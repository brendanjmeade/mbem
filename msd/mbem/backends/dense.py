"""Dense backend for BlockSystem: direct LU solve at oracle-parity quality.

Two assembly modes:

* ``mode="legacy"`` — every (field, source, kernel, material) block is
  produced by the legacy ``mollified_bem.assemble_BEM_matrices`` call,
  giving entrywise-identical blocks to the hand-written assemblers
  (the Phase-2 parity gate). Global scalar eps only.

* ``mode="basis"`` — geometry-only basis stacks are assembled ONCE per
  (field, source, kernel) pair with the numba kernels and recombined
  per material. ``rebuild_for_materials`` then re-solves with new
  region materials at recombination cost (no re-integration) — the
  primitive the viscoelastic Laplace sweep needs. Supports per-source-
  element eps arrays keyed by patch name.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .. import defaults
from ..kernels import basis as kb
from ..model.equations import BlockSystem


class DenseBackend:
    """``jump="half"`` adds the classical 1/2 I collocation term;
    ``jump="calibrated"`` instead sets each collocation diagonal so that
    constant displacement fields are annihilated EXACTLY (the rigid-body
    calibration standard in singular BEM). For mollified kernels the
    Gauss identity sum_j H_ij = -1/2 I holds only approximately — the
    blob leaks across the surface, catastrophically so for panels
    thinner than a few eps — and the resulting spectral perturbation is
    what creates the spurious material-resonance bands. Calibration
    repairs the identity by construction.

    On all-Neumann models the exact identity makes rigid translations an
    EXACT null space: "calibrated" there requires ``deflate=True``.
    """

    def __init__(self, mode: str = "basis", jump: str = "calibrated",
                 deflate: bool = False):
        # "legacy": legacy assembly calls (oracle parity; scalar eps).
        # "basis":  numba basis stacks, cached for cheap material rebuilds
        #           (memory ~9x one dense matrix across the pair set).
        # "direct": numba in-loop assembly, no basis storage — the
        #           memory-light choice for one-shot large dense solves;
        #           rebuild_for_materials re-assembles (still fast).
        # jump:    "calibrated" annihilates constant fields exactly; "half"
        #           with eps/h > ~0.5 is NON-convergent (the backends warn).
        # deflate: solve the rigid-translation-bordered system — REQUIRED
        #           for jump="calibrated" on all-Neumann models (the
        #           assembled operator raises otherwise).
        if mode not in ("legacy", "basis", "direct"):
            raise ValueError(mode)
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        self.mode = mode
        self.jump = jump
        self.deflate = deflate

    def assemble(self, system: BlockSystem, eps) -> "AssembledDense":
        return AssembledDense(system, eps, self.mode, jump=self.jump,
                              deflate=self.deflate)


def _eps_for(eps, patch) -> np.ndarray:
    if isinstance(eps, dict):
        e = eps[patch.name]
    else:
        e = eps
    return kb.resolve_eps(e, patch.mesh)


def require_anchor_or_deflate(system: BlockSystem, jump: str, deflate: bool):
    """``jump="calibrated"`` on an un-anchored (all-Neumann) model makes
    rigid translations an EXACT null space of A; refuse to solve it without
    ``deflate`` (the factorization would return |u| ~ 1e9 km). Shared by both
    backends: dense checks at assembly, H at ``solve`` where ``deflate`` lives.
    """
    if jump == "calibrated" and not deflate and not system.model.is_anchored():
        raise ValueError(
            "jump='calibrated' on an all-Neumann model (no "
            "PRESCRIBED_DISPLACEMENT patch): rigid translations are an EXACT "
            "null space of the calibrated operator. Pass deflate=True "
            "(bordered / projected solve) or prescribe displacement on a "
            "patch.")


def warn_half_jump_eps(system: BlockSystem, eps, jump: str):
    """``jump="half"`` with eps/h above ``defaults.HALF_JUMP_MAX_EPS_OVER_H``
    on any source patch is NON-convergent under h-refinement: warn, naming
    the offending patch. Shared by both backends.
    """
    if jump != "half":
        return
    worst_name, worst = None, 0.0
    for r in system.model.regions:
        for p in list(r.patches) + list(r.faults):
            try:
                ratio = float(np.max(_eps_for(eps, p)
                                     / kb.element_sizes(p.mesh)))
            except (KeyError, ValueError):
                continue          # the backend reports a bad spec itself
            if ratio > worst:
                worst_name, worst = p.name, ratio
    if worst > defaults.HALF_JUMP_MAX_EPS_OVER_H:
        import warnings
        warnings.warn(
            f"jump='half' with eps/h = {worst:.2f} on patch '{worst_name}' "
            f"(limit {defaults.HALF_JUMP_MAX_EPS_OVER_H}): this combination "
            f"was measured NON-convergent under h-refinement; use "
            f"jump='calibrated' or a smaller eps")


def translation_basis(layout) -> np.ndarray:
    """Orthonormal rigid-translation basis Z (n, 3) on the u-slots.

    With jump="calibrated" on an all-Neumann model these three vectors
    span the EXACT null space of A (the calibration annihilates constant
    displacement fields by construction; tractions stay zero)."""
    Z = np.zeros((layout.n_unknowns, 3))
    for slot in layout.slots:
        if slot.kind == "u":
            for k in range(3):
                Z[slot.offset + k:slot.stop:3, k] = 1.0
    Z /= np.linalg.norm(Z, axis=0, keepdims=True)
    return Z


class AssembledDense:
    def __init__(self, system: BlockSystem, eps, mode: str,
                 jump: str = "calibrated", deflate: bool = False,
                 _basis_cache: dict | None = None):
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        require_anchor_or_deflate(system, jump, deflate)
        warn_half_jump_eps(system, eps, jump)
        self.system = system
        self.layout = system.layout
        self.eps = eps
        self.mode = mode
        self.jump = jump
        # deflate=True solves the BORDERED system [[A, Z], [Z^T, 0]]
        # with Z the rigid-translation basis: the fix for all-Neumann +
        # calibrated models, whose A has translations as an EXACT null
        # space (the returned solution is the zero-mean-translation
        # representative).
        self.deflate = deflate
        self.materials = {r.name: r.material for r in system.model.regions}
        # basis cache: (id(field), id(source), kernel) -> UBasis | TBasis
        self._basis = _basis_cache if _basis_cache is not None else {}
        self._lu = None
        self.A = None
        self.b = None
        self._build()

    # -- block providers ------------------------------------------------

    def _block_legacy(self, field_patch, source_patch, kernel, material):
        import mollified_bem as mb
        if isinstance(self.eps, (dict, str)):
            raise ValueError("legacy mode supports only global scalar eps")
        kern = "T" if kernel == "H" else "U"
        return mb.assemble_BEM_matrices(field_patch.mesh, source_patch.mesh,
                                        material, float(self.eps), kern)

    def _block_basis(self, field_patch, source_patch, kernel, material):
        key = (id(field_patch), id(source_patch), kernel)
        b = self._basis.get(key)
        if b is None:
            eps_arr = _eps_for(self.eps, source_patch)
            if kernel == "H":
                b = kb.assemble_t_basis(field_patch.mesh, source_patch.mesh,
                                        eps_arr, arrays=self._arrays)
            else:
                b = kb.assemble_u_basis(field_patch.mesh, source_patch.mesh,
                                        eps_arr, arrays=self._arrays)
            self._basis[key] = b
        return b.combine(material)

    def _block_direct(self, field_patch, source_patch, kernel, material):
        eps_arr = _eps_for(self.eps, source_patch)
        if kernel == "H":
            return kb.assemble_t_matrix(field_patch.mesh, source_patch.mesh,
                                        material, eps_arr,
                                        arrays=self._arrays)
        return kb.assemble_u_matrix(field_patch.mesh, source_patch.mesh,
                                    material, eps_arr, arrays=self._arrays)

    def _block(self, field_patch, source_patch, kernel, region):
        material = self.materials[region.name]
        if self.mode == "legacy":
            return self._block_legacy(field_patch, source_patch, kernel,
                                      material)
        if self.mode == "direct":
            return self._block_direct(field_patch, source_patch, kernel,
                                      material)
        return self._block_basis(field_patch, source_patch, kernel, material)

    # -- assembly --------------------------------------------------------

    def _cached_block(self, cache: dict, field_patch, source_patch,
                      kernel, region):
        """``_block`` through a per-build cache keyed on
        (field, source, kernel, material) identity. Shared by the system
        terms, the RHS terms, AND the calibration pass -- calibration
        needs the same H blocks the equations do, and re-assembling them
        (the old behaviour) doubled the direct-mode assembly cost."""
        mat = self.materials[region.name]
        ck = (id(field_patch), id(source_patch), kernel, id(mat))
        blk = cache.get(ck)
        if blk is None:
            blk = self._block(field_patch, source_patch, kernel, region)
            cache[ck] = blk
        return blk

    def _build(self):
        n = self.layout.n_unknowns
        # Block caching ACROSS terms: the same (field, source, kernel,
        # material) block can appear in several equations (it does not in
        # the current models, but dedupe is free and protects wrappers).
        # The cache is per-build (local), NOT stored on self: material
        # identity is by id(), and rebuild_for_materials swaps material
        # objects -- an instance-lifetime cache would risk id-reuse
        # staleness. Same rule for the mesh-array memo (assembly-scoped;
        # TriMesh can be mutated in place between builds).
        block_cache: dict = {}
        self._arrays = kb.MeshArrays()

        calibrated = self.jump == "calibrated"

        A = np.zeros((n, n))
        for t in self.system.terms:
            blk = self._cached_block(block_cache, t.field_patch,
                                     t.source_patch, t.kernel, t.region)
            r0, r1 = t.row.offset, t.row.stop
            c0, c1 = t.col.offset, t.col.stop
            A[r0:r1, c0:c1] += t.scale * blk
            if t.diag_half and not calibrated:
                idx = np.arange(r1 - r0)
                A[r0 + idx, c0 + idx] += 0.5

        b = np.zeros(n)
        for rt in self.system.rhs_terms:
            blk = self._cached_block(block_cache, rt.field_patch,
                                     rt.source_patch, rt.kernel, rt.region)
            r0, r1 = rt.row.offset, rt.row.stop
            b[r0:r1] += rt.scale * (blk @ rt.vector)
            if rt.add_half_of_vector and not calibrated:
                b[r0:r1] += rt.scale_half * rt.vector

        if calibrated:
            self._apply_calibration(A, b, block_cache)

        self._arrays = None          # end of assembly scope
        self.A = A
        self.b = b

    def _apply_calibration(self, A: np.ndarray, b: np.ndarray,
                           block_cache: dict) -> None:
        """Rigid-body jump calibration.

        For each BIE row (region R, collocation patch q), set the
        collocation diagonal C_q so that a constant displacement field
        over ALL of dR (unknown and prescribed alike) with zero
        tractions is annihilated exactly:

            C_q = - sum_{p in dR} sigma(R,p) * rowsum_j H^{m(R)}_{qp}

        replacing the 1/2 I of the classical jump relation. The sums
        include prescribed-displacement patches (their H term lives on
        the RHS, so the constant-field identity needs them here too).
        """
        layout = self.layout
        model = self.system.model
        for region in model.regions:
            for q in region.patches:
                row = layout.row_slot(region, q)
                Nq = q.n_triangles
                C = np.zeros((Nq, 3, 3))
                for p in region.patches:
                    sigma = float(model.orientation(region, p))
                    blk = self._cached_block(block_cache, q, p, "H", region)
                    C -= sigma * blk.reshape(Nq, 3, p.n_triangles,
                                             3).sum(axis=2)
                if layout.has_slot(q, "u"):
                    col = layout.slot(q, "u")
                    for i in range(Nq):
                        A[row.offset + 3 * i: row.offset + 3 * i + 3,
                          col.offset + 3 * i: col.offset + 3 * i + 3] \
                            += C[i]
                else:
                    u_bar = q.value_array()
                    if np.any(u_bar):
                        contrib = np.einsum("nij,nj->ni", C, u_bar)
                        b[row.offset:row.stop] -= contrib.ravel()

    # -- solve -------------------------------------------------------

    def solve(self) -> dict:
        """LU-solve; returns {slot_name: (N_patch, 3) array}.

        Also stores ``self.cond_estimate`` (1-norm condition estimate,
        essentially free from the LU via LAPACK gecon) — the
        viscoelastic pipeline uses it to detect samples polluted by the
        spurious discretization resonance.
        """
        if self._lu is None:
            if self.deflate:
                Z = translation_basis(self.layout)
                n = self.A.shape[0]
                K = np.zeros((n + 3, n + 3))
                K[:n, :n] = self.A
                K[:n, n:] = Z
                K[n:, :n] = Z.T
                self._deflate_n = n
                A_solve = K
            else:
                A_solve = self.A
            anorm = float(np.linalg.norm(A_solve, 1))
            self._lu = lu_factor(A_solve)
            from scipy.linalg import get_lapack_funcs
            gecon = get_lapack_funcs(("gecon",), (self._lu[0],))[0]
            rcond, info = gecon(self._lu[0], anorm, norm="1")
            self.cond_estimate = (1.0 / rcond) if (info == 0 and rcond > 0) \
                else np.inf
            if self.cond_estimate > defaults.COND_WARN_THRESHOLD:
                import warnings
                warnings.warn(
                    f"dense solve: condition estimate "
                    f"{self.cond_estimate:.2e} exceeds "
                    f"{defaults.COND_WARN_THRESHOLD:.0e}; the solution may "
                    f"miss the {defaults.SOLUTION_RTOL:.0e} accuracy target "
                    f"(thin panels, near-fluid material, un-anchored "
                    f"calibrated jump, or discretization resonance)")
        if self.deflate:
            rhs = np.concatenate([self.b, np.zeros(3)])
            x = lu_solve(self._lu, rhs)[:self._deflate_n]
        else:
            x = lu_solve(self._lu, self.b)
        out = {}
        for slot in self.layout.slots:
            out[slot.name] = x[slot.offset:slot.stop].reshape(-1, 3)
        return out

    # -- viscoelastic hook --------------------------------------------

    def rebuild_for_materials(self, material_map: dict) -> "AssembledDense":
        """New AssembledDense with updated region materials.

        ``material_map`` maps region name -> ElasticMaterial. In basis
        mode the cached geometry bases are reused, so the rebuild costs
        only recombination + refactorization; in direct mode the blocks
        are re-assembled with the numba kernels (no cache, still fast).
        """
        if self.mode == "legacy":
            raise ValueError("rebuild_for_materials requires mode='basis' "
                             "or 'direct'")
        new = AssembledDense.__new__(AssembledDense)
        new.system = self.system
        new.layout = self.layout
        new.eps = self.eps
        new.mode = self.mode
        new.jump = self.jump
        new.deflate = self.deflate
        new.materials = dict(self.materials)
        for name, mat in material_map.items():
            if name not in new.materials:
                raise KeyError(f"unknown region '{name}'")
            new.materials[name] = mat
        new._basis = self._basis          # shared geometry cache
        new._lu = None
        new.A = None
        new.b = None
        new._build()
        return new
