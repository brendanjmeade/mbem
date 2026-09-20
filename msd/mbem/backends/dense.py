"""Dense backend for BlockSystem: direct LU solve at oracle-parity quality.

Three assembly modes:

* ``mode="direct"`` (default) — numba in-loop assembly, no basis storage:
  the memory-light choice for one-shot solves; ``rebuild_for_materials``
  re-assembles (still fast).

* ``mode="legacy"`` — every (field, source, kernel, material) block is
  produced by the legacy ``mollified_bem.assemble_BEM_matrices`` call,
  giving entrywise-identical blocks to the hand-written assemblers
  (the parity gate). Global scalar eps only, P0 patches only.

* ``mode="basis"`` — geometry-only basis stacks are assembled ONCE per
  (field, source, kernel) pair with the numba kernels and recombined
  per material (~9x the memory of one dense matrix across the pair
  set). ``rebuild_for_materials`` then re-solves with new region
  materials at recombination cost (no re-integration) — the primitive
  a material sweep needs.

Both backends share one API: ``Backend(jump=..., deflate=...).assemble(
system, eps).solve()`` returns the slot dict and leaves ``asm.report``.

Every block is field = the row patch's collocation points, source = the
column patch's (mesh, order): rows are collocation points, columns nodal
densities (``model/core.py``), any mix of P0/P1/P2 patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .. import defaults
from ..kernels import KERNEL_T
from ..kernels import basis as kb
from ..model.equations import (BlockSystem, add_block_diagonal,
                               add_jump_rhs, calibrated_diagonal,
                               term_diagonal)


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

    def __init__(self, mode: str = "direct", jump: str = "calibrated",
                 deflate: bool = False):
        # mode:    "direct" / "legacy" / "basis" (module docstring).
        # jump:    "calibrated" annihilates constant fields exactly; "half"
        #           with eps/h > ~0.5 is NON-convergent (the backends warn).
        # deflate: solve the rigid-translation-bordered system — REQUIRED
        #           for jump="calibrated" on all-Neumann models (assembly
        #           refuses otherwise).
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


def require_anchor_or_deflate(system: BlockSystem, jump: str, deflate: bool):
    """``jump="calibrated"`` on an un-anchored (all-Neumann) model makes
    rigid translations an EXACT null space of A; refuse to solve it without
    ``deflate`` (the factorization would return |u| ~ 1e9 km). Both backends
    check at construction, where ``deflate`` lives.
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
                ratio = float(np.max(kb.resolve_patch_eps(eps, p)
                                     / kb.element_sizes(p.mesh)))
            except ValueError:
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


def warn_collocation_near_fault(system: BlockSystem, eps):
    """Warn when a boundary collocation point lies within
    ``defaults.COLLOCATION_FAULT_CLEARANCE_EPS`` fault-eps of a fault element
    of its region (exact point-to-triangle distance, eps of the nearest fault
    element): that row sees the fault's blob average, not the one-sided value
    its boundary condition means (a fault outcrop). Shared by both backends;
    warned, not corrected.
    """
    from ..geometry import distance_to_mesh

    limit = defaults.COLLOCATION_FAULT_CLEARANCE_EPS
    n_close, worst, names = 0, np.inf, []
    for r in system.model.regions:
        if not r.faults:
            continue
        for q in r.patches:
            x = q.collocation_points()
            for f in r.faults:
                try:
                    eps_f = kb.resolve_patch_eps(eps, f)
                except ValueError:
                    continue          # the backend reports a bad spec itself
                d, idx = distance_to_mesh(x, f.mesh)
                ratio = d / eps_f[idx]
                close = int(np.sum(ratio < limit))
                if close:
                    n_close += close
                    worst = min(worst, float(ratio.min()))
                    names.append(f"{q.name}/{f.name}")
    if n_close:
        import warnings
        warnings.warn(
            f"{n_close} boundary collocation point(s) lie within {limit:g} "
            f"fault-eps of a fault element ({', '.join(names)}; min d/eps = "
            f"{worst:.2f}): the fault's mollified field there is the blob "
            f"average across the slip surface, not the one-sided value the "
            f"boundary condition means (fault outcrop); the row is not "
            f"corrected")


def translation_basis(layout) -> np.ndarray:
    """Orthonormal rigid-translation basis Z (n, 3) on the u-slots (every
    node of every element translates alike).

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


@dataclass
class DenseSolveReport:
    """``asm.report`` of a dense solve: the 1-norm condition estimate
    (LAPACK gecon on the LU, essentially free); the compressed backend's
    counterpart is ``la.solver.SolveReport``."""
    cond_estimate: float

    def __str__(self):
        return f"dense LU: cond ~ {self.cond_estimate:.2e}"


class AssembledDense:
    def __init__(self, system: BlockSystem, eps, mode: str,
                 jump: str = "calibrated", deflate: bool = False,
                 _basis_cache: dict | None = None):
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        require_anchor_or_deflate(system, jump, deflate)
        warn_half_jump_eps(system, eps, jump)
        warn_collocation_near_fault(system, eps)
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
        self.report = None
        self.A = None
        self.b = None
        self._build()

    # -- block providers ------------------------------------------------

    def _block_legacy(self, field_patch, source_patch, kernel, material):
        import mollified_bem as mb
        if isinstance(self.eps, (dict, str)):
            raise ValueError("legacy mode supports only global scalar eps")
        if field_patch.order or source_patch.order:
            raise ValueError("legacy mode supports only order-0 patches")
        kern = "T" if kernel == KERNEL_T else "U"
        return mb.assemble_BEM_matrices(field_patch.mesh, source_patch.mesh,
                                        material, float(self.eps), kern)

    def _block_basis(self, field_patch, source_patch, kernel, material):
        key = (id(field_patch), id(source_patch), kernel)
        b = self._basis.get(key)
        if b is None:
            eps_arr = kb.resolve_patch_eps(self.eps, source_patch)
            x_c = self._arrays.collocation_points(field_patch)
            if kernel == KERNEL_T:
                b = kb.assemble_t_basis(x_c, source_patch.mesh, eps_arr,
                                        arrays=self._arrays,
                                        order=source_patch.order)
            else:
                b = kb.assemble_u_basis(x_c, source_patch.mesh, eps_arr,
                                        arrays=self._arrays,
                                        order=source_patch.order)
            self._basis[key] = b
        return b.combine(material)

    def _block_direct(self, field_patch, source_patch, kernel, material):
        eps_arr = kb.resolve_patch_eps(self.eps, source_patch)
        x_c = self._arrays.collocation_points(field_patch)
        if kernel == KERNEL_T:
            return kb.assemble_t_matrix(x_c, source_patch.mesh, material,
                                        eps_arr, arrays=self._arrays,
                                        order=source_patch.order)
        return kb.assemble_u_matrix(x_c, source_patch.mesh, material, eps_arr,
                                    arrays=self._arrays,
                                    order=source_patch.order)

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
        terms, the RHS terms and the calibration row-sums, which need the
        same H blocks the equations do."""
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

        # Calibrated jump: row-sum the same H blocks the equations use
        # (through the cache, so nothing is assembled twice).
        self.calib = None
        if self.jump == "calibrated":
            def rowsum(region, q, p):
                blk = self._cached_block(block_cache, q, p, KERNEL_T, region)
                return blk.reshape(q.n_nodes, 3, p.n_nodes, 3).sum(axis=2)
            self.calib = calibrated_diagonal(self.system, rowsum)

        A = np.zeros((n, n))
        for t in self.system.terms:
            blk = self._cached_block(block_cache, t.field_patch,
                                     t.source_patch, t.kernel, t.region)
            r0, r1 = t.row.offset, t.row.stop
            c0, c1 = t.col.offset, t.col.stop
            A[r0:r1, c0:c1] += t.scale * blk
            D = term_diagonal(t, self.calib)
            if D is not None:
                add_block_diagonal(A, r0, c0, D,
                                   t.field_patch.collocation_shape())

        b = np.zeros(n)
        for rt in self.system.rhs_terms:
            blk = self._cached_block(block_cache, rt.field_patch,
                                     rt.source_patch, rt.kernel, rt.region)
            r0, r1 = rt.row.offset, rt.row.stop
            b[r0:r1] += rt.scale * (blk @ rt.vector)
        add_jump_rhs(self.system, self.calib, b)

        self._arrays = None          # end of assembly scope
        self.A = A
        self.b = b

    # -- solve -------------------------------------------------------

    def solve(self) -> dict:
        """LU-solve; returns {slot_name: (N_patch, 3) array} and leaves
        ``self.report`` (a ``DenseSolveReport`` with ``cond_estimate``,
        used to flag samples polluted by discretization resonance).
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
            cond = (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
            self.report = DenseSolveReport(cond_estimate=float(cond))
            if cond > defaults.COND_WARN_THRESHOLD:
                import warnings
                warnings.warn(
                    f"dense solve: condition estimate {cond:.2e} exceeds "
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
        new.report = None
        new.A = None
        new.b = None
        new._build()
        return new
