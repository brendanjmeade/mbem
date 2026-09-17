"""Compressed backend: block-compressed operators + preconditioned FGMRES.

Same BlockSystem contract as the dense backend, but every (field,
source, kernel) pair is stored ONCE as a material-basis PairCompressed;
materials only enter through coefficient vectors. ``rebuild_for_materials``
is therefore nearly free (the compressed geometry is shared), and the
preconditioner refactorizes only its small dense diagonal blocks.
"""

from __future__ import annotations

import numpy as np

from .. import defaults
from ..kernels import basis as kb
from ..la.hop import PairCompressed
from ..la.preconditioner import BlockGaussSeidel
from ..la.solver import fgmres
from ..model.equations import BlockSystem


class HBackend:
    """``jump="half"`` is the classical 1/2 I collocation term;
    ``jump="calibrated"`` replaces it with the rigid-body-calibrated
    diagonal ``C_q = -sum_p sigma(R,p) * rowsum_j H_qp`` (the same
    calibration as the dense backend). The row sums are taken through
    the COMPRESSED H pairs, so the operator that is actually applied
    annihilates constant displacement fields exactly; pairs absent from
    the compressed set (zero-valued prescribed patches appear in neither
    terms nor rhs) are summed with the exact matrix-free contraction
    kernels. Same all-Neumann null-space caveat as the dense backend.
    """

    def __init__(self, tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK,
                 n_workers: int | None = None,
                 jump: str = "half",
                 storage: str = "basis",
                 verbose: bool = False):
        # storage="basis": geometry-only per-basis factors (B-fold
        # memory, free material recombination -- best for sweeps).
        # storage="combined": material-combined payloads only (1x
        # memory -- the mode for very large models; material rebuilds
        # re-compress).
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        if storage not in ("basis", "combined"):
            raise ValueError(storage)
        self.opts = dict(tol=tol, min_leaf=min_leaf, eta=eta,
                         max_admissible=max_admissible, n_workers=n_workers)
        self.jump = jump
        self.storage = storage
        self.verbose = verbose

    def assemble(self, system: BlockSystem, eps) -> "AssembledH":
        return AssembledH(system, eps, self.opts, self.verbose,
                          jump=self.jump, storage=self.storage)


def _coeffs(kernel: str, mat) -> np.ndarray:
    return kb.t_coeffs(mat.mu, mat.lam) if kernel == "H" \
        else kb.u_coeffs(mat.mu, mat.lam)


class AssembledH:
    def __init__(self, system: BlockSystem, eps, opts: dict, verbose: bool,
                 jump: str = "half", storage: str = "basis", _shared=None):
        self.system = system
        self.layout = system.layout
        self.eps = eps
        self.opts = opts
        self.verbose = verbose
        self.jump = jump
        self.storage = storage
        self.materials = {r.name: r.material for r in system.model.regions}

        if _shared is None:
            self._pairs: dict = {}
            self._tree_cache: dict = {}
            arrays = kb.MeshArrays()      # assembly-scoped mesh-array memo
            pair_keys = {(id(t.field_patch), id(t.source_patch), t.kernel):
                         (t.field_patch, t.source_patch, t.kernel)
                         for t in system.terms}
            for rt in system.rhs_terms:
                pair_keys[(id(rt.field_patch), id(rt.source_patch),
                           rt.kernel)] = (rt.field_patch, rt.source_patch,
                                          rt.kernel)
            combos = self._pair_combos()
            for key, (fp, sp, kern) in pair_keys.items():
                pc = PairCompressed(
                    fp.mesh, sp.mesh, kern, self.eps_for(sp),
                    tree_cache=self._tree_cache, arrays=arrays,
                    storage=storage, combine_for=combos.get(key, []),
                    **opts)
                if verbose:
                    print(f"  {fp.name} <- {sp.name} [{kern}]: "
                          f"{pc.summary()}")
                self._pairs[key] = pc
        else:
            self._pairs, self._tree_cache = _shared

        self._refresh_material_state()
        self._precond = None

    def _pair_combos(self) -> dict:
        """{pair key: [coefficient vectors]} every solve stage touches --
        system terms, RHS terms, and (when calibrated) the calibration
        row-sums (t_coeffs of each region for its own (q,p) H pairs)."""
        combos: dict = {}

        def _add(key, c):
            lst = combos.setdefault(key, [])
            cb = np.asarray(c).tobytes()
            if not any(np.asarray(x).tobytes() == cb for x in lst):
                lst.append(np.asarray(c))

        for t in list(self.system.terms) + list(self.system.rhs_terms):
            key = (id(t.field_patch), id(t.source_patch), t.kernel)
            _add(key, _coeffs(t.kernel, self.materials[t.region.name]))
        if self.jump == "calibrated":
            for region in self.system.model.regions:
                c = kb.t_coeffs(self.materials[region.name].mu,
                                self.materials[region.name].lam)
                for q in region.patches:
                    for p in region.patches:
                        _add((id(q), id(p), "H"), c)
        return combos

    def _refresh_material_state(self):
        """(Re)compute everything that depends on the region materials:
        the calibration diagonal (if calibrated) and the RHS. In
        storage="combined" mode, first warm every pair's views for the
        new material set in one compression pass each."""
        if self.storage == "combined":
            combos = self._pair_combos()
            for key, pair in self._pairs.items():
                pair.warm_views(combos.get(key, []))
        self.calib = self._calibration() if self.jump == "calibrated" \
            else {}
        self.b = self._build_rhs()

    # -- calibrated jump ------------------------------------------------

    def _calibration(self) -> dict:
        """{(id(region), id(q)): C (Nq,3,3)}: the rigid-body-calibrated
        collocation diagonal per BIE row (region, collocation patch q).

        C = -sum_{p in dR} sigma(R,p) * rowsum_j H_qp, evaluated as
        three constant-field matvecs per (q,p) through the COMPRESSED
        pair when it exists (so the applied operator annihilates
        constants exactly), or through the exact matrix-free
        t_disp_contract for pairs the system never built (zero-valued
        prescribed-displacement patches).
        """
        from ..kernels import tri_kernels as tk

        model = self.system.model
        arrays = kb.MeshArrays()
        calib: dict = {}
        for region in model.regions:
            mat = self.materials[region.name]
            coeffs = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
            for q in region.patches:
                Nq = q.n_triangles
                C = np.zeros((Nq, 3, 3))
                xq = None
                for p in region.patches:
                    sigma = float(model.orientation(region, p))
                    pair = self._pairs.get((id(q), id(p), "H"))
                    for k in range(3):
                        if pair is not None:
                            const = np.zeros(3 * p.n_triangles)
                            const[k::3] = 1.0
                            col = pair.matvec(coeffs, const)
                        else:
                            if xq is None:
                                xq = arrays.field_points(q.mesh)
                            tv, nrm = arrays.source_arrays(p.mesh)
                            dens = np.zeros((p.n_triangles, 3))
                            dens[:, k] = 1.0
                            col = tk.t_disp_contract(
                                xq, tv, nrm, self.eps_for(p), dens,
                                *coeffs).ravel()
                        C[:, :, k] -= sigma * col.reshape(Nq, 3)
                calib[(id(region), id(q))] = C
        return calib

    def _calib_row(self, region, q):
        return self.calib[(id(region), id(q))]

    # -- helpers --------------------------------------------------------

    def eps_for(self, patch) -> np.ndarray:
        e = self.eps[patch.name] if isinstance(self.eps, dict) else self.eps
        return kb.resolve_eps(e, patch.mesh)

    def pair_for(self, term) -> PairCompressed:
        return self._pairs[(id(term.field_patch), id(term.source_patch),
                            term.kernel)]

    def _build_rhs(self) -> np.ndarray:
        calibrated = self.jump == "calibrated"
        b = np.zeros(self.layout.n_unknowns)
        for rt in self.system.rhs_terms:
            pair = self.pair_for(rt)
            mat = self.materials[rt.region.name]
            contrib = rt.scale * pair.matvec(_coeffs(rt.kernel, mat),
                                             rt.vector)
            b[rt.row.offset:rt.row.stop] += contrib
            if rt.add_half_of_vector and not calibrated:
                b[rt.row.offset:rt.row.stop] += rt.scale_half * rt.vector
        if calibrated:
            # prescribed-displacement collocation rows: the calibrated
            # diagonal multiplies the KNOWN u_bar, so it lands on the RHS
            # (mirrors AssembledDense._apply_calibration).
            layout = self.layout
            for region in self.system.model.regions:
                for q in region.patches:
                    if layout.has_slot(q, "u"):
                        continue
                    u_bar = q.value_array()
                    if not np.any(u_bar):
                        continue
                    C = self._calib_row(region, q)
                    row = layout.row_slot(region, q)
                    b[row.offset:row.stop] -= np.einsum(
                        "nij,nj->ni", C, u_bar).ravel()
        return b

    # -- operator ---------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        calibrated = self.jump == "calibrated"
        y = np.zeros_like(x)
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            seg = x[term.col.offset:term.col.stop]
            y[term.row.offset:term.row.stop] += \
                term.scale * pair.matvec(_coeffs(term.kernel, mat), seg)
            if term.diag_half and not calibrated:
                y[term.row.offset:term.row.stop] += 0.5 * seg
        if calibrated:
            layout = self.layout
            for region in self.system.model.regions:
                for q in region.patches:
                    if not layout.has_slot(q, "u"):
                        continue
                    C = self._calib_row(region, q)
                    row = layout.row_slot(region, q)
                    col = layout.slot(q, "u")
                    seg = x[col.offset:col.stop].reshape(-1, 3)
                    y[row.offset:row.stop] += np.einsum(
                        "nij,nj->ni", C, seg).ravel()
        return y

    def to_dense(self) -> np.ndarray:
        calibrated = self.jump == "calibrated"
        n = self.layout.n_unknowns
        A = np.zeros((n, n))
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            blk = pair.to_dense(_coeffs(term.kernel, mat))
            A[term.row.offset:term.row.stop,
              term.col.offset:term.col.stop] += term.scale * blk
            if term.diag_half and not calibrated:
                idx = np.arange(term.row.size)
                A[term.row.offset + idx, term.col.offset + idx] += 0.5
        if calibrated:
            layout = self.layout
            for region in self.system.model.regions:
                for q in region.patches:
                    if not layout.has_slot(q, "u"):
                        continue
                    C = self._calib_row(region, q)
                    row = layout.row_slot(region, q)
                    col = layout.slot(q, "u")
                    for i in range(q.n_triangles):
                        A[row.offset + 3 * i:row.offset + 3 * i + 3,
                          col.offset + 3 * i:col.offset + 3 * i + 3] += C[i]
        return A

    # -- solve -------------------------------------------------------

    def solve(self, rtol: float = defaults.GMRES_RTOL,
              restart: int = defaults.GMRES_RESTART,
              maxiter: int = defaults.GMRES_MAXITER,
              x0: np.ndarray | None = None,
              precond_max_dense: int = defaults.MAX_DENSE_PRECOND_DOF,
              precond_hodlr_max: int = defaults.PRECOND_HODLR_MAX_DOF,
              deflate: bool = False):
        """Preconditioned FGMRES. Returns (slot dict, SolveReport).

        ``deflate=True`` projects the rigid-translation null space out
        of the iteration (all-Neumann + calibrated models): solves
        P A P y = P b with P = I - Z Z^T and returns the zero-mean-
        translation representative P y.
        """
        if self._precond is None:
            self._precond = BlockGaussSeidel(self, max_dense=precond_max_dense,
                                             hodlr_max=precond_hodlr_max,
                                             verbose=self.verbose)
        if deflate:
            from .dense import translation_basis
            Z = translation_basis(self.layout)

            def proj(v):
                return v - Z @ (Z.T @ v)

            A_mv = lambda v: proj(self.matvec(proj(v)))       # noqa: E731
            M_mv = lambda r: proj(self._precond(proj(r)))     # noqa: E731
            x, report = fgmres(A_mv, proj(self.b), M=M_mv, x0=x0,
                               rtol=rtol, restart=restart, maxiter=maxiter)
            x = proj(x)
        else:
            x, report = fgmres(self.matvec, self.b, M=self._precond, x0=x0,
                               rtol=rtol, restart=restart, maxiter=maxiter)
        if self.verbose:
            print(f"  {report}")
        out = {s.name: x[s.offset:s.stop].reshape(-1, 3)
               for s in self.layout.slots}
        return out, report

    # -- viscoelastic hook --------------------------------------------

    def rebuild_for_materials(self, material_map: dict) -> "AssembledH":
        new = AssembledH(self.system, self.eps, self.opts, self.verbose,
                         jump=self.jump, storage=self.storage,
                         _shared=(self._pairs, self._tree_cache))
        for name, mat in material_map.items():
            if name not in new.materials:
                raise KeyError(f"unknown region '{name}'")
            new.materials[name] = mat
        # calibration + RHS are material-dependent: recompute AFTER the
        # material override (cheap: constant-field matvecs).
        new._refresh_material_state()
        return new

    # -- stats -------------------------------------------------------

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self._pairs.values())
