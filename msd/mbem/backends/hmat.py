"""Compressed backend: block-compressed operators + preconditioned FGMRES.

Same BlockSystem contract as the dense backend, but every (field,
source, kernel) pair is stored ONCE as a material-basis PairCompressed;
materials only enter through coefficient vectors. ``rebuild_for_materials``
is therefore nearly free (the compressed geometry is shared), and the
preconditioner refactorizes only its small dense diagonal blocks.
RHS terms (fault slip, prescribed values) are applied matrix-free by
default -- their pair is used once per material, so compressing it would
cost a full ACA build for one matvec -- and compressed only for a
declared material ``sweep``. P0 patches only
(``la.hop.require_order0``); higher order is the dense backend's.
"""

from __future__ import annotations

import numpy as np

from .. import defaults
from ..kernels import KERNEL_T, kernel_coeffs
from ..kernels import basis as kb
from ..la.hop import PairCompressed, require_order0
from ..la.preconditioner import BlockGaussSeidel
from ..la.solver import fgmres
from ..model.equations import (BlockSystem, add_block_diagonal,
                               add_jump_rhs, calibrated_diagonal,
                               diagonal_matvec, term_diagonal)


class HBackend:
    """``jump="half"`` is the classical collocation free term;
    ``jump="calibrated"`` replaces it with the rigid-body-calibrated
    diagonal of ``equations.calibrated_diagonal``, row-summed through the
    COMPRESSED H pairs so the operator actually applied annihilates
    constant displacement fields exactly. Same all-Neumann null-space
    caveat as the dense backend.
    """

    def __init__(self, tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK,
                 n_workers: int | None = None,
                 jump: str = "calibrated",
                 deflate: bool = False,
                 storage: str = "combined",
                 sweep: bool = False,
                 verbose: bool = False):
        # jump / deflate: as for the dense backend; an all-Neumann model
        # with jump="calibrated" needs deflate=True (assembly refuses
        # otherwise), and solve then projects the translations out.
        # storage="combined" (default): material-combined payloads only,
        # 1x memory -- a T pair in per-basis storage holds six factor
        # pairs per block; material rebuilds re-compress transiently.
        # storage="basis": geometry-only per-basis factors (B-fold
        # memory, free material recombination) when they fit.
        # sweep: declare a material sweep, so the RHS pairs (fault slip,
        # prescribed values) are compressed once and reused per material
        # instead of applied matrix-free at every rebuild -- at 100k
        # unknowns a fault RHS is ~2e9 kernel pairs, 70 s per material.
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        if storage not in ("basis", "combined"):
            raise ValueError(storage)
        self.opts = dict(tol=tol, min_leaf=min_leaf, eta=eta,
                         max_admissible=max_admissible, n_workers=n_workers)
        self.jump = jump
        self.deflate = deflate
        self.storage = storage
        self.sweep = sweep
        self.verbose = verbose

    def assemble(self, system: BlockSystem, eps) -> "AssembledH":
        return AssembledH(system, eps, self.opts, self.verbose,
                          jump=self.jump, deflate=self.deflate,
                          storage=self.storage, sweep=self.sweep)


class AssembledH:
    def __init__(self, system: BlockSystem, eps, opts: dict, verbose: bool,
                 jump: str = "calibrated", deflate: bool = False,
                 storage: str = "combined", sweep: bool = False,
                 _shared=None):
        from .dense import (require_anchor_or_deflate,
                            warn_collocation_near_fault, warn_half_jump_eps)
        if jump not in ("half", "calibrated"):
            raise ValueError(jump)
        require_order0(system.model)
        require_anchor_or_deflate(system, jump, deflate)
        warn_half_jump_eps(system, eps, jump)
        warn_collocation_near_fault(system, eps)
        self.system = system
        self.layout = system.layout
        self.eps = eps
        self.opts = opts
        self.verbose = verbose
        self.jump = jump
        self.deflate = deflate
        self.storage = storage
        self.sweep = sweep
        self.report = None
        self.materials = {r.name: r.material for r in system.model.regions}

        if _shared is None:
            self._pairs: dict = {}
            self._tree_cache: dict = {}
            arrays = kb.MeshArrays()      # assembly-scoped mesh-array memo
            pair_keys = {(id(t.field_patch), id(t.source_patch), t.kernel):
                         (t.field_patch, t.source_patch, t.kernel)
                         for t in system.terms}
            if sweep:                     # RHS pairs are compressed only
                for rt in system.rhs_terms:   # for a declared sweep
                    pair_keys[(id(rt.field_patch), id(rt.source_patch),
                               rt.kernel)] = (rt.field_patch,
                                              rt.source_patch, rt.kernel)
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
        system terms, RHS terms (only when they are compressed: ``sweep``),
        and (when calibrated) the calibration row-sums (t_coeffs of each
        region for its own (q,p) H pairs): the views built at assembly
        (all that combined storage keeps); each is certified for its own
        material when it is combined."""
        combos: dict = {}

        def _add(key, c):
            lst = combos.setdefault(key, [])
            cb = np.asarray(c).tobytes()
            if not any(np.asarray(x).tobytes() == cb for x in lst):
                lst.append(np.asarray(c))

        terms = list(self.system.terms)
        if self.sweep:
            terms += list(self.system.rhs_terms)
        for t in terms:
            key = (id(t.field_patch), id(t.source_patch), t.kernel)
            _add(key, kernel_coeffs(t.kernel, self.materials[t.region.name]))
        if self.jump == "calibrated":
            for region in self.system.model.regions:
                c = kb.t_coeffs(self.materials[region.name].mu,
                                self.materials[region.name].lam)
                for q in region.patches:
                    for p in region.patches:
                        _add((id(q), id(p), KERNEL_T), c)
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
            else None
        self.b = self._build_rhs()

    # -- calibrated jump ------------------------------------------------

    def _calibration(self) -> dict:
        """``calibrated_diagonal`` with the H row-sums taken as three
        constant-field matvecs through the COMPRESSED pair when it exists
        (so the operator actually applied annihilates constants exactly),
        or through the exact matrix-free ``t_disp_contract`` for pairs the
        system never built (zero-valued prescribed-displacement patches).
        """
        from ..kernels import tri_kernels as tk

        arrays = kb.MeshArrays()

        def rowsum(region, q, p):
            mat = self.materials[region.name]
            coeffs = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
            pair = self._pairs.get((id(q), id(p), KERNEL_T))
            S = np.zeros((q.n_nodes, 3, 3))
            for k in range(3):
                if pair is not None:
                    const = np.zeros(3 * p.n_nodes)
                    const[k::3] = 1.0
                    col = pair.matvec(coeffs, const)
                else:
                    xq = arrays.field_points(q.mesh)
                    tv, nrm = arrays.source_arrays(p.mesh)
                    dens = np.zeros((p.n_nodes, 3))
                    dens[:, k] = 1.0
                    col = tk.t_disp_contract(xq, tv, nrm, self.eps_for(p),
                                             dens, *coeffs).ravel()
                S[:, :, k] = col.reshape(q.n_nodes, 3)
            return S

        return calibrated_diagonal(self.system, rowsum)

    # -- helpers --------------------------------------------------------

    def eps_for(self, patch) -> np.ndarray:
        return kb.resolve_patch_eps(self.eps, patch)

    def pair_for(self, term) -> PairCompressed:
        return self._pairs[(id(term.field_patch), id(term.source_patch),
                            term.kernel)]

    def _build_rhs(self) -> np.ndarray:
        """Each RhsTerm's kernel applied to its KNOWN vector (fault slip,
        prescribed u / t) at the term's material. Through the compressed
        pair when one was built (``sweep``), else matrix-free through the
        nodal contraction drivers at the field patch's collocation points:
        the pair is applied once per material, so a compression that pays
        off only over repeated matvecs is not built for it."""
        from ..kernels import tri_nodal as tn

        arrays = kb.MeshArrays()
        b = np.zeros(self.layout.n_unknowns)
        for rt in self.system.rhs_terms:
            coeffs = kernel_coeffs(rt.kernel, self.materials[rt.region.name])
            pair = self._pairs.get((id(rt.field_patch), id(rt.source_patch),
                                    rt.kernel))
            if pair is not None:
                contrib = pair.matvec(coeffs, rt.vector)
            else:
                q, p = rt.field_patch, rt.source_patch
                xq = arrays.collocation_points(q)
                tv, nrm = arrays.source_arrays(p.mesh)
                dens = np.ascontiguousarray(rt.vector.reshape(-1, 3))
                if rt.kernel == KERNEL_T:
                    u = tn.t_disp_contract(xq, tv, nrm, self.eps_for(p), dens,
                                           *coeffs, p.order)
                else:
                    u = tn.u_disp_contract(xq, tv, self.eps_for(p), dens,
                                           *coeffs, p.order)
                contrib = u.ravel()
            b[rt.row.offset:rt.row.stop] += rt.scale * contrib
        add_jump_rhs(self.system, self.calib, b)
        return b

    # -- operator ---------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            seg = x[term.col.offset:term.col.stop]
            y[term.row.offset:term.row.stop] += \
                term.scale * pair.matvec(kernel_coeffs(term.kernel, mat), seg)
            D = term_diagonal(term, self.calib)
            if D is not None:
                y[term.row.offset:term.row.stop] += diagonal_matvec(D, seg)
        return y

    def matvec_exact_rows(self, x: np.ndarray, rows) -> np.ndarray:
        """Exact ``(A x)[rows]`` for global row indices ``rows``, matrix-free.

        Every term's kernel is contracted through the nodal drivers at the
        selected collocation points (no compressed block is touched) and the
        same collocation diagonal ``matvec`` applies is added, so this is the
        reference for the compressed operator where a dense matrix no longer
        fits (the bench harness samples 1,024 rows). Cost is
        O(len(rows) x N_source) kernel pairs.
        """
        from ..kernels import tri_nodal as tn

        rows = np.asarray(rows, dtype=int)
        x = np.asarray(x, dtype=float)
        out = np.zeros(rows.size)
        arrays = kb.MeshArrays()
        for term in self.system.terms:
            sel = np.nonzero((rows >= term.row.offset)
                             & (rows < term.row.stop))[0]
            if sel.size == 0:
                continue
            local = rows[sel] - term.row.offset
            c, comp = local // 3, local % 3
            pts, inv = np.unique(c, return_inverse=True)
            q, p = term.field_patch, term.source_patch
            dens = x[term.col.offset:term.col.stop].reshape(-1, 3)
            coeffs = kernel_coeffs(term.kernel, self.materials[term.region.name])
            xq = np.ascontiguousarray(arrays.collocation_points(q)[pts])
            tv, nrm = arrays.source_arrays(p.mesh)
            if term.kernel == KERNEL_T:
                u = tn.t_disp_contract(xq, tv, nrm, self.eps_for(p), dens,
                                       *coeffs, p.order)
            else:
                u = tn.u_disp_contract(xq, tv, self.eps_for(p), dens,
                                       *coeffs, p.order)
            vals = term.scale * u[inv, comp]
            D = term_diagonal(term, self.calib)
            if D is not None:
                # add_block_diagonal's rule, row by row.
                shape = q.collocation_shape()
                K = shape.shape[0]
                s, kc = c // K, c % K
                for k in range(K):
                    vals += shape[kc, k] * np.einsum(
                        "nb,nb->n", D[c, comp, :], dens[K * s + k])
            out[sel] += vals
        return out

    def to_dense(self) -> np.ndarray:
        n = self.layout.n_unknowns
        A = np.zeros((n, n))
        for term in self.system.terms:
            pair = self.pair_for(term)
            mat = self.materials[term.region.name]
            blk = pair.to_dense(kernel_coeffs(term.kernel, mat))
            A[term.row.offset:term.row.stop,
              term.col.offset:term.col.stop] += term.scale * blk
            D = term_diagonal(term, self.calib)
            if D is not None:
                add_block_diagonal(A, term.row.offset, term.col.offset, D,
                                   term.field_patch.collocation_shape())
        return A

    # -- solve -------------------------------------------------------

    def solve(self, rtol: float = defaults.GMRES_RTOL,
              restart: int = defaults.GMRES_RESTART,
              maxiter: int = defaults.GMRES_MAXITER,
              x0: np.ndarray | None = None,
              precond_max_dense: int = defaults.MAX_DENSE_PRECOND_DOF,
              precond_hodlr_max: int = defaults.PRECOND_HODLR_MAX_DOF):
        """Preconditioned FGMRES; returns {slot_name: (N_patch, 3) array}
        and leaves ``self.report`` (the ``SolveReport``).

        With ``deflate`` the rigid-translation null space is projected out
        of the iteration (all-Neumann + calibrated models): solves
        P A P y = P b with P = I - Z Z^T and returns the zero-mean-
        translation representative P y.
        """
        if self._precond is None:
            self._precond = BlockGaussSeidel(self, max_dense=precond_max_dense,
                                             hodlr_max=precond_hodlr_max,
                                             verbose=self.verbose)
        if self.deflate:
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
        report.precond_summary = self._precond.summary()
        if self.verbose:
            print(f"  {report}")
        self.report = report
        return {s.name: x[s.offset:s.stop].reshape(-1, 3)
                for s in self.layout.slots}

    # -- viscoelastic hook --------------------------------------------

    def rebuild_for_materials(self, material_map: dict) -> "AssembledH":
        new = AssembledH(self.system, self.eps, self.opts, self.verbose,
                         jump=self.jump, deflate=self.deflate,
                         storage=self.storage, sweep=self.sweep,
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
