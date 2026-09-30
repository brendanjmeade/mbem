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

import math

import numpy as np

from .. import defaults
from ..kernels import KERNEL_T, kernel_coeffs
from ..kernels import basis as kb
from ..la.hop import PairCompressed, require_order0
from ..la.preconditioner import BlockGaussSeidel
from ..la.solver import RecycleSpace, fgmres
from ..model.equations import (BlockSystem, add_block_diagonal,
                               add_jump_rhs, calibrated_diagonal,
                               diagonal_matvec, term_diagonal)


def material_step(old: dict, new: dict) -> float:
    """max |log(new/old)| over the moduli of the regions that changed.

    The one measure of "how far" a material sweep has moved, in the units
    the reuse policies are stated in (``defaults.PRECOND_REUSE_MAX_STEP``):
    scale-free, symmetric in the direction of the step, and infinite when
    a modulus changes sign or a region appears, which no reuse survives.
    """
    step = 0.0
    for name, mat in new.items():
        prev = old.get(name)
        if prev is None:
            return math.inf
        for a, b in ((mat.mu, prev.mu), (mat.lam, prev.lam)):
            if a == b:
                continue
            if a <= 0.0 or b <= 0.0:
                return math.inf
            step = max(step, abs(math.log(a / b)))
    return step


class _Lineage:
    """Cross-solve state shared by every ``AssembledH`` that one chain of
    ``rebuild_for_materials`` produces from a single assembly.

    Its members are tied to the unknown LAYOUT, not to the materials, so
    they survive a material change: the last solution vector (what
    ``solve(x0="previous")`` warm-starts from) and the GCRO-DR recycle
    space with the materials it was last harvested at. One chain is one
    assembly and so one layout, which is the invariant the length checks
    at the two use sites back up.
    """

    def __init__(self):
        self.x = None
        self.recycle: RecycleSpace | None = None
        self.recycle_materials: dict | None = None


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
                 min_aca: int = defaults.ACA_MIN_BLOCK,
                 precision=None,
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
        # precision: storage precision of the views, normally derived
        # from tol (la/flatview.storage_dtype); set it only to pin float64.
        self.opts = dict(tol=tol, min_leaf=min_leaf, eta=eta,
                         max_admissible=max_admissible, min_aca=min_aca,
                         precision=precision, n_workers=n_workers)
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
                 _shared=None, _groups=None, _lineage=None):
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
        # An FMM far field grouped by (region, kernel): one traversal for
        # every term of a group instead of one per pair. It REPLACES the
        # per-term pair.matvec below; the collocation diagonals are added
        # exactly as they were, from the unscaled segment.
        self._groups = _groups

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
        # The materials the preconditioner was BUILT at -- what
        # rebuild_for_materials measures a reuse step against -- and the
        # ladder caps it was built at, so a later solve asking for
        # different ones REBUILDS. Without that a sweep over one assembled
        # operator silently measures its first arm at every later point,
        # and agrees with itself wherever the caps happen to coincide.
        self._precond_materials: dict | None = None
        self._precond_caps: tuple | None = None
        self._lineage = _Lineage() if _lineage is None else _lineage

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
        if self.jump != "calibrated":
            self.calib = None
        elif self._groups is not None:
            self.calib = self._calibration_grouped()
        else:
            self.calib = self._calibration()
        self.b = self._build_rhs()

    # -- calibrated jump ------------------------------------------------

    def lower_applier(self, terms):
        """A callable ``(z, rk, local_offset)`` subtracting every term's
        ``scale * A_term @ z[col]``, or None to use the per-pair loop.

        The preconditioner's Gauss-Seidel sweep applies its strictly-lower
        couplings term by term, which for an FMM is one full traversal per
        TERM -- ~45 of them per apply on the bench model, and most of the
        solve. Grouped they share an upward pass and batch their M2L, the
        same win the operator matvec got.

        Returns None when the terms do not group (no FMM, or a subset whose
        coupling is incomplete, which ``FarGroups`` refuses): the caller
        then keeps the loop it had, which is always correct.
        """
        if self._groups is None or not terms:
            return None
        from ..la.fmm import FarGroups

        geom = self._groups.groups[0].pair.geom
        order = {g.kernel: g.pair.p for g in self._groups.groups}
        try:
            sub = FarGroups(self.system, self.materials, geom, order,
                            self.eps, terms=list(terms),
                            domain=geom.domain, m2l="table")
        except ValueError:
            return None                 # incomplete subset: keep the loop

        def apply(z, rk, local_offset):
            for g in sub.groups:
                c = np.asarray(kernel_coeffs(g.kernel,
                                             self.materials[g.region]))
                y = g.pair.matvec(c, np.concatenate(
                    [z[a:b] for a, b in g.cols]),
                    far=not defaults.PRECOND_LOWER_NEAR_ONLY)
                off = 0
                for name, (a, b) in zip(g.row_names, g.rows):
                    m = b - a
                    # by SLOT name: local_offset is keyed by slot, and a
                    # patch name is not a slot name. Getting this wrong
                    # subtracts NOTHING and the solve still converges, just
                    # slower -- 37 iterations became 52.
                    r0 = local_offset.get(name)
                    if r0 is not None:
                        rk[r0:r0 + m] -= y[off:off + m]
                    off += m
        return apply

    def _calibration_grouped(self) -> dict:
        """The calibrated diagonal from the GROUPED traversal.

        ``calibrated_diagonal`` sums ``-sigma(R, p) rowsum(R, q, p)`` over a
        region's patches, and a (region, T) group's matvec on a constant IS
        that sum -- sigma is folded into its source. So this costs three
        traversals per region where the per-pair route costs three per PAIR,
        which on the bench model is ~135 traversals and most of the build.

        A patch the system built no term for (a prescribed-displacement one)
        is not in the group and is added here through the exact matrix-free
        contraction, which is what the per-pair route does for it too.
        """
        from ..kernels import tri_kernels as tk

        arrays = kb.MeshArrays()
        model = self.system.model
        by_region = {g.region: g for g in self._groups.groups
                     if g.kernel == KERNEL_T}
        calib: dict = {}
        for region in model.regions:
            g = by_region.get(region.name)
            mat = self.materials[region.name]
            coeffs = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
            C = {id(q): np.zeros((q.n_nodes, 3, 3)) for q in region.patches}
            if g is not None:
                covered = {id(p) for p in g.source_patches}
                for k in range(3):
                    x = np.zeros(g.pair.shape[1])
                    x[k::3] = 1.0
                    y = g.pair.matvec(coeffs, x)       # sigma already in
                    off = 0
                    for q in g.field_patches:
                        m = 3 * q.n_nodes
                        if id(q) in C:
                            C[id(q)][:, :, k] -= y[off:off + m].reshape(
                                q.n_nodes, 3)
                        off += m
            else:
                covered = set()
            for p in region.patches:                   # the uncovered rest
                if id(p) in covered:
                    continue
                sg = float(model.orientation(region, p))
                tv, nrm = arrays.source_arrays(p.mesh)
                for q in region.patches:
                    xq = arrays.field_points(q.mesh)
                    for k in range(3):
                        dens = np.zeros((p.n_nodes, 3))
                        dens[:, k] = 1.0
                        col = tk.t_disp_contract(xq, tv, nrm, self.eps_for(p),
                                                 dens, *coeffs).ravel()
                        C[id(q)][:, :, k] -= sg * col.reshape(q.n_nodes, 3)
            for q in region.patches:
                calib[(id(region), id(q))] = C[id(q)]
        return calib

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
        if self._groups is not None:
            self._groups.matvec(x, y)
        for term in self.system.terms:
            seg = x[term.col.offset:term.col.stop]
            if self._groups is None:
                pair = self.pair_for(term)
                mat = self.materials[term.region.name]
                y[term.row.offset:term.row.stop] += term.scale * pair.matvec(
                    kernel_coeffs(term.kernel, mat), seg)
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
              x0: np.ndarray | str | None = None,
              precond_max_dense: int | None = None,
              precond_above_dense: str = defaults.PRECOND_RUNG_ABOVE_DENSE,
              precond_hodlr_max: int = defaults.PRECOND_HODLR_MAX_DOF,
              precond_bj_chunk: int | None = None,
              recycle: bool | None = None):
        """Preconditioned FGMRES; returns {slot_name: (N_patch, 3) array}
        and leaves ``self.report`` (the ``SolveReport``).

        With ``deflate`` the rigid-translation null space is projected out
        of the iteration (all-Neumann + calibrated models): solves
        P A P y = P b with P = I - Z Z^T and returns the zero-mean-
        translation representative P y.

        ``precond_max_dense`` overrides the preconditioner ladder's
        dense-LU cap, which is otherwise the machine's own
        (``la.preconditioner.dense_rung_max_dof``), and
        ``precond_above_dense`` names the rung past it ("block_jacobi" or
        "hodlr"; ``defaults.PRECOND_RUNG_ABOVE_DENSE``), and
        ``precond_bj_chunk`` that rung's chunk size
        (``defaults.PRECOND_BJ_CHUNK_DOF``) -- the term that sets the
        preconditioner's memory, O(N x chunk), and trades against its
        iteration count.

        Two sequence options, for the material sweeps (the solves this
        assembly's ``rebuild_for_materials`` chain produces):
        ``x0="previous"`` warm-starts from the last solution of the
        chain, and ``recycle`` (default ``defaults.GCRO_RECYCLE_DEFAULT``)
        carries a GCRO-DR subspace through it. Recycling costs k matvecs
        per solve before the first iteration, so it buys iterations, not
        necessarily wall time. The recycle space is dropped when the
        materials move further than ``defaults.PRECOND_REUSE_MAX_STEP``
        from the ones it was harvested at.
        """
        lineage = self._lineage
        if isinstance(x0, str):
            if x0 != "previous":
                raise ValueError(f"x0 must be an array, None or 'previous'"
                                 f" (got {x0!r})")
            x0 = lineage.x if (lineage.x is not None
                               and lineage.x.size == self.layout.n_unknowns) \
                else None
        rs = None
        if defaults.GCRO_RECYCLE_DEFAULT if recycle is None else recycle:
            # Harvested at materials too far from these? The outliers it
            # approximates have moved; start the space over.
            if lineage.recycle is None:
                lineage.recycle = RecycleSpace()
            elif lineage.recycle_materials is not None and material_step(
                    lineage.recycle_materials,
                    self.materials) > defaults.PRECOND_REUSE_MAX_STEP:
                lineage.recycle.reset()
            rs = lineage.recycle
            lineage.recycle_materials = dict(self.materials)

        caps = (precond_max_dense, precond_above_dense, precond_hodlr_max,
                precond_bj_chunk)
        if self._precond is not None and self._precond_caps != caps:
            self._precond = None          # asked for a different ladder
        if self._precond is None:
            self._precond = BlockGaussSeidel(self, max_dense=precond_max_dense,
                                             above_dense=precond_above_dense,
                                             hodlr_max=precond_hodlr_max,
                                             bj_chunk=precond_bj_chunk,
                                             verbose=self.verbose)
            self._precond_materials = dict(self.materials)
            self._precond_caps = caps
        if self.deflate:
            from .dense import translation_basis
            Z = translation_basis(self.layout)

            def proj(v):
                return v - Z @ (Z.T @ v)

            A_mv = lambda v: proj(self.matvec(proj(v)))       # noqa: E731
            M_mv = lambda r: proj(self._precond(proj(r)))     # noqa: E731
            x, report = fgmres(A_mv, proj(self.b), M=M_mv, x0=x0,
                               rtol=rtol, restart=restart, maxiter=maxiter,
                               recycle=rs)
            x = proj(x)
        else:
            x, report = fgmres(self.matvec, self.b, M=self._precond, x0=x0,
                               rtol=rtol, restart=restart, maxiter=maxiter,
                               recycle=rs)
        report.precond_summary = self._precond.summary()
        if self.verbose:
            print(f"  {report}")
        self.report = report
        self._lineage.x = x
        return {s.name: x[s.offset:s.stop].reshape(-1, 3)
                for s in self.layout.slots}

    # -- viscoelastic hook --------------------------------------------

    def rebuild_for_materials(self, material_map: dict) -> "AssembledH":
        """The same compressed geometry at new materials, sharing this
        assembly's pairs and its solve lineage (warm start, recycle
        space).

        The block-Gauss-Seidel ladder is carried over as well while the
        step from the materials it was BUILT at stays within
        ``defaults.PRECOND_REUSE_MAX_STEP``: it is the expensive part of
        a rebuild (dense LU / HODLR per super-block) and it is only a
        preconditioner, so a stale one costs iterations, never accuracy
        -- the solve still applies the new operator and still confirms
        the true residual. The whole object is kept, its off-diagonal
        Gauss-Seidel couplings included, so what is applied is exactly
        the previous materials' approximate inverse.

        How many solves that keeps per build is also what decides the
        ladder's rung above the dense cap, since the rungs trade build
        against solve: ``defaults.PRECOND_RUNG_ABOVE_DENSE`` carries the
        crossover, and this policy's ~3.5 solves per build is far below
        it at every size measured.
        """
        new = AssembledH(self.system, self.eps, self.opts, self.verbose,
                         jump=self.jump, deflate=self.deflate,
                         storage=self.storage, sweep=self.sweep,
                         _shared=(self._pairs, self._tree_cache),
                         _lineage=self._lineage)
        for name, mat in material_map.items():
            if name not in new.materials:
                raise KeyError(f"unknown region '{name}'")
            new.materials[name] = mat
        # calibration + RHS are material-dependent: recompute AFTER the
        # material override (cheap: constant-field matvecs).
        new._refresh_material_state()
        if self._precond is not None and material_step(
                self._precond_materials, new.materials) \
                <= defaults.PRECOND_REUSE_MAX_STEP:
            new._precond = self._precond
            new._precond_materials = self._precond_materials
            new._precond_caps = self._precond_caps   # or the next solve
            #                                          rebuilds what it kept
        return new

    # -- stats -------------------------------------------------------

    def nbytes(self) -> int:
        return sum(p.nbytes() for p in self._pairs.values())
