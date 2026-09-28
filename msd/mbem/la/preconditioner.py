"""Block Gauss-Seidel preconditioner over coupled super-blocks.

What shares a diagonal super-block is the GROUPING policy
(``defaults.PRECOND_GROUPING``), the one choice that sets the iteration
count:

``"patch"`` -- one super-block per patch. An INTERFACE patch's pair of
slots goes in together, so its diagonal block is the LOCAL 2x2
transmission system in its (u, t) unknowns,

    [ D + sA*H^A_pp    -sA*G^A_pp ]
    [ D + sB*H^B_pp    -sB*G^B_pp ]

(D the collocation diagonal of ``equations.term_diagonal``), which is
well-posed even where a first-kind G block alone is nearly singular (it
discretizes the locally well-posed two-sided transmission problem) and
contrast-robust. Non-interface slots form their own super-blocks
(second-kind D + sigma H for u-slots; -sigma G for a
prescribed-displacement t-slot).

``"region"`` -- one super-block per region: every patch of that region
together with the interfaces that close it, each interface owned by the
LAST incident region so a star graph's host block is its outer boundary
alone and every inclusion block is a closed surface. Its diagonal block
is the whole single-region exterior problem, so only the region-to-region
coupling is left to the sweep.

Either way one forward GS sweep in block order (regions in model order,
host first): the region graph of the models this solves is a path or a
star, so the sweep approximates chain elimination.

Diagonal solves form a two-rung ladder, ordered by what a rung COSTS and
not by how approximate it is:
  * size <= max_dense  : exact dense assembly (fast numba kernels) + LU,
                         the cheapest rung to BUILD at every size it
                         fits (``dense_rung_max_dof``); what it spends
                         is n^2 x 8 B of stored factor and a triangular
                         solve per iteration;
  * larger             : dense LU on cluster-tree chunks (block Jacobi),
                         O(N x chunk) in build and memory at ANY size.
                         It is the only rung whose iteration count grows
                         with N, and that growth is bounded and measured
                         (``defaults.GMRES_ITER_GROWTH_ALPHA``), not a
                         reason to prefer a rung that does not build.

``above_dense`` (``defaults.PRECOND_RUNG_ABOVE_DENSE``) names the second
one. Setting it to ``"hodlr"`` inserts the HODLR rung -- a loose-tolerance
HODLR solver over the local system, an order less memory than the dense
rung and a cheaper apply WHERE THE BLOCK COMPRESSES -- between the two,
up to ``hodlr_max`` DOFs. It is not the default because on these
geometries such a block is the exception: weak admissibility splits a
cluster into halves that TOUCH, the off-diagonal blocks are near field
and enter exactly, and the build is superlinear where the dense rung's
is not (the constant carries the ladder measurement). Choose it when
memory, not time, is what binds.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .. import defaults
from ..kernels import KERNEL_T, kernel_coeffs
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from ..model.core import BCType
from ..model.equations import term_diagonal
from .cluster import build_cluster_tree
from .hodlr import HodlrSolver


def group_slots(model, layout, grouping: str) -> list:
    """The super-blocks of a grouping policy, as lists of slots in sweep
    order (see the module docstring for what each policy means).

    ``"region"`` assigns every patch to the LAST region incident to it in
    model order, which for the star graphs this solves (host + 2-4
    inclusions) puts each interface in its inclusion's block: the host
    block is then its own outer boundary and each inclusion block a
    closed surface, and sweeping the regions in model order solves the
    host first, as chain elimination on a star wants.
    """
    if grouping == "patch":
        groups, seen = [], set()
        for slot in layout.slots:
            if id(slot.patch) in seen:
                continue
            if slot.patch.bc is BCType.INTERFACE:
                seen.add(id(slot.patch))
                groups.append([layout.slot(slot.patch, "u"),
                               layout.slot(slot.patch, "t")])
            else:
                groups.append([slot])
        return groups
    if grouping == "region":
        owner = {}
        for k, region in enumerate(model.regions):
            for patch in region.patches:
                owner[id(patch)] = k
        groups = []
        for k in range(len(model.regions)):
            slots = [s for s in layout.slots if owner[id(s.patch)] == k]
            if slots:
                groups.append(slots)
        return groups
    raise ValueError(f"unknown preconditioner grouping {grouping!r} "
                     f"(expected 'patch' or 'region')")


def dense_rung_max_dof(
        max_dense: int = defaults.MAX_DENSE_PRECOND_DOF,
        ram_fraction: float = defaults.PRECOND_DENSE_RAM_FRACTION) -> int:
    """Largest super-block the dense-LU rung takes on THIS machine: the
    DOF cap, lowered where one block's stored factorization (n^2 x 8 B)
    would exceed ``ram_fraction`` of physical RAM. The ladder holds every
    block's factorization at once and the build peaks at ~2.1x one of
    them, so the cap is a per-block share, not the whole budget."""
    from ..estimate import total_ram_bytes    # ..estimate imports la.*

    ram = total_ram_bytes()
    if ram is None:
        return max_dense
    return min(max_dense, int(np.sqrt(ram_fraction * ram / 8.0)))


def _run_start(sel: np.ndarray):
    """``sel[0]`` when the (ascending) positions are a unit-stride run,
    else None -- i.e. when they can be addressed by a slice."""
    if sel.size and sel[-1] - sel[0] == sel.size - 1:
        return int(sel[0])
    return None


class _SBEvaluator:
    """Dense evaluator of one super-block's local matrix.

    The block's rows are the concatenated collocation points of its
    slots' patches and its columns their concatenated slots, evaluated
    with the exact kernels at each term's own material and carrying the
    term's own collocation diagonal (``equations.term_diagonal``, spread
    over the element's own columns as ``add_block_diagonal`` spreads it,
    which at P0 -- all this backend assembles -- is the plain block
    diagonal), so the ladder inverts the operator the outer iteration
    actually applies.

    Local layout is by UNIT -- one unit per (slot group, element), d DOFs
    each -- because the HODLR and block-Jacobi rungs cluster and pivot in
    units, not in single DOFs. Slots of ONE patch share a unit, so an
    interface element's (u, t) pair is one 2x2 transmission unit and
    d = 3 x slots per group. A super-block whose patches carry DIFFERENT
    slot counts (a region block mixes u-only patches with (u, t)
    interfaces) has no such uniform d, so there every SLOT is its own
    group at d = 3; the (u, t) pair is recovered by the cluster tree
    instead, since both units sit at the same collocation point and a
    median split never separates coincident points.

    ``concat_perm`` maps this unit layout to the super-block's
    slot-concatenated layout (what the GS sweep passes in).
    """

    def __init__(self, assembled, slots, terms):
        slots = list(slots)
        by_patch, order = {}, []
        for s in slots:
            if id(s.patch) not in by_patch:
                by_patch[id(s.patch)] = []
                order.append(id(s.patch))
            by_patch[id(s.patch)].append(s)
        groups = [by_patch[k] for k in order]
        if len({len(g) for g in groups}) > 1:
            groups = [[s] for s in slots]
        self.groups = groups
        self.d = 3 * len(groups[0])

        arrays = kb.MeshArrays()
        self.points, self.tri_verts, self.normals, self.eps = [], [], [], []
        n_elem = []
        for g in groups:
            patch = g[0].patch
            self.points.append(np.ascontiguousarray(
                arrays.collocation_points(patch)))
            tv, nrm = arrays.source_arrays(patch.mesh)
            self.tri_verts.append(tv)
            self.normals.append(nrm)
            self.eps.append(assembled.eps_for(patch))
            n_elem.append(patch.n_triangles)
        self.n_elem = n_elem
        self.unit0 = np.concatenate([[0], np.cumsum(n_elem)])
        self.n_units = int(self.unit0[-1])
        self.centroids = (self.points[0] if len(groups) == 1
                          else np.ascontiguousarray(
                              np.concatenate(self.points)))
        self.group_id = np.repeat(np.arange(len(groups)), n_elem)
        self.elem_id = np.concatenate([np.arange(n) for n in n_elem])

        where = {s.name: (gi, 3 * idx)
                 for gi, g in enumerate(groups) for idx, s in enumerate(g)}
        calib = getattr(assembled, "calib", None)
        self.recipes = []
        for t in terms:
            g_row, r_off = where[t.row.name]
            g_col, c_off = where[t.col.name]
            mat = assembled.materials[t.region.name]
            self.recipes.append((
                t.kernel, kernel_coeffs(t.kernel, mat), t.scale,
                g_row, g_col, r_off, c_off, term_diagonal(t, calib)))

        # unit -> concat permutation: unit layout position
        # d*(unit0[g] + e) + 3*idx + a  holds concat entry
        # local_offset(slot) + 3*e + a.
        local_off, off = {}, 0
        for s in slots:
            local_off[s.name] = off
            off += s.size
        perm = np.empty(self.d * self.n_units, dtype=int)
        for gi, g in enumerate(groups):
            e = np.arange(n_elem[gi])
            base = self.d * (self.unit0[gi] + e)
            for idx, s in enumerate(g):
                for a in range(3):
                    perm[base + 3 * idx + a] = local_off[s.name] + 3 * e + a
        self.concat_perm = perm

    def eval_block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(d nr, d nc) exact sub-matrix for UNIT index arrays."""
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        d = self.d
        out = np.zeros((d * rows.size, d * cols.size))
        gr, er = self.group_id[rows], self.elem_id[rows]
        gc, ec = self.group_id[cols], self.elem_id[cols]
        a3 = np.arange(3)
        r_cache, c_cache = {}, {}
        for kernel, coeffs, scale, g_row, g_col, r_off, c_off, D \
                in self.recipes:
            if g_row not in r_cache:
                sel = np.flatnonzero(gr == g_row)
                r_cache[g_row] = (sel, er[sel], np.ascontiguousarray(
                    self.points[g_row][er[sel]]))
            sel_r, e_r, xf = r_cache[g_row]
            if g_col not in c_cache:
                sel = np.flatnonzero(gc == g_col)
                e = ec[sel]
                c_cache[g_col] = (
                    sel, e,
                    np.ascontiguousarray(self.tri_verts[g_col][e]),
                    np.ascontiguousarray(self.normals[g_col][e]),
                    np.ascontiguousarray(self.eps[g_col][e]))
            sel_c, e_c, tv, nv, ee = c_cache[g_col]
            if sel_r.size == 0 or sel_c.size == 0:
                continue
            if kernel == KERNEL_T:
                blk = tk.t_matrix_direct(xf, tv, nv, ee, *coeffs)
            else:
                blk = tk.u_matrix_direct(xf, tv, ee, *coeffs)
            # Accumulate one (a, b) component at a time, and through a
            # SLICE wherever the group's units are a run of positions --
            # which is every single-group block and every whole-block
            # evaluation. A slice is a view, so the only temporary is one
            # ninth of the block; scattering the whole (3nr, 3nc) block by
            # index arrays instead costs a gather, a sum and a scatter of
            # the FULL block (measured: +1.9 GB on a 0.47 GB block).
            p_r, p_c = _run_start(sel_r), _run_start(sel_c)
            if p_r is not None and p_c is not None:
                r0, c0 = d * p_r + r_off, d * p_c + c_off
                r1, c1 = r0 + d * sel_r.size, c0 + d * sel_c.size
                for a in range(3):
                    for b in range(3):
                        out[r0 + a:r1:d, c0 + b:c1:d] += scale * blk[a::3,
                                                                     b::3]
            else:
                ridx, cidx = d * sel_r + r_off, d * sel_c + c_off
                for a in range(3):
                    for b in range(3):
                        out[(ridx + a)[:, None], cidx + b] += \
                            scale * blk[a::3, b::3]
            if D is not None:
                # the free term lands where the collocation point and the
                # column node are the same element of the same patch
                ii, jj = np.nonzero(e_r[:, None] == e_c[None, :])
                rr = d * sel_r[ii][:, None] + r_off + a3
                cc = d * sel_c[jj][:, None] + c_off + a3
                out[rr[:, :, None], cc[:, None, :]] += D[e_r[ii]]
        return out


class _SuperBlock:
    def __init__(self, slots):
        self.slots = list(slots)
        self.global_idx = np.concatenate([
            np.arange(s.offset, s.stop) for s in self.slots])
        self.size = int(self.global_idx.size)
        self.local_offset = {}
        off = 0
        for s in self.slots:
            self.local_offset[s.name] = off
            off += s.size
        self.solve_fn = None
        self.terms = []         # terms with row AND col in this block
        self.lower_terms = []   # terms with col in an earlier super-block
        self.rung = None        # "dense_lu" | "hodlr" | "block_jacobi"
        self.build_s = 0.0
        self.detail: dict = {}  # rung-specific: HODLR max rank, BJ chunks


class BlockGaussSeidel:
    """Right preconditioner z = M(r) for an AssembledH system.

    ``grouping`` names what shares a diagonal super-block
    (``defaults.PRECOND_GROUPING``; module docstring). ``summary()``
    reports it together with which rung each super-block landed on and
    what its build cost, so a solve report can say why an iteration count
    moved (a grouping, or a super-block crossing a rung boundary) without
    re-running."""

    def __init__(self, assembled,
                 max_dense: int | None = None,
                 hodlr_tol: float = defaults.HODLR_PRECOND_TOL,
                 hodlr_max: int = defaults.PRECOND_HODLR_MAX_DOF,
                 bj_chunk: int | None = None,
                 grouping: str = defaults.PRECOND_GROUPING,
                 above_dense: str = defaults.PRECOND_RUNG_ABOVE_DENSE,
                 verbose: bool = False):
        # None = the machine's own cap; a caller passing one FORCES the
        # rung boundary (what the gates do to exercise a lower rung).
        if max_dense is None:
            max_dense = dense_rung_max_dof()
        # Read in the BODY, not in the signature. A default bound at def
        # time cannot be moved by rebinding `defaults`, so a sweep written
        # that way reports the chunk it set and measures the chunk that was
        # compiled in -- and is invisible at the default, where the two
        # agree. `check_precond_rungs` pins that the knob is live.
        if bj_chunk is None:
            bj_chunk = defaults.PRECOND_BJ_CHUNK_DOF
        if above_dense not in ("block_jacobi", "hodlr"):
            raise ValueError(
                f"unknown rung above the dense cap {above_dense!r} "
                f"(expected 'block_jacobi' or 'hodlr')")
        self.max_dense = int(max_dense)
        self.bj_chunk = int(bj_chunk)
        self.above_dense = above_dense
        self.asm = assembled
        # An all-Neumann model (``deflate``) has a SINGULAR region
        # operator -- the rigid translations the solve projects out -- and
        # a region super-block IS that operator, so its factorization is
        # meaningless, while the patch blocks stay second-kind and
        # invertible. The grouping is a performance choice, so this falls
        # back instead of refusing; ``summary()`` reports what was used.
        if grouping == "region" and getattr(assembled, "deflate", False):
            grouping = "patch"
        self.grouping = grouping
        self.sbs = [_SuperBlock(g) for g in
                    group_slots(assembled.system.model, assembled.layout,
                                grouping)]

        sb_of_slot = {}
        for k, sb in enumerate(self.sbs):
            for s in sb.slots:
                sb_of_slot[s.name] = k

        for term in assembled.system.terms:
            kr = sb_of_slot[term.row.name]
            kc = sb_of_slot[term.col.name]
            if kr == kc:
                self.sbs[kr].terms.append(term)
            elif kc < kr:
                self.sbs[kr].lower_terms.append(term)

        # ---- diagonal solves: the rung ladder ----
        t_start = time.perf_counter()
        for sb in self.sbs:
            t_sb = time.perf_counter()
            ev = _SBEvaluator(assembled, sb.slots, sb.terms)
            if sb.size <= max_dense:
                all_units = np.arange(ev.n_units)
                D_inter = ev.eval_block(all_units, all_units)
                lu = lu_factor(D_inter)
                # released before the NEXT block is evaluated: at the
                # cap this matrix is several GB and lu_factor already
                # holds its own copy of it.
                del D_inter
                perm = ev.concat_perm

                def solve_fn(r, lu=lu, perm=perm):
                    return _permuted_lu_solve(lu, r, perm)

                sb.solve_fn = solve_fn
                sb.rung = "dense_lu"
                if verbose:
                    print(f"  SB {[s.name for s in sb.slots]}: dense LU "
                          f"({sb.size} DOFs)")
            elif above_dense == "hodlr" and sb.size <= hodlr_max:
                hod = HodlrSolver(ev.eval_block, ev.centroids, ev.d,
                                  tol=hodlr_tol,
                                  leaf_elems=defaults.HODLR_LEAF_ELEMS)
                perm = ev.concat_perm

                def solve_fn(r, hod=hod, perm=perm):
                    return _permuted_solver(hod.solve, r, perm)

                sb.solve_fn = solve_fn
                sb.rung = "hodlr"
                sb.detail = {"hodlr_max_rank": int(hod.max_rank())}
                if verbose:
                    print(f"  SB {[s.name for s in sb.slots]}: HODLR "
                          f"({sb.size} DOFs, tol {hodlr_tol:g}); "
                          f"{hod.rank_summary()}")
            else:
                # Cluster BLOCK-JACOBI: dense LU on the diagonal blocks
                # of a cluster-tree chunking (chunks of <= bj_chunk
                # DOFs). Build memory and time are O(N x chunk) at ANY
                # super-block size: this is the rung that scales to
                # 1e5-1e6-element patches, where the HODLR build itself
                # (root rank ~ sqrt(N), dense half-matrix fallback)
                # becomes the wall. Costs outer iterations that GROW
                # with N; the diagonal chunks still capture the
                # near-singular local physics.
                chunk_units = max(32, bj_chunk // ev.d)
                tree = build_cluster_tree(ev.centroids, chunk_units)
                chunks: list = []

                def _leaves(node):
                    if node.is_leaf:
                        chunks.append(node.indices)
                    else:
                        _leaves(node.left)
                        _leaves(node.right)

                _leaves(tree)
                lus = []
                for ch in chunks:
                    D = ev.eval_block(ch, ch)
                    dof = (ev.d * ch[:, None]
                           + np.arange(ev.d)[None, :]).ravel()
                    lus.append((dof, lu_factor(D)))
                perm = ev.concat_perm

                def solve_fn(r, lus=lus, perm=perm):
                    r_inter = r[perm]
                    z_inter = np.empty_like(r_inter)
                    _bj_solve(lus, r_inter, z_inter)
                    z = np.empty_like(z_inter)
                    z[perm] = z_inter
                    return z

                sb.solve_fn = solve_fn
                sb.rung = "block_jacobi"
                sb.detail = {"chunks": len(chunks)}
                if verbose:
                    print(f"  SB {[s.name for s in sb.slots]}: cluster "
                          f"block-Jacobi ({sb.size} DOFs, "
                          f"{len(chunks)} chunks)")
            sb.build_s = time.perf_counter() - t_sb
        self.build_s = time.perf_counter() - t_start

    def diagonal_block(self, k: int) -> np.ndarray:
        """Super-block ``k``'s diagonal matrix, dense, in the
        slot-concatenated layout -- exactly what its rung approximates.

        Re-evaluated from the kernels (the rungs keep factorizations, not
        the matrix), so this is what a gate compares against the
        assembled operator's own sub-block: the two must agree entry for
        entry, or the ladder is preconditioning a different operator than
        the one being iterated."""
        sb = self.sbs[k]
        ev = _SBEvaluator(self.asm, sb.slots, sb.terms)
        units = np.arange(ev.n_units)
        block = ev.eval_block(units, units)
        out = np.empty_like(block)
        out[ev.concat_perm[:, None], ev.concat_perm] = block
        return out

    def summary(self) -> dict:
        """{"grouping", "above_dense", "max_dense", "bj_chunk", "build_s":
        total wall, "super_blocks": [{"slots", "size", "rung", "build_s",
        ...rung detail}]} in sweep order; JSON-ready. The policy and the
        caps are reported because the dense one is resolved per MACHINE
        (``dense_rung_max_dof``), so a rung that moved between two runs
        is readable from the record."""
        return {
            "grouping": self.grouping,
            "above_dense": self.above_dense,
            "max_dense": self.max_dense,
            "bj_chunk": self.bj_chunk,
            "build_s": self.build_s,
            "super_blocks": [
                {"slots": [s.name for s in sb.slots], "size": sb.size,
                 "rung": sb.rung, "build_s": sb.build_s, **sb.detail}
                for sb in self.sbs],
        }

    def __call__(self, r: np.ndarray) -> np.ndarray:
        z = np.zeros_like(r)
        for sb in self.sbs:
            rk = r[sb.global_idx].astype(z.dtype, copy=True)
            for term in sb.lower_terms:
                pair = self.asm.pair_for(term)
                coeffs = kernel_coeffs(term.kernel,
                                       self.asm.materials[term.region.name])
                xseg = z[term.col.offset:term.col.stop]
                contrib = term.scale * pair.matvec(coeffs, xseg)
                r0 = sb.local_offset[term.row.name]
                rk[r0:r0 + term.row.size] -= contrib
            z[sb.global_idx] = sb.solve_fn(rk)
        return z


_APPLY_POOL = None


def _apply_pool():
    """The process-wide pool the block-Jacobi apply runs on, or None.

    Built once and reused: a solve applies the preconditioner 100-200 times
    over several super-blocks, so a pool per apply would pay thread creation
    a thousand times over. ``defaults.PRECOND_APPLY_THREADS`` is read HERE
    rather than captured, so the count is a live knob.

    What runs on it is ONLY ``scipy.linalg.lu_solve``, never a numba
    ``parallel=True`` kernel (CLAUDE.md rule 8, the macOS workqueue crash);
    and every apply joins the pool before returning, so the Gauss-Seidel
    off-diagonal matvecs that follow are back on the main thread.
    """
    global _APPLY_POOL
    n = int(defaults.PRECOND_APPLY_THREADS)
    if n <= 1:
        return None
    if _APPLY_POOL is None or _APPLY_POOL._max_workers != n:
        if _APPLY_POOL is not None:
            _APPLY_POOL.shutdown(wait=True)
        _APPLY_POOL = ThreadPoolExecutor(max_workers=n,
                                         thread_name_prefix="bj-apply")
    return _APPLY_POOL


def _bj_solve(lus, r_inter, z_inter) -> None:
    """``z_inter[dof] = lu_solve(lu, r_inter[dof])`` over every chunk.

    The chunks own DISJOINT index sets, so this is a scatter with no
    reduction and threading it is bitwise identical to the loop -- which
    is what ``check_bj_rung`` pins, and why the partition is STATIC and
    CONTIGUOUS: each worker runs its own chunks in the serial order.
    ``scipy``'s ``lu_solve`` releases the GIL (measured against a
    GIL-bound control through the same harness), and with one right-hand
    side it is a level-2 solve that LAPACK never threads itself, so there
    is nothing to oversubscribe. Several right-hand sides would make it
    level 3 and that would stop being true.
    """
    pool = _apply_pool() if len(lus) >= defaults.PRECOND_APPLY_MIN_CHUNKS \
        else None
    if pool is None:
        for dof, lu in lus:
            z_inter[dof] = lu_solve(lu, r_inter[dof])
        return

    def run(rng):
        for dof, lu in lus[rng[0]:rng[1]]:
            z_inter[dof] = lu_solve(lu, r_inter[dof])

    w = min(pool._max_workers, len(lus))
    cut = [len(lus) * i // w for i in range(w + 1)]
    list(pool.map(run, list(zip(cut[:-1], cut[1:]))))   # joins before return


def _permuted_lu_solve(lu, r_concat, perm):
    """Solve the interleaved-layout LU for a concat-layout RHS."""
    r_inter = np.empty_like(r_concat)
    r_inter[np.arange(perm.size)] = r_concat[perm]
    x_inter = lu_solve(lu, r_inter)
    x_concat = np.empty_like(x_inter)
    x_concat[perm] = x_inter
    return x_concat


def _permuted_solver(solve, r_concat, perm):
    r_inter = r_concat[perm].copy()
    x_inter = solve(r_inter)
    x_concat = np.empty_like(x_inter)
    x_concat[perm] = x_inter
    return x_concat
