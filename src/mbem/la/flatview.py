"""Flat per-material view of one compressed pair, and its batched kernels.

A ``FlatView`` is the material-combined operator of one (field, source,
kernel) pair for ONE coefficient vector, laid out as flat buffers so one
numba kernel applies every block:

  low-rank block i < n_lr :  U_i (3 nr_i, k_i) row-major at ``u_ptr[i]`` of
                             ``U_flat``; V_i (3 nc_i, k_i) at ``v_ptr[i]``
                             of ``V_flat``
  dense block   i >= n_lr :  D_i (3 nr_i, 3 nc_i) row-major at
                             ``d_ptr[i - n_lr]`` of ``D_flat``

with the block's global row / column DOFs, ascending, at
``row_ptr[i]:row_ptr[i+1]`` of ``row_idx`` and ``col_ptr[i]:col_ptr[i+1]``
of ``col_idx`` (int32).

The matvec is bitwise deterministic at any thread count, and identical
whether it ran parallel or serial. Phase 1 forms ``w_i = V_i^T x[cols_i]``
per low-rank block (one thread per block, the column order fixed). Phase 2
OWNS ROWS: the output DOFs are cut into contiguous ranges of equal work
(``FLATVIEW_ROW_CHUNKS``, sized by ``FLATVIEW_MIN_CHUNK_WORK``, fixed by
the view and not by the thread count); each chunk visits the (block, local
row range) entries of its CSR in block order, and no two chunks write the
same row -- no per-thread accumulator, no reduction, no race.

Dense blocks -- the partition's near-field leaves and every admissible
block applied exactly (rank-capped, small certificate fallbacks, a
combined certificate failed for this material) -- are evaluated straight
into ``D_flat`` by ``dense_leaves``, one parallel kernel over all of them
with the material coefficients applied in-loop as ``t_matrix_direct``
does. The near field is therefore never stored per basis: a material
change re-evaluates the leaves from the kernels (their kernel work only;
~12 s at 1M unknowns) instead of holding B copies of them, and an
interface pair evaluates its near field once per region material.

The ``parallel=True`` kernels here are called from the main thread only
(rule 8); the per-block ACA runs on Python threads through the nogil
kernels of ``aca_numba``, not through these.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .. import defaults
from ..kernels import KERNEL_T, KERNEL_U
from ..kernels.tri_kernels import _t_basis_pair, _tri_frame, _u_basis_pair


def _dof_idx(elems: np.ndarray) -> np.ndarray:
    return (3 * np.asarray(elems, dtype=np.int64)[:, None]
            + np.arange(3)[None, :]).ravel()


# ---------------------------------------------------------------------
# dense blocks: one kernel over all (block, source column) work items
# ---------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _t_dense_blocks(x_field, tri_verts, normals, eps_arr, lrow_ptr, lrows,
                    lcol_ptr, lcols, d_ptr, D_flat, c):
    """T-kernel dense blocks with coefficients ``c`` (6,) in-loop. Work
    item = one source column of one block (its frame set up once, every
    field row evaluated); items are balanced to within one column."""
    n_items = lcol_ptr[lcol_ptr.shape[0] - 1]
    for it in prange(n_items):
        L = np.searchsorted(lcol_ptr, it, side="right") - 1
        j = it - lcol_ptr[L]
        s = lcols[it]
        nr = lrow_ptr[L + 1] - lrow_ptr[L]
        ld = 3 * (lcol_ptr[L + 1] - lcol_ptr[L])
        base = d_ptr[L] + 3 * j
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            for i in range(nr):
                for a in range(3):
                    for b in range(3):
                        D_flat[base + (3 * i + a) * ld + b] = 0.0
            continue
        eps = eps_arr[s]
        for i in range(nr):
            f = lrows[lrow_ptr[L] + i]
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for a in range(3):
                for b in range(3):
                    D_flat[base + (3 * i + a) * ld + b] = (
                        c[0] * blk[0, a, b] + c[1] * blk[1, a, b]
                        + c[2] * blk[2, a, b] + c[3] * blk[3, a, b]
                        + c[4] * blk[4, a, b] + c[5] * blk[5, a, b])


@njit(cache=True, parallel=True)
def _u_dense_blocks(x_field, tri_verts, eps_arr, lrow_ptr, lrows,
                    lcol_ptr, lcols, d_ptr, D_flat, c):
    """U-kernel twin of ``_t_dense_blocks`` (coefficients ``c`` (3,))."""
    n_items = lcol_ptr[lcol_ptr.shape[0] - 1]
    for it in prange(n_items):
        L = np.searchsorted(lcol_ptr, it, side="right") - 1
        j = it - lcol_ptr[L]
        s = lcols[it]
        nr = lrow_ptr[L + 1] - lrow_ptr[L]
        ld = 3 * (lcol_ptr[L + 1] - lcol_ptr[L])
        base = d_ptr[L] + 3 * j
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            for i in range(nr):
                for a in range(3):
                    for b in range(3):
                        D_flat[base + (3 * i + a) * ld + b] = 0.0
            continue
        eps = eps_arr[s]
        for i in range(nr):
            f = lrows[lrow_ptr[L] + i]
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for a in range(3):
                for b in range(3):
                    D_flat[base + (3 * i + a) * ld + b] = (
                        c[0] * blk[0, a, b] + c[1] * blk[1, a, b]
                        + c[2] * blk[2, a, b])


def storage_dtype(tol: float) -> np.dtype:
    """Storage precision of a view compressed at block tolerance ``tol``.

    Single once the tolerance is coarse enough to dominate float32's own
    ~6e-8 (``STORAGE_SINGLE_MIN_TOL``), double otherwise. Derived from the
    tolerance rather than chosen, so a tight build is never silently
    storage-limited. Arithmetic is float64 either way.
    """
    return (np.float32 if float(tol) >= defaults.STORAGE_SINGLE_MIN_TOL
            else np.float64)


def dense_leaves(kernel: str, x_field, tri_verts, normals, eps_arr,
                 blocks: list, coeffs,
                 dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """``(D_flat, d_ptr)`` of the dense blocks ``[(rows, cols)]`` (element
    index arrays, rows ascending) for coefficient vector ``coeffs``:
    block L is the (3 nr, 3 nc) row-major slice at ``d_ptr[L]``.

    ``dtype`` is the STORAGE precision (``storage_dtype``); the kernels
    combine their bases in float64 and the store rounds once, so this
    allocates at the final precision rather than casting afterwards --
    a cast would hold a float64 copy and a float32 copy of the largest
    buffer in the view at the same time.
    """
    c = np.ascontiguousarray(np.asarray(coeffs, dtype=float))
    n = len(blocks)
    lrow_ptr = np.zeros(n + 1, dtype=np.int64)
    lcol_ptr = np.zeros(n + 1, dtype=np.int64)
    d_ptr = np.zeros(n + 1, dtype=np.int64)
    for L, (rows, cols) in enumerate(blocks):
        lrow_ptr[L + 1] = lrow_ptr[L] + len(rows)
        lcol_ptr[L + 1] = lcol_ptr[L] + len(cols)
        d_ptr[L + 1] = d_ptr[L] + 9 * len(rows) * len(cols)
    lrows = (np.concatenate([np.asarray(r) for r, _ in blocks]).astype(np.int32)
             if n else np.zeros(0, dtype=np.int32))
    lcols = (np.concatenate([np.asarray(cc) for _, cc in blocks]).astype(np.int32)
             if n else np.zeros(0, dtype=np.int32))
    D_flat = np.empty(d_ptr[-1], dtype=dtype)
    if n:
        if kernel == KERNEL_T:
            _t_dense_blocks(x_field, tri_verts, normals, eps_arr, lrow_ptr,
                            lrows, lcol_ptr, lcols, d_ptr, D_flat, c)
        elif kernel == KERNEL_U:
            _u_dense_blocks(x_field, tri_verts, eps_arr, lrow_ptr, lrows,
                            lcol_ptr, lcols, d_ptr, D_flat, c)
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
    return D_flat, d_ptr


# ---------------------------------------------------------------------
# the flat matvec
# ---------------------------------------------------------------------

@njit(cache=True)
def _chunk_entries(row_ptr, row_idx, bounds):
    """(chunk, block, r0, r1) entries: block's local rows [r0, r1) fall in
    chunk's DOF range [bounds[c], bounds[c+1]); generated in block order."""
    nb = row_ptr.shape[0] - 1
    count = 0
    for i in range(nb):
        a = row_ptr[i]; b = row_ptr[i + 1]
        if b == a:
            continue
        c_lo = np.searchsorted(bounds, row_idx[a], side="right") - 1
        c_hi = np.searchsorted(bounds, row_idx[b - 1], side="right") - 1
        count += c_hi - c_lo + 1
    chunk = np.empty(count, np.int32); blk = np.empty(count, np.int32)
    r0 = np.empty(count, np.int32); r1 = np.empty(count, np.int32)
    e = 0
    for i in range(nb):
        a = row_ptr[i]; b = row_ptr[i + 1]
        if b == a:
            continue
        rows = row_idx[a:b]
        c_lo = np.searchsorted(bounds, rows[0], side="right") - 1
        c_hi = np.searchsorted(bounds, rows[rows.shape[0] - 1],
                               side="right") - 1
        for c in range(c_lo, c_hi + 1):
            s0 = np.searchsorted(rows, bounds[c])
            s1 = np.searchsorted(rows, bounds[c + 1])
            if s1 > s0:
                chunk[e] = c; blk[e] = i; r0[e] = s0; r1[e] = s1
                e += 1
    return chunk[:e], blk[:e], r0[:e], r1[:e]


@njit(cache=True, inline="always")
def _dot4(a, a0, b, b0, n):
    """``sum_j a[a0 + j] b[b0 + j]`` over FOUR accumulators reduced in a
    fixed order. A plain running sum is a floating-point reduction, which
    LLVM may not reorder and therefore may not vectorize (13x slower than
    BLAS gemv, measured); four independent chains vectorize and the result
    is still a function of the data alone."""
    s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0
    m = n - (n % 4)
    for j in range(0, m, 4):
        s0 += a[a0 + j] * b[b0 + j]
        s1 += a[a0 + j + 1] * b[b0 + j + 1]
        s2 += a[a0 + j + 2] * b[b0 + j + 2]
        s3 += a[a0 + j + 3] * b[b0 + j + 3]
    rest = 0.0
    for j in range(m, n):
        rest += a[a0 + j] * b[b0 + j]
    return ((s0 + s1) + (s2 + s3)) + rest


@njit(cache=True)
def _lowrank_w(i0, i1, x, col_ptr, col_idx, rank, v_ptr, w_ptr, V_flat, W):
    """``w_i = V_i^T x[cols_i]`` for the low-rank blocks [i0, i1)."""
    for i in range(i0, i1):
        k = rank[i]
        c0 = col_ptr[i]; nc = col_ptr[i + 1] - c0
        v0 = v_ptr[i]; w0 = w_ptr[i]
        for kk in range(k):
            W[w0 + kk] = 0.0
        for j in range(nc):
            xj = x[col_idx[c0 + j]]
            base = v0 + j * k
            for kk in range(k):
                W[w0 + kk] += V_flat[base + kk] * xj


@njit(cache=True)
def _apply_chunks(c0, c1, x, y, xl, n_lr, row_ptr, row_idx, col_ptr, col_idx,
                  rank, u_ptr, w_ptr, d_ptr, U_flat, D_flat, W,
                  chunk_ptr, chunk_blk, chunk_r0, chunk_r1):
    """Row chunks [c0, c1) applied into ``y``, entry by entry in block
    order. ``xl`` is scratch of at least the widest block."""
    for c in range(c0, c1):
        for e in range(chunk_ptr[c], chunk_ptr[c + 1]):
            i = chunk_blk[e]; r0 = chunk_r0[e]; r1 = chunk_r1[e]
            rp = row_ptr[i]
            if i < n_lr:
                k = rank[i]; u0 = u_ptr[i]; w0 = w_ptr[i]
                for r in range(r0, r1):
                    y[row_idx[rp + r]] += _dot4(U_flat, u0 + r * k,
                                                W, w0, k)
            else:
                cc = col_ptr[i]; nc = col_ptr[i + 1] - cc
                d0 = d_ptr[i - n_lr]
                for j in range(nc):
                    xl[j] = x[col_idx[cc + j]]
                for r in range(r0, r1):
                    y[row_idx[rp + r]] += _dot4(D_flat, d0 + r * nc,
                                                xl, 0, nc)


@njit(cache=True, parallel=True)
def _flat_matvec(x, n_rows, n_lr, row_ptr, row_idx, col_ptr, col_idx, rank,
                 u_ptr, v_ptr, w_ptr, d_ptr, U_flat, V_flat, D_flat,
                 chunk_ptr, chunk_blk, chunk_r0, chunk_r1, max_width):
    # W, y and the gathered x slice are float64 BY CONSTRUCTION: numba
    # types a bare np.empty(n) as float64, and that is what keeps every
    # accumulation double when U_flat/V_flat/D_flat are single. W is the
    # one that matters most -- it is an in-place running sum over up to
    # 3 nc terms (_lowrank_w), so typing it from V_flat.dtype, the
    # natural-looking edit, would silently make it a float32 reduction.
    W = np.empty(w_ptr[n_lr])
    for i in prange(n_lr):                 # phase 1: one thread per block
        _lowrank_w(i, i + 1, x, col_ptr, col_idx, rank, v_ptr, w_ptr,
                   V_flat, W)
    y = np.zeros(n_rows)
    n_chunks = chunk_ptr.shape[0] - 1
    for c in prange(n_chunks):             # phase 2: one thread per chunk
        _apply_chunks(c, c + 1, x, y, np.empty(max_width), n_lr, row_ptr,
                      row_idx, col_ptr, col_idx, rank, u_ptr, w_ptr, d_ptr,
                      U_flat, D_flat, W, chunk_ptr, chunk_blk, chunk_r0,
                      chunk_r1)
    return y


@njit(cache=True)
def _flat_matvec_serial(x, n_rows, n_lr, row_ptr, row_idx, col_ptr, col_idx,
                        rank, u_ptr, v_ptr, w_ptr, d_ptr, U_flat, V_flat,
                        D_flat, chunk_ptr, chunk_blk, chunk_r0, chunk_r1,
                        max_width):
    """The same two phases in one thread, for a view whose work is below
    ``FLATVIEW_PARALLEL_MIN_WORK``: an OpenMP fork/join costs ~0.1-0.3 ms,
    which a small pair's matvec (and every preconditioner lower term) pays
    many times per iteration."""
    W = np.empty(w_ptr[n_lr])
    _lowrank_w(0, n_lr, x, col_ptr, col_idx, rank, v_ptr, w_ptr, V_flat, W)
    y = np.zeros(n_rows)
    _apply_chunks(0, chunk_ptr.shape[0] - 1, x, y, np.empty(max_width), n_lr,
                  row_ptr, row_idx, col_ptr, col_idx, rank, u_ptr, w_ptr,
                  d_ptr, U_flat, D_flat, W, chunk_ptr, chunk_blk, chunk_r0,
                  chunk_r1)
    return y


class FlatView:
    """One material's operator of a pair in flat form (module docstring).

    ``lr_blocks``: ``[(rows, cols, U, V)]`` with element index arrays and
    the factors of ``U @ V.T`` on their DOFs; ``dense_blocks``:
    ``[(rows, cols)]`` whose ``(3 nr, 3 nc)`` values are already in
    ``D_flat`` at ``d_ptr`` (``dense_leaves``). Rows within a block must
    be ascending (a cluster's indices are); a low-rank block given
    otherwise is permuted here.
    """

    def __init__(self, shape: tuple, lr_blocks: list, dense_blocks: list,
                 D_flat: np.ndarray, d_ptr: np.ndarray,
                 dtype=np.float64):
        # One dtype for ALL THREE buffers: a view with, say, a float64
        # U_flat and a float32 D_flat is silently legal (numba just
        # compiles another specialization) and would differ from its
        # siblings with nothing to catch it.
        self.dtype = np.dtype(dtype)
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_lr = len(lr_blocks)
        self.n_dense = len(dense_blocks)
        nb = self.n_lr + self.n_dense

        row_lists, col_lists = [], []
        self.rank = np.zeros(nb, dtype=np.int32)
        self.u_ptr = np.zeros(nb + 1, dtype=np.int64)
        self.v_ptr = np.zeros(nb + 1, dtype=np.int64)
        self.w_ptr = np.zeros(nb + 1, dtype=np.int64)
        U_parts, V_parts = [], []
        for i, (rows, cols, U, V) in enumerate(lr_blocks):
            rows = np.asarray(rows)
            if rows.size > 1 and np.any(np.diff(rows) < 0):
                order = np.argsort(rows, kind="stable")
                rows = rows[order]
                U = U.reshape(rows.size, 3, -1)[order].reshape(U.shape)
            k = U.shape[1]
            self.rank[i] = k
            self.u_ptr[i + 1] = self.u_ptr[i] + U.size
            self.v_ptr[i + 1] = self.v_ptr[i] + V.size
            self.w_ptr[i + 1] = self.w_ptr[i] + k
            U_parts.append(np.ascontiguousarray(U, dtype=self.dtype).ravel())
            V_parts.append(np.ascontiguousarray(V, dtype=self.dtype).ravel())
            row_lists.append(_dof_idx(rows))
            col_lists.append(_dof_idx(cols))
        for i in range(self.n_lr, nb):
            self.u_ptr[i + 1] = self.u_ptr[i]
            self.v_ptr[i + 1] = self.v_ptr[i]
            self.w_ptr[i + 1] = self.w_ptr[i]
        for rows, cols in dense_blocks:
            rows = np.asarray(rows)
            if rows.size > 1 and np.any(np.diff(rows) < 0):
                raise ValueError("dense block rows must be ascending: "
                                 "D_flat already holds that row order")
            row_lists.append(_dof_idx(rows))
            col_lists.append(_dof_idx(cols))
        self.U_flat = (np.concatenate(U_parts) if U_parts
                       else np.zeros(0, dtype=self.dtype))
        self.V_flat = (np.concatenate(V_parts) if V_parts
                       else np.zeros(0, dtype=self.dtype))
        # ``dense_leaves`` already allocates at the storage dtype, so this
        # is a no-op cast on the normal path and a safety net otherwise.
        self.D_flat = np.ascontiguousarray(D_flat, dtype=self.dtype)
        self.d_ptr = np.asarray(d_ptr, dtype=np.int64)
        if self.d_ptr.size != self.n_dense + 1:
            raise ValueError("d_ptr does not match the dense block list")

        self.row_ptr = np.zeros(nb + 1, dtype=np.int64)
        self.col_ptr = np.zeros(nb + 1, dtype=np.int64)
        for i in range(nb):
            self.row_ptr[i + 1] = self.row_ptr[i] + row_lists[i].size
            self.col_ptr[i + 1] = self.col_ptr[i] + col_lists[i].size
        self.row_idx = (np.concatenate(row_lists).astype(np.int32) if nb
                        else np.zeros(0, dtype=np.int32))
        self.col_idx = (np.concatenate(col_lists).astype(np.int32) if nb
                        else np.zeros(0, dtype=np.int32))
        nr = np.diff(self.row_ptr)
        nc = np.diff(self.col_ptr)
        self.max_width = int(nc.max()) if nb else 0
        # Multiply-adds of one matvec: k (nr + nc) per low-rank block,
        # nr nc per dense one. It sets the chunk count and decides whether
        # the parallel kernel's fork/join is worth paying (``matvec``).
        k = self.rank.astype(np.int64)
        m = self.n_lr
        self.work = int((k[:m] * (nr[:m] + nc[:m])).sum()
                        + (nr[m:] * nc[m:]).sum())
        self._build_chunks()

    def _build_chunks(self) -> None:
        """Equal-work row chunks -- at most ``FLATVIEW_ROW_CHUNKS``, never
        finer than ``FLATVIEW_MIN_CHUNK_WORK`` each -- and their CSR of
        (block, local row range) entries in block order."""
        n_rows = self.shape[0]
        nb = self.n_lr + self.n_dense
        per_block = np.empty(nb, dtype=float)
        per_block[:self.n_lr] = self.rank[:self.n_lr]
        per_block[self.n_lr:] = np.diff(self.col_ptr)[self.n_lr:]
        work = np.zeros(n_rows)
        if nb:
            np.add.at(work, self.row_idx,
                      np.repeat(per_block, np.diff(self.row_ptr)))
        # Chunks are sized by WORK, not by thread count: every chunk that
        # touches a dense block re-gathers that block's x, so cutting a
        # small view into 256 chunks pays the gather 50 times over.
        n_chunks = max(1, min(defaults.FLATVIEW_ROW_CHUNKS, n_rows,
                              self.work // defaults.FLATVIEW_MIN_CHUNK_WORK))
        cum = np.cumsum(work)
        total = cum[-1] if n_rows else 0.0
        if total > 0.0:
            bounds = np.searchsorted(cum, np.linspace(0.0, total, n_chunks + 1),
                                     side="left")
        else:
            bounds = np.linspace(0, n_rows, n_chunks + 1).astype(np.int64)
        bounds = np.asarray(bounds, dtype=np.int64)
        bounds[0] = 0
        bounds[-1] = n_rows
        bounds = np.maximum.accumulate(bounds)
        chunk, blk, r0, r1 = _chunk_entries(self.row_ptr, self.row_idx, bounds)
        order = np.argsort(chunk, kind="stable")
        self.chunk_blk = blk[order]
        self.chunk_r0 = r0[order]
        self.chunk_r1 = r1[order]
        counts = np.bincount(chunk, minlength=n_chunks)
        self.chunk_ptr = np.zeros(n_chunks + 1, dtype=np.int64)
        np.cumsum(counts, out=self.chunk_ptr[1:])

    # -- operations -----------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """The whole view applied by ONE kernel (main thread, rule 8);
        serially below ``FLATVIEW_PARALLEL_MIN_WORK`` multiply-adds, where
        the fork/join costs more than the work. Both kernels apply the
        same chunks in the same order, so the result does not depend on
        which one ran, nor on the thread count."""
        x = np.ascontiguousarray(x, dtype=float)
        if x.shape != (self.shape[1],):
            raise ValueError(f"x has shape {x.shape}, operator {self.shape}")
        kernel = (_flat_matvec if self.work >= defaults.FLATVIEW_PARALLEL_MIN_WORK
                  else _flat_matvec_serial)
        return kernel(x, self.shape[0], self.n_lr, self.row_ptr,
                      self.row_idx, self.col_ptr, self.col_idx,
                      self.rank, self.u_ptr, self.v_ptr, self.w_ptr,
                      self.d_ptr, self.U_flat, self.V_flat, self.D_flat,
                      self.chunk_ptr, self.chunk_blk, self.chunk_r0,
                      self.chunk_r1, self.max_width)

    def blocks(self):
        """Yield ``(rdofs, cdofs, A, V)`` per block -- ``A @ V.T`` for a
        low-rank block, ``(D, None)`` for a dense one -- as views into
        the flat buffers (the block-loop form of this operator)."""
        for i in range(self.n_lr + self.n_dense):
            rdofs = self.row_idx[self.row_ptr[i]:self.row_ptr[i + 1]]
            cdofs = self.col_idx[self.col_ptr[i]:self.col_ptr[i + 1]]
            if i < self.n_lr:
                k = int(self.rank[i])
                U = self.U_flat[self.u_ptr[i]:self.u_ptr[i + 1]].reshape(
                    rdofs.size, k)
                V = self.V_flat[self.v_ptr[i]:self.v_ptr[i + 1]].reshape(
                    cdofs.size, k)
                yield rdofs, cdofs, U, V
            else:
                d = i - self.n_lr
                D = self.D_flat[self.d_ptr[d]:self.d_ptr[d + 1]].reshape(
                    rdofs.size, cdofs.size)
                yield rdofs, cdofs, D, None

    def to_dense(self) -> np.ndarray:
        """The view as a dense float64 matrix.

        The rank-k product is taken in float64 even when the factors are
        stored single: two float32 operands would dispatch to sgemm, whose
        inner sum over k accumulates at float32, and this is the reference
        every dense-parity gate compares against -- it has to sum the same
        way ``matvec`` does, which goes through ``_dot4`` against a float64
        partner.
        """
        M = np.zeros(self.shape)
        for rdofs, cdofs, A, V in self.blocks():
            if V is None:
                M[np.ix_(rdofs, cdofs)] += A
            else:
                M[np.ix_(rdofs, cdofs)] += (np.asarray(A, dtype=np.float64)
                                            @ np.asarray(V, dtype=np.float64).T)
        return M

    @property
    def ranks(self) -> np.ndarray:
        return self.rank[:self.n_lr]

    def lowrank_nbytes(self) -> int:
        return int(self.U_flat.nbytes + self.V_flat.nbytes)

    def dense_nbytes(self) -> int:
        return int(self.D_flat.nbytes)

    def nbytes(self) -> int:
        """Factors, dense blocks, index and pointer arrays."""
        return int(self.lowrank_nbytes() + self.dense_nbytes()
                   + self.row_idx.nbytes + self.col_idx.nbytes
                   + self.row_ptr.nbytes + self.col_ptr.nbytes
                   + self.rank.nbytes + self.u_ptr.nbytes + self.v_ptr.nbytes
                   + self.w_ptr.nbytes + self.d_ptr.nbytes
                   + self.chunk_ptr.nbytes + self.chunk_blk.nbytes
                   + self.chunk_r0.nbytes + self.chunk_r1.nbytes)
