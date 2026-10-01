"""Block ACA in one nogil numba kernel.

``aca_block`` is the ACA and certificate stage of ``aca.compress_block``
-- per-basis element-block cross approximation over a SHARED kernel
evaluation (one evaluation of a pivot row or column serves all B bases),
the persistent-sample stop, one QR+SVD recompression per basis, and the
full-row / full-column certificate -- with the Python loop (~0.5 ms per
pivot, GIL-serialized) gone: pivot rows and columns are evaluated inside
the kernel through the same ``tri_kernels`` serial kernels the Python
branch calls, and every admissible block of a pair therefore compresses
on the thread pool.

The stopping sample and the certificate lines are DRAWN IN PYTHON
(``aca.draw_lines``, from the block's own rng) and passed in, so the
factors depend only on the block, the tolerance and the seed: they are
the factors ``compress_block``'s Python branch produces, which is what
``verify_hbackend`` checks block by block.

Products and norms go through the same BLAS calls as that branch
(``np.dot``, and ``sqrt(dot(x, x))`` for a Frobenius norm, which is what
``np.linalg.norm`` does) instead of hand-written loops: the same
summation order is what makes the two paths agree to round-off rather
than to the block tolerance -- with one exception, ``np.linalg.qr``,
whose triangular factor numba returns as a read-after-free view and
which is therefore recomputed here (see ``_thin_qr``).

Own module, so ``cache=True`` here is invalidated by an edit to this file
rather than by one to the kernels it calls.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .. import defaults
from ..kernels import KERNEL_T, KERNEL_U
from ..kernels import tri_kernels as tk

# kernel_flag of the njit entry point (a numba kernel takes no strings)
KERNEL_FLAG = {KERNEL_T: 1, KERNEL_U: 0}

PINV_RCOND = defaults.ACA_PINV_RCOND     # a kernel reads no attributes


# ---------------------------------------------------------------------
# small helpers (all nogil: they run inside aca_block)
# ---------------------------------------------------------------------

@njit(nogil=True, cache=True)
def _gather2(a, idx):
    out = np.empty((idx.shape[0], a.shape[1]))
    for i in range(idx.shape[0]):
        out[i] = a[idx[i]]
    return out


@njit(nogil=True, cache=True)
def _gather3(a, idx):
    out = np.empty((idx.shape[0], a.shape[1], a.shape[2]))
    for i in range(idx.shape[0]):
        out[i] = a[idx[i]]
    return out


@njit(nogil=True, cache=True)
def _gather1(a, idx):
    out = np.empty(idx.shape[0])
    for i in range(idx.shape[0]):
        out[i] = a[idx[i]]
    return out


@njit(nogil=True, cache=True)
def _eval(kernel_flag, xf, tv, nrm, eps):
    """(B, 3 nf, 3 ns) exact basis stack of a subset, through the same
    serial kernels ``_BasisEval.stack_serial`` calls."""
    if kernel_flag == 1:
        return tk.t_basis_matrices_serial(xf, tv, nrm, eps)
    return tk.u_basis_matrices_serial(xf, tv, eps)


@njit(nogil=True, cache=True)
def _fro(a):
    """Frobenius norm as ``np.linalg.norm`` computes it: BLAS ddot on the
    flattened array (a hand-written loop sums in another order)."""
    v = np.ascontiguousarray(a).ravel()
    return np.sqrt(np.dot(v, v))


@njit(nogil=True, cache=True)
def _rel_err(approx, exact):
    denom = _fro(exact)
    if denom == 0.0:
        return 0.0
    return _fro(approx - exact) / denom


@njit(nogil=True, cache=True)
def _pinv3(P):
    """``aca._pinv3`` for a 3x3: the adjugate closed form of
    ``aca._adj_inv3``, expression for expression (the same bits as the
    Python branch, which is what keeps the pivot sequences together),
    falling back to the SVD pseudo-inverse at ``PINV_RCOND`` when the
    pivot block is too close to singular."""
    a = P[0, 0]; b = P[0, 1]; c = P[0, 2]
    d = P[1, 0]; e = P[1, 1]; f = P[1, 2]
    g = P[2, 0]; h = P[2, 1]; i = P[2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    det = a * A + b * B + c * C
    n2 = 0.0
    n2 += a * a
    n2 += b * b
    n2 += c * c
    n2 += d * d
    n2 += e * e
    n2 += f * f
    n2 += g * g
    n2 += h * h
    n2 += i * i
    scale = np.sqrt(n2)
    out = np.empty((3, 3))
    if abs(det) > PINV_RCOND * scale * scale * scale:
        out[0, 0] = A / det
        out[0, 1] = (c * h - b * i) / det
        out[0, 2] = (b * f - c * e) / det
        out[1, 0] = B / det
        out[1, 1] = (a * i - c * g) / det
        out[1, 2] = (c * d - a * f) / det
        out[2, 0] = C / det
        out[2, 1] = (b * g - a * h) / det
        out[2, 2] = (a * e - b * d) / det
        return out
    u, s, vt = np.linalg.svd(P)
    out[:, :] = 0.0
    if s.shape[0] == 0 or s[0] == 0.0:
        return out
    s_inv = np.zeros(s.shape[0])
    for k in range(s.shape[0]):
        if s[k] > PINV_RCOND * s[0]:
            s_inv[k] = 1.0 / s[k]
    return np.dot(np.ascontiguousarray(vt.T) * s_inv,
                  np.ascontiguousarray(u.T))


@njit(nogil=True, cache=True)
def _svd_keep(s, tol):
    """``aca._svd_keep``: smallest k with ||s[k:]|| <= tol * s[0]."""
    n = s.shape[0]
    if n == 0 or s[0] == 0.0:
        return 1
    tail = np.empty(n)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc += s[i] * s[i]
        tail[i] = np.sqrt(acc)
    thr = tol * s[0]
    keep = n
    for i in range(n):
        if tail[i] <= thr:      # searchsorted(-tail, -thr) on a sorted -tail
            keep = i
            break
    if keep < 1:
        keep = 1
    if keep > n:
        keep = n
    return keep


@njit(nogil=True, cache=True)
def _thin_qr(X):
    """Thin QR of a (m, k) factor: ``(Q, R)`` with ``Q`` orthonormal and
    ``Q R = X``, through LAPACK's Householder factorization.

    ``R`` is RECOMPUTED as ``Q^T X`` instead of taken from
    ``np.linalg.qr``, which returns it as a transposed view of a
    temporary its liveness guard does not cover (numba
    ``np/linalg.py``: the guard lists ``tau`` and ``q``, not the array
    behind ``r``). That view is read-after-free: measured garbage of
    order 1e198 in 2 of 12 calls on a (546, 375) factor, silently and
    nondeterministically, while ``Q`` from the same call is orthonormal
    to 2e-15 every time. One extra (m, k) product buys back the exact
    triangular factor -- and a Gram-matrix basis, which needs no R at
    all, is not an alternative here: it squares the conditioning of an
    ACA factor, whose columns span orders of magnitude, and cost four
    times as many blocks of the 10.9k inclusion model their certificate.
    """
    Q, _r_unsafe = np.linalg.qr(X)
    return Q, np.dot(np.ascontiguousarray(Q.T), X)


@njit(nogil=True, cache=True)
def _recompress(U, V, tol):
    """``aca.recompress``: thin QR of both factors, SVD of the core.

    Stays INSIDE this kernel. A pool of these over the operator's
    per-material recombinations is 10-15x faster than the serial numpy
    loop -- and leaves numba's own prange kernels segfaulting on the
    next call, on this machine's OpenMP-threaded OpenBLAS, at any BLAS
    thread count. LAPACK from several Python threads is what the whole
    block ACA already does through the same kernel; at recombination
    sizes it is not survivable."""
    Qu, Ru = _thin_qr(U)
    Qv, Rv = _thin_qr(V)
    u, s, vt = np.linalg.svd(np.dot(Ru, np.ascontiguousarray(Rv.T)), 0)
    keep = _svd_keep(s, tol)
    return (np.dot(Qu, np.ascontiguousarray(u[:, :keep] * s[:keep])),
            np.dot(Qv, np.ascontiguousarray(vt[:keep, :].T)))


# ---------------------------------------------------------------------
# the block kernel
# ---------------------------------------------------------------------

@njit(nogil=True, cache=True)
def aca_block(x_field, tri_verts, normals, eps, tol, srows, scols,
              cert_rows, cert_cols, max_rank, kernel_flag):
    """Compress every basis of one admissible block.

    ``x_field`` (nr, 3) are the block's field points and
    ``tri_verts`` / ``normals`` / ``eps`` its source triangles;
    ``srows`` / ``scols`` index the persistent stopping sample and
    ``cert_rows`` / ``cert_cols`` the certificate lines, both in
    block-local element numbering; ``max_rank`` is the element-pivot
    budget (``ACA_MAX_RANK_FRACTION``).

    Returns ``(U_flat, V_flat, ranks, err, capped)``: basis ``b`` holds
    ``(3 nr, ranks[b])`` row-major at ``sum(3 nr ranks[:b])`` of
    ``U_flat`` and ``(3 nc, ranks[b])`` in ``V_flat``; ``err`` is the
    largest certified relative Frobenius error over the bases and
    ``capped`` says a basis exhausted the budget (the block is not low
    rank at this tolerance and nothing was returned). The shared
    subspace the operator folds these into is built in ``aca``, on
    numpy's LAPACK: see ``aca.shared_subspace``.
    """
    n_rows = x_field.shape[0]
    n_cols = tri_verts.shape[0]
    nr3 = 3 * n_rows
    nc3 = 3 * n_cols

    # Persistent stopping sample: exact entries at srows x scols.
    stop_exact = _eval(kernel_flag, _gather2(x_field, srows),
                       _gather3(tri_verts, scols), _gather2(normals, scols),
                       _gather1(eps, scols))
    n_basis = stop_exact.shape[0]
    ns_r = srows.shape[0]
    ns_c = scols.shape[0]

    # Pivot row / column caches: one evaluation serves every basis, so
    # compressing B bases costs the kernel work of one. Slots grow by
    # doubling (a block's pivot count is its rank, not its size).
    row_slot = np.full(n_rows, -1, np.int64)
    col_slot = np.full(n_cols, -1, np.int64)
    row_data = np.empty((4, n_basis, 3, nc3))
    col_data = np.empty((4, n_basis, nr3, 3))
    n_row_slots = 0
    n_col_slots = 0

    ranks = np.zeros(n_basis, np.int64)
    U_flat = np.empty(0)
    V_flat = np.empty(0)
    n_u = 0
    n_v = 0
    stop_tol = 0.5 * tol

    for b in range(n_basis):
        R_samp = stop_exact[b].copy()
        samp_norm = _fro(R_samp)
        # Workspace for the accepted rank-3 updates. Grown by doubling
        # from a typical rank rather than allocated at ``max_rank``: that
        # is a third of the block side (22 MB per 1024-element T block,
        # 16 of them live under the pool) for a rank that is usually ten.
        cap = min(8, max_rank)
        U_work = np.zeros((nr3, 3 * cap))
        V_work = np.zeros((nc3, 3 * cap))
        used_rows = np.zeros(n_rows, np.uint8)
        used_cols = np.zeros(n_cols, np.uint8)
        steps = 0
        i_pivot = 0
        converged = False

        for _ in range(max_rank):
            used_rows[i_pivot] = 1
            s = row_slot[i_pivot]
            if s < 0:
                if n_row_slots == row_data.shape[0]:
                    grown = np.empty((2 * row_data.shape[0], n_basis, 3, nc3))
                    grown[:n_row_slots] = row_data
                    row_data = grown
                s = n_row_slots
                n_row_slots += 1
                row_slot[i_pivot] = s
                row_data[s] = _eval(kernel_flag,
                                    x_field[i_pivot:i_pivot + 1], tri_verts,
                                    normals, eps)
            R_row = row_data[s, b].copy()                   # (3, 3 nc)
            r0 = 3 * i_pivot
            for p in range(steps):
                Up = np.ascontiguousarray(U_work[r0:r0 + 3, 3 * p:3 * p + 3])
                VpT = np.ascontiguousarray(V_work[:, 3 * p:3 * p + 3].T)
                R_row -= np.dot(Up, VpT)

            j_pivot = -1
            best = 0.0
            for j in range(n_cols):
                if used_cols[j] == 1:
                    continue
                # aca._block_norms' order: over the block's leading index
                # first, then its trailing one (a tie here is decided by
                # the last ulp).
                t0 = 0.0
                t1 = 0.0
                t2 = 0.0
                for a in range(3):
                    v = R_row[a, 3 * j]
                    t0 += v * v
                    v = R_row[a, 3 * j + 1]
                    t1 += v * v
                    v = R_row[a, 3 * j + 2]
                    t2 += v * v
                score = np.sqrt((t0 + t1) + t2)
                if j_pivot < 0 or score > best:
                    best = score
                    j_pivot = j
            if j_pivot < 0 or best <= 1e-300:
                converged = True
                break
            used_cols[j_pivot] = 1

            s = col_slot[j_pivot]
            if s < 0:
                if n_col_slots == col_data.shape[0]:
                    grown = np.empty((2 * col_data.shape[0], n_basis, nr3, 3))
                    grown[:n_col_slots] = col_data
                    col_data = grown
                s = n_col_slots
                n_col_slots += 1
                col_slot[j_pivot] = s
                col_data[s] = _eval(kernel_flag, x_field,
                                    tri_verts[j_pivot:j_pivot + 1],
                                    normals[j_pivot:j_pivot + 1],
                                    eps[j_pivot:j_pivot + 1])
            C_col = col_data[s, b].copy()                   # (3 nr, 3)
            c0 = 3 * j_pivot
            for p in range(steps):
                Up = np.ascontiguousarray(U_work[:, 3 * p:3 * p + 3])
                Vc = np.ascontiguousarray(
                    V_work[c0:c0 + 3, 3 * p:3 * p + 3].T)
                C_col -= np.dot(Up, Vc)

            P = np.ascontiguousarray(R_row[:, c0:c0 + 3])   # (3, 3)
            V_new = np.ascontiguousarray(np.dot(_pinv3(P), R_row).T)
            if 3 * (steps + 1) > U_work.shape[1]:
                grown_u = np.zeros((nr3, 2 * U_work.shape[1]))
                grown_u[:, :3 * steps] = U_work[:, :3 * steps]
                U_work = grown_u
                grown_v = np.zeros((nc3, 2 * V_work.shape[1]))
                grown_v[:, :3 * steps] = V_work[:, :3 * steps]
                V_work = grown_v
            U_work[:, 3 * steps:3 * steps + 3] = C_col
            V_work[:, 3 * steps:3 * steps + 3] = V_new
            steps += 1

            # True residual on the persistent sample (the increment
            # heuristic alone plateaus on this kernel).
            US = np.empty((3 * ns_r, 3))
            for i in range(ns_r):
                for a in range(3):
                    for q in range(3):
                        US[3 * i + a, q] = C_col[3 * srows[i] + a, q]
            VS = np.empty((3, 3 * ns_c))
            for j in range(ns_c):
                for a in range(3):
                    for q in range(3):
                        VS[q, 3 * j + a] = V_new[3 * scols[j] + a, q]
            R_samp -= np.dot(US, VS)
            if samp_norm == 0.0 or _fro(R_samp) < stop_tol * samp_norm:
                converged = True
                break

            i_pivot = -1
            best = 0.0
            for i in range(n_rows):
                if used_rows[i] == 1:
                    continue
                t0 = 0.0
                t1 = 0.0
                t2 = 0.0
                for a in range(3):
                    v = C_col[3 * i + a, 0]
                    t0 += v * v
                    v = C_col[3 * i + a, 1]
                    t1 += v * v
                    v = C_col[3 * i + a, 2]
                    t2 += v * v
                score = np.sqrt((t0 + t1) + t2)
                if i_pivot < 0 or score > best:
                    best = score
                    i_pivot = i
            if i_pivot < 0 or best <= 0.0:
                converged = True
                break

        if not converged:
            return (np.empty(0), np.empty(0), ranks, 0.0, True)

        if steps == 0:
            U_b = np.zeros((nr3, 1))
            V_b = np.zeros((nc3, 1))
        else:
            U_b, V_b = _recompress(
                np.ascontiguousarray(U_work[:, :3 * steps]),
                np.ascontiguousarray(V_work[:, :3 * steps]), stop_tol)
        k = U_b.shape[1]
        ranks[b] = k
        if n_u + U_b.size > U_flat.shape[0]:
            grown = np.empty(max(2 * U_flat.shape[0], n_u + U_b.size))
            grown[:n_u] = U_flat[:n_u]
            U_flat = grown
        if n_v + V_b.size > V_flat.shape[0]:
            grown = np.empty(max(2 * V_flat.shape[0], n_v + V_b.size))
            grown[:n_v] = V_flat[:n_v]
            V_flat = grown
        U_flat[n_u:n_u + U_b.size] = U_b.ravel()
        V_flat[n_v:n_v + V_b.size] = V_b.ravel()
        n_u += U_b.size
        n_v += V_b.size

    # Certificate: exact full rows and columns against the factors.
    rows_exact = _eval(kernel_flag, _gather2(x_field, cert_rows), tri_verts,
                       normals, eps)
    cols_exact = _eval(kernel_flag, x_field, _gather3(tri_verts, cert_cols),
                       _gather2(normals, cert_cols), _gather1(eps, cert_cols))
    m_r = cert_rows.shape[0]
    m_c = cert_cols.shape[0]
    err = 0.0
    off_u = 0
    off_v = 0
    for b in range(n_basis):
        k = ranks[b]
        U_b = U_flat[off_u:off_u + nr3 * k].reshape(nr3, k)
        V_b = V_flat[off_v:off_v + nc3 * k].reshape(nc3, k)
        off_u += nr3 * k
        off_v += nc3 * k
        Ur = np.empty((3 * m_r, k))
        for i in range(m_r):
            for a in range(3):
                Ur[3 * i + a] = U_b[3 * cert_rows[i] + a]
        Vc = np.empty((3 * m_c, k))
        for j in range(m_c):
            for a in range(3):
                Vc[3 * j + a] = V_b[3 * cert_cols[j] + a]
        e_rows = _rel_err(np.dot(Ur, np.ascontiguousarray(V_b.T)),
                          rows_exact[b])
        e_cols = _rel_err(np.dot(U_b, np.ascontiguousarray(Vc.T)),
                          cols_exact[b])
        if e_rows > err:
            err = e_rows
        if e_cols > err:
            err = e_cols
    return U_flat[:n_u].copy(), V_flat[:n_v].copy(), ranks, err, False


def block_factors(x_field, tri_verts, normals, eps, kernel_flag, tol,
                  srows, scols, cert_rows, cert_cols, max_rank):
    """``aca_block`` with its ragged factors split per basis:
    ``([U_b], [V_b], err, capped)`` -- the contract ``compress_block``
    calls an alternative ACA implementation through."""
    U_flat, V_flat, ranks, err, capped = aca_block(
        x_field, tri_verts, normals, eps, float(tol),
        np.ascontiguousarray(srows, dtype=np.int64),
        np.ascontiguousarray(scols, dtype=np.int64),
        np.ascontiguousarray(cert_rows, dtype=np.int64),
        np.ascontiguousarray(cert_cols, dtype=np.int64),
        int(max_rank), int(kernel_flag))
    if capped:
        return [], [], 0.0, True
    nr3 = 3 * x_field.shape[0]
    nc3 = 3 * tri_verts.shape[0]
    Us, Vs = [], []
    off_u = off_v = 0
    for k in ranks:
        k = int(k)
        Us.append(U_flat[off_u:off_u + nr3 * k].reshape(nr3, k))
        Vs.append(V_flat[off_v:off_v + nc3 * k].reshape(nc3, k))
        off_u += nr3 * k
        off_v += nc3 * k
    return Us, Vs, float(err), False
