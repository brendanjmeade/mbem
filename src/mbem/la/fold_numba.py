"""The shared-subspace fold, entire, in one nogil numba kernel.

``aca.shared_subspace`` stays as the reference this is gated against
(``verify_hbackend.check_shared_subspace``). What this module exists for is
that the fold could not run on the compression pool: 68.9 % of the ACA phase
at 260,598 unknowns is the fold, and the ACA phase is 84-87 % of a build that
scales N^1.39, so the fold is the largest single item in the H path.

AND THE PART THAT DOES NOT POOL IS NOT THE QR. ``np.linalg.qr`` already
releases the GIL and is 86.8 % of a large fold; replacing it -- the fix WP5
named -- attacks the half that already scales 11x. What is GIL-held is the
WRAPPER: two Gram matrices, two ``eigh`` truncations and the core products,
~5.7 ms per fold at ANY block size, because they work on (K, K) matrices with
K = sum of the per-basis ranks (~102) independent of the block. That is 13 %
of a 1024-element block but 55 % of a 128-element one, and the production
partition is dominated by the small end -- so the measured collapse is convoy,
not plain serialization, and the only fix that reaches it is putting the WHOLE
fold in one kernel.

NO LAPACK FACTORIZATION ANYWHERE, which is forced rather than stylistic. Two
recorded failures bound this code: numba's ``np.linalg.qr`` returned memory
read after free (`c480c75`), and scipy's ``qr(mode="economic")`` is 3-25x
faster than numpy's but SILENTLY CORRUPTS 2-6 stored blocks per model,
nondeterministically. So the thin QR is block Gram-Schmidt and the symmetric
eigenproblem is cyclic Jacobi, both hand-written; only ``np.dot`` (BLAS GEMM,
which ``aca_numba`` already calls from inside nogil kernels) is borrowed.

TWO MEASUREMENTS SHAPED THE ALGORITHM, and the obvious version of it is 1.05x
-- i.e. nothing:

* A column-at-a-time Gram-Schmidt is 3.7x numpy's QR, because it is BLAS-2.
  The QR is therefore BLOCKED: a whole panel's two orthogonalization passes
  are four GEMMs instead of 2 nb dot/axpy pairs.
* A full (K, K) cyclic Jacobi is 23x ``eigh``, not the ~3x the record
  projected, and it is O(n^3) -- so the kernel stands or falls on the
  eigenproblem's SIZE. It is DEFLATED first by a pivoted Cholesky, which
  cuts n by ~1.3x per side, and the error bookkeeping is exact rather than
  heuristic (see ``_principal_defl``).

Everything long is stored TRANSPOSED (row-major over the long dimension). A
C-ordered Q indexed down its columns strides by n, and that alone measured
1.8x on an earlier prototype.

Own module, so ``cache=True`` here is invalidated by an edit to this file
rather than by one to the ACA kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .. import defaults

# A kernel reads no attributes, so the constants are module-level.
PANEL = defaults.ACA_FOLD_PANEL
CHOL_SLACK = defaults.ACA_FOLD_CHOL_SLACK
RANK_TOL = 1e-13          # residual / initial column norm, rank detection
JACOBI_SWEEPS = 40
JACOBI_TOL = 1e-32        # (a_pq^2 / |a_pp a_qq|) rotation floor


@njit(nogil=True, cache=True)
def _gather_t(flat, ranks, m):
    """(K, m) row-major transpose of the hstacked per-basis factors.

    Basis b occupies (m, ranks[b]) row-major at ``sum(m * ranks[:b])`` in
    the ragged buffer ``aca_numba.aca_block`` already produces; row j of the
    result is global column j, so every later dot and axpy is contiguous.
    """
    K = 0
    for b in range(ranks.shape[0]):
        K += ranks[b]
    Xt = np.empty((K, m))
    off = 0
    j0 = 0
    for b in range(ranks.shape[0]):
        k = ranks[b]
        for jl in range(k):
            row = Xt[j0 + jl]
            base = off + jl
            for i in range(m):
                row[i] = flat[base + i * k]
        off += m * k
        j0 += k
    return Xt


@njit(nogil=True, cache=True)
def _cgs2(Xt):
    """(Qt, R, r) thin QR of ``Xt.T`` by column-at-a-time Gram-Schmidt with
    one reorthogonalization. Used only on the small deflation factor; the
    long side goes through :func:`_bcgs2`."""
    K, m = Xt.shape
    Qt = np.empty((K, m))
    R = np.zeros((K, K))
    r = 0
    for j in range(K):
        x = Xt[j].copy()
        nrm0 = np.sqrt(np.dot(x, x))
        if nrm0 == 0.0:
            continue
        for _pass in range(2):
            if r > 0:
                Qr = Qt[:r]
                cc = np.dot(Qr, x)
                for p in range(r):
                    R[p, j] += cc[p]
                x -= np.dot(np.ascontiguousarray(Qr.T), cc)
        nrm = np.sqrt(np.dot(x, x))
        if nrm > RANK_TOL * nrm0:
            R[r, j] = nrm
            Qt[r] = x / nrm
            r += 1
    if r == 0:                    # an all-zero side: one arbitrary unit row
        Qt[0, :] = 0.0
        Qt[0, 0] = 1.0
        r = 1
    return Qt, R, r


@njit(nogil=True, cache=True)
def _bcgs2(Xt, panel):
    """(Qt, R, r) thin QR of ``Xt.T``: ``Qt[:r]`` orthonormal rows with
    ``Qt[:r].T @ R[:r] == Xt.T``.

    BLOCK Gram-Schmidt: a panel's two orthogonalization passes against the
    accepted basis are four GEMMs, which is what puts the O(m K^2) half of
    the fold at BLAS-3 speed. Rank detection is by residual norm, so a
    column adding no direction leaves an all-zero R row where LAPACK would
    leave a numerically-zero diagonal; the truncation downstream discards
    exactly those directions anyway.
    """
    K, m = Xt.shape
    Qt = np.empty((K, m))
    R = np.zeros((K, K))
    r = 0
    j0 = 0
    while j0 < K:
        j1 = min(j0 + panel, K)
        P = Xt[j0:j1].copy()                     # (nb, m)
        nb = j1 - j0
        if r > 0:
            for _pass in range(2):
                Qr = Qt[:r]
                C = np.dot(P, np.ascontiguousarray(Qr.T))       # (nb, r)
                for i in range(nb):
                    for p in range(r):
                        R[p, j0 + i] += C[i, p]
                P -= np.dot(C, Qr)
        r0 = r
        for i in range(nb):                      # inside the panel
            x = P[i].copy()
            nrm0 = np.sqrt(np.dot(x, x))
            if nrm0 == 0.0:
                continue
            for _pass in range(2):
                if r > r0:
                    Qr = Qt[r0:r]
                    cc = np.dot(Qr, x)
                    for p in range(r - r0):
                        R[r0 + p, j0 + i] += cc[p]
                    x -= np.dot(np.ascontiguousarray(Qr.T), cc)
            nrm = np.sqrt(np.dot(x, x))
            if nrm > RANK_TOL * nrm0:
                R[r, j0 + i] = nrm
                Qt[r] = x / nrm
                r += 1
        j0 = j1
    if r == 0:
        Qt[0, :] = 0.0
        Qt[0, 0] = 1.0
        r = 1
    return Qt, R, r


@njit(nogil=True, cache=True)
def _jacobi(A):
    """(w ascending, Zt with eigenvectors as ROWS) of a symmetric matrix, by
    cyclic Jacobi rotations. Hand-written because numba's LAPACK is not
    trusted here; 13-31x ``eigh`` at these sizes, which is why the caller
    deflates first.
    """
    n = A.shape[0]
    G = A.copy()
    Zt = np.eye(n)                 # ROWS are eigenvectors (contiguous)
    fro2 = 0.0
    for i in range(n):
        for j in range(n):
            fro2 += G[i, j] * G[i, j]
    if fro2 == 0.0:
        return np.zeros(n), Zt
    for _sweep in range(JACOBI_SWEEPS):
        rotations = 0
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = G[p, q]
                # Skipped only when the entry is negligible against ITS OWN
                # diagonal pair -- the classical criterion, and what gives
                # Jacobi its relative accuracy on the small eigenvalues. The
                # truncation cuts at ~1e-11 of the largest, so an off-norm
                # stopping test decides the RANK by noise: measured, it kept
                # 2-5 columns too many per block.
                if apq * apq <= JACOBI_TOL * abs(G[p, p] * G[q, q]):
                    continue
                rotations += 1
                app = G[p, p]
                aqq = G[q, q]
                theta = (aqq - app) / (2.0 * apq)
                if theta >= 0.0:
                    t = 1.0 / (theta + np.sqrt(theta * theta + 1.0))
                else:
                    t = -1.0 / (-theta + np.sqrt(theta * theta + 1.0))
                c = 1.0 / np.sqrt(t * t + 1.0)
                sn = t * c
                # Rows p and q (contiguous), then the 2x2 in closed form,
                # then the columns mirrored from the rows: the two-sided
                # update written down COLUMNS strides by n and measured
                # 1.6x slower for the same arithmetic.
                Gp = G[p]
                Gq = G[q]
                for k in range(n):
                    gp = Gp[k]
                    gq = Gq[k]
                    Gp[k] = c * gp - sn * gq
                    Gq[k] = sn * gp + c * gq
                Gp[p] = app - t * apq
                Gq[q] = aqq + t * apq
                Gp[q] = 0.0
                Gq[p] = 0.0
                for k in range(n):
                    G[k, p] = Gp[k]
                    G[k, q] = Gq[k]
                Zp = Zt[p]
                Zq = Zt[q]
                for k in range(n):
                    zp = Zp[k]
                    zq = Zq[k]
                    Zp[k] = c * zp - sn * zq
                    Zq[k] = sn * zp + c * zq
        if rotations == 0:
            break
    w = np.empty(n)
    for i in range(n):
        w[i] = G[i, i]
    order = np.argsort(w)
    ws = np.empty(n)
    Zs = np.empty((n, n))
    for i in range(n):
        ws[i] = w[order[i]]
        Zs[i] = Zt[order[i]]
    return ws, Zs


@njit(nogil=True, cache=True)
def _principal_t(G, delta):
    """``aca._principal`` on the full Gram, eigenvectors as ROWS."""
    w, Zt = _jacobi(G)
    n = w.shape[0]
    acc = 0.0
    drop = 0
    lim = delta * delta
    for i in range(n):
        acc += max(w[i], 0.0)
        if acc <= lim:
            drop = i + 1
        else:
            break
    keep = max(n - drop, 1)
    Wt = np.empty((keep, n))
    for p in range(keep):
        Wt[p] = Zt[n - 1 - p]
    return Wt


@njit(nogil=True, cache=True)
def _pivoted_chol(G, tau):
    """(Lt, k, resid) pivoted Cholesky of a PSD Gram, factor TRANSPOSED (row
    p is column p of L, so every update is contiguous), stopping when the
    Schur-complement trace falls to ``tau``; ``resid`` is that trace."""
    n = G.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = G[i, i]
    Lt = np.empty((n, n))
    tr = 0.0
    for i in range(n):
        tr += max(d[i], 0.0)
    k = 0
    while k < n and tr > tau:
        j = 0
        best = d[0]
        for i in range(1, n):
            if d[i] > best:
                best = d[i]
                j = i
        if best <= 0.0:
            break
        s = G[j].copy()                      # row j == column j (symmetric)
        for p in range(k):
            c = Lt[p, j]
            if c != 0.0:
                s -= c * Lt[p]
        s *= 1.0 / np.sqrt(best)
        Lt[k] = s
        tr = 0.0
        for i in range(n):
            d[i] -= s[i] * s[i]
            if d[i] > 0.0:
                tr += d[i]
        d[j] = 0.0
        k += 1
    return Lt, k, tr


@njit(nogil=True, cache=True)
def _principal_defl(G, delta, slack):
    """``aca._principal`` through a deflation, eigenvectors as ROWS.

    The Jacobi is O(n^3) and 23x ``eigh``, so the kept rank being ~0.4 K is
    worth exploiting: a pivoted Cholesky finds a subspace containing the
    leading eigenvectors in O(K k) work and the eigenproblem shrinks to
    (r, r).

    THE BOOKKEEPING IS EXACT, not a heuristic. With ``G = A A^T``, k steps of
    pivoted Cholesky leave a Schur complement ``S = A (I - Pi) A^T`` with
    ``Pi`` an orthogonal projector, and the Cholesky columns span a ``W``
    containing ``A Pi``, so ``||(I - P_W) A||_F^2 <= trace(S)``. The
    truncation then carries ``trace(S)`` as ALREADY-SPENT budget -- drop
    while ``acc + trace(S) <= delta^2`` -- and ``trace(S)`` is driven to
    ``slack`` times the budget, so the decision is the reference's. Measured:
    the keep count equals ``aca._principal``'s on all 90 real Gram matrices
    of 45 blocks x 2 sides.
    """
    n = G.shape[0]
    lim = delta * delta
    Lt, k, resid = _pivoted_chol(G, slack * lim)
    if k == 0:
        Wt = np.zeros((1, n))
        Wt[0, 0] = 1.0
        return Wt
    if k >= n:                              # nothing deflated; do it whole
        return _principal_t(G, delta)
    Yt, _R, r = _cgs2(np.ascontiguousarray(Lt[:k]))
    Y = np.ascontiguousarray(Yt[:r].T)
    Gs = np.dot(np.ascontiguousarray(Yt[:r]), np.dot(G, Y))
    for i in range(r):                      # symmetrize exactly
        for j in range(i + 1, r):
            v = 0.5 * (Gs[i, j] + Gs[j, i])
            Gs[i, j] = v
            Gs[j, i] = v
    w, Zt = _jacobi(np.ascontiguousarray(Gs))
    acc = resid
    drop = 0
    for i in range(r):
        acc += max(w[i], 0.0)
        if acc <= lim:
            drop = i + 1
        else:
            break
    keep = max(r - drop, 1)
    Wt = np.empty((keep, n))
    for p in range(keep):
        Wt[p] = np.dot(Zt[r - 1 - p], Yt[:r])
    return Wt


@njit(nogil=True, cache=True)
def _unit_gram_t(cores, axis):
    """``aca._unit_gram``: the Gram of the cores normalized to unit
    Frobenius norm, summed over the bases."""
    B, ku, kv = cores.shape
    n = ku if axis == 0 else kv
    out = np.zeros((n, n))
    for b in range(B):
        c = cores[b]
        v = c.ravel()
        nrm = np.sqrt(np.dot(v, v))
        if nrm == 0.0:
            continue
        m = c / nrm
        if axis == 0:
            out += np.dot(m, np.ascontiguousarray(m.T))
        else:
            out += np.dot(np.ascontiguousarray(m.T), m)
    return out


@njit(nogil=True, cache=True)
def fold_block(U_flat, V_flat, ranks, nr3, nc3, delta, panel, slack):
    """``aca.shared_subspace`` entire: ``(Qu, Qv, cores)`` with
    ``U_b V_b^T = Qu cores[b] Qv^T``.

    The cores come from column blocks of the JOINT R on BOTH sides, which is
    what makes them consistent with one shared basis pair.
    """
    B = ranks.shape[0]
    Qut, Ru, ru = _bcgs2(_gather_t(U_flat, ranks, nr3), panel)
    Qvt, Rv, rv = _bcgs2(_gather_t(V_flat, ranks, nc3), panel)

    cores = np.empty((B, ru, rv))
    off = 0
    for b in range(B):
        k = ranks[b]
        a = np.ascontiguousarray(Ru[:ru, off:off + k])
        d = np.ascontiguousarray(Rv[:rv, off:off + k])
        cores[b] = np.dot(a, np.ascontiguousarray(d.T))
        off += k

    Wut = _principal_defl(_unit_gram_t(cores, 0), delta, slack)
    ku = Wut.shape[0]
    c1 = np.empty((B, ku, rv))
    for b in range(B):
        c1[b] = np.dot(Wut, cores[b])
    Wvt = _principal_defl(_unit_gram_t(c1, 1), delta, slack)
    Wv = np.ascontiguousarray(Wvt.T)
    out = np.empty((B, ku, Wvt.shape[0]))
    for b in range(B):
        out[b] = np.dot(c1[b], Wv)

    Qu = np.dot(np.ascontiguousarray(Qut[:ru].T),
                np.ascontiguousarray(Wut.T))
    Qv = np.dot(np.ascontiguousarray(Qvt[:rv].T), Wv)
    return Qu, Qv, out


def shared_subspace_flat(U_flat, V_flat, ranks, nr3, nc3, delta):
    """Typed entry point: a kernel reads no Python attributes."""
    return fold_block(np.ascontiguousarray(U_flat, dtype=np.float64),
                      np.ascontiguousarray(V_flat, dtype=np.float64),
                      np.ascontiguousarray(ranks, dtype=np.int64),
                      int(nr3), int(nc3), float(delta), PANEL, CHOL_SLACK)


def flatten(Us, Vs):
    """The ragged flat form, for a caller holding per-basis lists (the
    Python reference ACA); the numba ACA already produces it."""
    return (np.concatenate([np.ascontiguousarray(u).ravel() for u in Us]),
            np.concatenate([np.ascontiguousarray(v).ravel() for v in Vs]),
            np.array([u.shape[1] for u in Us], np.int64))
