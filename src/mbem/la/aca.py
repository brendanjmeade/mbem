"""Block ACA compression of material-basis matrix stacks.

Each admissible block is compressed basis by basis,

    A_b ~ U_b @ V_b.T          b = 1..B   (B = 3 for U-kernel, 6 for T)

— a stacked factorization pays up to a Bx rank penalty, because the
basis matrices do not share row/column spaces exactly — and the B factor
pairs are then folded ONCE into a SHARED SUBSPACE (``shared_subspace``):

    A_b = Q_u @ cores[b] @ Q_v.T       Q_u, Q_v orthonormal

so a material's block is ``Q_u (sum_b c_b cores[b]) Q_v.T``, a k x k sum
and a k x k SVD instead of a QR+SVD of the (3n, sum_b k_b) factors. The
per-material recombination stops being the operator's dominant build
cost (measured 134x on 1024-element T blocks) and the far field shrinks
with it: the joint rank is 0.41x the summed per-basis rank on those
blocks and 0.52x over the 2,037 stored blocks of a 31k-unknown model,
since the bases of one kernel span nearly the same row/column spaces. The material's own epsilon-rank is unchanged — the
k x k SVD still truncates the combination — so the matvec is untouched.

Pivoting is 3x3 ELEMENT-block (scalar ACA on interleaved xyz DOFs is
what plateaued in the legacy code, hmatrix.py:493). Kernel evaluations
go through a shared row/column cache: one evaluation yields the rows of
ALL bases, so compressing 6 bases costs the same kernel work as one; the
cache is released as soon as the block is done.

``compress_block`` owns the POLICY (which lines are drawn, what an
outcome becomes); the ACA itself is pluggable. ``_aca_python`` here is
the reference implementation, and the operator path passes the nogil
numba kernel of :mod:`.aca_numba` instead, which is why a block's
Python-side cost is now its policy and nothing else.

Every block is CERTIFIED before it is accepted, on ``ACA_CERTIFY_LINES``
full random rows and as many full random columns evaluated exactly: per
basis here, at build, on the ACA's factors; and per MATERIAL when the
operator layer combines the cores (``certify_combined``), because a
per-basis tolerance does not bound a combination whose coefficients have
mixed signs (the T coefficients do). The per-material check is the
binding one -- it covers every material actually applied, declared at
build or not, and it certifies the block AS APPLIED, the recombined
``U @ V.T`` the matvec runs, with the fold inside it. What the fold adds
between the two is bounded by construction, not by sampling: its
truncation discards a tail of norm ``ACA_JOINT_TOL_FACTOR`` x tol from
each basis's normalized core. A block whose ACA hits the
rank cap is not low rank at this tolerance and is applied exactly, like a
near-field leaf. A FAILED CERTIFICATE is a stopping-rule failure, not a
verdict on the block: the ACA stops on a small random sample, and the
certificate lines are what catch a sample that fired early, so the block
is RE-RUN once at ``ACA_RETRY_TOL_FACTOR`` times the tolerance (the
lines never enter the ACA, so the retry must beat the same certificate)
and only a second failure is applied exactly, so correctness never
depends on the ACA's own stopping rule. What "exactly" costs is the
caller's choice: the operator path (``exact_payload=False``) keeps only
the block's indices and re-evaluates it per material with the near-field
leaves, while the HODLR rung takes the stack, SVD-truncated above
``ACA_SVD_FALLBACK_MIN_SIDE``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import defaults
from . import fold_numba


# Factorization counters, in the two sizes that matter: "subspace" is a
# BLOCK-SIZED pass (the ACA's recompressions and the joint QR/SVD of the
# (d n, sum_b k_b) factors, O(d n k^2)), "core" a k x k SVD of one
# material's combined core. A material view must add ONLY core ones --
# that is what the shared subspace buys, and what verify_hbackend
# asserts on the second material of a sweep. Incremented without a lock:
# the build count is approximate under the block pool, while the
# per-material delta a gate reads is taken over a serial loop.
FACTORIZATIONS = {"subspace": 0, "core": 0}


@dataclass
class PendingLR:
    """A block's per-basis factors, not yet folded into a subspace --
    what ``compress_block(fold=False)`` returns so the caller can choose
    the thread the fold runs on.

    It now runs ON the pool. The fold used to be main-thread because a pool
    of the NUMPY fold measured 0.6x of one thread -- recorded here as
    OpenBLAS's buffer lock, which was wrong: it is the GIL, held by the
    wrapper around the (K, K) Gram matrices and truncations rather than by
    the QR, which already releases it. ``la/fold_numba`` puts the whole fold
    in one nogil kernel, which is what makes pooling it worth anything.
    """
    U: list
    V: list
    cert_rows: np.ndarray
    cert_cols: np.ndarray
    flat: tuple | None = None

    def fold(self, tol: float):
        """This block's ``SharedLR``."""
        FACTORIZATIONS["subspace"] += 1
        ranks = tuple(u.shape[1] for u in self.U)
        if len(self.U) == 1:   # nothing to share: the basis IS the subspace
            Qu, Qv, cores = self.U[0], self.V[0], None
        else:
            delta = defaults.ACA_JOINT_TOL_FACTOR * tol
            # The numba ACA already holds the ragged flat buffers; the
            # Python reference ACA does not, and concatenating is ~0.4 % of
            # a fold.
            fu, fv, rk = self.flat if self.flat is not None \
                else fold_numba.flatten(self.U, self.V)
            Qu, Qv, cores = fold_numba.shared_subspace_flat(
                fu, fv, rk, self.U[0].shape[0], self.V[0].shape[0], delta)
        return SharedLR(Qu=Qu, Qv=Qv, cores=cores, basis_ranks=ranks,
                        cert_rows=self.cert_rows, cert_cols=self.cert_cols)


@dataclass
class SharedLR:
    """One block's bases in a SHARED subspace: ``A_b = Qu @ cores[b] @
    Qv.T`` with ``Qu`` (d nr, ku) and ``Qv`` (d nc, kv) orthonormal, plus
    the block-local element rows and columns (``cert_rows``,
    ``cert_cols``) every material combination is certified on.

    ``cores is None`` is the SINGLE-basis form (the HODLR rung's coupled
    blocks): one basis has nothing to share, so ``Qu``/``Qv`` are the
    ACA's own factors and no joint QR is paid for nothing.
    ``basis_ranks`` are the ACA's per-basis ranks, kept for the rank
    statistics and the implementation-parity gate.
    """
    Qu: np.ndarray
    Qv: np.ndarray
    cores: np.ndarray | None
    basis_ranks: tuple
    cert_rows: np.ndarray
    cert_cols: np.ndarray

    @property
    def rank(self) -> int:
        """Columns of the shared column basis (``Qv`` may differ by a
        few); the stored rank, against ``sum(basis_ranks)`` before."""
        return int(self.Qu.shape[1])

    def combine(self, c: np.ndarray, tol: float):
        """Factors ``(U, V)`` of the material block ``sum_b c_b A_b``,
        truncated at ``tol`` to the COMBINATION's own epsilon-rank: the
        cores are summed (B k^2) and that k x k sum is what the SVD
        truncates, so no factor of the block's own size is ever
        re-factorized for a material."""
        if self.cores is None:
            return float(c[0]) * self.Qu, self.Qv
        C = np.tensordot(np.asarray(c, dtype=float), self.cores, axes=1)
        FACTORIZATIONS["core"] += 1
        u, s, vt = np.linalg.svd(C, full_matrices=False)
        keep = _svd_keep(s, tol)
        return self.Qu @ (u[:, :keep] * s[:keep]), self.Qv @ vt[:keep, :].T

    def factors(self, b: int):
        """Low-rank factors ``(A, B)`` of basis ``b`` alone, ``A @ B.T``."""
        if self.cores is None:
            return self.Qu, self.Qv
        return self.Qu @ self.cores[b], self.Qv

    def basis_matrix(self, b: int) -> np.ndarray:
        """Basis ``b`` as stored, dense -- for gates and diagnostics."""
        A, B = self.factors(b)
        return A @ B.T

    def nbytes(self) -> int:
        return int(self.Qu.nbytes + self.Qv.nbytes
                   + (0 if self.cores is None else self.cores.nbytes))


@dataclass
class BlockResult:
    """Outcome of ``compress_block`` for one admissible block.

    ``payload`` is a ``SharedLR``, the exact ``(B, d nr, d nc)`` stack, or
    ``None`` when the caller asked for no exact payload
    (``exact_payload=False``): the block is not low rank at this
    tolerance and the caller re-evaluates it from the kernels.
    ``capped``: the ACA hit ``ACA_MAX_RANK_FRACTION``. ``retried``: the
    first certificate failed and the ACA was re-run at a tighter
    tolerance. ``fallback``: it failed again, and the block is applied
    exactly (or SVD-truncated from its exact stack). ``max_err`` is the
    largest certified relative Frobenius error over the bases (0 when
    exact).
    """
    payload: object
    capped: bool = False
    fallback: bool = False
    retried: bool = False
    max_err: float = 0.0


class BlockEvalCache:
    """Caches exact basis rows/cols/samples for one admissible block.

    ``stack_fn(rows, cols)`` -> (B, d*len(rows), d*len(cols)) where d is
    the DOFs per element (3 for kernel bases; 6 for coupled (u,t)
    transmission blocks used by the HODLR rung). The row/column dicts
    grow with the pivot count; ``clear`` releases them once the block's
    factors are final (or before its exact stack is materialized).

    ``aca_fn(tol, srows, scols, cert_rows, cert_cols, max_rank)`` ->
    ``([U_b], [V_b], certified error, capped)`` replaces the Python ACA
    and certificate of ``compress_block`` for this block: the operator
    path passes ``aca_numba.block_factors`` (one nogil kernel per block,
    the whole point of the thread pool), the HODLR rung and the gate
    leave it None and get the Python loop. The stack_fn stays: it is what
    an exact fallback is materialized through.
    """

    def __init__(self, stack_fn, rows: np.ndarray, cols: np.ndarray,
                 d: int = 3, aca_fn=None):
        self.stack_fn = stack_fn
        self.rows = rows
        self.cols = cols
        self.d = d
        self.aca_fn = aca_fn
        self._row: dict[int, np.ndarray] = {}
        self._col: dict[int, np.ndarray] = {}
        self.n_row_evals = 0
        self.n_col_evals = 0

    def row(self, b: int, i: int) -> np.ndarray:
        """(d, d*nc) exact rows of element i for basis b."""
        blk = self._row.get(i)
        if blk is None:
            blk = self.stack_fn(self.rows[i:i + 1], self.cols)
            self._row[i] = blk
            self.n_row_evals += 1
        return blk[b]

    def col(self, b: int, j: int) -> np.ndarray:
        """(d*nr, d) exact cols of element j for basis b."""
        blk = self._col.get(j)
        if blk is None:
            blk = self.stack_fn(self.rows, self.cols[j:j + 1])
            self._col[j] = blk
            self.n_col_evals += 1
        return blk[b]

    def dense(self) -> np.ndarray:
        return self.stack_fn(self.rows, self.cols)

    def clear(self) -> None:
        """Drop the cached pivot rows and columns."""
        self._row.clear()
        self._col.clear()


def _adj_inv3(P: np.ndarray, rcond: float) -> np.ndarray | None:
    """Inverse of a 3x3 by adjugate / determinant, or None when the block
    is too close to singular for it (|det| <= rcond ||P||_F^3).

    Scalar arithmetic in a fixed order, so numpy and numba compute the
    SAME bits -- which is what keeps the two ACA implementations on the
    same pivot sequence (see ``_block_norms``). LAPACK's SVD does not:
    numpy's and scipy's builds differ in the last ulp.
    """
    a = P[0, 0]; b = P[0, 1]; c = P[0, 2]
    d = P[1, 0]; e = P[1, 1]; f = P[1, 2]
    g = P[2, 0]; h = P[2, 1]; i = P[2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    det = a * A + b * B + c * C
    n2 = 0.0
    for v in (a, b, c, d, e, f, g, h, i):
        n2 += v * v
    scale = np.sqrt(n2)
    if not abs(det) > rcond * scale * scale * scale:
        return None
    out = np.empty((3, 3))
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


def _pinv3(P: np.ndarray,
           rcond: float = defaults.ACA_PINV_RCOND) -> np.ndarray:
    """Pseudo-inverse of the ACA's d x d pivot block: the closed form for
    an invertible 3x3 (``_adj_inv3``), else the SVD pseudo-inverse at
    ``rcond`` of the leading singular value (the HODLR rung's 6x6 blocks
    always land here)."""
    if P.shape[0] == 3 and P.shape[1] == 3:
        inv = _adj_inv3(P, rcond)
        if inv is not None:
            return inv
    u, s, vt = np.linalg.svd(P)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros_like(P.T)
    s_inv = np.where(s > rcond * s[0], 1.0 / np.where(s == 0, 1.0, s), 0.0)
    return (vt.T * s_inv) @ u.T


def _block_norms(sq: np.ndarray, axis: int) -> np.ndarray:
    """Frobenius norm of every d x d element block of ``sq`` (the ACA's
    pivot scores), summing over the leading index of the block first and
    the trailing one second, LEFT TO RIGHT.

    The order is part of the algorithm, not an implementation detail: as
    the residual approaches the stopping tolerance its element blocks
    become nearly tied in norm, and a one-ulp difference selects a
    different pivot -- equally accurate factors, but different ones.
    ``aca_numba`` mirrors this expression so the two implementations
    produce the same factorization (``verify_hbackend``).
    """
    x = sq ** 2
    t = x[0].copy() if axis == 0 else x[:, 0].copy()
    for k in range(1, x.shape[axis]):
        t += x[k] if axis == 0 else x[:, k]
    out = t[:, 0].copy()
    for k in range(1, t.shape[1]):
        out += t[:, k]
    return np.sqrt(out, out=out)


def aca_single(cache: BlockEvalCache, b: int, n_rows: int, n_cols: int,
               tol: float, srows: np.ndarray, scols: np.ndarray,
               sample_exact: np.ndarray, max_rank: int):
    """Element-block ACA of basis ``b``. Returns (U, V), or None when the
    element-pivot budget ``max_rank`` (``ACA_MAX_RANK_FRACTION`` of the
    shorter side, set by ``compress_block``) is spent without converging.

    Stopping is based on the TRUE residual tracked on a persistent
    random sample (``sample_exact``: exact entries at srows x scols) —
    the Frobenius-increment heuristic alone plateaus on this kernel
    (the failure mode recorded at hmatrix.py:493). The increment check
    remains only as a cheap accelerator for an early exit test.
    """
    d = cache.d
    nc3 = d * n_cols
    rsel = (d * srows[:, None] + np.arange(d)[None, :]).ravel()
    csel = (d * scols[:, None] + np.arange(d)[None, :]).ravel()
    R_sample = sample_exact.copy()
    sample_norm = np.linalg.norm(sample_exact)
    stop_tol = 0.5 * tol

    U_parts: list[np.ndarray] = []
    V_parts: list[np.ndarray] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    i_pivot = 0
    converged = False

    for _ in range(max_rank):
        used_rows.add(i_pivot)
        R_row = cache.row(b, i_pivot).copy()          # (d, d*nc)
        r0 = d * i_pivot
        for Up, Vp in zip(U_parts, V_parts):
            # The transposed factor is COPIED, not passed to BLAS as a
            # transpose flag: the two take different OpenBLAS kernels and
            # round differently, and a pivot near the stopping tolerance
            # is decided by that last ulp (``aca_numba`` copies too).
            R_row -= Up[r0:r0 + d, :] @ np.ascontiguousarray(Vp.T)

        scores = _block_norms(R_row.reshape(d, n_cols, d), 0)
        for j in used_cols:
            scores[j] = -1.0
        j_pivot = int(np.argmax(scores))
        if scores[j_pivot] <= 1e-300:
            converged = True
            break
        used_cols.add(j_pivot)

        C_col = cache.col(b, j_pivot).copy()          # (d*nr, d)
        c0 = d * j_pivot
        for Up, Vp in zip(U_parts, V_parts):
            C_col -= Up @ np.ascontiguousarray(Vp[c0:c0 + d, :].T)

        P = R_row[:, c0:c0 + d]                       # (d, d)
        V_new = (_pinv3(P) @ R_row).T                 # (d*nc, d)
        U_new = C_col                                 # (d*nr, d)

        U_parts.append(U_new)
        V_parts.append(V_new)

        # True-residual tracking on the persistent sample
        R_sample -= U_new[rsel, :] @ np.ascontiguousarray(V_new[csel, :].T)
        if sample_norm == 0.0 or \
                np.linalg.norm(R_sample) < stop_tol * sample_norm:
            converged = True
            break

        rscores = _block_norms(U_new.reshape(n_rows, d, d), 1)
        for i in used_rows:
            rscores[i] = -1.0
        i_pivot = int(np.argmax(rscores))
        if rscores[i_pivot] <= 0.0:
            converged = True
            break

    if not converged:
        return None      # rank cap reached without convergence

    if not U_parts:
        return (np.zeros((d * n_rows, 1)), np.zeros((nc3, 1)))

    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    return recompress(U, V, 0.5 * tol)


def recompress(U: np.ndarray, V: np.ndarray, tol: float):
    """Truncate U @ V.T via thin QR of both factors and an SVD."""
    Qu, Ru = np.linalg.qr(U)
    Qv, Rv = np.linalg.qr(V)
    u, s, vt = np.linalg.svd(Ru @ Rv.T, full_matrices=False)
    keep = _svd_keep(s, tol)
    return Qu @ (u[:, :keep] * s[:keep]), Qv @ vt[:keep, :].T


def _svd_keep(s: np.ndarray, tol: float) -> int:
    """Smallest k with ||s[k:]|| <= tol * s[0]."""
    if s.size == 0 or s[0] == 0.0:
        return 1
    tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
    keep = int(np.searchsorted(-tail, -tol * s[0]))
    return max(1, min(keep, s.size))


def _unit_gram(cores: list, axis: int) -> np.ndarray:
    """Gram matrix of the cores normalized to unit Frobenius norm,
    summed over the bases: ``axis=0`` gives ``sum_b C C^T`` (the column
    side), ``axis=1`` ``sum_b C^T C``.

    Normalizing is what makes the truncation PER BASIS -- the discarded
    eigenvalue sum is then each basis's squared relative error, and a raw
    stack would be free to discard a small-norm basis whose material
    coefficient is large. The Gram rather than the stack itself because
    it is the same spectrum k x k instead of k x Bk: measured 1.9 ms
    against 32 ms on a 78-column fold, 81 ms against 1.0 s on a
    378-column one.
    """
    out = None
    for c in cores:
        n = float(np.linalg.norm(c))
        if n == 0.0:
            continue
        m = np.ascontiguousarray(c) * (1.0 / n)
        g = m @ m.T if axis == 0 else m.T @ m
        out = g if out is None else out + g
    if out is None:
        k = cores[0].shape[axis]
        return np.zeros((k, k))
    return out


def _principal(G: np.ndarray, delta: float) -> np.ndarray:
    """Leading eigenvectors of a symmetric PSD Gram matrix, dropping as
    many as have eigenvalues summing to at most ``delta^2`` -- the
    tail-norm rule on the singular values of the matrix ``G`` is the
    Gram of."""
    w, Z = np.linalg.eigh(G)                      # ascending
    acc = np.cumsum(np.maximum(w, 0.0))
    drop = int(np.searchsorted(acc, delta * delta, side="right"))
    keep = max(1, w.size - drop)
    return np.ascontiguousarray(Z[:, ::-1][:, :keep])


def shared_subspace(Us: list, Vs: list, delta: float):
    """Fold the B factor pairs into one column and one row basis:
    ``(Qu, Qv, cores)`` with ``U_b V_b^T = Qu cores[b] Qv^T``.

    Thin QR of the concatenated factors gives the exact joint spans; the
    cores are then truncated, once per side, so that EVERY basis is
    reproduced to ``delta`` relative Frobenius error (``_unit_gram``,
    ``_principal``). The B cores that remain are k x k, so the
    material recombination that used to re-factorize (3n, sum_b k_b)
    factors becomes a k x k sum plus a k x k SVD.

    ``delta`` is spent out of the block's error budget, not on top of it:
    ``ACA_JOINT_TOL_FACTOR`` x the block tolerance.

    ``np.linalg.qr`` specifically, and in numpy rather than in the nogil
    kernel: the QR of a rank-deficient tall factor is where this
    machine's LAPACK misbehaves, and only numpy's wrapper survives it.
    numba's returns an R that is read after free
    (``aca_numba._thin_qr``); scipy's ``qr(mode="economic")`` is 3-25x
    faster than numpy's on the same shapes and silently corrupts a few
    blocks per model (measured: 2-6 of the 45 stored blocks of the
    refined fault zone, nondeterministically, which is what the
    stored-block scan of ``verify_hbackend`` catches). What numpy's
    slower QR buys is that scan coming back clean, so it stays.

    It is paid ONCE per block at build -- ``hop`` folds on the main
    thread, chunk by chunk behind the block pool, a pool of numpy LAPACK
    being 0.6x of one thread on OpenBLAS's buffer lock -- and no
    material after the first pays it again.
    """
    ks = [u.shape[1] for u in Us]
    Qu, Ru = np.linalg.qr(np.hstack(Us))
    Qv, Rv = np.linalg.qr(np.hstack(Vs))
    cores, off = [], 0
    for k in ks:
        cores.append(np.ascontiguousarray(Ru[:, off:off + k])
                     @ np.ascontiguousarray(Rv[:, off:off + k].T))
        off += k
    Wu = _principal(_unit_gram(cores, 0), delta)
    cores = [Wu.T @ c for c in cores]
    Wv = _principal(_unit_gram(cores, 1), delta)
    return Qu @ Wu, Qv @ Wv, np.array([c @ Wv for c in cores])


def _rel_err(approx: np.ndarray, exact: np.ndarray) -> float:
    denom = float(np.linalg.norm(exact))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(approx - exact)) / denom


def _dof_sel(elems: np.ndarray, d: int) -> np.ndarray:
    return (d * elems[:, None] + np.arange(d)[None, :]).ravel()


def _exact_stack(cache: BlockEvalCache) -> np.ndarray:
    """The block's exact basis stack, with the pivot caches released
    first. Only a caller that asked for an exact payload reaches it: the
    operator path re-evaluates such a block from the kernels instead."""
    cache.clear()
    return cache.dense()


def svd_from_stack(dense: np.ndarray, tol: float):
    """Per-basis SVD truncation of an exact block stack ``(B, m, n)`` at
    ``tol``: ``([U_b], [V_b], max relative tail norm)``. The last resort
    of a caller that needs a PAYLOAD (the HODLR rung) for a block whose
    retried ACA still failed its certificate and which is too large to
    keep exact. It costs B x O(m n min(m, n)) at one BLAS thread -- 13 s
    on a 404-element T block -- so the operator path does not use it: it
    applies such a block exactly from the kernels instead."""
    Us: list = []
    Vs: list = []
    err = 0.0
    for b in range(dense.shape[0]):
        u, s, vt = np.linalg.svd(dense[b], full_matrices=False)
        keep = _svd_keep(s, tol)
        Us.append(u[:, :keep] * s[:keep])
        Vs.append(vt[:keep, :].T)
        total = float(np.linalg.norm(s))
        if total > 0.0:
            err = max(err, float(np.linalg.norm(s[keep:])) / total)
    return Us, Vs, err


def certify_combined(U: np.ndarray, V: np.ndarray, c: np.ndarray,
                     rows_exact: np.ndarray, cols_exact: np.ndarray,
                     cert_rows: np.ndarray, cert_cols: np.ndarray,
                     d: int = 3) -> float:
    """Relative Frobenius error of the recombined block ``U @ V.T`` for
    material ``c`` against the exact combined rows and columns
    ``sum_b c_b rows_exact[b]`` / ``sum_b c_b cols_exact[b]`` (the
    ``(B, d m, d nc)`` and ``(B, d nr, d m)`` basis stacks at the
    block-local elements ``cert_rows`` / ``cert_cols``)."""
    rsel = _dof_sel(cert_rows, d)
    csel = _dof_sel(cert_cols, d)
    return max(_rel_err(U[rsel] @ V.T, np.tensordot(c, rows_exact, axes=1)),
               _rel_err(U @ V[csel].T, np.tensordot(c, cols_exact, axes=1)))


def draw_lines(rng, n_rows: int, n_cols: int):
    """(stopping sample rows, columns, certificate rows, columns) of one
    block, drawn from ``rng`` in this fixed order so that every
    implementation of the ACA sees the same lines from the same seed."""
    m = defaults.ACA_CERTIFY_LINES
    return (rng.choice(n_rows, size=min(10, n_rows), replace=False),
            rng.choice(n_cols, size=min(10, n_cols), replace=False),
            rng.choice(n_rows, size=min(m, n_rows), replace=False),
            rng.choice(n_cols, size=min(m, n_cols), replace=False))


def _aca_python(cache: BlockEvalCache, n_basis: int, tol: float, srows,
                scols, cert_rows, cert_cols, max_rank: int):
    """Per-basis ACA and certificate in Python: ``([U_b], [V_b],
    certified error, capped)``. The reference implementation --
    ``aca_numba.block_factors`` is gated against it block by block."""
    n_rows = len(cache.rows)
    n_cols = len(cache.cols)
    d = cache.d
    stop_exact = cache.stack_fn(cache.rows[srows], cache.cols[scols])

    Us: list = []
    Vs: list = []
    for b in range(n_basis):
        uv = aca_single(cache, b, n_rows, n_cols, tol,
                        srows, scols, stop_exact[b], max_rank)
        if uv is None:
            return [], [], 0.0, True
        Us.append(uv[0])
        Vs.append(uv[1])

    rsel = _dof_sel(cert_rows, d)
    csel = _dof_sel(cert_cols, d)
    rows_exact = cache.stack_fn(cache.rows[cert_rows], cache.cols)
    cols_exact = cache.stack_fn(cache.rows, cache.cols[cert_cols])
    err = 0.0
    for b in range(n_basis):
        err = max(err,
                  _rel_err(Us[b][rsel] @ np.ascontiguousarray(Vs[b].T),
                           rows_exact[b]),
                  _rel_err(Us[b] @ np.ascontiguousarray(Vs[b][csel].T),
                           cols_exact[b]))
    return Us, Vs, err, False


def compress_block(cache: BlockEvalCache, n_basis: int,
                   tol: float = defaults.BLOCK_COMPRESSION_TOL,
                   rng=None, exact_payload: bool = True,
                   fold: bool = True) -> BlockResult:
    """Compress all basis matrices of one admissible block: the ACA,
    shared-subspace and certificate stage (``cache.aca_fn``, else
    ``_aca_python``) plus the policy that turns its outcome into a
    payload.

    Per-basis ACA with a shared kernel-evaluation cache, stopped on one
    random sample, folded into one shared subspace
    (``shared_subspace``) and CERTIFIED per basis AS STORED on an
    independent draw of ``ACA_CERTIFY_LINES`` full rows and columns (the
    same rows and columns then certify each material combination at view
    time, where the block is certified as APPLIED). The
    lines are drawn here, from ``rng`` alone, so the factors are
    deterministic for a given seed and identical across implementations
    of the ACA.

    Outcomes (``BlockResult``): certified factors; the exact block when
    any basis hits the rank cap (not low rank at this tolerance); on a
    certificate error above ``ACA_CERTIFY_FACTOR * tol``, one RETRY of
    the same ACA at ``tol / ACA_RETRY_TOL_FACTOR`` -- a failed
    certificate means the sampled stop fired early, and the retry buys
    the missing pivots for a few per cent of the block's kernel work --
    and, if that fails too, the exact block.

    ``fold=False`` stops one step short of a ``SharedLR`` and returns
    the block's per-basis factors as a ``PendingLR``: the shared
    subspace is dense numpy algebra, so a caller compressing blocks on a
    thread pool folds them itself, off the pool.

    ``exact_payload=False`` (the operator path) returns ``payload=None``
    for every exact outcome: the caller re-evaluates such a block from
    the kernels with the near-field leaves (``la.flatview.dense_leaves``),
    so no (B, 3nr, 3nc) stack is materialized at all, and no size-dependent
    SVD fallback is reachable. The HODLR rung keeps the default -- it has
    no leaf kernel behind it -- and SVD-truncates a stack whose shorter
    side reaches ``ACA_SVD_FALLBACK_MIN_SIDE``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_rows = len(cache.rows)
    n_cols = len(cache.cols)
    srows, scols, cert_rows, cert_cols = draw_lines(rng, n_rows, n_cols)
    max_rank = min(max(8, int(min(n_rows, n_cols)
                              * defaults.ACA_MAX_RANK_FRACTION)),
                   n_rows, n_cols)

    def _aca(at_tol):
        if cache.aca_fn is None:
            return _aca_python(cache, n_basis, at_tol, srows, scols,
                               cert_rows, cert_cols, max_rank)
        return cache.aca_fn(at_tol, srows, scols, cert_rows, cert_cols,
                            max_rank)

    def _payload(Us, Vs):
        """The block's B factor pairs, folded into one shared subspace
        unless the caller asked to fold them itself (``fold=False``)."""
        pending = PendingLR(U=Us, V=Vs, cert_rows=cert_rows,
                            cert_cols=cert_cols)
        return pending if not fold else pending.fold(tol)

    limit = defaults.ACA_CERTIFY_FACTOR * tol
    Us, Vs, err, capped = _aca(tol)
    retried = not capped and err > limit
    if retried:
        Us, Vs, err, capped = _aca(tol / defaults.ACA_RETRY_TOL_FACTOR)
    cache.clear()
    if capped:
        return BlockResult(_exact_stack(cache) if exact_payload else None,
                           capped=True, retried=retried)
    if err <= limit:
        return BlockResult(_payload(Us, Vs), retried=retried, max_err=err)
    if not exact_payload:
        return BlockResult(None, fallback=True, retried=True)
    dense = _exact_stack(cache)
    if min(n_rows, n_cols) < defaults.ACA_SVD_FALLBACK_MIN_SIDE:
        return BlockResult(dense, fallback=True, retried=True)
    Us, Vs, err = svd_from_stack(dense, tol)
    return BlockResult(_payload(Us, Vs), fallback=True, retried=True,
                       max_err=err)
