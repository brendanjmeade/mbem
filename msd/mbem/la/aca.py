"""Block ACA compression of material-basis matrix stacks.

Each admissible block stores the basis matrices SEPARATELY in low rank:

    A_b ~ U_b @ V_b.T          b = 1..B   (B = 3 for U-kernel, 6 for T)

Per-basis storage beats compressing the stacked matrix because the basis
matrices do not share row/column spaces (a stacked factorization pays up
to a Bx rank penalty — measured, not hypothetical). The per-material
combined block sum_b c_b U_b V_b^T is re-truncated once per material by
QR+SVD and cached by the operator layer, so matvec rank stays at the
combination's own epsilon-rank.

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
full random rows and as many full random columns evaluated exactly:
per basis here, at build, and per MATERIAL when the operator layer
combines the factors (``certify_combined``), because a per-basis
tolerance does not bound a combination whose coefficients have mixed
signs (the T coefficients do); the combined check covers every material
actually applied, declared at build or not. A block whose ACA hits the
rank cap is not low rank at this tolerance and is applied exactly, like a
near-field leaf; a block that fails its per-basis certificate is applied
exactly when small and SVD-truncated from the exact stack when large
(``ACA_SVD_FALLBACK_MIN_SIDE``), so correctness never depends on the
ACA's own stopping rule. What "exactly" costs is the caller's choice:
the operator path (``exact_payload=False``) keeps only the block's
indices and re-evaluates it per material with the near-field leaves,
while the HODLR rung takes the stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import defaults


@dataclass
class BasisLR:
    """Per-basis low-rank factors: A_b ~ U[b] @ V[b].T (ragged ranks),
    with the block-local element rows and columns (``cert_rows``,
    ``cert_cols``) every material combination of them is certified on."""
    U: list          # B arrays (3nr, k_b)
    V: list          # B arrays (3nc, k_b)
    cert_rows: np.ndarray
    cert_cols: np.ndarray

    @property
    def ranks(self) -> list:
        return [u.shape[1] for u in self.U]

    def nbytes(self) -> int:
        return sum(u.nbytes + v.nbytes for u, v in zip(self.U, self.V))


@dataclass
class BlockResult:
    """Outcome of ``compress_block`` for one admissible block.

    ``payload`` is a ``BasisLR``, the exact ``(B, d nr, d nc)`` stack, or
    ``None`` when the caller asked for no exact payload
    (``exact_payload=False``): the block is not low rank at this
    tolerance and the caller re-evaluates it from the kernels.
    ``capped``: the ACA hit ``ACA_MAX_RANK_FRACTION``. ``fallback``: the
    per-basis certificate failed and the block was re-done from its exact
    stack. ``max_err`` is the largest certified relative Frobenius error
    over the bases (0 when exact).
    """
    payload: object
    capped: bool = False
    fallback: bool = False
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


def _rel_err(approx: np.ndarray, exact: np.ndarray) -> float:
    denom = float(np.linalg.norm(exact))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(approx - exact)) / denom


def _dof_sel(elems: np.ndarray, d: int) -> np.ndarray:
    return (d * elems[:, None] + np.arange(d)[None, :]).ravel()


def _exact_stack(cache: BlockEvalCache, fallback_gate) -> np.ndarray:
    """The block's exact basis stack, with the pivot caches released
    first and the materialization bounded by ``fallback_gate``."""
    cache.clear()
    if fallback_gate is None:
        return cache.dense()
    with fallback_gate:
        return cache.dense()


def svd_from_stack(dense: np.ndarray, tol: float):
    """Per-basis SVD truncation of an exact block stack ``(B, m, n)`` at
    ``tol``: ``([U_b], [V_b], max relative tail norm)``. The fallback for
    a block whose ACA factors failed their certificate and which is too
    large to keep exact."""
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
                   rng=None, fallback_gate=None,
                   exact_payload: bool = True) -> BlockResult:
    """Compress all basis matrices of one admissible block: the ACA and
    certificate stage (``cache.aca_fn``, else ``_aca_python``) plus the
    policy that turns its outcome into a payload.

    Per-basis ACA with a shared kernel-evaluation cache, stopped on one
    random sample and CERTIFIED per basis on an independent draw of
    ``ACA_CERTIFY_LINES`` full rows and columns (the same rows and
    columns then certify each material combination at view time). The
    lines are drawn here, from ``rng`` alone, so the factors are
    deterministic for a given seed and identical across implementations
    of the ACA.

    Outcomes (``BlockResult``): certified factors; the exact block when
    any basis hits the rank cap (not low rank at this tolerance); on a
    certificate error above ``ACA_CERTIFY_FACTOR * tol``, the exact block
    when its shorter side is below ``ACA_SVD_FALLBACK_MIN_SIDE`` elements,
    else its per-basis SVD truncation at ``tol``.

    ``exact_payload=False`` (the operator path) returns ``payload=None``
    instead of an exact stack: the caller re-evaluates such a block from
    the kernels with the near-field leaves (``la.flatview.dense_leaves``),
    so no (B, 3nr, 3nc) stack is materialized at all. The HODLR rung
    keeps the default -- it has no leaf kernel behind it.

    ``fallback_gate`` (a semaphore-like context manager) bounds how many
    exact stacks may be materialized CONCURRENTLY: one T stack is
    B x (3 n)^2 x 8 bytes, 0.45 GB at n = MAX_ADMISSIBLE_BLOCK = 1024.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_rows = len(cache.rows)
    n_cols = len(cache.cols)
    srows, scols, cert_rows, cert_cols = draw_lines(rng, n_rows, n_cols)
    max_rank = min(max(8, int(min(n_rows, n_cols)
                              * defaults.ACA_MAX_RANK_FRACTION)),
                   n_rows, n_cols)

    if cache.aca_fn is None:
        Us, Vs, err, capped = _aca_python(cache, n_basis, tol, srows, scols,
                                          cert_rows, cert_cols, max_rank)
    else:
        Us, Vs, err, capped = cache.aca_fn(tol, srows, scols, cert_rows,
                                           cert_cols, max_rank)
    cache.clear()
    if capped:
        return BlockResult(
            _exact_stack(cache, fallback_gate) if exact_payload else None,
            capped=True)
    if err <= defaults.ACA_CERTIFY_FACTOR * tol:
        return BlockResult(BasisLR(U=Us, V=Vs, cert_rows=cert_rows,
                                   cert_cols=cert_cols), max_err=err)

    small = min(n_rows, n_cols) < defaults.ACA_SVD_FALLBACK_MIN_SIDE
    if small and not exact_payload:
        return BlockResult(None, fallback=True)
    dense = _exact_stack(cache, fallback_gate)
    if small:
        return BlockResult(dense, fallback=True)
    Us, Vs, err = svd_from_stack(dense, tol)
    return BlockResult(BasisLR(U=Us, V=Vs, cert_rows=cert_rows,
                               cert_cols=cert_cols),
                       fallback=True, max_err=err)
