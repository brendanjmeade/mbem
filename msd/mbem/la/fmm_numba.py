"""The bbFMM far-field point kernel in one numba kernel.

``far_apply`` is ``fmm._far_apply`` -- the field at a box's target nodes from
point charges at another box's source nodes, summed over the eps passes --
with the nine ``np.einsum`` calls gone. Same arguments, same return, and the
reference stays in ``fmm.py`` as the oracle this is gated against
(``verify_fmm`` clause [b]); ``PairFMM(m2l="numba")`` selects this one.

WHY THIS AND NOT THE SHARED M2L TABLE FIRST. The table's arithmetic is not
cheaper -- 28.3 Mflop per V pair against 34.1 matrix-free at the shipping
orders, 1.2x -- so what a table buys is the RATE, by turning the sum into a
GEMM. This kernel buys the same rate without the bytes: the reference runs
its two dominant einsums (``ts,simk,tsm->tik`` and ``tsj,sjmk,tsm->tsk``,
88 % of the call) unoptimised on one core, materialising a (nt, ns, 3, 3, k)
temporary it reads once, while the same contraction fused per (target,
source) pair keeps the 3x3 charge in registers and touches memory once per
pair. The table is still worth taking afterwards, but it is worth it
key-major -- batched over the V pairs sharing a transfer offset -- and that
is a different data structure, not a faster inner loop.

THE EPS PASSES ARE STACKED, not looped in Python: ``K(eps) = K0 + eps^2 K1``
with both terms translation-invariant, so a pass differs only in a radial
power and a coefficient triple (``fmm._far_params``). Both ride one pass
over the geometry here, which is why ``d`` and ``1/r`` are formed once per
(target, source) pair and reused.

Two entry points per kernel, per the tree's rule that a ``parallel=True``
kernel is never called from a Python thread (macOS workqueue): the
``parallel`` form prange's over targets for a single-threaded caller, the
``_serial`` form releases the GIL for a caller that is itself pooled. Both
share one core, so there is one arithmetic to gate.

Own module, so ``cache=True`` here is invalidated by an edit to this file
rather than by one to ``fmm.py``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..kernels import KERNEL_U

# Parameter row: (radial power, coefficients). U fills 3 columns, T fills 4.
_PRM_COLS = 4


@njit(cache=True, nogil=True)
def _radial(inv, inv2, power):
    """``1/r^power`` from ``1/r`` and ``1/r^2``; power is odd."""
    out = inv
    for _ in range((power - 1) // 2):
        out = out * inv2
    return out


@njit(cache=True, nogil=True)
def _u_core(t0, t1, xt, ys, q, prm, out):
    """U-kernel field for targets ``[t0, t1)``, accumulated into ``out``."""
    ns = ys.shape[0]
    npass = q.shape[0]
    k = q.shape[3]
    for t in range(t0, t1):
        x0 = xt[t, 0]
        x1 = xt[t, 1]
        x2 = xt[t, 2]
        for s in range(ns):
            d0 = x0 - ys[s, 0]
            d1 = x1 - ys[s, 1]
            d2 = x2 - ys[s, 2]
            inv = 1.0 / np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            inv2 = inv * inv
            for ip in range(npass):
                wk = _radial(inv, inv2, int(prm[ip, 0]))
                wk2 = wk * inv2
                b1 = prm[ip, 1] * wk
                b2 = prm[ip, 2] * wk2
                for kk in range(k):
                    c0 = q[ip, s, 0, kk]
                    c1 = q[ip, s, 1, kk]
                    c2 = q[ip, s, 2, kk]
                    g = b2 * (d0 * c0 + d1 * c1 + d2 * c2)
                    out[t, 0, kk] += b1 * c0 + d0 * g
                    out[t, 1, kk] += b1 * c1 + d1 * g
                    out[t, 2, kk] += b1 * c2 + d2 * g


@njit(cache=True, nogil=True)
def _t_core(t0, t1, xt, ys, q, prm, out):
    """T-kernel field for targets ``[t0, t1)``, accumulated into ``out``.

    The three terms are the reference's, fused: ``a1 d_i tr(q)``,
    ``a2 (q + q^T)_im d_m`` and ``a3 d_i d_j q_jm d_m``, the first and third
    sharing the ``d_i`` factor.
    """
    ns = ys.shape[0]
    npass = q.shape[0]
    k = q.shape[4]
    for t in range(t0, t1):
        x0 = xt[t, 0]
        x1 = xt[t, 1]
        x2 = xt[t, 2]
        for s in range(ns):
            d0 = x0 - ys[s, 0]
            d1 = x1 - ys[s, 1]
            d2 = x2 - ys[s, 2]
            inv = 1.0 / np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            inv2 = inv * inv
            for ip in range(npass):
                wk = _radial(inv, inv2, int(prm[ip, 0]))
                wk2 = wk * inv2
                a1 = prm[ip, 1] * wk
                a2 = prm[ip, 2] * wk
                a3 = prm[ip, 3] * wk2
                for kk in range(k):
                    q00 = q[ip, s, 0, 0, kk]
                    q01 = q[ip, s, 0, 1, kk]
                    q02 = q[ip, s, 0, 2, kk]
                    q10 = q[ip, s, 1, 0, kk]
                    q11 = q[ip, s, 1, 1, kk]
                    q12 = q[ip, s, 1, 2, kk]
                    q20 = q[ip, s, 2, 0, kk]
                    q21 = q[ip, s, 2, 1, kk]
                    q22 = q[ip, s, 2, 2, kk]
                    r0 = q00 * d0 + q01 * d1 + q02 * d2
                    r1 = q10 * d0 + q11 * d1 + q12 * d2
                    r2 = q20 * d0 + q21 * d1 + q22 * d2
                    g = a1 * (q00 + q11 + q22) + a3 * (d0 * r0 + d1 * r1
                                                       + d2 * r2)
                    out[t, 0, kk] += d0 * g + a2 * (
                        r0 + q00 * d0 + q10 * d1 + q20 * d2)
                    out[t, 1, kk] += d1 * g + a2 * (
                        r1 + q01 * d0 + q11 * d1 + q21 * d2)
                    out[t, 2, kk] += d2 * g + a2 * (
                        r2 + q02 * d0 + q12 * d1 + q22 * d2)


@njit(cache=True, parallel=True)
def far_u(xt, ys, q, prm, out):
    """U kernel, prange over targets. Never call from a Python thread."""
    for t in prange(xt.shape[0]):
        _u_core(t, t + 1, xt, ys, q, prm, out)


@njit(cache=True, parallel=True)
def far_t(xt, ys, q, prm, out):
    """T kernel, prange over targets. Never call from a Python thread."""
    for t in prange(xt.shape[0]):
        _t_core(t, t + 1, xt, ys, q, prm, out)


@njit(cache=True, nogil=True)
def far_u_serial(xt, ys, q, prm, out):
    """Serial/nogil :func:`far_u`, for a caller on a thread pool."""
    _u_core(0, xt.shape[0], xt, ys, q, prm, out)


@njit(cache=True, nogil=True)
def far_t_serial(xt, ys, q, prm, out):
    """Serial/nogil :func:`far_t`, for a caller on a thread pool."""
    _t_core(0, xt.shape[0], xt, ys, q, prm, out)


def far_apply(kernel, xt, ys, charges, params, serial=False):
    """``fmm._far_apply`` on the numba kernels; identical arguments.

    ``charges`` and ``params`` are stacked into the contiguous arrays a
    kernel takes. The stack is O(n_source) against the O(n_target n_source)
    it feeds, so it is not on the critical path.
    """
    xt = np.ascontiguousarray(xt, dtype=float)
    ys = np.ascontiguousarray(ys, dtype=float)
    q = np.ascontiguousarray(np.stack([np.asarray(c, dtype=float)
                                       for c in charges]))
    prm = np.zeros((len(params), _PRM_COLS))
    for i, row in enumerate(params):
        prm[i, :len(row)] = row
    out = np.zeros((xt.shape[0], 3, q.shape[-1]))
    if kernel == KERNEL_U:
        (far_u_serial if serial else far_u)(xt, ys, q, prm, out)
    else:
        (far_t_serial if serial else far_t)(xt, ys, q, prm, out)
    return out
