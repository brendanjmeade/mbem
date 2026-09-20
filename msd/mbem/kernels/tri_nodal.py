"""Nodal (Lagrange P0/P1/P2) analytical triangle kernels, numba.

Per (field point, source triangle) pair the per-node weighted moment tables
W_k[n][a,b] = int_T N_k xi_1^a xi_2^b / R_eps^n dS come from the divergence-
theorem recursion on cancellation-free edge primitives (closed form, small-|u|
and large-|u| series) within NODAL_D_STAR * L of the centroid (L = longest
edge, distance sqrt(|x - c|^2 + eps^2)), and from a collapsed product Gauss
rule of the smooth integrand beyond; every kernel is tri_kernels' contraction
of W_k.  Nodes: p = 0 centroid; p = 1 v1, v2, v3; p = 2 v1, v2, v3, m12, m23,
m31.  Column 3 * (K * s + k) + j (element s, node k, component j); densities
(K * N_s, 3).  Basis slots and signs are tri_kernels' (U: [I1 d, eps^2 I3 d,
T2_3]; T: [N[P1..P3], R[P1..P3]]), so basis.u_coeffs / t_coeffs recombine them
unchanged.  Order 0 routes to tri_kernels' pair code.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import numpy as np
from numba import njit, prange

from .. import defaults
from . import tri_kernels as tk
from .tri_kernels import (_build_T2, _build_T3, _build_T4, _contract_NR,
                          _solid_angle, _tri_frame)

# Moment-table selectors (which degrees a driver needs, clq.moments
# kernel_degrees); NOT the kernel identifiers of kernels/__init__ (there
# KERNEL_T = "H" is the slip -> displacement kernel that clq calls "U").
KT = 1     # slip -> displacement basis (clq "U")
KU = 2     # force -> displacement basis (clq "G")
KH = 4     # slip -> TOTAL stress (clq "H")
KS = 8     # force -> stress (clq "S")
KE = 16    # eigenstress weight (clq "E")

_M_ROWS = (-1, 1, 3, 5)         # edge-primitive rows, index (m + 1) // 2
_MAX_EDGE_DEG = 5               # P2: {5: 5, 3: 4, 1: 3, -1: 1}
_MAX_TABLE_DEG = 6              # P2: n = 7 at degree 4 + 2
_GEO_LEN = 29                   # ex ey nhat cen (12) p2d (6) abc (9) L area2


# ---------------------------------------------------------------------
# Python-level spec: node lattice, degrees, series coefficients, far rules
# ---------------------------------------------------------------------

def n_nodes(order: int) -> int:
    """Lagrange nodes per triangle, 1 / 3 / 6 for order 0 / 1 / 2."""
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2 (got {order!r})")
    return (order + 1) * (order + 2) // 2


def _lattice(order: int) -> np.ndarray:
    """Multi-indices alpha (K, 3): vertices, then edges 12, 23, 31, then interior."""
    p = order
    if p == 0:
        return np.zeros((1, 3), dtype=np.int64)
    out = [[p, 0, 0], [0, p, 0], [0, 0, p]]
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        for s in range(1, p):
            a = [0, 0, 0]
            a[i] = p - s
            a[j] = s
            out.append(a)
    for a1 in range(1, p):
        for a2 in range(1, p - a1):
            if p - a1 - a2 >= 1:
                out.append([a1, a2, p - a1 - a2])
    return np.array(out, dtype=np.int64)


def _kernel_degrees(order: int, flags: int) -> dict:
    need: dict = {}

    def add(n, d):
        need[n] = max(need.get(n, -1), d)

    if flags & KT:
        add(3, 1 + order); add(5, 3 + order)
    if flags & KH:
        add(3, 0 + order); add(5, 2 + order); add(7, 4 + order)
    if flags & KE:
        add(7, 0 + order)
    if flags & KU:
        add(1, 0 + order); add(3, 2 + order)
    if flags & KS:
        add(3, 1 + order); add(5, 3 + order)
    return need


def _close_degrees(need: dict) -> dict:
    """Lower-n degrees the recursion consumes plus the vertical seeds."""
    deg = dict(need)
    n = max(deg)
    while True:
        d = deg.get(n, -1)
        if d >= 2:
            deg[n - 2] = max(deg.get(n - 2, -1), d - 2)
        if n - 2 < min(deg):
            break
        n -= 2
    deg[3] = max(deg.get(3, -1), 0)
    for n in list(deg):
        if n >= 5:
            for m in range(3, n, 2):
                deg[m] = max(deg.get(m, -1), 0)
        if n <= 1:
            for m in range(n, 4, 2):
                deg[m] = max(deg.get(m, -1), 0)
    return {n: d for n, d in deg.items() if d >= 0}


def _edge_degrees(deg: dict) -> dict:
    edge: dict = {}
    for n, d in deg.items():
        if d >= 1:
            edge[n - 2] = max(edge.get(n - 2, -1), d - 1)
    for n in deg:
        if n >= 5:
            edge[n - 2] = max(edge.get(n - 2, -1), 0)
        if n <= 1:
            edge[n] = max(edge.get(n, -1), 0)
    return edge


def _binom_half_table(n_terms: int) -> np.ndarray:
    """C(-m/2, j) for m in _M_ROWS, j < n_terms, exact then rounded once."""
    out = np.empty((len(_M_ROWS), n_terms))
    for mi, m in enumerate(_M_ROWS):
        a = Fraction(-m, 2)
        c = Fraction(1)
        for j in range(n_terms):
            out[mi, j] = float(c)
            c *= (a - j) / (j + 1)
    return out


def _gauss_triangle(n: int):
    """Collapsed product Gauss-Legendre rule on (0,0),(1,0),(0,1); sum w = 1/2."""
    g, w = np.polynomial.legendre.leggauss(n)
    g = 0.5 * (g + 1.0)
    w = 0.5 * w
    gi, gj = np.meshgrid(g, g, indexing="ij")
    wi, wj = np.meshgrid(w, w, indexing="ij")
    xi1 = gi.ravel()
    xi2 = (gj * (1.0 - gi)).ravel()
    ww = (wi * wj * (1.0 - gi)).ravel()
    return xi1, xi2, ww


def _lagrange_values(order: int, lam: np.ndarray) -> np.ndarray:
    """N_k(lam) (Q, K) from the lattice product formula."""
    lat = _lattice(order)
    Q = lam.shape[0]
    out = np.ones((Q, lat.shape[0]))
    for k, alpha in enumerate(lat):
        for i in range(3):
            for j in range(int(alpha[i])):
                out[:, k] *= (order * lam[:, i] - j) / (j + 1)
    return out


def _far_rules(order: int):
    """Both Gauss rules concatenated: lam (Q, 3), weights (Q,), N_k (Q, K), split."""
    lams, ws = [], []
    for n in (defaults.NODAL_FAR_GAUSS_N, defaults.NODAL_FAR_GAUSS_N_DISTANT):
        x1, x2, w = _gauss_triangle(n)
        lams.append(np.stack([1.0 - x1 - x2, x1, x2], axis=1))
        ws.append(w)
    lam = np.ascontiguousarray(np.concatenate(lams))
    w = np.ascontiguousarray(np.concatenate(ws))
    return lam, w, np.ascontiguousarray(_lagrange_values(order, lam)), ws[0].size


_SPEC_CACHE: dict = {}


def _spec(order: int, flags: int) -> tuple:
    """Immutable per-(order, kernel) spec tuple consumed by the njit drivers."""
    key = (int(order), int(flags))
    s = _SPEC_CACHE.get(key)
    if s is None:
        s = _build_spec(*key)
        _SPEC_CACHE[key] = s
    return s


def _build_spec(order: int, flags: int) -> tuple:
    K = n_nodes(order)
    need0 = _kernel_degrees(0, flags)
    deg = _close_degrees(_kernel_degrees(order, flags))
    edge = _edge_degrees(deg)
    if deg.get(-1, -1) >= 1 or min(edge) < -1 or max(edge.values()) > _MAX_EDGE_DEG:
        raise ValueError(f"order {order} kernel {flags}: degrees beyond the P2 spec")
    for m, kmax in edge.items():
        for k in range(2, kmax + 1):
            if edge.get(m - 2, -1) < k - 2:
                raise ValueError("edge-primitive spec is not closed under the recursion")
    q_lam, q_w, q_N, q_split = _far_rules(order)
    binom = _binom_half_table(defaults.NODAL_SERIES_TERMS)
    if binom.shape[1] != defaults.NODAL_SERIES_TERMS:
        raise ValueError("series coefficient table does not match NODAL_SERIES_TERMS")
    ispec = np.full(16, -1, dtype=np.int64)
    ispec[0] = order
    ispec[1] = K
    ispec[2] = q_split
    for n, d in deg.items():
        ispec[3 + (n + 1) // 2] = d           # rows n = -1, 1, 3, 5, 7 -> 3..7
    for m, d in edge.items():
        ispec[8 + (m + 1) // 2] = d           # rows m = -1, 1, 3, 5 -> 8..11
    for n, d in need0.items():
        ispec[12 + (n - 1) // 2] = d          # rows n = 1, 3, 5, 7 -> 12..15
    fspec = np.array([defaults.NODAL_D_STAR, defaults.NODAL_D_STAR_DISTANT,
                      defaults.NODAL_SMALL_U_OVER_RHO,
                      defaults.NODAL_SERIES_U_OVER_RHO])
    return (ispec, fspec, _lattice(order), binom, q_lam, q_w, q_N)


_COMB = np.array([[float(comb(a, i)) if i <= a else 0.0
                   for i in range(_MAX_EDGE_DEG + 1)]
                  for a in range(_MAX_EDGE_DEG + 1)])


# ---------------------------------------------------------------------
# Edge primitives: cancellation-free int_{ua}^{ub} u^k / R^m du
# ---------------------------------------------------------------------

@njit(cache=True, inline="always")
def _series_offset(n_terms):
    """Table index of q = 0 in the large-|u| chain: q runs from
    1 - m - 2 (n_terms - 1) >= -(2 n_terms + 2) up to _MAX_EDGE_DEG + 2."""
    return 2 * n_terms + 2


@njit(cache=True, inline="always")
def _series_len(n_terms):
    """Length of every series table: the q chain above and the small-|u|
    powers p <= _MAX_EDGE_DEG + 2 (n_terms - 1) + 1 both fit."""
    return _series_offset(n_terms) + _MAX_EDGE_DEG + 3


@njit(cache=True, inline="always")
def _ipow(x, n):
    """x ** n for a small int n by multiplication (1 / x for negative n)."""
    out = 1.0
    if n >= 0:
        for _ in range(n):
            out = out * x
    else:
        xi = 1.0 / x
        for _ in range(-n):
            out = out * xi
    return out


@njit(cache=True)
def _series_large_row(m, kmax, ua, ub, rho2, bm, n_terms, P, mi, SER):
    """Large-|u| series for k = 0..kmax at one m: 0 < ua < ub, rho2/ua^2 <= 1/4.

    sum_j C(-m/2, j) x^j [t^q / q]_1^r with x = rho^2/ua^2, r = ub/ua,
    q = k - m + 1 - 2j.  r^q - 1 comes from the one-step chains
    t_{q+1} = t_q r + (r - 1), t_{q-1} = t_q / r + (1/r - 1) seeded at the
    exact t_0 = 0: every term of each chain has one sign, so no cancellation
    and no transcendental per term; q = 0 is log r.
    """
    x = rho2 / (ua * ua)
    r = ub / ua
    log_r = np.log1p((ub - ua) / ua)
    up = (ub - ua) / ua                  # r - 1
    dn = -(ub - ua) / ub                 # 1/r - 1
    rinv = 1.0 / r
    T = SER[0]
    off = _series_offset(n_terms)
    T[off] = 0.0
    qmax = kmax - m + 1
    for q in range(1, qmax + 1):
        T[off + q] = T[off + q - 1] * r + up
    qmin = 1 - m - 2 * (n_terms - 1)
    for q in range(-1, qmin - 1, -1):
        T[off + q] = T[off + q + 1] * rinv + dn
    base = _ipow(ua, 1 - m)
    for k in range(kmax + 1):
        out = 0.0
        xj = 1.0
        for j in range(n_terms):
            q = k - m + 1 - 2 * j
            if q == 0:
                term = log_r
            else:
                term = T[off + q] / q
            out = out + bm[j] * xj * term
            xj = xj * x
        P[mi, k] = base * out
        base = base * ua


@njit(cache=True)
def _series_small_row(m, kmax, ua, ub, rho2, bm, n_terms, P, mi, SER):
    """Small-|u| series for k = 0..kmax at one m: max(|ua|, |ub|) <= rho/2.

    rho^(k+1-m) sum_j C(-m/2, j) [y^p / p]_{ya}^{yb}, p = k + 2j + 1, y = u/rho.
    Same-sign spans with 1/2 < ub/ua < 2 take yb^p - ya^p = ya^p (r^p - 1)
    with r^p - 1 from the one-sign chain t_{p+1} = t_p r + (r - 1).
    """
    rho = np.sqrt(rho2)
    ya = ua / rho
    yb = ub / rho
    pmax = kmax + 2 * (n_terms - 1) + 1
    YA = SER[1]
    YB = SER[2]
    E = SER[3]
    YA[0] = 1.0
    YB[0] = 1.0
    for p in range(1, pmax + 1):
        YA[p] = YA[p - 1] * ya
        YB[p] = YB[p - 1] * yb
    close = False
    if ua * ub > 0.0:
        ratio = ub / ua
        close = (ratio < 2.0) and (ratio > 0.5)
        if close:
            up = (ub - ua) / ua
            E[0] = 0.0
            for p in range(1, pmax + 1):
                E[p] = E[p - 1] * ratio + up
    scale = _ipow(rho, 1 - m)
    for k in range(kmax + 1):
        out = 0.0
        for j in range(n_terms):
            pw = k + 2 * j + 1
            if close:
                dpow = YA[pw] * E[pw]
            else:
                dpow = YB[pw] - YA[pw]
            out = out + bm[j] * dpow / pw
        P[mi, k] = scale * out
        scale = scale * rho


@njit(cache=True)
def _closed_form(ua, ub, rho2, ekmax, P):
    """Conjugate closed-form differences for every (m, k <= ekmax[m])."""
    Ra = np.sqrt(ua * ua + rho2)
    Rb = np.sqrt(ub * ub + rho2)
    dR = (ub - ua) * (ub + ua) / (Ra + Rb)
    if ua >= 0.0:
        dJ1 = np.log1p(((ub - ua) + dR) / (ua + Ra))
    elif ub <= 0.0:
        dJ1 = np.log1p(((ub - ua) - dR) / (Rb - ub))
    else:
        dJ1 = np.log(((ub + Rb) * (Ra - ua)) / rho2)
    if ua * ub > 0.0:
        dJ3 = ((ub - ua) * (ub + ua)) / (Ra * Rb * (ub * Ra + ua * Rb))
    else:
        dJ3 = (ub / Rb - ua / Ra) / rho2
    # clq's own operations (pow, not products): dJ5's difference of nearly
    # equal endpoint terms is naive there, so ulp-different inputs would be
    # amplified by it
    duRm = ub * Rb ** (-3.0) - ua * Ra ** (-3.0)
    dJ5 = (duRm + 2.0 * dJ3) / (3.0 * rho2)
    duR = 0.5 * (ub - ua) * (Ra + Rb) + 0.5 * (ua + ub) * dR
    dJm1 = 0.5 * (duR + rho2 * dJ1)
    RaRb = Ra * Rb
    S3 = (Rb * Rb + Ra * Rb) + Ra * Ra
    dK = (dR * S3 / 3.0, dR, dR / RaRb, dR * S3 / (3.0 * RaRb ** 3.0))
    dJ = (dJm1, dJ1, dJ3, dJ5)
    for mi in range(4):
        kmax = ekmax[mi]
        if kmax < 0:
            continue
        P[mi, 0] = dJ[mi]
        if kmax >= 1:
            P[mi, 1] = dK[mi]
    for k in range(2, _MAX_EDGE_DEG + 1):
        for mi in range(1, 4):
            if ekmax[mi] >= k:
                P[mi, k] = P[mi - 1, k - 2] - rho2 * P[mi, k - 2]


@njit(cache=True)
def _edge_primitives(ua, ub, rho2, ekmax, P, binom, small_u, series_u, SER):
    rho = np.sqrt(rho2)
    aua = abs(ua)
    aub = abs(ub)
    umax = max(aua, aub)
    umin = min(aua, aub)
    n_terms = binom.shape[1]
    if umax <= small_u * rho:
        for mi in range(4):
            if ekmax[mi] >= 0:
                _series_small_row(2 * mi - 1, ekmax[mi], ua, ub, rho2, binom[mi],
                                  n_terms, P, mi, SER)
    elif (ua * ub > 0.0) and umin >= series_u * rho:
        negative = ua < 0.0
        a = -ub if negative else ua
        b = -ua if negative else ub
        for mi in range(4):
            if ekmax[mi] >= 0:
                _series_large_row(2 * mi - 1, ekmax[mi], a, b, rho2, binom[mi],
                                  n_terms, P, mi, SER)
                if negative:
                    for k in range(1, ekmax[mi] + 1, 2):
                        P[mi, k] = -P[mi, k]
    else:
        _closed_form(ua, ub, rho2, ekmax, P)


# ---------------------------------------------------------------------
# Per-source frame record
# ---------------------------------------------------------------------

@njit(cache=True)
def _nodal_frame(tv, geo):
    """geo: ex(3) ey(3) nhat(3) centroid(3) p2d(3x2) abc(3x3) L area2; False if degenerate."""
    ex = geo[0:3]
    ey = geo[3:6]
    nhat = geo[6:9]
    if not _tri_frame(tv, ex, ey, nhat):
        return False
    for c in range(3):
        geo[9 + c] = ((tv[0, c] + tv[1, c]) + tv[2, c]) / 3.0
    for k in range(3):
        dx = tv[k, 0] - geo[9]; dy = tv[k, 1] - geo[10]; dz = tv[k, 2] - geo[11]
        geo[12 + 2 * k] = dx * ex[0] + dy * ex[1] + dz * ex[2]
        geo[13 + 2 * k] = dx * ey[0] + dy * ey[1] + dz * ey[2]
    L = 0.0
    for k in range(3):
        k2 = (k + 1) % 3
        dx = tv[k2, 0] - tv[k, 0]; dy = tv[k2, 1] - tv[k, 1]; dz = tv[k2, 2] - tv[k, 2]
        L = max(L, np.sqrt(dx * dx + dy * dy + dz * dz))
    geo[27] = L
    e1x = tv[1, 0] - tv[0, 0]; e1y = tv[1, 1] - tv[0, 1]; e1z = tv[1, 2] - tv[0, 2]
    e2x = tv[2, 0] - tv[0, 0]; e2y = tv[2, 1] - tv[0, 1]; e2z = tv[2, 2] - tv[0, 2]
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    geo[28] = np.sqrt(nx * nx + ny * ny + nz * nz)
    # barycentric coordinates are affine in the centroid frame:
    # lam_k(eta) = A_k + B_k eta_1 + C_k eta_2 (Cramer on [1; p_x; p_y])
    den = ((geo[14] - geo[12]) * (geo[17] - geo[13])
           - (geo[15] - geo[13]) * (geo[16] - geo[12]))
    for k in range(3):
        k1 = (k + 1) % 3
        k2 = (k + 2) % 3
        x1 = geo[12 + 2 * k1]; y1 = geo[13 + 2 * k1]
        x2 = geo[12 + 2 * k2]; y2 = geo[13 + 2 * k2]
        geo[18 + 3 * k] = (x1 * y2 - y1 * x2) / den
        geo[19 + 3 * k] = (y1 - y2) / den
        geo[20 + 3 * k] = (x2 - x1) / den
    return True


@njit(cache=True, parallel=True)
def _nodal_frames(tri_verts):
    N_s = tri_verts.shape[0]
    GEO = np.zeros((N_s, _GEO_LEN))
    OK = np.zeros(N_s, dtype=np.bool_)
    for s in prange(N_s):
        OK[s] = _nodal_frame(tri_verts[s], GEO[s])
    return GEO, OK


# ---------------------------------------------------------------------
# Shape coefficients, raw moment table, weighted tables (near path)
# ---------------------------------------------------------------------

@njit(cache=True)
def _shape_coeffs(order, lattice, geo, X1, X2, C, PQ):
    """C[k, a, b]: coefficients of N_k(xi) = sum c xi_1^a xi_2^b about the foot X."""
    K = C.shape[0]
    D = order + 1
    for k in range(K):
        for a in range(3):
            for b in range(3):
                C[k, a, b] = 0.0
    if order == 0:
        C[0, 0, 0] = 1.0
        return
    poly = PQ[0]
    fac = PQ[1]
    out = PQ[2]
    for k in range(K):
        for a in range(D):
            for b in range(D):
                poly[a, b] = 0.0
                fac[a, b] = 0.0
        poly[0, 0] = 1.0
        for i in range(3):
            l0 = (geo[18 + 3 * i] + X1 * geo[19 + 3 * i]) + X2 * geo[20 + 3 * i]
            for j in range(lattice[k, i]):
                fac[0, 0] = (order * l0) / (j + 1) - j / (j + 1)
                fac[1, 0] = (order * geo[19 + 3 * i]) / (j + 1)
                fac[0, 1] = (order * geo[20 + 3 * i]) / (j + 1)
                for a in range(D):
                    for b in range(D):
                        out[a, b] = 0.0
                for a in range(D):
                    for b in range(D - a):
                        pab = poly[a, b]
                        if pab == 0.0:
                            continue
                        for c in range(D - a):
                            for d in range(D - a - b - c):
                                out[a + c, b + d] += pab * fac[c, d]
                for a in range(D):
                    for b in range(D):
                        poly[a, b] = out[a, b]
        for a in range(D):
            for b in range(D):
                C[k, a, b] = poly[a, b]


@njit(cache=True)
def _moment_table(tv, geo, X1, X2, h2, ispec, fspec, binom, M, BN1, BN2, BD, P,
                  PW, SER):
    """M[ni, a, b] = int_T xi_1^a xi_2^b / R^n dS, rows n = -1, 1, 3, 5, 7."""
    small_u = fspec[2]
    series_u = fspec[3]
    ekmax = ispec[8:12]
    kall = 0
    for mi in range(4):
        kall = max(kall, ekmax[mi])
    h = np.sqrt(h2)
    ox = ((geo[9] + X1 * geo[0]) + X2 * geo[3]) + h * geo[6]
    oy = ((geo[10] + X1 * geo[1]) + X2 * geo[4]) + h * geo[7]
    oz = ((geo[11] + X1 * geo[2]) + X2 * geo[5]) + h * geo[8]
    I3 = -_solid_angle(tv[0], tv[1], tv[2], ox, oy, oz) / h

    for mi in range(4):
        BD[mi] = 0.0
        for a in range(_MAX_EDGE_DEG + 1):
            for b in range(_MAX_EDGE_DEG + 1):
                BN1[mi, a, b] = 0.0
                BN2[mi, a, b] = 0.0
    dpP = PW[0]
    c1P = PW[1]
    s1P = PW[2]
    c2P = PW[3]
    s2P = PW[4]
    dpP[0] = 1.0; c1P[0] = 1.0; s1P[0] = 1.0; c2P[0] = 1.0; s2P[0] = 1.0
    for e in range(3):
        e2 = (e + 1) % 3
        pax = geo[12 + 2 * e] - X1
        pay = geo[13 + 2 * e] - X2
        pbx = geo[12 + 2 * e2] - X1
        pby = geo[13 + 2 * e2] - X2
        evx = pbx - pax
        evy = pby - pay
        Le = np.sqrt(evx * evx + evy * evy)
        tx = evx / Le
        ty = evy / Le
        c1 = ty
        c2 = -tx
        dp = pax * c1 + pay * c2
        ua = pax * tx + pay * ty
        ub = pbx * tx + pby * ty
        rho2 = dp * dp + h2
        _edge_primitives(ua, ub, rho2, ekmax, P, binom, small_u, series_u, SER)
        for q in range(1, kall + 1):
            dpP[q] = dpP[q - 1] * dp
            c1P[q] = c1P[q - 1] * c1
            s1P[q] = s1P[q - 1] * tx
            c2P[q] = c2P[q - 1] * c2
            s2P[q] = s2P[q - 1] * ty
        for mi in range(4):
            kmax = ekmax[mi]
            if kmax < 0:
                continue
            BD[mi] = BD[mi] + dp * P[mi, 0]
            for a in range(kmax + 1):
                for b in range(kmax + 1 - a):
                    val = 0.0
                    for i in range(a + 1):
                        for j in range(b + 1):
                            pw = a + b - i - j
                            coeff = (_COMB[a, i] * _COMB[b, j]) * dpP[pw]
                            coeff = coeff * c1P[a - i]
                            coeff = coeff * s1P[i]
                            coeff = coeff * c2P[b - j]
                            coeff = coeff * s2P[j]
                            val = val + coeff * P[mi, i + j]
                    BN1[mi, a, b] = BN1[mi, a, b] + c1 * val
                    BN2[mi, a, b] = BN2[mi, a, b] + c2 * val

    # seeds: solid angle, vertical identity up (I5, I7) and down (I1, I_-1)
    degn = ispec[3:8]
    M[2, 0, 0] = I3
    if degn[3] >= 0:
        M[3, 0, 0] = (BD[2] + I3) / (3.0 * h2)
    if degn[4] >= 0:
        M[4, 0, 0] = (BD[3] + 3.0 * M[3, 0, 0]) / (5.0 * h2)
    if degn[1] >= 0:
        M[1, 0, 0] = (BD[1] - (1.0 * h2) * M[2, 0, 0]) / 1.0
    if degn[0] >= 0:
        M[0, 0, 0] = (BD[0] - (-1.0 * h2) * M[1, 0, 0]) / 3.0
    # horizontal recursion, ascending n
    for ni in range(5):
        d = degn[ni]
        if d < 1:
            continue
        n = 2 * ni - 1
        mi = ni - 1
        for q in range(1, d + 1):
            for a in range(q, -1, -1):
                b = q - a
                if a >= 1:
                    val = -BN1[mi, a - 1, b]
                    if a >= 2:
                        val = val + (a - 1) * M[mi, a - 2, b]
                else:
                    val = -BN2[mi, 0, b - 1]
                    if b >= 2:
                        val = val + (b - 1) * M[mi, 0, b - 2]
                M[ni, a, b] = val / (n - 2)


@njit(cache=True)
def _weighted_tables(order, K, need0, C, M, W):
    """W[ni, k, a, b] = sum_{a',b'} C[k, a', b'] M[n][a + a', b + b'], n = 2 ni + 1."""
    for ni2 in range(4):
        d = need0[ni2]
        if d < 0:
            continue
        ni = ni2 + 1
        for k in range(K):
            for a in range(d + 1):
                for b in range(d + 1 - a):
                    acc = 0.0
                    for ap in range(order + 1):
                        for bp in range(order + 1 - ap):
                            acc += C[k, ap, bp] * M[ni, a + ap, b + bp]
                    W[ni2, k, a, b] = acc


@njit(cache=True)
def _far_tables(geo, X1, X2, h2, K, need0, q_lam, q_w, q_N, q0, q1, W, PW):
    """Far-field producer: Gauss rule [q0, q1) of N_k xi_1^a xi_2^b / R^n."""
    area2 = geo[28]
    dmax = 0
    for ni2 in range(4):
        d = need0[ni2]
        dmax = max(dmax, d)
        if d < 0:
            continue
        for k in range(K):
            for a in range(d + 1):
                for b in range(d + 1 - a):
                    W[ni2, k, a, b] = 0.0
    x1p = PW[5]
    x2p = PW[6]
    x1p[0] = 1.0
    x2p[0] = 1.0
    for q in range(q0, q1):
        l1 = q_lam[q, 0]; l2 = q_lam[q, 1]; l3 = q_lam[q, 2]
        eta1 = l1 * geo[12] + l2 * geo[14] + l3 * geo[16]
        eta2 = l1 * geo[13] + l2 * geo[15] + l3 * geo[17]
        xi1 = eta1 - X1
        xi2 = eta2 - X2
        R2 = xi1 * xi1 + xi2 * xi2 + h2
        wq = q_w[q] * area2
        r2i = 1.0 / R2
        Rn = np.sqrt(r2i)                      # R^-1; R^-n follows by r2i steps
        for a in range(1, dmax + 1):
            x1p[a] = x1p[a - 1] * xi1
            x2p[a] = x2p[a - 1] * xi2
        for ni2 in range(4):
            d = need0[ni2]
            if d >= 0:
                for a in range(d + 1):
                    for b in range(d + 1 - a):
                        f = wq * x1p[a] * x2p[b] * Rn
                        for k in range(K):
                            W[ni2, k, a, b] += f * q_N[q, k]
            Rn = Rn * r2i


@njit(cache=True)
def _pair_tables(tv, geo, obs, eps, spec, W, M, BN1, BN2, BD, P, C, PQ, PW, SER):
    """Fill W (4, K, 5, 5) for one pair; returns the signed height z."""
    ispec, fspec, lattice, binom, q_lam, q_w, q_N = spec
    order = ispec[0]
    K = ispec[1]
    dx = obs[0] - geo[9]; dy = obs[1] - geo[10]; dz = obs[2] - geo[11]
    z = dx * geo[6] + dy * geo[7] + dz * geo[8]
    X1 = dx * geo[0] + dy * geo[1] + dz * geo[2]
    X2 = dx * geo[3] + dy * geo[4] + dz * geo[5]
    h2 = z * z + eps * eps
    D = np.sqrt((dx * dx + dy * dy + dz * dz) + eps * eps)
    L = geo[27]
    need0 = ispec[12:16]
    if D <= fspec[0] * L:
        _shape_coeffs(order, lattice, geo, X1, X2, C, PQ)
        _moment_table(tv, geo, X1, X2, h2, ispec, fspec, binom, M, BN1, BN2, BD, P,
                      PW, SER)
        _weighted_tables(order, K, need0, C, M, W)
    else:
        q_split = ispec[2]
        if D > fspec[1] * L:
            _far_tables(geo, X1, X2, h2, K, need0, q_lam, q_w, q_N,
                        q_split, q_lam.shape[0], W, PW)
        else:
            _far_tables(geo, X1, X2, h2, K, need0, q_lam, q_w, q_N, 0, q_split, W, PW)
    return z


@njit(cache=True)
def _scratch(K, n_terms):
    """Per-thread scratch of the pair machinery (allocated once per prange body)."""
    W = np.zeros((4, K, 5, 5))
    M = np.zeros((5, _MAX_TABLE_DEG + 1, _MAX_TABLE_DEG + 1))
    BN1 = np.zeros((4, _MAX_EDGE_DEG + 1, _MAX_EDGE_DEG + 1))
    BN2 = np.zeros((4, _MAX_EDGE_DEG + 1, _MAX_EDGE_DEG + 1))
    BD = np.zeros(4)
    P = np.zeros((4, _MAX_EDGE_DEG + 1))
    C = np.zeros((K, 3, 3))
    PQ = np.zeros((3, 3, 3))
    PW = np.zeros((7, _MAX_EDGE_DEG + 1))
    SER = np.zeros((4, _series_len(n_terms)))
    return W, M, BN1, BN2, BD, P, C, PQ, PW, SER


# ---------------------------------------------------------------------
# Per-node kernel contractions (the P0 formulas of tri_kernels on W_k)
# ---------------------------------------------------------------------

@njit(cache=True)
def _t_blocks_node(W, k, z, bas, nsrc, eps, out, T35, Pt):
    """6 T-kernel basis blocks [N[P1], N[P2], N[P3], R[P1], R[P2], R[P3]] of node k."""
    W3 = W[1, k]
    W5 = W[2, k]
    ex = bas[0]; ey = bas[1]; nhat = bas[2]
    V3 = (-ex[0] * W3[1, 0] - ey[0] * W3[0, 1] + z * nhat[0] * W3[0, 0],
          -ex[1] * W3[1, 0] - ey[1] * W3[0, 1] + z * nhat[1] * W3[0, 0],
          -ex[2] * W3[1, 0] - ey[2] * W3[0, 1] + z * nhat[2] * W3[0, 0])
    V5 = (-ex[0] * W5[1, 0] - ey[0] * W5[0, 1] + z * nhat[0] * W5[0, 0],
          -ex[1] * W5[1, 0] - ey[1] * W5[0, 1] + z * nhat[1] * W5[0, 0],
          -ex[2] * W5[1, 0] - ey[2] * W5[0, 1] + z * nhat[2] * W5[0, 0])
    _build_T3(W5, bas, z, T35)
    e2 = eps * eps
    for i in range(3):
        for kk in range(3):
            for m in range(3):
                Pt[i, kk, m] = V3[m] if i == kk else 0.0
    _contract_NR(Pt, nsrc, out, 0, 3)
    for i in range(3):
        for kk in range(3):
            for m in range(3):
                val = -3.0 * T35[i, kk, m]
                if i == m:
                    val += V3[kk]
                if kk == m:
                    val += V3[i]
                Pt[i, kk, m] = val
    _contract_NR(Pt, nsrc, out, 1, 4)
    for i in range(3):
        for kk in range(3):
            for m in range(3):
                Pt[i, kk, m] = e2 * V5[m] if i == kk else 0.0
    _contract_NR(Pt, nsrc, out, 2, 5)


@njit(cache=True)
def _u_blocks_node(W, k, z, bas, eps, out):
    """3 U-kernel basis blocks [I1 d, eps^2 I3 d, T2_3] of node k."""
    I1 = W[0, k, 0, 0]
    I3 = W[1, k, 0, 0]
    for i in range(3):
        for j in range(3):
            out[0, i, j] = 0.0
            out[1, i, j] = 0.0
        out[0, i, i] = I1
        out[1, i, i] = eps * eps * I3
    _build_T2(W[1, k], bas[0], bas[1], bas[2], z, out[2])


@njit(cache=True)
def _dd_stress_node(W, k, z, bas, nrm, eps, mu, nu, lam, Hout, T2_5, T2_7, T4_7,
                    ID2G, B):
    """Slip -> TOTAL stress H[m,n,j] (sigma_mn per unit slip_j) of node k."""
    I3 = W[1, k, 0, 0]
    I5 = W[2, k, 0, 0]
    _build_T2(W[2, k], bas[0], bas[1], bas[2], z, T2_5)
    _build_T2(W[3, k], bas[0], bas[1], bas[2], z, T2_7)
    _build_T4(W[3, k], bas, z, T4_7)
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cblob = 2.0 * (1.0 - nu) * eps * eps
    for r in range(3):
        for p in range(3):
            drp = 1.0 if r == p else 0.0
            for s in range(3):
                drs = 1.0 if r == s else 0.0
                dps = 1.0 if p == s else 0.0
                for q in range(3):
                    dsq = 1.0 if s == q else 0.0
                    dpq = 1.0 if p == q else 0.0
                    drq = 1.0 if r == q else 0.0
                    val = (
                        -c34 * drp * dsq * I3
                        + drs * dpq * I3
                        + dps * drq * I3
                        + c34 * drp * 3.0 * T2_5[s, q]
                        - drs * 3.0 * T2_5[p, q]
                        - dps * 3.0 * T2_5[r, q]
                        - 3.0 * drq * T2_5[p, s]
                        - 3.0 * dpq * T2_5[r, s]
                        - 3.0 * dsq * T2_5[r, p]
                        + 15.0 * T4_7[r, p, s, q]
                        + cblob * drp * (-3.0 * dsq * I5 + 15.0 * T2_7[s, q])
                    )
                    ID2G[r, p, s, q] = C1 * val
    for r in range(3):
        for s in range(3):
            tr = 0.0
            for p in range(3):
                tr += ID2G[r, p, s, p]
            for kk in range(3):
                nq = 0.0
                npd = 0.0
                for q in range(3):
                    nq += nrm[q] * ID2G[r, kk, s, q]
                for p in range(3):
                    npd += nrm[p] * ID2G[r, p, s, kk]
                B[r, s, kk] = lam * nrm[kk] * tr + mu * nq + mu * npd
    for kk in range(3):
        trB = B[0, 0, kk] + B[1, 1, kk] + B[2, 2, kk]
        for m in range(3):
            for n in range(3):
                dmn = 1.0 if m == n else 0.0
                Hout[m, n, kk] = -(lam * dmn * trB
                                   + mu * (B[m, n, kk] + B[n, m, kk]))


@njit(cache=True)
def _kelvin_stress_node(W, k, z, bas, eps, mu, nu, lam, Sout, T35, G1):
    """Force -> stress S[i,j,c] (sigma_ij per unit force_c) of node k."""
    W3 = W[1, k]
    W5 = W[2, k]
    ex = bas[0]; ey = bas[1]; nhat = bas[2]
    V3 = (-ex[0] * W3[1, 0] - ey[0] * W3[0, 1] + z * nhat[0] * W3[0, 0],
          -ex[1] * W3[1, 0] - ey[1] * W3[0, 1] + z * nhat[1] * W3[0, 0],
          -ex[2] * W3[1, 0] - ey[2] * W3[0, 1] + z * nhat[2] * W3[0, 0])
    V5 = (-ex[0] * W5[1, 0] - ey[0] * W5[0, 1] + z * nhat[0] * W5[0, 0],
          -ex[1] * W5[1, 0] - ey[1] * W5[0, 1] + z * nhat[1] * W5[0, 0],
          -ex[2] * W5[1, 0] - ey[2] * W5[0, 1] + z * nhat[2] * W5[0, 0])
    _build_T3(W5, bas, z, T35)
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cbdg = 6.0 * (1.0 - nu) * eps * eps
    for i in range(3):
        for j in range(3):
            dij = 1.0 if i == j else 0.0
            for m in range(3):
                dim = 1.0 if i == m else 0.0
                djm = 1.0 if j == m else 0.0
                G1[i, j, m] = C1 * (
                    -c34 * dij * V3[m]
                    + dim * V3[j]
                    + djm * V3[i]
                    - 3.0 * T35[i, j, m]
                    - cbdg * dij * V5[m]
                )
    for c in range(3):
        trace_c = G1[0, c, 0] + G1[1, c, 1] + G1[2, c, 2]
        for i in range(3):
            for j in range(3):
                dij = 1.0 if i == j else 0.0
                Sout[i, j, c] = (lam * dij * trace_c
                                 + mu * G1[i, c, j] + mu * G1[j, c, i])


@njit(cache=True)
def _eigen_node(W, k, nrm, eps, mu, lam, Hout):
    """Eigenstress H*[m,n,j] = (15 eps^4 / 8 pi) W_k[7][0,0] C:sym(e_j nrm)."""
    phi = (15.0 * eps * eps * eps * eps / (8.0 * np.pi)) * W[3, k, 0, 0]
    for m in range(3):
        for n in range(3):
            dmn = 1.0 if m == n else 0.0
            for j in range(3):
                dmj = 1.0 if m == j else 0.0
                dnj = 1.0 if n == j else 0.0
                Hout[m, n, j] = phi * (lam * dmn * nrm[j]
                                       + mu * (dmj * nrm[n] + dnj * nrm[m]))


# ---------------------------------------------------------------------
# Drivers, general order (source-parallel matrices, field-parallel contractions)
# ---------------------------------------------------------------------

@njit(cache=True)
def _t_basis_source(s, x_field, tv, geo, nsrc, eps, spec, out, coeffs, direct):
    """Column block of source s: basis stack (direct=False) or c . stack."""
    ispec = spec[0]
    K = ispec[1]
    N_f = x_field.shape[0]
    W, M, BN1, BN2, BD, P, C, PQ, PW, SER = _scratch(K, spec[3].shape[1])
    blk = np.empty((6, 3, 3))
    T35 = np.empty((3, 3, 3))
    Pt = np.empty((3, 3, 3))
    bas = np.empty((3, 3))
    for r in range(3):
        for c in range(3):
            bas[r, c] = geo[3 * r + c]
    for f in range(N_f):
        z = _pair_tables(tv, geo, x_field[f], eps, spec, W, M, BN1, BN2, BD, P, C, PQ, PW, SER)
        for k in range(K):
            _t_blocks_node(W, k, z, bas, nsrc, eps, blk, T35, Pt)
            col = 3 * (K * s + k)
            if direct:
                for i in range(3):
                    for j in range(3):
                        acc = 0.0
                        for b in range(6):
                            acc += coeffs[b] * blk[b, i, j]
                        out[0, 3 * f + i, col + j] = acc
            else:
                for b in range(6):
                    for i in range(3):
                        for j in range(3):
                            out[b, 3 * f + i, col + j] = blk[b, i, j]


@njit(cache=True)
def _u_basis_source(s, x_field, tv, geo, eps, spec, out, coeffs, direct):
    ispec = spec[0]
    K = ispec[1]
    N_f = x_field.shape[0]
    W, M, BN1, BN2, BD, P, C, PQ, PW, SER = _scratch(K, spec[3].shape[1])
    blk = np.empty((3, 3, 3))
    bas = np.empty((3, 3))
    for r in range(3):
        for c in range(3):
            bas[r, c] = geo[3 * r + c]
    for f in range(N_f):
        z = _pair_tables(tv, geo, x_field[f], eps, spec, W, M, BN1, BN2, BD, P, C, PQ, PW, SER)
        for k in range(K):
            _u_blocks_node(W, k, z, bas, eps, blk)
            col = 3 * (K * s + k)
            if direct:
                for i in range(3):
                    for j in range(3):
                        out[0, 3 * f + i, col + j] = (coeffs[0] * blk[0, i, j]
                                                      + coeffs[1] * blk[1, i, j]
                                                      + coeffs[2] * blk[2, i, j])
            else:
                for b in range(3):
                    for i in range(3):
                        for j in range(3):
                            out[b, 3 * f + i, col + j] = blk[b, i, j]


@njit(cache=True, parallel=True)
def _t_basis_nodal(x_field, tri_verts, normals, eps_arr, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((6, 3 * N_f, 3 * K * N_s))
    coeffs = np.zeros(6)
    for s in prange(N_s):
        geo = np.empty(_GEO_LEN)
        if _nodal_frame(tri_verts[s], geo):
            _t_basis_source(s, x_field, tri_verts[s], geo, normals[s], eps_arr[s],
                            spec, out, coeffs, False)
    return out


@njit(cache=True, nogil=True)
def _t_basis_nodal_serial(x_field, tri_verts, normals, eps_arr, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((6, 3 * N_f, 3 * K * N_s))
    coeffs = np.zeros(6)
    geo = np.empty(_GEO_LEN)
    for s in range(N_s):
        if _nodal_frame(tri_verts[s], geo):
            _t_basis_source(s, x_field, tri_verts[s], geo, normals[s], eps_arr[s],
                            spec, out, coeffs, False)
    return out


@njit(cache=True, parallel=True)
def _u_basis_nodal(x_field, tri_verts, eps_arr, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3, 3 * N_f, 3 * K * N_s))
    coeffs = np.zeros(3)
    for s in prange(N_s):
        geo = np.empty(_GEO_LEN)
        if _nodal_frame(tri_verts[s], geo):
            _u_basis_source(s, x_field, tri_verts[s], geo, eps_arr[s], spec, out,
                            coeffs, False)
    return out


@njit(cache=True, nogil=True)
def _u_basis_nodal_serial(x_field, tri_verts, eps_arr, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3, 3 * N_f, 3 * K * N_s))
    coeffs = np.zeros(3)
    geo = np.empty(_GEO_LEN)
    for s in range(N_s):
        if _nodal_frame(tri_verts[s], geo):
            _u_basis_source(s, x_field, tri_verts[s], geo, eps_arr[s], spec, out,
                            coeffs, False)
    return out


@njit(cache=True, parallel=True)
def _t_direct_nodal(x_field, tri_verts, normals, eps_arr, coeffs, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((1, 3 * N_f, 3 * K * N_s))
    for s in prange(N_s):
        geo = np.empty(_GEO_LEN)
        if _nodal_frame(tri_verts[s], geo):
            _t_basis_source(s, x_field, tri_verts[s], geo, normals[s], eps_arr[s],
                            spec, out, coeffs, True)
    return out[0]


@njit(cache=True, parallel=True)
def _u_direct_nodal(x_field, tri_verts, eps_arr, coeffs, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((1, 3 * N_f, 3 * K * N_s))
    for s in prange(N_s):
        geo = np.empty(_GEO_LEN)
        if _nodal_frame(tri_verts[s], geo):
            _u_basis_source(s, x_field, tri_verts[s], geo, eps_arr[s], spec, out,
                            coeffs, True)
    return out[0]


@njit(cache=True, inline="always")
def _block_is_zero(density, K, s):
    for k in range(K):
        for j in range(3):
            if density[K * s + k, j] != 0.0:
                return False
    return True


@njit(cache=True, parallel=True)
def _t_contract_nodal(x_field, tri_verts, normals, eps_arr, density, coeffs, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    GEO, OK = _nodal_frames(tri_verts)
    out = np.zeros((N_f, 3))
    for f in prange(N_f):
        W, M, BN1, BN2, BD, P, C, PQ, PW, SER = _scratch(K, spec[3].shape[1])
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3))
        Pt = np.empty((3, 3, 3))
        obs = x_field[f]
        a0 = 0.0; a1 = 0.0; a2 = 0.0
        for s in range(N_s):
            if not OK[s] or _block_is_zero(density, K, s):
                continue
            geo = GEO[s]
            bas = geo[0:9].reshape((3, 3))
            z = _pair_tables(tri_verts[s], geo, obs, eps_arr[s], spec,
                             W, M, BN1, BN2, BD, P, C, PQ, PW, SER)
            for k in range(K):
                _t_blocks_node(W, k, z, bas, normals[s], eps_arr[s], blk, T35, Pt)
                for j in range(3):
                    dj = density[K * s + k, j]
                    if dj == 0.0:
                        continue
                    for i in range(3):
                        v = 0.0
                        for b in range(6):
                            v += coeffs[b] * blk[b, i, j]
                        if i == 0:
                            a0 += v * dj
                        elif i == 1:
                            a1 += v * dj
                        else:
                            a2 += v * dj
        out[f, 0] = a0; out[f, 1] = a1; out[f, 2] = a2
    return out


@njit(cache=True, parallel=True)
def _u_contract_nodal(x_field, tri_verts, eps_arr, density, coeffs, spec):
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    GEO, OK = _nodal_frames(tri_verts)
    out = np.zeros((N_f, 3))
    for f in prange(N_f):
        W, M, BN1, BN2, BD, P, C, PQ, PW, SER = _scratch(K, spec[3].shape[1])
        blk = np.empty((3, 3, 3))
        obs = x_field[f]
        a0 = 0.0; a1 = 0.0; a2 = 0.0
        for s in range(N_s):
            if not OK[s] or _block_is_zero(density, K, s):
                continue
            geo = GEO[s]
            bas = geo[0:9].reshape((3, 3))
            z = _pair_tables(tri_verts[s], geo, obs, eps_arr[s], spec,
                             W, M, BN1, BN2, BD, P, C, PQ, PW, SER)
            for k in range(K):
                _u_blocks_node(W, k, z, bas, eps_arr[s], blk)
                for j in range(3):
                    dj = density[K * s + k, j]
                    if dj == 0.0:
                        continue
                    a0 += (coeffs[0] * blk[0, 0, j] + coeffs[1] * blk[1, 0, j]
                           + coeffs[2] * blk[2, 0, j]) * dj
                    a1 += (coeffs[0] * blk[0, 1, j] + coeffs[1] * blk[1, 1, j]
                           + coeffs[2] * blk[2, 1, j]) * dj
                    a2 += (coeffs[0] * blk[0, 2, j] + coeffs[1] * blk[1, 2, j]
                           + coeffs[2] * blk[2, 2, j]) * dj
        out[f, 0] = a0; out[f, 1] = a1; out[f, 2] = a2
    return out


@njit(cache=True, parallel=True)
def _stress_contract_nodal(x_field, tri_verts, normals, eps_arr, density, mu, nu, lam,
                           spec, which):
    """which: 0 = slip -> total stress, 1 = force -> stress, 2 = eigenstress.
    (mu, nu, lam) as the pair kernels take them; the public drivers derive nu."""
    K = spec[0][1]
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    GEO, OK = _nodal_frames(tri_verts)
    sig = np.zeros((N_f, 3, 3))
    for f in prange(N_f):
        W, M, BN1, BN2, BD, P, C, PQ, PW, SER = _scratch(K, spec[3].shape[1])
        H = np.empty((3, 3, 3))
        T2_5 = np.empty((3, 3)); T2_7 = np.empty((3, 3))
        T4_7 = np.empty((3, 3, 3, 3)); ID2G = np.empty((3, 3, 3, 3))
        B = np.empty((3, 3, 3)); T35 = np.empty((3, 3, 3)); G1 = np.empty((3, 3, 3))
        acc = np.zeros((3, 3))
        obs = x_field[f]
        for s in range(N_s):
            if not OK[s] or _block_is_zero(density, K, s):
                continue
            geo = GEO[s]
            bas = geo[0:9].reshape((3, 3))
            eps = eps_arr[s]
            z = _pair_tables(tri_verts[s], geo, obs, eps, spec,
                             W, M, BN1, BN2, BD, P, C, PQ, PW, SER)
            for k in range(K):
                if which == 0:
                    _dd_stress_node(W, k, z, bas, normals[s], eps, mu, nu, lam, H,
                                    T2_5, T2_7, T4_7, ID2G, B)
                elif which == 1:
                    _kelvin_stress_node(W, k, z, bas, eps, mu, nu, lam, H, T35, G1)
                else:
                    _eigen_node(W, k, normals[s], eps, mu, lam, H)
                row = K * s + k
                d0 = density[row, 0]; d1 = density[row, 1]; d2 = density[row, 2]
                for m in range(3):
                    for n in range(3):
                        acc[m, n] += H[m, n, 0] * d0 + H[m, n, 1] * d1 + H[m, n, 2] * d2
        for m in range(3):
            for n in range(3):
                sig[f, m, n] = acc[m, n]
    return sig


# ---------------------------------------------------------------------
# Public drivers: tri_kernels' contracts plus ``order``; order 0 stays on
# tri_kernels' pair code.
# ---------------------------------------------------------------------

def _as_coeffs(c):
    return np.ascontiguousarray(np.asarray(c, dtype=float))


def _check_shape(tri_verts):
    """Refuse needles for a nodal density: the P1/P2 closed form loses
    (L / height)^2 digits, so height / L below NODAL_MIN_HEIGHT_OVER_L
    (height = 2 area / L, the smallest altitude) raises instead of returning
    inaccurate columns."""
    tv = np.asarray(tri_verts, float)
    e = np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 1], tv[:, 0] - tv[:, 2]])
    L = np.linalg.norm(e, axis=2).max(axis=0)
    area2 = np.linalg.norm(np.cross(e[0], -e[2]), axis=1)
    ratio = area2 / np.where(L > 0.0, L * L, 1.0)
    bad = ratio < defaults.NODAL_MIN_HEIGHT_OVER_L
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} source triangle(s) have height / L < "
            f"{defaults.NODAL_MIN_HEIGHT_OVER_L:g} (min {ratio.min():.2e}): "
            "a nodal (order >= 1) density is not accurate on needles")


def u_basis_matrices(x_field, tri_verts, eps_arr, order):
    """U-kernel basis stack (3, 3 N_f, 3 K N_s)."""
    if n_nodes(order) == 1:
        return tk.u_basis_matrices(x_field, tri_verts, eps_arr)
    _check_shape(tri_verts)
    return _u_basis_nodal(x_field, tri_verts, eps_arr, _spec(order, KU))


def t_basis_matrices(x_field, tri_verts, normals, eps_arr, order):
    """T-kernel basis stack (6, 3 N_f, 3 K N_s)."""
    if n_nodes(order) == 1:
        return tk.t_basis_matrices(x_field, tri_verts, normals, eps_arr)
    _check_shape(tri_verts)
    return _t_basis_nodal(x_field, tri_verts, normals, eps_arr, _spec(order, KT))


def u_basis_matrices_serial(x_field, tri_verts, eps_arr, order):
    """Serial/nogil u_basis_matrices (for Python worker threads)."""
    if n_nodes(order) == 1:
        return tk.u_basis_matrices_serial(x_field, tri_verts, eps_arr)
    _check_shape(tri_verts)
    return _u_basis_nodal_serial(x_field, tri_verts, eps_arr, _spec(order, KU))


def t_basis_matrices_serial(x_field, tri_verts, normals, eps_arr, order):
    """Serial/nogil t_basis_matrices (for Python worker threads)."""
    if n_nodes(order) == 1:
        return tk.t_basis_matrices_serial(x_field, tri_verts, normals, eps_arr)
    _check_shape(tri_verts)
    return _t_basis_nodal_serial(x_field, tri_verts, normals, eps_arr, _spec(order, KT))


def u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3, order):
    """U-kernel matrix (3 N_f, 3 K N_s) with u_coeffs applied in-loop."""
    if n_nodes(order) == 1:
        return tk.u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3)
    _check_shape(tri_verts)
    return _u_direct_nodal(x_field, tri_verts, eps_arr, _as_coeffs((g1, g2, g3)),
                           _spec(order, KU))


def t_matrix_direct(x_field, tri_verts, normals, eps_arr, c1, c2, c3, c4, c5, c6,
                    order):
    """T-kernel matrix (3 N_f, 3 K N_s) with t_coeffs applied in-loop."""
    if n_nodes(order) == 1:
        return tk.t_matrix_direct(x_field, tri_verts, normals, eps_arr,
                                  c1, c2, c3, c4, c5, c6)
    _check_shape(tri_verts)
    return _t_direct_nodal(x_field, tri_verts, normals, eps_arr,
                           _as_coeffs((c1, c2, c3, c4, c5, c6)), _spec(order, KT))


def u_disp_contract(x_field, tri_verts, eps_arr, density, g1, g2, g3, order):
    """Displacement (N_f, 3) of a nodal FORCE density (K N_s, 3)."""
    if n_nodes(order) == 1:
        return tk.u_disp_contract(x_field, tri_verts, eps_arr, density, g1, g2, g3)
    _check_shape(tri_verts)
    return _u_contract_nodal(x_field, tri_verts, eps_arr, density,
                             _as_coeffs((g1, g2, g3)), _spec(order, KU))


def t_disp_contract(x_field, tri_verts, normals, eps_arr, density,
                    c1, c2, c3, c4, c5, c6, order):
    """Displacement (N_f, 3) of a nodal SLIP density (K N_s, 3)."""
    if n_nodes(order) == 1:
        return tk.t_disp_contract(x_field, tri_verts, normals, eps_arr, density,
                                  c1, c2, c3, c4, c5, c6)
    _check_shape(tri_verts)
    return _t_contract_nodal(x_field, tri_verts, normals, eps_arr, density,
                             _as_coeffs((c1, c2, c3, c4, c5, c6)), _spec(order, KT))


def _nu(mu, lam):
    """nu from (mu, lam): the safe direction (rule 7; kernels/basis.py)."""
    return lam / (2.0 * (lam + mu))


def kelvin_stress_contract(x_field, tri_verts, eps_arr, density, mu, lam, order):
    """Stress (N_f, 3, 3) of a nodal FORCE density. Material as (mu, lam)."""
    if n_nodes(order) == 1:
        return tk.kelvin_stress_contract(x_field, tri_verts, eps_arr, density, mu, lam)
    normals = np.zeros((tri_verts.shape[0], 3))
    _check_shape(tri_verts)
    return _stress_contract_nodal(x_field, tri_verts, normals, eps_arr, density,
                                  mu, _nu(mu, lam), lam, _spec(order, KS), 1)


def dd_stress_contract(x_field, tri_verts, normals, eps_arr, density, mu, lam, order):
    """TOTAL stress (N_f, 3, 3) of a nodal SLIP density (eigenstress included).
    Material as (mu, lam)."""
    if n_nodes(order) == 1:
        return tk.dd_stress_contract(x_field, tri_verts, normals, eps_arr, density,
                                     mu, lam)
    _check_shape(tri_verts)
    return _stress_contract_nodal(x_field, tri_verts, normals, eps_arr, density,
                                  mu, _nu(mu, lam), lam, _spec(order, KH), 0)


def eigenstress_contract(x_field, tri_verts, normals, eps_arr, density, mu, lam, order):
    """Eigenstress +C:eps* (N_f, 3, 3) of a nodal SLIP density. Material as
    (mu, lam)."""
    if n_nodes(order) == 1:
        return tk.eigenstress_contract(x_field, tri_verts, normals, eps_arr, density,
                                       mu, lam)
    _check_shape(tri_verts)
    return _stress_contract_nodal(x_field, tri_verts, normals, eps_arr, density,
                                  mu, _nu(mu, lam), lam, _spec(order, KE), 2)
