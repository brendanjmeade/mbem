"""Numba pair-level analytical integration of mollified Kelvin kernels.

Scalar port of ``mollified_kernel/analytical_batch.py`` restructured to
emit GEOMETRY-ONLY basis blocks (material coefficients are applied later
by :mod:`mbem.kernels.basis`):

U (Kelvin force->displacement) kernel over a source triangle::

    G(mu, nu) = g1*B1 + g2*B2 + g3*B3
    B1_ij = I1 * delta_ij            (I1 = integral of 1/R_eps dA)
    B2_ij = eps^2 * I3 * delta_ij    (I3 = integral of 1/R_eps^3 dA)
    B3_ij = T2[3]_ij                 (integral of d_i d_j / R_eps^3 dA)

T (slip->displacement, DD) kernel::

    U(mu, lam) = c1*N[P1] + c2*N[P2] + c3*N[P3]
               + c4*R[P1] + c5*R[P2] + c6*R[P3]
    P1_ikm = delta_ik V3_m
    P2_ikm = delta_im V3_k + delta_km V3_i - 3*T3[5]_ikm
    P3_ikm = eps^2 * delta_ik V5_m
    N[P]_ij = n_j * sum_m P_imm                      (coefficient ~ lam)
    R[P]_ij = sum_m n_m P_ijm + sum_k n_k P_ikj      (coefficient ~ mu)

i.e. the traction-operator contraction
    U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]
grouped by material constant (slip and normal share C's first index pair).
Before 2026-09-04 the lam slot held sum_m n_m P_ijm and the mu slot
n_j*tr + sum_k n_k P_ikj -- lam/mu swapped on two terms, invisible at
nu = 1/4 (lam == mu) where every gate ran; see verify/verify_dd_pairing.py.

where V3/V5 are first moments of d/R^3, d/R^5 and T3[5] the third moment
of d d d / R^5, all integrated analytically over the source triangle
with the Cortez mollification baked into R_eps = sqrt(r^2 + eps^2).
eps is PER SOURCE ELEMENT (an (N_src,) array at the assembly level).

The numerical guards (1e-300 in the log, 1e-30/1e-60 floors, degenerate
triangle/edge skips) replicate analytical_batch.py exactly so the two
paths agree to machine precision.

Matrix layout matches the legacy ``assemble_BEM_matrices``: row block
3*f..3*f+3 = field element f, column block 3*s..3*s+3 = source element s.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------
# Edge antiderivatives (orders 1 and 3 are all the U/T kernels consume)
# ---------------------------------------------------------------------

@njit(cache=True, inline="always")
def _J1(u, rho2):
    return np.log(u + np.sqrt(u * u + rho2) + 1e-300)


@njit(cache=True, inline="always")
def _J3(u, rho2):
    if rho2 < 1e-60:
        return 0.0
    return u / (rho2 * np.sqrt(u * u + rho2))


@njit(cache=True, inline="always")
def _K1(u, rho2):
    return np.sqrt(u * u + rho2)


@njit(cache=True, inline="always")
def _K3(u, rho2):
    R = np.sqrt(u * u + rho2)
    if R < 1e-30:
        R = 1e-30
    return -1.0 / R


@njit(cache=True, inline="always")
def _solid_angle(v1, v2, v3, ox, oy, oz):
    """van Oosterom signed solid angle of triangle (v1,v2,v3) from obs."""
    r1x = v1[0] - ox; r1y = v1[1] - oy; r1z = v1[2] - oz
    r2x = v2[0] - ox; r2y = v2[1] - oy; r2z = v2[2] - oz
    r3x = v3[0] - ox; r3y = v3[1] - oy; r3z = v3[2] - oz
    R1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    R2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)
    R3 = np.sqrt(r3x * r3x + r3y * r3y + r3z * r3z)
    cx = r2y * r3z - r2z * r3y
    cy = r2z * r3x - r2x * r3z
    cz = r2x * r3y - r2y * r3x
    numer = r1x * cx + r1y * cy + r1z * cz
    d12 = r1x * r2x + r1y * r2y + r1z * r2z
    d23 = r2x * r3x + r2y * r3y + r2z * r3z
    d13 = r1x * r3x + r1y * r3y + r1z * r3z
    denom = R1 * R2 * R3 + R3 * d12 + R1 * d23 + R2 * d13
    return 2.0 * np.arctan2(numer, denom)


@njit(cache=True, inline="always")
def _edge_int(a, b, dp, nx, ny, tx, ty, dJ, dK, dL):
    """Edge integral of xi1^a xi2^b / R^m along one edge.

    Mirrors analytical_batch.edge_integral: the integrand is expanded in
    the edge-local frame xi1 = nx*dp + tx*u, xi2 = ny*dp + ty*u; dJ/dK/dL
    are the order-m antiderivative differences for u^0, u^1, u^2.
    a + b <= 2 here, so binomials are 1 or 2 and u-powers stop at 2.
    """
    total = 0.0
    for i in range(a + 1):
        for j in range(b + 1):
            ca = 1.0
            if a == 2 and i == 1:
                ca = 2.0
            cb = 1.0
            if b == 2 and j == 1:
                cb = 2.0
            coeff = (ca * cb
                     * dp ** (a + b - i - j)
                     * nx ** (a - i) * tx ** i
                     * ny ** (b - j) * ty ** j)
            k = i + j
            if k == 0:
                delta = dJ
            elif k == 1:
                delta = dK
            else:
                delta = dL
            total += coeff * delta
    return total


# ---------------------------------------------------------------------
# Per-source-triangle frame setup
# ---------------------------------------------------------------------

@njit(cache=True)
def _tri_frame(tv, ex, ey, nhat):
    """In-plane basis of source triangle tv (3,3). Returns False if degenerate."""
    e1x = tv[1, 0] - tv[0, 0]; e1y = tv[1, 1] - tv[0, 1]; e1z = tv[1, 2] - tv[0, 2]
    e2x = tv[2, 0] - tv[0, 0]; e2y = tv[2, 1] - tv[0, 1]; e2z = tv[2, 2] - tv[0, 2]
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    area2 = np.sqrt(nx * nx + ny * ny + nz * nz)
    if area2 < 1e-30:
        return False
    nhat[0] = nx / area2; nhat[1] = ny / area2; nhat[2] = nz / area2
    e1n = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    ex[0] = e1x / e1n; ex[1] = e1y / e1n; ex[2] = e1z / e1n
    ey[0] = nhat[1] * ex[2] - nhat[2] * ex[1]
    ey[1] = nhat[2] * ex[0] - nhat[0] * ex[2]
    ey[2] = nhat[0] * ex[1] - nhat[1] * ex[0]
    return True


# ---------------------------------------------------------------------
# U-kernel basis blocks for one (source triangle, obs) pair
# ---------------------------------------------------------------------

@njit(cache=True)
def _u_basis_pair(tv, ex, ey, nhat, obs, eps, out):
    """Write the 3 U-kernel basis 3x3 blocks into out (3,3,3): [b,i,j]."""
    # Per-obs frame quantities
    dx = obs[0] - tv[0, 0]; dy = obs[1] - tv[0, 1]; dz = obs[2] - tv[0, 2]
    z = dx * nhat[0] + dy * nhat[1] + dz * nhat[2]
    h2 = z * z + eps * eps
    h = np.sqrt(h2)
    opx = obs[0] - z * nhat[0]
    opy = obs[1] - z * nhat[1]
    opz = obs[2] - z * nhat[2]

    # Projected triangle vertices in the (ex, ey) plane relative to obs_proj
    px = np.empty(3); py = np.empty(3)
    for k in range(3):
        wx = tv[k, 0] - opx; wy = tv[k, 1] - opy; wz = tv[k, 2] - opz
        px[k] = wx * ex[0] + wy * ex[1] + wz * ex[2]
        py[k] = wx * ey[0] + wy * ey[1] + wz * ey[2]

    # I3 from the solid angle at the effective (lifted) observation point
    oex = opx + h * nhat[0]; oey = opy + h * nhat[1]; oez = opz + h * nhat[2]
    Omega = _solid_angle(tv[0], tv[1], tv[2], oex, oey, oez)
    I3 = -Omega / h

    # Edge loop: accumulate BD(0,0,1), BN(0,0,1), BN(1,0,1), BN(0,1,1)
    BD001 = 0.0
    BN1_001 = 0.0; BN2_001 = 0.0
    BN1_101 = 0.0; BN2_101 = 0.0
    BN1_011 = 0.0; BN2_011 = 0.0
    for e in range(3):
        ax = px[e]; ay = py[e]
        bx = px[(e + 1) % 3]; by = py[(e + 1) % 3]
        evx = bx - ax; evy = by - ay
        L = np.sqrt(evx * evx + evy * evy)
        if L <= 1e-30:
            continue
        tx = evx / L; ty = evy / L
        nx = ty; ny = -tx
        dp = ax * nx + ay * ny
        ua = ax * tx + ay * ty
        ub = bx * tx + by * ty
        rho2 = dp * dp + h2
        dJ1 = _J1(ub, rho2) - _J1(ua, rho2)
        dK1 = _K1(ub, rho2) - _K1(ua, rho2)

        v00 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v10 = _edge_int(1, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v01 = _edge_int(0, 1, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)

        BD001 += dp * v00
        BN1_001 += nx * v00; BN2_001 += ny * v00
        BN1_101 += nx * v10; BN2_101 += ny * v10
        BN1_011 += nx * v01; BN2_011 += ny * v01

    I1 = BD001 - h2 * I3

    # 2D moments at n=3 (recursion constants: first order c=-1, second c=+1)
    m10 = -BN1_001
    m01 = -BN2_001
    m20 = I1 - BN1_101
    m11 = -BN1_011
    m02 = I1 - BN2_011

    # B1 = I1*delta, B2 = eps^2*I3*delta
    for i in range(3):
        for j in range(3):
            out[0, i, j] = 0.0
            out[1, i, j] = 0.0
            out[2, i, j] = 0.0
        out[0, i, i] = I1
        out[1, i, i] = eps * eps * I3

    # B3 = T2[3]: expand d_i d_j over the (ex, ey, nhat) frame,
    # d = -xi1*ex - xi2*ey + z*nhat
    for i in range(3):
        for j in range(3):
            out[2, i, j] = (
                ex[i] * ex[j] * m20
                + (ex[i] * ey[j] + ey[i] * ex[j]) * m11
                + ey[i] * ey[j] * m02
                - z * (ex[i] * nhat[j] + nhat[i] * ex[j]) * m10
                - z * (ey[i] * nhat[j] + nhat[i] * ey[j]) * m01
                + z * z * nhat[i] * nhat[j] * I3
            )


# ---------------------------------------------------------------------
# T-kernel basis blocks for one (source triangle, obs) pair
# ---------------------------------------------------------------------

@njit(cache=True)
def _t_basis_pair(tv, ex, ey, nhat, nsrc, obs, eps, out, T35, P):
    """Write the 6 T-kernel basis 3x3 blocks into out (6,3,3).

    Order: [N[P1], N[P2], N[P3], R[P1], R[P2], R[P3]] (lam slots, then mu slots).
    ``T35`` (3,3,3) and ``P`` (3,3,3) are caller-provided scratch.
    """
    dx = obs[0] - tv[0, 0]; dy = obs[1] - tv[0, 1]; dz = obs[2] - tv[0, 2]
    z = dx * nhat[0] + dy * nhat[1] + dz * nhat[2]
    h2 = z * z + eps * eps
    h = np.sqrt(h2)
    opx = obs[0] - z * nhat[0]
    opy = obs[1] - z * nhat[1]
    opz = obs[2] - z * nhat[2]

    px = np.empty(3); py = np.empty(3)
    for k in range(3):
        wx = tv[k, 0] - opx; wy = tv[k, 1] - opy; wz = tv[k, 2] - opz
        px[k] = wx * ex[0] + wy * ex[1] + wz * ex[2]
        py[k] = wx * ey[0] + wy * ey[1] + wz * ey[2]

    oex = opx + h * nhat[0]; oey = opy + h * nhat[1]; oez = opz + h * nhat[2]
    Omega = _solid_angle(tv[0], tv[1], tv[2], oex, oey, oez)
    I3 = -Omega / h

    # Edge accumulators: m=1 needs (0,0); m=3 needs (0,0),(1,0),(0,1),
    # (2,0),(1,1),(0,2) (BD only for (0,0,3); BN1/BN2 as consumed below).
    BN1_001 = 0.0; BN2_001 = 0.0
    BD003 = 0.0
    BN1_003 = 0.0; BN2_003 = 0.0
    BN1_103 = 0.0
    BN1_013 = 0.0; BN2_013 = 0.0
    BN1_203 = 0.0
    BN1_113 = 0.0
    BN1_023 = 0.0; BN2_023 = 0.0
    for e in range(3):
        ax = px[e]; ay = py[e]
        bx = px[(e + 1) % 3]; by = py[(e + 1) % 3]
        evx = bx - ax; evy = by - ay
        L = np.sqrt(evx * evx + evy * evy)
        if L <= 1e-30:
            continue
        tx = evx / L; ty = evy / L
        nx = ty; ny = -tx
        dp = ax * nx + ay * ny
        ua = ax * tx + ay * ty
        ub = bx * tx + by * ty
        rho2 = dp * dp + h2
        dJ1 = _J1(ub, rho2) - _J1(ua, rho2)
        dK1 = _K1(ub, rho2) - _K1(ua, rho2)
        dJ3 = _J3(ub, rho2) - _J3(ua, rho2)
        dK3 = _K3(ub, rho2) - _K3(ua, rho2)
        dL3 = dJ1 - rho2 * dJ3

        v00m1 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0)
        v00 = _edge_int(0, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v10 = _edge_int(1, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v01 = _edge_int(0, 1, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v20 = _edge_int(2, 0, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v11 = _edge_int(1, 1, dp, nx, ny, tx, ty, dJ3, dK3, dL3)
        v02 = _edge_int(0, 2, dp, nx, ny, tx, ty, dJ3, dK3, dL3)

        BN1_001 += nx * v00m1; BN2_001 += ny * v00m1
        BD003 += dp * v00
        BN1_003 += nx * v00; BN2_003 += ny * v00
        BN1_103 += nx * v10
        BN1_013 += nx * v01; BN2_013 += ny * v01
        BN1_203 += nx * v20
        BN1_113 += nx * v11
        BN1_023 += nx * v02; BN2_023 += ny * v02

    I5 = (BD003 + I3) / (3.0 * h2)

    # 2D moments. First order: M(.,n) = -1/(n-2) * BN(.,n-2).
    m10_3 = -BN1_001
    m01_3 = -BN2_001
    m10_5 = -BN1_003 / 3.0
    m01_5 = -BN2_003 / 3.0
    # Second order at n=5: c = 1/3
    m20_5 = (I3 - BN1_103) / 3.0
    m11_5 = -BN1_013 / 3.0
    m02_5 = (I3 - BN2_013) / 3.0
    # Third order at n=5: c = 1/3
    m30_5 = (2.0 * m10_3 - BN1_203) / 3.0
    m21_5 = (m01_3 - BN1_113) / 3.0
    m12_5 = -BN1_023 / 3.0
    m03_5 = (2.0 * m01_3 - BN2_023) / 3.0

    # First 3D moments V3, V5: d = -xi1*ex - xi2*ey + z*nhat
    V3x = -ex[0] * m10_3 - ey[0] * m01_3 + nhat[0] * z * I3
    V3y = -ex[1] * m10_3 - ey[1] * m01_3 + nhat[1] * z * I3
    V3z = -ex[2] * m10_3 - ey[2] * m01_3 + nhat[2] * z * I3
    V5x = -ex[0] * m10_5 - ey[0] * m01_5 + nhat[0] * z * I5
    V5y = -ex[1] * m10_5 - ey[1] * m01_5 + nhat[1] * z * I5
    V5z = -ex[2] * m10_5 - ey[2] * m01_5 + nhat[2] * z * I5

    # Third 3D moment T3[5]: sum over component triples of the frame.
    # Component c: 0 -> (-1, ex), 1 -> (-1, ey), 2 -> (z, nhat); for a
    # triple with a slots of xi1, b of xi2, and the rest z, the scalar
    # factor is (-1)^(a+b) * z^(#z) * M(a, b, 5).
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T35[i, j, k] = 0.0
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                a = (1 if c1 == 0 else 0) + (1 if c2 == 0 else 0) + (1 if c3 == 0 else 0)
                b = (1 if c1 == 1 else 0) + (1 if c2 == 1 else 0) + (1 if c3 == 1 else 0)
                nz = 3 - a - b
                if a == 3:
                    mom = m30_5
                elif a == 2 and b == 1:
                    mom = m21_5
                elif a == 2 and b == 0:
                    mom = m20_5
                elif a == 1 and b == 2:
                    mom = m12_5
                elif a == 1 and b == 1:
                    mom = m11_5
                elif a == 1 and b == 0:
                    mom = m10_5
                elif a == 0 and b == 3:
                    mom = m03_5
                elif a == 0 and b == 2:
                    mom = m02_5
                elif a == 0 and b == 1:
                    mom = m01_5
                else:
                    mom = I5
                sign = 1.0
                if (a + b) % 2 == 1:
                    sign = -1.0
                coeff = sign * z ** nz * mom
                if coeff == 0.0:
                    continue
                b1 = ex if c1 == 0 else (ey if c1 == 1 else nhat)
                b2 = ex if c2 == 0 else (ey if c2 == 1 else nhat)
                b3 = ex if c3 == 0 else (ey if c3 == 1 else nhat)
                for i in range(3):
                    ci = coeff * b1[i]
                    for j in range(3):
                        cij = ci * b2[j]
                        for k in range(3):
                            T35[i, j, k] += cij * b3[k]

    # ---- P tensors and their N / R contractions ----
    # N[P]_ij = n_j * sum_m P[i,m,m]                     (lam slot)
    # R[P]_ij = sum_m n_m P[i,j,m] + sum_k n_k P[i,k,j]  (mu slot)
    e2 = eps * eps
    V3 = (V3x, V3y, V3z)
    V5 = (V5x, V5y, V5z)

    # P1[i,k,m] = delta_ik V3_m
    for i in range(3):
        for k in range(3):
            for m in range(3):
                P[i, k, m] = V3[m] if i == k else 0.0
    _contract_NR(P, nsrc, out, 0, 3)

    # P2[i,k,m] = delta_im V3_k + delta_km V3_i - 3*T35[i,k,m]
    for i in range(3):
        for k in range(3):
            for m in range(3):
                val = -3.0 * T35[i, k, m]
                if i == m:
                    val += V3[k]
                if k == m:
                    val += V3[i]
                P[i, k, m] = val
    _contract_NR(P, nsrc, out, 1, 4)

    # P3[i,k,m] = eps^2 * delta_ik V5_m
    for i in range(3):
        for k in range(3):
            for m in range(3):
                P[i, k, m] = e2 * V5[m] if i == k else 0.0
    _contract_NR(P, nsrc, out, 2, 5)


@njit(cache=True, inline="always")
def _contract_NR(P, n, out, idx_N, idx_R):
    """N (lam-slot) and R (mu-slot) contractions of P (3,3,3) with the
    source normal n into out:
        N_ij = n_j * sum_m P[i,m,m]
        R_ij = sum_m n_m P[i,j,m] + sum_k n_k P[i,k,j]
    """
    for i in range(3):
        tr = P[i, 0, 0] + P[i, 1, 1] + P[i, 2, 2]
        for j in range(3):
            out[idx_N, i, j] = n[j] * tr
            out[idx_R, i, j] = (P[i, j, 0] * n[0] + P[i, j, 1] * n[1]
                                + P[i, j, 2] * n[2]
                                + n[0] * P[i, 0, j] + n[1] * P[i, 1, j]
                                + n[2] * P[i, 2, j])


# ---------------------------------------------------------------------
# Full-matrix basis assembly (parallel over source triangles)
# ---------------------------------------------------------------------

@njit(cache=True, parallel=True)
def u_basis_matrices(x_field, tri_verts, eps_arr):
    """U-kernel basis stack, shape (3, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3, 3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for b in range(3):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


@njit(cache=True, parallel=True)
def t_basis_matrices(x_field, tri_verts, normals, eps_arr):
    """T-kernel basis stack, shape (6, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((6, 3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for b in range(6):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


# Serial, GIL-releasing variants of the basis assemblers for use from
# Python worker THREADS (parallel ACA compression): numba's default
# macOS "workqueue" threading layer is NOT thread-safe under concurrent
# parallel=True calls (hard crash), and without nogil the threads would
# serialize on the GIL anyway. Across-block thread parallelism replaces
# the prange.

@njit(cache=True, nogil=True)
def u_basis_matrices_serial(x_field, tri_verts, eps_arr):
    """Serial/nogil u_basis_matrices, shape (3, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3, 3 * N_f, 3 * N_s))
    ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
    blk = np.empty((3, 3, 3))
    for s in range(N_s):
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for b in range(3):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


@njit(cache=True, nogil=True)
def t_basis_matrices_serial(x_field, tri_verts, normals, eps_arr):
    """Serial/nogil t_basis_matrices, shape (6, 3*N_f, 3*N_s)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((6, 3 * N_f, 3 * N_s))
    ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
    blk = np.empty((6, 3, 3))
    T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
    for s in range(N_s):
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for b in range(6):
                for i in range(3):
                    for j in range(3):
                        out[b, 3 * f + i, 3 * s + j] = blk[b, i, j]
    return out


@njit(cache=True, parallel=True)
def u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3):
    """U-kernel matrix with coefficients applied in-loop (no basis storage)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _u_basis_pair(tri_verts[s], ex, ey, nhat, x_field[f], eps, blk)
            for i in range(3):
                for j in range(3):
                    out[3 * f + i, 3 * s + j] = (g1 * blk[0, i, j]
                                                 + g2 * blk[1, i, j]
                                                 + g3 * blk[2, i, j])
    return out


@njit(cache=True, parallel=True)
def t_matrix_direct(x_field, tri_verts, normals, eps_arr,
                    c1, c2, c3, c4, c5, c6):
    """T-kernel matrix with coefficients applied in-loop (no basis storage)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    out = np.zeros((3 * N_f, 3 * N_s))
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nhat = np.empty(3)
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        if not _tri_frame(tri_verts[s], ex, ey, nhat):
            continue
        eps = eps_arr[s]
        for f in range(N_f):
            _t_basis_pair(tri_verts[s], ex, ey, nhat, normals[s],
                          x_field[f], eps, blk, T35, P)
            for i in range(3):
                for j in range(3):
                    out[3 * f + i, 3 * s + j] = (
                        c1 * blk[0, i, j] + c2 * blk[1, i, j]
                        + c3 * blk[2, i, j] + c4 * blk[3, i, j]
                        + c5 * blk[4, i, j] + c6 * blk[5, i, j])
    return out


# =====================================================================
# Stress kernels (force->stress and slip->stress), numba port of the
# scalar oracles in mollified_kernel/analytical_kernels.py
# (analytical_kelvin_stress / analytical_stress_kernel). The slip->stress
# kernel is the gradient of the T kernel, so it needs the moment
# recursion one order higher than _t_basis_pair: I7, T2[5], T2[7], and the
# rank-4 T4[7]. These are emitted MATERIAL-APPLIED (single material per
# call, like u_matrix_direct / t_matrix_direct), which is what
# evaluate_stress needs; a geometry-basis split is future work.
# =====================================================================


@njit(cache=True, inline="always")
def _J5(u, rho2):
    if rho2 < 1e-60:
        return 0.0
    R = np.sqrt(u * u + rho2)
    return u * (3.0 * rho2 + 2.0 * u * u) / (3.0 * rho2 * rho2 * R * R * R)


@njit(cache=True, inline="always")
def _K5(u, rho2):
    R = np.sqrt(u * u + rho2)
    if R < 1e-30:
        return 0.0
    return -1.0 / (3.0 * R * R * R)


@njit(cache=True, inline="always")
def _binom3(n, k):
    """Binomial C(n,k) for n,k <= 3."""
    if k <= 0 or k >= n:
        return 1.0
    if n == 2:
        return 2.0
    return 3.0          # n == 3, k == 1 or 2


@njit(cache=True, inline="always")
def _edge_int3(a, b, dp, nx, ny, tx, ty, dJ, dK, dL, dM):
    """Edge integral of xi1^a xi2^b / R^m, a+b <= 3 (u-powers up to 3)."""
    total = 0.0
    for i in range(a + 1):
        for j in range(b + 1):
            coeff = (_binom3(a, i) * _binom3(b, j)
                     * dp ** (a + b - i - j)
                     * nx ** (a - i) * tx ** i
                     * ny ** (b - j) * ty ** j)
            k = i + j
            if k == 0:
                delta = dJ
            elif k == 1:
                delta = dK
            elif k == 2:
                delta = dL
            else:
                delta = dM
            total += coeff * delta
    return total


@njit(cache=True)
def _stress_2d_moments(tv, ex, ey, nhat, obs, eps, M3, M5, M7, need7):
    """2D area moments for the stress kernels (mirror integrate_all_moments).

    Fills M3 (>=3x3), M5 (>=4x4), and -- if need7 -- M7 (>=5x5) with
    M[a,b] = integral of xi1^a xi2^b / R^n dA at n=3/5/7. Returns
    (I3, I5, I7, z); I7 is 0 when need7 is False.
    """
    dx = obs[0] - tv[0, 0]; dy = obs[1] - tv[0, 1]; dz = obs[2] - tv[0, 2]
    z = dx * nhat[0] + dy * nhat[1] + dz * nhat[2]
    h2 = z * z + eps * eps
    h = np.sqrt(h2)
    opx = obs[0] - z * nhat[0]
    opy = obs[1] - z * nhat[1]
    opz = obs[2] - z * nhat[2]

    px = np.empty(3); py = np.empty(3)
    for k in range(3):
        wx = tv[k, 0] - opx; wy = tv[k, 1] - opy; wz = tv[k, 2] - opz
        px[k] = wx * ex[0] + wy * ex[1] + wz * ex[2]
        py[k] = wx * ey[0] + wy * ey[1] + wz * ey[2]

    oex = opx + h * nhat[0]; oey = opy + h * nhat[1]; oez = opz + h * nhat[2]
    Omega = _solid_angle(tv[0], tv[1], tv[2], oex, oey, oez)
    I3 = -Omega / h

    BN1m1 = 0.0; BN2m1 = 0.0
    BD003 = 0.0; BD005 = 0.0
    bn1_3 = np.zeros((3, 3)); bn2_3 = np.zeros((3, 3))
    bn1_5 = np.zeros((4, 4)); bn2_5 = np.zeros((4, 4))
    for e in range(3):
        ax = px[e]; ay = py[e]
        bx = px[(e + 1) % 3]; by = py[(e + 1) % 3]
        evx = bx - ax; evy = by - ay
        L = np.sqrt(evx * evx + evy * evy)
        if L <= 1e-30:
            continue
        tx = evx / L; ty = evy / L
        nx = ty; ny = -tx
        dp = ax * nx + ay * ny
        ua = ax * tx + ay * ty
        ub = bx * tx + by * ty
        rho2 = dp * dp + h2

        dJ1 = _J1(ub, rho2) - _J1(ua, rho2)
        dK1 = _K1(ub, rho2) - _K1(ua, rho2)
        dJ3 = _J3(ub, rho2) - _J3(ua, rho2)
        dK3 = _K3(ub, rho2) - _K3(ua, rho2)
        dL3 = dJ1 - rho2 * dJ3

        # m = 1: (0,0)
        v = _edge_int3(0, 0, dp, nx, ny, tx, ty, dJ1, dK1, 0.0, 0.0)
        BN1m1 += nx * v; BN2m1 += ny * v

        # m = 3: a+b <= 2
        for a in range(3):
            for b in range(3 - a):
                v = _edge_int3(a, b, dp, nx, ny, tx, ty, dJ3, dK3, dL3, 0.0)
                bn1_3[a, b] += nx * v
                bn2_3[a, b] += ny * v
                if a == 0 and b == 0:
                    BD003 += dp * v

        if need7:
            dJ5 = _J5(ub, rho2) - _J5(ua, rho2)
            dK5 = _K5(ub, rho2) - _K5(ua, rho2)
            dL5 = dJ3 - rho2 * dJ5
            dM5 = dK3 - rho2 * dK5
            # m = 5: a+b <= 3
            for a in range(4):
                for b in range(4 - a):
                    v = _edge_int3(a, b, dp, nx, ny, tx, ty,
                                   dJ5, dK5, dL5, dM5)
                    bn1_5[a, b] += nx * v
                    bn2_5[a, b] += ny * v
                    if a == 0 and b == 0:
                        BD005 += dp * v

    I5 = (BD003 + I3) / (3.0 * h2) if h2 > 1e-60 else 0.0

    M3[0, 0] = I3
    M3[1, 0] = -BN1m1
    M3[0, 1] = -BN2m1

    M5[0, 0] = I5
    M5[1, 0] = -bn1_3[0, 0] / 3.0
    M5[0, 1] = -bn2_3[0, 0] / 3.0
    M5[2, 0] = (I3 - bn1_3[1, 0]) / 3.0
    M5[1, 1] = -bn1_3[0, 1] / 3.0
    M5[0, 2] = (I3 - bn2_3[0, 1]) / 3.0
    M5[3, 0] = (2.0 * M3[1, 0] - bn1_3[2, 0]) / 3.0
    M5[2, 1] = (M3[0, 1] - bn1_3[1, 1]) / 3.0
    M5[1, 2] = -bn1_3[0, 2] / 3.0
    M5[0, 3] = (2.0 * M3[0, 1] - bn2_3[0, 2]) / 3.0

    I7 = 0.0
    if need7:
        I7 = (BD005 + 3.0 * I5) / (5.0 * h2) if h2 > 1e-60 else 0.0
        M7[0, 0] = I7
        M7[1, 0] = -bn1_5[0, 0] / 5.0
        M7[0, 1] = -bn2_5[0, 0] / 5.0
        M7[2, 0] = (I5 - bn1_5[1, 0]) / 5.0
        M7[1, 1] = -bn1_5[0, 1] / 5.0
        M7[0, 2] = (I5 - bn2_5[0, 1]) / 5.0
        M7[3, 0] = (2.0 * M5[1, 0] - bn1_5[2, 0]) / 5.0
        M7[2, 1] = (M5[0, 1] - bn1_5[1, 1]) / 5.0
        M7[1, 2] = -bn1_5[0, 2] / 5.0
        M7[0, 3] = (2.0 * M5[0, 1] - bn2_5[0, 2]) / 5.0
        M7[4, 0] = (3.0 * M5[2, 0] - bn1_5[3, 0]) / 5.0
        M7[3, 1] = (2.0 * M5[1, 1] - bn1_5[2, 1]) / 5.0
        M7[2, 2] = (M5[0, 2] - bn1_5[1, 2]) / 5.0
        M7[1, 3] = -bn1_5[0, 3] / 5.0
        M7[0, 4] = (3.0 * M5[0, 2] - bn2_5[0, 3]) / 5.0

    return I3, I5, I7, z


@njit(cache=True, inline="always")
def _build_T2(M, ex, ey, nhat, z, out):
    """T2[n][i,j] = integral d_i d_j / R^n dA from 2D moments M[a,b]."""
    for i in range(3):
        for j in range(3):
            out[i, j] = (
                ex[i] * ex[j] * M[2, 0]
                + (ex[i] * ey[j] + ey[i] * ex[j]) * M[1, 1]
                + ey[i] * ey[j] * M[0, 2]
                - z * (ex[i] * nhat[j] + nhat[i] * ex[j]) * M[1, 0]
                - z * (ey[i] * nhat[j] + nhat[i] * ey[j]) * M[0, 1]
                + z * z * nhat[i] * nhat[j] * M[0, 0]
            )


@njit(cache=True, inline="always")
def _build_T3(M, bas, z, out):
    """T3[i,j,k] = integral d_i d_j d_k / R^5 dA; bas[(0,1,2)] = ex,ey,nhat."""
    for i in range(3):
        for j in range(3):
            for k in range(3):
                out[i, j, k] = 0.0
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                a = 0; b = 0
                if c1 == 0:
                    a += 1
                elif c1 == 1:
                    b += 1
                if c2 == 0:
                    a += 1
                elif c2 == 1:
                    b += 1
                if c3 == 0:
                    a += 1
                elif c3 == 1:
                    b += 1
                nz = 3 - a - b
                sign = -1.0 if (a + b) % 2 == 1 else 1.0
                coeff = sign * z ** nz * M[a, b]
                if coeff == 0.0:
                    continue
                for i in range(3):
                    ci = coeff * bas[c1, i]
                    for j in range(3):
                        cij = ci * bas[c2, j]
                        for k in range(3):
                            out[i, j, k] += cij * bas[c3, k]


@njit(cache=True, inline="always")
def _build_T4(M, bas, z, out):
    """T4[i,j,k,l] = integral d_i d_j d_k d_l / R^7 dA."""
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    out[i, j, k, l] = 0.0
    for c1 in range(3):
        for c2 in range(3):
            for c3 in range(3):
                for c4 in range(3):
                    a = 0; b = 0
                    for c in (c1, c2, c3, c4):
                        if c == 0:
                            a += 1
                        elif c == 1:
                            b += 1
                    nz = 4 - a - b
                    sign = -1.0 if (a + b) % 2 == 1 else 1.0
                    coeff = sign * z ** nz * M[a, b]
                    if coeff == 0.0:
                        continue
                    for i in range(3):
                        ci = coeff * bas[c1, i]
                        for j in range(3):
                            cij = ci * bas[c2, j]
                            for k in range(3):
                                cijk = cij * bas[c3, k]
                                for l in range(3):
                                    out[i, j, k, l] += cijk * bas[c4, l]


@njit(cache=True)
def _kelvin_stress_pair(tv, ex, ey, nhat, obs, eps, mu, nu, lam,
                        Sout, M3, M5, M7, bas, T35):
    """Force->stress S[i,j,k] (sigma_ij = S[i,j,k] f_k) for one pair."""
    I3, I5, I7, z = _stress_2d_moments(tv, ex, ey, nhat, obs, eps,
                                       M3, M5, M7, False)
    V3 = np.empty(3); V5 = np.empty(3)
    for i in range(3):
        V3[i] = -ex[i] * M3[1, 0] - ey[i] * M3[0, 1] + z * nhat[i] * M3[0, 0]
        V5[i] = -ex[i] * M5[1, 0] - ey[i] * M5[0, 1] + z * nhat[i] * M5[0, 0]
    bas[0, 0] = ex[0]; bas[0, 1] = ex[1]; bas[0, 2] = ex[2]
    bas[1, 0] = ey[0]; bas[1, 1] = ey[1]; bas[1, 2] = ey[2]
    bas[2, 0] = nhat[0]; bas[2, 1] = nhat[1]; bas[2, 2] = nhat[2]
    _build_T3(M5, bas, z, T35)

    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    cbdg = 6.0 * (1.0 - nu) * eps * eps

    # G1[i,j,m] = dG_ij/dx_m integrated
    G1 = np.empty((3, 3, 3))
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

    for k in range(3):
        trace_k = G1[0, k, 0] + G1[1, k, 1] + G1[2, k, 2]
        for i in range(3):
            for j in range(3):
                dij = 1.0 if i == j else 0.0
                Sout[i, j, k] = (lam * dij * trace_k
                                 + mu * G1[i, k, j] + mu * G1[j, k, i])


@njit(cache=True)
def _dd_stress_pair(tv, ex, ey, nhat, nrm, obs, eps, mu, nu, lam,
                    Hout, M3, M5, M7, bas, T2_5, T2_7, T4_7, ID2G, B):
    """Slip->stress H[m,n,k] (sigma_mn = H[m,n,k] slip_k) for one pair."""
    I3, I5, I7, z = _stress_2d_moments(tv, ex, ey, nhat, obs, eps,
                                       M3, M5, M7, True)
    bas[0, 0] = ex[0]; bas[0, 1] = ex[1]; bas[0, 2] = ex[2]
    bas[1, 0] = ey[0]; bas[1, 1] = ey[1]; bas[1, 2] = ey[2]
    bas[2, 0] = nhat[0]; bas[2, 1] = nhat[1]; bas[2, 2] = nhat[2]
    _build_T2(M5, ex, ey, nhat, z, T2_5)
    _build_T2(M7, ex, ey, nhat, z, T2_7)
    _build_T4(M7, bas, z, T4_7)

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

    # B[r,s,k] = C_kjpq n_j ID2G[r,p,s,q]
    for r in range(3):
        for s in range(3):
            tr = 0.0
            for p in range(3):
                tr += ID2G[r, p, s, p]
            for k in range(3):
                nq = 0.0
                npd = 0.0
                for q in range(3):
                    nq += nrm[q] * ID2G[r, k, s, q]
                for p in range(3):
                    npd += nrm[p] * ID2G[r, p, s, k]
                B[r, s, k] = lam * nrm[k] * tr + mu * nq + mu * npd

    # H[m,n,k] = -C_mnrs B[r,s,k]
    for k in range(3):
        trB = B[0, 0, k] + B[1, 1, k] + B[2, 2, k]
        for m in range(3):
            for n in range(3):
                dmn = 1.0 if m == n else 0.0
                Hout[m, n, k] = -(lam * dmn * trB
                                  + mu * (B[m, n, k] + B[n, m, k]))


# =====================================================================
# Matrix-free displacement contraction drivers: u(x) = sum_s K_xs @ d_s
# without materializing the (3N_f, 3N_s) influence matrix. O(N_f)
# memory at any N_s -- the evaluation path for very large models.
# Same pair kernels as the *_matrix_direct assemblers, so parity with
# `assemble_*_matrix(...) @ density` is machine precision.
# =====================================================================


@njit(cache=True, parallel=True)
def u_disp_contract(x_field, tri_verts, eps_arr, density, g1, g2, g3):
    """Displacement (N_f,3) from a triangulated FORCE source (U kernel).

    u(obs) = sum_s U[obs,s] @ density[s]. Parallel over obs points.
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    out = np.zeros((N_f, 3))
    for f in prange(N_f):
        blk = np.empty((3, 3, 3))
        obs = x_field[f]
        a0 = 0.0; a1 = 0.0; a2 = 0.0
        for s in range(N_s):
            if not OK[s]:
                continue
            d0 = density[s, 0]; d1 = density[s, 1]; d2 = density[s, 2]
            if d0 == 0.0 and d1 == 0.0 and d2 == 0.0:
                continue
            _u_basis_pair(tri_verts[s], EX[s], EY[s], NH[s], obs,
                          eps_arr[s], blk)
            for j in range(3):
                dj = density[s, j]
                if dj == 0.0:
                    continue
                a0 += (g1 * blk[0, 0, j] + g2 * blk[1, 0, j]
                       + g3 * blk[2, 0, j]) * dj
                a1 += (g1 * blk[0, 1, j] + g2 * blk[1, 1, j]
                       + g3 * blk[2, 1, j]) * dj
                a2 += (g1 * blk[0, 2, j] + g2 * blk[1, 2, j]
                       + g3 * blk[2, 2, j]) * dj
        out[f, 0] = a0; out[f, 1] = a1; out[f, 2] = a2
    return out


@njit(cache=True, parallel=True)
def t_disp_contract(x_field, tri_verts, normals, eps_arr, density,
                    c1, c2, c3, c4, c5, c6):
    """Displacement (N_f,3) from a triangulated SLIP/displacement source
    (T kernel). u(obs) = sum_s T[obs,s] @ density[s]. Parallel over obs.
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    out = np.zeros((N_f, 3))
    for f in prange(N_f):
        blk = np.empty((6, 3, 3))
        T35 = np.empty((3, 3, 3)); P = np.empty((3, 3, 3))
        obs = x_field[f]
        a0 = 0.0; a1 = 0.0; a2 = 0.0
        for s in range(N_s):
            if not OK[s]:
                continue
            d0 = density[s, 0]; d1 = density[s, 1]; d2 = density[s, 2]
            if d0 == 0.0 and d1 == 0.0 and d2 == 0.0:
                continue
            _t_basis_pair(tri_verts[s], EX[s], EY[s], NH[s], normals[s],
                          obs, eps_arr[s], blk, T35, P)
            for j in range(3):
                dj = density[s, j]
                if dj == 0.0:
                    continue
                a0 += (c1 * blk[0, 0, j] + c2 * blk[1, 0, j]
                       + c3 * blk[2, 0, j] + c4 * blk[3, 0, j]
                       + c5 * blk[4, 0, j] + c6 * blk[5, 0, j]) * dj
                a1 += (c1 * blk[0, 1, j] + c2 * blk[1, 1, j]
                       + c3 * blk[2, 1, j] + c4 * blk[3, 1, j]
                       + c5 * blk[4, 1, j] + c6 * blk[5, 1, j]) * dj
                a2 += (c1 * blk[0, 2, j] + c2 * blk[1, 2, j]
                       + c3 * blk[2, 2, j] + c4 * blk[3, 2, j]
                       + c5 * blk[4, 2, j] + c6 * blk[5, 2, j]) * dj
        out[f, 0] = a0; out[f, 1] = a1; out[f, 2] = a2
    return out


@njit(cache=True, parallel=True)
def _precompute_frames(tri_verts):
    """Per-source in-plane frames; OK[s] False for degenerate triangles."""
    N_s = tri_verts.shape[0]
    EX = np.zeros((N_s, 3)); EY = np.zeros((N_s, 3)); NH = np.zeros((N_s, 3))
    OK = np.zeros(N_s, dtype=np.bool_)
    for s in prange(N_s):
        ex = np.empty(3); ey = np.empty(3); nh = np.empty(3)
        if _tri_frame(tri_verts[s], ex, ey, nh):
            EX[s, 0] = ex[0]; EX[s, 1] = ex[1]; EX[s, 2] = ex[2]
            EY[s, 0] = ey[0]; EY[s, 1] = ey[1]; EY[s, 2] = ey[2]
            NH[s, 0] = nh[0]; NH[s, 1] = nh[1]; NH[s, 2] = nh[2]
            OK[s] = True
    return EX, EY, NH, OK


@njit(cache=True, parallel=True)
def kelvin_stress_contract(x_field, tri_verts, eps_arr, density, mu, lam):
    """Stress (N_f,3,3) from a triangulated FORCE source (single-layer).

    sigma(obs) = sum_s S[obs,s][:,:,k] density[s,k]. Parallel over obs.
    Material as (mu, lam); nu is derived, never lam from a 1/(1-2nu).
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    nu = lam / (2.0 * (lam + mu))
    sig = np.zeros((N_f, 3, 3))
    for f in prange(N_f):
        M3 = np.zeros((3, 3)); M5 = np.zeros((4, 4)); M7 = np.zeros((5, 5))
        bas = np.empty((3, 3)); T35 = np.empty((3, 3, 3))
        S = np.empty((3, 3, 3))
        obs = x_field[f]
        a00 = 0.0; a01 = 0.0; a02 = 0.0
        a10 = 0.0; a11 = 0.0; a12 = 0.0
        a20 = 0.0; a21 = 0.0; a22 = 0.0
        for s in range(N_s):
            if not OK[s]:
                continue
            d0 = density[s, 0]; d1 = density[s, 1]; d2 = density[s, 2]
            if d0 == 0.0 and d1 == 0.0 and d2 == 0.0:
                continue
            _kelvin_stress_pair(tri_verts[s], EX[s], EY[s], NH[s], obs,
                                eps_arr[s], mu, nu, lam, S, M3, M5, M7, bas, T35)
            a00 += S[0, 0, 0] * d0 + S[0, 0, 1] * d1 + S[0, 0, 2] * d2
            a01 += S[0, 1, 0] * d0 + S[0, 1, 1] * d1 + S[0, 1, 2] * d2
            a02 += S[0, 2, 0] * d0 + S[0, 2, 1] * d1 + S[0, 2, 2] * d2
            a10 += S[1, 0, 0] * d0 + S[1, 0, 1] * d1 + S[1, 0, 2] * d2
            a11 += S[1, 1, 0] * d0 + S[1, 1, 1] * d1 + S[1, 1, 2] * d2
            a12 += S[1, 2, 0] * d0 + S[1, 2, 1] * d1 + S[1, 2, 2] * d2
            a20 += S[2, 0, 0] * d0 + S[2, 0, 1] * d1 + S[2, 0, 2] * d2
            a21 += S[2, 1, 0] * d0 + S[2, 1, 1] * d1 + S[2, 1, 2] * d2
            a22 += S[2, 2, 0] * d0 + S[2, 2, 1] * d1 + S[2, 2, 2] * d2
        sig[f, 0, 0] = a00; sig[f, 0, 1] = a01; sig[f, 0, 2] = a02
        sig[f, 1, 0] = a10; sig[f, 1, 1] = a11; sig[f, 1, 2] = a12
        sig[f, 2, 0] = a20; sig[f, 2, 1] = a21; sig[f, 2, 2] = a22
    return sig


@njit(cache=True, parallel=True)
def dd_stress_contract(x_field, tri_verts, normals, eps_arr, density, mu, lam):
    """Stress (N_f,3,3) from a triangulated SLIP/displacement source (DD).

    sigma(obs) = sum_s H[obs,s][:,:,k] density[s,k]. Parallel over obs.
    Material as (mu, lam); nu is derived, never lam from a 1/(1-2nu).
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    nu = lam / (2.0 * (lam + mu))
    sig = np.zeros((N_f, 3, 3))
    for f in prange(N_f):
        M3 = np.zeros((3, 3)); M5 = np.zeros((4, 4)); M7 = np.zeros((5, 5))
        bas = np.empty((3, 3))
        T2_5 = np.empty((3, 3)); T2_7 = np.empty((3, 3))
        T4_7 = np.empty((3, 3, 3, 3)); ID2G = np.empty((3, 3, 3, 3))
        B = np.empty((3, 3, 3)); H = np.empty((3, 3, 3))
        obs = x_field[f]
        a00 = 0.0; a01 = 0.0; a02 = 0.0
        a10 = 0.0; a11 = 0.0; a12 = 0.0
        a20 = 0.0; a21 = 0.0; a22 = 0.0
        for s in range(N_s):
            if not OK[s]:
                continue
            d0 = density[s, 0]; d1 = density[s, 1]; d2 = density[s, 2]
            if d0 == 0.0 and d1 == 0.0 and d2 == 0.0:
                continue
            _dd_stress_pair(tri_verts[s], EX[s], EY[s], NH[s], normals[s], obs,
                            eps_arr[s], mu, nu, lam, H, M3, M5, M7, bas,
                            T2_5, T2_7, T4_7, ID2G, B)
            a00 += H[0, 0, 0] * d0 + H[0, 0, 1] * d1 + H[0, 0, 2] * d2
            a01 += H[0, 1, 0] * d0 + H[0, 1, 1] * d1 + H[0, 1, 2] * d2
            a02 += H[0, 2, 0] * d0 + H[0, 2, 1] * d1 + H[0, 2, 2] * d2
            a10 += H[1, 0, 0] * d0 + H[1, 0, 1] * d1 + H[1, 0, 2] * d2
            a11 += H[1, 1, 0] * d0 + H[1, 1, 1] * d1 + H[1, 1, 2] * d2
            a12 += H[1, 2, 0] * d0 + H[1, 2, 1] * d1 + H[1, 2, 2] * d2
            a20 += H[2, 0, 0] * d0 + H[2, 0, 1] * d1 + H[2, 0, 2] * d2
            a21 += H[2, 1, 0] * d0 + H[2, 1, 1] * d1 + H[2, 1, 2] * d2
            a22 += H[2, 2, 0] * d0 + H[2, 2, 1] * d1 + H[2, 2, 2] * d2
        sig[f, 0, 0] = a00; sig[f, 0, 1] = a01; sig[f, 0, 2] = a02
        sig[f, 1, 0] = a10; sig[f, 1, 1] = a11; sig[f, 1, 2] = a12
        sig[f, 2, 0] = a20; sig[f, 2, 1] = a21; sig[f, 2, 2] = a22
    return sig


# =====================================================================
# Anelastic (eigenstress) kernel of the mollified slip source.
#
# A mollified slip element is a smeared slip, i.e. an ANELASTIC
# (eigen-) strain
#
#     eps*_kl(x) = 1/2 (Du_k n_l + Du_l n_k) Phi_eps(x),
#     Phi_eps(x) = int int_T phi_eps(x - y) dA(y),
#
# with the Cortez blob phi_eps(r) = 15 eps^4 / (8 pi (r^2+eps^2)^(7/2)).
# Phi_eps is EXACTLY the seventh-order moment I7 already computed by
# `_stress_2d_moments` behind its need7 flag, so
#
#     Phi_eps = (15 eps^4 / 8 pi) * I7,
#
# and the eigenstress sigma* = C : eps* is
#
#     sigma*_mn = lam d_mn (n . Du) Phi_eps
#               + mu (Du_m n_n + Du_n n_m) Phi_eps
#               = H*[m,n,k] Du_k.
#
# This is the EXACT FINITE-TRIANGLE form: the blob is integrated over the
# actual triangle and summed over ALL elements. It replaces the
# infinite-plane / nearest-triangle approximation of the frozen
# `anelastic.py` (rho = 0.75 eps^4 / (d^2+eps^2)^2.5, one triangle per
# obs point), which is the d/L -> 0 limit of this integral: the two agree
# deep inside a large element and differ by up to ~2x near element edges
# (i.e. over the whole fault RIM). Parity oracles:
# moss/mollified_kernel/analytical_kernels.py::analytical_eigenstress_kernel,
# analytical_batch.py::eigenstress_batch, and clq.eigenstress; gate:
# verify/verify_eigenstress_exact.py.
#
# `eps` is per source element, like every other contraction driver here --
# nothing in the finite-triangle sum requires a uniform fault eps.
# =====================================================================


@njit(cache=True)
def _eigenstress_pair(tv, ex, ey, nhat, nrm, obs, eps, mu, lam,
                      Hout, M3, M5, M7):
    """Eigenstress H*[m,n,k] (sigma*_mn = H*[m,n,k] slip_k) for one pair."""
    if eps <= 0.0:
        for m in range(3):
            for n in range(3):
                for k in range(3):
                    Hout[m, n, k] = 0.0
        return
    I3, I5, I7, z = _stress_2d_moments(tv, ex, ey, nhat, obs, eps,
                                       M3, M5, M7, True)
    phi = (15.0 * eps * eps * eps * eps / (8.0 * np.pi)) * I7
    for m in range(3):
        for n in range(3):
            dmn = 1.0 if m == n else 0.0
            for k in range(3):
                dmk = 1.0 if m == k else 0.0
                dnk = 1.0 if n == k else 0.0
                Hout[m, n, k] = phi * (lam * dmn * nrm[k]
                                       + mu * (dmk * nrm[n] + dnk * nrm[m]))


@njit(cache=True, parallel=True)
def eigenstress_contract(x_field, tri_verts, normals, eps_arr, density,
                         mu, lam):
    """Eigenstress +C:eps* (N_f,3,3) of a triangulated SLIP source.

    sigma*(obs) = sum_s H*[obs,s][:,:,k] density[s,k], summed over ALL
    source elements (no nearest-triangle assignment). Parallel over obs
    points, per-element `eps_arr`; the twin of `dd_stress_contract`, whose
    divergent on-fault part this is. Returns +C:eps*, so
    sigma_elastic = dd_stress_contract(...) - eigenstress_contract(...).
    Material as (mu, lam), the only pair the eigenstress needs.
    """
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    sig = np.zeros((N_f, 3, 3))
    for f in prange(N_f):
        M3 = np.zeros((3, 3)); M5 = np.zeros((4, 4)); M7 = np.zeros((5, 5))
        H = np.empty((3, 3, 3))
        obs = x_field[f]
        a00 = 0.0; a01 = 0.0; a02 = 0.0
        a10 = 0.0; a11 = 0.0; a12 = 0.0
        a20 = 0.0; a21 = 0.0; a22 = 0.0
        for s in range(N_s):
            if not OK[s]:
                continue
            d0 = density[s, 0]; d1 = density[s, 1]; d2 = density[s, 2]
            if d0 == 0.0 and d1 == 0.0 and d2 == 0.0:
                continue
            _eigenstress_pair(tri_verts[s], EX[s], EY[s], NH[s], normals[s],
                              obs, eps_arr[s], mu, lam, H, M3, M5, M7)
            a00 += H[0, 0, 0] * d0 + H[0, 0, 1] * d1 + H[0, 0, 2] * d2
            a01 += H[0, 1, 0] * d0 + H[0, 1, 1] * d1 + H[0, 1, 2] * d2
            a02 += H[0, 2, 0] * d0 + H[0, 2, 1] * d1 + H[0, 2, 2] * d2
            a10 += H[1, 0, 0] * d0 + H[1, 0, 1] * d1 + H[1, 0, 2] * d2
            a11 += H[1, 1, 0] * d0 + H[1, 1, 1] * d1 + H[1, 1, 2] * d2
            a12 += H[1, 2, 0] * d0 + H[1, 2, 1] * d1 + H[1, 2, 2] * d2
            a20 += H[2, 0, 0] * d0 + H[2, 0, 1] * d1 + H[2, 0, 2] * d2
            a21 += H[2, 1, 0] * d0 + H[2, 1, 1] * d1 + H[2, 1, 2] * d2
            a22 += H[2, 2, 0] * d0 + H[2, 2, 1] * d1 + H[2, 2, 2] * d2
        sig[f, 0, 0] = a00; sig[f, 0, 1] = a01; sig[f, 0, 2] = a02
        sig[f, 1, 0] = a10; sig[f, 1, 1] = a11; sig[f, 1, 2] = a12
        sig[f, 2, 0] = a20; sig[f, 2, 1] = a21; sig[f, 2, 2] = a22
    return sig
