"""Gauss-quadrature assembly of the (singular) point kernels — the way
a CEDT BEM is assembled in practice when no analytic per-element
integral is available.

Provides drop-in replacements for
``mollified_kernel.analytical_batch.assemble_U_matrix_batch`` and
``assemble_T_matrix_batch`` that integrate the POINTWISE kernels with a
fixed-order collapsed tensor Gauss rule instead of the closed-form
per-triangle integrals.  The pointwise integrands follow the same
conventions (and the same Cortez-mollified forms, so eps > 0 also
works; the CEDT experiment uses eps = 0, where they reduce to the
classical singular kernels).

Used by fig12's system-level failure experiment: monkey-patch the two
assembly functions, solve the box BEM, restore.
"""

from __future__ import annotations

import numpy as np

from moss_kernel.analytical_batch import _place_blocks
from moss_kernel.mollified_elastic_kernels import triangle_quadrature


def _kelvin_G_point(d, mu, nu, eps):
    """Pointwise Kelvin displacement Green function, (N, 3, 3).

    Same integrand as the analytic path (kelvin_G_batch):
        G_ij = C1 [ (3-4nu) delta_ij / R + d_i d_j / R^3
                    + 2 (1-nu) eps^2 delta_ij / R^3 ].
    """
    R2 = np.sum(d * d, axis=-1) + eps * eps
    R = np.sqrt(R2)
    R3 = R2 * R
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps * eps
    eye3 = np.eye(3)
    diag = (c34 / R + c_blob / R3)
    G = diag[..., None, None] * eye3
    G += np.einsum("...i,...j->...ij", d, d) / R3[..., None, None]
    return C1 * G


def _dd_U_point(d, normal, mu, nu, eps):
    """Pointwise DD displacement kernel, (N, 3, 3): u_i = U[i, j] dU_j.

    Built from the same Kelvin-gradient form as dd_displacement_batch:
        dG[i,k,m] = C1 [ -c34 d_ik d_m/R^3 + d_im d_k/R^3
                          + d_km d_i/R^3 - 3 d_i d_k d_m/R^5
                          - 6(1-nu) eps^2 d_ik d_m/R^5 ]
    contracted with (lam, mu, n) exactly as in the analytic path.
    """
    R2 = np.sum(d * d, axis=-1) + eps * eps
    R3 = R2 * np.sqrt(R2)
    R5 = R2 * R3
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 6.0 * (1.0 - nu) * eps * eps
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    eye3 = np.eye(3)

    v3 = d / R3[..., None]
    v5 = d / R5[..., None]
    t35 = (np.einsum("...i,...k,...m->...ikm", d, d, d)
           / R5[..., None, None, None])
    dG = -c34 * np.einsum("ik,...m->...ikm", eye3, v3)
    dG += np.einsum("im,...k->...ikm", eye3, v3)
    dG += np.einsum("km,...i->...ikm", eye3, v3)
    dG += -3.0 * t35
    dG += -c_blob * np.einsum("ik,...m->...ikm", eye3, v5)
    dG *= C1

    n = normal
    # Traction-operator pairing (slip and normal in C's first index pair),
    # as in analytical_kernels.analytical_dd_displacement (lam/mu placement
    # fixed 2026-09-17; the swap is invisible at nu = 1/4).
    trace_im = np.einsum("...imm->...i", dG)
    term1 = mu * np.einsum("m,...ijm->...ij", n, dG)
    term2 = lam * np.einsum("j,...i->...ij", n, trace_im)
    term3 = mu * np.einsum("k,...ikj->...ij", n, dG)
    return -(term1 + term2 + term3)


def _quad_block(point_kernel, v1, v2, v3, obs, n_q):
    """Integrate a pointwise kernel over one triangle by the collapsed
    tensor Gauss rule (same rule as integrate_stress_kernel)."""
    xi1, xi2, wts = triangle_quadrature(n_q)
    nodes = ((1.0 - xi1 - xi2)[:, None] * v1
             + xi1[:, None] * v2 + xi2[:, None] * v3)     # (Q, 3)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    d = obs[:, None, :] - nodes[None, :, :]               # (N, Q, 3)
    K = point_kernel(d)                                    # (N, Q, 3, 3)
    return area2 * np.einsum("q,nqij->nij", wts, K)


def make_quad_assemblers(n_q):
    """Return (assemble_U, assemble_T) with the analytic-path
    signatures, integrating by fixed-order Gauss quadrature."""

    def assemble_U(x_field, tri_verts, mu, nu, eps):
        N_s = tri_verts.shape[0]
        blocks = np.empty((N_s, x_field.shape[0], 3, 3))
        for s in range(N_s):
            blocks[s] = _quad_block(
                lambda d: _kelvin_G_point(d, mu, nu, eps),
                tri_verts[s, 0], tri_verts[s, 1], tri_verts[s, 2],
                x_field, n_q)
        return _place_blocks(blocks)

    def assemble_T(x_field, tri_verts, normals_source, mu, nu, eps):
        N_s = tri_verts.shape[0]
        blocks = np.empty((N_s, x_field.shape[0], 3, 3))
        for s in range(N_s):
            blocks[s] = _quad_block(
                lambda d: _dd_U_point(d, normals_source[s], mu, nu, eps),
                tri_verts[s, 0], tri_verts[s, 1], tri_verts[s, 2],
                x_field, n_q)
        return _place_blocks(blocks)

    return assemble_U, assemble_T


if __name__ == "__main__":
    # Sanity: far pair -> quadrature must agree with the analytic
    # integrals; near pair at eps = 0 -> it must not.
    from moss_kernel.analytical_batch import (
        assemble_U_matrix_batch, assemble_T_matrix_batch)

    rng = np.random.default_rng(0)
    tri_far = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]]])
    normals = np.array([[0.0, 0.0, 1.0]])
    obs_far = np.array([[5.0, 4.0, 3.0], [-6.0, 2.0, 1.0]])
    obs_near = np.array([[0.55, 0.55, 0.05], [0.4, 0.4, 0.02]])

    aU, aT = make_quad_assemblers(8)
    for eps in (0.0, 0.3):
        for name, quad, exact in (
                ("U", aU, assemble_U_matrix_batch),
                ("T", lambda x, t, m, n, e: aT(x, t, normals, m, n, e),
                 lambda x, t, m, n, e: assemble_T_matrix_batch(
                     x, t, normals, m, n, e))):
            q = quad(obs_far, tri_far, 30.0, 0.25, eps)
            a = exact(obs_far, tri_far, 30.0, 0.25, eps)
            rel = np.abs(q - a).max() / np.abs(a).max()
            print(f"far  {name} eps={eps}: rel err {rel:.2e}")
            q = quad(obs_near, tri_far, 30.0, 0.25, eps)
            a = exact(obs_near, tri_far, 30.0, 0.25, eps)
            rel = np.abs(q - a).max() / np.abs(a).max()
            print(f"near {name} eps={eps}: rel err {rel:.2e}")
