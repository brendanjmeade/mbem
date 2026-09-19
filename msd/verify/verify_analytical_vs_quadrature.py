"""
Verify the corrected analytical-integration functions match high-order
quadrature of the corrected pointwise kernels.

This catches mistakes in the new Cortez-blob terms added to:
  - integrate_D2G        (uses new I5 and T2[7] moment contributions)
  - integrate_DG         (uses new V[5] moment contribution)
  - analytical_kelvin_G  (uses new I3 contribution)
"""

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mollified_kernel"))

from mollified_elastic_kernels import kelvin_d2G
from analytical_kernels import (
    analytical_kelvin_G,
    integrate_D2G,
    integrate_DG,
)


def gauss_triangle_pts(n):
    """n x n product Gauss rule mapped to the unit triangle (0,0),(1,0),(0,1)."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts01 = 0.5 * (pts + 1.0)
    wts01 = 0.5 * wts
    xi = []
    eta = []
    w = []
    for i in range(n):
        for j in range(n):
            xi1 = pts01[i]
            xi2 = pts01[j] * (1.0 - pts01[i])
            ww = wts01[i] * wts01[j] * (1.0 - pts01[i])
            xi.append(xi1)
            eta.append(xi2)
            w.append(ww)
    return np.array(xi), np.array(eta), np.array(w)


def numerical_integrate_pointwise(v1, v2, v3, obs, mu, nu, eps, n=24, kernel="d2g"):
    xi, eta, w = gauss_triangle_pts(n)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    out_d2g = np.zeros((3, 3, 3, 3))
    out_dg = np.zeros((3, 3, 3))
    out_g = np.zeros((3, 3))
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    c_blob = 2.0 * (1.0 - nu) * eps**2

    for k in range(len(w)):
        x = (1 - xi[k] - eta[k]) * v1 + xi[k] * v2 + eta[k] * v3
        d_vec = obs - x
        ww = w[k] * area2

        # Pointwise G_ij at this quadrature point (Galerkin/Cortez form)
        r2 = float(np.dot(d_vec, d_vec))
        re = np.sqrt(r2 + eps**2)
        g_ij = C1 * (
            c34 * np.eye(3) / re
            + np.outer(d_vec, d_vec) / re**3
            + c_blob * np.eye(3) / re**3
        )
        out_g += ww * g_ij

        # Pointwise D2G (uses corrected kelvin_d2G)
        out_d2g += ww * kelvin_d2G(d_vec, mu, nu, eps)

        # Pointwise DG = ∂G/∂x_m
        # = C1 * [-c34 δ_ij d_m/r^3 + δ_im d_j/r^3 + δ_jm d_i/r^3 - 3 d_i d_j d_m/r^5
        #         - 6(1-ν) ε² δ_ij d_m/r^5]
        re3 = re**3
        re5 = re**5
        dg = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for m in range(3):
                    val = (
                        -c34 * (i == j) * d_vec[m] / re3
                        + (i == m) * d_vec[j] / re3
                        + (j == m) * d_vec[i] / re3
                        - 3.0 * d_vec[i] * d_vec[j] * d_vec[m] / re5
                        - 3.0 * c_blob * (i == j) * d_vec[m] / re5
                    )
                    dg[i, j, m] = C1 * val
        out_dg += ww * dg

    return out_g, out_dg, out_d2g


def compare(label, A, B):
    diff = A - B
    max_abs = np.max(np.abs(diff))
    max_ref = np.max(np.abs(B))
    rel = max_abs / max(max_ref, 1e-30)
    print(f"  {label:30s} max|diff|={max_abs:.3e}  max|ref|={max_ref:.3e}  rel={rel:.3e}")
    return rel


def main():
    rng = np.random.default_rng(seed=42)

    # A general (non-axis-aligned, non-equilateral) triangle
    v1 = np.array([0.1, -0.2, 0.0])
    v2 = np.array([1.3, 0.0, 0.0])
    v3 = np.array([0.6, 1.1, 0.05])

    # Observation point off the plane (so quadrature is well-defined and converges fast)
    obs = np.array([0.5, 0.4, 0.7])

    mu = 1.0
    nu = 0.25
    eps = 0.3

    # Analytical
    G_a = analytical_kelvin_G(obs, v1, v2, v3, mu, nu, eps)
    DG_a = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    D2G_a = integrate_D2G(v1, v2, v3, obs, mu, nu, eps)

    # Numerical (high order)
    G_n, DG_n, D2G_n = numerical_integrate_pointwise(v1, v2, v3, obs, mu, nu, eps, n=24)

    print("Analytical vs 24-point Gauss quadrature")
    print(f"  obs = {obs}, eps = {eps}, nu = {nu}")
    print()
    rel_g  = compare("analytical_kelvin_G", G_a, G_n)
    rel_dg = compare("integrate_DG       ", DG_a, DG_n)
    rel_d2g = compare("integrate_D2G      ", D2G_a, D2G_n)

    ok = max(rel_g, rel_dg, rel_d2g) < 1e-8
    if ok:
        print("\nPASS: analytical and numerical integrations agree.")
    else:
        print("\nFAIL: analytical and numerical disagree above 1e-8.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
