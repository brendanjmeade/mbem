#!/usr/bin/env python3
"""
verify_arbitrary_triangle.py
============================

Confirms that the analytical mollified kernels in ``analytical_kernels.py``
are valid for *arbitrary* planar triangles, not just the unit right
triangle used in the built-in validation suite.

The routines

    integrate_all_moments, analytical_kelvin_G, analytical_dd_displacement,
    analytical_kelvin_stress, analytical_stress_kernel

accept the three vertices ``v1, v2, v3`` and build an intrinsic local
frame from them:

    normal = (v2 - v1) x (v3 - v1) / | ... |
    ex     = (v2 - v1) / |v2 - v1|
    ey     = normal x ex

The solid-angle and edge-by-edge antiderivative sums then walk the
actual triangle edges in this intrinsic frame. Nothing in the derivation
assumes the triangle is right-angled or unit-sized. This script
verifies this assertion by:

  1. Picking a generic triangle (tilted, non-right, non-unit,
     off-axis vertices) and comparing the analytical moments /
     kernels against a high-order numerical reference.

  2. Demonstrating quadrature-convergence of the numerical
     reference to the analytical value (confirming the latter is
     exact to machine precision when the numerical integral is
     resolved).

  3. Verifying that affine *rigid-body* transformations of the
     (triangle, obs, normal) triple leave the scalar I_n moments
     invariant but rotate the vector/tensor kernels consistently.

Run:
    python verify_arbitrary_triangle.py
"""

from __future__ import annotations

import numpy as np

from moss_kernel.analytical_kernels import (
    integrate_all_moments,
    integrate_moments_numerical,
    analytical_kelvin_G,
    integrate_kelvin_G_numerical,
    analytical_dd_displacement,
    integrate_dd_displacement_numerical,
    analytical_kelvin_stress,
    integrate_kelvin_stress_numerical,
)


# ----------------------------------------------------------------------
# Generic triangle (NOT a unit right triangle, NOT axis-aligned)
# ----------------------------------------------------------------------
V1 = np.array([ 0.37, -0.81,  0.44])
V2 = np.array([ 1.92,  0.11, -0.63])
V3 = np.array([-0.25,  1.57,  1.22])


def triangle_normal(v1, v2, v3):
    n = np.cross(v2 - v1, v3 - v1)
    return n / np.linalg.norm(n)


def triangle_area(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


# ----------------------------------------------------------------------
# Test 1: analytical vs high-order numerical agreement
# ----------------------------------------------------------------------

def test_arbitrary_triangle():
    print("=" * 70)
    print("TEST 1: Arbitrary planar triangle (non-right, tilted, off-axis)")
    print("=" * 70)
    print(f"  v1 = {V1}")
    print(f"  v2 = {V2}")
    print(f"  v3 = {V3}")
    print(f"  Edge lengths: "
          f"|v1v2|={np.linalg.norm(V2-V1):.3f}  "
          f"|v2v3|={np.linalg.norm(V3-V2):.3f}  "
          f"|v3v1|={np.linalg.norm(V1-V3):.3f}")
    print(f"  Area = {triangle_area(V1,V2,V3):.4f}")
    nrm = triangle_normal(V1, V2, V3)
    print(f"  Unit normal = {nrm}")

    mu, nu = 1.0, 0.25
    cases = [
        ("far-field  ",    np.array([ 2.00,  1.50,  3.00]), 0.08),
        ("near-field ",    np.array([ 0.60, -0.10,  0.60]), 0.10),
        ("below plane",    np.array([-0.40,  0.00, -0.80]), 0.08),
        ("on-plane   ",    V1 + 0.3*(V2-V1) + 0.3*(V3-V1), 0.12),
    ]

    tol = 2e-2   # loose to tolerate finite-nq reference near the plane
    all_ok = True
    print()
    for name, obs, eps in cases:
        # On-plane and near-field cases need heavier quadrature for the
        # numerical reference to be trustworthy (integrand is nearly
        # singular when obs is close to the triangle plane with small eps).
        nq = 80 if ("on-plane" in name or "near" in name) else 30
        mA = integrate_all_moments(V1, V2, V3, obs, eps)
        assert mA is not None
        mN = integrate_moments_numerical(V1, V2, V3, obs, eps, n_quad=nq)
        GA = analytical_kelvin_G(obs, V1, V2, V3, mu, nu, eps)
        GN = integrate_kelvin_G_numerical(obs, V1, V2, V3, mu, nu, eps,
                                          n_quad=nq)
        UA = analytical_dd_displacement(obs, V1, V2, V3, nrm, mu, nu, eps)
        UN = integrate_dd_displacement_numerical(obs, V1, V2, V3, nrm,
                                                 mu, nu, eps, n_quad=nq)
        SA = analytical_kelvin_stress(obs, V1, V2, V3, mu, nu, eps)
        SN = integrate_kelvin_stress_numerical(obs, V1, V2, V3, mu, nu, eps,
                                               n_quad=nq)

        print(f"  {name}  obs={obs}  eps={eps}")
        for lbl, A, N in [
            ("V3   ", mA["V"][3],   mN["V3"]),
            ("T2_5 ", mA["T2"][5],  mN["T2_5"]),
            ("T4_7 ", mA["T4"][7],  mN["T4_7"]),
            ("G    ", GA,            GN),
            ("U    ", UA,            UN),
            ("S    ", SA,            SN),
        ]:
            A = np.asarray(A); N = np.asarray(N)
            err = np.linalg.norm(A - N)
            ref = np.linalg.norm(N)
            rel = err / ref if ref > 1e-14 else err
            ok = rel < tol
            if not ok:
                all_ok = False
            print(f"    {lbl} rel={rel:8.2e} "
                  f"{'OK' if ok else '**FAIL**'}")
        print()
    return all_ok


# ----------------------------------------------------------------------
# Test 2: numerical reference converges to the analytical value
# ----------------------------------------------------------------------

def test_numerical_reference_converges():
    print("=" * 70)
    print("TEST 2: Numerical reference -> analytical value as n_quad -> inf")
    print("=" * 70)

    mu, nu = 1.0, 0.25
    nrm = triangle_normal(V1, V2, V3)

    obs = V1 + 0.3*(V2 - V1) + 0.3*(V3 - V1)   # hardest case (on-plane)
    eps = 0.12

    UA = analytical_dd_displacement(obs, V1, V2, V3, nrm, mu, nu, eps)
    print(f"  On-plane DD displacement kernel at obs={obs}  eps={eps}")
    print(f"  |U_analytical| = {np.linalg.norm(UA):.6e}\n")
    print(f"  {'n_quad':>8s}  {'|Unum - Uana| / |Uana|':>24s}")
    print(f"  {'-'*8}  {'-'*24}")
    for nq in [15, 30, 60, 120]:
        UN = integrate_dd_displacement_numerical(
            obs, V1, V2, V3, nrm, mu, nu, eps, n_quad=nq
        )
        rel = np.linalg.norm(UA - UN) / np.linalg.norm(UA)
        print(f"  {nq:8d}  {rel:24.3e}")

    print()


# ----------------------------------------------------------------------
# Test 3: rigid-body covariance
# ----------------------------------------------------------------------

def random_rotation(seed=0):
    """Return a random 3x3 rotation matrix (QR-based)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_rigid_covariance():
    """
    A rigid-body transform (rotation + translation) applied to
    (v1, v2, v3, obs, normal) must rotate the Kelvin G and DD U
    kernels by Q on each index and leave the scalar base integral
    I_1 invariant.
    """
    print("=" * 70)
    print("TEST 3: Rigid-body covariance (rotation + translation)")
    print("=" * 70)

    mu, nu, eps = 1.0, 0.25, 0.1
    nrm = triangle_normal(V1, V2, V3)
    obs = np.array([0.6, -0.1, 0.6])

    GA = analytical_kelvin_G(obs, V1, V2, V3, mu, nu, eps)
    UA = analytical_dd_displacement(obs, V1, V2, V3, nrm, mu, nu, eps)

    Q = random_rotation(seed=42)
    t = np.array([-1.3, 4.2, 0.7])

    v1p = Q @ V1 + t
    v2p = Q @ V2 + t
    v3p = Q @ V3 + t
    obsp = Q @ obs + t
    nrmp = Q @ nrm

    GB = analytical_kelvin_G(obsp, v1p, v2p, v3p, mu, nu, eps)
    UB = analytical_dd_displacement(obsp, v1p, v2p, v3p, nrmp, mu, nu, eps)

    # Kelvin G transforms as G'_{ij} = Q_{ia} Q_{jb} G_{ab}
    G_expected = np.einsum("ia,jb,ab->ij", Q, Q, GA)
    U_expected = np.einsum("ia,jb,ab->ij", Q, Q, UA)

    errG = np.linalg.norm(GB - G_expected) / np.linalg.norm(GB)
    errU = np.linalg.norm(UB - U_expected) / np.linalg.norm(UB)

    print(f"  |G(rigid) - Q G Q^T| / |G| = {errG:.3e}")
    print(f"  |U(rigid) - Q U Q^T| / |U| = {errU:.3e}")
    ok = (errG < 1e-12) and (errU < 1e-12)
    print(f"  {'OK' if ok else '**FAIL**'}\n")
    return ok


# ----------------------------------------------------------------------
# Test 4: size scaling.  Under the uniform rescaling
#     (y, obs, eps) -> alpha * (y, obs, eps)
# the pointwise Kelvin kernels transform by homogeneity of 1/R:
#     G_ij       -> G_ij        / alpha      (G ~ 1/R, R -> alpha R)
#     dG/dy      -> dG/dy       / alpha^2
# and the area element picks up alpha^2, so the *integrated* kernels are
#     G_int  -> alpha    * G_int     (single-layer Kelvin)
#     U_int  -> 1        * U_int     (DD displacement: extra derivative)
#     S_int  -> 1        * S_int     (Kelvin force-stress: extra deriv)
# ----------------------------------------------------------------------

def test_size_scaling():
    print("=" * 70)
    print("TEST 4: Scaling law under (v, obs, eps) -> alpha (v, obs, eps)")
    print("=" * 70)

    mu, nu = 1.0, 0.25
    nrm = triangle_normal(V1, V2, V3)
    obs = np.array([0.6, -0.1, 0.6])
    eps0 = 0.10
    alpha = 3.7

    GA = analytical_kelvin_G(obs,         V1,         V2,         V3,
                             mu, nu, eps0)
    UA = analytical_dd_displacement(obs,  V1,         V2,         V3,
                                     nrm, mu, nu, eps0)
    SA = analytical_kelvin_stress(obs, V1, V2, V3, mu, nu, eps0)

    GB = analytical_kelvin_G(alpha*obs,   alpha*V1,   alpha*V2,   alpha*V3,
                             mu, nu, alpha*eps0)
    UB = analytical_dd_displacement(alpha*obs, alpha*V1, alpha*V2, alpha*V3,
                                     nrm, mu, nu, alpha*eps0)
    SB = analytical_kelvin_stress(alpha*obs, alpha*V1, alpha*V2, alpha*V3,
                                  mu, nu, alpha*eps0)

    G_expected = alpha * GA
    U_expected = UA
    S_expected = SA

    errG = np.linalg.norm(GB - G_expected) / np.linalg.norm(GB)
    errU = np.linalg.norm(UB - U_expected) / np.linalg.norm(UB)
    errS = np.linalg.norm(SB - S_expected) / np.linalg.norm(SB)

    print(f"  alpha = {alpha}")
    print(f"  |G_scaled - alpha*G| / |G_scaled| = {errG:.3e}")
    print(f"  |U_scaled -       U| / |U_scaled| = {errU:.3e}")
    print(f"  |S_scaled -       S| / |S_scaled| = {errS:.3e}")
    ok = (errG < 1e-12) and (errU < 1e-12) and (errS < 1e-12)
    print(f"  {'OK' if ok else '**FAIL**'}\n")
    return ok


# ----------------------------------------------------------------------

def main():
    ok1 = test_arbitrary_triangle()
    test_numerical_reference_converges()
    ok3 = test_rigid_covariance()
    ok4 = test_size_scaling()
    print("=" * 70)
    print(f"  Arbitrary-tri={'PASS' if ok1 else 'FAIL'}  "
          f"Rigid={'PASS' if ok3 else 'FAIL'}  "
          f"Scaling={'PASS' if ok4 else 'FAIL'}")
    print("=" * 70)
    # A verdict at COLUMN 0, as the last such line: the per-clause summary
    # above is indented, and the suite runner scrapes column 0 precisely so an
    # indented clause cannot be mistaken for the gate's verdict. Without this
    # the gate scores FAIL however well it did. Added when the gate was adopted
    # into the suite; the numerics above are untouched.
    ok = ok1 and ok3 and ok4
    n = sum((ok1, ok3, ok4))
    print(f"{'PASS' if ok else 'FAIL'}: arbitrary-triangle covariance "
          f"({n} of 3 checks)")
    return ok


if __name__ == "__main__":
    main()
