"""verify_dd_pairing.py -- lambda/mu index pairing of the DD displacement kernel.

The slip -> displacement (T / double-layer) kernel is the traction operator
applied to the Kelvin solution: slip and normal share the FIRST index pair
of C, giving (analytical_kernels.analytical_dd_displacement)

    U_ij = -[ mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j ].

The scalar kernel has had this form since commit f721a6a (2026-06-12), but
the batch kernel (the T matrix of every BEM assembly), the numerical
reference integrator, the manuscript quadrature assembler and the Apostol
half-space comparator (mindlin_triangle) kept lam and mu swapped until
2026-09-17.  The swap is invisible at nu = 1/4 (lam == mu), where every other
check runs, and ~20-60 % off at nu = 0.3.  Adapted from msd's gate of the
same name; every check runs at nu != 1/4 against convention-free or
independent references:

  (a) Gauss closure: for a closed cube with outward normals and a uniform
      unit slip, the integrated kernel summed over the faces is -I at
      interior points and 0 outside (scalar and batch);
  (b) the classical Kelvin traction kernel (kelvin_T below, textbook form,
      independent of the DD code) integrated by Gauss quadrature equals
      analytical_dd_displacement at eps = 0;
  (c) Hooke consistency: C:sym(grad u) from finite differences of the
      displacement kernel equals analytical_stress_kernel;
  (d) batch, numerical integrator and manuscript quadrature assembler agree
      with the scalar kernel;
  (e) mindlin_triangle (Apostol comparator): far below the free surface the
      half-space DD kernel and its gradient reduce to the full-space ones.

Run from this directory:  python verify_dd_pairing.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from moss_kernel.analytical_kernels import (                   # noqa: E402
    analytical_dd_displacement, analytical_stress_kernel, integrate_DG,
    integrate_dd_displacement_numerical)
from moss_kernel.analytical_batch import dd_displacement_batch  # noqa: E402
from moss_kernel.mollified_elastic_kernels import triangle_quadrature  # noqa: E402
from moss_kernel.mindlin_triangle import (                     # noqa: E402
    integrate_mindlin_dd_kernel, integrate_mindlin_dd_grad_kernel)
from _quad_assembly import _dd_U_point            # noqa: E402

CHECKS = []


def check(name, val, tol):
    ok = bool(val < tol)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:62s} {val:9.2e} (tol {tol:.0e})")
    return ok


def lame(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def prefix_swapped_U(obs, v1, v2, v3, normal, mu, nu, eps):
    """The pre-fix contraction (lam on the normal-derivative term, mu on the
    trace term), rebuilt from integrate_DG as a tripwire."""
    G1 = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = lame(mu, nu)
    U = np.zeros((3, 3))
    for i in range(3):
        tr = sum(G1[i, m, m] for m in range(3))
        for j in range(3):
            U[i, j] = -(lam * sum(normal[m] * G1[i, j, m] for m in range(3))
                        + mu * normal[j] * tr
                        + mu * sum(normal[k] * G1[i, k, j] for k in range(3)))
    return U


def kelvin_T(r, n, nu):
    """Classical Kelvin traction kernel T_ij(r, n) in textbook form (check b)."""
    d = np.linalg.norm(r, axis=1)
    e, a = r / d[:, None], 1.0 - 2.0 * nu
    T = ((e @ n)[:, None, None] * (a * np.eye(3) + 3.0 * e[:, :, None] * e[:, None, :])
         + a * (n[None, :, None] * e[:, None, :] - e[:, :, None] * n[None, None, :]))
    return -T / (8.0 * np.pi * (1.0 - nu) * d[:, None, None] ** 2)


def cube_mesh():
    """Closed cube [-1, 1]^3 as 12 triangles with outward right-hand normals:
    (vertex triples (12, 3, 3), unit normals (12, 3))."""
    v = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0)
                  for z in (-1.0, 1.0)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6),
             (0, 2, 6, 4), (1, 5, 7, 3)]
    tv = v[np.array([t for a, b, c, d in faces for t in ([a, b, c], [a, c, d])])]
    normals = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    flip = np.einsum("ij,ij->i", normals, tv.mean(axis=1)) < 0
    tv[flip], normals[flip] = tv[flip][:, [0, 2, 1]], -normals[flip]
    return tv, normals / np.linalg.norm(normals, axis=1)[:, None]


def closure(mesh, points, mu, nu, eps):
    tv, normals = mesh
    S_scalar = np.zeros((points.shape[0], 3, 3))
    S_batch = np.zeros_like(S_scalar)
    for s in range(tv.shape[0]):
        v1, v2, v3 = tv[s]
        for i, o in enumerate(points):
            S_scalar[i] += analytical_dd_displacement(o, v1, v2, v3, normals[s],
                                                      mu, nu, eps)
        S_batch += dd_displacement_batch(v1, v2, v3, normals[s], points,
                                         mu, nu, eps)
    return S_scalar, S_batch


def main():
    print("=" * 80)
    print("moss DD displacement kernel: lambda/mu pairing gates")
    print("=" * 80)
    mu = 1.0
    eye = np.eye(3)
    v1 = np.array([0.1, -0.2, 0.0])
    v2 = np.array([1.3, 0.0, 0.0])
    v3 = np.array([0.6, 1.1, 0.05])
    n = np.cross(v2 - v1, v3 - v1)
    n /= np.linalg.norm(n)

    print("\n[a] closed-cube closure: sum over faces of the DD kernel = -I inside, 0 outside")
    mesh = cube_mesh()
    inside = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.4], [-0.5, 0.1, -0.3]])
    outside = np.array([[2.0, 0.5, -0.3], [-0.2, -2.5, 1.0]])
    for nu in (0.10, 0.25, 0.30, 0.45):
        Si = closure(mesh, inside, mu, nu, 0.02)
        So = closure(mesh, outside, mu, nu, 0.02)
        for lbl, Sin, Sout in zip(("scalar", "batch"), Si, So):
            check(f"nu={nu:.2f} {lbl}: |sum U + I| inside",
                  np.abs(Sin + eye[None]).max(), 1e-4)
            check(f"nu={nu:.2f} {lbl}: |sum U| outside", np.abs(Sout).max(), 1e-4)

    print("\n[b] Gauss quadrature of the classical Kelvin traction kernel T(y - x, n) at eps = 0")
    xi1, xi2, w = triangle_quadrature(30)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    y = (1 - xi1 - xi2)[:, None] * v1 + xi1[:, None] * v2 + xi2[:, None] * v3
    for nu in (0.25, 0.30, 0.45):
        for obs in (np.array([0.5, 0.4, 0.7]), np.array([-0.3, 0.9, -0.6])):
            U = analytical_dd_displacement(obs, v1, v2, v3, n, mu, nu, 0.0)
            Tq = np.einsum("q,qij->ij", w * area2,
                           kelvin_T(y - obs, n, nu))
            check(f"nu={nu:.2f} obs={obs}: analytic vs quad(T)",
                  np.abs(U - Tq).max() / np.abs(U).max(), 1e-8)
            Us = prefix_swapped_U(obs, v1, v2, v3, n, mu, nu, 0.0)
            swapped = np.abs(Us - Tq).max() / np.abs(U).max()
            if nu != 0.25:
                ok = swapped > 1e-2
                CHECKS.append(ok)
                print(f"  [{'ok' if ok else 'XX'}] nu={nu:.2f}: pre-fix swapped form "
                      f"differs by {swapped:.2e} (> 1e-2)")
            else:
                check(f"nu={nu:.2f}: pre-fix swapped form coincides (lam == mu)",
                      swapped, 1e-12)

    print("\n[c] Hooke consistency: C:sym(grad u) from FD of the displacement kernel vs stress kernel")
    eps = 0.3
    for nu in (0.25, 0.30):
        lam = lame(mu, nu)
        for obs in (np.array([0.5, 0.4, 0.7]), np.array([0.9, 0.2, -0.5])):
            H = analytical_stress_kernel(obs, v1, v2, v3, n, mu, nu, eps)
            grad = np.zeros((3, 3, 3))
            for b in range(3):
                for hh, wt in ((1e-3 * eps, -1.0 / 3.0), (5e-4 * eps, 4.0 / 3.0)):
                    d = np.zeros(3)
                    d[b] = hh
                    Up = analytical_dd_displacement(obs + d, v1, v2, v3, n, mu, nu, eps)
                    Um = analytical_dd_displacement(obs - d, v1, v2, v3, n, mu, nu, eps)
                    grad[:, b, :] += wt * (Up - Um) / (2 * hh)
            strain = 0.5 * (grad + np.swapaxes(grad, 0, 1))
            sig = (lam * np.einsum("aaj->j", strain)[None, None, :] * eye[:, :, None]
                   + 2 * mu * strain)
            check(f"nu={nu:.2f} obs={obs}: FD-Hooke vs stress kernel",
                  np.abs(sig - H).max() / np.abs(H).max(), 1e-7)

    print("\n[d] batch / numerical integrator / manuscript quadrature vs scalar")
    obs_pts = np.array([[0.5, 0.4, 0.7], [-0.3, 0.9, -0.6], [2.0, 1.0, 1.5]])
    for nu in (0.30, 0.45):
        Us = np.array([analytical_dd_displacement(o, v1, v2, v3, n, mu, nu, eps)
                       for o in obs_pts])
        scale = np.abs(Us).max()
        Ub = dd_displacement_batch(v1, v2, v3, n, obs_pts, mu, nu, eps)
        check(f"nu={nu:.2f} batch vs scalar", np.abs(Ub - Us).max() / scale, 1e-12)
        Un = np.array([integrate_dd_displacement_numerical(o, v1, v2, v3, n, mu, nu,
                                                           eps, n_quad=30)
                       for o in obs_pts])
        check(f"nu={nu:.2f} integrate_dd_displacement_numerical vs scalar",
              np.abs(Un - Us).max() / scale, 1e-8)
        Uq = np.array([np.einsum("q,qij->ij", w * area2,
                                 _dd_U_point(o - y, n, mu, nu, eps))
                       for o in obs_pts])
        check(f"nu={nu:.2f} manuscript _quad_assembly point kernel vs scalar",
              np.abs(Uq - Us).max() / scale, 1e-8)

    print("\n[e] mindlin_triangle far below the free surface -> full-space kernels")
    depth = 2.0e3
    shift = np.array([0.0, 0.0, -depth])
    w1, w2, w3 = v1 + shift, v2 + shift, v3 + shift
    for nu in (0.25, 0.30):
        lam = lame(mu, nu)
        for obs in (np.array([0.5, 0.4, 1.5]), np.array([0.9, 0.2, -1.2])):
            o = obs + shift
            Ufs = analytical_dd_displacement(obs, v1, v2, v3, n, mu, nu, 0.1)
            Uhs = integrate_mindlin_dd_kernel(o, w1, w2, w3, n, mu, nu, 0.1, n_quad=24)
            check(f"nu={nu:.2f} obs={obs}: mindlin DD kernel vs full space",
                  np.abs(Uhs - Ufs).max() / np.abs(Ufs).max(), 1e-5)
            Hfs = analytical_stress_kernel(obs, v1, v2, v3, n, mu, nu, 0.1)
            S = integrate_mindlin_dd_grad_kernel(o, w1, w2, w3, n, mu, nu, 0.1, n_quad=24)
            strain = 0.5 * (S + np.swapaxes(S, 0, 1))
            sig = (lam * np.einsum("aaj->j", strain)[None, None, :] * eye[:, :, None]
                   + 2 * mu * strain)
            check(f"nu={nu:.2f} obs={obs}: mindlin grad kernel (Hooke) vs full space",
                  np.abs(sig - Hfs).max() / np.abs(Hfs).max(), 1e-5)

    print("-" * 80)
    if all(CHECKS):
        print(f"PASS: DD displacement kernel pairing ({len(CHECKS)} checks)")
    else:
        bad = sum(1 for c in CHECKS if not c)
        print(f"FAIL: DD displacement kernel pairing ({bad} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
