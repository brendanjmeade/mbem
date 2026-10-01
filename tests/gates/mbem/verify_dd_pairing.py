"""verify_dd_pairing.py -- lambda/mu index pairing of the DD displacement kernel.

The slip -> displacement (T / double-layer) kernel is the traction operator
applied to the Kelvin solution: slip and normal share the FIRST index pair
of C, giving (analytical_kernels.analytical_dd_displacement)

    U_ij = -[ mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j ].

A lam/mu swap on the first two terms is invisible at nu = 1/4 (lam == mu),
where most gates run, and ~20-60 % off at nu = 0.3.  This gate runs at
several nu and uses only convention-free or independent references:

  (a) Gauss closure: for a closed cube with outward normals and a uniform
      unit slip, the integrated kernel summed over the faces is -I at interior
      points and 0 outside -- scalar, batch, and the mbem T basis / contraction;
  (b) the repo's own classical traction kernel mollified_bem.kelvin_T_mollified
      (which always had the correct pairing) integrated by Gauss quadrature
      over a triangle equals analytical_dd_displacement at eps = 0, nu = 0.3;
  (c) Hooke consistency: finite-difference gradient of the displacement kernel
      -> C:sym(grad u) equals analytical_stress_kernel (whose pairing was
      always correct) at nu = 0.3;
  (d) msd scalar == moss scalar at nu = 0.3 (a missing moss copy FAILS, so
      the parity is never silently skipped); batch == scalar;
      mbem assemble_t_matrix == batch, all at nu = 0.3;
  (e) the point kernel the triangle integrals are built from satisfies the
      regularized Navier equation L_ij G_jk + delta_ik phi_eps = 0 with the
      Cortez blob phi_eps = 15 eps^4 / (8 pi R_eps^7): kelvin_d2G directly and
      a 5-point finite difference of kelvin_dG_pointwise, at several nu.  The
      blob term 2(1-nu) eps^2 delta_ij / R_eps^3 in G is what makes this hold;
      a 5 % error in its coefficient leaves a residual of 3e-2 or more.

Run from the repo root:  python verify/verify_dd_pairing.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import mollified_bem as mb                                          # noqa: E402
from mollified_kernel.analytical_kernels import (                   # noqa: E402
    analytical_dd_displacement, analytical_stress_kernel, kelvin_dG_pointwise)
from mollified_kernel.analytical_batch import dd_displacement_batch  # noqa: E402
from mollified_kernel.mollified_elastic_kernels import (             # noqa: E402
    kelvin_d2G, triangle_quadrature)
from mbem.kernels import basis as kb                                 # noqa: E402
from mbem.kernels import tri_kernels as tk                           # noqa: E402

CHECKS = []


def check(name, val, tol):
    ok = bool(val < tol)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:58s} {val:9.2e} (tol {tol:.0e})")
    return ok


def _prefix_swapped_U(obs, v1, v2, v3, normal, mu, nu, eps):
    """The swapped contraction (lam on the normal-derivative term, mu on the
    trace term), rebuilt from integrate_DG for the tripwire."""
    from mollified_kernel.analytical_kernels import integrate_DG
    G1 = integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    U = np.zeros((3, 3))
    for i in range(3):
        tr = sum(G1[i, m, m] for m in range(3))
        for j in range(3):
            U[i, j] = -(lam * sum(n[m] * G1[i, j, m] for m in range(3))
                        + mu * n[j] * tr
                        + mu * sum(n[k] * G1[i, k, j] for k in range(3)))
    return U


def navier_residual(dG, d2G, mu, nu, eps, d, h_over_eps=0.005):
    """Relative residual of L_ij G_jk + delta_ik phi_eps at the point d, from
    the analytic second derivative d2G and from a 5-point central difference
    of the first derivative dG (step h_over_eps * eps), normalised by the
    largest term of the operator.  Returns (analytic, finite_difference)."""
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    phi = 15.0 * eps**4 / (8.0 * np.pi * (d @ d + eps**2) ** 3.5)
    hh = h_over_eps * eps
    D2fd = np.zeros((3, 3, 3, 3))                    # D2[i,j,s,q] = d2 G_ij / dx_s dx_q
    for q in range(3):
        e = np.zeros(3); e[q] = hh
        D2fd[:, :, :, q] = (-dG(d + 2 * e, mu, nu, eps) + 8 * dG(d + e, mu, nu, eps)
                            - 8 * dG(d - e, mu, nu, eps) + dG(d - 2 * e, mu, nu, eps)) / (12 * hh)
    out = []
    for D2 in (d2G(d, mu, nu, eps), D2fd):
        lap = mu * np.einsum("ikpp->ik", D2)
        div = (lam + mu) * np.einsum("jkij->ik", D2)
        res = lap + div + phi * np.eye(3)
        out.append(np.abs(res).max() / max(np.abs(lap).max(), np.abs(div).max(), phi))
    return out


def cube_mesh():
    v = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    # 12 triangles, orientation fixed below
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in faces:
        tris += [[a, b, c], [a, c, d]]
    mesh = mb.TriMesh(vertices=v, triangles=np.array(tris))
    mesh.ensure_outward_normals(np.zeros(3))
    return mesh


def closure(mesh, points, mu, nu, eps):
    """Sum over triangles of the integrated DD displacement kernel (N,3,3):
    scalar, batch, mbem basis matrix, mbem contraction."""
    tv = mesh.vertices[mesh.triangles]
    normals, _ = mesh.normals_and_areas()
    N = points.shape[0]
    S_scalar = np.zeros((N, 3, 3))
    S_batch = np.zeros((N, 3, 3))
    for s in range(tv.shape[0]):
        v1, v2, v3 = tv[s]
        for i, o in enumerate(points):
            S_scalar[i] += analytical_dd_displacement(o, v1, v2, v3, normals[s], mu, nu, eps)
        S_batch += dd_displacement_batch(v1, v2, v3, normals[s], points, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    mat = mb.ElasticMaterial(mu=mu, lam=lam)
    T = kb.assemble_t_matrix(points, mesh, mat, eps)                 # (3N, 3Ns)
    S_mbem = T.reshape(N, 3, tv.shape[0], 3).sum(axis=2)
    S_contract = np.zeros((N, 3, 3))
    c = kb.t_coeffs(mu, lam)
    eps_arr = np.full(tv.shape[0], eps)
    for j in range(3):
        dens = np.zeros((tv.shape[0], 3))
        dens[:, j] = 1.0
        S_contract[:, :, j] = tk.t_disp_contract(np.ascontiguousarray(points), np.ascontiguousarray(tv),
                                                 np.ascontiguousarray(normals), eps_arr, dens, *c)
    return S_scalar, S_batch, S_mbem, S_contract


def main():
    print("=" * 76)
    print("DD displacement kernel: lambda/mu pairing gates")
    print("=" * 76)
    mu = 1.0
    eye = np.eye(3)

    # (a) closure on a closed cube --------------------------------------
    mesh = cube_mesh()
    inside = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.4], [-0.5, 0.1, -0.3]])
    outside = np.array([[2.0, 0.5, -0.3], [-0.2, -2.5, 1.0]])
    eps = 0.02
    print("\n[a] closed-cube closure: sum over faces of the DD kernel = -I inside, 0 outside")
    for nu in (0.10, 0.25, 0.30, 0.45):
        Si = closure(mesh, inside, mu, nu, eps)
        So = closure(mesh, outside, mu, nu, eps)
        for lbl, Sin, Sout in zip(("scalar", "batch", "mbem basis", "mbem contract"), Si, So):
            e_in = np.max(np.abs(Sin + eye[None]))
            e_out = np.max(np.abs(Sout))
            check(f"nu={nu:.2f} {lbl}: |sum U + I| inside", e_in, 1e-4)
            check(f"nu={nu:.2f} {lbl}: |sum U| outside", e_out, 1e-4)

    # (b) legacy classical traction kernel, Gauss-integrated -------------
    print("\n[b] Gauss quadrature of mollified_bem.kelvin_T_mollified(y - x, n) at eps = 0")
    v1 = np.array([0.1, -0.2, 0.0]); v2 = np.array([1.3, 0.0, 0.0]); v3 = np.array([0.6, 1.1, 0.05])
    n = np.cross(v2 - v1, v3 - v1); n /= np.linalg.norm(n)
    xi1, xi2, w = triangle_quadrature(30)
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    y = (1 - xi1 - xi2)[:, None] * v1 + xi1[:, None] * v2 + xi2[:, None] * v3
    for nu in (0.25, 0.30, 0.45):
        for obs in (np.array([0.5, 0.4, 0.7]), np.array([-0.3, 0.9, -0.6])):
            U = analytical_dd_displacement(obs, v1, v2, v3, n, mu, nu, 0.0)
            Tq = np.einsum("q,qij->ij", w * area2, mb.kelvin_T_mollified(y - obs, n, mu, nu, 0.0))
            check(f"nu={nu:.2f} obs={obs}: analytic vs quad(T)", np.abs(U - Tq).max() / np.abs(U).max(), 1e-8)
            # the PRE-FIX contraction (lam/mu swapped on the first two terms) must NOT match
            Uswap = _prefix_swapped_U(obs, v1, v2, v3, n, mu, nu, 0.0)
            swapped = np.abs(Uswap - Tq).max() / np.abs(U).max()
            if nu != 0.25:
                CHECKS.append(swapped > 1e-2)
                print(f"  [{'ok' if swapped > 1e-2 else 'XX'}] nu={nu:.2f}: pre-fix swapped form differs by {swapped:.2e} (> 1e-2)")
            else:
                check(f"nu={nu:.2f}: pre-fix swapped form coincides (lam == mu)", swapped, 1e-12)

    # (c) FD-Hooke consistency with the (always correct) stress kernel ---
    print("\n[c] Hooke consistency: C:sym(grad u) from FD of the displacement kernel vs stress kernel")
    eps = 0.3
    for nu in (0.25, 0.30):
        lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
        for obs in (np.array([0.5, 0.4, 0.7]), np.array([0.9, 0.2, -0.5])):
            H = analytical_stress_kernel(obs, v1, v2, v3, n, mu, nu, eps)      # (3,3,3) sigma_mn per slip k
            grad = np.zeros((3, 3, 3))                                          # grad[i, b, j] = d u_i / d x_b per slip j
            for b in range(3):
                for hh, wt in ((1e-3 * eps, -1.0 / 3.0), (5e-4 * eps, 4.0 / 3.0)):
                    d = np.zeros(3); d[b] = hh
                    Up = analytical_dd_displacement(obs + d, v1, v2, v3, n, mu, nu, eps)
                    Um = analytical_dd_displacement(obs - d, v1, v2, v3, n, mu, nu, eps)
                    grad[:, b, :] += wt * (Up - Um) / (2 * hh)
            strain = 0.5 * (grad + np.swapaxes(grad, 0, 1))
            sig = lam * np.einsum("aaj->j", strain)[None, None, :] * eye[:, :, None] + 2 * mu * strain
            check(f"nu={nu:.2f} obs={obs}: FD-Hooke vs stress kernel", np.abs(sig - H).max() / np.abs(H).max(), 1e-7)

    # (d) parity: msd scalar == moss scalar; batch == scalar; mbem == batch at nu = 0.3
    print("\n[d] parity at nu = 0.3")
    # moss's copy is an installed package (moss_kernel), not a file path, so
    # this no longer requires msd and moss to be sibling directories. The
    # provenance assert is the point: the two copies are independent
    # implementations and this clause compares them, so resolving to msd's own
    # would compare a copy against itself and still pass.
    try:
        import moss_kernel.analytical_kernels as moss
        assert pathlib.Path(moss.__file__).parent.name == "moss_kernel", \
            moss.__file__
        moss_why = None
    except (ImportError, AssertionError) as exc:
        moss, moss_why = None, exc
    nu = 0.30
    obs_pts = np.array([[0.5, 0.4, 0.7], [-0.3, 0.9, -0.6], [2.0, 1.0, 1.5]])
    Us = np.array([analytical_dd_displacement(o, v1, v2, v3, n, mu, nu, eps) for o in obs_pts])
    Ub = dd_displacement_batch(v1, v2, v3, n, obs_pts, mu, nu, eps)
    check("batch vs scalar", np.abs(Ub - Us).max() / np.abs(Us).max(), 1e-12)
    if moss is not None:
        Um = np.array([moss.analytical_dd_displacement(o, v1, v2, v3, n, mu, nu, eps) for o in obs_pts])
        check("msd scalar vs moss scalar", np.abs(Um - Us).max() / np.abs(Us).max(), 1e-12)
    else:
        CHECKS.append(False)
        print(f"  [XX] moss oracle unavailable ({moss_why}) -- parity check counted as FAILED")
    mesh1 = mb.TriMesh(vertices=np.array([v1, v2, v3]), triangles=np.array([[0, 1, 2]]))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    T1 = kb.assemble_t_matrix(obs_pts, mesh1, mb.ElasticMaterial(mu=mu, lam=lam), eps)
    Umb = T1.reshape(len(obs_pts), 3, 3)
    check("mbem assemble_t_matrix vs batch", np.abs(Umb - Ub).max() / np.abs(Ub).max(), 1e-12)

    # (e) the point kernel satisfies the regularized Navier equation ---------
    print("\n[e] Navier residual L_ij G_jk + delta_ik phi_eps of the mollified point kernel")
    eps = 0.3
    for nu in (0.25, 0.30, 0.45):
        for d in (np.array([0.2, 0.1, -0.15]), np.array([0.5, 0.4, 0.7]), np.array([-0.3, 0.9, -0.6])):
            r_an, r_fd = navier_residual(kelvin_dG_pointwise, kelvin_d2G, mu, nu, eps, d)
            check(f"nu={nu:.2f} |d|/eps={np.linalg.norm(d) / eps:.1f}: kelvin_d2G analytic", r_an, 1e-12)
            check(f"nu={nu:.2f} |d|/eps={np.linalg.norm(d) / eps:.1f}: 5-pt FD of kelvin_dG_pointwise", r_fd, 1e-7)

    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: DD displacement kernel pairing ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: DD displacement kernel pairing ({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
