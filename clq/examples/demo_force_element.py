"""The force (Kelvin single-layer) element: the same three calls as for slip.

Shows, for constant / linear / quadratic force density on one triangle:

* the displacement and stress of a force element, and that the stress needs no
  eigenstress subtraction (a body force is not an eigenstrain);
* global equilibrium -- a closed surface around the element carries exactly
  minus the total force applied to it;
* what is continuous and what jumps across the mollified layer (displacement
  continuous, traction jumping by -f(z0/eps) * density): the reason a single
  layer carries no 1/2 I free term in a boundary element system;
* the exact reciprocity ``U[i,j] = -n_m S[j,m,i]`` linking the force stress
  kernel to the dislocation displacement kernel;
* eps = 0 evaluated ON the element, which the dislocation kernels cannot do.

    python examples/demo_force_element.py        (from the clq root)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import clq  # noqa: E402
from clq.frame import local_frame  # noqa: E402
from clq.moments import gauss_triangle  # noqa: E402
from clq.shape import shape_functions  # noqa: E402

np.set_printoptions(precision=5, suppress=True, linewidth=120)

MU, NU = 1.0, 0.25


def total_force(tri, order, dens, n_gauss=20):
    """int_T f dS = sum_k f_k int_T N_k dS."""
    fr = local_frame(tri)
    x1, x2, w = gauss_triangle(n_gauss)
    y = ((1 - x1 - x2)[:, None] * tri[0] + x1[:, None] * tri[1] + x2[:, None] * tri[2])
    return ((w * 2.0 * fr.area) @ shape_functions(tri, order, y)) @ dens


def sphere(radius, center, n_theta=60, n_phi=120):
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    st = np.sqrt(1.0 - ct ** 2)
    ph = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    nrm = np.stack([np.outer(st, np.cos(ph)).ravel(),
                    np.outer(st, np.sin(ph)).ravel(),
                    np.repeat(ct, n_phi)], axis=1)
    w = np.repeat(wt, n_phi) * (2.0 * np.pi / n_phi) * radius ** 2
    return center + radius * nrm, nrm, w


def main():
    tri = clq.equilateral(L=1.0)
    fr = local_frame(tri)
    eps = 0.05
    obs = np.array([[0.10, 0.05, 0.00],      # on the element
                    [0.10, 0.05, 0.20],      # above
                    [0.80, -0.30, -0.10]])   # off the patch, below

    forces = {
        "constant  (K=1)": np.array([[0.0, 0.0, 1.0]]),
        "linear    (K=3)": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "quadratic (K=6)": np.array([[0.0, 0.0, 0.0]] * 3 + [[0.0, 0.0, 1.0]] * 3),
    }
    for name, dens in forces.items():
        p = clq.order_from_count(len(dens))
        u = clq.force_displacement(obs, tri, dens, MU, NU, eps)
        sig = clq.force_stress(obs, tri, dens, MU, NU, eps)
        print(f"\n=== force density {name}, force per unit area")
        print("u at obs:\n", u)
        print("sigma at obs[0] (on the element), already elastic:\n", sig[0])

        # global equilibrium: int_S sigma.n dS = -int_T f dS
        F = total_force(tri, p, dens)
        pts, nrm, w = sphere(4.0, tri.mean(0))
        net = np.einsum("q,qij,qj->i", w, clq.force_stress(pts, tri, dens, MU, NU, eps), nrm)
        print(f"total force on the element   int_T f dS = {F}")
        print(f"flux through a sphere R = 4L            = {net}   (= -int_T f dS)")

    # --- what jumps and what does not -------------------------------------
    big = clq.equilateral(L=40.0)
    fb = local_frame(big)
    dens = np.array([[0.7, -0.4, 0.3]])
    e, z0 = 1.0e-3, 3.0e-3
    x = big.mean(0)
    up = clq.force_displacement((x + z0 * fb.nhat)[None], big, dens, MU, NU, e)[0]
    um = clq.force_displacement((x - z0 * fb.nhat)[None], big, dens, MU, NU, e)[0]
    tp = clq.force_stress((x + z0 * fb.nhat)[None], big, dens, MU, NU, e)[0] @ fb.nhat
    tm = clq.force_stress((x - z0 * fb.nhat)[None], big, dens, MU, NU, e)[0] @ fb.nhat
    prof = lambda t: t * (2 * t * t + 3) / (2 * (1 + t * t) ** 1.5)
    print("\n=== across the mollified layer (z0 = 3 eps)")
    print(f"displacement jump  = {up - um}   (continuous: no 1/2 I free term)")
    print(f"traction jump      = {tp - tm}")
    print(f"-f(z0/eps) * dens  = {-prof(z0 / e) * dens[0]}")

    # --- reciprocity with the dislocation kernel ---------------------------
    inf = clq.influence(obs, tri, MU, NU, eps, order=2, want=("U", "S"))
    lhs = inf.U
    rhs = -np.einsum("m,nkjmi->nkij", fr.nhat, inf.S)
    print("\n=== reciprocity  U[i,j] = -n_m S[j,m,i]")
    print(f"max |U + n.S| / max|U| = {np.max(np.abs(lhs - rhs)) / np.max(np.abs(lhs)):.2e}")

    # --- eps = 0 on the element --------------------------------------------
    print("\n=== eps = 0 evaluated ON the element (single layer is integrable)")
    on = np.vstack([tri.mean(0)[None, :], tri[0][None, :], 0.5 * (tri[1] + tri[2])[None, :]])
    u0 = clq.force_displacement(on, tri, forces["quadratic (K=6)"], MU, NU, 0.0)
    for label, row in zip(("centroid", "vertex v1", "edge midpoint"), u0):
        print(f"  u({label:14s}) = {row}")
    try:
        clq.displacement(on, tri, np.array([[1.0, 0, 0]]), MU, NU, 0.0)
    except ValueError as exc:
        print(f"  the dislocation kernel there still raises: {str(exc)[:60]}...")


if __name__ == "__main__":
    main()
