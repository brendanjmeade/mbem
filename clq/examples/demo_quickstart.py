"""Quickstart: the same three calls for constant, linear and quadratic slip.

    python examples/demo_quickstart.py        (from the clq root)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

import clq  # noqa: E402

np.set_printoptions(precision=5, suppress=True, linewidth=120)


def main():
    tri = clq.equilateral(L=1.0)                 # (3,3) vertices, nhat = +z
    mu, nu, eps = 1.0, 0.25, 0.05
    obs = np.array([[0.10, 0.05, 0.00],          # on the fault (mid-plane)
                    [0.10, 0.05, 0.20],          # above
                    [0.80, -0.30, -0.10]])       # off the patch, below

    slips = {
        "constant  (K=1)": np.array([[1.0, 0.0, 0.0]]),
        "linear    (K=3)": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "quadratic (K=6)": np.array([[0.0, 0.0, 0.0]] * 3 + [[1.0, 0.0, 0.0]] * 3),
    }
    for name, slip in slips.items():
        p = clq.order_from_count(len(slip))
        print(f"\n=== {name}: nodes at\n{clq.nodes(tri, p)}")
        u = clq.displacement(obs, tri, slip, mu, nu, eps)
        sig_el = clq.stress(obs, tri, slip, mu, nu, eps)                          # elastic (default)
        sig_tot = clq.stress(obs, tri, slip, mu, nu, eps, subtract_eigenstress=False)
        eig = clq.eigenstress(obs, tri, slip, mu, nu, eps)
        print("u at obs:\n", u)
        print("elastic sigma at obs[0] (on-fault):\n", sig_el[0])
        print("total   sigma at obs[0] (on-fault):\n", sig_tot[0])
        print("C:eps*  at obs[0]        :\n", eig[0])
        print("traction (elastic) on the fault plane at obs[0]:", clq.traction(sig_el, tri)[0])
        inf = clq.influence(obs, tri, mu, nu, eps, order=p)
        print("influence shapes: U", inf.U.shape, "H", inf.H.shape, "E", inf.E.shape)

    # a linear slip expressed on the quadratic nodes gives the same field
    s1 = np.array([[1.0, 0.2, 0.0], [0.3, 0.0, 0.0], [0.0, -0.4, 0.0]])
    s2 = clq.interpolate(tri, s1, clq.nodes(tri, 2))
    d = np.abs(clq.displacement(obs, tri, s2, mu, nu, eps) - clq.displacement(obs, tri, s1, mu, nu, eps)).max()
    print(f"\nlinear slip embedded in the quadratic basis: max |difference| = {d:.1e}")

    # nodal values from a callable
    s_fn = clq.nodal_values(tri, 2, lambda pts: np.column_stack([1.0 - pts[:, 0] ** 2, 0 * pts[:, 0], 0 * pts[:, 0]]))
    print("nodal_values of s_x = 1 - x^2 on the quadratic nodes:", s_fn[:, 0])

    # physical units (km, GPa): 1 m of strike-slip on a 10 km triangle, eps = 1 km
    tri_km = clq.equilateral(L=10.0)
    obs_km = np.array([[0.5, 0.0, 0.0]])
    sig = clq.stress(obs_km, tri_km, np.array([[1.0e-3, 0.0, 0.0]]), 30.0, 0.25, 1.0)
    print(f"\nkm/GPa call: on-fault elastic sigma_xz = {sig[0, 0, 2] * 1e3:+.3f} MPa for 1 m slip, "
          f"L = 10 km, eps = 1 km, mu = 30 GPa")


if __name__ == "__main__":
    main()
