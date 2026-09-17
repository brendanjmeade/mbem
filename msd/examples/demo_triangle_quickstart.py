"""Quickstart: call the mollified full-space triangular solution directly.

Given ONE triangle, a slip vector, and observation points, evaluate the
displacement and the stress with the analytic mollified displacement-
discontinuity kernels -- no mesh, no BEM.  This is the minimal "easy calling"
example for the core triangular full-space solution.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mollified_kernel.analytical_batch import dd_displacement_batch
from mollified_kernel.analytical_kernels import (analytical_dd_displacement,
                                                 analytical_stress_kernel)


def main():
    # One planar triangle (km), its unit normal, and a uniform slip (km).
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)
    slip = np.array([1.0e-3, 0.0, 0.0])     # 1 m of strike-slip

    mu, nu, eps = 30.0, 0.25, 0.1           # GPa, Poisson 1/4, mollification km

    # (1) Displacement at a batch of observation points (vectorized).
    obs = np.array([[0.5, 0.3, 0.2],
                    [0.5, 0.3, -0.2],
                    [2.0, 0.0, 0.5]])
    U = dd_displacement_batch(v1, v2, v3, normal, obs, mu, nu, eps)   # (N,3,3)
    u = U @ slip                                                      # (N,3)
    print("displacement (mm) at obs points:")
    for p, ui in zip(obs, u * 1e6):
        print(f"  {p} ->  ({ui[0]:+.4f}, {ui[1]:+.4f}, {ui[2]:+.4f})")

    # (2) Same point, scalar API.
    u0 = analytical_dd_displacement(obs[0], v1, v2, v3, normal,
                                    mu, nu, eps) @ slip
    print(f"\nscalar API matches batch at obs[0]: {np.allclose(u0, u[0])}")

    # (3) Full stress tensor at a point: sigma_mn = H[m,n,k] * slip_k.
    H = analytical_stress_kernel(obs[0], v1, v2, v3, normal, mu, nu, eps)
    sigma = H @ slip                                                  # (3,3) GPa
    print("\nstress tensor (MPa) at obs[0]:")
    for row in sigma * 1e3:
        print("  [{:+8.3f} {:+8.3f} {:+8.3f}]".format(*row))
    print(f"\nsymmetric: {np.allclose(sigma, sigma.T)}")


if __name__ == "__main__":
    main()
