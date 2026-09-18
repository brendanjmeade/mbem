"""Where to collocate a P1/P2 element -- a sweep, not a guess.

A P1 node is a vertex and a P2 node is a vertex or an edge midpoint, so at
``shrink = 0`` every higher-order collocation point sits exactly ON the element
boundary: clearance 0, shared with two or more neighbours, and inside its own
element's smeared edge for any ``eps``.  ``ddbem.mesh.barycentric_nodes`` pulls
it toward the centroid,

    lam_c = (1 - t) lam_node + t / 3,

which is fbem's parametrisation (it swept the same ``t`` for a single-layer
density, found the error varied by more than 2x over the range, and chose 0.5).
The BASIS does not move -- only the collocation point -- so the free term stops
being the identity and becomes the shape-function matrix ``N_k(x_c)``, which is
the whole reason this file exists rather than a constant in ``defaults``.

The test problem is a Kelvin point force OUTSIDE a closed icosphere
(``verify/_exact.py`` explains why that is the right exact solution), run both
ways: prescribed displacement (displacement rows) and prescribed traction
(hypersingular rows).  Errors are relative max-norm of the interior field.

    python bench/sweep_collocation.py [--nsub 1] [--quick]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))

import ddbem                                                   # noqa: E402
from _exact import (best_fit_rigid, kelvin_displacement,       # noqa: E402
                    kelvin_stress, kelvin_traction)

MU, NU = 1.0, 0.30
X0 = np.array([3.0, 1.2, -0.7])
FORCE = np.array([1.0, -0.5, 0.8])
OBS = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1], [-0.15, 0.2, 0.1]])


def _voigt(s):
    return np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2],
                     s[:, 1, 2], s[:, 0, 2], s[:, 0, 1]], axis=1)


def run_case(tv, order, shrink, eps, bc):
    K = ddbem.n_nodes(order)
    if bc == "dirichlet":
        patch = ddbem.Patch("sph", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                            value=lambda p: kelvin_displacement(p, X0, FORCE, MU, NU),
                            eps=eps)
    else:
        nrm = np.repeat(ddbem.element_normals(tv), K, axis=0)
        patch = ddbem.Patch("sph", tv, ddbem.BCType.FREE_TRACTION,
                            value=lambda p: kelvin_traction(p, nrm, X0, FORCE, MU, NU),
                            eps=eps)
    m = ddbem.Model([patch], MU, NU, order=order, shrink=shrink,
                    jump="calibrated")
    sol = m.solve()
    u = sol.displacement(OBS)
    ue = kelvin_displacement(OBS, X0, FORCE, MU, NU)
    s = sol.stress(OBS)
    se = _voigt(kelvin_stress(OBS, X0, FORCE, MU, NU))
    if bc == "dirichlet":
        eu = np.max(np.abs(u - ue)) / np.max(np.abs(ue))
    else:                       # unique only up to a rigid motion
        eu = np.max(np.abs(best_fit_rigid(u - ue, OBS))) / np.max(np.abs(ue))
    es = np.max(np.abs(s - se)) / np.max(np.abs(se))
    clr = float(np.min(ddbem.node_clearance(tv, order, shrink)) / np.max(eps))
    return eu, es, sol.cond, clr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    v, t = ddbem.icosphere(a.nsub, radius=1.0)
    tv = ddbem.tri_verts(v, t)
    h = float(ddbem.element_h(tv).mean())
    shrinks = ([0.0, 0.25, 0.5] if a.quick
               else [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8])
    eps_over_h = [0.3] if a.quick else [0.3, 0.15]

    print(f"icosphere nsub={a.nsub}: {tv.shape[0]} triangles, h = {h:.4f}")
    print(f"{'bc':10s} {'P':>2s} {'eps/h':>6s} {'shrink':>7s} {'clr/eps':>8s} "
          f"{'u err':>10s} {'sig err':>10s} {'cond':>10s}")
    best = {}
    for bc in ("dirichlet", "neumann"):
        for eh in eps_over_h:
            eps = eh * h
            for order in (1, 2):
                for sh in shrinks:
                    eu, es, cond, clr = run_case(tv, order, sh, eps, bc)
                    print(f"{bc:10s} {order:2d} {eh:6.3f} {sh:7.2f} {clr:8.3f} "
                          f"{eu:10.3e} {es:10.3e} {cond:10.2e}")
                    key = (bc, eh, order)
                    if key not in best or eu < best[key][1]:
                        best[key] = (sh, eu)
                print()
    print("best shrink per case (by displacement error):")
    for k, (sh, eu) in best.items():
        print(f"  {k[0]:10s} eps/h={k[1]:.3f} P{k[2]}: shrink {sh:.2f}  err {eu:.3e}")


if __name__ == "__main__":
    main()
