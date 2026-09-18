"""Convergence of the DD collocation solver in h, in p, and in eps.

This study exists because of ``../fbem/FINDINGS.md`` sec.2, which is the one
result from the dropped force-element work that transfers unchanged:

> eps is not only a regularisation.  It is a FLOOR on the resolvable structure
> of the unknown.  No basis can represent detail below eps.

There, holding ``eps = 0.3 h`` made P0, P1 and P2 indistinguishable (+0.31,
+0.32, +0.28 in h) while holding eps FIXED in absolute terms inverted the
ordering and P1 beat P0 by 1.37x.  So this script sweeps BOTH:

* ``--mode ratio``  : eps = (eps/h) * h, refined together -- the usual choice,
                      and the one that hides whether p-refinement pays;
* ``--mode absolute``: eps fixed in km while h shrinks, so eps/h GROWS with
                      refinement and the mollification floor is what is being
                      measured.

Test problem: a Kelvin point force outside a closed icosphere
(``verify/_exact.py`` for why that is the right exact solution), solved both as
a Dirichlet problem (displacement rows) and as a Neumann problem
(hypersingular rows).  The reported rate is a least-squares fit of
``log(err)`` on ``log(h)`` over the mesh ladder, and also on
``log(n_unknowns)`` -- at matched unknown count is the only fair comparison
between orders, since discontinuous P2 costs 18 unknowns per triangle against
P0's 3.

    python bench/study_convergence.py --mode ratio
    python bench/study_convergence.py --mode absolute --orders 0 1 2
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

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


def solve_case(tv, order, eps, bc, jump):
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
    m = ddbem.Model([patch], MU, NU, order=order, jump=jump)
    t0 = time.perf_counter()
    sol = m.solve()
    dt = time.perf_counter() - t0
    u = sol.displacement(OBS)
    ue = kelvin_displacement(OBS, X0, FORCE, MU, NU)
    s = sol.stress(OBS)
    se = _voigt(kelvin_stress(OBS, X0, FORCE, MU, NU))
    eu = (np.max(np.abs(u - ue)) if bc == "dirichlet"
          else np.max(np.abs(best_fit_rigid(u - ue, OBS)))) / np.max(np.abs(ue))
    es = np.max(np.abs(s - se)) / np.max(np.abs(se))
    return eu, es, sol.system.n_dof, sol.cond, dt


def rate(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("ratio", "absolute"), default="ratio")
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eps", type=float, nargs="+", default=None,
                    help="eps/h values (ratio mode) or absolute eps (absolute mode)")
    ap.add_argument("--jump", default="calibrated")
    ap.add_argument("--bc", nargs="+", default=["dirichlet", "neumann"])
    ap.add_argument("--max-dof", type=int, default=6000)
    a = ap.parse_args()

    eps_list = a.eps or ([0.5, 0.3, 0.15] if a.mode == "ratio" else [0.25, 0.12])
    meshes = []
    for lv in a.levels:
        v, t = ddbem.icosphere(lv, radius=1.0)
        tv = ddbem.tri_verts(v, t)
        meshes.append((tv, float(ddbem.element_h(tv).mean())))

    print(f"mode = {a.mode}, jump = {a.jump}, "
          f"meshes = {[m[0].shape[0] for m in meshes]} triangles, "
          f"h = {[round(m[1], 4) for m in meshes]}")
    for bc in a.bc:
        for ev in eps_list:
            print(f"\n--- {bc}, "
                  f"{'eps/h' if a.mode == 'ratio' else 'eps'} = {ev:g} ---")
            print(f"{'P':>2s} {'N_tri':>6s} {'dof':>6s} {'h':>8s} {'eps/h':>7s} "
                  f"{'clr/eps':>8s} {'u err':>10s} {'sig err':>10s} "
                  f"{'cond':>9s} {'t (s)':>7s}")
            for order in a.orders:
                hs, us, ss, ns = [], [], [], []
                for tv, h in meshes:
                    eps = ev * h if a.mode == "ratio" else ev
                    K = ddbem.n_nodes(order)
                    if 3 * K * tv.shape[0] > a.max_dof:
                        print(f"{order:2d} {tv.shape[0]:6d} "
                              f"{3 * K * tv.shape[0]:6d}   skipped (> --max-dof)")
                        continue
                    sh = ddbem.defaults.COLLOCATION_SHRINK_BY_ORDER[order]
                    clr = float(np.min(ddbem.node_clearance(tv, order, sh)) / eps)
                    eu, es, dof, cond, dt = solve_case(tv, order, eps, bc, a.jump)
                    print(f"{order:2d} {tv.shape[0]:6d} {dof:6d} {h:8.4f} "
                          f"{eps / h:7.3f} {clr:8.3f} {eu:10.3e} {es:10.3e} "
                          f"{cond:9.2e} {dt:7.1f}")
                    hs.append(h); us.append(eu); ss.append(es); ns.append(dof)
                if len(hs) >= 2:
                    print(f"   P{order} rate in h: u {rate(hs, us):+.2f}  "
                          f"sig {rate(hs, ss):+.2f}   |   per unknown: "
                          f"u {rate(ns, us):+.2f}  sig {rate(ns, ss):+.2f}")


if __name__ == "__main__":
    main()
