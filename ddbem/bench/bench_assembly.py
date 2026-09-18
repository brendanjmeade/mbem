"""Assembly timings for the ddbem influence matrices, per order.

    python bench/bench_assembly.py                    # 512 and 2048 triangles
    python bench/bench_assembly.py --n-tri 512 --orders 0 1 --kinds displacement
    python bench/bench_assembly.py --json out.json

The model is the realistic one for a DD collocation solve: a flat square fault
cut into ``2 n^2`` triangles, collocated at the ELEMENT CENTROIDS (N_f = N_tri)
so the matrix stays square-ish at P0 and the per-order cost is comparable.
Node collocation for P1/P2 would multiply the field count by K as well, which
is a solver-stage decision; the cost per (field point, source element, order)
reported here is what that scales.

The eps budget is printed with every run, because fbem/FINDINGS.md sec.2
measured that it -- not the polynomial order -- is what decides whether
p-refinement pays.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ddbem                                                    # noqa: E402

MU, NU = 30.0, 0.25                                             # GPa, tree units


def square_fault(n, side=10.0, tilt=0.15):
    """(2 n^2, 3, 3) triangles of a tilted square patch, side in km."""
    x = np.linspace(0.0, side, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = tilt * X                                                # a slight tilt
    V = np.stack([X, Y, Z], axis=-1)
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([V[i, j], V[i + 1, j], V[i, j + 1]])
            tris.append([V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]])
    return np.ascontiguousarray(np.array(tris))


def run(n_tri_list, orders, kinds, eps_over_h):
    rows = []
    for n_tri in n_tri_list:
        n = int(round(np.sqrt(n_tri / 2)))
        tri = square_fault(n)
        assert tri.shape[0] == 2 * n * n
        obs = np.ascontiguousarray(tri.mean(axis=1))
        nf = ddbem.element_normals(tri)
        eps = float(eps_over_h) * float(ddbem.element_h(tri).mean())
        print(f"\n### N_tri = {tri.shape[0]}, N_field = {obs.shape[0]} "
              f"(centroids), mu = {MU} GPa, nu = {NU}")
        for p in orders:
            print("   ", ddbem.eps_report(tri, eps, p).__str__().replace("\n", "\n    "))
            for kind in kinds:
                t0 = time.perf_counter()
                if kind == "displacement":
                    A = ddbem.displacement_matrix(obs, tri, eps, MU, NU, p)
                elif kind == "traction":
                    A = ddbem.traction_matrix(obs, nf, tri, eps, MU, NU, p)
                elif kind == "stress":
                    A = ddbem.stress_matrix(obs, tri, eps, MU, NU, p)
                else:
                    raise ValueError(kind)
                dt = time.perf_counter() - t0
                gb = A.nbytes / 2**30
                pairs = obs.shape[0] * tri.shape[0]
                rows.append(dict(n_tri=int(tri.shape[0]), n_field=int(obs.shape[0]),
                                 order=int(p), kind=kind, seconds=dt,
                                 shape=list(A.shape), gib=gb,
                                 pairs_per_s=pairs / dt, eps=eps))
                print(f"      P{p} {kind:13s} {dt:8.2f} s   {A.shape[0]:6d} x "
                      f"{A.shape[1]:6d}  {gb:6.2f} GiB   "
                      f"{pairs / dt / 1e3:8.1f} k pairs/s")
                del A
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tri", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--orders", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--kinds", nargs="+", default=["displacement", "traction"])
    ap.add_argument("--eps-over-h", type=float, default=ddbem.defaults.EPS_OVER_H)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = run(a.n_tri, a.orders, a.kinds, a.eps_over_h)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
