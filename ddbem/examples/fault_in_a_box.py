"""Worked example: a prescribed-slip fault inside a traction-free box.

The canonical earthquake-mechanics model, and the smallest complete use of the
ddbem solver stack: build patches, budget eps, assemble, solve, evaluate.

    python examples/fault_in_a_box.py            # from the ddbem root
    python examples/fault_in_a_box.py --order 1 --n 2

Everything the model decides is printed rather than assumed: the eps/h budget
and node clearance (``../fbem/FINDINGS.md`` sec.2 -- eps is a floor on what any
basis can resolve, so a p-refinement result read without this number means
nothing), the row type, the rigid-body row-sum defect the calibration removes,
and the condition number.

NOTE ON COST.  ddbem is pure Python + numpy around ``clq.influence``; msd's
numba P0 path assembles the same matrix 470-930x faster.  Keep ``--n`` small.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ddbem                                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--n", type=int, default=2, help="box subdivisions per axis")
    ap.add_argument("--nu", type=float, default=0.25)
    ap.add_argument("--eps-over-h", type=float, default=0.3)
    ap.add_argument("--row", default="traction", choices=("traction", "exterior"))
    a = ap.parse_args()

    mu = 30.0                                  # GPa
    # 10 km x 10 km x 10 km box, a 4 km x 3 km fault in the middle of it.
    bv, bt = ddbem.box((10.0, 10.0, 10.0), (a.n, a.n, a.n))
    box = ddbem.tri_verts(bv, bt)
    fv, ft = ddbem.rectangle([-2.0, -1.5, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0],
                             2, 2)
    fault = ddbem.tri_verts(fv, ft)
    slip = np.array([0.001, 0.0, 0.0])         # 1 m of strike slip, in km

    eps_b = a.eps_over_h * ddbem.element_h(box)
    eps_f = a.eps_over_h * ddbem.element_h(fault)

    patches = [
        ddbem.Patch("box", box, ddbem.BCType.FREE_TRACTION, eps=eps_b),
        ddbem.Patch("fault", fault, ddbem.BCType.FAULT, value=slip, eps=eps_f,
                    order=0),
    ]
    model = ddbem.Model(patches, mu, a.nu, order=a.order, neumann_row=a.row,
                        jump="calibrated")
    print(model.report())

    t0 = time.perf_counter()
    system = model.assemble()
    t_asm = time.perf_counter() - t0
    print(f"\nassembled {system.A.shape[0]} x {system.A.shape[1]} in {t_asm:.1f} s")
    trace = np.abs(np.einsum("cii->c", system.free_term)) / 3.0
    print(f"  rigid-body row-sum defect BEFORE calibration: "
          f"{system.rigid_defect():.3e}")
    print(f"  free term |F|/3 range: {trace.min():.4f} .. {trace.max():.4f}")
    if a.row == "traction":
        print("    (identically 0 for a hypersingular row -- the exact "
              "eigenstress subtraction already annihilates a rigid translation;")
        print("     the row-sum defect above is the measurement of that, and it "
              "is machine zero.  A displacement row instead gets")
        print("     F = -sigma (1 - phi) I with phi the discrete smoothed "
              "indicator, where the classical jump asserts phi = 1/2.)")

    t0 = time.perf_counter()
    sol = model.solve(system)
    print(f"solved in {time.perf_counter() - t0:.1f} s: cond {sol.cond:.2e}, "
          f"residual {sol.residual:.2e}, {sol.n_constraints} rigid constraints")

    # a profile across the top face, one element size below it
    z = 5.0 - 0.6 * float(ddbem.element_h(box).mean())
    line = np.stack([np.linspace(-4.0, 4.0, 9), np.zeros(9), np.full(9, z)], axis=1)
    u = sol.displacement(line)
    s = sol.stress(line)
    print("\n  x (km)      ux (mm)      uy (mm)      uz (mm)    sxz (MPa)")
    for p, uu, ss in zip(line, u, s):
        print(f"  {p[0]:7.2f} {1e6 * uu[0]:12.4f} {1e6 * uu[1]:12.4f} "
              f"{1e6 * uu[2]:12.4f} {1e3 * ss[4]:12.4f}")
    print("\n(units km / GPa: 0.001 km = 1 m of slip; displacements printed in mm, "
          "stress in MPa)")
    print("The box is only 2.5 fault-lengths across, so these are NOT half-space "
          "surface displacements -- the far boundary is traction-free at 5 km.")


if __name__ == "__main__":
    main()
