"""Demo: BEM epsilon-convergence at fixed mesh.

On one fixed fault-box triangulation, solve the mollified full-space BEM for a
ladder of mollification widths epsilon.  As epsilon decreases (while staying
above ~h/3 so the zone is resolved) the surface displacement converges; the
RMS difference from the finest-epsilon solution shrinks ~ O(epsilon^2).  This
is the BEM-level companion to the kernel-level eps-h decoupling demo.

Output (repo root): fig_bem_eps_convergence.png/.pdf
  (a) RMS surface-displacement difference from finest eps vs eps (log-log)
  (b) across-fault transect u_y(x) for each eps (curves converge)
"""
from __future__ import annotations

import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

import mollified_bem as mb                                       # noqa: E402
from mbem.cases.fault_box import build_fault_box, build_model             # noqa: E402
from mbem.backends.dense import AssembledDense                  # noqa: E402
from mbem.model import generate_system                          # noqa: E402

try:
    from _paper_style import set_paper_style
    set_paper_style()
except Exception:
    pass

EPS_LADDER = [12.0, 8.0, 6.0, 4.0, 3.0, 2.0]    # km, decreasing
SLIP_MAG = 0.01
MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)
KM_TO_MM = 1.0e6


def main():
    meshes = build_fault_box(edge_fault=6.0)     # eps down to 2 km stays > h/3
    model = build_model(meshes, SLIP_MAG, MAT)
    system = generate_system(model)
    print(f"unknowns: {system.layout.n_unknowns}; sweeping eps={EPS_LADDER}",
          flush=True)

    top_c = meshes["top"].centroids()
    sols = {}
    for eps in EPS_LADDER:
        t0 = time.time()
        asm = AssembledDense(system, eps, "direct", jump="calibrated")
        sols[eps] = asm.solve()["u:top"]
        print(f"  eps={eps:5.2f}: solved in {time.time()-t0:.0f} s", flush=True)

    eps_ref = min(EPS_LADDER)
    u_ref = sols[eps_ref]
    coarser = [e for e in EPS_LADDER if e > eps_ref]
    rms = [float(np.sqrt(np.mean((sols[e] - u_ref) ** 2))) * KM_TO_MM
           for e in coarser]
    print("\n  eps (km)   RMS |u - u(eps_min)| (mm)")
    for e, r in zip(coarser, rms):
        print(f"   {e:6.2f}     {r:.4g}")

    # across-fault transect u_y(x) along y=0
    xs = np.linspace(-120.0, 120.0, 200)
    line = np.column_stack([xs, np.zeros_like(xs)])
    transects = {e: griddata(top_c[:, :2], sols[e][:, 1], line,
                             method="linear") * KM_TO_MM for e in EPS_LADDER}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.8, 4.3))
    ax0.loglog(coarser, rms, "o-", color="#1f5fa6", lw=1.4, ms=5)
    e0 = np.array(coarser, float)
    ref2 = rms[0] * (e0 / e0[0]) ** 2
    ax0.loglog(e0, ref2, "--", color="0.55", lw=0.9)
    ax0.text(e0[-1], ref2[-1], r"$\propto \varepsilon^{2}$", color="0.4",
             ha="left", va="bottom", fontsize=8)
    ax0.set_xlabel(r"$\varepsilon$ (km)")
    ax0.set_ylabel(r"RMS $|u - u(\varepsilon_{\min})|$ (mm)")
    ax0.set_box_aspect(1)
    ax0.text(0.96, 0.96, "a", transform=ax0.transAxes, ha="right", va="top")

    cmap = plt.get_cmap("viridis")
    for i, e in enumerate(EPS_LADDER):
        ax1.plot(xs, transects[e], "-", lw=1.3,
                 color=cmap(i / (len(EPS_LADDER) - 1)),
                 label=rf"$\varepsilon={e:g}$")
    ax1.set_xlabel(r"$x$ (km, across fault)")
    ax1.set_ylabel(r"$u_y$ (mm) at $y=0$")
    ax1.set_xlim(-120, 120); ax1.set_xticks([-120, 0, 120])
    ax1.set_box_aspect(1)
    ax1.legend(frameon=False, fontsize=7, loc="upper right")
    ax1.text(0.04, 0.96, "b", transform=ax1.transAxes, ha="left", va="top")

    fig.suptitle(r"BEM $\varepsilon$-convergence at fixed mesh", fontsize=11,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        out = ROOT / f"fig_bem_eps_convergence.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
