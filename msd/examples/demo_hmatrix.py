"""Demo: block-compressed (H-matrix) iterative solve on the soft inclusion.

The mollified BEM operator is assembled in mbem's block-compressed form
(mbem.backends.HBackend: ACA low-rank far-field blocks + per-material bases)
and solved matrix-free with preconditioned FGMRES.  This solves the
fault + soft (mu/10) cylindrical-inclusion problem and checks that the
compressed FGMRES solution reproduces the dense LU reference.

Notes
-----
* HBackend uses the half-jump formulation, so the dense reference here also
  uses jump="half" (an apples-to-apples comparison of the SAME operator).
* eta=0.8 (stricter admissibility) routes near-field blocks to exact dense;
  this is required for correctness -- the looser default admissibility can
  accept inaccurate low-rank factors for a near-field block.
* The compressed operator stores a per-material basis per block (cheap material
  recombination); net memory compression vs a single-material dense matrix
  emerges at larger problem sizes, not at this ~11k-unknown demo size.

Output (repo root): fig_hmatrix.png/.pdf
  (a) surface u_x from the compressed FGMRES solve (host + inclusion tops)
  (b) dense LU vs H-matrix agreement, with iterations + error annotated
"""
from __future__ import annotations

import pathlib
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mollified_bem as mb                                       # noqa: E402
from assess_fig06_inclusion import build, build_model, EPS      # noqa: E402
from mbem.backends import AssembledDense, HBackend              # noqa: E402
from mbem.model import generate_system                          # noqa: E402

try:
    from _paper_style import set_paper_style
    set_paper_style()
except Exception:
    pass

MU_INC = 3.0          # mu/10 inclusion (host mu = 30)
KM_TO_MM = 1.0e6
INC_X, INC_Y, INC_R = -100.0, 100.0, 75.0


def main():
    meshes, fault, n_hat, s_hat = build()
    model = build_model(meshes, fault, s_hat,
                        mat_inc=mb.ElasticMaterial(mu=MU_INC, lam=MU_INC))
    system = generate_system(model)
    n = system.layout.n_unknowns
    print(f"fault + mu/10 inclusion: {n} unknowns, eps={EPS}", flush=True)

    t0 = time.time()
    dense = AssembledDense(system, EPS, "basis", jump="half").solve()
    print(f"dense LU (half jump): {time.time()-t0:.0f} s", flush=True)

    t0 = time.time()
    hasm = HBackend(eta=0.8, tol=1e-6, verbose=False).assemble(system, EPS)
    sol, report = hasm.solve(rtol=1e-8)
    t_h = time.time() - t0
    n_iter = getattr(report, "iterations", getattr(report, "n_iter", None))
    print(f"H-matrix FGMRES (eta=0.8): {t_h:.0f} s, {n_iter} iters", flush=True)

    worst = scale = 0.0
    for k in ("u:host_top", "u:inclusion_top"):
        worst = max(worst, float(np.max(np.abs(sol[k] - dense[k]))))
        scale = max(scale, float(np.max(np.abs(dense[k]))))
    rel_err = worst / scale
    print(f"  operator storage   : {hasm.nbytes()/1e9:.2f} GB "
          f"(dense single-material {(n*n*8)/1e9:.2f} GB)")
    print(f"  FGMRES iterations  : {n_iter}")
    print(f"  max rel diff vs dense LU: {rel_err:.2e}\n")

    # --- figure ---
    th = mtri.Triangulation(meshes["host_top"].vertices[:, 0],
                            meshes["host_top"].vertices[:, 1],
                            meshes["host_top"].triangles)
    ti = mtri.Triangulation(meshes["inclusion_top"].vertices[:, 0],
                            meshes["inclusion_top"].vertices[:, 1],
                            meshes["inclusion_top"].triangles)
    uh = sol["u:host_top"][:, 0] * KM_TO_MM
    ui = sol["u:inclusion_top"][:, 0] * KM_TO_MM
    vmax = max(float(np.percentile(np.abs(np.concatenate([uh, ui])), 99.0)), 1e-6)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 4.5))
    tcf = ax0.tripcolor(th, uh, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        shading="flat", rasterized=True)
    ax0.tripcolor(ti, ui, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  shading="flat", rasterized=True)
    th2 = np.linspace(0, 2 * np.pi, 200)
    ax0.plot([0, 0], [-100, 100], "k-", lw=0.8)
    ax0.plot(INC_X + INC_R * np.cos(th2), INC_Y + INC_R * np.sin(th2),
             "k--", lw=0.7)
    ax0.set_aspect("equal"); ax0.set_xlim(-200, 200); ax0.set_ylim(-200, 200)
    ax0.set_xticks([-200, 0, 200]); ax0.set_yticks([-200, 0, 200])
    ax0.set_xlabel(r"$x$ (km)"); ax0.set_ylabel(r"$y$ (km)")
    ax0.set_title(r"$u_x$ (mm), H-matrix FGMRES", fontsize=9)
    fig.colorbar(tcf, ax=ax0, fraction=0.045, pad=0.04, shrink=0.85)
    ax0.text(0.96, 0.96, "a", transform=ax0.transAxes, ha="right", va="top")

    d = np.concatenate([dense["u:host_top"].ravel(),
                        dense["u:inclusion_top"].ravel()]) * KM_TO_MM
    h = np.concatenate([sol["u:host_top"].ravel(),
                        sol["u:inclusion_top"].ravel()]) * KM_TO_MM
    lim = max(np.abs(d).max(), 1e-6)
    ax1.plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=0.8)
    ax1.plot(d, h, ".", color="#1f5fa6", ms=2, alpha=0.4)
    ax1.set_xlabel(r"dense LU  $u$ (mm)")
    ax1.set_ylabel(r"H-matrix FGMRES  $u$ (mm)")
    ax1.set_box_aspect(1)
    ax1.text(0.05, 0.95,
             f"{n} unknowns\nFGMRES iters {n_iter}\nmax rel diff {rel_err:.0e}",
             transform=ax1.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.5", lw=0.6))
    ax1.text(0.96, 0.96, "b", transform=ax1.transAxes, ha="right", va="top")

    fig.suptitle(r"Block-compressed FGMRES reproduces dense LU "
                 r"($\mu/10$ inclusion)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        out = ROOT / f"fig_hmatrix.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
