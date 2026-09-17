"""Demo: fault-only BEM in a homogeneous full-space box.

A vertical right-lateral strike-slip fault inside a Cartesian box (free top,
traction-free sides, clamped base) is solved with the mollified full-space BEM
(mbem, calibrated jump).  We plot the free-surface displacement and the
ELASTIC surface stress -- the anelastic (eigenstrain) term is subtracted
(anelastic.py) so the near-fault stress is the genuine elastic field, not the
spurious 1/eps zone-loading band.

Output (repo root): fig_fault_only.png/.pdf
  row 1: surface displacement u_x, u_y, u_z (mm)
  row 2: elastic surface stress sigma_xx, sigma_yy, sigma_xy (MPa)
"""
from __future__ import annotations

import pathlib
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import griddata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mollified_bem as mb                                         # noqa: E402
from _fault_box import build_fault_box, build_model               # noqa: E402
from anelastic import eigenstress_at_points                       # noqa: E402
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.model import generate_system                            # noqa: E402

try:
    from _paper_style import set_paper_style
    set_paper_style()
except Exception:
    pass

EPS = 3.0                       # km
SLIP_MAG = 0.01                 # km == 10 m
MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)    # nu = 1/4
KM_TO_MM = 1.0e6
GPA_TO_MPA = 1.0e3
VIEW = 100.0                    # km plotting half-window
GRID_N = 161


def free_surface_stress(meshes, sol):
    """Elastic surface stress on a regular grid: interpolate the free-surface
    displacement, apply the free-surface Hooke law to the in-plane gradients,
    then remove the anelastic eigenstress."""
    top = meshes["top"]
    c = top.centroids()
    u = sol["u:top"]
    g = np.linspace(-VIEW, VIEW, GRID_N)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    ux = griddata(c[:, :2], u[:, 0], pts, method="linear")
    uy = griddata(c[:, :2], u[:, 1], pts, method="linear")
    nn_x = griddata(c[:, :2], u[:, 0], pts, method="nearest")
    nn_y = griddata(c[:, :2], u[:, 1], pts, method="nearest")
    ux = np.where(np.isfinite(ux), ux, nn_x).reshape(X.shape)
    uy = np.where(np.isfinite(uy), uy, nn_y).reshape(X.shape)

    dh = g[1] - g[0]
    dux_dy, dux_dx = np.gradient(ux, dh, dh)
    duy_dy, duy_dx = np.gradient(uy, dh, dh)
    mu, lam = MAT.mu, MAT.lam
    coef = 2.0 * mu * lam / (lam + 2.0 * mu)          # free-surface plane stress
    div = dux_dx + duy_dy
    sig = np.zeros((X.size, 3, 3))
    sig[:, 0, 0] = (coef * div + 2.0 * mu * dux_dx).ravel()
    sig[:, 1, 1] = (coef * div + 2.0 * mu * duy_dy).ravel()
    sig[:, 0, 1] = sig[:, 1, 0] = (mu * (dux_dy + duy_dx)).ravel()

    obs = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    slip_cart = SLIP_MAG * np.asarray(meshes["s_hat"], float)
    # ``sig`` is differentiated from the BEM displacement field, whose fault
    # term is -H@slip, so its divergent near-fault part is MINUS the
    # eigenstress C:eps_star; recovering the elastic field therefore ADDS the
    # eigenstress back (the same sign convention as mbem.evaluate_stress, where
    # the fault stress term is -Sdd@slip). The eigenstress is negligible off
    # the surface-breaking trace and the trace strip is masked below, so this
    # only matters in a thin band hugging the mask -- but the sign is kept
    # consistent so a future on/near-fault evaluation does not double the spike.
    sig_el = sig + eigenstress_at_points(obs, meshes["fault"], slip_cart,
                                         mu, MAT.nu, EPS)
    comps = {"xx": sig_el[:, 0, 0], "yy": sig_el[:, 1, 1], "xy": sig_el[:, 0, 1]}
    S = {k: (v * GPA_TO_MPA).reshape(X.shape) for k, v in comps.items()}
    # Mask the thin surface-breaking fault strip: the slip is DISCONTINUOUS
    # across x=0, so grid-differentiating it there is meaningless (the band is
    # a rendering artifact of the jump, not stress).
    yext = float(np.abs(meshes["fault"].vertices[:, 1]).max())
    strip = (np.abs(X) < 8.0) & (np.abs(Y) <= yext + 10.0)
    for k in S:
        S[k][strip] = np.nan
    return X, Y, S


def main():
    meshes = build_fault_box(fault_depth=25.0, edge_fault=6.0,
                             edge_near=15.0, edge_far=35.0)
    model = build_model(meshes, SLIP_MAG, MAT)
    system = generate_system(model)
    nt = {k: meshes[k].n_triangles for k in ("top", "sides", "base", "fault")}
    print(f"meshes: {nt}; unknowns: {system.layout.n_unknowns}", flush=True)

    t0 = time.time()
    asm = AssembledDense(system, EPS, "direct", jump="calibrated")
    sol = asm.solve()
    print(f"solved in {time.time()-t0:.0f} s; cond ~ {asm.cond_estimate:.2e}",
          flush=True)

    # --- figure ---
    top = meshes["top"]
    tri = mtri.Triangulation(top.vertices[:, 0], top.vertices[:, 1],
                             top.triangles)
    u = sol["u:top"] * KM_TO_MM
    X, Y, S = free_surface_stress(meshes, sol)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2),
                             gridspec_kw=dict(wspace=0.32, hspace=0.28))
    disp_titles = [r"$u_x$ (mm)", r"$u_y$ (mm)", r"$u_z$ (mm)"]
    for k in range(3):
        ax = axes[0, k]
        vmax = max(float(np.percentile(np.abs(u[:, k]), 99.0)), 1e-6)
        tcf = ax.tripcolor(tri, u[:, k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="flat", rasterized=True)
        ax.plot([0, 0], [-50, 50], "k-", lw=0.8)
        _style(ax, disp_titles[k]); _cbar(fig, ax, tcf, vmax)
    stress_titles = [(r"$\sigma_{xx}$ (MPa)", "xx"),
                     (r"$\sigma_{yy}$ (MPa)", "yy"),
                     (r"$\sigma_{xy}$ (MPa)", "xy")]
    for k, (title, key) in enumerate(stress_titles):
        ax = axes[1, k]
        vmax = max(float(np.nanpercentile(np.abs(S[key]), 98.0)), 1e-6)
        cf = ax.contourf(X, Y, np.clip(S[key], -vmax, vmax),
                         levels=np.linspace(-vmax, vmax, 13), cmap="RdBu_r",
                         extend="both")
        ax.plot([0, 0], [-50, 50], "k-", lw=0.8)
        _style(ax, title); _cbar(fig, ax, cf, vmax)
    for ax, letter in zip(axes.flat, "abcdef"):
        ax.text(0.95, 0.95, letter, transform=ax.transAxes, ha="right",
                va="top")
    fig.suptitle(r"Fault-only mollified BEM (elastic surface stress: "
                 r"anelastic term subtracted)", fontsize=11, y=0.98)
    for ext in ("png", "pdf"):
        out = ROOT / f"fig_fault_only.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


def _style(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW, VIEW)
    ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])
    ax.set_xlabel(r"$x$ (km)"); ax.set_ylabel(r"$y$ (km)")
    ax.set_title(title, fontsize=9)
    ax.tick_params(direction="out", length=3, width=0.8)


def _cbar(fig, ax, mappable, vmax):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04, shrink=0.85)
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels([f"{-vmax:.2g}", "0", f"{vmax:.2g}"])
    cb.ax.tick_params(labelsize=7)


if __name__ == "__main__":
    main()
