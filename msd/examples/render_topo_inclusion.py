"""Render the topography + inclusion showcase figure.

Reads benchmarks/topo_inclusion_fields.npz (make_topo_inclusion.py).

Output (repo root): fig_topo_inclusion_showcase.png/.pdf — 3 x 3:
  row 1: u(topo, het)                    raw fields with hill + inclusion
  row 2: Du_inc  = u(topo,het) - u(topo,hom)   inclusion effect
         (NOTE both states include the hill)
  row 3: Du_topo = u(topo,het) - u(flat,het)   topography effect

Overlays: fault trace (solid), inclusion footprint (dashed circle),
topography contours at 0.5 / 1.0 / 1.5 km (thin gray).
"""

import pathlib
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

NPZ = pathlib.Path(__file__).parent / "topo_inclusion_fields_mu10.npz"

INC_X, INC_Y, INC_R = -100.0, 100.0, 75.0


def _style(ax, tri_h, h_vertex):
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot([0.0, 0.0], [-100.0, 100.0], "k-", lw=0.8)
    ax.plot(INC_X + INC_R * np.cos(th), INC_Y + INC_R * np.sin(th),
            "k--", lw=0.7)
    ax.tricontour(tri_h, h_vertex, levels=[0.5, 1.0, 1.5],
                  colors="0.35", linewidths=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-200.0, 200.0)
    ax.set_ylim(-200.0, 200.0)
    ax.set_xticks([-200, 0, 200])
    ax.set_yticks([-200, 0, 200])
    ax.tick_params(direction="out", length=3, width=0.8)


def _panel(fig, ax, tri_h, tri_i, fh, fi, vmax, h_vertex, title=None):
    cf = ax.tripcolor(tri_h, fh, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      shading="flat", rasterized=True)
    ax.tripcolor(tri_i, fi, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 shading="flat", rasterized=True)
    _style(ax, tri_h, h_vertex)
    if title:
        ax.set_title(title, fontsize=9)
    cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.04, shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-vmax, 0, vmax])
    cb.set_ticklabels([f"{-vmax:.0f}", "0", f"{vmax:.0f}"])


def _vmax99(*arrays, floor=0.5):
    return max(max(float(np.percentile(np.abs(a), 99.0)) for a in arrays),
               floor)


def main():
    d = np.load(NPZ)
    print("condition estimates:",
          {k: f"{float(d[k]):.2e}" for k in d.files if k.startswith("cond")})

    tri_h = mtri.Triangulation(d["host_top_vertices"][:, 0],
                               d["host_top_vertices"][:, 1],
                               d["host_top_triangles"])
    tri_i = mtri.Triangulation(d["inclusion_top_vertices"][:, 0],
                               d["inclusion_top_vertices"][:, 1],
                               d["inclusion_top_triangles"])
    h_vertex = d["host_top_h"]

    # km -> mm
    rows = {
        "raw": (d["u_host_top_topo_het"] * 1e6,
                d["u_inclusion_top_topo_het"] * 1e6),
        "inc": ((d["u_host_top_topo_het"]
                 - d["u_host_top_topo_hom"]) * 1e6,
                (d["u_inclusion_top_topo_het"]
                 - d["u_inclusion_top_topo_hom"]) * 1e6),
        "topo": ((d["u_host_top_topo_het"]
                  - d["u_host_top_flat_het"]) * 1e6,
                 (d["u_inclusion_top_topo_het"]
                  - d["u_inclusion_top_flat_het"]) * 1e6),
    }
    titles = {
        "raw": [r"$u_x$ (mm)", r"$u_y$ (mm)", r"$u_z$ (mm)"],
        "inc": [r"$\Delta u_x^{\rm inc}$ (mm)",
                r"$\Delta u_y^{\rm inc}$ (mm)",
                r"$\Delta u_z^{\rm inc}$ (mm)"],
        "topo": [r"$\Delta u_x^{\rm topo}$ (mm)",
                 r"$\Delta u_y^{\rm topo}$ (mm)",
                 r"$\Delta u_z^{\rm topo}$ (mm)"],
    }

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.0),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.20, hspace=0.22))
    for r, key in enumerate(("raw", "inc", "topo")):
        Uh, Ui = rows[key]
        for k in range(3):
            vmax = _vmax99(Uh[:, k], Ui[:, k],
                           floor=1.0 if key == "raw" else 0.5)
            _panel(fig, axes[r, k], tri_h, tri_i, Uh[:, k], Ui[:, k],
                   vmax, h_vertex, titles[key][k])
        axes[r, 0].set_ylabel(r"$y$ (km)")
    for k in range(3):
        axes[2, k].set_xlabel(r"$x$ (km)")
    for ax, letter in zip(axes.flat, "abcdefghi"):
        ax.text(0.95, 0.95, letter, transform=ax.transAxes,
                ha="right", va="top")

    for ext in ("png", "pdf"):
        fname = ROOT / f"fig_topo_inclusion_showcase.{ext}"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
