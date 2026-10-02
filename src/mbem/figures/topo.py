"""Box + topography + fault + inclusion: the four-state decomposition.

A STUDY figure, not a run figure, and that distinction is the whole reason the
study machinery exists. The two effects drawn here are

    inclusion effect   Du_inc  = u(topo, het) - u(topo, hom)
    topography effect  Du_topo = u(topo, het) - u(flat, het)

The first differences two MATERIAL STATES, which share one assembly, so it lives
inside a run. The second differences two SURFACES -- warped against flat -- and
those are different meshes, hence different operators, hence different runs. A
figure that needs two operators cannot be a property of either one.

The old script solved all four states in one process and wrote one npz, which is
why a committed panel had no recoverable provenance: nothing said which code,
which eps, or which backend produced it. Here each surface is a run with its own
``resolved.json``, and this figure reads them back off disk.

WHAT THE CANCELLATION COSTS, which anyone reading the lower row should know:
Du_topo differences two fields that agree to 43-72x their difference, so fields
accurate to ~2e-6 give a decomposition accurate to ~2e-4. That is a property of
the quantity, not of the backend -- both the ACA and the FMM far fields land in
the same place (BACKLOG).
"""

from __future__ import annotations

import numpy as np

from mbem.figures import save_figure
from mbem.figures.style import set_paper_style

# The inclusion rim and the fault trace, drawn for reference on every panel.
INC_X, INC_Y, INC_R = -100.0, 100.0, 75.0
HOST = "u:host_top"
INCL = "u:inclusion_top"


def _tris(run):
    """The FLAT triangulation both surfaces are plotted on, and the relief.

    Flat for both: this is a map view, so the warped run is drawn in the frame
    of the undeformed surface and the relief appears as contours rather than as
    displaced vertices. The bundle carries the flat frame for exactly this.
    """
    import matplotlib.tri as mtri
    m = run.meshes
    tri_h = mtri.Triangulation(m["arr__host_top_flat_v"][:, 0],
                               m["arr__host_top_flat_v"][:, 1],
                               m["arr__host_top_flat_t"])
    tri_i = mtri.Triangulation(m["inclusion_top__v"][:, 0],
                               m["inclusion_top__v"][:, 1],
                               m["inclusion_top__t"])
    return tri_h, tri_i, m["arr__host_top_h"]


def _style(ax, tri_h, h_vertex):
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot([0.0, 0.0], [-100.0, 100.0], "k-", lw=0.8)          # fault trace
    ax.plot(INC_X + INC_R * np.cos(th), INC_Y + INC_R * np.sin(th),
            "k--", lw=0.7)                                       # inclusion rim
    ax.tricontour(tri_h, h_vertex, levels=[0.5, 1.0, 1.5],
                  colors="0.35", linewidths=0.5)                 # relief
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
    return max(max(float(np.percentile(np.abs(a), 99.0)) for a in arrays), floor)


def _decompose(study):
    """``{row: (host, inclusion)}`` in mm, from the two surface runs.

    Raises rather than guessing if a surface or a state is missing: a panel
    silently drawn from the wrong pair is the failure worth preventing here.
    """
    topo, flat = study.by("topo"), study.by("flat")
    for run, need in ((topo, ("het", "hom")), (flat, ("het",))):
        missing = [s for s in need if s not in run.fields]
        if missing:
            raise KeyError(f"{run.run_dir.name} lacks state(s) {missing}; "
                           f"has {sorted(run.fields)}")
    mm = 1e6                                      # km -> mm
    th, ti = topo.fields["het"][HOST], topo.fields["het"][INCL]
    oh, oi = topo.fields["hom"][HOST], topo.fields["hom"][INCL]
    fh, fi = flat.fields["het"][HOST], flat.fields["het"][INCL]
    return {"raw": (th * mm, ti * mm),
            "inc": ((th - oh) * mm, (ti - oi) * mm),
            "topo": ((th - fh) * mm, (ti - fi) * mm)}


def showcase(study):
    """The 3x3 showcase: raw field, inclusion effect, topography effect."""
    import matplotlib.pyplot as plt
    set_paper_style()

    rows = _decompose(study)
    tri_h, tri_i, h_vertex = _tris(study.by("topo"))
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
    out = save_figure(fig, study.study_dir, "topo_inclusion_showcase")
    plt.close(fig)
    return out


def contour(study):
    """The topography effect alone, as filled contours with the relief over it."""
    import matplotlib.pyplot as plt
    set_paper_style()

    rows = _decompose(study)
    tri_h, tri_i, h_vertex = _tris(study.by("topo"))
    Uh, Ui = rows["topo"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True,
                             gridspec_kw=dict(wspace=0.18))
    comp = [r"$\Delta u_x^{\rm topo}$ (mm)", r"$\Delta u_y^{\rm topo}$ (mm)",
            r"$\Delta u_z^{\rm topo}$ (mm)"]
    for k, ax in enumerate(axes):
        vmax = _vmax99(Uh[:, k], Ui[:, k], floor=0.5)
        _panel(fig, ax, tri_h, tri_i, Uh[:, k], Ui[:, k], vmax, h_vertex,
               comp[k])
        ax.set_xlabel(r"$x$ (km)")
    axes[0].set_ylabel(r"$y$ (km)")
    out = save_figure(fig, study.study_dir, "topo_inclusion_contour")
    plt.close(fig)
    return out
