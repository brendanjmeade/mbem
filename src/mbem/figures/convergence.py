"""Convergence in the mollification width: a STUDY swept over eps.

``eps`` is a per-source-element parameter of the model, so changing it changes
the operator -- the mollified kernel itself differs -- and a different operator
is a different run. The old demo reassembled inside a loop in one process, which
is the same arithmetic but leaves one figure and no record of the ladder.

The reference is the SMALLEST eps in the sweep, not an analytic solution: what
the panel shows is self-convergence, and the dashed guide is eps^2 because that
is the order the Cortez regularization is expected to approach the singular
kernel at. A ladder that flattens instead of following it has hit the
resolution floor -- eps below ~h/3 is no longer resolved by the mesh, which is
why the demo's own geometry refines the fault to keep eps > h/3 over the whole
ladder.
"""

from __future__ import annotations

import numpy as np

from mbem.figures import save_figure
from mbem.figures.style import set_paper_style

KM_TO_MM = 1.0e6
SLOT = "u:top"


def _eps_sorted(study):
    """``[(eps, run), ...]`` ascending. The sweep keys are strings."""
    rows = []
    for key, run in study.runs.items():
        try:
            rows.append((float(key), run))
        except ValueError:
            raise ValueError(f"eps_convergence needs a numeric sweep; got "
                             f"key {key!r}") from None
    return sorted(rows)


def eps_convergence(study):
    """RMS self-convergence against the smallest eps, and an across-fault cut."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    set_paper_style()

    rows = _eps_sorted(study)
    if len(rows) < 3:
        raise ValueError(f"eps_convergence wants at least 3 rungs, got "
                         f"{len(rows)}")
    state = next(iter(rows[0][1].fields))
    sols = {e: r.fields[state][SLOT] for e, r in rows}
    eps_ref, ref_run = rows[0]
    u_ref = sols[eps_ref]
    coarser = [e for e, _r in rows[1:]]
    rms = [float(np.sqrt(np.mean((sols[e] - u_ref) ** 2))) * KM_TO_MM
           for e in coarser]

    m = ref_run.meshes
    top_c = np.asarray(m["arr__top_centroids"]) if "arr__top_centroids" in m \
        else None
    if top_c is None:                       # centroids from the saved mesh
        v, t = m["top__v"], m["top__t"]
        top_c = v[t].mean(axis=1)

    xs = np.linspace(-120.0, 120.0, 200)
    line = np.column_stack([xs, np.zeros_like(xs)])
    transects = {e: griddata(top_c[:, :2], sols[e][:, 1], line,
                             method="linear") * KM_TO_MM for e, _r in rows}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.8, 4.3))
    ax0.loglog(coarser, rms, "o-", color="#1f5fa6", lw=1.4, ms=5)
    e0 = np.array(coarser, float)
    ref2 = rms[0] * (e0 / e0[0]) ** 2
    ax0.loglog(e0, ref2, "--", color="0.55", lw=0.9)
    ax0.text(e0[-1], ref2[-1], r"$\propto \varepsilon^{2}$", color="0.4",
             ha="left", va="bottom", fontsize=8)
    ax0.set_xlabel(r"$\varepsilon$ (km)")
    ax0.set_ylabel(rf"RMS $|u - u(\varepsilon={eps_ref:g})|$ (mm)")
    ax0.set_title("self-convergence of the surface field", fontsize=9)
    ax0.grid(True, which="both", lw=0.3, color="0.85")
    ax0.text(0.05, 0.95, "a", transform=ax0.transAxes, ha="left", va="top")

    cmap = plt.get_cmap("viridis")
    for i, (e, _r) in enumerate(rows):
        ax1.plot(xs, transects[e], lw=1.3,
                 color=cmap(i / max(len(rows) - 1, 1)),
                 label=rf"$\varepsilon={e:g}$")
    ax1.set_xlabel(r"$x$ (km)")
    ax1.set_ylabel(r"$u_y$ (mm) along $y=0$")
    ax1.set_title("across-fault transect", fontsize=9)
    ax1.legend(fontsize=7, frameon=False, ncol=2)
    ax1.grid(True, lw=0.3, color="0.9")
    ax1.text(0.05, 0.95, "b", transform=ax1.transAxes, ha="left", va="top")

    fig.suptitle("Mollified BEM: surface field converges as eps falls "
                 "(while eps stays above ~h/3)", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_figure(fig, study.study_dir, "eps_convergence")
    plt.close(fig)
    return out
