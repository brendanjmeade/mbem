"""One model, two backends: does the fast operator reproduce the reference?

A STUDY figure swept over ``backend``, because two backends are two operators
and this repo's rule is that a different operator is a different run. The old
demo assembled both in one process and compared them in memory, which works but
leaves no record: nothing afterwards could say which tolerance, which eps or
which commit either side used. As a sweep, each backend has its own
``resolved.json`` and this figure differences them off disk.

The scatter is the point rather than a single norm. A relative max difference
says how big the worst disagreement is; the cross-plot says whether the
disagreement is NOISE about the diagonal or a systematic tilt, and those have
different causes -- the first is the far-field tolerance doing its job, the
second is a wrong material, a wrong eps or a wrong sign.
"""

from __future__ import annotations

import numpy as np

from mbem.figures import save_figure
from mbem.figures.style import set_paper_style

INC_X, INC_Y, INC_R = -100.0, 100.0, 75.0
KM_TO_MM = 1.0e6
SLOTS = ("u:host_top", "u:inclusion_top")


def _flat(run, state: str, slots=SLOTS) -> np.ndarray:
    f = run.fields[state]
    return np.concatenate([f[s].ravel() for s in slots]) * KM_TO_MM


def backend_agreement(study):
    """Map of the fast solution, and it cross-plotted against the reference.

    The reference is whichever of ``dense`` / ``hmat`` the sweep contains in
    that order of preference, so ``--sweep backend=hmat,dense`` and
    ``backend=fmm,hmat`` both do the sensible thing instead of needing a flag.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    set_paper_style()

    keys = list(study.runs)
    ref_key = next((k for k in ("dense", "hmat") if k in keys), keys[0])
    fast_key = next((k for k in keys if k != ref_key), ref_key)
    ref, fast = study.by(ref_key), study.by(fast_key)
    state = next(iter(fast.fields))

    m = fast.meshes
    th = mtri.Triangulation(m["host_top__v"][:, 0], m["host_top__v"][:, 1],
                            m["host_top__t"])
    ti = mtri.Triangulation(m["inclusion_top__v"][:, 0],
                            m["inclusion_top__v"][:, 1], m["inclusion_top__t"])
    uh = fast.fields[state]["u:host_top"][:, 0] * KM_TO_MM
    ui = fast.fields[state]["u:inclusion_top"][:, 0] * KM_TO_MM
    vmax = max(float(np.percentile(np.abs(np.concatenate([uh, ui])), 99.0)), 1e-6)

    a, b = _flat(fast, state), _flat(ref, state)
    rel = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    n = fast.resolved.get("model", {}).get("n_unknowns", 0)
    iters = next((s.get("iterations") for s in fast.report.get("states", [])
                  if s.get("label") == state), None)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 4.5))
    tcf = ax0.tripcolor(th, uh, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        shading="flat", rasterized=True)
    ax0.tripcolor(ti, ui, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  shading="flat", rasterized=True)
    t2 = np.linspace(0, 2 * np.pi, 200)
    ax0.plot([0, 0], [-100, 100], "k-", lw=0.8)
    ax0.plot(INC_X + INC_R * np.cos(t2), INC_Y + INC_R * np.sin(t2),
             "k--", lw=0.7)
    ax0.set_aspect("equal"); ax0.set_xlim(-200, 200); ax0.set_ylim(-200, 200)
    ax0.set_xticks([-200, 0, 200]); ax0.set_yticks([-200, 0, 200])
    ax0.set_xlabel(r"$x$ (km)"); ax0.set_ylabel(r"$y$ (km)")
    ax0.set_title(rf"$u_x$ (mm), {fast_key}", fontsize=9)
    fig.colorbar(tcf, ax=ax0, fraction=0.045, pad=0.04, shrink=0.85)
    ax0.text(0.96, 0.96, "a", transform=ax0.transAxes, ha="right", va="top")

    lim = max(np.abs(b).max(), 1e-6)
    ax1.plot([-lim, lim], [-lim, lim], "-", color="0.6", lw=0.8)
    ax1.plot(b, a, ".", color="#1f5fa6", ms=2, alpha=0.4)
    ax1.set_xlabel(rf"{ref_key}  $u$ (mm)")
    ax1.set_ylabel(rf"{fast_key}  $u$ (mm)")
    ax1.set_box_aspect(1)
    label = f"{n} unknowns\nmax rel diff {rel:.0e}"
    if iters is not None:
        label += f"\nFGMRES iters {iters}"
    ax1.text(0.05, 0.95, label, transform=ax1.transAxes, ha="left", va="top",
             fontsize=8, bbox=dict(boxstyle="square,pad=0.3", fc="white",
                                   ec="0.5", lw=0.6))
    ax1.text(0.96, 0.96, "b", transform=ax1.transAxes, ha="right", va="top")

    fig.suptitle(f"{fast_key} reproduces {ref_key} "
                 r"($\mu/10$ inclusion)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_figure(fig, study.study_dir, "backend_agreement")
    plt.close(fig)
    return out
