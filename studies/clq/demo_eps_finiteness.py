"""Figure: the eigenstress subtraction keeps the on-fault elastic stress
finite as eps -> 0 for constant, linear and quadratic slip.

(a) interior peak |sigma_xz| on the mid-plane vs eps/L: raw (total) grows
    like 1/eps, elastic stays bounded, for each slip order;
(b) elastic sigma_xz along the line y = 0 for eps/L in {0.1, 0.05, 0.025}:
    the profiles collapse (colour = order, line style = eps);
(c) the raw profiles: spikes ~ (3/4) mu s(x)/eps that follow the slip shape.

Output: fig_eps_finiteness.png/.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _fields import (L, MU, NU, TRI, ORDER_LABELS, representative_slips, savefig)
from _paper_style import set_paper_style, text_size
import clq

set_paper_style()

EPS_LADDER = np.array([0.2, 0.1, 0.05, 0.025, 0.0125]) * L
PROFILE_EPS = [0.1 * L, 0.05 * L, 0.025 * L]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
STYLES = ["-", "--", ":"]


def main():
    slips = representative_slips()
    grid = clq.triangle_grid(TRI, 30)
    interior = grid[clq.inside(TRI, grid, margin=0.2 * L)]
    xs = np.linspace(-0.6 * L, 0.6 * L, 481)
    line = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])

    peaks_raw = np.zeros((3, len(EPS_LADDER)))
    peaks_el = np.zeros((3, len(EPS_LADDER)))
    print("interior peak |sigma_xz| (units mu s/L): raw vs elastic")
    for k, slip in enumerate(slips):
        for j, eps in enumerate(EPS_LADDER):
            tot = clq.stress(interior, TRI, slip, MU, NU, eps, subtract_eigenstress=False)
            el = clq.stress(interior, TRI, slip, MU, NU, eps)
            peaks_raw[k, j] = np.abs(tot[:, 0, 2]).max()
            peaks_el[k, j] = np.abs(el[:, 0, 2]).max()
        print(f"  {ORDER_LABELS[k]:9s} eps/L: " + "  ".join(f"{e:.4g}" for e in EPS_LADDER / L))
        print(f"  {'':9s} raw:   " + "  ".join(f"{v:.3g}" for v in peaks_raw[k]))
        print(f"  {'':9s} elast: " + "  ".join(f"{v:.3g}" for v in peaks_el[k]))

    fig, axes = plt.subplots(1, 3, figsize=text_size(2.6), gridspec_kw=dict(wspace=0.35))
    ax = axes[0]
    for k in range(3):
        ax.loglog(EPS_LADDER / L, peaks_raw[k], "o-", color=COLORS[k], lw=1.0, ms=3.5, mfc="w")
        ax.loglog(EPS_LADDER / L, peaks_el[k], "s-", color=COLORS[k], lw=1.2, ms=3.5)
    ref = peaks_raw[0, 0] * (EPS_LADDER[0] / EPS_LADDER)
    ax.loglog(EPS_LADDER / L, ref, "--", color="0.55", lw=0.8)
    ax.text(0.0185, 8.5, r"$\propto 1/\varepsilon$", color="0.4", fontsize=8,
            ha="left", va="center")
    ax.set_xlabel(r"$\varepsilon / L$")
    ax.set_ylabel(r"peak on-fault $|\sigma_{xz}|\;(\mu s_{\max}/L)$")
    ax.set_xticks([0.01, 0.1])
    ax.set_xlim(0.01, 0.25)
    ax.set_ylim(0.3, 100.0)
    ax.set_yticks([1.0, 10.0, 100.0])
    ax.text(0.05, 0.05, "filled: elastic\nopen: raw", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7)
    ax.set_box_aspect(1)
    ax.text(0.95, 0.95, "a", transform=ax.transAxes, ha="right", va="top")

    for panel, raw in ((1, False), (2, True)):
        ax = axes[panel]
        for k, slip in enumerate(slips):
            for j, eps in enumerate(PROFILE_EPS):
                sig = clq.stress(line, TRI, slip, MU, NU, eps, subtract_eigenstress=(not raw))
                ax.plot(xs / L, sig[:, 0, 2], STYLES[j], color=COLORS[k], lw=1.0)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([-0.6, 0.0, 0.6])
        if raw:
            ax.set_ylim(-2, 40)
            ax.set_yticks([0, 20, 40])
        else:
            ax.set_ylim(-6, 6)
            ax.set_yticks([-6, 0, 6])
        ax.set_xlabel(r"$x / L$  ($y = 0$)")
        ax.set_ylabel((r"raw " if raw else r"elastic ") + r"$\sigma_{xz}\;(\mu s_{\max}/L)$")
        ax.set_box_aspect(1)
        ax.text(0.95, 0.95, "bc"[panel - 1], transform=ax.transAxes, ha="right", va="top")
        if panel == 1:
            for k in range(3):
                ax.plot([], [], "-", color=COLORS[k], lw=1.2, label=ORDER_LABELS[k])
            ax.legend(frameon=False, fontsize=7, loc="upper left")
        if panel == 2:
            for j, eps in enumerate(PROFILE_EPS):
                ax.plot([], [], STYLES[j], color="k", lw=1.0, label=rf"$\varepsilon/L = {eps/L:g}$")
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    savefig(fig, "fig_eps_finiteness")
    plt.close(fig)


if __name__ == "__main__":
    main()
