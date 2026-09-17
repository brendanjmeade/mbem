"""Figure: the displacement jump [u_x] = u_x(+2 eps) - u_x(-2 eps) along the
line y = 0 converges to the prescribed slip as eps -> 0, for constant, linear
and quadratic slip (one panel per order; eps/L in {0.04, 0.02, 0.01, 0.005}
in viridis shades, prescribed interpolant dashed black).

Output: fig_jump_profiles.png/.pdf
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _fields import L, MU, NU, TRI, NHAT, ORDER_LABELS, representative_slips, savefig
from _paper_style import set_paper_style, text_size
import clq

set_paper_style()

EPS_LIST = np.array([0.04, 0.02, 0.01, 0.005]) * L


def main():
    xs = np.linspace(-0.6 * L, 0.6 * L, 601)
    line = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    inside = clq.inside(TRI, line)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=text_size(2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    for k, slip in enumerate(representative_slips()):
        ax = axes[k]
        s = clq.interpolate(TRI, slip, line)[:, 0]
        s[~inside] = 0.0
        for j, eps in enumerate(EPS_LIST):
            z0 = 2.0 * eps
            jump = (clq.displacement(line + z0 * NHAT, TRI, slip, MU, NU, eps)
                    - clq.displacement(line - z0 * NHAT, TRI, slip, MU, NU, eps))[:, 0]
            ax.plot(xs / L, jump, "-", color=cmap(0.15 + 0.7 * j / (len(EPS_LIST) - 1)), lw=1.0,
                    label=rf"$\varepsilon/L = {eps/L:g}$")
        ax.plot(xs / L, s, "--", color="k", lw=0.9, label="prescribed")
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([-0.6, 0.0, 0.6])
        ax.set_ylim(-0.1, 1.5)
        ax.set_yticks([0.0, 0.5, 1.0, 1.5])
        ax.set_xlabel(r"$x / L$  ($y = 0$)")
        ax.set_box_aspect(1)
        ax.set_title(ORDER_LABELS[k], fontsize=9)
        ax.text(0.95, 0.95, "abc"[k], transform=ax.transAxes, ha="right", va="top")
        if k == 0:
            ax.set_ylabel(r"$[u_x] / s_{\max}$")
        if k == 1:
            ax.legend(frameon=False, fontsize=7, loc="upper left")
    savefig(fig, "fig_jump_profiles")
    plt.close(fig)


if __name__ == "__main__":
    main()
