"""Figure: displacement jump across the fault for constant, linear and
quadratic slip on one equilateral triangle.

On the mid-plane the slip-parallel displacement of a full-space planar
dislocation vanishes by antisymmetry, so the on-fault displacement is shown
as the JUMP across the mollified zone,

    [u](x) = u(x + 2 eps nhat) - u(x - 2 eps nhat),

which in the interior recovers f(2) = 0.984 of the prescribed slip from the
mollified profile f(t) = t(2t^2+3)/(2(1+t^2)^{3/2}) times a finite-size
factor (~0.88 here: the field still varies across +-2 eps = +-0.02 L on a
triangle whose inradius is 0.29 L; fig_jump_profiles shows the convergence
as eps -> 0) and smooths the edges over ~eps.  Rows: constant / linear / quadratic slip; columns:
prescribed s_x, [u_x], [u_y], and u_z on the mid-plane z = 0.  [u_z] vanishes
identically (u_z is symmetric in z for in-plane slip), so the fourth column
shows the surviving mid-plane component u_z(z=0) instead; [u_y] is the
finite-offset residual 2 u_y(2 eps), confined to the mixed-mode (inclined)
edges and vanishing with eps.  Units: L = mu = s_max = 1, eps = 0.01 L.

Output: fig_onfault_displacement.png/.pdf in the clq root.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _fields import (L, MU, NU, TRI, NHAT, ORDER_LABELS, representative_slips,
                     square_grid, chunked, slip_field, draw_outline, style_panel,
                     column_colorbar, savefig)
from _paper_style import set_paper_style, text_size
import clq

set_paper_style()

EPS = 0.01 * L
Z0 = 2.0 * EPS


def main():
    xs, X, Y, pts = square_grid()
    n = len(xs)
    slips = representative_slips()
    fields = []   # per order: (s_x, [u_x], [u_y], [u_z]) on the grid
    for k, slip in enumerate(slips):
        s = slip_field(slip, pts)[:, 0]
        up = chunked(lambda p: clq.displacement(p + Z0 * NHAT, TRI, slip, MU, NU, EPS), pts)
        um = chunked(lambda p: clq.displacement(p - Z0 * NHAT, TRI, slip, MU, NU, EPS), pts)
        jump = up - um
        umid = chunked(lambda p: clq.displacement(p, TRI, slip, MU, NU, EPS), pts)
        fields.append([s.reshape(n, n), jump[:, 0].reshape(n, n), jump[:, 1].reshape(n, n),
                       umid[:, 2].reshape(n, n)])
        inside = clq.inside(TRI, pts, margin=4 * EPS)
        ratio = jump[inside, 0] / np.where(np.abs(s[inside]) > 0.3, s[inside], np.nan)
        print(f"{ORDER_LABELS[k]:9s}: interior [u_x]/s_x median = {np.nanmedian(ratio):.4f} "
              f"(f(2) = {2*(2*4+3)/(2*5**1.5):.4f}), max |[u_y]| = {np.abs(jump[:,1]).max():.2e}, "
              f"max |[u_z]| = {np.abs(jump[:,2]).max():.2e}")

    col_lbl = [r"$s_x / s_{\max}$", r"$[u_x] / s_{\max}$", r"$[u_y] / s_{\max}$", r"$u_z(z{=}0) / s_{\max}$"]
    fig, axes = plt.subplots(3, 4, figsize=text_size(6.0), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.10))
    letters = "abcdefghijkl"
    for c in range(4):
        if c <= 1:
            vmax = float(np.nanmax(np.abs(np.stack([f[0] for f in fields]))))   # 4/3 for the dome
        else:
            vmax = max(float(np.nanpercentile(np.abs(np.stack([f[c] for f in fields])), 99)), 1e-6)
        for r in range(3):
            ax = axes[r, c]
            pm = ax.pcolormesh(X / L, Y / L, fields[r][c], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="nearest", rasterized=True)
            draw_outline(ax)
            if c == 0:
                nd = clq.nodes(TRI, r)
                ax.plot(nd[:, 0] / L, nd[:, 1] / L, "o", ms=2.5, mfc="w", mec="k", mew=0.6)
            style_panel(ax)
            ax.text(0.95, 0.95, letters[r * 4 + c], transform=ax.transAxes, ha="right", va="top")
            if r == 0:
                ax.set_title(col_lbl[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(ORDER_LABELS[r] + "\n" + r"$y / L$")
            if r == 2:
                ax.set_xlabel(r"$x / L$")
        column_colorbar(fig, axes[:, c], pm, vmax)
    fig.text(0.5, 0.995, r"$[u] = u(x + 2\varepsilon\hat n) - u(x - 2\varepsilon\hat n)$, "
             r"$\varepsilon = 0.01\,L$", ha="center", va="top", fontsize=8.5)
    savefig(fig, "fig_onfault_displacement")
    plt.close(fig)


if __name__ == "__main__":
    main()
