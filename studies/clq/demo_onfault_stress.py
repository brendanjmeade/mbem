"""Figures: on-fault and near-fault stress for constant, linear and quadratic
slip on one equilateral triangle (mu = s_max = L = 1, eps = 0.05 L).

On the mid-plane z = 0 of a planar dislocation in a full space, reflection
symmetry makes the components with an even number of z indices
(sigma_xx, sigma_yy, sigma_xy, sigma_zz) antisymmetric in z for in-plane
slip, so they vanish identically there (checked numerically: |.| < 1e-12);
only the shear tractions sigma_xz, sigma_yz survive.  Hence:

  fig_onfault_stress_elastic   -- z = 0, rows constant/linear/quadratic,
                                  columns sigma_xz, sigma_yz; ELASTIC stress
                                  sigma_el = sigma_tot - C:eps* (exact
                                  eigenstress of the smeared slip subtracted,
                                  tree-wide policy);
  fig_onfault_stress_total     -- same layout, the raw kernel (TOTAL) stress,
                                  dominated on the fault by C:eps* ~ (3/4) mu s/eps;
  fig_nearfault_stress_elastic -- z = +2 eps (one side, just outside the
                                  smeared zone), all six components, elastic.

Colour limits per column: 98th percentile of |sigma| over interior points
(2 eps from the edges) pooled over the three rows, so the orders are
directly comparable and the ~1/eps edge caps do not set the scale.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _fields import (L, MU, NU, TRI, NHAT, ORDER_LABELS, representative_slips, square_grid,
                     chunked, draw_outline, style_panel, column_colorbar, savefig)
from _paper_style import set_paper_style, text_size
import clq

set_paper_style()

EPS = 0.05 * L
Z_NEAR = 2.0 * EPS
COMP6 = [(0, 2), (1, 2), (2, 2), (0, 0), (1, 1), (0, 1)]
LBL6 = [r"$\sigma_{xz}$", r"$\sigma_{yz}$", r"$\sigma_{zz}$",
        r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]


def compute(z_offset):
    xs, X, Y, pts = square_grid()
    n = len(xs)
    interior = clq.inside(TRI, pts, margin=2 * EPS)
    shifted = pts + z_offset * NHAT
    out = {"elastic": [], "total": []}
    for k, slip in enumerate(representative_slips()):
        tot = chunked(lambda p: clq.stress(p, TRI, slip, MU, NU, EPS, subtract_eigenstress=False), shifted)
        eig = chunked(lambda p: clq.eigenstress(p, TRI, slip, MU, NU, EPS), shifted)
        el = tot - eig
        out["total"].append([tot[:, a, b].reshape(n, n) for a, b in COMP6])
        out["elastic"].append([el[:, a, b].reshape(n, n) for a, b in COMP6])
        if z_offset == 0.0:
            even = max(np.abs(el[:, a, b]).max() for a, b in COMP6[2:])
            c = TRI.mean(axis=0)[None, :]
            sc = clq.stress(c, TRI, slip, MU, NU, EPS)[0]
            st = clq.stress(c, TRI, slip, MU, NU, EPS, subtract_eigenstress=False)[0]
            print(f"{ORDER_LABELS[k]:9s} z=0: max|sigma_zz,xx,yy,xy| = {even:.1e} (symmetry); centroid "
                  f"elastic sigma_xz = {sc[0,2]:+.4f}, total sigma_xz = {st[0,2]:+.4f} "
                  f"(3/4 mu s/eps = {0.75/EPS:.1f})")
    return X, Y, interior.reshape(n, n), out


def plot(X, Y, interior, fields, cols, name, title, height):
    ncol = len(cols)
    fig, axes = plt.subplots(3, ncol, figsize=text_size(height) if ncol > 2 else (4.2, height),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.10))
    letters = "abcdefghijklmnopqr"
    for ci, c in enumerate(cols):
        pool = np.concatenate([np.abs(fields[r][c][interior]) for r in range(3)])
        vmax = max(float(np.percentile(pool, 98)), 1e-6)
        for r in range(3):
            ax = axes[r, ci]
            pm = ax.pcolormesh(X / L, Y / L, fields[r][c], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="nearest", rasterized=True)
            draw_outline(ax)
            style_panel(ax)
            ax.text(0.95, 0.95, letters[r * ncol + ci], transform=ax.transAxes, ha="right", va="top",
                    fontsize=8)
            if r == 0:
                ax.set_title(LBL6[c], fontsize=9)
            if ci == 0:
                ax.set_ylabel(ORDER_LABELS[r] + "\n" + r"$y / L$")
            if r == 2:
                ax.set_xlabel(r"$x / L$")
        column_colorbar(fig, axes[:, ci], pm, vmax)
    fig.text(0.5, 0.995, title, ha="center", va="top", fontsize=8.5)
    savefig(fig, name)
    plt.close(fig)


def main():
    X, Y, interior, on = compute(0.0)
    plot(X, Y, interior, on["elastic"], [0, 1], "fig_onfault_stress_elastic",
         r"elastic $\sigma_{\rm el}$ on $z = 0$ ($\mu s_{\max}/L$, $\varepsilon = 0.05\,L$); "
         r"$\sigma_{zz},\sigma_{xx},\sigma_{yy},\sigma_{xy} \equiv 0$ there", 7.6)
    plot(X, Y, interior, on["total"], [0, 1], "fig_onfault_stress_total",
         r"total $\sigma_{\rm tot}$ on $z = 0$ ($\mu s_{\max}/L$, $\varepsilon = 0.05\,L$)", 7.6)
    X, Y, interior, near = compute(Z_NEAR)
    plot(X, Y, interior, near["elastic"], [0, 1, 2, 3, 4, 5], "fig_nearfault_stress_elastic",
         r"elastic $\sigma$ on $z = +2\varepsilon$ ($\mu s_{\max}/L$, $\varepsilon = 0.05\,L$)", 4.6)


if __name__ == "__main__":
    main()
