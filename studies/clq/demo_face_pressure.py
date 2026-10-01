"""Figures: slip as filled contours, and the mean stress (pressure) on the
fault faces for constant, linear and quadratic slip.

Physics.  For in-plane slip on a planar fault in a homogeneous full space the
stress components with an even number of z indices are odd in z (reflection
symmetry through the fault plane), so on the mid-plane z = 0 the mean stress
is exactly zero for EVERY slip distribution.  Across the plane those
components jump by the slip-gradient (eigen)strain, and the face values are
half the jump:

    p(0+-) = -tr(sigma)/3 |_{0+-} = -+ mu (1 + nu) / (3 (1 - nu)) * div_par(Delta u),

with div_par the in-plane divergence of the slip vector.  Hence the face
pressure is zero in the interior of a constant-slip element (only edge
bands remain, ~1/eps in the mollified model, a line singularity in the sharp
limit), a constant on a linear element whose gradient has a component along
the slip, zero for a linear ramp perpendicular to the slip, and a linear
function on a quadratic element.  The eigenstress C:eps* of in-plane slip is
trace-free, so total and elastic mean stress coincide.

  fig_slip_contours   -- the four slip distributions as filled contours on the
                         triangle (nodes marked);
  fig_face_pressure   -- rows: constant / linear hat (div = 0) / linear ramp
                         along the slip (div = 1/L) / quadratic dome;
                         columns: slip, face pressure p+ at z = +3 eps from
                         clq, sharp-limit prediction from the slip divergence.
Units: L = mu = s_max = 1, nu = 1/4, eps = 0.005 L.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from _fields import L, MU, NU, TRI, NHAT, chunked, savefig
from _paper_style import set_paper_style, text_size
import clq
from clq.frame import local_frame
from clq.shape import shape_coefficients

set_paper_style()

EPS = 0.005 * L
Z0 = 3.0 * EPS
NGRID = 120
CASES = [
    ("constant", np.array([[1.0, 0.0, 0.0]])),
    ("linear, hat at $v_1$\n($\\nabla_\\parallel\\!\\cdot\\!\\Delta u = 0$)",
     np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    ("linear, ramp along $x$\n($\\nabla_\\parallel\\!\\cdot\\!\\Delta u = 1/L$)",
     np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
    ("quadratic dome", np.array([[0.0, 0.0, 0.0]] * 3 + [[1.0, 0.0, 0.0]] * 3)),
]
SHORT = ["constant", "linear (hat)", "linear (ramp)", "quadratic"]


def slip_divergence(slip, pts):
    """In-plane divergence of the interpolated slip at points on the plane."""
    fr = local_frame(TRI)
    p = clq.order_from_count(len(slip))
    _, X = fr.to_plane(pts)
    c = shape_coefficients(fr, p, X)              # (M, K, p+1, p+1) about each point
    d1 = np.einsum("mk,kj->mj", c[:, :, 1, 0] if p >= 1 else np.zeros((len(pts), len(slip))), slip)
    d2 = np.einsum("mk,kj->mj", c[:, :, 0, 1] if p >= 1 else np.zeros((len(pts), len(slip))), slip)
    return d1 @ fr.e1 + d2 @ fr.e2


def face_pressure(slip, pts):
    sig = chunked(lambda q: clq.stress(q + Z0 * NHAT, TRI, slip, MU, NU, EPS), pts)
    return -np.einsum("nii->n", sig) / 3.0


def style(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks([-0.6, 0.0, 0.6])
    ax.set_yticks([-0.6, 0.0, 0.6])
    ax.tick_params(direction="out", length=3, width=0.8)


def outline(ax):
    x = np.r_[TRI[:, 0], TRI[0, 0]]
    y = np.r_[TRI[:, 1], TRI[0, 1]]
    ax.plot(x, y, color="k", lw=0.7)


def fill(ax, triang, vals, levels, cmap, lines=True):
    cf = ax.tricontourf(triang, vals, levels=levels, cmap=cmap, extend="both")
    if lines:
        ax.tricontour(triang, vals, levels=levels, colors="k", linewidths=0.25, alpha=0.6)
    return cf


def main():
    pts = clq.triangle_grid(TRI, NGRID)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], clq.grid_triangles(NGRID))
    interior = clq.inside(TRI, pts, margin=0.15 * L)      # deep interior: outside the edge-singularity tails
    coef = MU * (1.0 + NU) / (3.0 * (1.0 - NU))

    slips = [clq.interpolate(TRI, s, pts)[:, 0] for _, s in CASES]
    smax = max(v.max() for v in slips)

    # ---------------- figure 1: slip as filled contours -------------------
    fig, axes = plt.subplots(1, 4, figsize=text_size(2.3), sharey=True, gridspec_kw=dict(wspace=0.28))
    levels = np.linspace(0.0, smax, 15)
    for k, ((name, s), ax) in enumerate(zip(CASES, axes)):
        cf = fill(ax, triang, slips[k], levels, "viridis")
        outline(ax)
        nd = clq.nodes(TRI, clq.order_from_count(len(s)))
        ax.plot(nd[:, 0], nd[:, 1], "o", ms=3, mfc="w", mec="k", mew=0.6)
        style(ax)
        ax.set_title(SHORT[k], fontsize=9)
        ax.set_xlabel(r"$x / L$")
        ax.text(0.95, 0.95, "abcd"[k], transform=ax.transAxes, ha="right", va="top")
    axes[0].set_ylabel(r"$y / L$")
    cb = fig.colorbar(cf, ax=list(axes), orientation="vertical", fraction=0.025, pad=0.02, shrink=0.8)
    cb.set_ticks([0.0, 0.5, 1.0, smax])
    cb.set_ticklabels(["0", "0.5", "1", f"{smax:.2g}"])
    cb.set_label(r"$s_x / s_{\max}$", fontsize=8)
    cb.ax.tick_params(length=2, width=0.6, labelsize=7)
    cb.outline.set_linewidth(0.6)
    savefig(fig, "fig_slip_contours")
    plt.close(fig)

    # ---------------- figure 2: face pressure ----------------------------
    fig, axes = plt.subplots(4, 3, figsize=text_size(8.6), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.28, hspace=0.12))
    col_titles = [r"slip $s_x / s_{\max}$",
                  r"face pressure $p^+ = -\mathrm{tr}\,\sigma/3$ at $z = +3\varepsilon$",
                  r"sharp limit $-\frac{\mu(1+\nu)}{3(1-\nu)}\,\nabla_\parallel\!\cdot\!\Delta u$"]
    letters = "abcdefghijkl"
    print(f"face pressure at z = +3 eps (eps = {EPS/L:g} L), units mu s_max / L; interior = 0.15 L from the edges")
    rows = []
    for r, (name, s) in enumerate(CASES):
        p_plus = face_pressure(s, pts)
        p_pred = -coef * slip_divergence(s, pts)
        err = np.abs(p_plus[interior] - p_pred[interior]).max()
        print(f"  {SHORT[r]:14s}: interior p+ range [{p_plus[interior].min():+.3f}, {p_plus[interior].max():+.3f}], "
              f"prediction range [{p_pred[interior].min():+.3f}, {p_pred[interior].max():+.3f}], "
              f"max |clq - prediction| = {err:.3f}, edge peak |p+| = {np.abs(p_plus).max():.1f}")
        rows.append((p_plus, p_pred))
    # one colour scale for every row, set by the deep interior (edges saturate)
    vmax = max(max(float(np.percentile(np.abs(pp[interior]), 98)), float(np.abs(pr[interior]).max()))
               for pp, pr in rows)
    vmax = float(f"{vmax:.1g}")
    lev = np.linspace(-vmax, vmax, 21)
    for r, (name, s) in enumerate(CASES):
        p_plus, p_pred = rows[r]
        ax = axes[r, 0]
        fill(ax, triang, slips[r], np.linspace(0.0, smax, 15), "viridis")
        nd = clq.nodes(TRI, clq.order_from_count(len(s)))
        ax.plot(nd[:, 0], nd[:, 1], "o", ms=2.5, mfc="w", mec="k", mew=0.6)
        for c, vals in ((1, p_plus), (2, p_pred)):
            cf = fill(axes[r, c], triang, vals, lev, "RdBu_r", lines=(c == 1))
        for c in range(3):
            ax = axes[r, c]
            outline(ax)
            style(ax)
            ax.text(0.95, 0.95, letters[r * 3 + c], transform=ax.transAxes, ha="right", va="top")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8.5)
            if r == 3:
                ax.set_xlabel(r"$x / L$")
        axes[r, 0].set_ylabel(name + "\n" + r"$y / L$", fontsize=8)
    cb = fig.colorbar(cf, ax=list(axes[:, 2]), orientation="vertical", fraction=0.03, pad=0.03, shrink=0.5)
    cb.set_ticks([-vmax, -vmax / 2, 0.0, vmax / 2, vmax])
    cb.set_ticklabels([f"{-vmax:g}", f"{-vmax/2:g}", "0", f"{vmax/2:g}", f"{vmax:g}"])
    cb.ax.tick_params(length=2, width=0.6, labelsize=7)
    cb.outline.set_linewidth(0.6)
    cb.set_label(r"$p\;(\mu s_{\max}/L)$", fontsize=8)
    fig.text(0.5, 0.995, r"$p^- = -p^+$ on the other face; $p \equiv 0$ on $z = 0$; "
             r"$\varepsilon = 0.005\,L$, $\nu = 1/4$", ha="center", va="top", fontsize=8.5)
    savefig(fig, "fig_face_pressure")
    plt.close(fig)


if __name__ == "__main__":
    main()
