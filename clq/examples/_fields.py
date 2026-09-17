"""Shared pieces for the clq example figures: the representative slip
distributions, on-fault evaluation grids, chunked field evaluation and small
plotting helpers in the house style."""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clq  # noqa: E402

L = 1.0
MU, NU = 1.0, 0.25
TRI = clq.equilateral(L)
NHAT = clq.unit_normal(TRI)
ORDER_LABELS = ["constant", "linear", "quadratic"]


def representative_slips():
    """Nodal slips per order (all strike-slip along x, s_max = 1):
    constant: uniform; linear: vertex hat at v1; quadratic: dome (0 at the
    vertices, 1 at the edge midpoints, 4/3 at the centroid)."""
    return [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]] * 3 + [[1.0, 0.0, 0.0]] * 3),
    ]


def square_grid(n=161, half=0.6 * L):
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    return xs, X, Y, pts


def chunked(fn, pts, chunk=3000):
    out = [fn(pts[i:i + chunk]) for i in range(0, pts.shape[0], chunk)]
    return np.concatenate(out, axis=0)


def slip_field(slip, pts):
    """Prescribed slip on the grid, NaN outside the triangle."""
    s = clq.interpolate(TRI, slip, pts)
    s[~clq.inside(TRI, pts)] = np.nan
    return s


def draw_outline(ax, lw=0.6):
    x = np.r_[TRI[:, 0], TRI[0, 0]]
    y = np.r_[TRI[:, 1], TRI[0, 1]]
    ax.plot(x / L, y / L, color="k", lw=lw)


def style_panel(ax, half=0.6):
    ax.set_aspect("equal")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_xticks([-half, 0.0, half])
    ax.set_yticks([-half, 0.0, half])
    ax.tick_params(direction="out", length=3, width=0.8)


def column_colorbar(fig, axes_column, mappable, vmax, fmt="{:.2g}"):
    """One horizontal colorbar below a column of panels, three ticks."""
    cb = fig.colorbar(mappable, ax=list(axes_column), orientation="horizontal",
                      fraction=0.035, pad=0.10, shrink=0.85, aspect=18)
    cb.set_ticks([-vmax, 0.0, vmax])
    cb.set_ticklabels([fmt.format(-vmax), "0", fmt.format(vmax)])
    cb.ax.tick_params(length=2, width=0.6, labelsize=7)
    cb.outline.set_linewidth(0.6)
    return cb


def savefig(fig, name):
    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, f"{name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
