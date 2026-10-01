"""Figure 1 — Pointwise singular vs mollified DD kernels.

We place an equilateral triangle of edge length L = 1 in the z = 0 plane,
centred at the origin, with outward normal +z. We then walk a vertical
line piercing the triangle through its centroid (an evaluation locus that
the singular kernels cannot tolerate) and a horizontal line offset slightly
above the triangle plane. Along these loci we evaluate

  - the pointwise mollified DD displacement kernel
        U_pt_{ij}(d) = - [ μ n_m DG_{ijm} + λ n_j DG_{imm} + μ n_k DG_{ikj} ]
  - the pointwise mollified DD stress kernel K_{mn,k}(d)

for ε ∈ {0, 0.05L, 0.1L, 0.5L}. ε = 0 reduces to the classical singular
Kelvin form. The figure shows that the singular kernels diverge near the
element while the mollified kernels remain bounded by O(ε^{-2}) (displacement)
and O(ε^{-3}) (stress).
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Repo root on path so we can import the existing modules.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

from _paper_style import (  # noqa: E402
    set_paper_style, text_size, panel_letter,
)
from mollified_kernel.analytical_kernels import kelvin_dG_pointwise  # noqa: E402
from mollified_kernel.mollified_elastic_kernels import dd_stress_kernel  # noqa: E402

set_paper_style()

# Suppress the expected divide-by-zero warnings emitted at the singular
# point (eps = 0, d = 0). Those values become +/-inf in the plot, which
# matplotlib clips at the axis limits — exactly the visual story we want.
np.seterr(divide="ignore", invalid="ignore")


def equilateral_triangle(L: float = 1.0):
    """Return vertices and outward (+z) normal of a unit-edge equilateral
    triangle centred at the origin in the z = 0 plane.
    """
    h = L * np.sqrt(3.0) / 2.0
    centroid_y = h / 3.0
    v1 = np.array([0.0, h - centroid_y, 0.0])
    v2 = np.array([-L / 2.0, -centroid_y, 0.0])
    v3 = np.array([L / 2.0, -centroid_y, 0.0])
    n = np.array([0.0, 0.0, 1.0])
    return v1, v2, v3, n


def dd_displacement_kernel_pt(d, normal, mu, nu, eps):
    """Pointwise mollified DD displacement kernel U[i,j] such that
    u_i(obs) per unit slip Δu_j (per unit source area) at d = obs - source.
    """
    DG = kelvin_dG_pointwise(d, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    U = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            term1 = mu * sum(n[m] * DG[i, j, m] for m in range(3))
            term2 = lam * n[j] * sum(DG[i, m, m] for m in range(3))
            term3 = mu * sum(n[k] * DG[i, k, j] for k in range(3))
            U[i, j] = -(term1 + term2 + term3)
    return U


def main():
    L = 1.0
    mu = 1.0
    nu = 0.25
    v1, v2, v3, n_vec = equilateral_triangle(L)
    centroid = (v1 + v2 + v3) / 3.0

    # Vertical line piercing the triangle through its centroid.
    z_line = np.linspace(-1.5 * L, 1.5 * L, 601)

    # Horizontal line slightly above the plane (z = 0.02 L), along x.
    x_line = np.linspace(-2.0 * L, 2.0 * L, 601)

    eps_list = [0.0, 0.05, 0.1, 0.5]
    # SKILL.md palette: black + restrained warm/cool. Black for singular
    # baseline; cool→warm for increasing eps.
    colors = ["#000000", "#1f77b4", "#d62728", "#ff7f0e"]

    fig, axes = plt.subplots(2, 2, figsize=text_size(5.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.10,
                         wspace=0.32, hspace=0.30)

    def style_panel(ax, letter, x_extent, x_ticks, y_lim, x_label, y_label):
        ax.set_box_aspect(1)
        ax.set_xlim(*x_extent)
        ax.set_xticks(x_ticks)
        ax.set_ylim(*y_lim)
        ax.set_yticks([y_lim[0], 0.0, y_lim[1]])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        panel_letter(ax, letter)

    # -------- (a) U_xx along vertical line through centroid ---------
    ax = axes[0, 0]
    for eps, c in zip(eps_list, colors):
        vals = np.zeros_like(z_line)
        for i, z in enumerate(z_line):
            obs = centroid + np.array([0.0, 0.0, z])
            d = obs - centroid
            U = dd_displacement_kernel_pt(d, n_vec, mu, nu, eps)
            vals[i] = U[0, 0]
        label = (r"$\varepsilon = 0$" if eps == 0.0
                  else rf"$\varepsilon/L = {eps}$")
        ax.plot(z_line / L, vals, color=c, label=label, lw=1.2)
    ax.axvline(0.0, color="0.6", lw=0.5, ls="--")
    style_panel(ax, "a", (-1.5, 1.5), [-1.5, 0, 1.5], (-20.0, 20.0),
                 r"$z / L$", r"$U^\varepsilon_{xx}$")
    ax.legend(loc="upper right", frameon=False, fontsize=7)

    # -------- (b) U_xx along horizontal line, z = 0.02 L -------------
    ax = axes[0, 1]
    for eps, c in zip(eps_list, colors):
        vals = np.zeros_like(x_line)
        z0 = 0.02 * L
        for i, x in enumerate(x_line):
            obs = centroid + np.array([x, 0.0, z0])
            d = obs - centroid
            U = dd_displacement_kernel_pt(d, n_vec, mu, nu, eps)
            vals[i] = U[0, 0]
        label = (r"$\varepsilon = 0$" if eps == 0.0
                  else rf"$\varepsilon/L = {eps}$")
        ax.plot(x_line / L, vals, color=c, label=label, lw=1.2)
    style_panel(ax, "b", (-2.0, 2.0), [-2, 0, 2], (-20.0, 20.0),
                 r"$x / L$ at $z = 0.02\,L$", r"$U^\varepsilon_{xx}$")

    # -------- (c) Stress K[0,2,0] (sigma_xz) vertical line -----------
    ax = axes[1, 0]
    for eps, c in zip(eps_list, colors):
        vals = np.zeros_like(z_line)
        for i, z in enumerate(z_line):
            obs = centroid + np.array([0.0, 0.0, z])
            K = dd_stress_kernel(obs, centroid, n_vec, mu, nu, eps)
            vals[i] = K[0, 2, 0]
        label = (r"$\varepsilon = 0$" if eps == 0.0
                  else rf"$\varepsilon/L = {eps}$")
        ax.plot(z_line / L, vals, color=c, label=label, lw=1.2)
    ax.axvline(0.0, color="0.6", lw=0.5, ls="--")
    style_panel(ax, "c", (-1.5, 1.5), [-1.5, 0, 1.5], (-2500.0, 2500.0),
                 r"$z / L$", r"$K^\varepsilon_{xz,x}$")

    # -------- (d) Stress K[0,2,0] horizontal line --------------------
    ax = axes[1, 1]
    for eps, c in zip(eps_list, colors):
        vals = np.zeros_like(x_line)
        z0 = 0.02 * L
        for i, x in enumerate(x_line):
            obs = centroid + np.array([x, 0.0, z0])
            K = dd_stress_kernel(obs, centroid, n_vec, mu, nu, eps)
            vals[i] = K[0, 2, 0]
        label = (r"$\varepsilon = 0$" if eps == 0.0
                  else rf"$\varepsilon/L = {eps}$")
        ax.plot(x_line / L, vals, color=c, label=label, lw=1.2)
    style_panel(ax, "d", (-2.0, 2.0), [-2, 0, 2], (-2500.0, 2500.0),
                 r"$x / L$ at $z = 0.02\,L$", r"$K^\varepsilon_{xz,x}$")

    out_dir = os.path.abspath(os.path.join(HERE, ".."))
    fig.savefig(os.path.join(out_dir, "fig_point_kernel.pdf"))
    fig.savefig(os.path.join(out_dir, "fig_point_kernel.png"))
    plt.close(fig)
    print("wrote fig_point_kernel.pdf/png")


if __name__ == "__main__":
    main()
