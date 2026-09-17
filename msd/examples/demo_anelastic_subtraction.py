"""Demo: subtracting the anelastic (eigenstrain) term keeps on-fault stress finite.

A fault slip is an anelastic strain, so the mollified displacement-discontinuity
stress kernel returns the TOTAL stress inside the ~eps fault zone -- on the fault
it is dominated by the eigenstress, peaking at (3/4) mu s / eps and DIVERGING as
eps -> 0.  Subtracting the anelastic eigenstress C:eps_star (anelastic.py) leaves
the genuine ELASTIC stress, which is smooth, BOUNDED, and eps-independent in the
fault interior.

This is the full-space analogue of the elastic-vs-total correction: it evaluates
the on-fault shear of a finite strike-slip patch on a fault-normal profile through
the interior centroid, for an eps ladder, raw vs corrected.

Output (repo root): fig_anelastic_subtraction.png/.pdf
  (a) peak on-fault |sigma_xy| vs eps -- raw ~ 1/eps, corrected ~ flat (bounded)
  (b) fault-normal sigma_xy(x) profiles -- raw spikes (~1/eps), corrected collapse
"""
from __future__ import annotations

import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from anelastic import eigenstress_at_points                       # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mollified_kernel.analytical_kernels import (                 # noqa: E402
    analytical_stress_kernel,
)

try:
    from _paper_style import set_paper_style
    set_paper_style()
except Exception:
    pass

RAW_C = "#c1272d"     # warm  -> raw (with anelastic term)
COR_C = "#1f5fa6"     # cool  -> corrected (elastic only)

MU, NU = 30.0, 0.25                 # GPa, Poisson 1/4
SLIP_MAG = 1.0e-3                   # km == 1 m, right-lateral strike-slip
L, D = 50.0, 50.0                   # fault half-length and depth (km)
EPS_LADDER = [2.0, 1.0, 0.5, 0.25, 0.125]   # km
GPA_TO_MPA = 1.0e3


def fault_stress(obs, fault, slip_cart, mu, nu, eps):
    """Total stress (N,3,3) from the mollified DD stress kernel summed over the
    fault triangles: sigma_mn = sum_tri H[m,n,k] * slip_k."""
    obs = np.asarray(obs, float)
    verts = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    normals, _ = fault.normals_and_areas()
    slip_cart = np.asarray(slip_cart, float)
    sig = np.zeros((obs.shape[0], 3, 3))
    for i in range(obs.shape[0]):
        for m in range(verts.shape[0]):
            A, B, C = verts[m]
            H = analytical_stress_kernel(obs[i], A, B, C, normals[m],
                                         mu, nu, eps)        # (3,3,3)
            sig[i] += H @ slip_cart
    return sig


def main():
    # Vertical strike-slip fault in the y-z plane (normal +x, slip +y).
    fault, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=2.0 * L, depth_range=(-D, 0.0), target_edge=0.5 * L)
    slip_cart = SLIP_MAG * np.asarray(s_hat, float)
    print(f"fault: {fault.n_triangles} triangles; n_hat={n_hat}, s_hat={s_hat}")

    # Fault-normal profile through the interior centroid (x varies; y=0, z=-D/2).
    xs = np.linspace(-3.0, 3.0, 241)
    obs = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, -0.5 * D)])

    rows, prof_raw, prof_cor = [], [], []
    for eps in EPS_LADDER:
        sig_raw = fault_stress(obs, fault, slip_cart, MU, NU, eps)
        sig_cor = sig_raw - eigenstress_at_points(obs, fault, slip_cart,
                                                  MU, NU, eps)
        sxy_raw = sig_raw[:, 0, 1] * GPA_TO_MPA
        sxy_cor = sig_cor[:, 0, 1] * GPA_TO_MPA
        pk_raw, pk_cor = float(np.max(np.abs(sxy_raw))), float(np.max(np.abs(sxy_cor)))
        rows.append((eps, pk_raw, pk_cor, pk_raw / pk_cor))
        prof_raw.append(sxy_raw)
        prof_cor.append(sxy_cor)

    print("\n  on-fault interior |sigma_xy| (MPa): raw (with anelastic) vs "
          "corrected (elastic)")
    print(f"    {'eps (km)':>9} {'raw':>11} {'corrected':>11} {'raw/corr':>9}")
    for eps, pr, pc, ra in rows:
        print(f"    {eps:9.4g} {pr:11.4g} {pc:11.4g} {ra:9.3g}")
    print("    raw peak ~ (3/4) mu s / eps  ->  grows ~1/eps;   "
          "corrected peak  ->  bounded elastic stress drop\n")

    # --- figure ---
    eps_arr = np.array(EPS_LADDER, float)
    pk_raw = np.array([r[1] for r in rows])
    pk_cor = np.array([r[2] for r in rows])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 4.3))
    ax0.loglog(eps_arr, pk_raw, "o-", color=RAW_C, lw=1.4, ms=5,
               label="raw (with anelastic term)")
    ax0.loglog(eps_arr, pk_cor, "s-", color=COR_C, lw=1.4, ms=5,
               label="corrected (elastic)")
    ref = pk_raw[np.argmax(eps_arr)] * (eps_arr.max() / eps_arr)
    ax0.loglog(eps_arr, ref, "--", color="0.55", lw=0.9)
    ax0.text(eps_arr.min(), ref.max(), r"$\propto 1/\varepsilon$",
             color="0.4", ha="left", va="bottom", fontsize=8)
    ax0.set_xlabel(r"$\varepsilon$ (km)")
    ax0.set_ylabel(r"peak on-fault $|\sigma_{xy}|$ (MPa)")
    ax0.set_box_aspect(1)
    ax0.legend(frameon=False, fontsize=8, loc="lower left")
    ax0.text(0.96, 0.96, "a", transform=ax0.transAxes, ha="right", va="top")

    cor_all = np.concatenate(prof_cor)
    c_lo, c_hi = float(cor_all.min()), float(cor_all.max())
    span = max(c_hi - c_lo, 1e-9)
    for k in range(len(eps_arr)):
        ax1.plot(xs, prof_raw[k], "-", color=RAW_C, lw=1.0, alpha=0.85)
        ax1.plot(xs, prof_cor[k], "-", color=COR_C, lw=1.3)
    ax1.axhline(0.0, color="0.6", lw=0.6)
    ax1.set_ylim(c_lo - 0.35 * span, c_hi + 1.1 * span)
    ax1.set_xlim(-3, 3)
    ax1.set_xticks([-3, 0, 3])
    ax1.set_xlabel(r"$x$ (km, fault-normal)")
    ax1.set_ylabel(r"$\sigma_{xy}$ (MPa)")
    ax1.set_box_aspect(1)
    ax1.text(0.5, 0.97, r"raw spike $\to$ off-scale ($\propto 1/\varepsilon$)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=7.5,
             color=RAW_C)
    ax1.text(0.96, 0.96, "b", transform=ax1.transAxes, ha="right", va="top")

    fig.suptitle(r"Subtracting the anelastic term keeps on-fault stress finite "
                 r"as $\varepsilon\to0$", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        out = ROOT / f"fig_anelastic_subtraction.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
