"""Demo: on-fault ELASTIC stress converges to a constant as eps -> 0.

A fault slip is an anelastic strain, so the mollified slip->stress kernel returns
the TOTAL stress inside the ~eps fault zone -- on the fault it is dominated by the
eigenstress, growing ~ (3/4) mu s / eps and DIVERGING as eps -> 0.  Subtracting
the exact finite-triangle anelastic eigenstress C:eps_star (the "eigen" kernel of
mbem.evaluate._stress_from_source) leaves the genuine ELASTIC stress.  This demo
shows, for a finite full-space strike-slip patch, that the
corrected on-fault stress at the fault interior:

  * converges to a CONSTANT (observed order ~ eps^2), and
  * converges to the PHYSICALLY CORRECT value -- the finite part of the classical
    (singular-kernel) dislocation stress, computed independently with cutde's
    full-space triangular-dislocation solution (tde_reference.py).

(The interior converges to a constant; the genuine elastic stress still
concentrates at the fault TIPS, which the edge-tapered subtraction preserves --
so the interior, not the peak, is the right convergence metric.)

Output (repo root): fig_onfault_convergence.png/.pdf
  (a) corrected center sigma_xy vs eps -> plateau on the cutde finite part
  (b) fault-normal sigma_xy(x) profiles: raw spikes (~1/eps), corrected collapse
      onto the classical cutde field, finite through the fault.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem.evaluate import _stress_from_source                     # noqa: E402
from tde_reference import classical_tde_stress                    # noqa: E402

from mbem.figures import save_figure                            # noqa: E402

try:
    from mbem.figures.style import set_paper_style
    set_paper_style()
except Exception:                                                  # noqa: BLE001
    pass

RAW_C = "#c1272d"     # warm -> raw (with anelastic term)
COR_C = "#1f5fa6"     # cool -> corrected (elastic)
REF_C = "0.25"        # cutde classical reference

MU, NU = 30.0, 0.25                  # GPa, Poisson 1/4
LAM = 2.0 * MU * NU / (1.0 - 2.0 * NU)   # the mbem drivers take (mu, lam)
SLIP_MAG = 1.0e-3                    # km == 1 m; b = +SLIP_MAG s_hat = u(+n) - u(-n)
#                                      (left-lateral on this n = +x, s_hat = +y fault)
L, D = 50.0, 50.0                    # fault half-length and depth (km)
TARGET_EDGE = 10.0                   # fault element size (km)
EPS_LADDER = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]   # km (center sweep)
PROFILE_EPS = [1.0, 0.5, 0.25]       # subset drawn on the profile panel
GPA_TO_MPA = 1.0e3


def dd_stress(obs, fault, slip, eps):
    """Total (raw) mollified slip->stress, (N,3,3), via the mbem assembler."""
    nt = fault.n_triangles
    return _stress_from_source(obs, fault, slip, "dd", MU, LAM,
                               np.full(nt, float(eps)))


def eigenstress(obs, fault, slip, eps):
    """Exact finite-triangle anelastic eigenstress +C:eps_star, (N,3,3)."""
    nt = fault.n_triangles
    return _stress_from_source(obs, fault, slip, "eigen", MU, LAM,
                               np.full(nt, float(eps)))


def figure(out_dir):
    fault, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=2.0 * L, depth_range=(-D, 0.0), target_edge=TARGET_EDGE)
    s_hat = np.asarray(s_hat, float)
    slip_vec = SLIP_MAG * s_hat
    slip = np.broadcast_to(slip_vec, (fault.n_triangles, 3))
    print(f"fault: {fault.n_triangles} triangles; n_hat={n_hat}, s_hat={s_hat}")

    # --- independent reference: classical TDE finite part at the fault center ---
    center = np.array([[0.0, 0.0, -0.5 * D]])
    rp = classical_tde_stress(center + [0.01, 0, 0], fault, slip_vec, MU, NU)
    rm = classical_tde_stress(center - [0.01, 0, 0], fault, slip_vec, MU, NU)
    sxy_ref = 0.5 * (rp[0, 0, 1] + rm[0, 0, 1]) * GPA_TO_MPA

    # --- (a) center-point convergence over the eps ladder ---
    raw_c, cor_c = [], []
    for eps in EPS_LADDER:
        tot = dd_stress(center, fault, slip, eps)
        cor = tot - eigenstress(center, fault, slip, eps)
        raw_c.append(tot[0, 0, 1] * GPA_TO_MPA)
        cor_c.append(cor[0, 0, 1] * GPA_TO_MPA)
    raw_c, cor_c = np.array(raw_c), np.array(cor_c)
    eps_arr = np.array(EPS_LADDER)

    # Richardson (eps^2) limit and observed order from the three finest.
    sigma0 = (4.0 * cor_c[-1] - cor_c[-2]) / 3.0
    order = float(np.log2(abs((cor_c[-3] - cor_c[-2]) / (cor_c[-2] - cor_c[-1]))))

    print("\n  on-fault CENTER sigma_xy (MPa): raw (total) vs corrected (elastic)")
    print(f"    {'eps (km)':>9} {'raw':>12} {'corrected':>12}")
    for eps, r, c in zip(EPS_LADDER, raw_c, cor_c):
        print(f"    {eps:9.4g} {r:12.4f} {c:12.6f}")
    print(f"\n    Richardson limit (eps^2)      : {sigma0:.6f} MPa")
    print(f"    observed convergence order p  : {order:.2f}")
    print(f"    cutde classical finite part   : {sxy_ref:.6f} MPa")
    print(f"    |corrected_limit - reference| : "
          f"{abs(sigma0 - sxy_ref):.2e} MPa\n")

    # --- (b) fault-normal profiles through the center ---
    xs = np.linspace(-3.0, 3.0, 41)
    obs = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, -0.5 * D)])
    prof_raw, prof_cor = [], []
    for eps in PROFILE_EPS:
        tot = dd_stress(obs, fault, slip, eps)
        cor = tot - eigenstress(obs, fault, slip, eps)
        prof_raw.append(tot[:, 0, 1] * GPA_TO_MPA)
        prof_cor.append(cor[:, 0, 1] * GPA_TO_MPA)
    # classical reference on the profile (mask the singular on-fault plane).
    ref_prof = classical_tde_stress(obs, fault, slip_vec, MU, NU)[:, 0, 1]
    ref_prof = ref_prof * GPA_TO_MPA
    ref_prof[np.abs(xs) < 0.05] = np.nan

    # ---------------------------- figure -----------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 4.3))

    # (a) convergence to a constant
    ax0.axhline(sxy_ref, color=REF_C, lw=1.0, ls="--",
                label="cutde classical (finite part)")
    ax0.semilogx(eps_arr, cor_c, "s-", color=COR_C, lw=1.4, ms=5,
                 label="corrected (elastic)")
    ax0.set_xlabel(r"$\varepsilon$ (km)")
    ax0.set_ylabel(r"on-fault center $\sigma_{xy}$ (MPa)")
    ax0.set_xticks([0.0625, 0.25, 1.0, 4.0])
    ax0.set_xticklabels(["0.06", "0.25", "1", "4"])
    span = max(abs(cor_c.max() - cor_c.min()), 1e-4)
    ax0.set_ylim(min(cor_c.min(), sxy_ref) - 0.6 * span,
                 max(cor_c.max(), sxy_ref) + 0.6 * span)
    ax0.set_box_aspect(1)
    ax0.legend(frameon=False, fontsize=8, loc="upper right")
    ax0.text(0.04, 0.5,
             f"raw $\\propto 1/\\varepsilon$\n(off scale: "
             f"{raw_c[-1]:.0f} MPa\nat $\\varepsilon={eps_arr[-1]:g}$)",
             transform=ax0.transAxes, ha="left", va="center", fontsize=7.5,
             color=RAW_C)
    ax0.text(0.96, 0.06,
             f"$\\sigma_0={sigma0:.4f}$ MPa\norder $p={order:.2f}$",
             transform=ax0.transAxes, ha="right", va="bottom", fontsize=7.5,
             color=COR_C)
    ax0.text(0.04, 0.96, "a", transform=ax0.transAxes, ha="left", va="top")

    # (b) profile collapse onto the classical field: corrected curves for all
    # eps (they collapse), cutde classical underneath, and ONE raw curve
    # (smallest eps) to show the off-scale spike.
    ax1.plot(xs, prof_raw[-1], "-", color=RAW_C, lw=1.0, alpha=0.8,
             label=fr"raw ($\varepsilon={PROFILE_EPS[-1]:g}$)")
    for k in range(len(PROFILE_EPS)):
        ax1.plot(xs, prof_cor[k], "-", color=COR_C, lw=1.3,
                 label="corrected (elastic)" if k == 0 else None)
    ax1.plot(xs, ref_prof, ":", color=REF_C, lw=1.8,
             label="cutde classical")
    ax1.axhline(0.0, color="0.6", lw=0.6)
    cor_all = np.concatenate(prof_cor)
    lo, hi = float(cor_all.min()), float(cor_all.max())
    s = max(hi - lo, 1e-6)
    ax1.set_ylim(lo - 0.35 * s, hi + 1.2 * s)
    ax1.set_xlim(-3, 3)
    ax1.set_xticks([-3, 0, 3])
    ax1.set_xlabel(r"$x$ (km, fault-normal)")
    ax1.set_ylabel(r"$\sigma_{xy}$ (MPa)")
    ax1.set_box_aspect(1)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.text(0.5, 0.97, r"raw spike $\to$ off-scale ($\propto 1/\varepsilon$)",
             transform=ax1.transAxes, ha="center", va="top", fontsize=7.5,
             color=RAW_C)
    ax1.text(0.04, 0.96, "b", transform=ax1.transAxes, ha="left", va="top")

    fig.suptitle(r"On-fault elastic stress converges to a constant as "
                 r"$\varepsilon\to0$ (matches classical TDE)",
                 fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written = save_figure(fig, out_dir, "onfault_convergence")
    plt.close(fig)
    return written

