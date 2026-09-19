"""Demo: ON-FAULT elastic stress from a mollified BEM solution.

The mbem BEM is solved in DISPLACEMENT (a fault enters as a smooth slip->
displacement source on the RHS), so the eigenstress never pollutes the solve.
The anelastic subtraction belongs in the STRESS READOUT: ``evaluate_stress``
sums the boundary (free surface + base) and fault stress kernels and then
removes the fault eigenstress, giving the genuine ELASTIC on-fault stress --
the Coulomb-relevant field the BEM could not previously produce.

The fault box has a FREE TOP surface at z=0, i.e. it is a half-space, so the
independent reference is cutde's half-space triangular dislocation
(tde_reference.py, halfspace=True).  We show:

  (a) on-fault elastic shear at the fault center converges with eps (a scalar
      ladder, then the production spec eps="auto") and lands
      on the half-space value (to box-truncation accuracy);
  (b) the down-dip on-fault elastic shear profile at the production eps
      matches the half-space dislocation; the full-space TDE is drawn too, so
      the gap between them is the free-surface contribution the BEM captures.

The mbem fault (-H@slip) and cutde's TDE use OPPOSITE slip-sign conventions, so
the reference is sign-aligned to the BEM with a global g = +-1 measured at one
off-fault point.  eps must be small relative to the fault (so the planar-fault
eigenstress marginal is valid and the fault interior is many eps from the free
surface) -- the opposite regime to a free-surface displacement plot.

Output (repo root): fig_bem_onfault_stress.png/.pdf
"""
from __future__ import annotations

import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mollified_bem as mb                                        # noqa: E402
from _fault_box import build_fault_box, build_model  # noqa: E402
from mbem.backends.dense import AssembledDense                   # noqa: E402
from mbem.evaluate import evaluate_stress                        # noqa: E402
from mbem.kernels import basis as kb                             # noqa: E402
from mbem.model import generate_system                           # noqa: E402
from tde_reference import classical_tde_stress                   # noqa: E402

try:
    from _paper_style import set_paper_style
    set_paper_style()
except Exception:
    pass

BEM_C = "#1f5fa6"     # cool -> BEM elastic
HS_C = "#117733"      # green -> half-space reference
FS_C = "0.45"         # gray -> full-space reference

MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)     # nu = 1/4
SLIP_MAG = 0.01                                  # km == 10 m
GPA_TO_MPA = 1.0e3

FAULT_HALF_LEN = 20.0
FAULT_DEPTH = 20.0
EDGE_FAULT = 2.0                                  # fault element size h (km)
EPS_LADDER = [4.0, 3.0, 2.0, 1.5, 1.0]           # km scalars; eps="auto" last


def resolved_shear(sig, n_hat, s_hat):
    """On-fault shear traction resolved in the slip direction (N,)."""
    return np.einsum("nij,i,j->n", sig, n_hat, s_hat)


def main():
    meshes = build_fault_box(
        half_x=120.0, z_bottom=-80.0, fault_half_len=FAULT_HALF_LEN,
        fault_depth=FAULT_DEPTH, edge_fault=EDGE_FAULT, edge_near=20.0,
        edge_far=60.0, edge_side=60.0, near_field_radius=60.0)
    n_hat = np.asarray(meshes["n_hat"], float)
    s_hat = np.asarray(meshes["s_hat"], float)
    slip_vec = SLIP_MAG * s_hat
    fault = meshes["fault"]

    model = build_model(meshes, SLIP_MAG, MAT)
    system = generate_system(model)
    region = model.regions[0]
    nt = {k: meshes[k].n_triangles for k in ("top", "sides", "base", "fault")}
    print(f"meshes: {nt}; unknowns: {system.layout.n_unknowns}", flush=True)

    def ref_shear(obs, halfspace):
        sig = classical_tde_stress(obs, fault, slip_vec, MAT.mu, MAT.nu,
                                   halfspace=halfspace)
        return resolved_shear(sig, n_hat, s_hat) * GPA_TO_MPA

    # On-fault observation sets.
    c = fault.centroids()
    cz = c[:, 2]
    target = np.array([0.0, 0.0, -0.5 * FAULT_DEPTH])
    center = c[int(np.argmin(np.linalg.norm(c - target, axis=1)))][None]
    # Down-dip line at mid-strike, INTERIOR only: drop the shallowest and
    # deepest element rows, where the tip stress concentration + the
    # surface-breaking edge are resolution-limited in every method.
    margin = 1.5 * EDGE_FAULT
    band = (np.abs(c[:, 1]) < EDGE_FAULT) & (cz < -margin) \
        & (cz > -(FAULT_DEPTH - margin))
    dd_idx = np.where(band)[0]
    dd_idx = dd_idx[np.argsort(cz[dd_idx])]
    dd = c[dd_idx]

    # Reconcile the BEM<->cutde slip-sign convention at one off-fault point.
    off = np.array([[15.0, 6.0, -28.0]])
    eps_prod = "auto"
    eps_f_prod = float(kb.resolve_patch_eps(eps_prod, region.faults[0])[0])
    asm0 = AssembledDense(system, eps_prod, "direct", jump="calibrated")
    sol0 = asm0.solve()
    ev_off = resolved_shear(evaluate_stress(model, region, sol0, off, eps_prod,
                                            subtract_anelastic=True),
                            n_hat, s_hat)[0] * GPA_TO_MPA
    g = float(np.sign(ev_off * ref_shear(off, True)[0]))
    print(f"  slip-sign reconciliation: g = {g:+.0f}", flush=True)

    hs_center = g * ref_shear(center, True)[0]

    # ---- (a) eps sweep: BEM on-fault elastic shear at the center ----
    # (spec, fault eps for the axis): the scalar ladder, then eps="auto".
    ladder = [(e, e) for e in EPS_LADDER] + [(eps_prod, eps_f_prod)]
    eps_axis = [e for _, e in ladder]
    print("\n  BEM on-fault CENTER elastic shear vs eps")
    print(f"    {'eps_f (km)':>10} {'tau_center (MPa)':>18}")
    tau_center = []
    for eps, eps_f in ladder:
        t0 = time.time()
        asm = AssembledDense(system, eps, "direct", jump="calibrated")
        sol = asm.solve()
        sig = evaluate_stress(model, region, sol, center, eps,
                              subtract_anelastic=True)
        tau = resolved_shear(sig, n_hat, s_hat)[0] * GPA_TO_MPA
        tau_center.append(tau)
        print(f"    {eps_f:10.3g} {tau:18.5f}   ({time.time()-t0:.0f}s)",
              flush=True)
    tau_center = np.array(tau_center)
    print(f"\n    half-space TDE reference      : {hs_center:.5f} MPa")
    print(f"    |BEM(finest) - half-space|    : "
          f"{abs(tau_center[-1] - hs_center):.3e} MPa "
          f"({abs(tau_center[-1]/hs_center - 1)*100:.1f} %)\n")

    # ---- (b) down-dip profile at the production eps (the last rung's sol) --
    eps_f = eps_f_prod
    sig_dd = evaluate_stress(model, region, sol, dd, eps_prod,
                             subtract_anelastic=True)
    tau_dd = resolved_shear(sig_dd, n_hat, s_hat) * GPA_TO_MPA
    tau_hs = g * ref_shear(dd, True)
    tau_fs = g * ref_shear(dd, False)
    depth = -dd[:, 2]

    # ------------------------------ figure ---------------------------------
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 4.3))

    ax0.axhline(hs_center, color=HS_C, lw=1.0, ls="--",
                label="half-space TDE")
    ax0.plot(eps_axis, tau_center, "s-", color=BEM_C, lw=1.4, ms=5,
             label="BEM (elastic)")
    ax0.set_xlabel(r"fault $\varepsilon$ (km)")
    ax0.set_ylabel(r"on-fault center shear $\tau$ (MPa)")
    ax0.set_xticks([0, 1, 2, 3, 4])
    sp = max(abs(tau_center.max() - tau_center.min()), 1e-3)
    ax0.set_ylim(min(tau_center.min(), hs_center) - 1.2 * sp,
                 max(tau_center.max(), hs_center) + 1.2 * sp)
    ax0.set_box_aspect(1)
    ax0.legend(frameon=False, fontsize=8, loc="best")
    ax0.text(0.96, 0.06,
             f"$\\to$ half-space\n({abs(tau_center[-1]/hs_center - 1)*100:.1f}% "
             f"at $\\varepsilon={eps_f:.2g}$;\nrest is box truncation)",
             transform=ax0.transAxes, ha="right", va="bottom", fontsize=7.0,
             color=BEM_C)
    ax0.text(0.04, 0.96, "a", transform=ax0.transAxes, ha="left", va="top")

    ax1.plot(tau_fs, depth, ":", color=FS_C, lw=1.4, label="full-space TDE")
    ax1.plot(tau_hs, depth, "--", color=HS_C, lw=1.4, label="half-space TDE")
    ax1.plot(tau_dd, depth, "o-", color=BEM_C, lw=1.3, ms=3.5,
             label=fr"BEM ($\varepsilon={eps_f:.2g}$)")
    ax1.invert_yaxis()
    ax1.set_xlabel(r"on-fault shear $\tau$ (MPa)")
    ax1.set_ylabel(r"depth (km)")
    ax1.set_box_aspect(1)
    ax1.legend(frameon=False, fontsize=8, loc="best")
    ax1.text(0.04, 0.96, "b", transform=ax1.transAxes, ha="left", va="top")

    fig.suptitle(r"BEM on-fault elastic stress: converges in $\varepsilon$ and "
                 r"matches the half-space dislocation", fontsize=10.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        out = ROOT / f"fig_bem_onfault_stress.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
