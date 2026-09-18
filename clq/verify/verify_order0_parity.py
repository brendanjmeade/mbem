"""Constant-slip / constant-force (order 0) parity with the legacy oracles.

  * U and H vs the msd scalar `analytical_dd_displacement` / `analytical_stress_kernel`
    and the msd batch `dd_displacement_batch` at nu = 0.25 AND nu = 0.3 (msd
    carries the corrected traction-operator pairing since 2026-09-04), 1e-12;
  * U vs the moss scalar (corrected pairing, commit f721a6a) at both nu, 1e-12;
  * tripwire: msd's PRE-FIX contraction (lam and mu swapped on the first two
    terms, rebuilt here from `integrate_DG`) must differ by > 1e-2 at nu = 0.3
    and coincide at nu = 0.25 -- documents why gates at nu = 1/4 are blind;
  * FORCE (Kelvin single-layer) element: G and S vs the msd AND moss
    `analytical_kelvin_G` / `analytical_kelvin_stress` at both nu, 1e-12.  Those
    oracles take the observation point FIRST and carry no normal argument (the
    force kernels have no nhat dependence).  Measured worst over the five
    observers: G 5.0e-16, S 1.2e-15 at eps = 0.12, and G 3.1e-16, S 6.8e-16 at
    eps = 0 off the plane -- the shared `integrate_DG`/moment machinery of the
    three codebases agrees to rounding.
"""
from __future__ import annotations

import numpy as np

from _common import TRI, MU, Report, relmax, msd_analytical, msd_batch, moss_analytical
import clq
from clq.kernels import nodal_influence


def swapped_pre_fix_U(ak, obs, v1, v2, v3, normal, mu, nu, eps):
    """msd's pre-fix contraction (lam on the normal-derivative term)."""
    G1 = ak.integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    n = normal
    U = np.zeros((3, 3))
    for i in range(3):
        tr = sum(G1[i, m, m] for m in range(3))
        for j in range(3):
            t1 = lam * sum(n[m] * G1[i, j, m] for m in range(3))
            t2 = mu * n[j] * tr
            t3 = mu * sum(n[k] * G1[i, k, j] for k in range(3))
            U[i, j] = -(t1 + t2 + t3)
    return U


def main():
    rep = Report("order-0 parity with msd / moss oracles")
    msd = msd_analytical()
    msdb = msd_batch()
    moss = moss_analytical()
    v1, v2, v3 = TRI
    nrm = clq.unit_normal(TRI)
    obs = np.array([[0.60, -0.10, 0.60],
                    [2.00, 1.50, 3.00],
                    [-0.40, 0.00, -0.80],
                    v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),   # on-plane, inside
                    v1 + 1.2 * (v2 - v1) - 0.4 * (v3 - v1)])  # on-plane, outside
    eps = 0.12
    for nu in (0.25, 0.30):
        res = nodal_influence(obs, TRI, 0, MU, nu, eps, far_field="analytic")
        U = res["U"][:, 0]
        H = res["H"][:, 0]
        Um = np.array([msd.analytical_dd_displacement(o, v1, v2, v3, nrm, MU, nu, eps) for o in obs])
        Hm = np.array([msd.analytical_stress_kernel(o, v1, v2, v3, nrm, MU, nu, eps) for o in obs])
        Uo = np.array([moss.analytical_dd_displacement(o, v1, v2, v3, nrm, MU, nu, eps) for o in obs])
        Ub = msdb.dd_displacement_batch(v1, v2, v3, nrm, obs, MU, nu, eps)
        Uswap = np.array([swapped_pre_fix_U(msd, o, v1, v2, v3, nrm, MU, nu, eps) for o in obs])
        rep.check(f"nu={nu}: H vs msd analytical_stress_kernel", relmax(H, Hm), 1e-12)
        rep.check(f"nu={nu}: U vs moss analytical_dd_displacement", relmax(U, Uo), 1e-12)
        rep.check(f"nu={nu}: U vs msd analytical_dd_displacement (post-fix)", relmax(U, Um), 1e-12)
        rep.check(f"nu={nu}: U vs msd dd_displacement_batch (post-fix)", relmax(U, Ub), 1e-12)
        d = relmax(U, Uswap)
        if nu == 0.25:
            rep.check(f"nu={nu}: pre-fix swapped form coincides (lam == mu)", d, 1e-12)
        else:
            rep.check_bool(f"nu={nu}: tripwire, pre-fix swapped form differs by > 1e-2",
                           d > 1e-2, f"(rel diff {d:.2e})")

    # --- force (Kelvin single-layer) element --------------------------------
    # analytical_kelvin_G / analytical_kelvin_stress take (obs, v1, v2, v3, mu,
    # nu, eps): the observation point FIRST and no normal, because the force
    # kernels carry no nhat dependence at all.  S[i,j,k] is the stress ij per
    # unit force in direction k -- clq's index order for S as well.
    obs_off = obs[:3]                              # eps = 0 needs off-plane rows
    for nu in (0.25, 0.30):
        for eps_f, rows, tag in ((eps, obs, f"eps={eps}"), (0.0, obs_off, "eps=0 off-plane")):
            res = nodal_influence(rows, TRI, 0, MU, nu, eps_f, want=("G", "S"),
                                  far_field="analytic")
            G, S = res["G"][:, 0], res["S"][:, 0]
            Gm = np.array([msd.analytical_kelvin_G(o, v1, v2, v3, MU, nu, eps_f) for o in rows])
            Sm = np.array([msd.analytical_kelvin_stress(o, v1, v2, v3, MU, nu, eps_f) for o in rows])
            Go = np.array([moss.analytical_kelvin_G(o, v1, v2, v3, MU, nu, eps_f) for o in rows])
            So = np.array([moss.analytical_kelvin_stress(o, v1, v2, v3, MU, nu, eps_f) for o in rows])
            rep.check(f"nu={nu} {tag}: G vs msd analytical_kelvin_G", relmax(G, Gm), 1e-12)
            rep.check(f"nu={nu} {tag}: G vs moss analytical_kelvin_G", relmax(G, Go), 1e-12)
            rep.check(f"nu={nu} {tag}: S vs msd analytical_kelvin_stress", relmax(S, Sm), 1e-12)
            rep.check(f"nu={nu} {tag}: S vs moss analytical_kelvin_stress", relmax(S, So), 1e-12)
    rep.finish()


if __name__ == "__main__":
    main()
