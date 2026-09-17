"""Constant-slip (order 0) parity with the legacy oracles.

  * U and H vs the msd scalar `analytical_dd_displacement` / `analytical_stress_kernel`
    and the msd batch `dd_displacement_batch` at nu = 0.25 AND nu = 0.3 (msd
    carries the corrected traction-operator pairing since 2026-09-04), 1e-12;
  * U vs the moss scalar (corrected pairing, commit f721a6a) at both nu, 1e-12;
  * tripwire: msd's PRE-FIX contraction (lam and mu swapped on the first two
    terms, rebuilt here from `integrate_DG`) must differ by > 1e-2 at nu = 0.3
    and coincide at nu = 0.25 -- documents why gates at nu = 1/4 are blind.
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
    # Kelvin force kernels are not part of clq; nothing else to compare.
    rep.finish()


if __name__ == "__main__":
    main()
