"""Regressions from the 2026-09-05 adversarial review.

  1. edge primitives are scale-free and NaN-free at extreme absolute scales
     (rho from 1e-8 to 1e8, tiny same-sign endpoint ratios) vs 40-digit mpmath;
  2. slip orders 5 and 6 (negative-n seeds I_{-3}) agree with quadrature;
  3. eps = 0 with an on-plane observer raises on every far_field path;
  4. translating triangle and observers by 1e6 L leaves the fields unchanged
     to double-precision rounding of the coordinates;
  5. eps >> L: the hybrid producer routes to quadrature (effective distance
     sqrt(D^2 + eps^2)) and agrees with the closed form to 1e-7.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np

from _common import TRI, MU, Report, relmax
import clq
from clq.primitives import edge_table
from clq.kernels import nodal_influence
from clq.quadrature import quadrature_influence


def mp_ref(k, m, ua, ub, rho2):
    mp.mp.dps = 40
    rho = mp.sqrt(rho2)
    f = lambda u: mp.mpf(u) ** k / (mp.mpf(u) ** 2 + mp.mpf(rho2)) ** (mp.mpf(m) / 2)
    pts = sorted(set([mp.mpf(ua), mp.mpf(ub)] + [x for x in (-10 * rho, -rho, 0, rho, 10 * rho) if ua < x < ub]))
    return float(mp.quad(f, pts))


def main():
    rep = Report("review regressions (scale-free primitives, orders 5-6, guards, invariance)")
    spec = {5: 5, 3: 4, 1: 2, -1: 0}
    cases = [(1e-9, 1.0, 100.0), (1e-7, 3e-7, 1e-16), (1e7, 3e7, 1e16), (1e10, 1e10 + 1, 1e-4),
             (2e-8, 3e-8, 1e-16), (-0.7, 0.9, 0.04), (1e3, 1e3 + 1, 1e-4), (0.05, 0.06, 1.0), (0.5, 1e5, 1e-4)]
    worst = 0.0
    finite = True
    for ua, ub, rho2 in cases:
        tab = edge_table(np.array([ua]), np.array([ub]), np.array([rho2]), spec)
        for m, kmax in spec.items():
            for k in range(kmax + 1):
                v = tab[m][0, k]
                finite &= bool(np.isfinite(v))
                r = mp_ref(k, m, ua, ub, rho2)
                worst = max(worst, abs(v - r) / max(abs(r), 1e-300))
    rep.check_bool("edge primitives finite at extreme scales", finite)
    rep.check("edge primitives vs mpmath, extreme scales", worst, 1e-12)

    obs = np.array([[0.6, -0.1, 0.6], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    for p in (5, 6):
        r = nodal_influence(obs, TRI, p, MU, 0.3, 0.3, far_field="analytic")
        q = quadrature_influence(obs, TRI, p, MU, 0.3, 0.3, n_gauss=60)
        rep.check(f"order {p}: U vs quadrature", relmax(r["U"], q["U"]), 1e-8)
        rep.check(f"order {p}: H vs quadrature", relmax(r["H"], q["H"]), 1e-8)
        rep.check(f"order {p}: E vs quadrature", relmax(r["E"], q["E"]), 1e-8)

    T = clq.equilateral(1.0)
    for ff in ("hybrid", "analytic", "quadrature"):
        try:
            clq.displacement(np.array([[0.05, 0.02, 0.0]]), T, [[1.0, 0, 0]], MU, 0.25, 0.0, far_field=ff)
            rep.check_bool(f"eps=0 on-plane raises (far_field={ff})", False)
        except ValueError:
            rep.check_bool(f"eps=0 on-plane raises (far_field={ff})", True)

    o = np.array([[0.1, 0.05, 0.0], [0.3, -0.2, 0.2]])
    s6 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    sh = np.array([1e6, -2e6, 3e6])
    u0 = clq.displacement(o, T, s6, MU, 0.3, 0.05)
    u1 = clq.displacement(o + sh, T + sh, s6, MU, 0.3, 0.05)
    rep.check("translation by 1e6 L: displacement invariant", relmax(u1, u0), 1e-7)
    g0 = clq.stress(o, T, s6, MU, 0.3, 0.05)
    g1 = clq.stress(o + sh, T + sh, s6, MU, 0.3, 0.05)
    rep.check("translation by 1e6 L: stress invariant", relmax(g1, g0), 1e-7)

    ua = clq.displacement(o, T, s6, MU, 0.3, 30.0, far_field="analytic")
    uq = clq.displacement(o, T, s6, MU, 0.3, 30.0, far_field="quadrature")
    uh = clq.displacement(o, T, s6, MU, 0.3, 30.0, far_field="hybrid")
    rep.check("eps = 30 L: hybrid == quadrature producer", relmax(uh, uq), 1e-14)
    rep.check("eps = 30 L: analytic vs quadrature", relmax(ua, uq), 1e-7)
    rep.finish()


if __name__ == "__main__":
    main()
