"""Far-field accuracy sweep of the closed-form ("analytic") moment producer,
the Gauss ("quadrature") producer and the "hybrid" crossover.

TRI, eps = 0.05 L, nu = 0.3, nodal orders p = 0, 1, 2 (generic nodal values,
seed 0).  Observers at D/L in {2, 5, 10, 20, 50, 100, 200, 500, 1000} from the
centroid in three directions: broadside (nhat), along the longest edge, and an
oblique global direction (0.3, 0.5, 0.81)/|.|.  Four per-node tensors of every
producer are compared with ``clq.quadrature.quadrature_influence`` (point
kernels x shape functions, n_gauss = 40 for D/L <= 10, 24 beyond) by
``relmax``: the SLIP tensors U (N,K,3,3) and H (N,K,3,3,3), and the FORCE
(Kelvin single-layer) tensors G (N,K,3,3) and S (N,K,3,3,3).

Gates (all four tensors, same thresholds)
  * analytic branch: relmax <= 1e-8 for every D/L <= defaults.D_STAR;
  * hybrid branch:   relmax <= 1e-8 for every D/L in the sweep;
  * quadrature producer: relmax <= 1e-9 for every D/L >= 5;
  * hybrid == analytic below D_STAR and == quadrature above (branch identity);
  * reference self-consistency (n_gauss 40 vs 60 at D/L = 2, 24 vs 40 at 20).

The force tensors lose digits with distance the same way the slip ones do, but
about an order of magnitude more slowly, so they need no separate thresholds:
within D_STAR = 10 the worst analytic-branch value at p = 2 is 2.9e-10 for
G/S against 4.6e-9 for U/H, and TOL_ANALYTIC = 1e-8 carries over unchanged.
The quadrature producer sits at ~5e-15 for G and S.  The slip and force gates
are reported separately so a regression in one family cannot hide behind the
other.

For information the sweep prints the D/L at which the analytic branch first
exceeds 1e-8 and 1e-6.  This documents the cancellation in the
divergence-theorem closed form that motivates D_STAR.  It grows roughly like
(D/L)^4-5 in-plane.  On this triangle the worst values are ~5e-9 at 10,
2e-4 at 100, 0.6 at 500 and ~6 at 1000, so the analytic branch is unusable
far out; the README "Far field" note has the thin-triangle numbers.
"""
from __future__ import annotations

import time

import numpy as np

from _common import TRI, MU, Report, relmax
from clq import defaults
from clq.frame import local_frame
from clq.kernels import nodal_influence
from clq.quadrature import quadrature_influence

NU = 0.3
EPS_OVER_L = 0.05
RATIOS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
ORDERS = (0, 1, 2)
WANT = ("U", "H", "G", "S")
N_REF_NEAR, N_REF_FAR = 40, 24        # reference rule for D/L <= 10 / beyond
TOL_ANALYTIC = 1e-8                   # within D_STAR
TOL_HYBRID = 1e-8                     # everywhere
TOL_QUAD = 1e-9                       # D/L >= QUAD_FROM
QUAD_FROM = 5.0
TOL_IDENTITY = 1e-12                  # hybrid vs the branch it should equal
TOL_REF_SELF = 1e-10                  # reference rule refinement


def directions(fr):
    v = fr.v
    edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]]
    e_long = edges[int(np.argmax([np.linalg.norm(e) for e in edges]))]
    e_long = e_long / np.linalg.norm(e_long)
    obl = np.array([0.3, 0.5, 0.81])
    obl = obl / np.linalg.norm(obl)
    return [("normal", fr.nhat), ("edge", e_long), ("oblique", obl)]


def reference(obs, near_mask, p, eps):
    """quadrature_influence with n_gauss = 40 (near) / 24 (far), all of WANT."""
    K = (p + 1) * (p + 2) // 2
    shapes = {"U": (3, 3), "H": (3, 3, 3), "G": (3, 3), "S": (3, 3, 3)}
    ref = {k: np.empty((obs.shape[0], K) + shapes[k]) for k in WANT}
    for mask, n in ((near_mask, N_REF_NEAR), (~near_mask, N_REF_FAR)):
        if np.any(mask):
            r = quadrature_influence(obs[mask], TRI, p, MU, NU, eps, n_gauss=n, want=WANT)
            for k in WANT:
                ref[k][mask] = r[k]
    return ref


def per_obs(a, b):
    """relmax per observation point (first axis)."""
    return np.array([relmax(a[n], b[n]) for n in range(a.shape[0])])


def main():
    t0 = time.perf_counter()
    rep = Report("far-field sweep: analytic / quadrature / hybrid producers vs 40x40 Gauss")
    fr = local_frame(TRI)
    L = fr.L
    eps = EPS_OVER_L * L
    dirs = directions(fr)
    print(f"  L = {L:.4f}, eps = {eps:.4f} (= {EPS_OVER_L} L), nu = {NU}, D_STAR = {defaults.D_STAR}")
    for name, d in dirs:
        print(f"  direction {name:8s} d.nhat = {d @ fr.nhat:+.3f}")

    # observation set: index n = i_ratio * 3 + i_dir
    obs = np.array([fr.centroid + r * L * d for r in RATIOS for _, d in dirs])
    ratio_of = np.array([r for r in RATIOS for _ in dirs], float)
    dir_of = [name for _ in RATIOS for name, _ in dirs]
    near_ref = ratio_of <= 10.0
    D_actual = np.linalg.norm(obs - fr.centroid, axis=1) / L

    rng = np.random.default_rng(0)
    rows = {}          # (ratio, dir, p) -> dict of relmax values
    worst = {}         # bookkeeping for the gates
    ref_self = {}
    for p in ORDERS:
        K = (p + 1) * (p + 2) // 2
        slip = rng.standard_normal((K, 3))
        ref = reference(obs, near_ref, p, eps)
        res = {}
        for ff in ("analytic", "quadrature", "hybrid"):
            res[ff] = nodal_influence(obs, TRI, p, MU, NU, eps, want=WANT, far_field=ff)
        # reference self-consistency: refine the rule at D/L = 2 (near) and 20 (far)
        for r_chk, n_ref, n_fine in ((2.0, N_REF_NEAR, 60), (20.0, N_REF_FAR, 40)):
            sel = ratio_of == r_chk
            fine = quadrature_influence(obs[sel], TRI, p, MU, NU, eps, n_gauss=n_fine, want=WANT)
            ref_self[(p, r_chk)] = max(relmax(fine[k], ref[k][sel]) for k in WANT)
        err = {k: {ff: per_obs(res[ff][k], ref[k]) for ff in res} for k in WANT}
        # slip-contracted fields (information only)
        u_ref = np.einsum("nkij,kj->ni", ref["U"], slip)
        s_ref = np.einsum("nkmlj,kj->nml", ref["H"], slip)
        u_an = np.einsum("nkij,kj->ni", res["analytic"]["U"], slip)
        s_an = np.einsum("nkmlj,kj->nml", res["analytic"]["H"], slip)
        e_slip = np.maximum(per_obs(u_an, u_ref), per_obs(s_an, s_ref))
        # branch identity of the hybrid producer, over all four tensors
        # hybrid crossover uses the effective distance sqrt(D^2 + eps^2) (clq.moments.weighted_tables)
        below = np.sqrt(D_actual ** 2 + EPS_OVER_L ** 2) <= defaults.D_STAR
        ident = np.zeros(obs.shape[0])
        for k in WANT:
            ident = np.maximum(ident, np.where(
                below, per_obs(res["hybrid"][k], res["analytic"][k]),
                per_obs(res["hybrid"][k], res["quadrature"][k])))
        worst[(p, "identity")] = float(ident.max())
        for n in range(obs.shape[0]):
            rows[(ratio_of[n], dir_of[n], p)] = dict(
                U_an=err["U"]["analytic"][n], H_an=err["H"]["analytic"][n],
                U_qd=err["U"]["quadrature"][n], H_qd=err["H"]["quadrature"][n],
                G_an=err["G"]["analytic"][n], S_an=err["S"]["analytic"][n],
                G_qd=err["G"]["quadrature"][n], S_qd=err["S"]["quadrature"][n],
                hyb=max(err[k]["hybrid"][n] for k in WANT),
                slip=e_slip[n])

    # ---- table -------------------------------------------------------------
    print()
    print(f"  {'D/L':>6s} {'dir':8s} {'p':>2s} | {'analytic U':>11s} {'analytic H':>11s} "
          f"{'analytic G':>11s} {'analytic S':>11s} | {'quadr. U':>11s} {'quadr. H':>11s} "
          f"{'quadr. G':>11s} {'quadr. S':>11s} | {'hybrid':>9s} | {'slip-contr. an.':>15s}")
    print("  " + "-" * 150)
    for r in RATIOS:
        for name, _ in dirs:
            for p in ORDERS:
                v = rows[(float(r), name, p)]
                print(f"  {r:6d} {name:8s} {p:2d} | {v['U_an']:11.2e} {v['H_an']:11.2e} "
                      f"{v['G_an']:11.2e} {v['S_an']:11.2e} | "
                      f"{v['U_qd']:11.2e} {v['H_qd']:11.2e} "
                      f"{v['G_qd']:11.2e} {v['S_qd']:11.2e} | {v['hyb']:9.2e} | {v['slip']:15.2e}")
        print("  " + "-" * 150)

    # ---- information: where the analytic branch first exceeds 1e-8 / 1e-6 ----
    an_by_ratio = {r: max(rows[(float(r), nm, p)][f"{k}_an"]
                          for nm, _ in dirs for p in ORDERS for k in ("U", "H", "G", "S"))
                   for r in RATIOS}
    print()
    print("  analytic branch, worst over directions and orders:")
    print("   " + "  ".join(f"D/L={r}: {an_by_ratio[r]:.1e}" for r in RATIOS))
    for thr in (1e-8, 1e-6):
        hit = [r for r in RATIOS if an_by_ratio[r] > thr]
        where = f"D/L = {hit[0]}" if hit else "never within the sweep (D/L <= 1000)"
        print(f"  analytic branch first exceeds {thr:.0e} at {where}")
    print()

    # ---- gates (per slip order) ------------------------------------------------
    for p in ORDERS:
        sub = {k: v for k, v in rows.items() if k[2] == p}

        def w(pred, keys, sub=sub):
            best, arg = -1.0, None
            for (r, nm, pp), v in sub.items():
                if not pred(r):
                    continue
                val = max(v[k] for k in keys)
                if val > best:
                    best, arg = val, (r, nm)
            return best, arg

        for fam, an_keys, qd_keys in (("slip U/H", ("U_an", "H_an"), ("U_qd", "H_qd")),
                                      ("force G/S", ("G_an", "S_an"), ("G_qd", "S_qd"))):
            val, arg = w(lambda r: r <= defaults.D_STAR, an_keys)
            rep.check(f"p={p}: analytic branch {fam}, D/L <= {defaults.D_STAR:g}",
                      val, TOL_ANALYTIC, f"worst at D/L={arg[0]:g} {arg[1]}")
            val, arg = w(lambda r: r >= QUAD_FROM, qd_keys)
            rep.check(f"p={p}: quadrature producer {fam}, D/L >= {QUAD_FROM:g}",
                      val, TOL_QUAD, f"worst at D/L={arg[0]:g} {arg[1]}")
        val, arg = w(lambda r: True, ("hyb",))
        rep.check(f"p={p}: hybrid branch (all four tensors), all D/L", val, TOL_HYBRID,
                  f"worst at D/L={arg[0]:g} {arg[1]}")
        rep.check(f"p={p}: hybrid == analytic below / quadrature above D_STAR "
                  f"(all four tensors)", worst[(p, "identity")], TOL_IDENTITY)
    for p in ORDERS:
        rep.check(f"p={p}: reference 40x40 vs 60x60 at D/L = 2",
                  ref_self[(p, 2.0)], TOL_REF_SELF)
        rep.check(f"p={p}: reference 24x24 vs 40x40 at D/L = 20",
                  ref_self[(p, 20.0)], TOL_REF_SELF)
    print(f"  runtime {time.perf_counter() - t0:.1f} s")
    rep.finish()


if __name__ == "__main__":
    main()
