"""Linear / quadratic slip vs a uniform subdivision into congruent order-0
sub-triangles (midpoint rule on the slip).

For p in {1, 2} the order-p closed form on the tilted TRI (nodal slip s_k,
eps = 0.1 L, three off-plane observers at |z| ~ 0.3 L) is compared with the
sum over the N^2 congruent sub-triangles of the barycentric lattice of the
ORDER-0 closed form with constant slip = clq.interpolate(TRI, s_k, centroid of
the sub-triangle).  Only the constant-slip machinery enters the reference, so
this is an independent check of the shape-function weighted tables.  The
midpoint rule converges as h^2 (~ N^-2): gate 1e-3 at N = 64 for u, the total
stress and the eigenstress, monotone decrease from N = 8 on, and the observed
order is printed (expected ~ 2).
"""
from __future__ import annotations

import time

import numpy as np

from _common import TRI, MU, Report, relmax
import clq

NU = 0.3
N_LIST = (2, 4, 8, 16, 32, 64)
GATE_N = 64
GATE_TOL = 1e-3
MONO_FROM = 8


def subdivide(tri, n):
    """The n^2 congruent sub-triangles (T, 3, 3) of the barycentric lattice
    with n subdivisions per edge, vertex order chosen so every sub-triangle
    has the parent's orientation (checked by the caller)."""
    v1, v2, v3 = tri

    def P(i, j):
        return (i * v1 + j * v2 + (n - i - j) * v3) / n

    subs = []
    for i in range(n):
        for j in range(n - i):
            subs.append([P(i, j), P(i + 1, j), P(i, j + 1)])           # "up"
            if j < n - i - 1:
                subs.append([P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])   # "down"
    return np.array(subs)


def subdivided_sum(obs, tri, slip_nodes, n, mu, nu, eps):
    """Sum over the n^2 sub-triangles of the order-0 clq result with constant
    slip equal to the interpolated slip at the sub-triangle centroid."""
    subs = subdivide(tri, n)
    cents = subs.mean(axis=1)                                   # (T, 3)
    slip_c = clq.interpolate(tri, slip_nodes, cents)            # (T, 3)
    u = np.zeros((len(obs), 3))
    sig = np.zeros((len(obs), 3, 3))
    eig = np.zeros((len(obs), 3, 3))
    for t in range(subs.shape[0]):
        s = slip_c[t]                                           # (3,) -> constant slip
        u += clq.displacement(obs, subs[t], s, mu, nu, eps)
        sig += clq.stress(obs, subs[t], s, mu, nu, eps, subtract_eigenstress=False)
        eig += clq.eigenstress(obs, subs[t], s, mu, nu, eps)
    return u, sig, eig


def fit_order(ns, errs):
    """Least-squares slope of log(err) vs log(N) (negated -> convergence order)."""
    x = np.log(np.asarray(ns, float))
    y = np.log(np.asarray(errs, float))
    return -np.polyfit(x, y, 1)[0]


def main():
    rep = Report("linear/quadratic slip vs order-0 uniform subdivision")
    fr = clq.local_frame(TRI)
    L = fr.L
    eps = 0.1 * L
    nhat = fr.nhat
    obs = np.array([fr.centroid + 0.3 * L * nhat,                                    # above
                    fr.centroid - 0.3 * L * nhat,                                    # below
                    fr.centroid + 0.3 * L * nhat + 0.4 * L * fr.e1 + 0.2 * L * fr.e2])  # oblique
    print(f"  TRI: L = {L:.4f}, area = {fr.area:.4f}, eps = 0.1 L = {eps:.4f}, nu = {NU}")
    print(f"  observers (z/L): {np.round(fr.to_plane(obs)[0] / L, 3)}")

    # geometry sanity of the subdivision: count, congruent areas, orientation
    geo_ok = True
    for n in N_LIST:
        subs = subdivide(TRI, n)
        areas = np.array([clq.local_frame(s).area for s in subs])
        normals = np.array([clq.unit_normal(s) for s in subs])
        geo_ok &= subs.shape[0] == n * n
        geo_ok &= np.allclose(areas, fr.area / n ** 2, rtol=1e-12, atol=0.0)
        geo_ok &= np.allclose(normals, nhat[None], atol=1e-12)
        # sub-triangle vertices tile the parent: every lattice vertex lies on TRI's plane
        z_sub, _ = fr.to_plane(subs.reshape(-1, 3))
        geo_ok &= np.max(np.abs(z_sub)) < 1e-12 * L
    rep.check_bool("subdivision: n^2 congruent sub-triangles, parent orientation", geo_ok)

    rng = np.random.default_rng(0)
    cases = {
        1: np.array([[1.0, 0.3, -0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),   # hat at v1
        2: rng.standard_normal((6, 3)),                                        # generic quadratic
    }
    for p, slip_nodes in cases.items():
        if p == 2:
            rep.check_bool("p=2: generic nodal slip, all 18 components nonzero",
                           np.all(slip_nodes != 0.0))
        t0 = time.time()
        u_ref = clq.displacement(obs, TRI, slip_nodes, MU, NU, eps)
        sig_ref = clq.stress(obs, TRI, slip_nodes, MU, NU, eps, subtract_eigenstress=False)
        eig_ref = clq.eigenstress(obs, TRI, slip_nodes, MU, NU, eps)
        # the interpolant must reproduce the nodal values (independent of clq.kernels)
        interp_nodes = clq.interpolate(TRI, slip_nodes, clq.nodes(TRI, p))
        rep.check(f"p={p}: interpolate reproduces nodal slip", relmax(interp_nodes, slip_nodes), 1e-13)

        print(f"\n  p = {p}: relative error of the order-0 subdivision sum vs the order-{p} closed form")
        print(f"  {'N':>4s} {'n_sub':>6s} {'u':>12s} {'sigma_tot':>12s} {'eigenstress':>12s}")
        errs = {"u": [], "sigma": [], "eig": []}
        for n in N_LIST:
            u_n, sig_n, eig_n = subdivided_sum(obs, TRI, slip_nodes, n, MU, NU, eps)
            eu, es, ee = relmax(u_n, u_ref), relmax(sig_n, sig_ref), relmax(eig_n, eig_ref)
            errs["u"].append(eu)
            errs["sigma"].append(es)
            errs["eig"].append(ee)
            print(f"  {n:4d} {n * n:6d} {eu:12.3e} {es:12.3e} {ee:12.3e}")
        print(f"  (p = {p} subdivision sums took {time.time() - t0:.1f} s)")

        idx_gate = N_LIST.index(GATE_N)
        idx_mono = N_LIST.index(MONO_FROM)
        ns_fit = N_LIST[idx_mono:]
        for name, label in (("u", "displacement"), ("sigma", "total stress"), ("eig", "eigenstress")):
            e = errs[name]
            order_ls = fit_order(ns_fit, e[idx_mono:])
            order_last = np.log2(e[idx_gate - 1] / e[idx_gate])
            print(f"  p={p} {label:13s}: observed order (LS fit N>={MONO_FROM}) = {order_ls:.2f}, "
                  f"(N={N_LIST[idx_gate - 1]}->{GATE_N}) = {order_last:.2f}")
            rep.check(f"p={p}: {label} error at N={GATE_N}", e[idx_gate], GATE_TOL)
            mono = all(e[k] > e[k + 1] for k in range(idx_mono, len(N_LIST) - 1))
            rep.check_bool(f"p={p}: {label} error decreasing monotonically from N={MONO_FROM}",
                           mono, f"(order ~ {order_ls:.2f})")
    rep.finish()


if __name__ == "__main__":
    main()
