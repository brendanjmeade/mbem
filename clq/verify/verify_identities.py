"""Algebraic identities of the nodal influence tensors and the public API on
the tilted triangle (nu = 0.3, eps = 0.12, far_field = "analytic", four
observers: three off-plane and one on the plane inside the triangle):

  1. partition of unity: sum_k U/H/E[:, k] at order p (P1, P2) equals the
     order-0 tensor, 1e-13 relative;
  2. P1 embedded in P2: P2 nodal values interpolated from a P1 nodal slip give
     the same displacement / total stress / eigenstress / elastic stress, 1e-13;
  3. cyclic vertex relabel (v2, v3, v1) with correspondingly permuted nodal
     values leaves the nodal tensors and the fields unchanged, 1e-13;
  4. reversed orientation (v1, v3, v2): with the slip negated u, sigma_total
     and C:eps* are unchanged; with the slip kept, u and sigma_total flip sign
     (nodal level: U, H odd and E even in nhat), 1e-13;
  5. rigid motion x -> Q x + t of triangle and observers: U' = Q U Q^T,
     H'_{mlj} = Q_ma Q_lb Q_jc H_abc, E' = E, 1e-12;
  6. scaling alpha of (tri, obs, eps): U invariant, H and E scale as 1/alpha,
     1e-12;
  7. interpolate(nodal_values(f)) reproduces polynomials of degree <= p in the
     in-plane coordinates at random points (P1 affine, P2 quadratic), 1e-13.
"""
from __future__ import annotations

import numpy as np

from _common import TRI, MU, Report, relmax, random_rotation
import clq
from clq.kernels import nodal_influence

NU = 0.3
EPS = 0.12
FF = "analytic"
ORDERS = (0, 1, 2)
KEYS = ("U", "H", "E")


def observers(tri):
    v1, v2, v3 = tri
    return np.array([[0.60, -0.10, 0.60],
                     [2.00, 1.50, 3.00],
                     [-0.40, 0.00, -0.80],
                     v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1)])   # on-plane, inside


def node_permutation(tri_a, tri_b, p):
    """perm such that nodes(tri_b, p) == nodes(tri_a, p)[perm]."""
    na = clq.nodes(tri_a, p)
    nb = clq.nodes(tri_b, p)
    perm = []
    for x in nb:
        d = np.linalg.norm(na - x, axis=1)
        k = int(np.argmin(d))
        if d[k] > 1e-12:
            raise RuntimeError("node sets of the relabelled triangles differ")
        perm.append(k)
    perm = np.array(perm)
    if len(set(perm.tolist())) != len(perm):
        raise RuntimeError("node permutation is not a bijection")
    return perm


def fields(obs, tri, slip):
    """Displacement, total stress, eigenstress C:eps*, elastic stress."""
    return {
        "u": clq.displacement(obs, tri, slip, MU, NU, EPS, far_field=FF),
        "sigma_total": clq.stress(obs, tri, slip, MU, NU, EPS,
                                  subtract_eigenstress=False, far_field=FF),
        "C:eps*": clq.eigenstress(obs, tri, slip, MU, NU, EPS, far_field=FF),
        "sigma_elastic": clq.stress(obs, tri, slip, MU, NU, EPS, far_field=FF),
    }


def influence_all(obs, tri):
    return {p: nodal_influence(obs, tri, p, MU, NU, EPS, want=KEYS, far_field=FF)
            for p in ORDERS}


def main():
    rep = Report("algebraic identities (nu = 0.3, eps = 0.12, tilted triangle)")
    rng = np.random.default_rng(7)
    obs = observers(TRI)
    inf = influence_all(obs, TRI)
    slips = {p: rng.standard_normal((clq.n_nodes(p), 3)) for p in ORDERS}
    fld = {p: fields(obs, TRI, slips[p]) for p in ORDERS}

    # ------------------------------------------------------------------ (1)
    for p in (1, 2):
        for key in KEYS:
            rep.check(f"(1) P{p} partition of unity, {key}",
                      relmax(inf[p][key].sum(axis=1), inf[0][key][:, 0]), 1e-13)
    # shape-function level: sum_k N_k = 1 at random points
    fr = clq.local_frame(TRI)
    Xr = rng.uniform(-1.5 * fr.L, 1.5 * fr.L, size=(40, 2))
    pts = fr.from_plane(Xr, rng.uniform(-1.0, 1.0, size=40))
    for p in ORDERS:
        Nk = clq.shape_functions(TRI, p, pts)
        rep.check(f"(1) P{p} sum_k N_k = 1 at random points",
                  np.max(np.abs(Nk.sum(axis=1) - 1.0)), 1e-13)

    # ------------------------------------------------------------------ (2)
    s1 = slips[1]
    s2 = clq.interpolate(TRI, s1, clq.nodes(TRI, 2))
    f2 = fields(obs, TRI, s2)
    for name in fld[1]:
        rep.check(f"(2) P1 embedded in P2: {name}", relmax(f2[name], fld[1][name]), 1e-13)
    # constant slip embedded in P1 and P2
    s0 = slips[0]
    for p in (1, 2):
        sp = np.repeat(s0, clq.n_nodes(p), axis=0)
        fp = fields(obs, TRI, sp)
        rep.check(f"(2) P0 embedded in P{p}: worst over u/sigma/C:eps*",
                  max(relmax(fp[k], fld[0][k]) for k in fp), 1e-13)

    # ------------------------------------------------------------------ (3)
    tri_c = TRI[[1, 2, 0]]
    rep.check("(3) cyclic relabel keeps nhat",
              relmax(clq.unit_normal(tri_c), clq.unit_normal(TRI)), 1e-15)
    inf_c = influence_all(obs, tri_c)
    expected_perm = {0: [0], 1: [1, 2, 0], 2: [1, 2, 0, 4, 5, 3]}
    for p in ORDERS:
        perm = node_permutation(TRI, tri_c, p)
        rep.check_bool(f"(3) P{p} cyclic node permutation is {expected_perm[p]}",
                       perm.tolist() == expected_perm[p], f"(got {perm.tolist()})")
        for key in KEYS:
            rep.check(f"(3) P{p} cyclic relabel, nodal {key}",
                      relmax(inf_c[p][key], inf[p][key][:, perm]), 1e-13)
        fc = fields(obs, tri_c, slips[p][perm])
        rep.check(f"(3) P{p} cyclic relabel, fields (worst of 4)",
                  max(relmax(fc[k], fld[p][k]) for k in fc), 1e-13)

    # ------------------------------------------------------------------ (4)
    tri_r = TRI[[0, 2, 1]]
    rep.check("(4) reversed orientation flips nhat",
              relmax(clq.unit_normal(tri_r), -clq.unit_normal(TRI)), 1e-15)
    inf_r = influence_all(obs, tri_r)
    expected_perm = {0: [0], 1: [0, 2, 1], 2: [0, 2, 1, 5, 4, 3]}
    for p in ORDERS:
        perm = node_permutation(TRI, tri_r, p)
        rep.check_bool(f"(4) P{p} reversed node permutation is {expected_perm[p]}",
                       perm.tolist() == expected_perm[p], f"(got {perm.tolist()})")
        rep.check(f"(4) P{p} reversed, nodal U -> -U",
                  relmax(inf_r[p]["U"], -inf[p]["U"][:, perm]), 1e-13)
        rep.check(f"(4) P{p} reversed, nodal H -> -H",
                  relmax(inf_r[p]["H"], -inf[p]["H"][:, perm]), 1e-13)
        rep.check(f"(4) P{p} reversed, nodal E -> +E",
                  relmax(inf_r[p]["E"], inf[p]["E"][:, perm]), 1e-13)
        # invariant statement: reverse orientation AND negate the slip
        fneg = fields(obs, tri_r, -slips[p][perm])
        for name in fneg:
            rep.check(f"(4) P{p} reversed + negated slip: {name} unchanged",
                      relmax(fneg[name], fld[p][name]), 1e-13)
        # same slip, reversed orientation: odd fields flip sign
        fpos = fields(obs, tri_r, slips[p][perm])
        for name in ("u", "sigma_total", "C:eps*"):
            rep.check(f"(4) P{p} reversed, slip kept: {name} flips sign",
                      relmax(fpos[name], -fld[p][name]), 1e-13)

    # ------------------------------------------------------------------ (5)
    Q = random_rotation(3)
    t = np.array([0.7, -1.3, 0.4])
    rep.check("(5) Q is a proper rotation", max(relmax(Q @ Q.T, np.eye(3)),
                                               abs(np.linalg.det(Q) - 1.0)), 1e-14)
    tri_q = TRI @ Q.T + t
    obs_q = obs @ Q.T + t
    inf_q = influence_all(obs_q, tri_q)
    for p in ORDERS:
        rep.check(f"(5) P{p} rigid motion, U' = Q U Q^T",
                  relmax(inf_q[p]["U"], np.einsum("ia,jb,nkab->nkij", Q, Q, inf[p]["U"])), 1e-12)
        rep.check(f"(5) P{p} rigid motion, H' = Q Q Q H",
                  relmax(inf_q[p]["H"], np.einsum("ma,lb,jc,nkabc->nkmlj", Q, Q, Q, inf[p]["H"])), 1e-12)
        rep.check(f"(5) P{p} rigid motion, E' = E",
                  relmax(inf_q[p]["E"], inf[p]["E"]), 1e-12)
        rep.check(f"(5) P{p} rigid motion, nodes' = Q nodes + t",
                  relmax(inf_q[p]["nodes"], inf[p]["nodes"] @ Q.T + t), 1e-14)
    # API level: rotated slip vectors give rotated fields
    fq = fields(obs_q, tri_q, slips[2] @ Q.T)
    rep.check("(5) P2 rigid motion, u' = Q u", relmax(fq["u"], fld[2]["u"] @ Q.T), 1e-12)
    for name in ("sigma_total", "C:eps*", "sigma_elastic"):
        rep.check(f"(5) P2 rigid motion, {name}' = Q sigma Q^T",
                  relmax(fq[name], np.einsum("ia,jb,nab->nij", Q, Q, fld[2][name])), 1e-12)

    # ------------------------------------------------------------------ (6)
    alpha = 3.7
    inf_s = {p: nodal_influence(alpha * obs, alpha * TRI, p, MU, NU, alpha * EPS,
                                want=KEYS, far_field=FF) for p in ORDERS}
    for p in ORDERS:
        rep.check(f"(6) P{p} scaling, U invariant", relmax(inf_s[p]["U"], inf[p]["U"]), 1e-12)
        rep.check(f"(6) P{p} scaling, H -> H/alpha", relmax(inf_s[p]["H"], inf[p]["H"] / alpha), 1e-12)
        rep.check(f"(6) P{p} scaling, E -> E/alpha", relmax(inf_s[p]["E"], inf[p]["E"] / alpha), 1e-12)

    # ------------------------------------------------------------------ (7)
    def basis(P, p):
        _, X = fr.to_plane(P)
        cols = [np.ones(X.shape[0])]
        if p >= 1:
            cols += [X[:, 0], X[:, 1]]
        if p >= 2:
            cols += [X[:, 0] ** 2, X[:, 0] * X[:, 1], X[:, 1] ** 2]
        return np.stack(cols, axis=1)

    for p in ORDERS:
        coef = rng.standard_normal((basis(pts[:1], p).shape[1], 3))
        f = lambda P, p=p, coef=coef: basis(P, p) @ coef
        vals = clq.nodal_values(TRI, p, f)
        rep.check(f"(7) P{p} interpolate(nodal_values(f)) reproduces degree-{p} f",
                  relmax(clq.interpolate(TRI, vals, pts), f(pts)), 1e-13)
    # tripwire: P1 does not reproduce a generic quadratic (the check is not vacuous)
    coef = rng.standard_normal((6, 3))
    f2 = lambda P: basis(P, 2) @ coef
    d = relmax(clq.interpolate(TRI, clq.nodal_values(TRI, 1, f2), pts), f2(pts))
    rep.check_bool("(7) tripwire: P1 interpolation of a quadratic differs by > 1e-2",
                   d > 1e-2, f"(rel diff {d:.2e})")

    rep.finish()


if __name__ == "__main__":
    main()
