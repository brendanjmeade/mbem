"""Public API surface of ``clq`` (:mod:`clq.api`, :mod:`clq.shape`, :mod:`clq.frame`).

  * order inference: slips of shape (3,), (1,3), (3,3), (6,3), (10,3) give
    orders 0, 0, 1, 2, 3 (``order_from_count``) and ``displacement`` agrees with
    the explicit ``influence(order=p)`` contraction (1e-13);
  * argument validation raises ValueError: explicit ``order=`` disagreeing
    with K, slip (4,3) / (3,2), eps < 0, nu = 1/2, mu <= 0, a degenerate
    triangle, eps = 0 with an on-plane observer, far_field='bogus';
  * eps = 0 off the plane works and equals the eps -> 0 limit (eps = 1e-9, 1e-6);
  * a single (3,) observer returns squeezed (3,) / (3,3) outputs equal to the
    first row of the batched call;
  * ``stress`` (elastic, default) == total - ``eigenstress`` (1e-14), and
    ``eigenstress`` matches the E-weight formula written out by hand;
  * ``traction(sigma, tri)`` == sigma @ nhat;
  * constant slip written as (1,3), (3,3) identical rows, (6,3) identical rows
    gives the same displacement / stress (1e-13, partition of unity);
  * BEM block ``U.transpose(0,2,1,3).reshape(3N,3K) @ slip.ravel()`` equals
    ``displacement`` (1e-14), same for the stress block from H;
  * ``nodes(tri, 2)`` = vertices, then midpoints of edges 12, 23, 31;
  * FORCE (Kelvin single-layer) surface: ``force_displacement`` /
    ``force_stress`` infer the order from a (3,) / (1,3) / (3,3) / (6,3)
    nodal force density exactly as the slip entry points do, squeeze a single
    (3,) observer the same way, raise ValueError on the same bad shapes (with
    "force" -- not "slip" -- named in the message wherever the API spells the
    argument out), and reduce to the same BEM block layout,
    ``G.transpose(0,2,1,3).reshape(3N,3K) @ force.ravel()``.  ``influence``
    with ``want=("G",)`` leaves U, H, E and S as ``None``.  Finally
    ``force_stress`` must REJECT ``subtract_eigenstress`` with a TypeError: a
    mollified body force is a genuine body force, not an eigenstrain, so there
    is no eigenstress to subtract and the keyword does not exist (the slip
    ``stress`` does take it -- both halves are gated so the asymmetry cannot
    be "fixed" away silently).
"""
from __future__ import annotations

import numpy as np

from _common import TRI, MU, Report, relmax
import clq
import sys


def raises(fn, exc=ValueError):
    """True if ``fn()`` raises ``exc`` (and nothing else)."""
    try:
        fn()
    except exc:
        return True
    except Exception as e:                     # noqa: BLE001 - report the wrong type
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


def raises_with(fn, word, exc=ValueError):
    """True if ``fn()`` raises ``exc`` whose message contains ``word``."""
    try:
        fn()
    except exc as e:
        ok = word in str(e)
        if not ok:
            print(f"      ({type(e).__name__} message lacks {word!r}: {e})")
        return ok
    except Exception as e:                     # noqa: BLE001
        print(f"      (raised {type(e).__name__}: {e})")
        return False
    return False


def main():
    rep = Report("public API surface (clq.api / clq.shape / clq.frame)")
    rng = np.random.default_rng(7)

    # --- geometry / material for the equilateral checks ---------------------
    tri = clq.equilateral(1.0)
    nu, eps = 0.25, 0.05
    # off-plane observers: over the face, near a vertex, outside the footprint,
    # moderately far, and one beyond D_STAR * L (hybrid -> quadrature branch)
    obs = np.array([[0.10, 0.05, 0.30],
                    [-0.20, 0.10, -0.40],
                    [0.02, 0.55, 0.02],
                    [0.60, -0.30, 0.15],
                    [1.50, 1.00, 0.80],
                    [12.0, -3.00, 5.00]])
    N = obs.shape[0]

    # ------------------------------------------------------------------ (1)
    # order inference from the slip shape + parity with influence(order=p)
    worst_infer = 0.0
    for shape, p_expect in (((3,), 0), ((1, 3), 0), ((3, 3), 1), ((6, 3), 2), ((10, 3), 3)):
        slip = rng.standard_normal(shape)
        K = 1 if len(shape) == 1 else shape[0]
        p = clq.order_from_count(K)
        rep.check_bool(f"order_from_count({K}) == {p_expect}", p == p_expect, f"(got {p})")
        rep.check_bool(f"n_nodes({p_expect}) == {K}", clq.n_nodes(p_expect) == K)
        inf = clq.influence(obs, tri, MU, nu, eps, order=p_expect, want=("U",))
        rep.check_bool(f"influence(order={p_expect}) shapes / order",
                       inf.order == p_expect and inf.U.shape == (N, K, 3, 3)
                       and inf.nodes.shape == (K, 3) and inf.H is None and inf.E is None,
                       f"(U {inf.U.shape}, nodes {inf.nodes.shape})")
        u_ref = np.einsum("nkij,kj->ni", inf.U, np.atleast_2d(slip))
        u = clq.displacement(obs, tri, slip, MU, nu, eps)
        d = relmax(u, u_ref)
        worst_infer = max(worst_infer, d)
        rep.check(f"displacement(slip {shape}) vs influence(order={p_expect})", d, 1e-13)
        # explicit, agreeing order= is accepted
        u2 = clq.displacement(obs, tri, slip, MU, nu, eps, order=p_expect)
        rep.check(f"displacement(slip {shape}, order={p_expect}) identical", relmax(u2, u), 1e-15)

    # ------------------------------------------------------------------ (2)
    # argument validation
    slip3 = rng.standard_normal((3, 3))
    slip1 = rng.standard_normal(3)
    rep.check_bool("order=0 with a (3,3) slip raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, slip3, MU, nu, eps, order=0)))
    rep.check_bool("order=2 with a (3,3) slip raises ValueError (stress)",
                   raises(lambda: clq.stress(obs, tri, slip3, MU, nu, eps, order=2)))
    rep.check_bool("order=1 with a (3,) slip raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, slip1, MU, nu, eps, order=1)))
    rep.check_bool("slip of shape (4,3) raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, rng.standard_normal((4, 3)), MU, nu, eps)))
    rep.check_bool("slip of shape (3,2) raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, rng.standard_normal((3, 2)), MU, nu, eps)))
    rep.check_bool("slip of shape (2,) raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, np.ones(2), MU, nu, eps)))
    rep.check_bool("order_from_count(4) raises ValueError", raises(lambda: clq.order_from_count(4)))
    rep.check_bool("eps < 0 raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, slip1, MU, nu, -0.05)))
    rep.check_bool("nu = 0.5 raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, slip1, MU, 0.5, eps)))
    rep.check_bool("mu = 0 raises ValueError",
                   raises(lambda: clq.displacement(obs, tri, slip1, 0.0, nu, eps)))
    tri_degen = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])     # collinear
    tri_sliver = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1e-16, 0.0]])  # area ~ 1e-16
    rep.check_bool("collinear triangle raises ValueError (displacement)",
                   raises(lambda: clq.displacement(obs, tri_degen, slip1, MU, nu, eps)))
    rep.check_bool("collinear triangle raises ValueError (unit_normal)",
                   raises(lambda: clq.unit_normal(tri_degen)))
    rep.check_bool("sliver triangle (area/L^2 ~ 1e-17) raises ValueError",
                   raises(lambda: clq.influence(obs, tri_sliver, MU, nu, eps)))
    rep.check_bool("tri of shape (4,3) raises ValueError",
                   raises(lambda: clq.displacement(obs, np.zeros((4, 3)), slip1, MU, nu, eps)))
    # eps = 0: on-plane observer must raise; off-plane must work and be the eps -> 0 limit
    on_plane = np.array([[0.05, 0.02, 0.0]])                    # inside the face, z = 0
    rep.check_bool("eps = 0 with an on-plane observer raises ValueError",
                   raises(lambda: clq.displacement(on_plane, tri, slip1, MU, nu, 0.0)))
    rep.check_bool("eps = 0 with an on-plane observer raises ValueError (stress)",
                   raises(lambda: clq.stress(on_plane, tri, slip3, MU, nu, 0.0)))
    mixed = np.vstack([obs, on_plane])
    rep.check_bool("eps = 0 with one on-plane observer in a batch raises",
                   raises(lambda: clq.displacement(mixed, tri, slip1, MU, nu, 0.0)))
    slip6 = rng.standard_normal((6, 3))
    worst_eps0 = 0.0
    for name, fn in (("displacement", clq.displacement), ("stress", clq.stress)):
        out0 = fn(obs, tri, slip6, MU, nu, 0.0)
        out9 = fn(obs, tri, slip6, MU, nu, 1e-9)
        d = relmax(out0, out9)
        worst_eps0 = max(worst_eps0, d)
        rep.check(f"eps = 0 off-plane {name} == eps = 1e-9 (quadratic slip)", d, 1e-6)
        rep.check_bool(f"eps = 0 off-plane {name} is finite", np.all(np.isfinite(out0)))
    e0 = clq.eigenstress(obs, tri, slip6, MU, nu, 0.0)
    rep.check("eps = 0 eigenstress is identically zero", np.max(np.abs(e0)), 1e-300)

    # ------------------------------------------------------------------ (3)
    # single (3,) observer -> squeezed outputs
    o1 = obs[3]
    u1 = clq.displacement(o1, tri, slip6, MU, nu, eps)
    s1 = clq.stress(o1, tri, slip6, MU, nu, eps)
    e1 = clq.eigenstress(o1, tri, slip6, MU, nu, eps)
    uN = clq.displacement(obs, tri, slip6, MU, nu, eps)
    sN = clq.stress(obs, tri, slip6, MU, nu, eps)
    eN = clq.eigenstress(obs, tri, slip6, MU, nu, eps)
    rep.check_bool("single observer: displacement shape (3,)", u1.shape == (3,), f"(got {u1.shape})")
    rep.check_bool("single observer: stress shape (3,3)", s1.shape == (3, 3), f"(got {s1.shape})")
    rep.check_bool("single observer: eigenstress shape (3,3)", e1.shape == (3, 3), f"(got {e1.shape})")
    rep.check_bool("batched observer: shapes (N,3) / (N,3,3) / (N,3,3)",
                   uN.shape == (N, 3) and sN.shape == (N, 3, 3) and eN.shape == (N, 3, 3))
    rep.check("single observer displacement == batched row", relmax(u1, uN[3]), 1e-14)
    rep.check("single observer stress == batched row", relmax(s1, sN[3]), 1e-14)
    rep.check("single observer eigenstress == batched row", relmax(e1, eN[3]), 1e-14)
    # a (1,3) observer stays 2-D
    u11 = clq.displacement(obs[3:4], tri, slip6, MU, nu, eps)
    rep.check_bool("(1,3) observer: displacement shape (1,3)", u11.shape == (1, 3))

    # ------------------------------------------------------------------ (4)
    # elastic stress = total - eigenstress; eigenstress = E-weight formula.
    # Use the tilted TRI at nu = 0.3 (lam != mu) with an observer set that
    # includes an on-plane point (eigenstress nonzero there) and a far point.
    nu_t, eps_t = 0.3, 0.12
    v1, v2, v3 = TRI
    nhat_t = clq.unit_normal(TRI)
    obs_t = np.array([[0.60, -0.10, 0.60],
                      [2.00, 1.50, 3.00],
                      [-0.40, 0.00, -0.80],
                      v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),          # on-plane, inside
                      v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1) + 0.05 * nhat_t,
                      TRI.mean(0) + 30.0 * nhat_t])                     # far: quadrature branch
    worst_split = 0.0
    for shape in ((1, 3), (3, 3), (6, 3)):
        sl = rng.standard_normal(shape)
        p = clq.order_from_count(shape[0])
        sig_el = clq.stress(obs_t, TRI, sl, MU, nu_t, eps_t)
        sig_tot = clq.stress(obs_t, TRI, sl, MU, nu_t, eps_t, subtract_eigenstress=False)
        sig_eig = clq.eigenstress(obs_t, TRI, sl, MU, nu_t, eps_t)
        d = relmax(sig_el, sig_tot - sig_eig)
        worst_split = max(worst_split, d)
        rep.check(f"order {p}: stress (elastic) == total - eigenstress", d, 1e-14)
        rep.check_bool(f"order {p}: eigenstress nonzero on the plane, ~0 far away",
                       np.max(np.abs(sig_eig[3])) > 1e-3 * np.max(np.abs(sig_tot[3]))
                       and np.max(np.abs(sig_eig[5])) < 1e-12 * np.max(np.abs(sig_eig[3])))
        # E-weight formula written out by hand
        inf = clq.influence(obs_t, TRI, MU, nu_t, eps_t, order=p, want=("E",))
        lam = 2.0 * MU * nu_t / (1.0 - 2.0 * nu_t)
        eig_hand = np.zeros((obs_t.shape[0], 3, 3))
        for k in range(sl.shape[0]):
            sk = sl[k]
            sig_k = lam * (sk @ nhat_t) * np.eye(3) + MU * (np.outer(sk, nhat_t) + np.outer(nhat_t, sk))
            eig_hand += inf.E[:, k][:, None, None] * sig_k[None]
        rep.check(f"order {p}: eigenstress == sum_k E_k [lam (s.n) I + mu (s n^T + n s^T)]",
                  relmax(sig_eig, eig_hand), 1e-14)
        # symmetry of the stress tensors
        rep.check(f"order {p}: stress tensors symmetric",
                  max(relmax(sig_el, np.swapaxes(sig_el, 1, 2)),
                      relmax(sig_tot, np.swapaxes(sig_tot, 1, 2))), 1e-14)
        # total stress: H contracted directly
        infH = clq.influence(obs_t, TRI, MU, nu_t, eps_t, order=p, want=("H",))
        rep.check(f"order {p}: total stress == einsum(H, slip)",
                  relmax(sig_tot, np.einsum("nkmlj,kj->nml", infH.H, sl)), 1e-14)
        # an on-plane point with 3 (K,3) slips: elastic != total (eigenstress is subtracted)
        rep.check_bool(f"order {p}: elastic != total on the plane",
                       relmax(sig_el[3], sig_tot[3]) > 1e-6)

    # ------------------------------------------------------------------ (5)
    # traction
    sig = clq.stress(obs_t, TRI, rng.standard_normal((6, 3)), MU, nu_t, eps_t)
    t = clq.traction(sig, TRI)
    rep.check_bool("traction shape (N,3)", t.shape == (obs_t.shape[0], 3))
    rep.check("traction == sigma @ nhat (batched)", relmax(t, sig @ nhat_t), 1e-15)
    rep.check("traction == sigma @ nhat (single (3,3))", relmax(clq.traction(sig[0], TRI), sig[0] @ nhat_t), 1e-15)
    rep.check("traction == einsum(sigma_ij n_j)", relmax(t, np.einsum("nij,j->ni", sig, nhat_t)), 1e-15)
    # nhat convention: (v2-v1) x (v3-v1) normalised; equilateral -> +z
    nn = np.cross(v2 - v1, v3 - v1)
    rep.check("unit_normal == (v2-v1)x(v3-v1)/|.|", relmax(nhat_t, nn / np.linalg.norm(nn)), 1e-15)
    rep.check("equilateral(1) normal == +z", relmax(clq.unit_normal(tri), [0, 0, 1.0]), 1e-15)

    # ------------------------------------------------------------------ (6)
    # constant slip written at three orders: (1,3), (3,3) identical rows,
    # (6,3) identical rows -> identical displacement / stress (partition of unity)
    s0 = rng.standard_normal(3)
    worst_pou = 0.0
    for geom_name, T, nu_g, eps_g, O in (("equilateral", tri, nu, eps, obs),
                                         ("tilted TRI", TRI, nu_t, eps_t, obs_t)):
        u_ref = clq.displacement(O, T, s0[None, :], MU, nu_g, eps_g)
        s_ref = clq.stress(O, T, s0[None, :], MU, nu_g, eps_g)
        e_ref = clq.eigenstress(O, T, s0[None, :], MU, nu_g, eps_g)
        u_vec = clq.displacement(O, T, s0, MU, nu_g, eps_g)
        rep.check(f"{geom_name}: (3,) slip == (1,3) slip displacement", relmax(u_vec, u_ref), 1e-15)
        for K in (3, 6):
            sK = np.tile(s0, (K, 1))
            uK = clq.displacement(O, T, sK, MU, nu_g, eps_g)
            sKs = clq.stress(O, T, sK, MU, nu_g, eps_g)
            eK = clq.eigenstress(O, T, sK, MU, nu_g, eps_g)
            du, ds, de = relmax(uK, u_ref), relmax(sKs, s_ref), relmax(eK, e_ref)
            worst_pou = max(worst_pou, du, ds, de)
            rep.check(f"{geom_name}: constant slip via ({K},3) == (1,3): displacement", du, 1e-13)
            rep.check(f"{geom_name}: constant slip via ({K},3) == (1,3): stress", ds, 1e-13)
            rep.check(f"{geom_name}: constant slip via ({K},3) == (1,3): eigenstress", de, 1e-13)
    # a genuinely varying slip must NOT collapse to the constant answer
    s_var = np.tile(s0, (3, 1)) + 0.3 * rng.standard_normal((3, 3))
    rep.check_bool("linear slip with varying rows differs from the constant result",
                   relmax(clq.displacement(obs, tri, s_var, MU, nu, eps),
                          clq.displacement(obs, tri, s0, MU, nu, eps)) > 1e-3)
    # and the three-shape calls run with random nodal values (stress + displacement)
    ok_run = True
    for K in (1, 3, 6):
        sl = rng.standard_normal((K, 3))
        uu = clq.displacement(obs, tri, sl, MU, nu, eps)
        ss = clq.stress(obs, tri, sl, MU, nu, eps)
        ok_run &= uu.shape == (N, 3) and ss.shape == (N, 3, 3) and np.all(np.isfinite(uu)) and np.all(np.isfinite(ss))
    rep.check_bool("(1,3)/(3,3)/(6,3) random slips: displacement + stress run, finite, right shapes", ok_run)

    # ------------------------------------------------------------------ (7)
    # BEM block reshape
    worst_block = 0.0
    for p in (0, 1, 2):
        K = clq.n_nodes(p)
        sl = rng.standard_normal((K, 3))
        inf = clq.influence(obs_t, TRI, MU, nu_t, eps_t, order=p)
        Nt = obs_t.shape[0]
        Ublock = inf.U.transpose(0, 2, 1, 3).reshape(3 * Nt, 3 * K)
        u_blk = (Ublock @ sl.ravel()).reshape(Nt, 3)
        u_api = clq.displacement(obs_t, TRI, sl, MU, nu_t, eps_t)
        d = relmax(u_blk, u_api)
        worst_block = max(worst_block, d)
        rep.check(f"order {p}: U block (3N,3K) @ slip.ravel() == displacement", d, 1e-14)
        Hblock = inf.H.transpose(0, 2, 3, 1, 4).reshape(9 * Nt, 3 * K)
        s_blk = (Hblock @ sl.ravel()).reshape(Nt, 3, 3)
        s_api = clq.stress(obs_t, TRI, sl, MU, nu_t, eps_t, subtract_eigenstress=False)
        d = relmax(s_blk, s_api)
        worst_block = max(worst_block, d)
        rep.check(f"order {p}: H block (9N,3K) @ slip.ravel() == total stress", d, 1e-14)
        # column K-index sanity: the block column for node k, component j is U[:, k, :, j]
        k, j = K - 1, 2
        col = Ublock[:, 3 * k + j].reshape(Nt, 3)
        rep.check(f"order {p}: block column (k={k}, j={j}) == U[:, k, :, j]", relmax(col, inf.U[:, k, :, j]), 1e-15)

    # ------------------------------------------------------------------ (8)
    # node ordering
    n2 = clq.nodes(TRI, 2)
    n2_expect = np.array([v1, v2, v3, 0.5 * (v1 + v2), 0.5 * (v2 + v3), 0.5 * (v3 + v1)])
    rep.check_bool("nodes(tri, 2) shape (6,3)", n2.shape == (6, 3))
    rep.check("nodes(tri, 2) == [v1, v2, v3, m12, m23, m31]", relmax(n2, n2_expect), 1e-15)
    rep.check("nodes(tri, 1) == [v1, v2, v3]", relmax(clq.nodes(TRI, 1), TRI), 1e-15)
    rep.check("nodes(tri, 0) == [centroid]", relmax(clq.nodes(TRI, 0), TRI.mean(0, keepdims=True)), 1e-15)
    rep.check_bool("Influence.nodes == nodes(tri, p) for p = 0, 1, 2",
                   all(np.array_equal(clq.influence(obs_t[:1], TRI, MU, nu_t, eps_t, order=p, want=("E",)).nodes,
                                      clq.nodes(TRI, p)) for p in (0, 1, 2)))
    # shape functions are Lagrange (Kronecker delta at the nodes) in the same order
    for p in (1, 2):
        Nk = clq.shape_functions(TRI, p, clq.nodes(TRI, p))
        rep.check(f"shape_functions(order {p}) at nodes == identity", relmax(Nk, np.eye(clq.n_nodes(p))), 1e-13)

    # ------------------------------------------------------------------ (9)
    # far_field validation
    rep.check_bool("far_field='bogus' raises ValueError (influence)",
                   raises(lambda: clq.influence(obs, tri, MU, nu, eps, far_field="bogus")))
    rep.check_bool("far_field='bogus' raises ValueError (displacement)",
                   raises(lambda: clq.displacement(obs, tri, slip1, MU, nu, eps, far_field="bogus")))
    rep.check_bool("far_field='bogus' raises ValueError (stress)",
                   raises(lambda: clq.stress(obs, tri, slip1, MU, nu, eps, far_field="bogus")))
    rep.check_bool("far_field in {'hybrid','analytic','quadrature'} all run",
                   all(np.all(np.isfinite(clq.displacement(obs, tri, slip6, MU, nu, eps, far_field=ff)))
                       for ff in ("hybrid", "analytic", "quadrature")))

    # ------------------------------------------------------------------ (10)
    # FORCE (Kelvin single-layer) entry points.  Same order inference, same
    # squeeze, same BEM block layout, same validation -- and one deliberate
    # asymmetry: no subtract_eigenstress.
    worst_force = 0.0
    for shape, p_expect in (((3,), 0), ((1, 3), 0), ((3, 3), 1), ((6, 3), 2)):
        force = rng.standard_normal(shape)
        K = 1 if len(shape) == 1 else shape[0]
        inf = clq.influence(obs, tri, MU, nu, eps, order=p_expect, want=("G",))
        rep.check_bool(f"influence(order={p_expect}, want=('G',)): G shape, U/H/E/S None",
                       inf.G.shape == (N, K, 3, 3) and inf.U is None and inf.H is None
                       and inf.E is None and inf.S is None,
                       f"(G {None if inf.G is None else inf.G.shape})")
        u_ref = np.einsum("nkij,kj->ni", inf.G, np.atleast_2d(force))
        u = clq.force_displacement(obs, tri, force, MU, nu, eps)
        d = relmax(u, u_ref)
        worst_force = max(worst_force, d)
        rep.check(f"force_displacement(force {shape}) vs influence(order={p_expect})", d, 1e-13)
        u2 = clq.force_displacement(obs, tri, force, MU, nu, eps, order=p_expect)
        rep.check(f"force_displacement(force {shape}, order={p_expect}) identical",
                  relmax(u2, u), 1e-15)
        infS = clq.influence(obs, tri, MU, nu, eps, order=p_expect, want=("S",))
        rep.check_bool(f"influence(order={p_expect}, want=('S',)): S shape, U/H/E/G None",
                       infS.S.shape == (N, K, 3, 3, 3) and infS.U is None
                       and infS.H is None and infS.E is None and infS.G is None)
        rep.check(f"force_stress(force {shape}) vs einsum(S, force)",
                  relmax(clq.force_stress(obs, tri, force, MU, nu, eps),
                         np.einsum("nkijc,kc->nij", infS.S, np.atleast_2d(force))), 1e-13)
    # validation: "force", not "slip", wherever the API names the argument
    for fn, name in ((clq.force_displacement, "force_displacement"),
                     (clq.force_stress, "force_stress")):
        rep.check_bool(f"{name}: force of shape (3,2) raises ValueError naming 'force'",
                       raises_with(lambda fn=fn: fn(obs, tri, rng.standard_normal((3, 2)),
                                                    MU, nu, eps), "force"))
        rep.check_bool(f"{name}: force of shape (2,) raises ValueError naming 'force'",
                       raises_with(lambda fn=fn: fn(obs, tri, np.ones(2), MU, nu, eps), "force"))
        # these two go through order_from_count / the order check, which are
        # argument-name agnostic: gate the exception, not the wording
        rep.check_bool(f"{name}: force of shape (4,3) raises ValueError",
                       raises(lambda fn=fn: fn(obs, tri, rng.standard_normal((4, 3)),
                                               MU, nu, eps)))
        rep.check_bool(f"{name}: order=0 with a (3,3) force raises ValueError",
                       raises(lambda fn=fn: fn(obs, tri, rng.standard_normal((3, 3)),
                                               MU, nu, eps, order=0)))
        rep.check_bool(f"{name}: far_field='bogus' raises ValueError",
                       raises(lambda fn=fn: fn(obs, tri, slip1, MU, nu, eps, far_field="bogus")))
        rep.check_bool(f"{name}: eps < 0 raises ValueError",
                       raises(lambda fn=fn: fn(obs, tri, slip1, MU, nu, -0.05)))
    # the one deliberate asymmetry with the slip path
    force6 = rng.standard_normal((6, 3))
    for flag in (True, False):
        rep.check_bool(f"force_stress(subtract_eigenstress={flag}) raises TypeError",
                       raises(lambda flag=flag: clq.force_stress(obs, tri, force6, MU, nu, eps,
                                                                 subtract_eigenstress=flag),
                              exc=TypeError))
    rep.check_bool("stress(subtract_eigenstress=...) DOES exist (asymmetry is deliberate)",
                   np.all(np.isfinite(clq.stress(obs, tri, slip6, MU, nu, eps,
                                                 subtract_eigenstress=False))))
    # squeeze: a single (3,) observer
    uf1 = clq.force_displacement(obs[3], tri, force6, MU, nu, eps)
    sf1 = clq.force_stress(obs[3], tri, force6, MU, nu, eps)
    ufN = clq.force_displacement(obs, tri, force6, MU, nu, eps)
    sfN = clq.force_stress(obs, tri, force6, MU, nu, eps)
    rep.check_bool("single observer: force_displacement shape (3,), force_stress (3,3)",
                   uf1.shape == (3,) and sf1.shape == (3, 3), f"(got {uf1.shape}, {sf1.shape})")
    rep.check_bool("batched observer: force shapes (N,3) / (N,3,3)",
                   ufN.shape == (N, 3) and sfN.shape == (N, 3, 3))
    rep.check("single observer force_displacement == batched row", relmax(uf1, ufN[3]), 1e-14)
    rep.check("single observer force_stress == batched row", relmax(sf1, sfN[3]), 1e-14)
    rep.check_bool("(1,3) observer: force_displacement shape (1,3)",
                   clq.force_displacement(obs[3:4], tri, force6, MU, nu, eps).shape == (1, 3))
    # constant force density written at three orders (partition of unity)
    f0 = rng.standard_normal(3)
    uf_ref = clq.force_displacement(obs, tri, f0[None, :], MU, nu, eps)
    sf_ref = clq.force_stress(obs, tri, f0[None, :], MU, nu, eps)
    rep.check("(3,) force == (1,3) force: displacement",
              relmax(clq.force_displacement(obs, tri, f0, MU, nu, eps), uf_ref), 1e-15)
    for K in (3, 6):
        fK = np.tile(f0, (K, 1))
        rep.check(f"constant force via ({K},3) == (1,3): displacement",
                  relmax(clq.force_displacement(obs, tri, fK, MU, nu, eps), uf_ref), 1e-13)
        rep.check(f"constant force via ({K},3) == (1,3): stress",
                  relmax(clq.force_stress(obs, tri, fK, MU, nu, eps), sf_ref), 1e-13)
    # BEM block reshape, force side
    for p in (0, 1, 2):
        K = clq.n_nodes(p)
        f = rng.standard_normal((K, 3))
        inf = clq.influence(obs_t, TRI, MU, nu_t, eps_t, order=p, want=("G", "S"))
        Nt = obs_t.shape[0]
        Gblock = inf.G.transpose(0, 2, 1, 3).reshape(3 * Nt, 3 * K)
        d = relmax((Gblock @ f.ravel()).reshape(Nt, 3),
                   clq.force_displacement(obs_t, TRI, f, MU, nu_t, eps_t))
        worst_force = max(worst_force, d)
        rep.check(f"order {p}: G block (3N,3K) @ force.ravel() == force_displacement", d, 1e-14)
        Sblock = inf.S.transpose(0, 2, 3, 1, 4).reshape(9 * Nt, 3 * K)
        d = relmax((Sblock @ f.ravel()).reshape(Nt, 3, 3),
                   clq.force_stress(obs_t, TRI, f, MU, nu_t, eps_t))
        worst_force = max(worst_force, d)
        rep.check(f"order {p}: S block (9N,3K) @ force.ravel() == force_stress", d, 1e-14)
        k, j = K - 1, 2
        rep.check(f"order {p}: G block column (k={k}, j={j}) == G[:, k, :, j]",
                  relmax(Gblock[:, 3 * k + j].reshape(Nt, 3), inf.G[:, k, :, j]), 1e-15)

    print(f"  worst: order-inference parity {worst_infer:.2e}, eps=0 limit {worst_eps0:.2e}, "
          f"elastic/total split {worst_split:.2e}, partition of unity {worst_pou:.2e}, "
          f"BEM block {worst_block:.2e}, force surface {worst_force:.2e}")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
