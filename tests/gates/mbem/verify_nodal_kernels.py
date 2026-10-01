"""verify_nodal_kernels.py -- nodal (P0/P1/P2) triangle kernels vs the clq oracle.

``mbem/kernels/tri_nodal.py`` is the numba port of clq's closed-form moment
machinery (cancellation-free edge primitives, degree-generic recursion,
per-node weighted tables, Gauss far field).  This gate pins it to clq, loaded
BY FILE PATH from ``../clq`` (a missing clq is a FAIL, never a skip).

  [0]   the edge primitives int_{ua}^{ub} u^k / R^m du against
        clq.primitives.edge_table on a (ua, ub, rho) grid that enters all three
        regimes (closed form, small-|u| series, large-|u| series: spans crossing
        zero, same-sign, same-sign with nearly equal endpoints, both signs) for
        every (m, k) of the P2 spec, entrywise relative 1e-14;
  [i]   all five kernels at p = 0, 1, 2 against clq.influence at nu = 0.25 and
        0.30 on three triangles (equilateral, tilted scalene, sliver h/L ~ 0.1),
        eps/h in {0.05, 0.3}, observers near (|z| = 0.3 eps and 1 eps inside the
        footprint, just off the edges), broadside at 2 L (the small-|u| edge
        regime at kernel level), mid (2-9.9 L in plane and oblique, closed
        form) and far (10.1, 15, 60 L: both Gauss rules).  mbem's T basis
        recombined with t_coeffs is clq's U, its U basis with u_coeffs is clq's
        G, the stress drivers are H and S, the eigenstress driver is E.
        Block-wise max|diff| / max|clq| < 1e-12 over the near and far
        observers (E over all observers), and the near and far groups each
        meet 1e-12 on their own scale.  Away from the element the
        divergence-theorem closed form amplifies ulp-level differences between
        two implementations (more with the order and with L / height; the
        reason for the Gauss crossover at defaults.NODAL_D_STAR), so the
        broadside group is held on its own scale to 1e-11 on the well-shaped
        triangles and 1e-9 on the sliver -- the small-|u| regime itself is
        pinned by [0] -- and the mid band is gated per observer against
        clq.quadrature (40 x 40 Gauss) for U, G, H, S at 1e-8 on the
        well-shaped triangles and 1e-6 on the sliver, with E (~(eps/R)^4 of its
        near value there) block-wise only;
  [ii]  the general machinery at order 0 against tri_kernels' P0 pair code on a
        random 40-triangle mesh with per-element eps (basis stacks, direct
        matrices, contraction, stress and eigenstress drivers), 1e-12; the
        public drivers at order 0 are bitwise tri_kernels';
  [iii] parallel == serial basis drivers, bitwise;
  [iv]  *_disp_contract == *_matrix_direct @ density, 1e-13;
  [v]   lam/mu tripwire at nu = 0.30: the swapped pairing (lam and mu exchanged
        on the first two traction-operator terms) differs from the port by
        > 1e-3 and coincides at nu = 0.25 -- the gate can see the pairing;
  [vi]  partition of unity: a uniform nodal density at P1 and P2 reproduces P0
        through every public driver, 1e-13.

Run from the repo root:  python verify/verify_nodal_kernels.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

from mbem.kernels import basis as kb                                # noqa: E402
from mbem.kernels import tri_kernels as tk                          # noqa: E402
from mbem.kernels import tri_nodal as tn                            # noqa: E402
from mbem import defaults                                           # noqa: E402

MU = 1.0
NUS = (0.25, 0.30)
ORDERS = (0, 1, 2)
EPS_OVER_H = (0.05, 0.3)
TOL_PRIM = 1e-14         # edge primitives vs clq, entrywise
TOL_BLOCK = 1e-12        # near + far observers, U G H S; all observers, E
TOL_GROUP = 1e-12        # near and far groups on their own scale
TOL_BROAD = {"equilateral": 1e-11, "scalene": 1e-11, "sliver": 1e-9}    # 2 L broadside, own scale
TOL_MID_QUAD = {"equilateral": 1e-8, "scalene": 1e-8, "sliver": 1e-6}   # 2-9.9 L vs Gauss
N_GAUSS_REF = 40
GROUPS = ("near", "broad", "mid", "far")
PINNED = ("near", "far")

CHECKS = []


def check(name, val, tol):
    ok = bool(val < tol)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:64s} {val:9.2e} (tol {tol:.0e})")
    return ok


def check_bool(name, ok, extra=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:64s} {extra}")
    return ok


def relmax(a, b):
    ref = float(np.max(np.abs(b)))
    diff = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    return diff / ref if ref > 0.0 else diff


def _load_clq():
    """clq is an installed package, not a sibling directory on sys.path.

    Still None rather than a raised ImportError, because this gate counts a
    missing oracle as FAIL and says so -- never a silent skip.
    """
    try:
        import clq
    except ImportError:
        return None
    return clq


def _lame(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def _rotation(seed):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _c(a):
    return np.ascontiguousarray(np.asarray(a, float))


# ---------------------------------------------------------------------------
# mbem nodal tensors of ONE triangle in clq's layout, through the general
# (order-generic) machinery at every order, so p = 0 exercises it too
# ---------------------------------------------------------------------------

def nodal_tensors(obs, tri, nrm, eps, mu, nu, p):
    K = tn.n_nodes(p)
    N = obs.shape[0]
    tv, nr, ea = _c(tri[None]), _c(nrm[None]), np.full(1, float(eps))
    lam = _lame(mu, nu)
    Tb = tn._t_basis_nodal(obs, tv, nr, ea, tn._spec(p, tn.KT))
    Ub = tn._u_basis_nodal(obs, tv, ea, tn._spec(p, tn.KU))
    U = np.tensordot(kb.t_coeffs(mu, lam), Tb, axes=1).reshape(N, 3, K, 3).transpose(0, 2, 1, 3)
    G = np.tensordot(kb.u_coeffs(mu, lam), Ub, axes=1).reshape(N, 3, K, 3).transpose(0, 2, 1, 3)
    H = np.zeros((N, K, 3, 3, 3))
    S = np.zeros((N, K, 3, 3, 3))
    Estar = np.zeros((N, K, 3, 3, 3))
    for k in range(K):
        for j in range(3):
            d = np.zeros((K, 3))
            d[k, j] = 1.0
            H[:, k, :, :, j] = tn._stress_contract_nodal(obs, tv, nr, ea, d, mu, nu, lam, tn._spec(p, tn.KH), 0)
            S[:, k, :, :, j] = tn._stress_contract_nodal(obs, tv, nr, ea, d, mu, nu, lam, tn._spec(p, tn.KS), 1)
            Estar[:, k, :, :, j] = tn._stress_contract_nodal(obs, tv, nr, ea, d, mu, nu, lam, tn._spec(p, tn.KE), 2)
    return dict(U=U, G=G, H=H, S=S, Estar=Estar)


def clq_eigen_tensor(E, nrm, mu, nu):
    """sigma*[n,k,m,l,j] = E[n,k] (lam n_j d_ml + mu (d_mj n_l + n_m d_lj))."""
    lam = _lame(mu, nu)
    eye = np.eye(3)
    sig0 = (lam * np.einsum("ml,j->mlj", eye, nrm)
            + mu * (np.einsum("mj,l->mlj", eye, nrm) + np.einsum("m,lj->mlj", nrm, eye)))
    return E[:, :, None, None, None] * sig0[None, None]


# ---------------------------------------------------------------------------
# [i] fixtures: three triangles, observers by range
# ---------------------------------------------------------------------------

def triangles(clq):
    Q = _rotation(3)
    eq = clq.frame.equilateral(1.0) @ Q.T + np.array([0.3, -0.2, 0.5])
    scalene = np.array([[0.37, -0.81, 0.44], [1.92, 0.11, -0.63], [-0.25, 1.57, 1.22]])
    sliver = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.1, 0.2, 0.0]]) @ _rotation(5).T \
        + np.array([-0.4, 0.7, 0.1])
    return (("equilateral", eq), ("scalene", scalene), ("sliver", sliver))


def observers(clq, tri, eps):
    fr = clq.frame.local_frame(tri)
    L, c, n, e1, e2 = fr.L, fr.centroid, fr.nhat, fr.e1, fr.e2
    obl = (e1 + 0.6 * e2 + 0.8 * n)
    obl /= np.linalg.norm(obl)
    v = fr.v
    mids = [0.5 * (v[i] + v[(i + 1) % 3]) for i in range(3)]
    near, mid, far = [], [], []
    for zf in (0.3, 1.0):
        for sgn in (1.0, -1.0):
            near.append(c + sgn * zf * eps * n)
            near.append(v[0] + 0.1 * (v[1] - v[0]) + 0.1 * (v[2] - v[0]) + sgn * zf * eps * n)
            for m in mids:
                out = m - c
                out /= np.linalg.norm(out)
                near.append(m - 0.3 * eps * out + sgn * zf * eps * n)      # inside, by the edge
                near.append(m + 0.5 * eps * out + sgn * zf * eps * n)      # just off the edge
    broad = [c + 2.0 * L * n, c - 2.0 * L * n]
    for r, d in ((2.0, e1), (5.0, e1), (5.0, obl), (8.0, obl), (8.0, e2), (9.9, obl)):
        mid.append(c + r * L * d)
    for r, d in ((10.1, obl), (15.0, e1), (15.0, n), (60.0, obl), (60.0, n)):
        far.append(c + r * L * d)
    return {"near": _c(near), "broad": _c(broad), "mid": _c(mid), "far": _c(far)}


def zero_edge_primitives(clq):
    print("\n[0] edge primitives vs clq.primitives.edge_table, all three regimes")
    from clq.primitives import edge_table
    spec = {5: 5, 3: 4, 1: 3, -1: 1}                     # the P2 spec
    binom = tn._spec(2, tn.KT)[3]
    SER = tn._scratch(1, binom.shape[1])[9]
    P = np.zeros((4, tn._MAX_EDGE_DEG + 1))
    ekmax = np.array([spec[m] for m in (-1, 1, 3, 5)], dtype=np.int64)
    small_u = defaults.NODAL_SMALL_U_OVER_RHO
    series_u = defaults.NODAL_SERIES_U_OVER_RHO
    shapes = ((-0.7, 1.0), (-1.0, 0.3), (0.0, 1.0), (-1.0, 0.0),        # crossing / touching zero
              (1.0 / 3.0, 1.0), (0.5, 1.0), (0.05, 1.0),               # same sign
              (0.999, 1.0), (-1.0, -0.999),                            # nearly equal endpoints
              (-1.0, -1.0 / 3.0), (-1.0, -0.5))                        # negative spans
    worst = {"closed": 0.0, "small": 0.0, "large": 0.0}
    count = {"closed": 0, "small": 0, "large": 0}
    for ratio in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 20.0):
        for rho in (1.0, 0.37):
            umax = ratio * rho
            for fa, fb in shapes:
                ua, ub, rho2 = fa * umax, fb * umax, rho * rho
                if max(abs(ua), abs(ub)) <= small_u * rho:
                    regime = "small"
                elif ua * ub > 0.0 and min(abs(ua), abs(ub)) >= series_u * rho:
                    regime = "large"
                else:
                    regime = "closed"
                tn._edge_primitives(ua, ub, rho2, ekmax, P, binom, small_u, series_u, SER)
                ref = edge_table(np.array([ua]), np.array([ub]), np.array([rho2]), spec)
                for m, kmax in spec.items():
                    for k in range(kmax + 1):
                        rv = ref[m][0, k]
                        worst[regime] = max(worst[regime], abs(P[(m + 1) // 2, k] - rv) / abs(rv))
                count[regime] += 1
    for regime in ("closed", "small", "large"):
        check(f"{regime} regime ({count[regime]} spans, all (m, k) of the P2 spec), entrywise",
              worst[regime], TOL_PRIM)


def i_clq_parity(clq):
    print("\n[i] all five kernels vs clq.influence, p = 0, 1, 2")
    from clq.quadrature import quadrature_influence
    want = ("U", "H", "E", "G", "S")
    keys = ("U", "G", "H", "S")
    worst = {}
    for name, tri in triangles(clq):
        nrm = clq.unit_normal(tri)
        h = np.mean([np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[1]),
                     np.linalg.norm(tri[0] - tri[2])])
        for eh in EPS_OVER_H:
            eps = eh * h
            groups = observers(clq, tri, eps)
            obs = np.concatenate([groups[g] for g in GROUPS])
            sl = {}
            i0 = 0
            for g in GROUPS:
                sl[g] = slice(i0, i0 + groups[g].shape[0])
                i0 += groups[g].shape[0]
            nf = np.concatenate([np.arange(sl[g].start, sl[g].stop) for g in PINNED])
            for nu in NUS:
                for p in ORDERS:
                    ref = clq.influence(obs, tri, MU, nu, eps, order=p, want=want)
                    quad = quadrature_influence(groups["mid"], tri, p, MU, nu, eps,
                                                n_gauss=N_GAUSS_REF, want=keys)
                    got = nodal_tensors(obs, tri, nrm, eps, MU, nu, p)
                    refd = dict(U=ref.U, G=ref.G, H=ref.H, S=ref.S,
                                Estar=clq_eigen_tensor(ref.E, nrm, MU, nu))
                    for key in keys + ("Estar",):
                        for g in PINNED:
                            worst[(g, key)] = max(worst.get((g, key), 0.0),
                                                  relmax(got[key][sl[g]], refd[key][sl[g]]))
                        worst[("broad", name)] = max(
                            worst.get(("broad", name), 0.0),
                            relmax(got[key][sl["broad"]], refd[key][sl["broad"]]))
                    mid_clq = max(relmax(got[k][sl["mid"]], refd[k][sl["mid"]]) for k in keys)
                    mid_quad = max(relmax(got[k][sl["mid"]][i], quad[k][i])
                                   for k in keys for i in range(groups["mid"].shape[0]))
                    clq_quad = max(relmax(refd[k][sl["mid"]][i], quad[k][i])
                                   for k in keys for i in range(groups["mid"].shape[0]))
                    worst["mid_clq"] = max(worst.get("mid_clq", 0.0), mid_clq)
                    worst["mid_quad"] = max(worst.get("mid_quad", 0.0), mid_quad)
                    worst["clq_quad"] = max(worst.get("clq_quad", 0.0), clq_quad)
                    label = f"{name} eps/h={eh:g} nu={nu:.2f} p={p}"
                    val = max(max(relmax(got[k][nf], refd[k][nf]) for k in keys),
                              relmax(got["Estar"], refd["Estar"]))
                    check(f"{label}: U,G,H,S near+far, E all, block-wise", val, TOL_BLOCK)
                    check(f"{label}: U,G,H,S mid 2-9.9 L per observer vs Gauss-{N_GAUSS_REF}",
                          mid_quad, TOL_MID_QUAD[name])
    for key in keys + ("Estar",):
        check(f"{key}: near group on its own scale (worst over all)", worst[("near", key)], TOL_GROUP)
        check(f"{key}: far group 10.1-60 L on its own scale (Gauss)", worst[("far", key)], TOL_GROUP)
    for name, _tri in triangles(clq):
        check(f"{name}: broadside 2 L group on its own scale, U G H S E", worst[("broad", name)],
              TOL_BROAD[name])
    print(f"      mid band, worst over all: port vs clq {worst['mid_clq']:.1e}, "
          f"port vs Gauss-{N_GAUSS_REF} {worst['mid_quad']:.1e}, "
          f"clq vs Gauss-{N_GAUSS_REF} {worst['clq_quad']:.1e} (closed-form conditioning)")


# ---------------------------------------------------------------------------
# [ii]-[iv], [vi]: a random 40-triangle mesh, per-element eps
# ---------------------------------------------------------------------------

def random_mesh(seed=11, n_tri=40):
    rng = np.random.default_rng(seed)
    tris = []
    while len(tris) < n_tri:
        c = rng.uniform(0.0, 1.0, 3)
        t = c + rng.normal(scale=0.2, size=(3, 3))
        area = 0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))
        L = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]),
                np.linalg.norm(t[0] - t[2]))
        if area > 0.05 * L * L:
            tris.append(t)
    tv = _c(np.array(tris))
    nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    nrm = _c(nrm / np.linalg.norm(nrm, axis=1, keepdims=True))
    h = np.linalg.norm(np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 1],
                                 tv[:, 0] - tv[:, 2]]), axis=2).mean(axis=0)
    eps = _c(rng.uniform(0.05, 0.3, n_tri) * h)
    obs = _c(rng.uniform(-0.2, 1.2, (50, 3)))
    cen = tv.mean(axis=1)
    L = np.linalg.norm(np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 1],
                                 tv[:, 0] - tv[:, 2]]), axis=2).max(axis=0)
    D = np.sqrt(((obs[:, None, :] - cen[None]) ** 2).sum(axis=2) + eps[None] ** 2)
    print(f"      {n_tri} random triangles, {obs.shape[0]} field points, "
          f"eps/h in [{(eps / h).min():.3f}, {(eps / h).max():.3f}], "
          f"max D/L = {(D / L[None]).max():.2f} (closed form below {defaults.NODAL_D_STAR})")
    return tv, nrm, eps, obs


def ii_order0_vs_tri_kernels(tv, nrm, eps, obs):
    print("\n[ii] general machinery at order 0 vs tri_kernels' P0 pair code")
    nu = 0.30
    lam = _lame(MU, nu)
    ct = kb.t_coeffs(MU, lam)
    cu = kb.u_coeffs(MU, lam)
    rng = np.random.default_rng(1)
    dens = rng.normal(size=(tv.shape[0], 3))
    dens[5] = 0.0                                          # zero-block skip
    pairs = [
        ("t_basis_matrices", tn._t_basis_nodal(obs, tv, nrm, eps, tn._spec(0, tn.KT)),
         tk.t_basis_matrices(obs, tv, nrm, eps)),
        ("u_basis_matrices", tn._u_basis_nodal(obs, tv, eps, tn._spec(0, tn.KU)),
         tk.u_basis_matrices(obs, tv, eps)),
        ("t_basis_matrices_serial", tn._t_basis_nodal_serial(obs, tv, nrm, eps, tn._spec(0, tn.KT)),
         tk.t_basis_matrices_serial(obs, tv, nrm, eps)),
        ("u_basis_matrices_serial", tn._u_basis_nodal_serial(obs, tv, eps, tn._spec(0, tn.KU)),
         tk.u_basis_matrices_serial(obs, tv, eps)),
        ("t_matrix_direct", tn._t_direct_nodal(obs, tv, nrm, eps, _c(ct), tn._spec(0, tn.KT)),
         tk.t_matrix_direct(obs, tv, nrm, eps, *ct)),
        ("u_matrix_direct", tn._u_direct_nodal(obs, tv, eps, _c(cu), tn._spec(0, tn.KU)),
         tk.u_matrix_direct(obs, tv, eps, *cu)),
        ("t_disp_contract", tn._t_contract_nodal(obs, tv, nrm, eps, dens, _c(ct), tn._spec(0, tn.KT)),
         tk.t_disp_contract(obs, tv, nrm, eps, dens, *ct)),
        ("u_disp_contract", tn._u_contract_nodal(obs, tv, eps, dens, _c(cu), tn._spec(0, tn.KU)),
         tk.u_disp_contract(obs, tv, eps, dens, *cu)),
        ("dd_stress_contract", tn._stress_contract_nodal(obs, tv, nrm, eps, dens, MU, nu, lam, tn._spec(0, tn.KH), 0),
         tk.dd_stress_contract(obs, tv, nrm, eps, dens, MU, lam)),
        ("kelvin_stress_contract", tn._stress_contract_nodal(obs, tv, nrm, eps, dens, MU, nu, lam, tn._spec(0, tn.KS), 1),
         tk.kelvin_stress_contract(obs, tv, eps, dens, MU, lam)),
        ("eigenstress_contract", tn._stress_contract_nodal(obs, tv, nrm, eps, dens, MU, nu, lam, tn._spec(0, tn.KE), 2),
         tk.eigenstress_contract(obs, tv, nrm, eps, dens, MU, lam)),
    ]
    for name, new, old in pairs:
        check(f"nodal order 0 vs tri_kernels: {name}", relmax(new, old), 1e-12)
    # the public drivers at order 0 ARE tri_kernels
    same = (np.array_equal(tn.t_basis_matrices(obs, tv, nrm, eps, 0), tk.t_basis_matrices(obs, tv, nrm, eps))
            and np.array_equal(tn.dd_stress_contract(obs, tv, nrm, eps, dens, MU, lam, 0),
                               tk.dd_stress_contract(obs, tv, nrm, eps, dens, MU, lam))
            and np.array_equal(tn.u_disp_contract(obs, tv, eps, dens, *cu, 0),
                               tk.u_disp_contract(obs, tv, eps, dens, *cu)))
    check_bool("public drivers at order 0 are bitwise tri_kernels'", same)


def iii_parallel_vs_serial(tv, nrm, eps, obs):
    print("\n[iii] parallel == serial basis drivers, bitwise")
    for p in (1, 2):
        a = tn.t_basis_matrices(obs, tv, nrm, eps, p)
        b = tn.t_basis_matrices_serial(obs, tv, nrm, eps, p)
        check_bool(f"p={p}: t_basis_matrices == t_basis_matrices_serial", np.array_equal(a, b),
                   f"shape {a.shape}")
        a = tn.u_basis_matrices(obs, tv, eps, p)
        b = tn.u_basis_matrices_serial(obs, tv, eps, p)
        check_bool(f"p={p}: u_basis_matrices == u_basis_matrices_serial", np.array_equal(a, b),
                   f"shape {a.shape}")


def iv_contract_vs_direct(tv, nrm, eps, obs):
    print("\n[iv] *_disp_contract == *_matrix_direct @ density")
    nu = 0.30
    lam = _lame(MU, nu)
    ct = kb.t_coeffs(MU, lam)
    cu = kb.u_coeffs(MU, lam)
    rng = np.random.default_rng(2)
    for p in (1, 2):
        K = tn.n_nodes(p)
        dens = rng.normal(size=(K * tv.shape[0], 3))
        dens[K * 7:K * 8] = 0.0
        T = tn.t_matrix_direct(obs, tv, nrm, eps, *ct, p)
        U = tn.u_matrix_direct(obs, tv, eps, *cu, p)
        check(f"p={p}: t_disp_contract vs t_matrix_direct @ density",
              relmax(tn.t_disp_contract(obs, tv, nrm, eps, dens, *ct, p),
                     (T @ dens.ravel()).reshape(-1, 3)), 1e-13)
        check(f"p={p}: u_disp_contract vs u_matrix_direct @ density",
              relmax(tn.u_disp_contract(obs, tv, eps, dens, *cu, p),
                     (U @ dens.ravel()).reshape(-1, 3)), 1e-13)
        # and the direct matrices are the recombined basis stacks
        check(f"p={p}: t_matrix_direct == t_coeffs . t_basis_matrices",
              relmax(T, np.tensordot(ct, tn.t_basis_matrices(obs, tv, nrm, eps, p), axes=1)), 1e-13)
        check(f"p={p}: u_matrix_direct == u_coeffs . u_basis_matrices",
              relmax(U, np.tensordot(cu, tn.u_basis_matrices(obs, tv, eps, p), axes=1)), 1e-13)


def v_pairing_tripwire(clq):
    print("\n[v] lam/mu pairing tripwire (nu = 0.30 sees a swap, nu = 0.25 cannot)")
    from clq.frame import local_frame
    from clq.kernels import lift
    from clq.moments import weighted_tables

    tri = np.array([[0.37, -0.81, 0.44], [1.92, 0.11, -0.63], [-0.25, 1.57, 1.22]])
    nrm = clq.unit_normal(tri)
    frame = local_frame(tri)
    fr = frame
    obs = _c([fr.centroid + 0.4 * fr.nhat, fr.centroid + 0.3 * fr.e1 + 0.05 * fr.nhat,
              fr.centroid + 1.5 * fr.L * (fr.e2 + fr.nhat)])
    eps = 0.1
    eye = np.eye(3)
    for p in ORDERS:
        for nu in NUS:
            lam = _lame(MU, nu)
            C1 = 1.0 / (16.0 * np.pi * MU * (1.0 - nu))
            c34 = 3.0 - 4.0 * nu
            W, z, _X, _r, _h0 = weighted_tables(frame, obs, eps, p, ("U",))
            V3 = lift(W, z, frame, 1, 3)
            V5 = lift(W, z, frame, 1, 5)
            T35 = lift(W, z, frame, 3, 5)
            G1 = C1 * (-c34 * np.einsum("ij,nkm->nkijm", eye, V3)
                       + np.einsum("im,nkj->nkijm", eye, V3)
                       + np.einsum("jm,nki->nkijm", eye, V3)
                       - 3.0 * T35
                       - 6.0 * (1.0 - nu) * eps * eps * np.einsum("ij,nkm->nkijm", eye, V5))
            t_grad = np.einsum("m,nkijm->nkij", nrm, G1)       # sum_m n_m dG_ij/dx_m
            t_trace = np.einsum("j,nkimm->nkij", nrm, G1)      # n_j sum_m dG_im/dx_m
            t_mix = np.einsum("m,nkimj->nkij", nrm, G1)        # sum_m n_m dG_im/dx_j
            U_ok = -(MU * t_grad + lam * t_trace + MU * t_mix)
            U_swap = -(lam * t_grad + MU * t_trace + MU * t_mix)
            got = nodal_tensors(obs, tri, nrm, eps, MU, nu, p)["U"]
            check(f"p={p} nu={nu:.2f}: port == traction-operator pairing", relmax(got, U_ok), 1e-12)
            d = relmax(got, U_swap)
            if nu == 0.25:
                check(f"p={p} nu={nu:.2f}: swapped form coincides (lam == mu)", d, 1e-12)
            else:
                check_bool(f"p={p} nu={nu:.2f}: swapped form differs by > 1e-3", d > 1e-3,
                           f"(rel diff {d:.2e})")


def vi_partition_of_unity(tv, nrm, eps, obs):
    print("\n[vi] partition of unity: uniform nodal density at P1/P2 == P0")
    nu = 0.30
    lam = _lame(MU, nu)
    ct = kb.t_coeffs(MU, lam)
    cu = kb.u_coeffs(MU, lam)
    rng = np.random.default_rng(4)
    d0 = rng.normal(size=(tv.shape[0], 3))
    ref = dict(
        t=tn.t_disp_contract(obs, tv, nrm, eps, d0, *ct, 0),
        u=tn.u_disp_contract(obs, tv, eps, d0, *cu, 0),
        dd=tn.dd_stress_contract(obs, tv, nrm, eps, d0, MU, lam, 0),
        kel=tn.kelvin_stress_contract(obs, tv, eps, d0, MU, lam, 0),
        eig=tn.eigenstress_contract(obs, tv, nrm, eps, d0, MU, lam, 0))
    for p in (1, 2):
        K = tn.n_nodes(p)
        d = np.repeat(d0, K, axis=0)                     # row K s + k = d0[s]
        got = dict(
            t=tn.t_disp_contract(obs, tv, nrm, eps, d, *ct, p),
            u=tn.u_disp_contract(obs, tv, eps, d, *cu, p),
            dd=tn.dd_stress_contract(obs, tv, nrm, eps, d, MU, lam, p),
            kel=tn.kelvin_stress_contract(obs, tv, eps, d, MU, lam, p),
            eig=tn.eigenstress_contract(obs, tv, nrm, eps, d, MU, lam, p))
        for key, label in (("t", "t_disp_contract"), ("u", "u_disp_contract"),
                           ("dd", "dd_stress_contract"), ("kel", "kelvin_stress_contract"),
                           ("eig", "eigenstress_contract")):
            check(f"p={p}: {label} with uniform nodal density == P0", relmax(got[key], ref[key]), 1e-13)


def main():
    print("=" * 76)
    print("Nodal P0/P1/P2 triangle kernels (mbem tri_nodal) vs clq")
    print("=" * 76)
    clq = _load_clq()
    if clq is None:
        print("  [XX] clq oracle not importable")
        print("-" * 76)
        print("FAIL: nodal triangle kernels (clq oracle missing)")
        return False
    zero_edge_primitives(clq)
    i_clq_parity(clq)
    print("\n[fixture for ii-iv, vi]")
    tv, nrm, eps, obs = random_mesh()
    ii_order0_vs_tri_kernels(tv, nrm, eps, obs)
    iii_parallel_vs_serial(tv, nrm, eps, obs)
    iv_contract_vs_direct(tv, nrm, eps, obs)
    v_pairing_tripwire(clq)
    vi_partition_of_unity(tv, nrm, eps, obs)
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: nodal triangle kernels vs clq ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: nodal triangle kernels vs clq "
              f"({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
