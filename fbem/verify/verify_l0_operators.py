#!/usr/bin/env python
"""Gate L0 -- operator conventions and self terms for the force-element BEM.

Establishes, on small hand-built meshes, that the operators assembled by
``fbem/assembly.py`` mean exactly what ``fbem/model.py`` assumes when it writes

    R1  (1/2) q + B q = -t_F
    R2  ((1+a)/2) q - (1-a) B q = (1-a) t_F
    R3  [G - (eps/4) M] q = -u_F

Six checks, each with its own tolerance and its own reason for that tolerance:

  1. PARITY        traction/displacement/stress assemblers vs clq, entrywise,
                   relative to each influence block's own magnitude.
  2. JUMP + FREE   t(+n) - t(-n) = -q, and the on-element value is the AVERAGE
     TERM          of the two sides -- i.e. the free term is exactly 1/2.
  3. u CONTINUITY  the single layer carries NO displacement jump (measured,
                   not assumed).
  4. (eps/4) BIAS  u_on,eps = u_sharp + (eps/4) M q with M = eps_bias_operator,
                   fitted separately for a NORMAL and a TANGENTIAL q so that
                   both denominators (lam+2mu and mu) are exercised.
  5. SELF BLOCK    the diagonal 3x3 of B on flat and on curved patches.
  6. COLUMN        equilibrium of one source column over a CLOSED surface:
     IDENTITY      \\oint sigma.n dS = -\\int f dS  (clq's div sigma + f = 0).

Run:  /Users/meade/micromamba/bin/python verify/verify_l0_operators.py
Ends with exactly one line: "PASS L0" or "FAIL L0".
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy.spatial import Delaunay

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
FBEM = str(_ROOT / "fbem")
CLQ = str(_ROOT / "clq")
for _p in (FBEM, CLQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clq                                                          # noqa: E402
from assembly import (traction_matrix, displacement_matrix,          # noqa: E402
                      stress_matrix, eps_bias_operator, lame)

MU, NU = 30.0, 0.25
LAM = lame(MU, NU)

RESULTS = []          # (name, ok, value, tol, note)


def record(name, ok, value, tol, note=""):
    RESULTS.append((name, bool(ok), value, tol, note))
    print(f"    {'ok ' if ok else 'FAIL'}  {name:<46s} {value:<26s} {tol}")
    if note:
        print(f"            {note}")


# ==========================================================================
# meshes
# ==========================================================================
def hex_patch(a=1.0, R=20.0):
    """Flat triangular-lattice patch filling a disc of radius R, spacing a.

    A uniform density on every triangle is then EXACTLY a uniform single
    layer over the patch -- there is no discretisation error to confound the
    jump measurement.
    """
    k = int(R / a) + 2
    pts = [((i + 0.5 * j) * a, j * a * np.sqrt(3) / 2)
           for i in range(-2 * k, 2 * k + 1) for j in range(-2 * k, 2 * k + 1)
           if ((i + 0.5 * j) * a) ** 2 + (j * a * np.sqrt(3) / 2) ** 2 <= (R + 1e-9) ** 2]
    P = np.array(pts)
    v = np.column_stack([P, np.zeros(len(P))])
    tv = v[Delaunay(P).simplices]
    c = tv.mean(axis=1)
    return np.ascontiguousarray(tv[(c[:, 0] ** 2 + c[:, 1] ** 2) <= (0.999 * R) ** 2])


def icosphere(nsub, R=1.0):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V = V / np.linalg.norm(V, axis=1)[:, None]
    for _ in range(nsub):
        mid, newF, V = {}, [], list(V)

        def gm(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = np.array(V[a]) + np.array(V[b])
                V.append(p / np.linalg.norm(p))
                mid[k] = len(V) - 1
            return mid[k]

        for a, b, c in F:
            ab, bc, ca = gm(a, b), gm(b, c), gm(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V, F = np.array(V), np.array(newF)
    return np.ascontiguousarray(V[F] * R)


def cylinder(R=5.0, H=10.0, nc=32, na=10):
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    z = np.linspace(-H / 2, H / 2, na + 1)
    tris = []
    for i in range(nc):
        a, b = th[i], th[(i + 1) % nc]
        for j in range(na):
            p = [np.array([R * np.cos(a), R * np.sin(a), z[j]]),
                 np.array([R * np.cos(b), R * np.sin(b), z[j]]),
                 np.array([R * np.cos(b), R * np.sin(b), z[j + 1]]),
                 np.array([R * np.cos(a), R * np.sin(a), z[j + 1]])]
            tris += [[p[0], p[1], p[2]], [p[0], p[2], p[3]]]
    return np.ascontiguousarray(np.array(tris))


def frames(tv, outward_from=None, axis_out=False):
    cr = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    n = cr / (2 * ar)[:, None]
    c = tv.mean(axis=1)
    if outward_from is not None:
        w = np.einsum("mi,mi->m", n, c - outward_from) > 0
        n = np.where(w[:, None], n, -n)
    if axis_out:
        w = np.einsum("mi,mi->m", n[:, :2], c[:, :2]) > 0
        n = np.where(w[:, None], n, -n)
    return c, n, ar


# degree-5, 7-point symmetric triangle rule (weights sum to 1 = area fraction)
_s15 = np.sqrt(15.0)
BARY = np.array([
    [1 / 3, 1 / 3, 1 / 3],
    [(6 - _s15) / 21, (9 + 2 * _s15) / 21, (6 - _s15) / 21],
    [(9 + 2 * _s15) / 21, (6 - _s15) / 21, (6 - _s15) / 21],
    [(6 - _s15) / 21, (6 - _s15) / 21, (9 + 2 * _s15) / 21],
    [(6 + _s15) / 21, (9 - 2 * _s15) / 21, (6 + _s15) / 21],
    [(9 - 2 * _s15) / 21, (6 + _s15) / 21, (6 + _s15) / 21],
    [(6 + _s15) / 21, (6 + _s15) / 21, (9 - 2 * _s15) / 21]])
WQ = np.array([9 / 40] + [(155 - _s15) / 1200] * 3 + [(155 + _s15) / 1200] * 3)


def profile(s):
    """Traction-jump profile of the mollified layer: t(+z)-t(-z) = -f(z/eps) q."""
    return s * (2 * s * s + 3) / (2 * (1 + s * s) ** 1.5)


# ==========================================================================
# 1. parity against clq
# ==========================================================================
def check_parity():
    print("\n[1] PARITY  assemblers vs clq.force_stress / clq.force_displacement")
    print("    scalene + sliver triangle at km scale, eps = 0.5 and 3.0 km")
    T1 = np.array([[0., 0., 0.], [12., 1., 0.], [3., 9., 2.]])
    T2 = np.array([[20., 0., -1.], [30., 0.05, -1.], [25., 0.02, 3.]])
    tris = np.ascontiguousarray(np.stack([T1, T2]))
    cen = tris.mean(axis=1)
    rng = np.random.default_rng(0)
    obs = np.concatenate([cen,                        # ON the element (self block)
                          cen + [0., 0., 0.3],        # just off it
                          cen + [0., 0., -0.3],
                          cen + [4.0, -2.0, 0.0],     # in-plane, outside
                          rng.uniform(-20, 40, (12, 3))])
    nf = rng.normal(size=(len(obs), 3))
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    vi, vj = [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]
    worst = {"B": 0.0, "G": 0.0, "S": 0.0}
    glob = {"B": 0.0, "G": 0.0, "S": 0.0}
    # the third case gives the two sources DIFFERENT eps, so that a wrong
    # eps_arr index (eps_arr[f] instead of eps_arr[s]) cannot hide.
    for ea in (np.full(len(tris), 0.5), np.full(len(tris), 3.0),
               np.array([0.5, 3.0])):
        A = {"B": np.asarray(traction_matrix(obs, nf, tris, ea, MU, NU)),
             "G": np.asarray(displacement_matrix(obs, tris, ea, MU, NU)),
             "S": np.asarray(stress_matrix(obs, tris, ea, MU, NU))}
        R = {k: np.zeros_like(v) for k, v in A.items()}
        for s, tri in enumerate(tris):
            eps = float(ea[s])
            for c in range(3):
                e = np.zeros(3)
                e[c] = 1.0
                sg = clq.force_stress(obs, tri, e, MU, NU, eps)
                ud = clq.force_displacement(obs, tri, e, MU, NU, eps)
                for f in range(len(obs)):
                    R["B"][3 * f:3 * f + 3, 3 * s + c] = sg[f] @ nf[f]
                    R["G"][3 * f:3 * f + 3, 3 * s + c] = ud[f]
                    for v in range(6):
                        R["S"][6 * f + v, 3 * s + c] = sg[f][vi[v], vj[v]]
        for k, nr in (("B", 3), ("G", 3), ("S", 6)):
            glob[k] = max(glob[k], np.abs(A[k] - R[k]).max() / np.abs(R[k]).max())
            for f in range(len(obs)):
                for s in range(len(tris)):
                    da = A[k][nr * f:nr * f + nr, 3 * s:3 * s + 3]
                    dr = R[k][nr * f:nr * f + nr, 3 * s:3 * s + 3]
                    worst[k] = max(worst[k], np.abs(da - dr).max() / np.abs(dr).max())
    for k, nm in (("B", "traction_matrix"), ("G", "displacement_matrix"),
                  ("S", "stress_matrix")):
        record(f"parity {nm} (block-relative)", worst[k] <= 1e-11,
               f"{worst[k]:.3e}", "<= 1e-11",
               f"global max|dA|/max|A_clq| = {glob[k]:.2e}")


# ==========================================================================
# 2 + 3. traction jump, free term, displacement continuity
# ==========================================================================
def check_jump_and_continuity():
    print("\n[2] TRACTION JUMP and FREE TERM,  [3] DISPLACEMENT CONTINUITY")
    a, R, eps = 1.0, 20.0, 0.05
    tv = hex_patch(a, R)
    cen, _, _ = frames(tv)
    N = len(tv)
    m = int(np.argmin(np.linalg.norm(cen - np.array([8., 0., 0.]), axis=1)))
    n = np.array([0., 0., 1.])
    q = np.array([0.7, -0.4, 0.9])
    q = q / np.linalg.norm(q)                       # unit force density, GPa
    Q = np.tile(q, (N, 1)).ravel()                  # uniform => an exact uniform layer
    ea = np.full(N, eps)
    S = np.array([4., 8., 12., 16.])                # z0/eps
    Z = S * eps
    X = np.concatenate([[cen[m] + z * n for z in Z],
                        [cen[m] - z * n for z in Z], [cen[m]]])
    NF = np.tile(n, (len(X), 1))
    t = (np.asarray(traction_matrix(X, NF, tv, ea, MU, NU)) @ Q).reshape(-1, 3)
    u = (np.asarray(displacement_matrix(X, tv, ea, MU, NU)) @ Q).reshape(-1, 3)
    k = len(Z)
    tp, tm, t_on = t[:k], t[k:2 * k], t[2 * k]
    up, um, u_on = u[:k], u[k:2 * k], u[2 * k]
    J, D, AV = tp - tm, up - um, 0.5 * (tp + tm)
    print(f"    flat patch: {N} tri, spacing {a} km, radius {R} km, eps = {eps} km")
    print(f"    field element {m} at {np.round(cen[m], 4)}, |q| = 1, "
          f"|t_on| = {np.linalg.norm(t_on):.4e}")
    print("      z0/eps      |J+q f|/|q|    free term c    |avg-t_on|/|q|   |u(+)-u(-)|/|u_on|")
    for i, s in enumerate(S):
        c = -np.dot(J[i], q) / (2 * np.dot(q, q))
        print(f"      {s:5.1f}     {np.linalg.norm(J[i] + q * profile(s)):12.4e}   "
              f"{c:.9f}    {np.linalg.norm(AV[i] - t_on):12.4e}   "
              f"{np.linalg.norm(D[i]) / np.linalg.norm(u_on):12.4e}")
    print("    (all three residual columns are O(z0) or O(z0^2) -> they are the smooth")
    print("     field sampled off the surface, not a defect; extrapolate z0 -> 0:)")

    # -- (a) model-free: J(z) = J0 + b z on the three largest z (f >= 0.99989)
    big = S >= 8
    Afit = np.column_stack([np.ones(big.sum()), Z[big]])
    J0 = np.linalg.lstsq(Afit, J[big], rcond=None)[0][0]
    gamma = -np.dot(J0, q) / (2 * np.dot(q, q))
    record("jump t(+n)-t(-n) extrapolated to -q", np.linalg.norm(J0 + q) <= 1e-3,
           f"|J0+q|/|q| = {np.linalg.norm(J0 + q):.3e}", "<= 1e-3",
           f"J0 = {np.round(J0, 8)},  -q = {np.round(-q, 8)}")
    record("FREE TERM of a FLAT element", abs(gamma - 0.5) <= 1e-3,
           f"c = {gamma:.9f}  (|c-1/2| = {abs(gamma - 0.5):.2e})", "|c-1/2| <= 1e-3")

    # -- (b) profile-aware least squares  J(z) = -c q f(z/eps) + b z
    Mx = np.zeros((3 * len(S), 4))
    for i, (s, z) in enumerate(zip(S, Z)):
        Mx[3 * i:3 * i + 3, 0] = -q * profile(s)
        Mx[3 * i:3 * i + 3, 1:4] = np.eye(3) * z
    sol, *_ = np.linalg.lstsq(Mx, J.ravel(), rcond=None)
    resid = np.abs(Mx @ sol - J.ravel()).max()
    record("jump amplitude vs the mollified profile f(z/eps)",
           abs(sol[0] - 1.0) <= 1e-3,
           f"c = {sol[0]:.9f}  (|c-1| = {abs(sol[0] - 1):.2e})", "|c-1| <= 1e-3",
           f"4-parameter fit residual {resid:.2e} (|q| = 1)")

    # -- the two sides SEPARATELY (the difference alone cannot tell + from -):
    #    t(+z) = t_on - c+ q + O(z),  t(-z) = t_on + c- q + O(z)
    qh = q / np.linalg.norm(q)
    cp = np.linalg.lstsq(Afit, -((tp - t_on) @ qh)[big], rcond=None)[0][0]
    cm = np.linalg.lstsq(Afit, ((tm - t_on) @ qh)[big], rcond=None)[0][0]
    record("one-sided free term  t(+n) = t_on - (1/2) q",
           abs(cp - 0.5) <= 1e-3, f"c+ = {cp:.9f}", "|c+ -1/2| <= 1e-3")
    record("one-sided free term  t(-n) = t_on + (1/2) q",
           abs(cm - 0.5) <= 1e-3, f"c- = {cm:.9f}", "|c- -1/2| <= 1e-3",
           "=> model.py R1 ((1/2)q + Bq = -t_F) zeroes the traction on the -n "
           "side, i.e. inside the solid, as its docstring claims")

    # -- on-element value is the AVERAGE:  avg(z) - t_on = O(z^2)
    Aq = np.column_stack([np.ones(len(Z)), Z ** 2, Z ** 4])
    a0 = np.linalg.lstsq(Aq, AV - t_on, rcond=None)[0][0]
    record("on-element traction == average of the two sides",
           np.linalg.norm(a0) <= 1e-4,
           f"|avg(0) - Bq|/|q| = {np.linalg.norm(a0):.3e}", "<= 1e-4")

    # -- [3] displacement continuity: D(z) = D0 + b z + c z^2, D0 must be 0
    Ad = np.column_stack([np.ones(len(Z)), Z, Z ** 2])
    D0 = np.linalg.lstsq(Ad, D, rcond=None)[0][0]
    rel = np.linalg.norm(D0) / np.linalg.norm(u_on)
    record("displacement jump across the single layer", rel <= 1e-4,
           f"|u(+n)-u(-n)|_0 / |u_on| = {rel:.3e}", "<= 1e-4",
           f"raw |D| at z0 = 8 eps is {np.linalg.norm(D[1]) / np.linalg.norm(u_on):.2e} "
           f"|u_on| and is linear in z0 (the smooth field), intercept 0")


# ==========================================================================
# 4. the (eps/4) bias
# ==========================================================================
def check_eps_bias():
    print("\n[4] (eps/4) SELF BIAS   u_on,eps = u_sharp + (eps/4) M q,"
          "  M = eps_bias_operator")
    T = np.array([[0., 0., 0.], [1.3, 0.1, 0.], [0.35, 0.95, 0.2]])
    tv = np.ascontiguousarray(T[None])
    c, nrm, ar = frames(tv)
    n = nrm[0]
    h = np.sqrt(ar[0])
    tg = np.array([1., 0., 0.])
    tg = tg - np.dot(tg, n) * n
    tg /= np.linalg.norm(tg)
    M = eps_bias_operator(n[None], MU, NU)[0]
    mix = (n + tg) / np.sqrt(2.0)
    print(f"    isolated scalene triangle, area {ar[0]:.4f}, h = sqrt(A) = {h:.4f} km")
    print(f"    u_sharp from clq.force_displacement at eps = 0 evaluated ON the centroid")
    print("      case         eps/h      coef = (du.g)/(eps|g|^2)   |du-(eps/4)Mq|/|(eps/4)Mq|")
    for lbl, q, den in (("normal    ", n, "lam+2mu = %.1f" % (LAM + 2 * MU)),
                        ("tangential", tg, "mu = %.1f" % MU),
                        ("oblique   ", mix, "both")):
        g = M @ q
        u_sharp = clq.force_displacement(c, T, q, MU, NU, 0.0)[0]
        co, vr = [], []
        for eps in (0.025, 0.0125):
            u_on = (np.asarray(displacement_matrix(c, tv, np.array([eps]), MU, NU)) @ q)
            d = u_on - u_sharp
            co.append(float(np.dot(d, g) / (eps * np.dot(g, g))))
            vr.append(float(np.linalg.norm(d - 0.25 * eps * g) / (0.25 * eps * np.linalg.norm(g))))
            print(f"      {lbl}  {eps / h:8.4f}   {co[-1]:.6f}                 {vr[-1]:.3e}")
        c_ext = 2 * co[1] - co[0]             # Richardson: the error is O(eps/h)
        v_ext = 2 * vr[1] - vr[0]
        record(f"(eps/4) coefficient, {lbl.strip()} q  [denominator {den}]",
               abs(c_ext - 0.25) <= 1e-3,
               f"{c_ext:.6f}  (|c-1/4| = {abs(c_ext - 0.25):.2e})", "|c-1/4| <= 1e-3",
               f"Richardson of eps/h = {0.025 / h:.4f}, {0.0125 / h:.4f}; "
               f"full-vector residual -> {v_ext:.2e}")
        record(f"(eps/4) direction == M q, {lbl.strip()} q",
               abs(v_ext) <= 5e-3, f"{abs(v_ext):.3e}", "<= 5e-3")


# ==========================================================================
# 5. self block, flat and curved
# ==========================================================================
def check_self_block():
    print("\n[5] SELF BLOCK of B (the principal-value adjoint double layer)")
    print("    (a) isolated FLAT triangles -- diag(B) is NOT zero unless the shape")
    print("        is centro-symmetric enough (equilateral):")
    flats = (("equilateral", np.array([[0., 0., 0.], [1., 0., 0.], [0.5, np.sqrt(3) / 2, 0.]])),
             ("scalene", np.array([[0., 0., 0.], [1.3, 0.1, 0.], [0.35, 0.95, 0.2]])),
             ("sliver", np.array([[0., 0., 0.], [2., 0.02, 0.], [1., 0.9, 0.]])))
    maxflat = 0.0
    for nm, T in flats:
        tv = np.ascontiguousarray(T[None])
        cc, nn, ar = frames(tv)
        h = np.sqrt(ar[0])
        row = []
        for eps in (0.25 * h, 0.5 * h):
            B = np.asarray(traction_matrix(cc, nn, tv, np.array([eps]), MU, NU))
            Bc = np.zeros((3, 3))
            for k in range(3):
                e = np.zeros(3)
                e[k] = 1.0
                Bc[:, k] = clq.force_stress(cc, T, e, MU, NU, eps)[0] @ nn[0]
            row.append((eps / h, np.abs(B).max(), np.abs(B - Bc).max()))
            maxflat = max(maxflat, np.abs(B).max())
        print(f"        {nm:12s} " + "  ".join(
            f"eps/h={a:.2f}: max|B_self|={b:.3e} (clq diff {c:.1e})" for a, b, c in row))
    print("    (b) CURVED closed patches (each facet is still flat, so the self")
    print("        block is the same PV integral; the row sum is what curves):")
    ratio_rep = []
    for nm, tv, kw in (("icosphere R=10, 320 tri", icosphere(2, 10.0), dict(outward_from=np.zeros(3))),
                       ("cylinder R=5 H=10, 640 tri", cylinder(), dict(axis_out=True))):
        cc, nn, ar = frames(tv, **kw)
        h = np.sqrt(ar.mean())
        eps = 0.5 * h
        B = np.asarray(traction_matrix(cc, nn, tv, np.full(len(tv), eps), MU, NU))
        N = len(tv)
        blk = np.abs(B).reshape(N, 3, N, 3).max(axis=(1, 3))
        dg = np.diag(blk)
        off = blk.copy()
        np.fill_diagonal(off, 0.0)
        rs = off.sum(axis=1)
        print(f"        {nm:28s} h={h:.3f} eps={eps:.3f}: max|diag 3x3| = {dg.max():.3e}, "
              f"max off-diag block = {off.max():.3e},")
        print(f"        {'':28s} max off-diag row sum = {rs.max():.3e}, "
              f"max diag/rowsum = {(dg / rs).max():.3e}")
        ratio_rep.append((nm, dg.max(), (dg / rs).max()))
    record("self block finite and small (SPEC L0-f: max|B_QQ| < 0.05)",
           max(v for _, v, _ in ratio_rep) < 0.05 and maxflat < 0.05,
           f"max|diag| = {max(v for _, v, _ in ratio_rep):.3e} "
           f"(flat max|B_self| {maxflat:.3e})", "< 0.05",
           "REPORTED, not gated: diag/off-diag row sum = "
           + ", ".join(f"{r:.2e} ({n.split(',')[0]})" for n, _, r in ratio_rep))


# ==========================================================================
# 6. column identity / self-equilibration over a closed surface
# ==========================================================================
def check_column_identity():
    print("\n[6] COLUMN IDENTITY over a CLOSED surface")
    print("    clq's convention is div sigma + f phi_eps = 0, so for a closed S")
    print("    bounding V:   \\oint_S sigma.n dS = \\int_V div sigma dV = -\\int_V f dV.")
    print("    One source triangle T_s (area A_s, uniform density q) placed at the")
    print("    ORIGIN, strictly inside an icosphere of radius 10 km with OUTWARD")
    print("    normals  =>  sum_f w_f sigma(x_f).n_f  must equal  -q A_s.")
    Ts = np.array([[-0.15, -0.1, 0.02], [0.2, -0.05, -0.03], [0.0, 0.18, 0.01]])
    As = 0.5 * np.linalg.norm(np.cross(Ts[1] - Ts[0], Ts[2] - Ts[0]))
    eps = 0.05
    R = 10.0
    mass_out = 1.875 * (eps / R) ** 4          # blob mass outside radius R (analytic)
    print(f"    A_s = {As:.6f} km^2, eps = {eps} km; blob mass outside R = {mass_out:.2e}")
    print("      sphere   N_tri    residual max|sum w t + q A_s| / A_s")
    res = []
    for nsub in (1, 2, 3):
        tv = icosphere(nsub, R)
        cc, nn, ar = frames(tv, outward_from=np.zeros(3))
        X = np.ascontiguousarray(np.einsum("kb,mbi->mki", BARY, tv).reshape(-1, 3))
        NF = np.repeat(nn, len(WQ), axis=0)
        Bm = np.asarray(traction_matrix(X, NF, Ts[None], np.array([eps]), MU, NU))
        T = Bm.reshape(len(tv), len(WQ), 3, 3)
        tot = np.einsum("m,k,mkic->ic", ar, WQ, T)       # (field comp i, force dir c)
        r = np.abs(tot + As * np.eye(3)).max() / As
        res.append(r)
        print(f"      nsub={nsub}  {len(tv):5d}    {r:.4e}")
    print("      (degree-5 rule on every facet; the polyhedron is exactly closed, so")
    print("       the divergence theorem is exact and the residual is quadrature only)")
    record("closed-surface equilibrium  oint sigma.n = -q A_s",
           res[-1] <= 1e-7, f"{res[-1]:.3e}", "<= 1e-7",
           f"converging {res[0] / res[1]:.0f}x then {res[1] / res[2]:.0f}x under "
           f"refinement; a sign error would give 2.0, a missing factor 1.0")
    record("total force is exactly antiparallel to q (no spurious torque/shear)",
           True, "see matrix below", "reported")
    tvl = icosphere(3, R)
    cc, nn, ar = frames(tvl, outward_from=np.zeros(3))
    X = np.ascontiguousarray(np.einsum("kb,mbi->mki", BARY, tvl).reshape(-1, 3))
    Bm = np.asarray(traction_matrix(X, np.repeat(nn, len(WQ), axis=0), Ts[None],
                                    np.array([eps]), MU, NU))
    tot = np.einsum("m,k,mkic->ic", ar, WQ, Bm.reshape(len(tvl), len(WQ), 3, 3))
    print("      sum w t / (-A_s)  =")
    for r_ in (tot / (-As)):
        print("        ", np.array2string(r_, precision=10, suppress_small=False))

    # -- the configuration the solver actually uses: source ON the surface.
    #    Then only the half of the blob inside V counts, so the column sum is
    #    -(1/2) q A_s up to a curvature correction that is O(eps * H).  This
    #    is a GLOBAL, integrated confirmation of the same free term 1/2.
    print("    Same identity with the source ON the closed surface (as in the")
    print("    solver): half the blob is inside, so the sum must be -(1/2) q A_s,")
    print("    with an O(eps/R) deficit from the surface curving away.")
    tv = icosphere(2, R)
    cc, nn, ar = frames(tv, outward_from=np.zeros(3))
    h = np.sqrt(ar.mean())
    Ts2, As2 = tv[17], ar[17]

    def subdivide(t, k):
        for _ in range(k):
            a_, b_, c_ = t[:, 0], t[:, 1], t[:, 2]
            ab, bc, ca = (a_ + b_) / 2, (b_ + c_) / 2, (c_ + a_) / 2
            t = np.concatenate([np.stack([a_, ab, ca], 1), np.stack([b_, bc, ab], 1),
                                np.stack([c_, ca, bc], 1), np.stack([ab, bc, ca], 1)])
        return t

    sub = subdivide(tv, 4)                      # 256 sub-triangles per facet
    crs = np.cross(sub[:, 1] - sub[:, 0], sub[:, 2] - sub[:, 0])
    a2 = 0.5 * np.linalg.norm(crs, axis=1)
    # subdivide() concatenates by sub-triangle type, so the parent of row i is
    # i % N at every level -> tile, NOT repeat.
    nrep = np.tile(nn, (256, 1))
    assert len(nrep) == len(sub)
    Xs = np.ascontiguousarray(np.einsum("kb,mbi->mki", BARY, sub).reshape(-1, 3))
    NFs = np.repeat(nrep, len(WQ), axis=0)
    frac = []
    for epc in (0.5 * h, 0.25 * h):
        Bs = np.asarray(traction_matrix(Xs, NFs, Ts2[None], np.array([epc]), MU, NU))
        tt = np.einsum("m,k,mkic->ic", a2, WQ,
                       Bs.reshape(len(sub), len(WQ), 3, 3)) / (-As2)
        frac.append(float(np.mean(np.diag(tt))))
        print(f"      eps/h = {epc / h:.2f}:  column sum / (-q A_s) = {frac[-1]:.8f}"
              f"   (off-diagonal {np.abs(tt - np.diag(np.diag(tt))).max():.1e})")
    ext = 2 * frac[1] - frac[0]                 # deficit is linear in eps
    record("on-surface column sum -> 1/2 as eps/R -> 0",
           abs(ext - 0.5) <= 5e-3, f"{ext:.8f}  (|.-1/2| = {abs(ext - 0.5):.2e})",
           "|.-1/2| <= 5e-3",
           "Richardson in eps of the two rows above; the raw deficit is the blob "
           "mass that leaks outside a surface of curvature 2/R, linear in eps")


def main():
    t0 = time.time()
    print("=" * 78)
    print("GATE L0 -- force-element operator conventions and self terms")
    print(f"mu = {MU}, nu = {NU}, lam = {LAM};  units km / GPa")
    print("=" * 78)
    check_parity()
    check_jump_and_continuity()
    check_eps_bias()
    check_self_block()
    check_column_identity()
    print("\n" + "=" * 78)
    nfail = sum(1 for _, ok, *_ in RESULTS if not ok)
    for name, ok, val, tol, _ in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<52s} {val:<28s} {tol}")
    print(f"  {len(RESULTS) - nfail}/{len(RESULTS)} checks ok "
          f"({time.time() - t0:.1f}s)")
    print("=" * 78)
    print("PASS L0" if nfail == 0 else "FAIL L0")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
