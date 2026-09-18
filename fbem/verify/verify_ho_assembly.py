#!/usr/bin/env python
"""Gate ho_assembly -- the P0/P1/P2 nodal force-density assembler.

Pins ``fbem/assembly_ho.py`` -- influence matrices for a DISCONTINUOUS
(per-element) Lagrange nodal force density built on ``clq.influence`` -- before
anything is asked of it about convergence.  Five checks:

  1. P0 REDUCTION      order=0 reproduces ``assembly.traction_matrix``,
                       ``displacement_matrix`` and ``stress_matrix`` entrywise,
                       block-relative, <= 1e-11.  Three eps settings, the third
                       giving the two sources DIFFERENT eps so a wrong eps index
                       cannot hide.  This is the only check that ties the new
                       path to the numba assembler the project already trusts.

  2. CONSTANT DENSITY  a P1/P2 density with every nodal value equal to the same
                       vector gives the same field as a P0 density of that
                       vector: sum_k of the (3,3) sub-blocks over the K nodes
                       equals the P0 block.  Partition of unity.  NOTE it is
                       invariant under a PERMUTATION of the nodes, so it cannot
                       by itself pin the node order -- check 3 does that.

  3. QUADRATURE        one unit density at ONE node at a time, against an
                       independent Gauss product rule on the source triangle:
                       this file's own Duffy rule, its own transcription of the
                       mollified Kelvin G and its stress, and its own
                       barycentric shape functions.  Nothing of clq's
                       integration or shape machinery is reused, so a wrong
                       shape-function ordering, a wrong (node, component)
                       column packing or a wrong stress contraction all fail
                       here.  The rule is self-validated by refining it.

  4. RECIPROCITY       U[i,j] = -n_m S[j,m,i] (clq's own identity, gate (8) of
                       clq/verify/verify_identities.py) read off the ASSEMBLED
                       traction matrix with the field normal set to the source
                       triangle's own normal, against clq's independent slip
                       kernel U.  A sign tripwire comes with it.

  5. PARALLEL PATH     workers > 1 returns a bitwise-identical matrix.

then TIMINGS: seconds to assemble a dense traction matrix on a box mesh at
N_tri = 512 and 2048 for each order, field points at the centroids -- i.e. the
operator the next stage actually builds -- so it can size its meshes.  That
section dominates the runtime (~6 min); ``--quick`` drops the N_tri = 2048 row.

Run:  /Users/meade/micromamba/bin/python verify/verify_ho_assembly.py [--quick]
Ends with exactly one line: "PASS ho_assembly" or "FAIL ho_assembly".
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "fbem"), str(_ROOT / "clq")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clq                                                          # noqa: E402
from assembly import (traction_matrix, displacement_matrix,          # noqa: E402
                      stress_matrix, lame)
import assembly_ho as aho                                            # noqa: E402

MU, NU = 30.0, 0.25
LAM = lame(MU, NU)
ORDERS = (0, 1, 2)

RESULTS = []


def record(name, ok, value, tol, note=""):
    RESULTS.append((name, bool(ok), value, tol, note))
    print(f"    {'ok ' if ok else 'FAIL'}  {name:<52s} {value:<14s} {tol}")
    if note:
        print(f"            {note}")


def block_rel(A, B, nrow, ncol, n_f, n_s):
    """Worst block-relative difference over (field, source) blocks.

    Entries pass through zero, so each (nrow, ncol) block is scaled by its OWN
    largest reference entry rather than by a global norm.
    """
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape:
        raise AssertionError(f"shape {A.shape} vs {B.shape}")
    worst = 0.0
    for f in range(n_f):
        for s in range(n_s):
            a = A[nrow * f:nrow * (f + 1), ncol * s:ncol * (s + 1)]
            b = B[nrow * f:nrow * (f + 1), ncol * s:ncol * (s + 1)]
            scale = np.abs(b).max()
            if scale == 0.0:
                worst = max(worst, np.abs(a).max())
            else:
                worst = max(worst, np.abs(a - b).max() / scale)
    return float(worst)


def relmax(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.abs(b).max()
    return float(np.abs(a - b).max() / s) if s > 0 else float(np.abs(a).max())


# ==========================================================================
# geometry
# ==========================================================================
T1 = np.array([[0., 0., 0.], [12., 1., 0.], [3., 9., 2.]])
T2 = np.array([[20., 0., -1.], [30., 0.05, -1.], [25., 0.02, 3.]])
TRIS2 = np.ascontiguousarray(np.stack([T1, T2]))


def parity_obs():
    """Observation points that include the self (on-element) point."""
    cen = TRIS2.mean(axis=1)
    rng = np.random.default_rng(0)
    obs = np.concatenate([cen,                       # ON the element
                          cen + [0., 0., 0.3],       # just off it
                          cen + [0., 0., -0.3],
                          cen + [4.0, -2.0, 0.0],    # in-plane, outside
                          rng.uniform(-20, 40, (12, 3))])
    nf = rng.normal(size=(len(obs), 3))
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    return np.ascontiguousarray(obs), np.ascontiguousarray(nf)


def quad_face(origin, e1, e2, n1, n2):
    """Structured triangulation of a parallelogram; normals along e1 x e2."""
    a, b = np.linspace(0, 1, n1 + 1), np.linspace(0, 1, n2 + 1)
    A, B = np.meshgrid(a, b, indexing="ij")
    V = (np.asarray(origin, float)[None, None, :]
         + A[..., None] * np.asarray(e1, float)[None, None, :]
         + B[..., None] * np.asarray(e2, float)[None, None, :])
    idx = lambda i, j: i * (n2 + 1) + j                          # noqa: E731
    tris = [t for i in range(n1) for j in range(n2)
            for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                      [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])]
    return V.reshape(-1, 3)[np.array(tris, int)]


def box_mesh(h, L=20.0, D=20.0):
    """Closed box, same construction as verify_l2's box_faces: 4 nxy^2 + 8 nxy nz
    triangles, so h = 5 gives exactly 512 and h = 2.5 exactly 2048."""
    nxy, nz = max(2, int(round(2 * L / h))), max(2, int(round(D / h)))
    X = 2 * L
    faces = [quad_face((-L, -L, 0.0), (X, 0, 0), (0, X, 0), nxy, nxy),
             quad_face((-L, -L, -D), (0, X, 0), (X, 0, 0), nxy, nxy),
             quad_face((L, -L, -D), (0, X, 0), (0, 0, D), nxy, nz),
             quad_face((-L, -L, -D), (0, 0, D), (0, X, 0), nz, nxy),
             quad_face((-L, L, -D), (0, 0, D), (X, 0, 0), nz, nxy),
             quad_face((-L, -L, -D), (X, 0, 0), (0, 0, D), nxy, nz)]
    return np.ascontiguousarray(np.concatenate(faces))


def normals_of(tv):
    n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    return n / np.linalg.norm(n, axis=1)[:, None]


def small_mesh(n=4):
    """A closed-ish scatter of generic triangles (no two alike)."""
    rng = np.random.default_rng(11)
    base = rng.uniform(-6.0, 6.0, (n, 3))
    tv = base[:, None, :] + rng.uniform(-2.5, 2.5, (n, 3, 3))
    return np.ascontiguousarray(tv)


# ==========================================================================
# 1. P0 reduction against the numba assembler
# ==========================================================================
def check_p0_reduction():
    print("\n[1] P0 REDUCTION  order=0 vs assembly.py (block-relative)")
    obs, nf = parity_obs()
    n_f, n_s = len(obs), len(TRIS2)
    worst = {"traction": 0.0, "displacement": 0.0, "stress": 0.0}
    for ea in (np.full(n_s, 0.5), np.full(n_s, 3.0), np.array([0.5, 3.0])):
        ref_t = np.asarray(traction_matrix(obs, nf, TRIS2, ea, MU, NU))
        ref_g = np.asarray(displacement_matrix(obs, TRIS2, ea, MU, NU))
        ref_s = np.asarray(stress_matrix(obs, TRIS2, ea, MU, NU))
        got_t = aho.traction_matrix_ho(obs, nf, TRIS2, ea, MU, NU, 0)
        got_g = aho.displacement_matrix_ho(obs, TRIS2, ea, MU, NU, 0)
        got_s = aho.stress_matrix_ho(obs, TRIS2, ea, MU, NU, 0)
        worst["traction"] = max(worst["traction"],
                                block_rel(got_t, ref_t, 3, 3, n_f, n_s))
        worst["displacement"] = max(worst["displacement"],
                                    block_rel(got_g, ref_g, 3, 3, n_f, n_s))
        worst["stress"] = max(worst["stress"],
                              block_rel(got_s, ref_s, 6, 3, n_f, n_s))
    for k, v in worst.items():
        record(f"P0 reduction: {k}_matrix_ho == assembly.{k}_matrix",
               v <= 1e-11, f"{v:.3e}", "<= 1e-11")
    # shape and unknown-count bookkeeping
    ok = True
    for p in ORDERS:
        K = aho.n_nodes(p)
        A = aho.traction_matrix_ho(obs[:3], nf[:3], TRIS2, 1.0, MU, NU, p)
        ok &= A.shape == (9, 3 * K * n_s) and A.flags.f_contiguous
        ok &= aho.n_unknowns(n_s, p) == 3 * K * n_s
        ok &= aho.element_nodes(TRIS2, p).shape == (n_s, K, 3)
    record("shapes: (3 N_f, 3 K N_tri), Fortran-ordered, K = 1/3/6",
           ok, "exact" if ok else "mismatch", "exact")
    # the nodes= contract must REJECT a wrong node convention
    bad = aho.element_nodes(TRIS2, 2).copy()
    bad[:, [3, 4]] = bad[:, [4, 3]]                   # swap two edge midpoints
    caught = False
    try:
        aho.traction_matrix_ho(obs[:3], nf[:3], TRIS2, 1.0, MU, NU, 2, nodes=bad)
    except ValueError:
        caught = True
    good = True
    try:
        aho.traction_matrix_ho(obs[:3], nf[:3], TRIS2, 1.0, MU, NU, 2,
                               nodes=aho.element_nodes(TRIS2, 2))
    except ValueError:
        good = False
    record("nodes= contract: right one accepted, permuted one rejected",
           caught and good, f"{caught}/{good}", "True/True")


# ==========================================================================
# 2. constant-density consistency (partition of unity)
# ==========================================================================
def check_constant_density():
    print("\n[2] CONSTANT DENSITY  sum_k of the P1/P2 sub-blocks == the P0 block")
    print("    (permutation-invariant by construction: check 3 pins the node order)")
    tv = small_mesh(5)
    rng = np.random.default_rng(3)
    obs = np.concatenate([tv.mean(axis=1), rng.uniform(-14, 14, (10, 3))])
    nf = rng.normal(size=(len(obs), 3))
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    n_f, n_s = len(obs), len(tv)
    ea = rng.uniform(0.3, 1.2, n_s)
    A0 = {"traction": aho.traction_matrix_ho(obs, nf, tv, ea, MU, NU, 0),
          "displacement": aho.displacement_matrix_ho(obs, tv, ea, MU, NU, 0),
          "stress": aho.stress_matrix_ho(obs, tv, ea, MU, NU, 0)}
    rows = {"traction": 3, "displacement": 3, "stress": 6}
    q = np.array([0.31, -0.77, 0.52])
    for p in (1, 2):
        K = aho.n_nodes(p)
        Ap = {"traction": aho.traction_matrix_ho(obs, nf, tv, ea, MU, NU, p),
              "displacement": aho.displacement_matrix_ho(obs, tv, ea, MU, NU, p),
              "stress": aho.stress_matrix_ho(obs, tv, ea, MU, NU, p)}
        for key, A in Ap.items():
            nr = rows[key]
            summed = A.reshape(nr * n_f, n_s, K, 3).sum(axis=2).reshape(nr * n_f, 3 * n_s)
            w = block_rel(summed, A0[key], nr, 3, n_f, n_s)
            record(f"P{p} sum_k block == P0 block: {key}", w <= 1e-11,
                   f"{w:.3e}", "<= 1e-11")
        # the same statement on the contracted field, which is what a solver sees
        f_p = Ap["traction"] @ aho.uniform_density(q, n_s, p)
        f_0 = A0["traction"] @ aho.uniform_density(q, n_s, 0)
        r = relmax(f_p, f_0)
        record(f"P{p} uniform nodal density -> same traction field", r <= 1e-11,
               f"{r:.3e}", "<= 1e-11")
    # tripwire: the check is not vacuous -- a NON-constant nodal density differs
    rngq = np.random.default_rng(5)
    Kp = aho.n_nodes(2)
    Ap2 = aho.traction_matrix_ho(obs, nf, tv, ea, MU, NU, 2)
    var = rngq.normal(size=3 * Kp * n_s)
    d = relmax(Ap2 @ var, Ap2 @ aho.uniform_density(q, n_s, 2))
    record("tripwire: a non-constant nodal density gives a DIFFERENT field",
           d > 1e-2, f"{d:.2e}", "> 1e-2")


# ==========================================================================
# 3. independent Gauss quadrature on the source triangle
# ==========================================================================
def shape_bary(order, lam):
    """Lagrange shape functions (Q, K) at barycentric lam (Q, 3).

    Node order, transcribed from clq/shape.py's documented lattice and NOT
    imported from it: p=0 centroid; p=1 v1, v2, v3; p=2 v1, v2, v3, m12, m23, m31.
    """
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    if order == 0:
        return np.ones((lam.shape[0], 1))
    if order == 1:
        return np.stack([l1, l2, l3], axis=1)
    if order == 2:
        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                         4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)
    raise ValueError(order)


def tri_rule(tri, n):
    """Duffy-collapsed n x n Gauss product rule: points (Q,3), weights (Q,),
    barycentric coordinates (Q,3).  Weights already carry the area element."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = 0.5 * (x + 1.0), 0.5 * w                      # to [0, 1]
    U, V = np.meshgrid(x, x, indexing="ij")
    WU, WV = np.meshgrid(w, w, indexing="ij")
    a = U.ravel()
    b = (V * (1.0 - U)).ravel()
    jac = (1.0 - U).ravel()
    lam = np.stack([1.0 - a - b, a, b], axis=1)
    y = lam @ tri
    area2 = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    wq = (WU * WV).ravel() * jac * area2
    return y, wq, lam


def kelvin_pair(d, mu, nu, eps):
    """G[n,i,j] and S[n,i,j,c] of the Cortez-mollified point force, transcribed
    from the closed form (not imported): R = sqrt(|d|^2 + eps^2), d = x - y."""
    d = np.asarray(d, float).reshape(-1, 3)
    R = np.sqrt(np.einsum("ni,ni->n", d, d) + eps ** 2)
    iR, iR3, iR5 = 1.0 / R, 1.0 / R ** 3, 1.0 / R ** 5
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    a34, be = 3.0 - 4.0 * nu, 2.0 * (1.0 - nu) * eps ** 2
    I = np.eye(3)
    G = C1 * ((a34 * iR + be * iR3)[:, None, None] * I[None]
              + np.einsum("ni,nj->nij", d, d) * iR3[:, None, None])
    # dG[n,i,j,m] = dG_ij/dx_m
    dG = C1 * (-a34 * np.einsum("ij,nm,n->nijm", I, d, iR3)
               + np.einsum("im,nj,n->nijm", I, d, iR3)
               + np.einsum("jm,ni,n->nijm", I, d, iR3)
               - 3.0 * np.einsum("ni,nj,nm,n->nijm", d, d, d, iR5)
               - 3.0 * be * np.einsum("ij,nm,n->nijm", I, d, iR5))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    div = np.einsum("naca->nc", dG)                      # dG_ac/dx_a
    S = (lam * np.einsum("ij,nc->nijc", I, div)
         + mu * np.einsum("nicj->nijc", dG)
         + mu * np.einsum("njci->nijc", dG))
    return G, S


def quad_influence(obs, tri, order, mu, nu, eps, n_gauss):
    """G[n,k,i,j], S[n,k,i,j,c] by the rule above."""
    y, wq, lam = tri_rule(tri, n_gauss)
    N = shape_bary(order, lam)                            # (Q, K)
    n_f, K = len(obs), N.shape[1]
    G = np.zeros((n_f, K, 3, 3))
    S = np.zeros((n_f, K, 3, 3, 3))
    for q in range(len(y)):
        g, s = kelvin_pair(obs - y[q], mu, nu, eps)
        wk = wq[q] * N[q]
        G += np.einsum("k,nij->nkij", wk, g)
        S += np.einsum("k,nijc->nkijc", wk, s)
    return G, S


def check_quadrature():
    print("\n[3] QUADRATURE  one node at a time vs this file's own Gauss rule")
    tri = np.array([[0.0, 0.0, 0.0], [10.0, 1.5, 0.0], [2.5, 8.0, 1.5]])
    L = float(np.sqrt(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))))
    cen = tri.mean(axis=0)
    nh = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nh = nh / np.linalg.norm(nh)
    ip = (tri[1] - tri[0]) / np.linalg.norm(tri[1] - tri[0])
    obs = np.array([cen + 0.4 * L * nh,
                    cen - 0.4 * L * nh,
                    cen + 1.0 * L * (0.6 * nh + 0.8 * ip),
                    cen + 3.0 * L * (0.5 * nh - 0.87 * ip),
                    cen + 15.0 * L * (0.3 * nh + 0.95 * ip),
                    tri[0] + 0.5 * L * nh,
                    tri[2] - 0.7 * L * nh,
                    cen + 6.0 * L * np.array([0.3, -0.5, 0.81])])
    rng = np.random.default_rng(17)
    nf = rng.normal(size=(len(obs), 3))
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    eps = 0.25 * L
    tv = tri[None]

    dist = np.linalg.norm(obs - cen, axis=1) / L

    def per_obs_rel(got, ref, where=False):
        """max over field points of (that point's worst entry error) / (that
        point's own largest reference entry) -- so the distant observer, whose
        entries are orders of magnitude smaller, is judged on its own scale."""
        got, ref = np.asarray(got), np.asarray(ref)
        ax = tuple(range(1, ref.ndim))
        den = np.abs(ref).max(axis=ax)
        num = np.abs(got - ref).max(axis=ax)
        r = num / np.where(den > 0, den, 1.0)
        if where:
            n = int(np.argmax(r))
            return float(r[n]), f"worst observer at D/L = {dist[n]:.2f}"
        return float(r.max())

    # self-validation of the oracle: refine the rule and see it stop moving
    g48, s48 = quad_influence(obs, tri, 2, MU, NU, eps, 48)
    g72, s72 = quad_influence(obs, tri, 2, MU, NU, eps, 72)
    conv = max(per_obs_rel(g48, g72), per_obs_rel(s48, s72))
    record("oracle self-check: 48x48 vs 72x72 Gauss rule", conv <= 1e-12,
           f"{conv:.3e}", "<= 1e-12")
    for p in ORDERS:
        K = aho.n_nodes(p)
        Gq, Sq = quad_influence(obs, tri, p, MU, NU, eps, 72)
        Tq = np.einsum("nkijc,nj->nikc", Sq, nf)          # (n_f, 3, K, 3)
        Gm = aho.displacement_matrix_ho(obs, tv, eps, MU, NU, p)
        Tm = aho.traction_matrix_ho(obs, nf, tv, eps, MU, NU, p)
        Gr = Gm.reshape(len(obs), 3, K, 3).transpose(0, 2, 1, 3)   # [n,k,i,j]
        Tr = np.asarray(Tm).reshape(len(obs), 3, K, 3)             # [n,i,k,c]
        wg, wg_at = per_obs_rel(Gr, Gq, where=True)   # entrywise: every node,
        wt, wt_at = per_obs_rel(Tr, Tq, where=True)   # component, observer
        record(f"P{p} displacement, entrywise over all {K} nodes", wg <= 1e-9,
               f"{wg:.3e}", "<= 1e-9", wg_at)
        record(f"P{p} traction,     entrywise over all {K} nodes", wt <= 1e-9,
               f"{wt:.3e}", "<= 1e-9", wt_at)
    print("        (the residual grows with D/L, not with proximity: clq's"
          " closed form loses\n         digits to cancellation with distance"
          " -- its own README says ~2e-4 at 100 L.\n         far_field='hybrid'"
          " switches to quadrature at D_STAR = 10 L, which is why the\n"
          "         15 L observer is CLEANER than the 6 L one.)")
    # tripwire: reversing the node order must break P2 (pins the ORDER, not just
    # the magnitudes -- the partition-of-unity check in [2] cannot see this)
    K = 6
    Gq, _ = quad_influence(obs, tri, 2, MU, NU, eps, 48)
    Gm = aho.displacement_matrix_ho(obs, tv, eps, MU, NU, 2).reshape(len(obs), 3, K, 3)
    perm = [0, 1, 2, 4, 5, 3]                              # rotate the midpoints
    d = relmax(Gm.transpose(0, 2, 1, 3), Gq[:, perm])
    record("tripwire: a rotated P2 midpoint order differs (> 1e-2)", d > 1e-2,
           f"{d:.2e}", "> 1e-2")


# ==========================================================================
# 4. reciprocity U[i,j] = -n_m S[j,m,i], read off the assembled matrix
# ==========================================================================
def check_reciprocity():
    print("\n[4] RECIPROCITY  U[i,j] == -n_m S[j,m,i] on the assembled traction block")
    tv = small_mesh(3)
    rng = np.random.default_rng(23)
    obs = rng.uniform(-14, 14, (9, 3))
    eps = 0.6
    worst, worst_sign = 0.0, np.inf
    for s in range(len(tv)):
        tri = tv[s]
        nh = clq.unit_normal(tri)
        nf = np.tile(nh, (len(obs), 1))
        for p in ORDERS:
            K = aho.n_nodes(p)
            B = np.asarray(aho.traction_matrix_ho(obs, nf, tri[None], eps,
                                                  MU, NU, p))
            # B[3n+j, 3k+i] = n_m S[n,k,j,m,i]; the identity wants U = -that
            rhs = -B.reshape(len(obs), 3, K, 3).transpose(0, 2, 3, 1)  # [n,k,i,j]
            U = clq.influence(obs, tri, MU, NU, eps, order=p, want=("U",)).U
            worst = max(worst, relmax(rhs, U))
            worst_sign = min(worst_sign, relmax(-rhs, U))
    record("U == -n_m S[j,m,i] from traction_matrix_ho, P0/P1/P2",
           worst <= 1e-12, f"{worst:.3e}", "<= 1e-12")
    record("tripwire: the sign matters", worst_sign > 1e-2,
           f"{worst_sign:.2e}", "> 1e-2")


# ==========================================================================
# 5. the parallel path
# ==========================================================================
def check_parallel():
    print("\n[5] PARALLEL PATH  workers > 1 must be bitwise identical")
    tv = small_mesh(9)
    rng = np.random.default_rng(31)
    obs = rng.uniform(-14, 14, (12, 3))
    nf = rng.normal(size=(len(obs), 3))
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    ea = rng.uniform(0.3, 1.0, len(tv))
    ok = True
    try:
        for p in (0, 2):
            a = np.asarray(aho.traction_matrix_ho(obs, nf, tv, ea, MU, NU, p))
            b = np.asarray(aho.traction_matrix_ho(obs, nf, tv, ea, MU, NU, p,
                                                  workers=4))
            ok &= bool(np.array_equal(a, b))
        note = ""
    except Exception as exc:                                  # noqa: BLE001
        ok, note = False, f"raised {type(exc).__name__}: {exc}"
    record("workers=4 == workers=1, bitwise", ok, "equal" if ok else "differ",
           "exact", note)


# ==========================================================================
# timings
# ==========================================================================
def timings(quick):
    print("\n[T] TIMINGS  dense traction matrix, box mesh, field = centroids,")
    print("    eps = 0.3 h, SERIAL (workers=1).  Cost is N_tri clq calls.")
    rows = []
    sizes = [5.0] if quick else [5.0, 2.5]
    for h in sizes:
        tv = box_mesh(h)
        cen = tv.mean(axis=1)
        nrm = normals_of(tv)
        n_s = len(tv)
        for p in ORDERS:
            K = aho.n_nodes(p)
            ncol = 3 * K * n_s
            t0 = time.perf_counter()
            A = aho.traction_matrix_ho(cen, nrm, tv, 0.3 * h, MU, NU, p)
            dt = time.perf_counter() - t0
            gb = A.nbytes / 2 ** 30
            del A
            rows.append((n_s, p, ncol, dt, gb))
            print(f"    N_tri={n_s:5d}  P{p}  unknowns={ncol:6d}  "
                  f"{dt:8.2f} s  ({dt/n_s*1e3:6.1f} ms/source)  {gb:5.2f} GB",
                  flush=True)
    return rows


# ==========================================================================
def main():
    quick = "--quick" in sys.argv[1:]
    t0 = time.time()
    print("=" * 74)
    print("GATE ho_assembly -- P0/P1/P2 nodal force-density assembler "
          "(fbem/assembly_ho.py)")
    print("=" * 74)
    check_p0_reduction()
    check_constant_density()
    check_quadrature()
    check_reciprocity()
    check_parallel()
    rows = timings(quick)

    print("\n" + "-" * 74)
    bad = [r for r in RESULTS if not r[1]]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed "
          f"in {time.time() - t0:.1f}s")
    for name, _, value, tol, _ in bad:
        print(f"    FAILED  {name}: {value} (needed {tol})")
    print("\n    timing summary (seconds, serial):")
    for n_s, p, ncol, dt, gb in rows:
        print(f"      N_tri={n_s:5d}  order={p}  unknowns={ncol:6d}  "
              f"{dt:8.2f} s   matrix {gb:.2f} GB")
    if quick:
        print("      (--quick: the N_tri = 2048 row was skipped)")
    print("-" * 74)
    print("PASS ho_assembly" if not bad else "FAIL ho_assembly")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
