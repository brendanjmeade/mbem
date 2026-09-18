"""Gate for ddbem's conventions: column layout, Fortran ordering, node
geometry, the eps/h budget, and the discontinuous -> continuous scatter.

None of this is kernel mathematics; all of it is the part a caller has to be
able to trust without reading the source.

  (a) COLUMN LAYOUT.  A one-hot slip at column ``3 * (K s + k) + j`` must
      reproduce ``clq.displacement`` / ``clq.stress`` for that single element
      with that single nodal slip.  This pins the documented column index
      against clq's own contracted API, independently of how the block is
      transposed into the matrix.
  (b) FORTRAN ORDERING.  Every returned matrix is F-contiguous, so
      ``scipy.linalg.lu_factor(overwrite_a=True)`` factors a square collocation
      system in place.  Also checked: ``lu_factor(overwrite_a=True)`` really
      does overwrite (the array changes), which is the only observable
      consequence.
  (c) NODE GEOMETRY.  ``element_nodes`` matches ``clq.nodes`` per element, and
      ``node_clearance`` matches the closed form ``min_i lam_i a_i`` computed
      from the altitudes -- P1 vertices and P2 midpoints have clearance exactly
      0, which is the geometric statement behind fbem/FINDINGS.md sec.2.
  (d) eps HANDLING.  A scalar eps and the equivalent (N_tri,) array give
      bitwise identical matrices; a genuinely graded eps array matches
      per-element assembly.
  (e) SCATTER.  On a closed subdivided icosahedron: continuous P1/P2 node
      counts follow Euler (V ~ T/2, E ~ 3T/2), so continuous P1 costs ~1.5
      slip unknowns per triangle against P0's 3 and discontinuous P1's 9 --
      the claim the layout docstring makes.  Shared nodes must be
      GEOMETRICALLY coincident (this is what would catch a P2 edge-node
      numbering that disagreed with clq's 12/23/31 order), and
      ``condense(A) @ x_global`` must equal ``A @ gather(x_global)``.
  (f) INTERNAL CONSISTENCY.  ``traction_matrix`` equals ``stress_matrix``
      contracted with the field normal, for both the elastic and the total
      variant -- two different code paths (contract-then-Voigt vs
      Voigt-then-contract).
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.linalg as sla

from _common import MU, Report, relmax, test_triangles, test_observers

import ddbem
from ddbem._clq import CLQ as clq

EPS = 0.15
NU = 0.30


# ---------------------------------------------------------------------------
def part_columns(rep):
    tri = test_triangles()[:3]
    obs = test_observers()
    n_s = tri.shape[0]
    for p in (0, 1, 2):
        K = ddbem.n_nodes(p)
        A = ddbem.displacement_matrix(obs, tri, EPS, MU, NU, p)
        S = ddbem.stress_matrix(obs, tri, EPS, MU, NU, p)
        worst_u = worst_s = 0.0
        for s in range(n_s):
            for k in range(K):
                for j in range(3):
                    slip = np.zeros((K, 3))
                    slip[k, j] = 1.0
                    col = 3 * (K * s + k) + j
                    u_ref = clq.displacement(obs, tri[s], slip, MU, NU, EPS, order=p)
                    s_ref = clq.stress(obs, tri[s], slip, MU, NU, EPS, order=p)
                    worst_u = max(worst_u, relmax(A[:, col].reshape(-1, 3), u_ref))
                    worst_s = max(worst_s, relmax(
                        ddbem.voigt_to_tensor(S[:, col].reshape(-1, 6)), s_ref))
        rep.check(f"P{p}: column 3*(K*s+k)+j == clq.displacement of that node",
                  worst_u, 1e-13)
        rep.check(f"P{p}: column 3*(K*s+k)+j == clq.stress (elastic) of that node",
                  worst_s, 1e-13)


def part_ordering(rep):
    tri = test_triangles()[:4]
    obs = ddbem.collocation_points(tri, 0)
    nf = ddbem.element_normals(tri)
    A = ddbem.displacement_matrix(obs, tri, EPS, MU, NU, 0)
    T = ddbem.traction_matrix(obs, nf, tri, EPS, MU, NU, 0)
    S = ddbem.stress_matrix(obs, tri, EPS, MU, NU, 0)
    for name, M in (("displacement", A), ("traction", T), ("stress", S)):
        rep.check_bool(f"{name}_matrix is F-contiguous", M.flags["F_CONTIGUOUS"])
    rep.check_bool("traction_matrix is square at node collocation",
                   T.shape[0] == T.shape[1], f"{T.shape}")
    before = T.copy()
    sla.lu_factor(T, overwrite_a=True)
    rep.check_bool("lu_factor(overwrite_a=True) factors it in place",
                   not np.array_equal(before, T))


def part_nodes(rep):
    tri = test_triangles()
    for p in (0, 1, 2):
        got = ddbem.element_nodes(tri, p)
        ref = np.array([clq.nodes(t, p) for t in tri])
        rep.check(f"P{p}: element_nodes == clq.nodes per element", relmax(got, ref), 1e-15)
    # node clearance against the altitude formula
    area = ddbem.element_areas(tri)
    opp = np.stack([np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                    np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
                    np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)], axis=1)
    alt = 2.0 * area[:, None] / opp
    rep.check("P0: clearance == min altitude / 3",
              relmax(ddbem.node_clearance(tri, 0)[:, 0], alt.min(axis=1) / 3.0), 1e-14)
    for p in (1, 2):
        rep.check_bool(f"P{p}: every node sits ON the element boundary "
                       f"(clearance 0)",
                       float(np.abs(ddbem.node_clearance(tri, p)).max()) < 1e-15,
                       f"(max {np.abs(ddbem.node_clearance(tri, p)).max():.1e})")
        c = ddbem.node_clearance(tri, p, shrink=0.25)
        rep.check_bool(f"P{p}: shrink=0.25 lifts every node off the boundary",
                       float(c.min()) > 0.0, f"(min {c.min():.3f})")
    # the eps budget the fbem study turned on
    a = 1.0
    eq = np.array([[[0.0, 0.0, 0.0], [a, 0.0, 0.0], [a / 2, a * np.sqrt(3) / 2, 0.0]]])
    b = ddbem.eps_report(eq, 0.3 * a, 0)
    rep.check("equilateral P0 at eps=0.3h: clearance/eps == 1/(0.3*2*sqrt(3))",
              abs(b.clearance_over_eps_min - 1.0 / (0.3 * 2 * np.sqrt(3))), 1e-12)
    rep.note(f"P0 {b.clearance_over_eps_min:.3f} eps of clearance at eps = 0.3 h; "
             f"P1/P2 have 0 -- fbem/FINDINGS.md sec.2")


def part_eps(rep):
    tri = test_triangles()
    obs = test_observers()
    for p in (0, 1, 2):
        A_s = ddbem.displacement_matrix(obs, tri, EPS, MU, NU, p)
        A_a = ddbem.displacement_matrix(obs, tri, np.full(tri.shape[0], EPS), MU, NU, p)
        rep.check_bool(f"P{p}: scalar eps == constant eps array (bitwise)",
                       np.array_equal(A_s, A_a))
    graded = ddbem.eps_auto(tri, 0.25)
    A_g = ddbem.displacement_matrix(obs, tri, graded, MU, NU, 1)
    K = ddbem.n_nodes(1)
    ref = np.zeros_like(A_g)
    for s in range(tri.shape[0]):
        ref[:, 3 * K * s:3 * K * (s + 1)] = ddbem.displacement_matrix(
            obs, tri[s][None], graded[s], MU, NU, 1)
    rep.check_bool("P1: graded eps array == per-element assembly (bitwise)",
                   np.array_equal(A_g, ref))
    b = ddbem.eps_report(tri, graded, 1)
    rep.note(str(b).replace("\n", "\n       .  "))


# ---------------------------------------------------------------------------
def icosphere(n_sub=2):
    """Closed triangulated sphere: (vertices, triangles).  Euler: V - E + T = 2."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(n_sub):
        mid = {}
        new_f = []
        v = list(v)

        def m(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = (v[a] + v[b]) / 2.0
                v.append(p / np.linalg.norm(p))
                mid[key] = len(v) - 1
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(v)
        f = np.array(new_f)
    return v, f


def part_scatter(rep):
    V, F = icosphere(2)
    tri = V[F]
    n_tri = F.shape[0]
    n_edge = len(set(tuple(sorted(e)) for f in F
                     for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))))
    rep.check_bool("icosphere is closed (Euler V - E + T = 2)",
                   V.shape[0] - n_edge + n_tri == 2,
                   f"(V {V.shape[0]}, E {n_edge}, T {n_tri})")

    lay1 = ddbem.continuous(F, 1)
    lay2 = ddbem.continuous(F, 2)
    per_tri_p0 = 3.0
    per_tri_c1 = lay1.n_dof / n_tri
    per_tri_c2 = lay2.n_dof / n_tri
    rep.check_bool("continuous P1 costs < 1.6 slip unknowns per triangle (P0: 3)",
                   per_tri_c1 < 1.6, f"({per_tri_c1:.3f} vs {per_tri_p0:.1f}; "
                                     f"discontinuous P1 is 9)")
    rep.check_bool("continuous P2 costs < 6.2 slip unknowns per triangle",
                   per_tri_c2 < 6.2, f"({per_tri_c2:.3f}; discontinuous P2 is 18)")

    # shared nodes must be geometrically coincident (pins the P2 edge numbering
    # against clq's 12 / 23 / 31 node order)
    for lay, p in ((lay1, 1), (lay2, 2)):
        pts = ddbem.element_nodes(tri, p).reshape(-1, 3)
        idx = lay.scatter.reshape(-1)
        acc = np.zeros((lay.n_global, 3))
        cnt = np.zeros(lay.n_global)
        np.add.at(acc, idx, pts)
        np.add.at(cnt, idx, 1.0)
        mean = acc / cnt[:, None]
        rep.check(f"P{p}: nodes sharing a global index are coincident",
                  float(np.abs(pts - mean[idx]).max()), 1e-14)

    # condense(A) @ x_global == A @ gather(x_global)
    obs = 1.7 * V[:20]
    rng = np.random.default_rng(3)
    for lay, p in ((ddbem.discontinuous(n_tri, 1), 1), (lay1, 1), (lay2, 2)):
        A = ddbem.displacement_matrix(obs, tri, 0.25, MU, NU, p)
        Ac = ddbem.displacement_matrix(obs, tri, 0.25, MU, NU, p, layout=lay)
        x = rng.normal(size=lay.n_dof)
        gathered = x.reshape(lay.n_global, 3)[lay.scatter].reshape(-1)
        tag = "continuous" if lay.continuous else "discontinuous"
        rep.check(f"P{p} {tag}: condense(A) @ x == A @ gather(x)",
                  relmax(Ac @ x, A @ gathered), 1e-13)
        rep.check_bool(f"P{p} {tag}: condensed matrix is F-contiguous",
                       Ac.flags["F_CONTIGUOUS"], f"shape {Ac.shape}")


def part_consistency(rep):
    tri = test_triangles()
    obs = test_observers()
    rng = np.random.default_rng(11)
    nf = rng.normal(size=(obs.shape[0], 3))
    nf /= np.linalg.norm(nf, axis=1, keepdims=True)
    for p in (0, 1, 2):
        for sub in (True, False):
            S = ddbem.stress_matrix(obs, tri, EPS, MU, NU, p, subtract_eigenstress=sub)
            T = ddbem.traction_matrix(obs, nf, tri, EPS, MU, NU, p,
                                      subtract_eigenstress=sub)
            sig = ddbem.voigt_to_tensor(S.reshape(obs.shape[0], 6, -1).transpose(0, 2, 1))
            ref = np.einsum("ncml,nl->ncm", sig, nf).transpose(0, 2, 1).reshape(T.shape)
            tag = "elastic" if sub else "total"
            rep.check(f"P{p}: traction_matrix == stress_matrix . n ({tag})",
                      relmax(T, ref), 1e-13)
    # Voigt round trip
    a = rng.normal(size=(5, 3, 3))
    a = 0.5 * (a + a.transpose(0, 2, 1))
    rep.check("voigt round trip", relmax(ddbem.voigt_to_tensor(
        ddbem.tensor_to_voigt(a)), a), 1e-15)


def main():
    rep = Report("ddbem conventions: columns, ordering, nodes, eps, scatter")
    part_columns(rep)
    part_ordering(rep)
    part_nodes(rep)
    part_eps(rep)
    part_scatter(rep)
    part_consistency(rep)
    sys.exit(0 if rep.finish() else 1)


if __name__ == "__main__":
    main()
