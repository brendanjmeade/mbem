"""Verify P0/P1/P2 (Lagrange nodal) patches and faults end to end through the
model layer and the dense backend: layout, collocation, free term, assembly,
solve, evaluation, and the refusals of the compressed path.

  [i]   Plumbing. P0 is bitwise the P0 kernels: A and b of the level-1
        icosphere (Dirichlet half/calibrated, Neumann calibrated) and of
        verify_solved_bvp's anchor-C closed box with a fault (half/calibrated)
        equal a hand-written centroid assembly on tri_kernels, np.array_equal.
        Shape functions: the Lagrange node lattice equals tri_nodal's column
        lattice, lagrange_shape is bitwise tri_nodal._lagrange_values at
        random barycentric points and the identity at the nodes (the lattice
        alone does not pin a permuted shape function), the collocation shape
        rows sum to one, slots are 3 K N_tri. Nodal values at P1/P2: an
        (N_tri, 3) value fills every node of its element (== value[:, None, :]
        bitwise; a node-major tiling is otherwise invisible), a callable
        value(points, normals) sees the element normal at every node, and
        collocation_values reproduces a linear field at the collocation
        points to 1e-14 (measured 1.4e-16 / 3.7e-16; un-interpolated nodal
        values are otherwise absorbed by [iv]'s ratios). P1/P2 assembly on
        the level-1 icosphere: the Neumann b from an (N_tri, 3) value is
        bitwise the b from np.repeat(value, K), and the Dirichlet calibrated
        A, b equal a hand assembly on the nodal kernels at the collocation
        points, A = -sigma G and b = -sigma H u_bar - C_c (N u_bar)_c with
        the backend's own calib and N = lagrange_shape at the shrunk nodes
        (bitwise measured, gated at 1e-14).
        The check count printed at the end is this in-tree pin; with
        MBEM_P0_ORACLE=<npz> set, A, b and the solution of the five P0 cases
        are also compared to that file (the pre-change tree's numbers, kept
        outside the tree), 5 more checks.
  [ii]  Patch test, u = c on the closed icosphere, calibrated, P0/P1/P2,
        eps/h = 0.15 and 0.3: the assembled RHS annihilates the constant,
        max|b| / |c| < 1e-14 (measured 3e-16 .. 7e-16), and the solved
        traction max|t| / (mu |c|) < 1e-11 -- that residual is the RHS
        roundoff amplified by cond(A) of the first-kind Dirichlet system,
        1.2e2 / 1.9e3 / 2.6e4 at P0 / P1 / P2 (eps/h = 0.3), measured
        9.7e-15 / 2.5e-13 / 1.8e-12.  Tripwire: jump="half" is NOT exact
        (> 1e-3; measured 0.10 / 0.95 / 10.3).
  [iii] Rigid-motion covariance: the P1 Neumann solve of a rigidly rotated and
        translated problem is the rotated solution to 1e-10 (nodal u and
        interior u).
  [iv]  Exterior Kelvin point force, levels 1 -> 2, eps/h = 0.15 and 0.3,
        Dirichlet and Neumann at P1 and P2 (calibrated; Neumann deflated and
        rigid-stripped): the interior displacement error at r = 0.5 falls
        with h, ratio > 1.3 (measured 1.44 .. 3.23), and on the Neumann
        problem at eps/h = 0.15, level 1, the P1 and P2 errors are < 0.8 of
        P0's at the same mesh (measured 0.66 / 0.48); the level-2 ratios
        (0.75 / 0.66, within 6 % of the bar) are printed, not gated.
        Dirichlet P1 / P2 are NOT gated against P0: msd's Dirichlet unknown
        is the traction of a first-kind single-layer system, and the
        discontinuous nodal traction is poorly controlled by the mollified
        kernel (P1 interior error above P0's at every eps/h, its recovered
        traction 0.3 .. 1.3 off); it converges in h, which is what is gated.
  [v]   A P1 fault with a linear slip field (callable value) in the anchor-C
        box (free top and sides, clamped base) against the same field on a
        16x-refined P0 fault (each fault at its own "auto" eps): interior u
        on anchor C's cloud and the surface u agree to 1e-3 (measured
        3.5e-4 / 4.7e-4; the coarse P0 fault gives 7.6e-3 / 9.3e-3), and
        P1 vs P2 agree to 1e-10 (measured 6e-15 / 9e-15).
  [vi]  Layout tripwires: permuting the node order of a P2 patch's value
        changes the solution (> 1e-2); the identity free term instead of
        N_k(x_c) changes a P1 Neumann solution (> 1e-3).
  [vii] Refusals: HBackend, DisplacementEvaluator, estimate_memory("hmat")
        and mode="legacy" raise on an order > 0 patch; order 3 and order 1.5
        raise ValueError (a wrong exception type is a labelled FAIL, not a
        traceback); the collocation-near-fault guard
        warns on a fault reaching 0.2 km below a 5 km-element top (eps_f =
        0.4) and is silent on the production fault box of
        examples/demo_bem_onfault_stress.py with eps="auto" at a P0 and at
        the production P1 top (its nearest collocation point is 3.1 fault-eps
        from the trace).

Run from the repo root:  python verify/verify_nodal_solve.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import _sphere as S                                                 # noqa: E402
import verify_solved_bvp as VB                                      # noqa: E402
from mbem.cases.fault_box import build_fault_box, build_model                 # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq                # noqa: E402
from mbem import defaults                                           # noqa: E402
from mbem.backends.dense import (AssembledDense, DenseBackend,      # noqa: E402
                                 warn_collocation_near_fault)
from mbem.backends.hmat import HBackend                             # noqa: E402
from mbem.estimate import estimate_memory                           # noqa: E402
from mbem.evaluate import DisplacementEvaluator, evaluate_displacement  # noqa: E402
from mbem.kernels import basis as kb                                # noqa: E402
from mbem.kernels import tri_kernels as tk                          # noqa: E402
from mbem.kernels import tri_nodal as tn                            # noqa: E402
from mbem.model import BCType, Patch, RegionModel, generate_system  # noqa: E402
from mbem.model.equations import COLLOCATION_JUMP                   # noqa: E402

NU = 0.30                      # away from 1/4 (rule 6)
ORDERS = (0, 1, 2)
TOL_PATCH_B = 1e-14            # RHS of the constant field, / |c|
TOL_PATCH_T = 1e-11            # solved traction, / (mu |c|): cond(A) x roundoff
TOL_HALF_NOT_EXACT = 1e-3
TOL_LINEAR = 1e-14             # collocation_values of a linear field; hand-assembled b
TOL_COVARIANCE = 1e-10
RATIO_MIN = 1.3                # level-1 / level-2 interior error
# P1/P2 Neumann error below P0's, gated at level 1 only: measured 0.66 / 0.48
# there, but 0.75 / 0.66 at level 2 -- within 6 % of the bar, where a real
# defect landed at 0.80 -- so level 2 is printed, not gated.
NEUMANN_VS_P0 = 0.8
TOL_FAULT = 1e-3               # P1 fault vs 16x P0 fault, far cloud and surface
TOL_P1_P2 = 1e-10
TOL_PERMUTE = 1e-2
TOL_IDENTITY = 1e-3

CHECKS = []


def check(name, val, tol):
    ok = bool(np.isfinite(val)) and val < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:66s} {val:9.2e} < {tol:.0e}")
    return ok


def check_gt(name, val, floor):
    ok = bool(np.isfinite(val)) and val > floor
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:66s} {val:9.2e} > {floor:.0e}")
    return ok


def check_bool(name, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:66s} {note}")
    return ok


def relmax(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def relnorm(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# [i] P0 plumbing vs a hand-written centroid assembly on tri_kernels
# ---------------------------------------------------------------------------

def p0_reference(model, eps, jump):
    """A, b of a single-region P0 model assembled at the centroids with
    tri_kernels, accumulated in the dense backend's order."""
    region = model.regions[0]
    mat = region.material
    tc, uc = kb.t_coeffs(mat.mu, mat.lam), kb.u_coeffs(mat.mu, mat.lam)
    off, col = 0, {}
    for p in region.patches:
        kind = "u" if p.bc is BCType.FREE_TRACTION else "t"
        col[id(p)] = (kind, off)
        off += 3 * p.n_triangles
    A, b = np.zeros((off, off)), np.zeros(off)

    def H(x, p):
        tv, nrm = kb._source_arrays(p.mesh)
        return tk.t_matrix_direct(x, tv, nrm, kb.resolve_patch_eps(eps, p), *tc)

    def G(x, p):
        tv, _ = kb._source_arrays(p.mesh)
        return tk.u_matrix_direct(x, tv, kb.resolve_patch_eps(eps, p), *uc)

    for q in region.patches:
        x = np.ascontiguousarray(q.mesh.centroids())
        kind, r0 = col[id(q)]
        r = slice(r0, r0 + 3 * q.n_triangles)
        D = None
        for p in region.patches:
            sigma = float(model.orientation(region, p))
            Hqp = H(x, p)
            if jump == "calibrated":
                dq = -sigma * Hqp.reshape(q.n_triangles, 3, p.n_triangles, 3).sum(axis=2)
                D = dq if D is None else D + dq
            ck, c0 = col[id(p)]
            c = slice(c0, c0 + 3 * p.n_triangles)
            if ck == "u":
                A[r, c] += sigma * Hqp
                if p is q:
                    Dq = D if jump == "calibrated" else None
                    Dq = np.broadcast_to(COLLOCATION_JUMP * np.eye(3),
                                         (q.n_triangles, 3, 3)) if Dq is None else Dq
                    idx = 3 * np.arange(q.n_triangles)
                    for a_ in range(3):
                        for b_ in range(3):
                            A[r0 + idx + a_, c0 + idx + b_] += 1.0 * Dq[:, a_, b_]
            else:
                u_bar = p.value_array().ravel()
                if np.any(u_bar):
                    b[r] += -sigma * (Hqp @ u_bar)
            if ck == "t":
                A[r, c] += -sigma * G(x, p)
            else:
                t_bar = p.value_array().ravel()
                if np.any(t_bar):
                    b[r] += sigma * (G(x, p) @ t_bar)
        for f in region.faults:
            sigma = float(model.orientation(region, f))
            slip = f.value_array().ravel()
            if np.any(slip):
                b[r] += -sigma * (H(x, f) @ slip)
        if kind == "t":                          # prescribed u on q's own row
            u_bar = q.value_array().ravel()
            if np.any(u_bar):
                if jump == "half":
                    b[r] += -COLLOCATION_JUMP * u_bar
                else:
                    b[r] -= np.einsum("nij,nj->ni", D, u_bar.reshape(-1, 3)).ravel()
    return A, b


def box_c_model():
    """verify_solved_bvp anchor C: the full-space field of an interior fault
    prescribed on the closed box."""
    box = VB._closed_box(VB.EDGE_C)
    fault, _, s_hat = make_vertical_fault_eq(**VB.FAULT_C)
    slip = np.broadcast_to(VB.SLIP * s_hat, (fault.n_triangles, 3))
    keys = ("top", "sides", "base")
    zero = {f"t:{k}": np.zeros((box[k].n_triangles, 3)) for k in keys}
    mat = VB.material(NU)
    ref, r_ref = VB._build_box_model(
        box, mat, dict(top="d", sides="d", base="d"), {k: None for k in keys},
        faults=[Patch("fault", fault, BCType.FAULT, value=slip)])
    u_ref = {k: evaluate_displacement(ref, r_ref, zero, box[k].centroids(),
                                      VB.EPS_C, warn_near=False) for k in keys}
    model, _ = VB._build_box_model(
        box, mat, dict(top="d", sides="d", base="d"), u_ref,
        faults=[Patch("fault", fault, BCType.FAULT, value=slip)])
    return model


def i_p0_plumbing():
    print("\n[i] P0 plumbing == hand-written centroid assembly on tri_kernels, bitwise")
    rng = np.random.default_rng(11)
    lam_r = rng.random((7, 3))
    lam_r /= lam_r.sum(axis=1, keepdims=True)                 # random barycentric
    for p in ORDERS:
        K = kb.n_nodes(p)
        lam = kb.lagrange_nodes(p)
        same = p == 0 or np.array_equal(lam, tn._lattice(p) / p)
        check_bool(f"P{p}: Lagrange node lattice == tri_nodal column lattice", same)
        check_bool(f"P{p}: lagrange_shape == tri_nodal._lagrange_values at random lam, bitwise",
                   np.array_equal(kb.lagrange_shape(p, lam_r), tn._lagrange_values(p, lam_r)))
        check_bool(f"P{p}: lagrange_shape at the nodes == I (nodes and columns agree)",
                   np.array_equal(kb.lagrange_shape(p, lam), np.eye(K)))
        N = kb.lagrange_shape(p, kb.lagrange_nodes(p, defaults.COLLOCATION_SHRINK_BY_ORDER[p]))
        check(f"P{p}: collocation shape rows sum to 1 (partition of unity)",
              float(np.max(np.abs(N.sum(axis=1) - 1.0))), 1e-15)
    mesh = S.icosphere(0)
    for p in ORDERS:
        pt = Patch("s", mesh, BCType.FREE_TRACTION, order=p)
        sysm = generate_system(RegionModel([__import__("mbem.model", fromlist=["Region"]).Region(
            "b", S.material(NU), [pt], np.zeros(3))]))
        check_bool(f"P{p}: slot size == 3 K N_tri, rows == collocation points",
                   sysm.layout.slots[0].size == 3 * kb.n_nodes(p) * mesh.n_triangles
                   and pt.collocation_points().shape == (kb.n_nodes(p) * mesh.n_triangles, 3))

    # Nodal values: an (N_tri, 3) value fills every node of its element
    # (element-major, not node-major), a two-argument callable sees the
    # element normal at every node, and collocation_values interpolates
    # (a linear field is reproduced at the shrunk collocation points).
    nrm, _ = mesh.normals_and_areas()
    per_el = rng.standard_normal((mesh.n_triangles, 3))
    A_lin, c_lin = rng.standard_normal((3, 3)), rng.standard_normal(3)

    def f_lin(x):
        return x @ A_lin.T + c_lin

    for p in (1, 2):
        K = kb.n_nodes(p)
        pt = Patch("s", mesh, BCType.FREE_TRACTION, value=per_el, order=p)
        check_bool(f"P{p}: value_array of an (N_tri, 3) value == value[:, None, :], bitwise",
                   np.array_equal(pt.value_array().reshape(mesh.n_triangles, K, 3),
                                  np.broadcast_to(per_el[:, None, :], (mesh.n_triangles, K, 3))))
        pt = Patch("s", mesh, BCType.FREE_TRACTION, value=lambda x, n: n, order=p)
        check_bool(f"P{p}: callable value(points, normals) gets the element normal at every node",
                   np.array_equal(pt.value_array(), np.repeat(nrm, K, axis=0)))
        check(f"P{p}: collocation_values(f(nodes)) == f(collocation points), f linear",
              relmax(pt.collocation_values(f_lin(pt.nodes())), f_lin(pt.collocation_points())),
              TOL_LINEAR)

    oracle = os.environ.get("MBEM_P0_ORACLE")
    orc = np.load(oracle) if oracle else None
    cases = []
    for bc, jump in (("dirichlet", "half"), ("dirichlet", "calibrated"),
                     ("neumann", "calibrated")):
        m, _r, sol, es, _h, asm = quiet(S.sphere_model, 1, bc, NU, eps_over_h=0.3,
                                        jump=jump, return_asm=True)
        cases.append((f"sphere_{bc}_{jump}", m, es, jump, asm, sol))
    box = box_c_model()
    system = generate_system(box)
    for jump in ("half", "calibrated"):
        asm = AssembledDense(system, VB.EPS_C, "direct", jump=jump)
        cases.append((f"boxC_{jump}", box, VB.EPS_C, jump, asm, asm.solve()))
    for name, model, eps, jump, asm, sol in cases:
        A, b = p0_reference(model, eps, jump)
        check_bool(f"{name}: A bitwise == tri_kernels reference", np.array_equal(asm.A, A),
                   f"shape {A.shape}")
        check_bool(f"{name}: b bitwise == tri_kernels reference", np.array_equal(asm.b, b))
        if orc is not None:
            x = np.concatenate([sol[k].ravel() for k in
                                (sorted(sol) if name.startswith("boxC") else sol)])
            check_bool(f"{name}: A, b, x bitwise == MBEM_P0_ORACLE",
                       np.array_equal(asm.A, orc[name + "_A"])
                       and np.array_equal(asm.b, orc[name + "_b"])
                       and np.array_equal(x, orc[name + "_x"]))
    if orc is None:
        print("      (MBEM_P0_ORACLE not set: pre-change oracle file not compared)")

    # P1/P2 assembly: the Neumann RHS from an (N_tri, 3) value is the RHS
    # from its node-repeated (K N_tri, 3) form, and the Dirichlet A, b equal
    # a hand assembly on the nodal kernels at the collocation points,
    # b = -sigma H u_bar - C_c (N u_bar)_c with C the backend's own calib
    # and N = lagrange_shape at the shrunk nodes (pinned above).
    mesh = S.icosphere(1)
    nrm, _ = mesh.normals_and_areas()
    t_c = np.einsum("nij,nj->ni", S.kelvin_sigma(mesh.centroids(), NU), nrm)
    for p in (1, 2):
        K = kb.n_nodes(p)
        asm_el = quiet(S.sphere_model, 1, "neumann", NU, eps_over_h=0.3, value=t_c,
                       order=p, return_asm=True)[-1]
        asm_nd = quiet(S.sphere_model, 1, "neumann", NU, eps_over_h=0.3,
                       value=np.repeat(t_c, K, axis=0), order=p, return_asm=True)[-1]
        check_bool(f"P{p} Neumann: b from an (N_tri, 3) value == b from np.repeat(value, K), bitwise",
                   np.array_equal(asm_el.b, asm_nd.b))
        m, r, _s, es, _h, asm = quiet(S.sphere_model, 1, "dirichlet", NU, eps_over_h=0.3,
                                      jump="calibrated", order=p, return_asm=True)
        q = r.patches[0]
        sigma = float(m.orientation(r, q))
        x_c = np.ascontiguousarray(q.collocation_points())
        eps_arr = kb.resolve_patch_eps(es, q)
        H = kb.assemble_t_matrix(x_c, mesh, r.material, eps_arr, order=p)
        G = kb.assemble_u_matrix(x_c, mesh, r.material, eps_arr, order=p)
        u_bar = q.value_array()
        N = kb.lagrange_shape(p, kb.lagrange_nodes(p, defaults.COLLOCATION_SHRINK_BY_ORDER[p]))
        u_c = np.einsum("ck,skj->scj", N, u_bar.reshape(mesh.n_triangles, K, 3)).reshape(-1, 3)
        b = np.zeros(asm.b.size)
        b += -sigma * (H @ u_bar.ravel())
        b -= np.einsum("nij,nj->ni", asm.calib[(id(r), id(q))], u_c).ravel()
        check_bool(f"P{p} Dirichlet calibrated: A == -sigma G (nodal U kernel), bitwise",
                   np.array_equal(asm.A, -sigma * G))
        check(f"P{p} Dirichlet calibrated: b == -sigma H u_bar - C_c (N u_bar)_c, hand-assembled",
              relmax(asm.b, b), TOL_LINEAR)


# ---------------------------------------------------------------------------
# [ii] patch test
# ---------------------------------------------------------------------------

def ii_patch_test():
    print("\n[ii] patch test u = c on the closed icosphere (level 1)")
    c = np.array([0.003, -0.002, 0.001])
    scale = S.MU * np.linalg.norm(c)
    for eh in (0.15, 0.3):
        for p in ORDERS:
            _m, _r, sol, _e, _h, asm = quiet(S.sphere_model, 1, "dirichlet", NU, eps_over_h=eh,
                                             value=c, jump="calibrated", order=p, return_asm=True)
            check(f"P{p} eps/h={eh}: calibrated RHS annihilates the constant, max|b|/|c|",
                  float(np.abs(asm.b).max() / np.linalg.norm(c)), TOL_PATCH_B)
            check(f"P{p} eps/h={eh}: calibrated max|t|/(mu|c|) (cond {asm.report.cond_estimate:.1e})",
                  float(np.abs(sol['t:sphere']).max() / scale), TOL_PATCH_T)
            if eh == 0.3:
                _m, _r, sol, _e, _h = quiet(S.sphere_model, 1, "dirichlet", NU, eps_over_h=eh,
                                            value=c, jump="half", order=p)
                check_gt(f"P{p} eps/h={eh}: jump='half' is NOT exact (tripwire), max|t|/(mu|c|)",
                         float(np.abs(sol['t:sphere']).max() / scale), TOL_HALF_NOT_EXACT)


# ---------------------------------------------------------------------------
# [iii] rigid-motion covariance of a P1 Neumann solve
# ---------------------------------------------------------------------------

def iii_covariance():
    print("\n[iii] rigid-motion covariance, P1 Neumann, level 1, eps/h = 0.15")
    rng = np.random.default_rng(7)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    d = np.array([0.4, -1.1, 0.7])
    P = S.shell(0.5)

    def solve(mesh, x0, force, nodes_of):
        patch = Patch("sphere", mesh, BCType.FREE_TRACTION, order=1)
        nrm, _ = mesh.normals_and_areas()
        nd = patch.nodes()
        r = nd - x0
        Rn = np.linalg.norm(r, axis=1)
        n = r / Rn[:, None]
        nF = n @ force
        eye = np.eye(3)
        cc = -1.0 / (8.0 * np.pi * (1.0 - NU) * Rn ** 2)
        sig = cc[:, None, None] * (
            (1.0 - 2.0 * NU) * (np.einsum("i,nj->nij", force, n) + np.einsum("j,ni->nij", force, n)
                                - nF[:, None, None] * eye[None])
            + 3.0 * nF[:, None, None] * np.einsum("ni,nj->nij", n, n))
        patch.value = np.einsum("nij,nj->ni", sig, np.repeat(nrm, 3, axis=0))
        region = __import__("mbem.model", fromlist=["Region"]).Region(
            "body", S.material(NU), [patch], nodes_of, faults=[])
        model = RegionModel([region])
        h = float(kb.element_sizes(mesh).mean())
        eps = {"sphere": 0.15 * h}
        sol = DenseBackend(jump="calibrated", deflate=True).assemble(
            generate_system(model), eps).solve()
        return model, region, sol, eps

    mesh = S.icosphere(1)
    m0, r0, s0, e0 = quiet(solve, mesh, S.X0, S.FORCE, np.zeros(3))
    import mollified_bem as mb
    mesh_r = mb.TriMesh(vertices=np.ascontiguousarray(mesh.vertices @ Q.T + d),
                        triangles=mesh.triangles.copy())
    m1, r1, s1, e1 = quiet(solve, mesh_r, Q @ S.X0 + d, Q @ S.FORCE, d)
    check("rotated nodal u == Q u (rigid-stripped both pinned by deflation)",
          relmax(s1["u:sphere"], s0["u:sphere"] @ Q.T), TOL_COVARIANCE)
    u0 = evaluate_displacement(m0, r0, s0, P, e0, warn_near=False)
    u1 = evaluate_displacement(m1, r1, s1, P @ Q.T + d, e1, warn_near=False)
    check("rotated interior u(Q x + d) == Q u(x)", relmax(u1, u0 @ Q.T), TOL_COVARIANCE)


# ---------------------------------------------------------------------------
# [iv] Kelvin ladder at P0 / P1 / P2
# ---------------------------------------------------------------------------

def iv_kelvin_ladder():
    print("\n[iv] exterior Kelvin point force, levels 1 -> 2, interior u at r = 0.5")
    P = S.shell(0.5)
    ue = S.kelvin_u(P, NU)
    for eh in (0.15, 0.3):
        for bc in ("dirichlet", "neumann"):
            err = {}
            for p in ORDERS:
                for level in (1, 2):
                    m, r, sol, es, _h = quiet(S.sphere_model, level, bc, NU, eps_over_h=eh,
                                              jump="calibrated", order=p)
                    du = evaluate_displacement(m, r, sol, P, es, warn_near=False) - ue
                    if bc == "neumann":
                        du = S.best_fit_rigid(du, P)
                    err[(p, level)] = float(np.abs(du).max() / np.abs(ue).max())
                print(f"      eps/h={eh:4.2f} {bc:9s} P{p}: {err[(p, 1)]:.3e} -> {err[(p, 2)]:.3e}")
            for p in (1, 2):
                check_gt(f"eps/h={eh} {bc} P{p}: error falls with h, level1/level2",
                         err[(p, 1)] / err[(p, 2)], RATIO_MIN)
                if bc == "neumann" and eh == 0.15:
                    check(f"eps/h={eh} {bc} P{p} level 1: error / P0 error",
                          err[(p, 1)] / err[(0, 1)], NEUMANN_VS_P0)
                    print(f"      eps/h={eh} {bc} P{p} level 2: error / P0 error = "
                          f"{err[(p, 2)] / err[(0, 2)]:.3f} (diagnostic, not gated)")


# ---------------------------------------------------------------------------
# [v] P1 fault with a linear slip field vs a 16x-refined P0 fault
# ---------------------------------------------------------------------------

def v_p1_fault():
    print("\n[v] P1 fault, linear slip field, anchor-C box (free top/sides, clamped base)")
    mat = VB.material(NU)
    box = VB._closed_box(VB.EDGE_C)
    keys = ("top", "sides", "base")
    fc = VB.FAULT_C
    zc = 0.5 * sum(fc["depth_range"])
    D = fc["depth_range"][1] - fc["depth_range"][0]
    L = fc["strike_length"]

    def slip_field(x):
        x = np.asarray(x, float)
        out = np.zeros((x.shape[0], 3))
        out[:, 1] = VB.SLIP * (1.0 + 0.5 * x[:, 1] / (0.5 * L)
                               + 0.3 * (x[:, 2] - zc) / (0.5 * D))
        return out

    rng = np.random.default_rng(31415)
    obs = np.column_stack([rng.uniform(-34.0, 34.0, 600), rng.uniform(-34.0, 34.0, 600),
                           rng.uniform(-32.0, -8.0, 600)])
    obs = obs[np.max(np.abs(obs[:, :2]), axis=1) > 26.0][:80]
    eps = {"top": VB.EPS_C, "sides": VB.EPS_C, "base": VB.EPS_C, "fault": "auto"}

    def solve(fault_mesh, order):
        fp = Patch("fault", fault_mesh, BCType.FAULT, value=slip_field, order=order)
        model, region = VB._build_box_model(box, mat, dict(top="t", sides="t", base="d"),
                                            {k: None for k in keys}, faults=[fp])
        sol = AssembledDense(generate_system(model), eps, "direct", jump="calibrated").solve()
        return sol["u:top"], evaluate_displacement(model, region, sol, obs, eps, warn_near=False)

    f1, _, _ = make_vertical_fault_eq(**fc)
    f16, _, _ = make_vertical_fault_eq(**dict(fc, target_edge=fc["target_edge"] / 4.0))
    print(f"      fault: {f1.n_triangles} tri (P0/P1/P2) vs {f16.n_triangles} tri P0")
    top16, u16 = solve(f16, 0)
    top0, u0 = solve(f1, 0)
    top1, u1 = solve(f1, 1)
    top2, u2 = solve(f1, 2)
    print(f"      coarse P0 vs refined P0: interior {relnorm(u0, u16):.3e}, surface {relnorm(top0, top16):.3e}")
    check("P1 fault vs 16x-refined P0 fault: interior u (anchor-C cloud)", relnorm(u1, u16), TOL_FAULT)
    check("P1 fault vs 16x-refined P0 fault: surface u:top", relnorm(top1, top16), TOL_FAULT)
    check("P1 vs P2 fault (both exact for a linear field): interior u", relnorm(u1, u2), TOL_P1_P2)
    check("P1 vs P2 fault: surface u:top", relnorm(top1, top2), TOL_P1_P2)


# ---------------------------------------------------------------------------
# [vi] layout tripwires
# ---------------------------------------------------------------------------

def vi_tripwires():
    print("\n[vi] nodal-layout tripwires")
    m, r, sol, es, _h = quiet(S.sphere_model, 1, "dirichlet", NU, eps_over_h=0.15,
                              jump="calibrated", order=2)
    patch = r.patches[0]
    val = patch.value_array().reshape(patch.n_triangles, 6, 3)
    patch.value = np.roll(val, 1, axis=1).reshape(-1, 3)      # node k -> k + 1 in each element
    sol_perm = DenseBackend(jump="calibrated").assemble(generate_system(m), es).solve()
    check_gt("P2: permuting the node order of the value changes the solution",
             relmax(sol_perm["t:sphere"], sol["t:sphere"]), TOL_PERMUTE)

    m, r, sol, es, _h = quiet(S.sphere_model, 2, "neumann", NU, eps_over_h=0.15,
                              jump="calibrated", order=1)
    orig = Patch.collocation_shape
    try:
        Patch.collocation_shape = lambda self: np.eye(kb.n_nodes(self.order))
        sol_id = DenseBackend(jump="calibrated", deflate=True).assemble(
            generate_system(m), es).solve()
    finally:
        Patch.collocation_shape = orig
    check_gt("P1 Neumann: identity free term instead of N_k(x_c) changes the solution",
             relmax(sol_id["u:sphere"], sol["u:sphere"]), TOL_IDENTITY)


# ---------------------------------------------------------------------------
# [vii] refusals and the collocation-near-fault guard
# ---------------------------------------------------------------------------

def vii_refusals():
    print("\n[vii] refusals and the collocation-near-fault guard")
    mesh = S.icosphere(1)
    patch = Patch("sphere", mesh, BCType.FREE_TRACTION, order=1)
    Region = __import__("mbem.model", fromlist=["Region"]).Region
    model = RegionModel([Region("body", S.material(NU), [patch], np.zeros(3))])
    system = generate_system(model)
    eps = 0.3 * float(kb.element_sizes(mesh).mean())

    def raises(fn, exc, needle=""):
        """(ok, note): ok iff ``fn`` raises ``exc`` with ``needle`` in its
        message. A wrong type or no raise is a labelled FAIL, never a
        traceback (a no-op require_order0 surfaces as a ValueError from the
        cluster tree, and the gate must still print FAIL)."""
        try:
            fn()
        except exc as e:
            ok = needle in str(e)
            return ok, "" if ok else f"{type(e).__name__} without {needle!r}: {e}"[:90]
        except Exception as e:                        # noqa: BLE001
            return False, f"raised {type(e).__name__}, not {exc.__name__}: {e}"[:90]
        return False, "did not raise"

    check_bool("HBackend refuses an order-1 patch (NotImplementedError)",
               *raises(lambda: HBackend(jump="calibrated", deflate=True).assemble(system, eps),
                       NotImplementedError, "compressed backend"))
    check_bool("DisplacementEvaluator refuses an order-1 patch (NotImplementedError)",
               *raises(lambda: DisplacementEvaluator(model, "body", S.shell(0.5), eps),
                       NotImplementedError, "compressed backend"))
    check_bool("estimate_memory(mode='hmat') refuses an order-1 patch (NotImplementedError)",
               *raises(lambda: estimate_memory(system, mode="hmat"),
                       NotImplementedError, "compressed backend"))
    check_bool("mode='legacy' refuses an order-1 patch (ValueError)",
               *raises(lambda: AssembledDense(system, eps, "legacy", jump="half"),
                       ValueError, "order-0"))
    check_bool("Patch(order=3) raises ValueError",
               *raises(lambda: Patch("s", mesh, BCType.FREE_TRACTION, order=3), ValueError))
    check_bool("Patch(order=1.5) raises ValueError (not truncated to P1)",
               *raises(lambda: Patch("s", mesh, BCType.FREE_TRACTION, order=1.5), ValueError,
                       "integer"))

    def warns(system, eps):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_collocation_near_fault(system, eps)
        return any("fault-eps" in str(x.message) for x in w)

    box = VB._closed_box(VB.EDGE_C)
    shallow, _, s_hat = make_vertical_fault_eq(
        strike_length=10.0, depth_range=(-10.0, -0.2), target_edge=5.0)
    fp = Patch("fault", shallow, BCType.FAULT, value=VB.SLIP * s_hat)
    near, _ = VB._build_box_model(box, VB.material(NU), dict(top="t", sides="t", base="d"),
                                  {k: None for k in ("top", "sides", "base")}, faults=[fp])
    eps_f = float(kb.resolve_patch_eps("auto", fp)[0])
    check_bool(f"guard warns: fault top 0.2 km below the surface, eps_f = {eps_f:.2f}",
               warns(generate_system(near), "auto"))
    meshes = build_fault_box(half_x=120.0, z_bottom=-80.0, fault_half_len=20.0,
                             fault_depth=20.0, edge_fault=2.0, edge_near=20.0,
                             edge_far=60.0, edge_side=60.0, near_field_radius=60.0)
    for order_top in (0, 1):
        prod = build_model(meshes, 0.01, VB.material(0.25), order_top=order_top)
        check_bool(f"guard silent on the production fault box (demo_bem_onfault_stress, "
                   f"eps='auto', P{order_top} top)", not warns(generate_system(prod), "auto"))


def main():
    print("=" * 76)
    print("P0/P1/P2 nodal patches and faults: model layer + dense backend end to end")
    print("=" * 76)
    i_p0_plumbing()
    ii_patch_test()
    iii_covariance()
    iv_kelvin_ladder()
    v_p1_fault()
    vi_tripwires()
    vii_refusals()
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: nodal P0/P1/P2 solve ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: nodal P0/P1/P2 solve "
              f"({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
