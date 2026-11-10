"""verify_penny_crack.py -- the traction-BC fault and the continuous P2 layout on
the penny-shaped crack, and their coupling to a boundary.

A circular shear crack of radius a in the full space (a region with no
boundary patch), Eshelby's exact slip delta0 sqrt(1 - r^2 / a^2) under the
uniform drop dtau = pi (2 - nu) mu delta0 / (8 (1 - nu) a) (nu 0.30, mu 30,
a 1 km, delta0 1 m), meshed by ``mbem.cases.penny``: the q30 CDT at a/h 8 and
16, a jittered CDT and a hexagonal lattice at 16 (the mesh-independence
controls). The slip error is the rms over the FREE nodes of
(b_x - exact) / delta0, the error over the peak slip, so the numbers compare
with the scratchpad ladder that chose the element (memory note
c0-p2-mollified-collocation). Every tolerance carries its measurement.

  [a] moments: Patch.integrate at P2 (the 3-midpoint rule) equals a 36-point
      Gauss rule on 1 - r^2/a^2 to 1e-14; the vertex rule misses by > 1e-3.
  [b] readout: the Eshelby slip PRESCRIBED on the continuous P2 fault, the
      elastic traction at the shared nodes (evaluate_fault_traction) at
      eps = 0.1 h over the nodes with r < a - h, e = |t_x + dtau| / dtau in
      units of eps/a: median in [1.3, 2.2] on every mesh (measured 1.65 at
      a/h 8; 1.75 / 1.74 / 1.75 on the CDT / jittered / lattice at 16), the
      vertices (kinks) and midpoints within 0.15 of each other (measured
      0.02-0.04: the kink costs nothing), the raw median falling from a/h 8
      to 16 by a factor in [1.5, 2.4] (measured 1.88: first order in eps),
      median t_x < 0 hardcoded. TRIPWIRES: the P0 centroid readout does NOT
      converge (its median at 16 over its median at 8 > 0.8; measured 1.15,
      2.17 % -> 2.50 %) while P2's falls below 0.65 (0.53); on the jittered
      mesh P0's median is > 2x P2's (3.4); the eigenstress left in gives a
      median > 20 (49 / 94).
  [c] stress-drop solve in the full-space region, order 2 continuous and the
      P0 control, eps = 0.1 h and "auto" (0.07 min h = 0.052 h): slip rms
      and moment against Eshelby (table in the constants, 1.5x measured);
      b_x at the centre within 10 % of delta0 and > 0 HARDCODED (a drop of
      the +x load on the +z face slips +x); b_y, b_z small; the locked rim
      exactly 0; jump="half" gives the same b bitwise (no jump enters a
      fault row); eps "auto" equals the explicit 0.07 min h array bitwise;
      DenseBackend() defaults run on a model with no u-slot. TRIPWIRES: P0's
      rms > 2x P2's at eps 0.1 h (3.0 / 2.8); P0's moment error > 2.5x P2's
      at a/h 8 (4.8); the eigenstress left in the rows (kind "total") changes
      the a/h 8 solve by > 20 % (measured 102 %); rolled edge nodes change
      it by > 1e-2 (12).
  [d] consistency: the free rows' residual A x - b < 1e-10 relative; the
      tangential elastic traction read back by evaluate_fault_traction (the
      contract kernels, near-list eigenstress) equals the target to 1e-6
      (the rows use the full-sum eigenstress; measured 1e-13 .. 1e-12).
  [e] coupled round trip on verify_solved_bvp's anchor-C box (768 P0 boundary
      triangles, free top and sides, clamped base) with a 48-triangle fault
      buried 15 km: the prescribed-slip solve (P0 and continuous P2, both
      jumps), its tangential traction read at the nodes, then the same box
      with a FAULT_TRACTION fault carrying that traction, nothing locked:
      slip, u:top, interior u and stress recovered to 1e-8 (an identity;
      measured 1e-14). TRIPWIRES: zeroing the (fault rows, boundary columns)
      block moves b by > 5e-4 (measured 1.4e-3 / 1.8e-3: the fault is deep);
      zeroing (boundary rows, b columns) moves u:top by 1.
  [f] translation: a free-traction box (all three patches Neumann,
      deflated) around a P0 traction fault: the fault rows applied to a unit
      translation of the box (the deflation basis Z) vanish (1e-12, measured
      1.6e-18 of the self-block scale), and with the eigenstress left in
      (kind "total") they do not (> 100x larger; measured 5e8): the reason
      rule 4 subtracts it in traction rows.
  [g] faceted fixture: the 6 x 6 square fault with its interior vertices
      displaced out of plane by up to 0.25 h (dihedral angles to 49 degrees),
      a smooth slip MAGNITUDE along the local dip direction, read at the
      interior nodes. With the direction per FACET (the discontinuous twin,
      callable of the element normals) the Burgers VECTOR jumps across every
      crease -- a line dislocation -- and the readout differs from the
      per-NODE one (the continuous layout, callable of the node normals) by a
      term that grows as a 1/eps core: at the vertices the median
      difference grows by > 2x when eps/h falls from 0.1 to 0.025 (measured
      3.5; 0.038 -> 0.132 of the traction scale), while the per-node readout
      itself moves by < 1.3x over the same range (measured 1.14: the
      gradient kink on a kinked surface, a log, not a core). A symmetric
      apex or a straight crease shows none of it at its own nodes (the cores
      cancel there), which is why the fixture is random roughness.
  [h] the tensile penny: allow_opening=True, value -p z_hat at a/h 8, the
      opening against Sneddon's 4 (1 - nu) p / (pi mu) sqrt(a^2 - r^2):
      b_z(centre) > 0 within 10 % (0.991), rms over the free nodes / peak <
      2.5 % (1.50), |b_x|, |b_y| rms < 1e-3 of the peak (0), P0's rms > 1.5x
      P2's (1.88; P0 2.81 %).

Run from the repo root:  python tests/gates/mbem/verify_penny_crack.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
from scipy.linalg import lu_factor, lu_solve

ROOT = pathlib.Path(__file__).resolve().parents[1]

import _sphere as S                                                 # noqa: E402
import verify_solved_bvp as VB                                      # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq                # noqa: E402
from mollified_bem import TriMesh                                   # noqa: E402
from mbem import ElasticMaterial                                    # noqa: E402
from mbem.backends.dense import AssembledDense, DenseBackend, translation_basis  # noqa: E402
from mbem.cases import penny as P                                   # noqa: E402
from mbem.evaluate import (evaluate_displacement, evaluate_fault_traction,  # noqa: E402
                           evaluate_stress)
from mbem.kernels import basis as kb                                # noqa: E402
from mbem.kernels import tri_nodal as tn                            # noqa: E402
from mbem.model import BCType, Patch, Region, RegionModel, generate_system  # noqa: E402
from mbem import defaults                                           # noqa: E402

MU, NU = 30.0, 0.30
LAM = 2.0 * MU * NU / (1.0 - 2.0 * NU)
MAT = ElasticMaterial(mu=MU, lam=LAM)
A = 1.0                        # km
DELTA0 = 1e-3                  # km = 1 m
DTAU = P.stress_drop(MU, LAM, DELTA0, A)
M_EXACT = P.eshelby_moment(DELTA0, A)
LEVELS = (8, 16)

# [b] readout, e / (eps/a) medians at eps = 0.1 h over r < a - h (nu 0.30):
# measured 1.645 (cdt 8), 1.753 / 1.742 / 1.747 (cdt / jitter / lattice 16)
READOUT_WINDOW = (1.3, 2.2)
READOUT_KINK_DIFF = 0.15       # |vertices - midpoints|, measured 0.016-0.034
READOUT_RATIO = (1.5, 2.4)     # raw median a/h 8 over 16, measured 1.88
P0_NO_CONVERGENCE = 0.8        # P0 median 16 / 8, measured 1.15
P2_CONVERGENCE = 0.65          # P2 median 16 / 8, measured 0.53
P0_VS_P2_JITTER = 2.0          # measured 3.4
EIGEN_LEFT_IN = 20.0           # measured 48.6 / 93.5

# [c] solve: (order, eps label, a/h) -> (rms tolerance, moment tolerance),
# 1.5x the measured rms and ~2x the measured moment (in % of delta0 / exact):
#   P2 0.1h   8: rms 1.59, moment +1.54   16: 0.97, +1.14
#   P2 auto   8: 0.66, -0.56               16: 0.36, -0.04
#   P0 0.1h   8: 4.77, +7.41               16: 2.68, +3.93
#   P0 auto   8: 4.13, +6.39               16: 2.33, +3.35
SOLVE_TOL = {
    (2, "0.1h", 8): (2.4, 3.0), (2, "0.1h", 16): (1.5, 2.5),
    (2, "auto", 8): (1.0, 1.5), (2, "auto", 16): (0.55, 1.0),
    (0, "0.1h", 8): (7.2, 11.0), (0, "0.1h", 16): (4.0, 6.0),
    (0, "auto", 8): (6.2, 9.5), (0, "auto", 16): (3.5, 5.0),
}
P0_VS_P2_RMS = 2.0             # measured 3.0 / 2.8
P0_VS_P2_MOMENT = 2.5          # measured 4.8 at a/h 8
EIGEN_SOLVE_MOVE = 0.2         # measured 1.02
ROLL_MOVE = 1e-2               # measured 12
TOL_RESIDUAL = 1e-10
TOL_READOUT_VS_ROWS = 1e-6     # near-list vs full-sum eigenstress; measured 1e-13 .. 1e-12
TOL_ROUNDTRIP = 1e-8           # measured 1e-14
TRIP_MOVE = 5e-4               # measured 1.36e-3 / 1.75e-3 (the fault is 15 km deep) and 1.0
TOL_TRANSLATION = 1e-12        # measured 1.6e-18 (the identity is exact for the smeared layer too)
TRANSLATION_CONTRAST = 100.0   # measured 5e8
FACET_GROWTH = 2.0             # |T - V| at vertices, eps/h 0.1 -> 0.025: measured 3.5
NODE_GROWTH = 1.3              # |V| at vertices over the same range: measured 1.14
SNEDDON_CENTRE = 0.10
SNEDDON_RMS = 2.5              # % of the peak, measured 1.50 (P0: 2.81)

CHECKS = []


def check(name, val, tol):
    ok = bool(np.isfinite(val)) and val < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3e} < {tol:.0e}")
    return ok


def check_gt(name, val, floor):
    ok = bool(np.isfinite(val)) and val > floor
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3e} > {floor:.0e}")
    return ok


def check_in(name, val, lo, hi):
    ok = bool(np.isfinite(val)) and lo < val < hi
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3f} in ({lo:g}, {hi:g})")
    return ok


def check_bool(name, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {note}")
    return ok


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def solve(model, eps, jump="calibrated", deflate=False, return_asm=False):
    asm = quiet(AssembledDense, generate_system(model), eps, "direct", jump=jump,
                deflate=deflate)
    sol = asm.solve()
    return (sol, asm) if return_asm else sol


def slip_error(fault, b, region):
    """rms over the free nodes of (b_x - exact) / delta0, in %."""
    x = fault.collocation_points()
    ex = P.eshelby_slip(A, DELTA0)(x)
    free = ~fault.locked_mask(region.patches)
    err = (b[:, 0] - ex[:, 0]) / DELTA0
    return 100.0 * float(np.sqrt(np.mean(err[free] ** 2)))


_MESHES: dict = {}


def mesh(ah, kind="cdt"):
    key = (ah, kind)
    if key not in _MESHES:
        _MESHES[key] = P.penny_mesh(A, A / ah, kind)
    return _MESHES[key]


def h_mean(m):
    return float(kb.element_sizes(m).mean())


# ---------------------------------------------------------------------------
# [a] moments
# ---------------------------------------------------------------------------

def a_moments():
    print("\n[a] Patch.integrate: the midpoint rule is exact for quadratics")
    m = mesh(8)
    pt = Patch("penny", m, BCType.FAULT, order=2, layout="continuous")
    x = pt.shared.coords
    f = (1.0 - (x[:, 0] ** 2 + x[:, 1] ** 2) / (A * A))[:, None] * np.ones((1, 3))
    x1, x2, w = tn._gauss_triangle(6)
    lam = np.stack([1.0 - x1 - x2, x1, x2], axis=1)
    tv = np.asarray(m.vertices, float)[np.asarray(m.triangles)]
    pts = np.einsum("qv,svc->sqc", lam, tv)
    fq = 1.0 - (pts[..., 0] ** 2 + pts[..., 1] ** 2) / (A * A)
    gauss = float((pt.shared.areas * (fq @ w) / w.sum()).sum())
    check("P2 integrate(1 - r^2/a^2) == 36-point Gauss", abs(pt.integrate(f)[0] - gauss) / gauss, 1e-14)
    vert = float((pt.shared.areas * f[pt.shared.ids[:, :3], 0].mean(axis=1)).sum())
    check_gt("TRIPWIRE: the vertex rule misses the quadratic", abs(vert - gauss) / gauss, 1e-3)
    p0 = Patch("penny", m, BCType.FAULT, order=0)
    c = p0.collocation_points()
    f0 = (1.0 - (c[:, 0] ** 2 + c[:, 1] ** 2) / (A * A))[:, None] * np.ones((1, 3))
    _, areas = m.normals_and_areas()
    check("P0 integrate == sum area x centroid value",
          abs(p0.integrate(f0)[0] - float((areas * f0[:, 0]).sum())) / gauss, 1e-14)


# ---------------------------------------------------------------------------
# [b] readout
# ---------------------------------------------------------------------------

def b_readout():
    print("\n[b] readout of the prescribed Eshelby slip at the shared nodes, eps = 0.1 h")
    med = {}
    for ah, kind in ((8, "cdt"), (16, "cdt"), (16, "jitter"), (16, "lattice")):
        m = mesh(ah, kind)
        hh = A / ah
        eps = 0.1 * h_mean(m)
        for order in (2, 0):
            model = P.build_slip_model(m, MAT, delta0=DELTA0, order=order, a=A)
            r = model.regions[0]
            f = r.faults[0]
            t = evaluate_fault_traction(model, r, {}, f, eps)
            x = f.collocation_points()
            inner = np.hypot(x[:, 0], x[:, 1]) < A - hh
            e = np.abs(t[:, 0] + DTAU) / DTAU
            med[(ah, kind, order)] = float(np.median(e[inner]))
            if order == 2:
                u = med[(ah, kind, 2)] / (eps / A)
                isv = np.arange(f.n_dofs) < f.shared.n_vertices
                ev = float(np.median(e[inner & isv])) / (eps / A)
                em = float(np.median(e[inner & ~isv])) / (eps / A)
                check_in(f"{kind} a/h {ah}: P2 median e / (eps/a)", u, *READOUT_WINDOW)
                check(f"{kind} a/h {ah}: |vertices - midpoints| / (eps/a) (the kink costs nothing)",
                      abs(ev - em), READOUT_KINK_DIFF)
                check_bool(f"{kind} a/h {ah}: median t_x < 0 (slip +x reads a drop of the +x load)",
                           np.median(t[inner, 0]) < 0.0)
                if (ah, kind) == (16, "cdt"):
                    t_tot = evaluate_fault_traction(model, r, {}, f, eps, subtract_anelastic=False)
                    e_tot = np.abs(t_tot[:, 0] + DTAU) / DTAU
                    check_gt("TRIPWIRE: eigenstress left in, median e", float(np.median(e_tot[inner])),
                             EIGEN_LEFT_IN)
            else:
                print(f"      {kind} a/h {ah}: P0 centroid median e {med[(ah, kind, 0)]:.4f}, "
                      f"p90 {np.percentile(e[inner], 90):.4f}")
    check_in("cdt: P2 raw median a/h 8 over a/h 16 (first order in eps)",
             med[(8, 'cdt', 2)] / med[(16, 'cdt', 2)], *READOUT_RATIO)
    check_gt("TRIPWIRE: P0 median a/h 16 over a/h 8 (the centroid floor does not fall)",
             med[(16, 'cdt', 0)] / med[(8, 'cdt', 0)], P0_NO_CONVERGENCE)
    check("P2 median a/h 16 over a/h 8 (it converges)", med[(16, 'cdt', 2)] / med[(8, 'cdt', 2)],
          P2_CONVERGENCE)
    check_gt("TRIPWIRE: jittered mesh, P0 median over P2 median",
             med[(16, 'jitter', 0)] / med[(16, 'jitter', 2)], P0_VS_P2_JITTER)


# ---------------------------------------------------------------------------
# [c] + [d] the stress-drop solve
# ---------------------------------------------------------------------------

def cd_solve():
    print("\n[c] the stress-drop solve in the full space; [d] its consistency")
    rms, mom = {}, {}
    for ah in LEVELS:
        m = mesh(ah)
        hm = h_mean(m)
        for order in (2, 0):
            model = P.build_model(m, MAT, drop=DTAU, order=order)
            r = model.regions[0]
            f = r.faults[0]
            for eps_spec, lab in ((0.1 * hm, "0.1h"), ("auto", "auto")):
                sol, asm = solve(model, eps_spec, return_asm=True)
                b = sol["b:penny"]
                rms[(order, lab, ah)] = slip_error(f, b, r)
                mom[(order, lab, ah)] = 100.0 * (f.integrate(b)[0] / M_EXACT - 1.0)
                tol_rms, tol_mom = SOLVE_TOL[(order, lab, ah)]
                check(f"a/h {ah} P{order} eps {lab}: slip rms / delta0 at the free nodes (%)",
                      rms[(order, lab, ah)], tol_rms)
                check(f"a/h {ah} P{order} eps {lab}: |moment / exact - 1| (%)",
                      abs(mom[(order, lab, ah)]), tol_mom)
                if lab == "0.1h":
                    x = f.collocation_points()
                    ic = int(np.argmin(np.hypot(x[:, 0], x[:, 1])))
                    check_bool(f"a/h {ah} P{order}: b_x(centre) > 0 and within 10 % of delta0",
                               b[ic, 0] > 0.0 and abs(b[ic, 0] / DELTA0 - 1.0) < 0.1,
                               f"{b[ic, 0] / DELTA0:.4f}")
                    check(f"a/h {ah} P{order}: |b_y|, |b_z| max / delta0",
                          float(np.abs(b[:, 1:]).max()) / DELTA0, 3e-2)
                    lock = f.locked_mask(r.patches)
                    check_bool(f"a/h {ah} P{order}: locked rim nodes exactly 0 ({int(lock.sum())} nodes)",
                               (not lock.any()) or np.all(b[lock] == 0.0))
                    # [d] residual of the free rows, and the readout against the target
                    slot = asm.layout.slot(f, "b")
                    res = asm.A @ np.concatenate([sol[s.name].ravel() for s in asm.layout.slots]) - asm.b
                    free_rows = np.repeat(~lock, 3)
                    check(f"a/h {ah} P{order}: [d] free-row residual |A x - b| / |b|",
                          float(np.abs(res[slot.offset:slot.stop][free_rows]).max()
                                / np.abs(asm.b).max()), TOL_RESIDUAL)
                    t = evaluate_fault_traction(model, r, sol, f, eps_spec)
                    nrm = f.collocation_normals()
                    t_tan = t - np.einsum("ni,ni->n", t, nrm)[:, None] * nrm
                    target = f.value_array()
                    check(f"a/h {ah} P{order}: [d] evaluate_fault_traction (tangential) == target",
                          float(np.abs((t_tan - target)[~lock]).max() / np.abs(target).max()),
                          TOL_READOUT_VS_ROWS)
                if ah == 8 and lab == "0.1h":
                    sol_h = solve(model, eps_spec, jump="half")
                    check_bool(f"a/h {ah} P{order}: jump='half' gives b bitwise (no jump on a fault row)",
                               np.array_equal(sol_h["b:penny"], b))
                if ah == 8 and lab == "auto":
                    explicit = np.full(m.n_triangles, defaults.FAULT_EPS_OVER_H
                                       * float(kb.element_sizes(m).min()))
                    check_bool(f"a/h {ah} P{order}: eps 'auto' == explicit 0.07 min h, bitwise solution",
                               np.array_equal(solve(model, explicit)["b:penny"], b))
        if ah == 8:
            check_bool("DenseBackend() defaults run on a model with no u-slot",
                       "b:penny" in quiet(DenseBackend().assemble,
                                          generate_system(P.build_model(m, MAT, drop=DTAU)),
                                          "auto").solve())
    for ah in LEVELS:
        check_gt(f"TRIPWIRE a/h {ah}: P0 rms over P2 rms at eps 0.1 h", rms[(0, '0.1h', ah)] / rms[(2, '0.1h', ah)],
                 P0_VS_P2_RMS)
    check_gt("TRIPWIRE a/h 8: P0 moment error over P2's", abs(mom[(0, '0.1h', 8)]) / abs(mom[(2, '0.1h', 8)]),
             P0_VS_P2_MOMENT)
    # the eigenstress left in the rows: route the rows' kernel to the TOTAL traction
    m = mesh(8)
    model = P.build_model(m, MAT, drop=DTAU, order=2)
    r = model.regions[0]
    f = r.faults[0]
    ref = solve(model, 0.1 * h_mean(m))["b:penny"]
    import mbem.backends.dense as dense_mod
    orig = dense_mod.kb.assemble_traction_matrix

    def total(*a, **k):
        if k.get("kind") == "slip":
            k["kind"] = "total"
        return orig(*a, **k)

    dense_mod.kb.assemble_traction_matrix = total
    try:
        bad = solve(model, 0.1 * h_mean(m))["b:penny"]
    finally:
        dense_mod.kb.assemble_traction_matrix = orig
    check_gt("TRIPWIRE: eigenstress left in the rows moves the a/h 8 P2 solve (relative)",
             float(np.linalg.norm(bad - ref) / np.linalg.norm(ref)), EIGEN_SOLVE_MOVE)
    # rolled edge nodes: the shared table with its m12 m23 m31 columns rolled
    f._shared = None
    tab = f.shared
    ids_ok = tab.ids.copy()
    tab.ids[:, 3:6] = np.roll(tab.ids[:, 3:6], 1, axis=1)
    try:
        rolled = solve(model, 0.1 * h_mean(m))["b:penny"]
    finally:
        tab.ids[:] = ids_ok
    check_gt("TRIPWIRE: rolled edge-node ids change the solve (relative)",
             float(np.linalg.norm(rolled - ref) / np.linalg.norm(ref)), ROLL_MOVE)


# ---------------------------------------------------------------------------
# [e] the coupled round trip
# ---------------------------------------------------------------------------

def e_round_trip():
    print("\n[e] coupled round trip on the anchor-C box: prescribed slip -> traction -> solved slip")
    mat = VB.material(NU)
    box = VB._closed_box(VB.EDGE_C)
    keys = ("top", "sides", "base")
    fc = dict(VB.FAULT_C, target_edge=2.5)
    fmesh, _, _ = make_vertical_fault_eq(**fc)
    zc = 0.5 * sum(fc["depth_range"])
    D = fc["depth_range"][1] - fc["depth_range"][0]
    L = fc["strike_length"]

    def slip_field(x):
        x = np.asarray(x, float)
        out = np.zeros((x.shape[0], 3))
        out[:, 1] = VB.SLIP * (1.0 + 0.5 * x[:, 1] / (0.5 * L) + 0.3 * (x[:, 2] - zc) / (0.5 * D))
        return out

    eps = {"top": VB.EPS_C, "sides": VB.EPS_C, "base": VB.EPS_C, "fault": "auto"}
    rng = np.random.default_rng(31415)
    obs = np.column_stack([rng.uniform(-34.0, 34.0, 600), rng.uniform(-34.0, 34.0, 600),
                           rng.uniform(-32.0, -8.0, 600)])
    obs = obs[np.max(np.abs(obs[:, :2]), axis=1) > 26.0][:80]
    bcs = dict(top="t", sides="t", base="d")
    print(f"      fault {fmesh.n_triangles} triangles, box {sum(box[k].n_triangles for k in keys)}")
    for order, layout in ((0, "discontinuous"), (2, "continuous")):
        for jump in ("half", "calibrated"):
            fA = Patch("fault", fmesh, BCType.FAULT, value=slip_field, order=order, layout=layout)
            mA, rA = VB._build_box_model(box, mat, bcs, {k: None for k in keys}, faults=[fA])
            solA = solve(mA, eps, jump=jump)
            tA = evaluate_fault_traction(mA, rA, solA, fA, eps)
            nrm = fA.collocation_normals()
            tA_t = tA - np.einsum("ni,ni->n", tA, nrm)[:, None] * nrm
            fB = Patch("fault", fmesh, BCType.FAULT_TRACTION, value=tA_t, order=order,
                       layout=layout, locked=np.zeros(fA.n_dofs, bool))
            mB, rB = VB._build_box_model(box, mat, bcs, {k: None for k in keys}, faults=[fB])
            solB, asmB = solve(mB, eps, jump=jump, return_asm=True)
            label = f"P{order} {layout} {jump}"
            bA = fA.value_array()
            check(f"{label}: b recovered", float(np.linalg.norm(solB['b:fault'] - bA) / np.linalg.norm(bA)),
                  TOL_ROUNDTRIP)
            check(f"{label}: u:top recovered",
                  float(np.linalg.norm(solB['u:top'] - solA['u:top']) / np.linalg.norm(solA['u:top'])),
                  TOL_ROUNDTRIP)
            uA = evaluate_displacement(mA, rA, solA, obs, eps, warn_near=False)
            uB = evaluate_displacement(mB, rB, solB, obs, eps, warn_near=False)
            check(f"{label}: interior u recovered", float(np.linalg.norm(uB - uA) / np.linalg.norm(uA)),
                  TOL_ROUNDTRIP)
            if jump == "calibrated":
                sA = evaluate_stress(mA, rA, solA, obs, eps, warn_near=False)
                sB = evaluate_stress(mB, rB, solB, obs, eps, warn_near=False)
                check(f"{label}: interior stress recovered", float(np.linalg.norm(sB - sA) / np.linalg.norm(sA)),
                      TOL_ROUNDTRIP)
                # tripwires: the coupling blocks matter
                lay = asmB.layout
                bslot, uslot = lay.slot(fB, "b"), lay.slot(rB.patches[0], "u")
                x_ref = np.concatenate([solB[s.name].ravel() for s in lay.slots])
                for name, rows, cols, read in (("fault rows <- boundary columns", bslot,
                                                [s for s in lay.slots if s.kind != "b"], "b:fault"),
                                               ("boundary rows <- b columns",
                                                [s for s in lay.slots if s.kind != "b"], [bslot], "u:top")):
                    A2 = asmB.A.copy()
                    rr = [rows] if not isinstance(rows, list) else rows
                    for rs in rr:
                        for cs in cols:
                            A2[rs.offset:rs.stop, cs.offset:cs.stop] = 0.0
                    x2 = lu_solve(lu_factor(A2), asmB.b)
                    sl = lay.slot(fB, "b") if read == "b:fault" else uslot
                    moved = np.linalg.norm(x2[sl.offset:sl.stop] - x_ref[sl.offset:sl.stop]) \
                        / np.linalg.norm(x_ref[sl.offset:sl.stop])
                    check_gt(f"TRIPWIRE {label}: zeroing {name} moves {read}", float(moved), TRIP_MOVE)


# ---------------------------------------------------------------------------
# [f] translation
# ---------------------------------------------------------------------------

def f_translation():
    print("\n[f] a rigid translation of a free-traction box is annihilated by the fault rows")
    mat = VB.material(NU)
    box = VB._closed_box(VB.EDGE_C)
    keys = ("top", "sides", "base")
    fmesh, _, s_hat = make_vertical_fault_eq(**VB.FAULT_C)
    eps = {"top": VB.EPS_C, "sides": VB.EPS_C, "base": VB.EPS_C, "fault": "auto"}
    fB = Patch("fault", fmesh, BCType.FAULT_TRACTION, value=np.zeros(3))
    mB, rB = VB._build_box_model(box, mat, dict(top="t", sides="t", base="t"),
                                 {k: None for k in keys}, faults=[fB])
    import mbem.backends.dense as dense_mod
    orig = dense_mod.kb.assemble_traction_matrix
    out = {}
    for label, kind in (("eigenstress removed", "slip"), ("eigenstress left in", "total")):
        def patched(*a, **k):
            if k.get("kind") == "slip":
                k["kind"] = kind
            return orig(*a, **k)
        dense_mod.kb.assemble_traction_matrix = patched
        try:
            sol, asm = solve(mB, eps, deflate=True, return_asm=True)
        finally:
            dense_mod.kb.assemble_traction_matrix = orig
        slot = asm.layout.slot(fB, "b")
        Z = translation_basis(asm.layout)
        rows = asm.A[slot.offset:slot.stop]
        scale = float(np.abs(rows[:, slot.offset:slot.stop]).max())
        out[kind] = float(np.abs(rows @ Z).max() / scale)
        print(f"      {label}: max |fault rows . Z| / self-block scale = {out[kind]:.3e}, "
              f"solved |b| max {np.abs(sol['b:fault']).max():.2e}")
    check("fault rows annihilate the unit translation (eigenstress removed)", out["slip"], TOL_TRANSLATION)
    check_gt("TRIPWIRE: with the eigenstress left in they do not (ratio)", out["total"] / out["slip"],
             TRANSLATION_CONTRAST)


# ---------------------------------------------------------------------------
# [g] the faceted fixture
# ---------------------------------------------------------------------------

def g_faceted():
    print("\n[g] a roughened square fault: slip direction per facet vs per node, as eps falls")
    half, n, amp = 0.3, 6, 0.25
    hh = 2.0 * half / n
    base = S.square_fault(half=half, n=n)
    v = np.asarray(base.vertices, float).copy()
    rng = np.random.default_rng(1)
    interior = (np.abs(v[:, 1]) < half - 1e-12) & (np.abs(v[:, 2]) < half - 1e-12)
    v[interior, 0] += rng.uniform(-amp * hh, amp * hh, int(interior.sum()))
    rough = TriMesh(vertices=np.ascontiguousarray(v), triangles=base.triangles.copy())
    nrm_e, _ = rough.normals_and_areas()
    dihedral = np.degrees(np.arccos(np.clip((nrm_e @ nrm_e.T).min(), -1.0, 1.0)))

    def magnitude(x):
        return DELTA0 * np.clip(1.0 - (x[:, 1] / half) ** 2, 0.0, None) \
            * np.clip(1.0 - (x[:, 2] / half) ** 2, 0.0, None)

    def value(points, normals):
        d = np.cross(normals, np.array([0.0, 1.0, 0.0]))       # the local dip direction
        d /= np.linalg.norm(d, axis=1)[:, None]
        return magnitude(np.asarray(points, float))[:, None] * d

    pc = Patch("f", rough, BCType.FAULT, value=value, order=2, layout="continuous")
    pd = Patch("f", rough, BCType.FAULT, value=value, order=2)
    tab = pc.shared
    inner = ~tab.boundary & (np.abs(tab.coords[:, 1]) < half - 0.9 * hh) \
        & (np.abs(tab.coords[:, 2]) < half - 0.9 * hh)
    isv = (np.arange(tab.n) < tab.n_vertices)[inner]
    pts, nn = tab.coords[inner], tab.normals[inner]
    print(f"      {rough.n_triangles} facets, largest dihedral angle {dihedral:.0f} deg, "
          f"{int(isv.sum())} vertices and {int((~isv).sum())} midpoints read")
    out = {}
    for name, patch in (("node", pc), ("facet", pd)):
        model = RegionModel([Region("space", MAT, [], np.zeros(3), faults=[patch])])
        r = model.regions[0]
        for eh in (0.1, 0.025):
            sig = evaluate_stress(model, r, {}, pts, eh * hh, warn_near=False)
            out[(name, eh)] = np.einsum("nij,nj->ni", sig, nn)
    scale = float(np.median(np.linalg.norm(out[("node", 0.1)], axis=1)))
    diff = {eh: np.linalg.norm(out[("facet", eh)] - out[("node", eh)], axis=1) / scale
            for eh in (0.1, 0.025)}
    own = {eh: np.linalg.norm(out[("node", eh)], axis=1) / scale for eh in (0.1, 0.025)}
    for eh in (0.1, 0.025):
        print(f"      eps/h {eh}: median |t_facet - t_node| / scale at vertices {np.median(diff[eh][isv]):.3f}, "
              f"midpoints {np.median(diff[eh][~isv]):.3f}; median |t_node| at vertices {np.median(own[eh][isv]):.3f}")
    check_gt("TRIPWIRE: |per-facet - per-node| at the vertices grows as eps falls 4x (a 1/eps core)",
             float(np.median(diff[0.025][isv]) / np.median(diff[0.1][isv])), FACET_GROWTH)
    check("the per-node (continuous) readout at the vertices over the same 4x in eps",
          float(np.median(own[0.025][isv]) / np.median(own[0.1][isv])), NODE_GROWTH)


# ---------------------------------------------------------------------------
# [h] the tensile penny
# ---------------------------------------------------------------------------

def h_tensile():
    print("\n[h] the pressurised penny (allow_opening) against Sneddon")
    m = mesh(8)
    p = DTAU
    peak = 4.0 * (1.0 - NU) * p * A / (np.pi * MU)
    rms = {}
    for order in (2, 0):
        model = P.build_model(m, MAT, pressure=p, order=order, allow_opening=True)
        r = model.regions[0]
        f = r.faults[0]
        b = solve(model, 0.1 * h_mean(m))["b:penny"]
        x = f.collocation_points()
        ex = P.sneddon_opening(A, p, MU, LAM)(x)
        free = ~f.locked_mask(r.patches)
        rms[order] = 100.0 * float(np.sqrt(np.mean(((b[free, 2] - ex[free, 2]) / peak) ** 2)))
        ic = int(np.argmin(np.hypot(x[:, 0], x[:, 1])))
        if order == 2:
            check_bool("P2: b_z(centre) > 0 and within 10 % of Sneddon's peak",
                       b[ic, 2] > 0.0 and abs(b[ic, 2] / peak - 1.0) < SNEDDON_CENTRE,
                       f"{b[ic, 2] / peak:.4f}")
            check("P2: opening rms / peak at the free nodes (%)", rms[2], SNEDDON_RMS)
            check("P2: |b_x|, |b_y| rms / peak", float(np.sqrt(np.mean((b[free, :2] / peak) ** 2))), 1e-3)
        else:
            print(f"      P0: opening rms / peak {rms[0]:.2f} %")
    check_gt("TRIPWIRE: P0 rms over P2 rms", rms[0] / rms[2], 1.5)   # measured 1.88


def main():
    print("=" * 76)
    print("The penny-shaped crack: traction-BC fault, continuous P2 layout, coupling")
    print("=" * 76)
    print(f"  dtau {DTAU * 1e3:.3f} MPa for delta0 {DELTA0 * 1e3:.1f} m, exact moment {M_EXACT:.5f}")
    a_moments()
    b_readout()
    cd_solve()
    e_round_trip()
    f_translation()
    g_faceted()
    h_tensile()
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: penny crack ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: penny crack ({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
