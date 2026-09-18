"""Verify a SOLVED boundary-value problem end to end: assembly -> boundary
conditions -> solve -> readout.

Every other gate in ``verify/`` checks one link of the chain -- a kernel, an
assembler, a contraction, or an internal consistency identity.  None of them
closes the loop, so a defect in the *right-hand side* (a prescribed value on
the wrong side of the equation, a fault slip entering with the wrong sign) is
invisible: flipping ``scale=-1.0`` -> ``+1.0`` on the fault RhsTerm in
``mbem/model/equations.py`` leaves all 13 msd gates PASSing with byte-identical
stdout.  This gate exists to close that hole.

It uses TWO anchors because they fail differently:

  A. EXTERNAL, absolute, loose.  Solve the standard ``_fault_box`` model (free
     top, free sides, clamped base, one vertical strike-slip fault) and compare
     the displacement against ``cutde``'s half-space triangular dislocation
     elements on the IDENTICAL fault mesh.  cutde is an independent
     implementation of different mathematics (classical singular Okada-family
     TDEs in a true half space), so it pins the ABSOLUTE convention -- sign,
     orientation, slip direction, the collocation free term and the evaluate
     path -- all at once.  It is loose because msd's box is a TRUNCATED half
     space and its slip is mollified: both are physics, not error.
     A third sub-check pins the slip sense with NO external reference at all.

  B. INTERNAL, manufactured, tight.  Impose the exact data of a known
     elasticity solution on a closed box and check that the solve reproduces
     it.  Two solutions are used: a rigid translation (u and t BOTH exactly P0
     representable, so ``t == 0`` must come out at machine precision) and a
     uniform strain ``u = A x`` with A symmetric (traction piecewise constant,
     hence exactly P0 representable; the displacement is linear, so its P0
     collocation error is the method's own and converges).  The uniform strain
     is run all-Dirichlet (B2), MIXED (B3) -- because all-Dirichlet never
     touches the prescribed-traction RHS path, nor the LHS free term, since it
     has no u unknowns -- and across a TWO-REGION INTERFACE (B4), which no
     single-region model can probe.  Tight enough to catch what anchor A's
     few-percent truncation floor would hide: BC application, the free term,
     the interface orientation sign, per-patch eps routing.

Measured detection (each seeded one at a time in a sandbox copy):
  fault RhsTerm sign      equations.py:117  -> 30/144 fail, A only  (cos -> -1)
  lam/mu swap in t_coeffs basis.py:64       -> 61/144, first at nu=0.30
  LHS free term 1/2->1/4  dense.py:208      -> 31/144, A + B3
  prescribed-u RHS sign   equations.py:92   -> 78/144, B only
  prescribed-u half term  equations.py:94   -> 36/144, B only
  prescribed-t RHS sign   equations.py:108  -> 36/144, B3 only
  orientation sign on G   equations.py:101  -> 48/144, B2 + B3
  interface-only sigma    equations.py:101  -> 24/144, B4 only

Both anchors run at nu = 1/4 AND nu != 1/4 (0.30, 0.45) -- the lam/mu pairing
is invisible at nu = 1/4, which is how a swapped kernel shipped for months --
and under BOTH jump conventions ("half" and "calibrated").

All checks PASS/FAIL.
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import mollified_bem as mb                                        # noqa: E402
from _fault_box import build_fault_box, build_model               # noqa: E402
from local_box_mesh_eq import (                                   # noqa: E402
    _concatenate_meshes,
    make_rectangular_patch_eq,
    make_vertical_panel_eq,
)
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.evaluate import evaluate_displacement                   # noqa: E402
from mbem.model import (                                          # noqa: E402
    BCType,
    Patch,
    Region,
    RegionModel,
    generate_system,
)

CHECKS = []

MU = 30.0
NU_LIST = (0.25, 0.30, 0.45)      # 1/4 hides the lam/mu pairing; 0.30/0.45 do not
JUMPS = ("half", "calibrated")

# ---------------------------------------------------------------- anchor A --
SLIP = 0.01                        # km == 10 m
EPS_A = 3.0                        # km, == the fault element size
# cutde carries slip as [strike, dip, tensile] in the per-triangle TDCS frame
# and uses the OPPOSITE displacement-discontinuity sign convention to the mbem
# fault source (mbem: u = -H @ slip).  This factor is FROZEN, not fitted: if it
# were fitted from the data the gate would be sign-blind, which is the very
# defect it exists to catch.
CUTDE_SLIP_SIGN = -1.0
# Surface points closer to the trace than this are mollification-limited: the
# mollified slip smears the surface step over ~eps, so msd under-predicts the
# jump there by construction while cutde's singular kernel does not.  Measured:
# including them drops the cosine from 0.9995 to 0.956 while leaving the median
# unchanged, i.e. it is a near-field amplitude deficit, not a direction error.
TRACE_EXCLUSION = 1.5 * EPS_A

COS_TOL = 0.995        # measured worst 0.99908 (surface) / 0.99951 (interior);
#                        a sign flip drives this negative, so the margin is huge
MED_TOL_SURF = 0.12    # measured worst 0.046 -- box truncation + mollification
MED_TOL_EVAL = 0.10    # measured worst 0.019

# ---------------------------------------------------------------- anchor B --
BOX_L, BOX_H, BOX_EDGE = 40.0, 40.0, 10.0
EPS_B = 1.25                                        # eps/h = 0.125
EPS_B_DICT = {"top": 1.0, "sides": 1.6, "base": 0.8}   # per-patch eps routing
# u = A x, A symmetric -> sigma = lam tr(A) I + 2 mu A is CONSTANT, so the
# traction is piecewise constant and exactly representable at P0.
STRAIN_A = 1.0e-4 * np.array([[2.0, -1.0, 0.5],
                              [-1.0, 1.0, -1.5],
                              [0.5, -1.5, -3.0]])
RIGID_C = np.array([0.003, -0.002, 0.001])

TOL_RIGID_T = 1.0e-10   # calibrated jump annihilates constants EXACTLY (5e-14)
TOL_RIGID_U = 5.0e-3    # measured worst 1.3e-3
TOL_DIR_U = 1.0e-2      # measured worst 3.4e-3
TOL_DIR_T = 0.12        # measured worst 0.076 (box edges excluded)
TOL_MIX_U = 4.0e-2      # measured worst 1.9e-2
TOL_MIX_UB = 0.10       # measured worst 0.057
TOL_MIX_T = 0.10        # measured worst 0.048 (edges cut; 0.137 with edges in)
TOL_IFACE_U = 2.0e-2    # measured worst 9.8e-3 (upper region, nu=0.45)
TOL_IFACE_T = 0.15      # measured worst 0.076 (interface rim excluded)


def check(label, value, tol, fmt="{:.3e}"):
    ok = bool(np.isfinite(value)) and value < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:58s} "
          f"{fmt.format(value)} < {fmt.format(tol)}")
    return ok


def check_true(label, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:58s} {note}")
    return ok


def material(nu):
    return mb.ElasticMaterial(mu=MU, lam=2.0 * MU * nu / (1.0 - 2.0 * nu))


def relerr(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def median_rel(a, b):
    nb = np.linalg.norm(b, axis=1)
    return float(np.median(np.linalg.norm(a - b, axis=1) / nb))


# ============================================================== ANCHOR A =====

def _cutde_halfspace_disp(obs, fault, slip_cart, nu):
    """Half-space TDE displacement at ``obs`` for the SAME triangulated fault."""
    import cutde.geometry as cg
    import cutde.halfspace as hs

    tris = np.ascontiguousarray(
        np.asarray(fault.vertices, float)[np.asarray(fault.triangles)])
    slip_cart = np.broadcast_to(np.asarray(slip_cart, float), (tris.shape[0], 3))
    rot = cg.compute_efcs_to_tdcs_rotations(tris)
    slip_tdcs = np.ascontiguousarray(np.einsum("sij,sj->si", rot, slip_cart))
    mat = hs.disp_matrix(np.ascontiguousarray(np.asarray(obs, float)), tris, nu)
    return CUTDE_SLIP_SIGN * np.einsum("oisc,sc->oi", mat, slip_tdcs)


def a_external_cutde(meshes):
    """Solved free-surface + interior displacement vs cutde half-space TDEs."""
    print("\n[A] EXTERNAL anchor: solved fault box vs cutde half-space TDE")
    try:
        import cutde.halfspace  # noqa: F401
    except ImportError as exc:
        # Do NOT skip quietly: without the external anchor the fault sign is
        # unguarded again, which is the whole point of this gate.
        print(f"  (cutde not importable -- {exc}); external anchor FAILED")
        CHECKS.append(False)
        return
    fault = meshes["fault"]
    slip_vec = SLIP * np.asarray(meshes["s_hat"], float)

    top_c = meshes["top"].centroids()
    keep = np.abs(top_c[:, 0]) > TRACE_EXCLUSION
    obs_surf = top_c[keep].copy()
    obs_surf[:, 2] = -1.0e-6            # cutde wants z < 0; msd's are at z = 0

    rng = np.random.default_rng(20260918)
    obs_int = np.column_stack([rng.uniform(-60.0, 60.0, 80),
                               rng.uniform(-60.0, 60.0, 80),
                               rng.uniform(-45.0, -12.0, 80)])
    obs_int = obs_int[np.abs(obs_int[:, 0]) > TRACE_EXCLUSION]

    for nu in NU_LIST:
        mat = material(nu)
        model = build_model(meshes, SLIP, mat)
        region = model.regions[0]
        system = generate_system(model)
        ref_surf = _cutde_halfspace_disp(obs_surf, fault, slip_vec, nu)
        ref_int = _cutde_halfspace_disp(obs_int, fault, slip_vec, nu)
        for jump in JUMPS:
            sol = AssembledDense(system, EPS_A, "direct", jump=jump).solve()
            u_surf = sol["u:top"][keep]
            u_int = evaluate_displacement(model, region, sol, obs_int, EPS_A,
                                          warn_near=False)
            tag = f"nu={nu:.2f} {jump:10s}"
            check(f"A1 surface u vs cutde: 1-cos          {tag}",
                  1.0 - cosine(u_surf, ref_surf), 1.0 - COS_TOL)
            check(f"A1 surface u vs cutde: median rel     {tag}",
                  median_rel(u_surf, ref_surf), MED_TOL_SURF, "{:.4f}")
            check(f"A2 interior u vs cutde: 1-cos         {tag}",
                  1.0 - cosine(u_int, ref_int), 1.0 - COS_TOL)
            check(f"A2 interior u vs cutde: median rel    {tag}",
                  median_rel(u_int, ref_int), MED_TOL_EVAL, "{:.4f}")


def a_slip_sense(meshes):
    """Absolute slip sense -- no external reference, no fitted sign.

    Fault normal n_hat = +x_hat, slip vector Du = +SLIP * y_hat.  msd's fault
    source is u = -H @ slip, whose jump is u(+n) - u(-n) = -Du: the block on
    the +x side must translate toward -y and the block on the -x side toward
    +y, each by a fraction of |Du| at the free surface.  This is the single
    statement the seeded RHS sign flip inverts, and it is frozen in the source
    below rather than read off the data.
    """
    print("\n[A3] ABSOLUTE slip sense at the free surface (no external ref)")
    n_hat = np.asarray(meshes["n_hat"], float)
    s_hat = np.asarray(meshes["s_hat"], float)
    assert np.allclose(n_hat, [1.0, 0.0, 0.0]) and np.allclose(s_hat, [0, 1, 0])
    c = meshes["top"].centroids()
    band = np.abs(c[:, 1]) < 6.0
    plus = band & (c[:, 0] > TRACE_EXCLUSION) & (c[:, 0] < 20.0)
    minus = band & (c[:, 0] < -TRACE_EXCLUSION) & (c[:, 0] > -20.0)

    for nu in NU_LIST:
        model = build_model(meshes, SLIP, material(nu))
        system = generate_system(model)
        for jump in JUMPS:
            u = AssembledDense(system, EPS_A, "direct", jump=jump).solve()["u:top"]
            up = float(u[plus, 1].mean())
            um = float(u[minus, 1].mean())
            tag = f"nu={nu:.2f} {jump:10s}"
            check_true(f"A3 +n side moves -Du, -n side +Du   {tag}",
                       up < 0.0 < um, f"u_y(+x)={up:+.3e} u_y(-x)={um:+.3e}")
            # near-antisymmetry about the fault plane
            check(f"A3 antisymmetry |sum|/|diff|         {tag}",
                  abs(up + um) / abs(up - um), 0.15, "{:.4f}")
            # the surface really moves an appreciable fraction of the slip
            frac = 0.5 * (abs(up) + abs(um)) / SLIP
            check_true(f"A3 surface |u_y| is 0.10-0.45 of slip {tag}",
                       0.10 < frac < 0.45, f"frac={frac:.3f}")


# ============================================================== ANCHOR B =====

def _closed_box():
    xr, yr = (-BOX_L, BOX_L), (-BOX_L, BOX_L)
    zr = (-BOX_H, 0.0)
    sides = _concatenate_meshes([
        make_vertical_panel_eq("x", xr[1], yr, zr, BOX_EDGE, +1),
        make_vertical_panel_eq("x", xr[0], yr, zr, BOX_EDGE, -1),
        make_vertical_panel_eq("y", yr[1], xr, zr, BOX_EDGE, +1),
        make_vertical_panel_eq("y", yr[0], xr, zr, BOX_EDGE, -1),
    ])
    return {
        "top": make_rectangular_patch_eq(xr, yr, 0.0, BOX_EDGE, normal_up=True),
        "base": make_rectangular_patch_eq(xr, yr, -BOX_H, BOX_EDGE,
                                          normal_up=False),
        "sides": sides,
    }


def _edge_distance(mesh):
    """Distance from each centroid to the nearest box EDGE (12 of them).

    A centroid lies on one face, so its distance to that face's plane is 0 and
    the SECOND smallest of the six face-plane distances is the distance to the
    nearest box edge.
    """
    c = mesh.centroids()
    d = np.abs(np.stack([BOX_L - c[:, 0], BOX_L + c[:, 0],
                         BOX_L - c[:, 1], BOX_L + c[:, 1],
                         -c[:, 2], c[:, 2] + BOX_H], axis=1))
    return np.sort(d, axis=1)[:, 1]


def _rim_distance(mesh):
    """Distance from each centroid to the rim of a horizontal interior patch."""
    c = mesh.centroids()
    return np.minimum(BOX_L - np.abs(c[:, 0]), BOX_L - np.abs(c[:, 1]))


def _exact_u(x):
    return np.asarray(x, float) @ STRAIN_A.T


def _exact_sigma(mat):
    return mat.lam * np.trace(STRAIN_A) * np.eye(3) + 2.0 * mat.mu * STRAIN_A


def _exact_t(mesh, mat):
    n, _ = mesh.normals_and_areas()
    return n @ _exact_sigma(mat).T


def _build_box_model(box, mat, bcs, values):
    patches = [Patch(k, box[k],
                     BCType.PRESCRIBED_DISPLACEMENT if bcs[k] == "d"
                     else BCType.FREE_TRACTION,
                     value=values[k])
               for k in ("top", "sides", "base")]
    region = Region("block", mat, patches,
                    probe_point=np.array([1.0, 2.0, -0.5 * BOX_H]))
    return RegionModel([region]), region


def _interior_points(n=60):
    rng = np.random.default_rng(31415)
    return np.column_stack([rng.uniform(-0.55 * BOX_L, 0.55 * BOX_L, n),
                            rng.uniform(-0.55 * BOX_L, 0.55 * BOX_L, n),
                            rng.uniform(-0.75 * BOX_H, -0.25 * BOX_H, n)])


def _interface_points(n=30):
    rng = np.random.default_rng(27182)
    def cloud(zlo, zhi):
        return np.column_stack([rng.uniform(-0.5 * BOX_L, 0.5 * BOX_L, n),
                                rng.uniform(-0.5 * BOX_L, 0.5 * BOX_L, n),
                                rng.uniform(zlo, zhi, n)])
    return (cloud(-0.40 * BOX_H, -0.10 * BOX_H),
            cloud(-0.90 * BOX_H, -0.60 * BOX_H))


def b_rigid_translation(box, obs):
    """u = const on a closed box: t == 0 and u == const, both exactly P0."""
    print("\n[B1] MANUFACTURED: rigid translation (data AND answer exact at P0)")
    keys = ("top", "sides", "base")
    for nu in NU_LIST:
        mat = material(nu)
        vals = {k: np.broadcast_to(RIGID_C, (box[k].n_triangles, 3))
                for k in keys}
        model, region = _build_box_model(
            box, mat, dict(top="d", sides="d", base="d"), vals)
        system = generate_system(model)
        scale = mat.mu * np.linalg.norm(RIGID_C) / BOX_L
        for eps, elabel in ((EPS_B, "eps"), (EPS_B_DICT, "eps{}")):
            for jump in JUMPS:
                sol = AssembledDense(system, eps, "direct", jump=jump).solve()
                tmax = max(np.abs(sol[f"t:{k}"]).max() for k in keys) / scale
                u = evaluate_displacement(model, region, sol, obs, eps,
                                          warn_near=False)
                tag = f"nu={nu:.2f} {elabel:5s} {jump:10s}"
                if jump == "calibrated":
                    # calibration makes the operator annihilate constants
                    # EXACTLY, so the recovered traction must be machine zero.
                    check(f"B1 t==0 (machine)                  {tag}",
                          tmax, TOL_RIGID_T)
                else:
                    # the half jump carries the Gauss-identity error at finite
                    # eps; not gated on t, only reported.
                    print(f"  [--] {'B1 t==0 (half jump: Gauss error, reported)':58s} "
                          f"{tmax:.3e}")
                check(f"B1 interior u == const               {tag}",
                      relerr(u, np.broadcast_to(RIGID_C, (obs.shape[0], 3))),
                      TOL_RIGID_U)


def b_uniform_strain_dirichlet(box, obs):
    """u = A x prescribed on the whole closed boundary; recover t and interior u."""
    print("\n[B2] MANUFACTURED: uniform strain, ALL-DIRICHLET")
    keys = ("top", "sides", "base")
    u_ex = {k: _exact_u(box[k].centroids()) for k in keys}
    masks = {k: _edge_distance(box[k]) > BOX_EDGE for k in keys}
    for nu in NU_LIST:
        mat = material(nu)
        t_ex = {k: _exact_t(box[k], mat) for k in keys}
        model, region = _build_box_model(
            box, mat, dict(top="d", sides="d", base="d"), u_ex)
        system = generate_system(model)
        for eps, elabel in ((EPS_B, "eps"), (EPS_B_DICT, "eps{}")):
            for jump in JUMPS:
                sol = AssembledDense(system, eps, "direct", jump=jump).solve()
                num = sum(float(np.sum((sol[f"t:{k}"][masks[k]]
                                        - t_ex[k][masks[k]]) ** 2))
                          for k in keys)
                den = sum(float(np.sum(t_ex[k][masks[k]] ** 2)) for k in keys)
                u = evaluate_displacement(model, region, sol, obs, eps,
                                          warn_near=False)
                tag = f"nu={nu:.2f} {elabel:5s} {jump:10s}"
                check(f"B2 interior u == A x                 {tag}",
                      relerr(u, _exact_u(obs)), TOL_DIR_U)
                # tractions within one element of a box EDGE are excluded: the
                # exact traction jumps there with the normal and P0 collocation
                # cannot resolve it.
                check(f"B2 recovered t == C:A (edges cut)    {tag}",
                      float(np.sqrt(num / den)), TOL_DIR_T, "{:.4f}")


def b_uniform_strain_mixed(box, obs):
    """Mixed BCs: exercises the PRESCRIBED-TRACTION RHS path all-Dirichlet skips."""
    print("\n[B3] MANUFACTURED: uniform strain, MIXED (base u given, rest t given)")
    base_mask = _edge_distance(box["base"]) > BOX_EDGE
    for nu in NU_LIST:
        mat = material(nu)
        vals = {"top": _exact_t(box["top"], mat),
                "sides": _exact_t(box["sides"], mat),
                "base": _exact_u(box["base"].centroids())}
        model, region = _build_box_model(
            box, mat, dict(top="t", sides="t", base="d"), vals)
        system = generate_system(model)
        for eps, elabel in ((EPS_B, "eps"), (EPS_B_DICT, "eps{}")):
            for jump in JUMPS:
                sol = AssembledDense(system, eps, "direct", jump=jump).solve()
                u = evaluate_displacement(model, region, sol, obs, eps,
                                          warn_near=False)
                tag = f"nu={nu:.2f} {elabel:5s} {jump:10s}"
                check(f"B3 interior u == A x                 {tag}",
                      relerr(u, _exact_u(obs)), TOL_MIX_U)
                check(f"B3 boundary u:top == A x             {tag}",
                      relerr(sol["u:top"], _exact_u(box["top"].centroids())),
                      TOL_MIX_UB, "{:.4f}")
                check(f"B3 boundary t:base == C:A n (edges cut) {tag}",
                      relerr(sol["t:base"][base_mask],
                             _exact_t(box["base"], mat)[base_mask]),
                      TOL_MIX_T, "{:.4f}")


def b_two_region_interface(obs_pair):
    """Same manufactured strain across a TWO-REGION model joined by an INTERFACE.

    No other solved-BVP check touches the INTERFACE branch of generate_system:
    the shared traction unknown t_p = sigma_stored * n_stored is seen by the two
    regions with OPPOSITE orientation signs, so a sign error there is invisible
    to any single-region model.  Both regions carry the same material, so the
    exact solution is still u = A x with continuous u and t across the seam.
    """
    print("\n[B4] MANUFACTURED: uniform strain across a TWO-REGION INTERFACE")
    z_mid = -0.5 * BOX_H
    xr, yr = (-BOX_L, BOX_L), (-BOX_L, BOX_L)

    def panels(zr):
        return _concatenate_meshes([
            make_vertical_panel_eq("x", xr[1], yr, zr, BOX_EDGE, +1),
            make_vertical_panel_eq("x", xr[0], yr, zr, BOX_EDGE, -1),
            make_vertical_panel_eq("y", yr[1], xr, zr, BOX_EDGE, +1),
            make_vertical_panel_eq("y", yr[0], xr, zr, BOX_EDGE, -1),
        ])

    top = make_rectangular_patch_eq(xr, yr, 0.0, BOX_EDGE, normal_up=True)
    base = make_rectangular_patch_eq(xr, yr, -BOX_H, BOX_EDGE, normal_up=False)
    iface = make_rectangular_patch_eq(xr, yr, z_mid, BOX_EDGE, normal_up=True)
    sides_u, sides_l = panels((z_mid, 0.0)), panels((-BOX_H, z_mid))
    rim = _rim_distance(iface) > BOX_EDGE
    obs_u, obs_l = obs_pair

    for nu in NU_LIST:
        mat = material(nu)
        # the interface Patch object must be SHARED by both regions (incidence
        # is by id()), which is exactly what this check is here to exercise
        p_if = Patch("iface", iface, BCType.INTERFACE)
        outer = [("top", top), ("sides_u", sides_u),
                 ("sides_l", sides_l), ("base", base)]
        pats = {n: Patch(n, m, BCType.PRESCRIBED_DISPLACEMENT,
                         value=_exact_u(m.centroids())) for n, m in outer}
        r_up = Region("upper", mat, [pats["top"], pats["sides_u"], p_if],
                      probe_point=np.array([1.0, 2.0, -0.25 * BOX_H]))
        r_lo = Region("lower", mat, [p_if, pats["sides_l"], pats["base"]],
                      probe_point=np.array([1.0, 2.0, -0.75 * BOX_H]))
        model = RegionModel([r_up, r_lo])
        system = generate_system(model)
        for jump in JUMPS:
            sol = AssembledDense(system, EPS_B, "direct", jump=jump).solve()
            u_up = evaluate_displacement(model, r_up, sol, obs_u, EPS_B,
                                         warn_near=False)
            u_lo = evaluate_displacement(model, r_lo, sol, obs_l, EPS_B,
                                         warn_near=False)
            tag = f"nu={nu:.2f} {jump:10s}"
            check(f"B4 upper-region interior u == A x   {tag}",
                  relerr(u_up, _exact_u(obs_u)), TOL_IFACE_U)
            check(f"B4 lower-region interior u == A x   {tag}",
                  relerr(u_lo, _exact_u(obs_l)), TOL_IFACE_U)
            check(f"B4 interface u == A x               {tag}",
                  relerr(sol["u:iface"], _exact_u(iface.centroids())),
                  TOL_IFACE_U)
            check(f"B4 interface t == C:A n (rim cut)   {tag}",
                  relerr(sol["t:iface"][rim], _exact_t(iface, mat)[rim]),
                  TOL_IFACE_T, "{:.4f}")


def main():
    t0 = time.time()
    print("=" * 76)
    print("Solved boundary-value problem: assembly -> BCs -> solve -> readout")
    print("=" * 76)

    meshes = build_fault_box(half_x=200.0, z_bottom=-120.0, fault_half_len=20.0,
                             fault_depth=18.0, edge_fault=3.0, edge_near=12.0,
                             edge_far=50.0, edge_side=50.0,
                             near_field_radius=80.0)
    nt = {k: meshes[k].n_triangles
          for k in ("top", "sides", "base", "fault")}
    print(f"  fault box triangles: {nt}")
    a_external_cutde(meshes)
    a_slip_sense(meshes)

    box = _closed_box()
    print(f"\n  closed box triangles: "
          f"{ {k: box[k].n_triangles for k in ('top', 'sides', 'base')} }")
    obs = _interior_points()
    b_rigid_translation(box, obs)
    b_uniform_strain_dirichlet(box, obs)
    b_uniform_strain_mixed(box, obs)
    b_two_region_interface(_interface_points())

    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    if all(CHECKS):
        print(f"PASS: solved BVP matches both anchors ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: solved BVP disagrees "
              f"({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")


if __name__ == "__main__":
    main()
