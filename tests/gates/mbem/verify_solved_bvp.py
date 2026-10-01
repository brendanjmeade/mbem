"""Verify a SOLVED boundary-value problem end to end: assembly -> boundary
conditions -> solve -> readout.

Every other gate in ``verify/`` checks one link of the chain -- a kernel, an
assembler, a contraction, or an internal consistency identity.  None of them
closes the loop, so a defect in the *right-hand side* (a prescribed value on
the wrong side of the equation, a fault slip entering with the wrong sign) is
invisible: flipping the sign of the fault RhsTerm scale in
``mbem/model/equations.py::generate_system`` leaves every other msd gate
PASSing with byte-identical stdout.  This gate exists to close that hole.

It uses THREE anchors because they fail differently:

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
     A fourth (A4) is the FIRST ELEMENT ROW of on-fault elastic shear below
     the free surface against cutde's half-space stress: a P0 top cannot
     follow the slope of the surface displacement at the trace and the row
     is tens of percent high (the error is amplified by the image/total
     ratio ~D/2z); a P1 top (``build_model(order_top=1)``) at eps="auto"
     resolves the row below ~3 eps_top, and at eps_top = 0.025 h_top the
     whole row. Both the defect (tripwires) and the cure are gated.

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
     single-region model can probe.  B5 repeats the interface with a MATERIAL
     CONTRAST (bimaterial simple shear, exact and lam-independent): with equal
     materials B4 cannot tell which region's material an interface block was
     assembled with.  B6 splits the Anchor A fault box by a same-material
     interface below the fault tip and requires the two-region solve to
     reproduce the one-region solve (fault-in-multi-region assembly).  Tight
     enough to catch what anchor A's few-percent truncation floor would hide:
     BC application, the free term, the interface orientation sign, per-patch
     eps routing, per-region materials.

  C. INTERNAL, manufactured, ABSOLUTE FAULT AMPLITUDE.  A small fault inside
     the closed box; its full-space mollified field (msd's own kernels, which
     are gated separately against cutde and the legacy oracle) is prescribed
     on all six faces.  The fault term is then P0-exact and the boundary data
     exact, so the interior displacement must reproduce the full-space field
     with no truncation floor -- a few-percent AMPLITUDE error on the fault
     source, invisible to A's direction (cosine) and truncation-limited
     magnitude checks, shows up here at near full strength.

Seeded defects (one at a time, in a sandbox copy) and the anchors that fail:
  fault RhsTerm scale sign  (generate_system)        A only  (cos -> -1)
  fault RhsTerm scale x1.03 (generate_system)        C only  (A, B unchanged)
  lam/mu swap in t_coeffs   (basis.py)               first at nu=0.30
  LHS free term 1/2 -> 1/4  (dense backend)          A + B3
  prescribed-u RHS sign     (generate_system)        B only
  prescribed-u half term    (generate_system)        B only
  prescribed-t RHS sign     (generate_system)        B3 only
  orientation sign on G     (generate_system)        B2 + B3
  interface-only sigma      (generate_system)        B4 only
  interface G with the wrong region's material       B5 only
  fault RHS skipped on INTERFACE rows                B6 only

The anchors run at nu = 1/4 AND nu != 1/4 (0.30, 0.45) -- the lam/mu pairing
is invisible at nu = 1/4, which is how a swapped kernel shipped for months --
and under BOTH jump conventions ("half" and "calibrated").

All checks PASS/FAIL.
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import mollified_bem as mb                                        # noqa: E402
from mbem.cases.fault_box import build_fault_box, build_model               # noqa: E402
from local_box_mesh_eq import (                                   # noqa: E402
    _concatenate_meshes,
    make_rectangular_patch_eq,
    make_vertical_fault_eq,
    make_vertical_panel_eq,
)
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.evaluate import evaluate_displacement, evaluate_stress  # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.model import (                                          # noqa: E402
    BCType,
    Patch,
    Region,
    RegionModel,
    generate_system,
)
from tde_reference import classical_tde_stress                    # noqa: E402

CHECKS = []

MU = 30.0
NU_LIST = (0.25, 0.30, 0.45)      # 1/4 hides the lam/mu pairing; 0.30/0.45 do not
JUMPS = ("half", "calibrated")

# ---------------------------------------------------------------- anchor A --
SLIP = 0.01                        # km == 10 m
# Anchor A runs at eps="auto" (the production rule: 0.1 h on the boundary
# patches, one 0.07 min h on the fault); EPS_A is the scalar B6 uses.
EPS_A = 3.0                        # km, == the fault element size
FAULT_BOX_KW = dict(half_x=200.0, z_bottom=-120.0, fault_half_len=20.0,
                    fault_depth=18.0, edge_fault=3.0, edge_near=12.0,
                    edge_far=50.0, edge_side=50.0, near_field_radius=80.0)
# cutde carries slip as [strike, dip, tensile] in the per-triangle TDCS frame
# and the SAME Burgers-vector sign as a FAULT Patch.value (b = u(+n) - u(-n)),
# so the reference is fed the model's own slip with no factor: a sign fitted
# from the data, or written here by hand, would make the gate sign-blind,
# which is the very defect it exists to catch.
# Surface points within this many FAULT eps of the trace are mollification-
# limited (the smeared slip under-predicts the surface step there); at the
# production eps the band excludes only the x = 0 centroids beyond the fault
# ends, and the trace-adjacent elements sit within 1 % of cutde.
TRACE_EXCLUSION_EPS = 1.5

COS_TOL = 0.9998       # measured worst 1-cos 9.8e-5 (surface) / 8.0e-5 (interior);
#                        a sign flip drives the cosine negative
MED_TOL_SURF = 0.10    # measured worst 0.051 -- box truncation
MED_TOL_EVAL = 0.04    # measured worst 0.019
# A4, the first element row: fault elements with a vertex on z = 0 within
# Y_FIRST_ROW of mid-strike (the lateral tips are resolution-limited in every
# method), elastic shear n.sigma.s at their centroids (0.38-1.64 km deep on
# this Triangle mesh) vs cutde's half-space TDE, at ONE nu != 1/4. The top
# double layer resolves a centroid only beyond ~FIRST_ROW_EPS_DEPTH eps_top
# (eps_top = the "auto" eps of the top elements on the trace, 0.31 km here),
# so the row splits into a resolved part (z <= -3 eps_top, 7 of 14 centroids)
# and a mollification-limited part. Measured (nu = 0.30 / 0.25):
#   P0 top, auto:  whole-row median +0.27 / +0.28, resolved-part median
#                  +0.074 / +0.080 (max +0.13), shallow part +0.4 .. +3.6
#   P1 top, auto:  resolved part max |err| 0.031 / 0.035 (median 0.002 /
#                  0.004), shallow part +0.17 .. +3.0 (mollification-limited)
#   P1 top, eps_top = EPS_TOP_ROW h_top: whole row max |err| 0.10 at the
#                  shallowest centroid (4.9 eps_top deep), <= 0.03 elsewhere;
#                  a P0 top at 0.05 h_top stays at +0.57 (shallow median)
# Tolerances at ~1.5-2x the measurement; the P0 tripwires prove the check
# sees the defect (a top that followed the trace would pass P1 and fail them).
NU_FIRST_ROW = 0.30
Y_FIRST_ROW = 10.0
FIRST_ROW_EPS_DEPTH = 3.0
EPS_TOP_ROW = 0.025
TOL_ROW_P0 = 0.15          # tripwire: P0 whole-row median ABOVE this
TOL_ROW_P0_DEEP = 0.04     # tripwire: P0 resolved-part median ABOVE this
TOL_ROW_P1_DEEP = 0.06     # P1 auto, resolved part, max |rel err|
TOL_ROW_P1_SMALL = 0.15    # P1 at EPS_TOP_ROW, whole row, max |rel err|

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
Z_MID = -0.5 * BOX_H
TAU = 1.0e-3            # GPa == 1 MPa, the bimaterial shear stress (B5)
TOL_BIMAT_U = 3.0e-2    # measured worst 1.6e-2 (upper region, nu=0.45)
TOL_BIMAT_T = 8.0e-2    # measured worst 3.6e-2 (interface rim excluded)
Z_IFACE_A = -40.0       # B6 interface: below the 18 km fault tip, 80 km of
#                         lower region beneath it (a thin slab is ~10 % off)
EDGE_IFACE_A = 25.0
TOL_TRANSP_U = 1.5e-2   # measured worst 6.4e-3
TOL_TRANSP_S = 1.0e-2   # measured worst 4.8e-3
TOL_TRANSP_UIF = 2.5e-2  # measured worst 1.1e-2; dropping the fault term from
#                          the interface rows gives 0.58 while the two interior
#                          checks above barely move (the fault field is weak
#                          22 km below the tip), so this is the one that sees it

# ---------------------------------------------------------------- anchor C --
# a 10 x 10 km vertical fault (8 triangles) centred in the closed box, >= 15 km
# from every face; slip SLIP along strike.  The box is meshed at half of
# BOX_EDGE (eps/h kept at 0.125): the x1.03 seed moves interior u by a fixed
# ~2e-2 while the P0 floor halves, so the finer box doubles the margin.
FAULT_C = dict(strike_length=10.0, depth_range=(-25.0, -15.0), target_edge=5.0)
EDGE_C, EPS_C = 0.5 * BOX_EDGE, 0.5 * EPS_B
TOL_AMP_U = 9.0e-3      # measured worst 4.4e-3; a x1.03 fault source gives 2.2e-2
TOL_AMP_T = 0.17        # measured worst 0.084 (edges cut)


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


def material(nu, mu=MU):
    return mb.ElasticMaterial(mu=mu, lam=2.0 * mu * nu / (1.0 - 2.0 * nu))


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
    return np.einsum("oisc,sc->oi", mat, slip_tdcs)


def a_external_cutde(meshes):
    """Solved free-surface + interior displacement vs cutde half-space TDEs
    carrying the model's own Burgers vector."""
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
    eps = "auto"
    exclusion = TRACE_EXCLUSION_EPS * float(
        kb.resolve_patch_eps(eps, Patch("fault", meshes["fault"], BCType.FAULT))[0])

    top_c = meshes["top"].centroids()
    keep = np.abs(top_c[:, 0]) > exclusion
    obs_surf = top_c[keep].copy()
    obs_surf[:, 2] = -1.0e-6            # cutde wants z < 0; msd's are at z = 0

    rng = np.random.default_rng(20260918)
    obs_int = np.column_stack([rng.uniform(-60.0, 60.0, 80),
                               rng.uniform(-60.0, 60.0, 80),
                               rng.uniform(-45.0, -12.0, 80)])
    obs_int = obs_int[np.abs(obs_int[:, 0]) > exclusion]

    for nu in NU_LIST:
        mat = material(nu)
        model = build_model(meshes, SLIP, mat)
        region = model.regions[0]
        system = generate_system(model)
        b = region.faults[0].value_array()          # the fault's Burgers vector
        ref_surf = _cutde_halfspace_disp(obs_surf, fault, b, nu)
        ref_int = _cutde_halfspace_disp(obs_int, fault, b, nu)
        for jump in JUMPS:
            sol = AssembledDense(system, eps, "direct", jump=jump).solve()
            u_surf = sol["u:top"][keep]
            u_int = evaluate_displacement(model, region, sol, obs_int, eps,
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

    Fault normal n_hat = +x_hat, s_hat = +y_hat, and ``build_model`` gives the
    fault the RIGHT-lateral Burgers vector b = u(+n) - u(-n) = -SLIP * y_hat
    (``_fault_box.py``; msd's fault source ``-sigma H @ b`` produces exactly
    that jump): the block on the +x side must translate toward -y and the
    block on the -x side toward +y, each by a fraction of SLIP at the free
    surface.  This is the single statement the seeded RHS sign flip inverts,
    and it is frozen in the source below rather than read off the data.
    """
    print("\n[A3] ABSOLUTE slip sense at the free surface (no external ref)")
    n_hat = np.asarray(meshes["n_hat"], float)
    s_hat = np.asarray(meshes["s_hat"], float)
    assert np.allclose(n_hat, [1.0, 0.0, 0.0]) and np.allclose(s_hat, [0, 1, 0])
    eps = "auto"
    exclusion = TRACE_EXCLUSION_EPS * float(
        kb.resolve_patch_eps(eps, Patch("fault", meshes["fault"], BCType.FAULT))[0])
    c = meshes["top"].centroids()
    band = np.abs(c[:, 1]) < 6.0
    plus = band & (c[:, 0] > exclusion) & (c[:, 0] < 20.0)
    minus = band & (c[:, 0] < -exclusion) & (c[:, 0] > -20.0)

    for nu in NU_LIST:
        model = build_model(meshes, SLIP, material(nu))
        system = generate_system(model)
        for jump in JUMPS:
            u = AssembledDense(system, eps, "direct", jump=jump).solve()["u:top"]
            up = float(u[plus, 1].mean())
            um = float(u[minus, 1].mean())
            tag = f"nu={nu:.2f} {jump:10s}"
            check_true(f"A3 right-lateral: +x side -y, -x side +y {tag}",
                       up < 0.0 < um, f"u_y(+x)={up:+.3e} u_y(-x)={um:+.3e}")
            # near-antisymmetry about the fault plane
            check(f"A3 antisymmetry |sum|/|diff|         {tag}",
                  abs(up + um) / abs(up - um), 0.15, "{:.4f}")
            # the surface really moves an appreciable fraction of the slip
            frac = 0.5 * (abs(up) + abs(um)) / SLIP
            check_true(f"A3 surface |u_y| is 0.10-0.45 of slip {tag}",
                       0.10 < frac < 0.45, f"frac={frac:.3f}")


def _first_row(fault, y_max):
    """Centroids of the fault elements with a vertex on z = 0 and |y| < y_max,
    shallowest first."""
    c = fault.centroids()
    tv = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    sel = (tv[:, :, 2].max(axis=1) > -1e-9) & (np.abs(c[:, 1]) < y_max)
    obs = c[sel]
    return obs[np.argsort(-obs[:, 2])]


def _trace_eps_top(top, half_len):
    """Smallest "auto" eps among the top elements with an edge on the trace:
    0.1 x the trace spacing (edge_fault). Triangle also puts a few larger
    slivers on the trace; they do not set the band (centroids 2.3-2.7 eps of
    a 4.8-km trace element are within 3 % at P1)."""
    eps = kb.resolve_patch_eps("auto", top)
    tv = np.asarray(top.mesh.vertices, float)[np.asarray(top.mesh.triangles)]
    on = (np.abs(tv[:, :, 0]) < 1e-9) & (np.abs(tv[:, :, 1]) <= half_len + 1e-9)
    return float(eps[on.sum(axis=1) >= 2].min())


def a_first_row(meshes):
    """First element row of on-fault elastic shear vs cutde half-space stress
    (constants and measurements above): P0 tripwires, P1 cure."""
    print("\n[A4] FIRST ELEMENT ROW of on-fault elastic shear vs cutde half-space TDE")
    try:
        import cutde.halfspace  # noqa: F401
    except ImportError as exc:
        print(f"  (cutde not importable -- {exc}); external anchor FAILED")
        CHECKS.append(False)
        return
    fault = meshes["fault"]
    n_hat = np.asarray(meshes["n_hat"], float)
    s_hat = np.asarray(meshes["s_hat"], float)
    obs = _first_row(fault, Y_FIRST_ROW)
    mat = material(NU_FIRST_ROW)

    def shear(order_top, eps):
        model = build_model(meshes, SLIP, mat, order_top=order_top)
        region = model.regions[0]
        system = generate_system(model)
        sol = AssembledDense(system, eps, "direct", jump="calibrated").solve()
        # the row sits inside the near-boundary band by construction
        sig = evaluate_stress(model, region, sol, obs, eps,
                              subtract_anelastic=True, warn_near=False)
        return region, np.einsum("nij,i,j->n", sig, n_hat, s_hat)

    region, tau_p0 = shear(0, "auto")
    _, tau_p1 = shear(1, "auto")
    top = next(p for p in region.patches if p.name == "top")
    eps_small = {"top": EPS_TOP_ROW * kb.element_sizes(top.mesh),
                 "sides": "auto", "base": "auto", "fault": "auto"}
    _, tau_p1s = shear(1, eps_small)
    b = region.faults[0].value_array()
    ref = np.einsum("nij,i,j->n",
                    classical_tde_stress(obs, fault, b, mat.mu, mat.nu,
                                         halfspace=True), n_hat, s_hat)
    eps_top = _trace_eps_top(top, FAULT_BOX_KW["fault_half_len"])
    deep = obs[:, 2] <= -FIRST_ROW_EPS_DEPTH * eps_top
    r0, r1, r1s = tau_p0 / ref - 1.0, tau_p1 / ref - 1.0, tau_p1s / ref - 1.0
    print(f"  nu = {NU_FIRST_ROW}, {obs.shape[0]} centroids, trace eps_top = "
          f"{eps_top:.3f} km, resolved below {FIRST_ROW_EPS_DEPTH * eps_top:.2f} km "
          f"({int(deep.sum())} centroids); rel err of n.sigma.s vs half space")
    print(f"  {'z (km)':>8} {'hs (MPa)':>9} {'P0 auto':>9} {'P1 auto':>9} "
          f"{'P1 ' + str(EPS_TOP_ROW) + 'h':>10}")
    for i in range(obs.shape[0]):
        print(f"  {obs[i, 2]:8.3f} {ref[i] * 1e3:9.4f} {r0[i]:+9.3f} {r1[i]:+9.3f} "
              f"{r1s[i]:+10.3f}{'' if deep[i] else '   (mollification band)'}")
    check_true("A4 P0 top, auto: whole-row median ABOVE tripwire",
               float(np.median(r0)) > TOL_ROW_P0,
               f"{np.median(r0):+.3f} > {TOL_ROW_P0}")
    check_true("A4 P0 top, auto: resolved-part median ABOVE tripwire",
               float(np.median(r0[deep])) > TOL_ROW_P0_DEEP,
               f"{np.median(r0[deep]):+.3f} > {TOL_ROW_P0_DEEP}")
    check("A4 P1 top, auto: resolved part max |rel err|",
          float(np.abs(r1[deep]).max()), TOL_ROW_P1_DEEP, "{:.4f}")
    check(f"A4 P1 top, eps_top = {EPS_TOP_ROW} h: whole row max |rel err|",
          float(np.abs(r1s).max()), TOL_ROW_P1_SMALL, "{:.4f}")


# ============================================================== ANCHOR B =====

def _panels(zr, half=BOX_L, edge=BOX_EDGE):
    """The four vertical sides of a square box of half-width ``half``, over
    the depth range ``zr``, stored normals outward."""
    xr = yr = (-half, half)
    return _concatenate_meshes([
        make_vertical_panel_eq("x", xr[1], yr, zr, edge, +1),
        make_vertical_panel_eq("x", xr[0], yr, zr, edge, -1),
        make_vertical_panel_eq("y", yr[1], xr, zr, edge, +1),
        make_vertical_panel_eq("y", yr[0], xr, zr, edge, -1),
    ])


def _closed_box(edge=BOX_EDGE):
    xr, yr = (-BOX_L, BOX_L), (-BOX_L, BOX_L)
    return {
        "top": make_rectangular_patch_eq(xr, yr, 0.0, edge, normal_up=True),
        "base": make_rectangular_patch_eq(xr, yr, -BOX_H, edge,
                                          normal_up=False),
        "sides": _panels((-BOX_H, 0.0), edge=edge),
    }


def _split_box():
    """The closed box cut at Z_MID by a horizontal INTERFACE patch (normal up)."""
    xr, yr = (-BOX_L, BOX_L), (-BOX_L, BOX_L)
    return {
        "top": make_rectangular_patch_eq(xr, yr, 0.0, BOX_EDGE, normal_up=True),
        "base": make_rectangular_patch_eq(xr, yr, -BOX_H, BOX_EDGE,
                                          normal_up=False),
        "iface": make_rectangular_patch_eq(xr, yr, Z_MID, BOX_EDGE,
                                           normal_up=True),
        "sides_u": _panels((Z_MID, 0.0)),
        "sides_l": _panels((-BOX_H, Z_MID)),
    }


def _two_region_model(sbox, mat_up, mat_lo, u_up, u_lo):
    """Upper/lower regions of ``_split_box`` joined by ONE shared interface
    Patch (incidence is by id()); every outer face is Dirichlet with the
    region's exact displacement ``u_up(x)`` / ``u_lo(x)`` at its centroids."""
    p_if = Patch("iface", sbox["iface"], BCType.INTERFACE)

    def dirichlet(name, u_of):
        return Patch(name, sbox[name], BCType.PRESCRIBED_DISPLACEMENT,
                     value=u_of(sbox[name].centroids()))

    r_up = Region("upper", mat_up,
                  [dirichlet("top", u_up), dirichlet("sides_u", u_up), p_if],
                  probe_point=np.array([1.0, 2.0, -0.25 * BOX_H]))
    r_lo = Region("lower", mat_lo,
                  [p_if, dirichlet("sides_l", u_lo), dirichlet("base", u_lo)],
                  probe_point=np.array([1.0, 2.0, -0.75 * BOX_H]))
    return RegionModel([r_up, r_lo]), r_up, r_lo


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


def _build_box_model(box, mat, bcs, values, faults=()):
    patches = [Patch(k, box[k],
                     BCType.PRESCRIBED_DISPLACEMENT if bcs[k] == "d"
                     else BCType.FREE_TRACTION,
                     value=values[k])
               for k in ("top", "sides", "base")]
    region = Region("block", mat, patches,
                    probe_point=np.array([1.0, 2.0, -0.5 * BOX_H]),
                    faults=list(faults))
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


def b_two_region_interface(sbox, obs_pair):
    """Same manufactured strain across a TWO-REGION model joined by an INTERFACE.

    No other solved-BVP check touches the INTERFACE branch of generate_system:
    the shared traction unknown t_p = sigma_stored * n_stored is seen by the two
    regions with OPPOSITE orientation signs, so a sign error there is invisible
    to any single-region model.  Both regions carry the same material, so the
    exact solution is still u = A x with continuous u and t across the seam.
    """
    print("\n[B4] MANUFACTURED: uniform strain across a TWO-REGION INTERFACE")
    iface = sbox["iface"]
    rim = _rim_distance(iface) > BOX_EDGE
    obs_u, obs_l = obs_pair

    for nu in NU_LIST:
        mat = material(nu)
        model, r_up, r_lo = _two_region_model(sbox, mat, mat,
                                              _exact_u, _exact_u)
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


def _shear_u(mu):
    """u = (TAU/mu)(z - Z_MID) x_hat: simple shear with sigma_xz = TAU."""
    def u_of(x):
        u = np.zeros((len(x), 3))
        u[:, 0] = (TAU / mu) * (np.asarray(x, float)[:, 2] - Z_MID)
        return u
    return u_of


def _shear_t(mesh):
    n, _ = mesh.normals_and_areas()
    sig = np.zeros((3, 3))
    sig[0, 2] = sig[2, 0] = TAU
    return n @ sig.T


def b_bimaterial_shear(sbox, obs_pair):
    """B4's split box with mu_upper != mu_lower under uniform shear sigma_xz.

    Exact, lam-independent, u and t continuous across the seam.  With equal
    materials (B4) an interface block assembled with the WRONG region's
    material is invisible; here it is not.  Also run with the contrast
    inverted and unequal nu.
    """
    print("\n[B5] MANUFACTURED: bimaterial simple shear across the INTERFACE")
    iface = sbox["iface"]
    rim = _rim_distance(iface) > BOX_EDGE
    obs_u, obs_l = obs_pair
    cases = [(30.0, 10.0, nu, nu) for nu in NU_LIST] + [(10.0, 30.0, 0.45, 0.25)]
    for mu_up, mu_lo, nu_up, nu_lo in cases:
        model, r_up, r_lo = _two_region_model(
            sbox, material(nu_up, mu_up), material(nu_lo, mu_lo),
            _shear_u(mu_up), _shear_u(mu_lo))
        system = generate_system(model)
        for jump in JUMPS:
            sol = AssembledDense(system, EPS_B, "direct", jump=jump).solve()
            u_up = evaluate_displacement(model, r_up, sol, obs_u, EPS_B,
                                         warn_near=False)
            u_lo = evaluate_displacement(model, r_lo, sol, obs_l, EPS_B,
                                         warn_near=False)
            tag = f"mu {mu_up:.0f}/{mu_lo:.0f} nu {nu_up:.2f}/{nu_lo:.2f} {jump:10s}"
            check(f"B5 upper interior u == shear  {tag}",
                  relerr(u_up, _shear_u(mu_up)(obs_u)), TOL_BIMAT_U)
            check(f"B5 lower interior u == shear  {tag}",
                  relerr(u_lo, _shear_u(mu_lo)(obs_l)), TOL_BIMAT_U)
            check(f"B5 interface t == tau (rim cut) {tag}",
                  relerr(sol["t:iface"][rim], _shear_t(iface)[rim]),
                  TOL_BIMAT_T, "{:.4f}")


def b_fault_interface_transparency(meshes):
    """Anchor A's fault box cut by a same-material horizontal interface BELOW
    the fault tip: the two-region solve must reproduce the one-region solve
    (identical meshes on every matching face) in the fault region's interior
    displacement and elastic stress, and on the interface itself, where the
    one-region model is evaluated as an interior field far from all of its
    boundaries.  Pins fault-in-multi-region assembly, which B4/B5 (no fault)
    and A (one region) cannot.
    """
    print("\n[B6] TRANSPARENCY: fault box split by a same-material interface")
    kw = FAULT_BOX_KW
    sides_u = _panels((Z_IFACE_A, 0.0), kw["half_x"], kw["edge_side"])
    sides_l = _panels((kw["z_bottom"], Z_IFACE_A), kw["half_x"], kw["edge_side"])
    xr = meshes["x_range"]
    iface = make_rectangular_patch_eq(xr, xr, Z_IFACE_A, EDGE_IFACE_A,
                                      normal_up=True)
    fault = meshes["fault"]
    slip = np.broadcast_to(SLIP * np.asarray(meshes["s_hat"], float),
                           (fault.n_triangles, 3))
    # obs in the fault region: >= 3 eps from the fault plane, clear of the top
    # (near-field h = 12) and the interface (h = 25), inside |x|,|y| < 60
    rng = np.random.default_rng(16180)
    obs = np.column_stack([rng.uniform(-60.0, 60.0, 60),
                           rng.uniform(-60.0, 60.0, 60),
                           rng.uniform(-25.0, -8.0, 60)])
    obs = obs[np.abs(obs[:, 0]) > 3.0 * EPS_A]

    def patch(name, mesh, bc):
        return Patch(name, mesh, bc)

    for nu in (0.25, 0.30):
        mat = material(nu)
        one = RegionModel([Region(
            "crust", mat,
            [patch("top", meshes["top"], BCType.FREE_TRACTION),
             patch("sides", _concatenate_meshes([sides_u, sides_l]),
                   BCType.FREE_TRACTION),
             patch("base", meshes["base"], BCType.PRESCRIBED_DISPLACEMENT)],
            probe_point=np.array([90.0, 90.0, -60.0]),
            faults=[Patch("fault", fault, BCType.FAULT, value=slip)])])
        p_if = patch("iface", iface, BCType.INTERFACE)
        r_up = Region("upper", mat,
                      [patch("top", meshes["top"], BCType.FREE_TRACTION),
                       patch("sides_u", sides_u, BCType.FREE_TRACTION), p_if],
                      probe_point=np.array([90.0, 90.0, -20.0]),
                      faults=[Patch("fault", fault, BCType.FAULT, value=slip)])
        r_lo = Region("lower", mat,
                      [p_if, patch("sides_l", sides_l, BCType.FREE_TRACTION),
                       patch("base", meshes["base"],
                             BCType.PRESCRIBED_DISPLACEMENT)],
                      probe_point=np.array([90.0, 90.0, -80.0]))
        two = RegionModel([r_up, r_lo])
        sys_one, sys_two = generate_system(one), generate_system(two)
        for jump in JUMPS:
            s1 = AssembledDense(sys_one, EPS_A, "direct", jump=jump).solve()
            s2 = AssembledDense(sys_two, EPS_A, "direct", jump=jump).solve()
            u1 = evaluate_displacement(one, one.regions[0], s1, obs, EPS_A,
                                       warn_near=False)
            u2 = evaluate_displacement(two, r_up, s2, obs, EPS_A,
                                       warn_near=False)
            sg1 = evaluate_stress(one, one.regions[0], s1, obs, EPS_A,
                                  warn_near=False)
            sg2 = evaluate_stress(two, r_up, s2, obs, EPS_A, warn_near=False)
            u1_if = evaluate_displacement(one, one.regions[0], s1,
                                          iface.centroids(), EPS_A,
                                          warn_near=False)
            tag = f"nu={nu:.2f} {jump:10s}"
            check(f"B6 fault-region u: 2 == 1 region    {tag}",
                  relerr(u2, u1), TOL_TRANSP_U)
            check(f"B6 fault-region sigma: 2 == 1 region {tag}",
                  relerr(sg2, sg1), TOL_TRANSP_S)
            check(f"B6 interface u == 1-region field    {tag}",
                  relerr(s2["u:iface"], u1_if), TOL_TRANSP_UIF)


# ============================================================== ANCHOR C =====

def c_fault_amplitude():
    """Full-space field of a small interior fault prescribed on the closed box.

    The reference is msd's own fault kernel with the boundary densities zero
    (separately gated against cutde and the legacy oracle).  Solving with that
    displacement on all six faces must give it back inside: the fault term is
    P0-exact and the data exact, so only the P0 error of the smooth boundary
    integral remains.  A fault source scaled by (1 + a) leaves the interior
    field short by a * u_D, u_D being the interior extension of the face data,
    so the obs cloud sits 1.2-2.8 h inside the side faces, where u_D is of the
    order of the fault field and the amplitude error shows at near full
    strength (and the cloud is automatically far from the fault).
    """
    print("\n[C] MANUFACTURED: full-space fault field prescribed on the closed box")
    box = _closed_box(EDGE_C)
    fault, _, s_hat = make_vertical_fault_eq(**FAULT_C)
    slip = np.broadcast_to(SLIP * s_hat, (fault.n_triangles, 3))
    keys = ("top", "sides", "base")
    rng = np.random.default_rng(31415)
    obs = np.column_stack([rng.uniform(-34.0, 34.0, 600),
                           rng.uniform(-34.0, 34.0, 600),
                           rng.uniform(-32.0, -8.0, 600)])
    obs = obs[np.max(np.abs(obs[:, :2]), axis=1) > 26.0][:80]
    masks = {k: _edge_distance(box[k]) > EDGE_C for k in keys}
    zero = {f"t:{k}": np.zeros((box[k].n_triangles, 3)) for k in keys}

    def fpatch():
        return Patch("fault", fault, BCType.FAULT, value=slip)

    for nu in (0.25, 0.30):
        mat = material(nu)
        ref, r_ref = _build_box_model(box, mat, dict(top="d", sides="d", base="d"),
                                      {k: None for k in keys}, faults=[fpatch()])
        u_ref = {k: evaluate_displacement(ref, r_ref, zero, box[k].centroids(),
                                          EPS_C, warn_near=False) for k in keys}
        t_ref = {k: np.einsum("nij,nj->ni",
                              evaluate_stress(ref, r_ref, zero,
                                              box[k].centroids(), EPS_C,
                                              warn_near=False),
                              box[k].normals_and_areas()[0]) for k in keys}
        u_ref_obs = evaluate_displacement(ref, r_ref, zero, obs, EPS_C,
                                          warn_near=False)
        model, region = _build_box_model(
            box, mat, dict(top="d", sides="d", base="d"), u_ref,
            faults=[fpatch()])
        system = generate_system(model)
        for jump in JUMPS:
            sol = AssembledDense(system, EPS_C, "direct", jump=jump).solve()
            u = evaluate_displacement(model, region, sol, obs, EPS_C,
                                      warn_near=False)
            num = sum(float(np.sum((sol[f"t:{k}"][masks[k]]
                                    - t_ref[k][masks[k]]) ** 2)) for k in keys)
            den = sum(float(np.sum(t_ref[k][masks[k]] ** 2)) for k in keys)
            tag = f"nu={nu:.2f} {jump:10s}"
            check(f"C interior u == full-space field   {tag}",
                  relerr(u, u_ref_obs), TOL_AMP_U)
            check(f"C face t == full-space t (edges cut) {tag}",
                  float(np.sqrt(num / den)), TOL_AMP_T, "{:.4f}")


def main():
    t0 = time.time()
    print("=" * 76)
    print("Solved boundary-value problem: assembly -> BCs -> solve -> readout")
    print("=" * 76)

    meshes = build_fault_box(**FAULT_BOX_KW)
    nt = {k: meshes[k].n_triangles
          for k in ("top", "sides", "base", "fault")}
    print(f"  fault box triangles: {nt}")
    a_external_cutde(meshes)
    a_slip_sense(meshes)
    a_first_row(meshes)

    box = _closed_box()
    print(f"\n  closed box triangles: "
          f"{ {k: box[k].n_triangles for k in ('top', 'sides', 'base')} }")
    obs = _interior_points()
    b_rigid_translation(box, obs)
    b_uniform_strain_dirichlet(box, obs)
    b_uniform_strain_mixed(box, obs)
    sbox, obs_pair = _split_box(), _interface_points()
    b_two_region_interface(sbox, obs_pair)
    b_bimaterial_shear(sbox, obs_pair)
    b_fault_interface_transparency(meshes)
    c_fault_amplitude()

    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    if all(CHECKS):
        print(f"PASS: solved BVP matches all anchors ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: solved BVP disagrees "
              f"({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
