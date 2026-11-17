"""verify_interface_slip.py -- an interface that slips (``Patch.slip``).

Two regions moving past each other along the patch they share, where a fault
is slip inside one region. The displacement jumps by the Burgers vector, the
traction stays one unknown, and the region the patch's normal points into
sees ``u + b`` (``model/equations.py``). A solved slip's own rows are
algebraic -- the traction on an interface is an unknown already -- so it is
a FAULT_TRACTION fault in everything but its equations: a drag, a place in
the reduction, a law in ``mbem.stepper``.

The exact cases are two blocks of DIFFERENT materials, one on the other, the
lower one's base clamped and every side free. Slip on the plane between them
is then a rigid motion of the upper block, whatever the materials:

  [a] PRESCRIBED slip b: the upper block's free surface moves by b to
      rounding (5e-15), the lower block does not move, the plane carries no
      traction, and a readout inside each block returns b and 0 (1e-5: the
      mollified closure, the readout's own). The slot ``u`` of the plane is
      the -n side's: zero. The slip side is the region the normal points
      into.
  [b] SOLVED slip, the top face held displaced by U. Let go (no drag) the
      plane slips by exactly U and carries nothing (1e-14); held by an
      infinite drag it is the welded interface, the same model without the
      flag (1e-14); with a finite drag its rows hold to rounding, its slip
      has no normal part, and prescribing the slip it found returns the
      solution it came from (the round trip, 1e-12). TRIPWIRE: the dragging
      plane slips by a fraction of U that is neither 0 nor 1.
  [c] the reduction and the stepper take it as they take a fault: the
      reduced operator returns the dense backend's dragged solve (1e-13),
      its stiffness resists, and ``ViscousFault`` stepped in time is the
      matrix exponential of that operator (1e-6), ending at the free slip.
  [d] refusals: a slip on a patch that is not an interface, at an order
      above 0, of an unknown kind; the compressed backend; a slipping
      interface between two densities under gravity.

And the megathrust that needs it, ``layered_box`` with a slab and the fault
taken deeper than the plate: below the plate's base the megathrust runs down
the slab's upper face, which is an interface.

  [e] a slab of the mantle's own material is no slab, so slip PRESCRIBED on
      its face is the fault through a flat interface (``fault_lower`` as a
      FAULT of the substrate), two formulations with nothing in common but
      the kernels: the free surface within 2.1 % where the reach itself is
      20 % of it (tripwire), on this gate's coarse box (30 km interface
      elements; 0.9 % and 36 % with 15 km ones). SOLVED and let go, the
      reach slips the way the rupture above it did, in its plane (3e-15),
      by 0.24 of the rupture's slip, as the solved fault in the substrate
      does (0.24), and the two surfaces agree to 0.7 %. That the two means
      agree here is the mesh's doing: uniform slip ends in a line of
      unbounded traction, and what a freed reach does against it is set by
      the mollification width -- on the face the interface's (0.23 / 0.30 /
      0.34 of the slip at 3 / 1.5 / 0.75 km, whatever the element: 10, 7.5
      and 5 km agree to 3 %), on the fault its own (0.28 / 0.33 / 0.39 at
      0.35 / 1.5 / 3 km), measured on the finer box. Hence a band.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import mollified_bem as mb
import verify_viscoelastic as VV               # sibling: the slab's box
from local_box_mesh_eq import (_concatenate_meshes, make_rectangular_patch_eq,
                               make_vertical_panel_eq)
from mbem import rheology, stepper
from mbem.backends.dense import AssembledDense
from mbem.cases import layered_box
from mbem.evaluate import evaluate_displacement
from mbem.la.hop import require_compressible
from mbem.model import (BCType, Gravity, Patch, Region, RegionModel,
                        generate_system)

L, H, EDGE, EPS = 10.0, 6.0, 2.0, 0.2          # the blocks, km
UPPER = mb.ElasticMaterial(mu=30.0, lam=45.0)
LOWER = mb.ElasticMaterial(mu=12.0, lam=30.0)
B = np.array([1.0e-3, 0.4e-3, 0.0])            # a slip in the plane
U = np.array([2.0e-3, 0.0, 0.0])               # the top's displacement
KAPPA = 3.0                                    # GPa/km
TOL_EXACT = 1.0e-11      # a rigid block, the welded limit; 5e-15 .. 2e-14
TOL_READOUT = 1.0e-4     # the readout inside a block; 7e-6
TOL_ROUND = 1.0e-10      # the round trip; 6e-13
TOL_EXPM = 1.0e-5        # the stepper vs the matrix exponential; 1e-7
TOL_NULL = 0.03          # slab of the mantle's material: face vs fault; 0.021
REACH_THERE = 0.10       # ... what the reach is of the surface; 0.195
FREED = (0.15, 0.40)     # mean slip of a freed reach, of the slip; 0.238, 0.242
TOL_FREED_TOP = 0.05     # the two let-go models' surfaces; 0.007

CHECKS = []


def check(name, val, tol):
    ok = bool(np.isfinite(val)) and val < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3e} < {tol:.0e}")
    return ok


def check_in(name, val, lo, hi):
    ok = bool(np.isfinite(val)) and lo < val < hi
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.4f} in ({lo:g}, {hi:g})")
    return ok


def check_bool(name, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {note}")
    return ok


def refuses(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - b).max() / np.abs(b).max())


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def blocks(slip=None, value=None, top=None, gravity=None):
    """Two blocks, the plane z = -H between them an interface (``slip``)."""
    def sides(z0, z1):
        return _concatenate_meshes([
            make_vertical_panel_eq(ax, s * L, (-L, L), (z0, z1), EDGE, s)
            for ax in ("x", "y") for s in (+1, -1)])

    def flat(z, up):
        return make_rectangular_patch_eq((-L, L), (-L, L), z, EDGE,
                                         normal_up=up)
    mid = Patch("mid", flat(-H, True), BCType.INTERFACE, slip=slip, value=value)
    upper = Region("upper", UPPER, [
        Patch("top", flat(0.0, True), BCType.FREE_TRACTION if top is None
              else BCType.PRESCRIBED_DISPLACEMENT, value=top),
        Patch("sides_upper", sides(-H, 0.0), BCType.FREE_TRACTION), mid],
        probe_point=np.array([0.1, 0.2, -0.5 * H]))
    lower = Region("lower", LOWER, [
        mid, Patch("sides_lower", sides(-2.0 * H, -H), BCType.FREE_TRACTION),
        Patch("base", flat(-2.0 * H, False), BCType.PRESCRIBED_DISPLACEMENT)],
        probe_point=np.array([0.1, 0.2, -1.5 * H]))
    return RegionModel([upper, lower], gravity=gravity), mid


def dense(model, **kw):
    return quiet(AssembledDense, generate_system(model), EPS, "direct", **kw)


def a_prescribed():
    print("\n[a] prescribed slip between two blocks: the upper one moves, rigidly")
    model, mid = blocks("prescribed", B)
    check_bool("the slip side is the region the normal points into",
               model.slip_side(mid).name == "upper"
               and model.orientation(model.regions[0], mid) == -1)
    sol = dense(model).solve()
    nb = float(np.linalg.norm(B))
    check("the free surface of the upper block moves by b",
          float(np.abs(sol["u:top"] - B).max() / nb), TOL_EXACT)
    check("the plane's own u is the -n side's: the lower block's, zero",
          float(np.abs(sol["u:mid"]).max() / nb), TOL_EXACT)
    check("the plane carries no traction (GPa, of mu b / H)",
          float(np.abs(sol["t:mid"]).max() / (UPPER.mu * nb / H)), TOL_EXACT)
    x = np.array([[0.3, -0.2, -0.5 * H], [0.3, -0.2, -1.5 * H]])
    up = evaluate_displacement(model, model.regions[0], sol, x[:1], EPS,
                               warn_near=False)[0]
    lo = evaluate_displacement(model, model.regions[1], sol, x[1:], EPS,
                               warn_near=False)[0]
    check("a readout inside the upper block is b",
          float(np.abs(up - B).max() / nb), TOL_READOUT)
    check("... and inside the lower block nothing",
          float(np.abs(lo).max() / nb), TOL_READOUT)


def b_solved():
    print("\n[b] solved slip, the top face held displaced by U")
    model, mid = blocks("solved", top=U)
    asm = dense(model)
    nu = float(np.linalg.norm(U))
    free = asm.solve()
    check("let go, the plane slips by U", rel(free["b:mid"], np.tile(U, (
        free["b:mid"].shape[0], 1))), TOL_EXACT)
    check("... and carries nothing",
          float(np.abs(free["t:mid"]).max() / (UPPER.mu * nu / H)), TOL_EXACT)
    welded = dense(blocks(None, top=U)[0]).solve()
    held = asm.rebuild_for_materials({}, fault_drag={"mid": np.inf}).solve()
    check("an infinite drag is the welded interface: its displacement",
          rel(held["u:mid"], welded["u:mid"]), TOL_EXACT)
    check("... its traction", rel(held["t:mid"], welded["t:mid"]), TOL_EXACT)
    check_bool("... and no slip", not np.any(held["b:mid"]))
    drag = asm.rebuild_for_materials({}, fault_drag={"mid": KAPPA}).solve()
    t, b = drag["t:mid"], drag["b:mid"]
    check("a finite drag: its rows hold, t = kappa b in the plane",
          float(np.abs(t[:, :2] - KAPPA * b[:, :2]).max() / np.abs(t).max()),
          TOL_EXACT)
    check_bool("... the slip has no normal part", not np.any(b[:, 2]))
    check_in("... and the plane slips by a fraction of U (tripwire)",
             float(b[:, 0].mean() / U[0]), 0.1, 0.9)
    again = dense(blocks("prescribed", b, top=U)[0]).solve()
    check("round trip: that slip PRESCRIBED returns the plane's traction",
          rel(again["t:mid"], t), TOL_ROUND)
    check("... and the traction on the held top",
          rel(again["t:top"], drag["t:top"]), TOL_ROUND)
    return asm, drag


def c_reduction(asm, drag):
    print("\n[c] the reduction and the stepper take it as they take a fault")
    from scipy.linalg import expm
    c = asm.condense()
    beta = c.dragged(KAPPA)
    check("the reduced operator returns the dense backend's dragged slip",
          rel(c.cartesian(beta)["mid"], drag["b:mid"]), TOL_ROUND)
    check("... and its traction on the plane",
          rel(c.slots(c.x0 + c.X @ beta)["t:mid"], drag["t:mid"]), TOL_ROUND)
    check("its stiffness resists (largest real part of its eigenvalues)",
          float(np.linalg.eigvals(c.S).real.max()), 0.0)
    law = rheology.ViscousFault(tau=1.0, width=10.0)
    eta = LOWER.mu * law.tau / law.width
    body = stepper.Body([c])
    t = np.array([0.03, 0.3, 3.0, 300.0])
    h = stepper.march(body, stepper.law_rate(body, {"mid": law},
                                             {"mid": LOWER.mu}), t)
    end = -np.linalg.solve(c.S, c.L)
    exact = np.array([end - expm(c.S * tt / eta) @ end for tt in t])
    check("ViscousFault stepped is the matrix exponential of that operator",
          float(np.abs(h.slip - exact).max() / np.abs(end).max()), TOL_EXPM)
    check("... and ends at the free slip, U",
          rel(h.fault("mid")[-1], np.tile(U, (drag["b:mid"].shape[0], 1))),
          1.0e-6)


def d_refusals():
    print("\n[d] refusals")
    flat = make_rectangular_patch_eq((-L, L), (-L, L), 0.0, EDGE, normal_up=True)
    check_bool("a slip on a patch that is not an interface",
               refuses(lambda: Patch("p", flat, BCType.FREE_TRACTION,
                                     slip="solved"), ValueError))
    check_bool("... at an order above 0",
               refuses(lambda: Patch("p", flat, BCType.INTERFACE, order=1,
                                     slip="solved"), ValueError))
    check_bool("... of a kind there is not",
               refuses(lambda: Patch("p", flat, BCType.INTERFACE,
                                     slip="free"), ValueError))
    model, _mid = blocks("prescribed", B)
    check_bool("the compressed backend",
               refuses(lambda: require_compressible(model), NotImplementedError))
    heavy = blocks("solved", gravity=Gravity(
        density={"upper": 2700.0, "lower": 3300.0}, surface=("top",)))[0]
    check_bool("a slipping interface between two densities, under gravity",
               refuses(lambda: generate_system(heavy), NotImplementedError))
    level = blocks("solved", gravity=Gravity(
        density={"upper": 3000.0, "lower": 3000.0}, surface=("top",)))[0]
    check_bool("... and one density across it is taken",
               "b:mid" in dense(level).solve())


def e_slab_face():
    print("\n[e] a megathrust down a slab's upper face")
    geo = dict(VV.SLAB, fault_depth=50.0, slab_half_len=VV.SLAB["fault_half_len"],
               edge_lower=10.0)
    flat = {k: v for k, v in geo.items()
            if not k.startswith("slab") and k != "edge_lower"}
    plate = mb.ElasticMaterial(VV.MU, VV.LAM)

    def solve(g, lower):
        meshes = layered_box.build_layered_box(**g)
        model = layered_box.build_model(meshes, 1.0e-3, plate, plate,
                                        sides="clamped", rake=90.0, lower=lower)
        eps = dict(top="auto", sides_upper="auto", fault="auto",
                   interface=VV.SLAB_EPS, sides_lower=VV.SLAB_EPS,
                   base=VV.SLAB_EPS)
        if "fault_lower" in meshes:
            eps["fault_lower"] = VV.SLAB_EPS if g.get("slab_depth") else "auto"
        return meshes, model, quiet(AssembledDense, generate_system(model),
                                    eps, "direct").solve()

    mA, modA, sA = solve(geo, "slip")
    _m, _mod, sB = solve(flat, "slip")
    _m, _mod, sC = solve(dict(geo, fault_depth=-geo["z_interface"]), "slip")
    reach = next(p for p in modA.regions[0].patches if p.name == "fault_lower")
    check_bool("the reach is an interface of both regions, that slips",
               reach.bc is BCType.INTERFACE and reach.slip == "prescribed"
               and modA.slip_side(reach).name == layered_box.SUBSTRATE
               and all(any(p is reach for p in r.patches)
                       for r in modA.regions))
    x = mA["top"].vertices[mA["top"].triangles].mean(axis=1)
    near = (np.abs(x[:, 1]) < 40.0) & (np.abs(x[:, 0] - 40.0) < 120.0)

    def apart(a, b):
        return float(np.linalg.norm((a - b)[near]) / np.linalg.norm(b[near]))

    check("a slab of the mantle's material: slip on its face is the fault "
          "through", apart(sA["u:top"], sB["u:top"]), TOL_NULL)
    ok = apart(sA["u:top"], sC["u:top"]) > REACH_THERE
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {'... of which the reach itself is (tripwire)':70s} "
          f"{apart(sA['u:top'], sC['u:top']):9.3e} > {REACH_THERE:.0e}")

    mD, _mod, sD = solve(geo, "creep")
    mE, _mod, sE = solve(flat, "creep")
    dd = np.deg2rad(geo["dip"])
    up_dip = np.array([-np.cos(dd), 0.0, np.sin(dd)])
    normal = np.array([np.sin(dd), 0.0, np.cos(dd)])
    on_face = sD["b:fault_lower"] @ up_dip / 1.0e-3
    on_fault = sE["b:fault_lower"] @ up_dip / 1.0e-3
    check_in("let go, the reach on the face slips as the rupture did, by",
             float(on_face.mean()), *FREED)
    check_in("... the solved fault in the substrate by",
             float(on_fault.mean()), *FREED)
    check("... in its plane", float(np.abs(sD["b:fault_lower"] @ normal).max()
                                    / 1.0e-3 / on_face.mean()), 1.0e-10)
    check("... and the two surfaces agree",
          apart(sD["u:top"], sE["u:top"]), TOL_FREED_TOP)


def main():
    print("=" * 76)
    print("An interface that slips: prescribed, solved, reduced, stepped")
    print("=" * 76)
    a_prescribed()
    asm, drag = b_solved()
    c_reduction(asm, drag)
    d_refusals()
    e_slab_face()
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: interface slip ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: interface slip ({sum(1 for c in CHECKS if not c)} of "
              f"{len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
