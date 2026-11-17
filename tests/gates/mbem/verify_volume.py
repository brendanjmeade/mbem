"""Gate: the volume sampler evaluates the same field, everywhere it claims to.

The field itself is already gated -- ``verify_evaluate_stress`` against cutde
and the legacy oracle, ``verify_solved_bvp`` against manufactured solutions,
``verify_eigenstress_exact`` against two independent closed forms. What is NEW
here is a layer of bookkeeping between a grid and those evaluators, and every
part of it fails quietly rather than loudly:

  a  the chunk loop must be transparent. A serial loop over observation points
     is the only safe shape (rule 9: the contraction kernels are parallel=True
     and no *_contract_serial variants exist), and a chunked result that merely
     LOOKS right is the failure mode -- so it is compared bitwise against one
     unchunked call. A SECTION (``section_grid``, ``mbem sample --section``)
     is one plane of the same lattice and must be that plane of the volume.
  b  region classification decides which material and which patch set a point
     is evaluated with. Ask for the host field inside the inclusion and the
     answer is wrong, plausible, and unremarked by anything in the evaluator.
     Gated against the solid-angle closure identity, against the analytic
     cylinder of the topo_inclusion model, and -- because the classifier reads
     the WARPED top mesh -- against points above and below a topographic hill.
     A point ON a patch has no solid angle to go by, and is the first
     region's that owns the patch, by rule.
  c  MANUFACTURED: u = A x prescribed on a closed box has a known interior
     field, a known CONSTANT stress C:A, and a known constant strain A_sym. So
     the whole volume path is checked against arithmetic rather than against
     itself: displacement, stress, the Hooke inversion and the derived scalars
     all have exact answers here.
  d  the eigenstress returned by ``parts=True`` is the term actually subtracted
     -- the one-pass path must agree with the two separate calls it replaces.
  e  the .vti round-trips, because a writer nobody reads back is a writer that
     is wrong in a way no test can see.
  f  the near-field metric is the solver's own: ``clearance_h`` must reproduce
     ``evaluate._warn_near_boundary``'s d/h and compare against the existing
     NEAR_BOUNDARY_H_RATIO, not a second threshold.
  g  the website's uint8 encoding is lossy by a known amount. The payload the
     site renders cannot crash on a bad encoding, it just draws the wrong
     numbers convincingly -- so the error is pinned, and the two encodings
     someone might "simplify" to are shown losing 99.7 % and 54.7 %.
  h  a MATERIAL STATE is read back with its own moduli. A solution solved
     after ``rebuild_for_materials`` carries that state's densities, and the
     evaluators take the coefficients from the model they are handed -- so
     through the base model the two come from different states and the field
     is wrong by tens of per cent, with nothing to flag it. ``mbem sample``
     did exactly that for every non-base state. Manufactured again, because
     u = A x is exact at ANY moduli: the state readout must return A x and
     C_state:A, the base-model readout of the same solution must MISS, and
     the function the CLI resolves a stored state's model with is pinned.
  i  a Laplace run in TIME reaches a browser as modes and tables. Time is a
     map whose weights sum to ~200 in magnitude, so what a reader may combine
     is not the states: it is an orthonormal basis of them, with the weights
     already applied here. The decode a reader does must reproduce
     ``Prony``'s value, rate and locked-fault velocity; the obvious payload
     -- each state quantised, combined in the page -- is shown losing it; and
     the sheet a viewer warps must not average a displacement across the
     fault it jumps on.
  j  NEAR A FACE the readout resolves the element-constant displacement it
     integrates: every element edge is a small dislocation line, and within
     half an element the stress is tens of per cent wrong with nothing to say
     so. ``recover.NearField`` reads the near pairs from a continuous
     displacement recovered from the same values, at a narrowed width, and
     leaves the traction and the solve alone. Each of the three is held to
     the exact field of the manufactured box, with what it replaces shown
     missing: the element values, the narrowed width WITHOUT the recovery
     (worse than doing nothing), a fit ACROSS a fault. And the bookkeeping:
     far pairs untouched bitwise, chunks transparent, a point in what is left
     of the mollified layer read from just inside it.

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import sys
import tempfile
import time

import numpy as np

from local_box_mesh_eq import (_concatenate_meshes,
                              make_rectangular_patch_eq,
                              make_vertical_panel_eq)
from mbem import ElasticMaterial, cli, defaults, volume as V, vti
from mbem.backends.dense import AssembledDense
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.geometry import distance_to_mesh, solid_angle_batch
from mbem.kernels import basis as kb
from mbem.model import BCType, Patch, Region, RegionModel, generate_system

CHECKS: list = []

# The manufactured box: same shape as verify_solved_bvp's [B2], restated here
# because a gate is not a library. A is asymmetric so a transposed strain or a
# lam/mu swap cannot hide, and nu != 1/4 for the same reason (rule 7).
BOX_L, BOX_H, BOX_EDGE = 40.0, 40.0, 10.0
EPS_BOX = 1.25                                   # eps/h = 0.125
# tr(A) MUST be non-zero. With a trace-free A, C:A = lam tr(A) I + 2 mu A
# collapses to 2 mu A, so "sigma == C:A" would pass without lam entering it at
# all, and the dilatation reference would be 0 -- a relative error against
# nothing. Asymmetric so a transposed strain cannot hide either.
STRAIN_A = 1.0e-4 * np.array([[2.0, -1.0, 0.5],
                              [-1.0, -3.0, 1.5],
                              [0.5, 1.5, 3.0]])
NU = 0.30

# Measured, then pinned with headroom -- see the commit message.
TOL_CHUNK = 0.0          # bitwise: a chunked sum must be the same sum
# The reference here is the weak side: C:eps* is ~1e-3 of the total stress,
# so "total - elastic" is a difference of two nearly-equal float64 numbers and
# loses ~3 digits. 4.6e-13 measured; the one-pass value is the ACCURATE one.
TOL_PARTS = 1.0e-11
TOL_MANU_U = 1.0e-2      # interior u vs A x (P0 boundary integral)
TOL_MANU_S = 5.0e-2      # interior sigma vs C:A
TOL_MANU_E = 5.0e-2      # strain vs A_sym, and dilatation vs tr A
TOL_CLEAR = 1.0e-12      # clearance_h vs the evaluator's own d/h
TOL_WEB_LOG = 3.0e-2     # log10-uint8 over 5 decades; measured 2.4 %
TOL_WEB_MODES = 5.0e-3   # [i] decode vs Prony, of the quantity's peak; the
#                          rate is the worst, 2.6e-3 on the stored megathrust
# [j] depths below the free top, in elements, and what is asked there.
# Measured: with the recovered readout the worst point is 1.2e-2 in stress
# and 1.3e-2 in displacement (the solve's own floor); the element-constant
# readout is 0.19 / 0.11 / 0.033 rms in stress at these depths.
NEAR_DEPTHS = (0.1, 0.2, 0.5)
TOL_NEAR_S = 3.0e-2      # stress vs C:A, worst point, recovered readout
TOL_NEAR_U = 2.5e-2      # displacement vs A x, of the plane's peak
MISS_NEAR_S = 0.10       # the element-constant readout at 0.1 h must miss
TOL_RECOVER = 1.0e-10    # a quadratic through the recovery
# [h]: a second material for the same box, softer and at another nu, so that
# both kernels move (the slip kernel sees nu only, the force kernel 1/mu).
STATE_MU, STATE_NU = 12.0, 0.18
MISS_STATE = 0.10        # the base-model readout must be at least this wrong


def check(name, ok, extra="") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:58s} {extra}")
    return ok


def close(name, got, tol, extra="") -> bool:
    return check(name, got <= tol, f"{got:.3e} <= {tol:.0e} {extra}")


def relerr(a, b) -> float:
    b = np.asarray(b, float)
    return float(np.linalg.norm(np.asarray(a, float) - b) /
                 max(np.linalg.norm(b), 1e-300))


# ---------------------------------------------------------------- fixtures --

def _closed_box(edge=BOX_EDGE):
    """Top, base and the four outward-facing sides of a square box."""
    xr = (-BOX_L, BOX_L)
    return {
        "top": make_rectangular_patch_eq(xr, xr, 0.0, edge, normal_up=True),
        "base": make_rectangular_patch_eq(xr, xr, -BOX_H, edge,
                                          normal_up=False),
        "sides": _concatenate_meshes([
            make_vertical_panel_eq("x", xr[1], xr, (-BOX_H, 0.0), edge, +1),
            make_vertical_panel_eq("x", xr[0], xr, (-BOX_H, 0.0), edge, -1),
            make_vertical_panel_eq("y", xr[1], xr, (-BOX_H, 0.0), edge, +1),
            make_vertical_panel_eq("y", xr[0], xr, (-BOX_H, 0.0), edge, -1),
        ]),
    }


def _cut_sheet():
    """``(vertices, triangles, fault)``: a flat square of P0 elements, cut
    for half its length by a vertical fault under the line x = 0."""
    from mbem import TriMesh
    n = 8
    g = np.linspace(-1.0, 1.0, 2 * n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    v = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    idx = np.arange(X.size).reshape(X.shape)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tri = np.concatenate([np.column_stack([a.ravel(), b.ravel(), c.ravel()]),
                          np.column_stack([a.ravel(), c.ravel(), d.ravel()])])
    fault = TriMesh(np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0],
                              [0.0, 0.5, -1.0], [0.0, -0.5, -1.0]]),
                    np.array([[0, 1, 2], [0, 2, 3]]))
    return v, tri, fault


def _free_top():
    """The manufactured box with its top FREE under the exact traction of
    u = A x, so that the top's displacement is SOLVED: the density the
    readout near a face recovers. Returns (model, region, sol)."""
    box = _closed_box()
    mat = ElasticMaterial(mu=30.0, lam=2.0 * 30.0 * NU / (1.0 - 2.0 * NU))
    sig = mat.lam * np.trace(STRAIN_A) * np.eye(3) \
        + mat.mu * (STRAIN_A + STRAIN_A.T)
    patches = [Patch("top", box["top"], BCType.FREE_TRACTION,
                     value=sig[:, 2])]                 # C:A on the normal +z
    patches += [Patch(k, box[k], BCType.PRESCRIBED_DISPLACEMENT,
                      value=np.asarray(box[k].centroids(), float) @ STRAIN_A.T)
                for k in ("sides", "base")]
    region = Region("block", mat, patches,
                    probe_point=np.array([1.0, 2.0, -0.5 * BOX_H]))
    model = RegionModel([region])
    sol = AssembledDense(generate_system(model), EPS_BOX, "direct",
                         jump="calibrated").solve()
    return model, region, sol


def _manufactured():
    """Closed box, u = A x on every face. Returns (model, region, sol, eps,
    asm); the assembly is kept for [h], which rebuilds it at a second material."""
    box = _closed_box()
    # lam from (mu, nu) once, here, to set up the manufactured case; the
    # solver and the sampler only ever see (mu, lam). Rule 8 forbids a
    # 1/(1-2nu) INSIDE the kernels, not in a gate's own data.
    mat = ElasticMaterial(mu=30.0, lam=2.0 * 30.0 * NU / (1.0 - 2.0 * NU))
    vals = {k: np.asarray(box[k].centroids(), float) @ STRAIN_A.T
            for k in box}
    patches = [Patch(k, box[k], BCType.PRESCRIBED_DISPLACEMENT, value=vals[k])
               for k in ("top", "sides", "base")]
    region = Region("block", mat, patches,
                    probe_point=np.array([1.0, 2.0, -0.5 * BOX_H]))
    model = RegionModel([region])
    asm = AssembledDense(generate_system(model), EPS_BOX, "direct",
                         jump="calibrated")
    return model, region, asm.solve(), EPS_BOX, asm


# ------------------------------------------------------------------ clauses --

def a_chunking(model, region, sol, eps) -> None:
    """[a] the chunk loop is transparent, bitwise."""
    print("\n[a] the serial chunk loop changes nothing")
    grid = V.make_grid(model, 8.0)
    code = V.classify(model, grid)
    inside = np.flatnonzero(code > 0)
    check("the grid has interior points", inside.size > 200, f"{inside.size}")

    one = V.sample(model, sol, eps, grid, code, chunk=10 ** 9)
    many = V.sample(model, sol, eps, grid, code, chunk=137)
    for key in ("u", "sigma", "strain", "eigenstress"):
        a, b = one[key][inside], many[key][inside]
        check(f"{key:12s} chunked == unchunked (bitwise)",
              np.array_equal(a, b),
              "" if np.array_equal(a, b) else f"max {np.abs(a - b).max():.2e}")
    # and against the evaluator called directly, with no sampler in the way
    pts = grid.points[inside]
    u_direct = evaluate_displacement(model, region, sol, pts, eps,
                                     warn_near=False)
    s_direct = evaluate_stress(model, region, sol, pts, eps, warn_near=False)
    close("u    == evaluate_displacement", relerr(one["u"][inside], u_direct),
          TOL_CHUNK)
    close("sigma == evaluate_stress", relerr(one["sigma"][inside], s_direct),
          TOL_CHUNK)
    check("outside points are NaN, not 0",
          bool(np.all(np.isnan(one["u"][code == 0]))) if (code == 0).any()
          else True)

    # A SECTION is one plane of the same lattice, sampled by the same code:
    # at a plane the volume grid has, it must be that plane of the volume.
    y0 = float(np.unique(grid.points[:, 1])[2])
    sec = V.section_grid(model, "y", y0, 8.0)
    on = np.flatnonzero(grid.points[:, 1] == y0)
    check("a section is one plane of the volume's lattice",
          sec.dims[1] == 1 and sec.points.shape[0] == on.size
          and np.array_equal(sec.points, grid.points[on]),
          f"{sec.dims}")
    cut = V.sample(model, sol, eps, sec, V.classify(model, sec))
    for key in ("u", "sigma"):
        check(f"{key:12s} on the section == in the volume (bitwise)",
              np.array_equal(cut[key], one[key][on], equal_nan=True))
    lo, _hi = V.model_bounds(model)
    refused = 0
    for axis, at in (("y", float(lo[1])), ("w", 0.0)):
        try:
            V.section_grid(model, axis, at, 8.0)
        except ValueError:
            refused += 1
    check("a section in a face of the body, or across no axis, is refused",
          refused == 2)


def b_classification() -> None:
    """[b] which region a point is in -- closure identity, cylinder, hill."""
    print("\n[b] region classification")
    from mbem.cases.registry import topo_inclusion_meshes, topo_inclusion_model
    bundle = topo_inclusion_meshes()
    model = topo_inclusion_model(bundle)
    names = [r.name for r in model.regions]
    check("topo_inclusion has host and inclusion",
          names == ["host", "inclusion"], f"{names}")

    rng = np.random.default_rng(2718)
    pts = np.column_stack([rng.uniform(-200, 200, 4000),
                           rng.uniform(-200, 200, 4000),
                           rng.uniform(-200, 2, 4000)])
    inc = model.point_in_region("inclusion", pts)
    host = model.point_in_region("host", pts)

    # The inclusion of this case IS a cylinder: centre (-100, 100), radius 75,
    # depth 50. An analytic test is legitimate HERE, as an independent oracle
    # for a geometry we happen to know; the classifier itself stays general.
    r2 = (pts[:, 0] + 100.0) ** 2 + (pts[:, 1] - 100.0) ** 2
    analytic = (r2 < 75.0 ** 2) & (pts[:, 2] > -50.0) & (pts[:, 2] < 0.0)
    # points within one element of the cylinder wall, its base or z=0 are
    # ambiguous by solid angle and excluded, exactly as validate() excludes
    # them with a distance test.
    skin = 8.0
    clear = (np.abs(np.sqrt(r2) - 75.0) > skin) & \
            (np.abs(pts[:, 2] + 50.0) > skin) & (np.abs(pts[:, 2]) > skin)
    bad = int((inc[clear] != analytic[clear]).sum())
    check("inclusion == analytic cylinder (skin cut)", bad == 0,
          f"{bad} of {int(clear.sum())} disagree")
    check("no point is in both regions", not bool((inc & host).any()))
    check("the inclusion is actually found", int(inc.sum()) > 50,
          f"{int(inc.sum())} points")

    # TOPOGRAPHY, free: the classifier reads the warped top mesh, so a point
    # under the hill but above z=0 is inside and one above the hill is not.
    # No z > h(x,y) test anywhere -- that would be a second statement of the
    # surface.
    top = next(p for p in model.regions[0].patches if p.name == "host_top")
    zmax = float(np.asarray(top.mesh.vertices)[:, 2].max())
    check("the top mesh really is warped above z=0", zmax > 1.0,
          f"max z = {zmax:.2f} km")
    v = np.asarray(top.mesh.vertices, float)
    summit = v[np.argmax(v[:, 2])]
    below = summit + np.array([0.0, 0.0, -2.0])
    above = summit + np.array([0.0, 0.0, +2.0])
    check("a point just under the hill crest is inside",
          bool(model.point_in_region("host", below[None])[0]))
    check("a point just above the hill crest is outside",
          not bool(model.point_in_region("host", above[None])[0]))

    # NO SAMPLE MAY LIE ON A PATCH. A point on the surface has a genuinely
    # ambiguous solid angle (2 pi, not 0 or 4 pi), so its side is decided by
    # floating point -- measured 149 of 400 samples on one box face landing
    # "inside" before make_grid was offset by half a cell. The symptom was a
    # ragged one-cell jitter along every axis-aligned face, which looks like a
    # renderer bug and is not one.
    from mbem.cases.registry import topo_inclusion_meshes as _tim  # noqa: F401
    grid = V.make_grid(model, 8.0, pad=8.0)
    worst = np.inf
    for rr in model.regions:
        for pp in rr.patches:
            dd, _ = distance_to_mesh(grid.points, pp.mesh)
            worst = min(worst, float(dd.min()))
    check("no grid sample lies ON a patch", worst > 1e-6,
          f"closest {worst:.4f} km")

    # ... AND ONE THAT DOES IS SETTLED BY RULE. The half cell keeps samples
    # off the OUTER faces; it cannot keep them off an interior one, and at
    # 4 km -- the website's own spacing -- a whole plane of samples lies on
    # the inclusion's base. The closure sum there is 0 or 4 pi as rounding
    # has it, point by point: the site drew the patchwork as a speckled
    # disk. ``classify`` gives such a point to the first region that owns
    # the patch, and an outer face keeps its own points.
    fine = V.make_grid(model, 4.0, pad=4.0)
    code = V.classify(model, fine)
    base = next(p for p in model.regions[1].patches
                if p.name == "interface_bot")
    on = V.on_patch(fine.points, base)
    check("at 4 km a plane of samples lies ON the inclusion's base",
          int(on.sum()) > 100, f"{int(on.sum())} samples")
    alone = model.point_in_region("host", fine.points[on])
    check("every one of them is the host's: the first region that owns it",
          bool(np.all(code[on] == 1)),
          f"the closure identity alone calls {int(alone.sum())} of "
          f"{int(on.sum())} the host's")
    for name, want in (("host_top", 1), ("interface_side", 1),
                       ("inclusion_top", 2)):
        pp = next(p for r in model.regions for p in r.patches
                  if p.name == name)
        cen = np.asarray(pp.mesh.centroids(), float)   # ON their elements
        got = V.classify(model, V.Grid(origin=(0.0,) * 3, spacing=(1.0,) * 3,
                                       dims=(cen.shape[0], 1, 1), points=cen))
        check(f"a point ON {name} is region {want}, every one",
              bool(np.all(got == want)),
              f"{int((got != want).sum())} of {cen.shape[0]} otherwise")
    # On an axis-aligned body the interior mask must be a clean rectangle: a
    # ragged row width is the tie above, showing up as geometry.
    code2 = V.classify(model, grid)
    mid = code2.reshape(grid.shape)[grid.shape[0] // 2] > 0
    rows = mid.sum(axis=1)
    occ = rows[rows > 0]
    check("the interior mask is not ragged", occ.min() == occ.max(),
          f"row widths {occ.min()}..{occ.max()}")

    # And the identity itself, read raw: 4*pi inside, 0 outside.
    inside_pt = np.array([[0.0, 0.0, -100.0]])
    s_in = sum(model.orientation(model.regions[0], p)
               * solid_angle_batch(p.mesh, inside_pt)[0]
               for p in model.regions[0].patches)
    s_out = sum(model.orientation(model.regions[0], p)
                * solid_angle_batch(p.mesh, np.array([[0.0, 0.0, 500.0]]))[0]
                for p in model.regions[0].patches)
    close("closure sum == 4 pi at an interior point",
          abs(s_in - 4.0 * np.pi), 1e-6)
    close("closure sum == 0 far outside", abs(s_out), 1e-6)


def c_manufactured(model, region, sol, eps) -> None:
    """[c] u = A x: the interior field, stress and strain are all known."""
    print("\n[c] MANUFACTURED uniform strain: exact u, sigma AND strain")
    mat = region.material
    grid = V.make_grid(model, 5.0)
    code = V.classify(model, grid)
    c_h, _, _ = V.clearance(model, grid, eps)
    f = V.sample(model, sol, eps, grid, code)

    # Interior AND clear of the boundary: the P0 density limits the
    # representation within ~0.5 h of a patch, which is what c_h measures and
    # what NEAR_BOUNDARY_H_RATIO is set from.
    sel = np.flatnonzero((code > 0) & (c_h > 2.0 * defaults.NEAR_BOUNDARY_H_RATIO))
    check("clear interior points to test on", sel.size > 100, f"{sel.size}")
    pts = grid.points[sel]

    sig_ex = mat.lam * np.trace(STRAIN_A) * np.eye(3) + 2.0 * mat.mu * STRAIN_A
    e_ex = 0.5 * (STRAIN_A + STRAIN_A.T)
    n = sel.size
    close("u      == A x", relerr(f["u"][sel], pts @ STRAIN_A.T), TOL_MANU_U)
    close("sigma  == C:A (a constant tensor)",
          relerr(f["sigma"][sel], np.broadcast_to(sig_ex, (n, 3, 3))),
          TOL_MANU_S)
    close("strain == sym(A)  [the Hooke inversion, vs arithmetic]",
          relerr(f["strain"][sel], np.broadcast_to(e_ex, (n, 3, 3))),
          TOL_MANU_E)
    close("dilatation == tr A",
          relerr(f["dilatation"][sel],
                 np.full(n, float(np.trace(STRAIN_A)))), TOL_MANU_E)
    dev = sig_ex - np.trace(sig_ex) / 3.0 * np.eye(3)
    close("von_mises == sqrt(3/2 s:s) of C:A",
          relerr(f["von_mises"][sel],
                 np.full(n, float(np.sqrt(1.5 * np.sum(dev * dev))))),
          TOL_MANU_S)
    close("u_mag == |A x|",
          relerr(f["u_mag"][sel], np.linalg.norm(pts @ STRAIN_A.T, axis=1)),
          TOL_MANU_U)
    # Hooke must also close on itself, which catches a (mu, lam) mix-up that
    # a constant-field test could absorb: 2 mu e + lam tr(e) I == sigma.
    e = f["strain"][sel]
    back = 2.0 * mat.mu * e + mat.lam * np.trace(
        e, axis1=1, axis2=2)[:, None, None] * np.eye(3)
    close("2 mu e + lam tr(e) I == sigma (machine)",
          relerr(back, f["sigma"][sel]), 1e-12)


def d_eigenstress(model, region, sol, eps) -> None:
    """[d] the one-pass eigenstress is the term actually subtracted."""
    print("\n[d] parts=True returns what subtract_anelastic removes")
    rng = np.random.default_rng(161803)
    pts = np.column_stack([rng.uniform(-0.5 * BOX_L, 0.5 * BOX_L, 200),
                           rng.uniform(-0.5 * BOX_L, 0.5 * BOX_L, 200),
                           rng.uniform(-0.8 * BOX_H, -0.2 * BOX_H, 200)])
    el = evaluate_stress(model, region, sol, pts, eps, True, False)
    tot = evaluate_stress(model, region, sol, pts, eps, False, False)
    el2, eig = evaluate_stress(model, region, sol, pts, eps, True, False,
                               parts=True)
    check("parts=True leaves the elastic field bitwise unchanged",
          np.array_equal(el, el2))
    close("eigenstress == total - elastic", relerr(eig, el - tot), TOL_PARTS)
    check("the eigenstress is not identically zero",
          float(np.abs(eig).max()) > 0.0, f"max {np.abs(eig).max():.3e}")


def e_vti_roundtrip(model, region, sol, eps) -> None:
    """[e] the writer is readable, and reads back what went in."""
    print("\n[e] .vti round trip")
    import xml.dom.minidom
    grid = V.make_grid(model, 10.0)
    code = V.classify(model, grid)
    c_h, c_eps, c_fault = V.clearance(model, grid, eps)
    f = V.sample(model, sol, eps, grid, code)
    arrays = V.as_vti_arrays(grid, f, code, c_h, c_eps, c_fault)
    with tempfile.TemporaryDirectory(prefix="mbem_vti_") as tmp:
        p = vti.write(f"{tmp}/v.vti", grid.origin, grid.spacing, grid.dims,
                      arrays)
        back = vti.read(p)
        check("dims survive", back["dims"] == grid.dims, f"{back['dims']}")
        check("origin survives", back["origin"] == grid.origin)
        check("spacing survives", back["spacing"] == grid.spacing)
        check("every array survives", set(back["arrays"]) == set(arrays),
              f"{len(arrays)} arrays")
        # equal_nan: points outside every region are NaN BY DESIGN, and
        # NaN != NaN would report a correct round trip as broken.
        bad = [k for k, a in arrays.items()
               if not np.array_equal(back["arrays"][k],
                                     np.asarray(a, dtype="<f4"),
                                     equal_nan=True)]
        check("float32 round trip is bitwise", not bad, f"{bad[:3]}")
        head = p.read_bytes().split(b"  <AppendedData")[0]
        xml.dom.minidom.parseString(head + b"</VTKFile>")
        check("the header is well-formed XML", True)
        check("a 3x3 tensor becomes 6 named components",
              all(f"sigma_{c}" in arrays for c, _i, _j in V.SYM6))

        # INDEPENDENT reader. Everything above proves vti.read inverts
        # vti.write, which is self-consistency, not correctness: a wrong
        # header_type, extent convention or appended-offset base would be
        # symmetric in both and invisible. VTK is the reference implementation
        # and the only thing that settles whether a viewer can open the file.
        # vtk is a `[viz]` extra and this clause FAILS without it rather than
        # skipping, as the cutde gates do.
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        rd = vtk.vtkXMLImageDataReader()
        rd.SetFileName(str(p))
        rd.Update()
        img = rd.GetOutput()
        check("VTK reads it without error", rd.GetErrorCode() == 0,
              f"error code {rd.GetErrorCode()}")
        check("VTK agrees on dims, origin, spacing",
              img.GetDimensions() == grid.dims
              and img.GetOrigin() == grid.origin
              and img.GetSpacing() == grid.spacing,
              f"{img.GetDimensions()} {img.GetOrigin()}")
        pdata = img.GetPointData()
        check("VTK sees every array", pdata.GetNumberOfArrays() == len(arrays),
              f"{pdata.GetNumberOfArrays()} of {len(arrays)}")
        wrong = []
        for i in range(pdata.GetNumberOfArrays()):
            arr = pdata.GetArray(i)
            ours = back["arrays"][arr.GetName()]
            if not np.array_equal(vtk_to_numpy(arr).reshape(ours.shape), ours,
                                  equal_nan=True):
                wrong.append(arr.GetName())
        check("VTK's values == ours, array for array", not wrong,
              f"{wrong[:3]}")


def f_clearance_is_the_solver_s(model, region, sol, eps) -> None:
    """[f] clearance_h is the evaluator's own d/h, not a second metric."""
    print("\n[f] the near-field metric is the one already in defaults")
    grid = V.make_grid(model, 7.0)
    c_h, c_eps, c_fault = V.clearance(model, grid, eps)
    # Recompute exactly as evaluate._warn_near_boundary does: min over
    # BOUNDARY patches of d / element_size.
    ref = np.full(grid.points.shape[0], np.inf)
    for p in region.patches:
        d, idx = distance_to_mesh(grid.points, p.mesh)
        ref = np.minimum(ref, d / kb.element_sizes(p.mesh)[idx])
    # clearance_h is BOUNDARY-ONLY by construction, exactly as
    # _warn_near_boundary is, so this is an identity on any model rather than a
    # coincidence on a fault-free one.
    close("clearance_h == _warn_near_boundary's d/h",
          float(np.abs(c_h - ref).max()), TOL_CLEAR)
    check("clearance_eps differs from clearance_h (eps is not h)",
          float(np.abs(c_eps - c_h).max()) > 1.0,
          f"max |diff| {np.abs(c_eps - c_h).max():.1f}")
    on = int((c_h < defaults.NEAR_BOUNDARY_H_RATIO).sum())
    check("some grid points fall inside the flagged band", on > 0,
          f"{on} of {grid.points.shape[0]} below "
          f"NEAR_BOUNDARY_H_RATIO={defaults.NEAR_BOUNDARY_H_RATIO}")
    check("the band is flagged, never blanked (no NaN from clearance)",
          not bool(np.isnan(c_h).any() or np.isnan(c_eps).any()))


def g_webexport() -> None:
    """[g] the website's quantisation is lossy by a KNOWN amount, not silently.

    The payload the site renders is uint8, and a wrong encoding does not crash:
    it draws a plausible picture of the wrong numbers. The fields span five
    decades, which is the whole reason the encoding is log10 -- linear uint8
    loses 99.7 % and even linear uint16 loses 39-54 %, measured. So the gate
    pins the error of the encoder that is actually used and demonstrates that
    the alternatives are not usable, in case someone ever "simplifies" it.
    """
    print("\n[g] the web payload's encoding")
    from mbem import webexport as W
    rng = np.random.default_rng(20261002)
    v = 10.0 ** rng.uniform(-6.0, -0.8, 20000)       # ~5.2 decades, as measured
    v[::11] = np.nan                                  # outside the body

    buf, meta = W.encode_log_u8(v)
    back = W.decode_log_u8(buf, meta)
    good = np.isfinite(v)
    rel = float(np.abs(back[good] - v[good]).max() / 1.0) if False else float(
        np.max(np.abs(back[good] - v[good]) / v[good]))
    close("log10_u8 round trip", rel, TOL_WEB_LOG)
    check("absent samples stay absent (NaN -> 0 -> NaN)",
          np.array_equal(np.isnan(back), np.isnan(v)))
    check("no sample collides with the no-data level",
          not bool((np.frombuffer(buf, np.uint8)[good] == W.NODATA).any()))

    # The two encodings that would look reasonable and destroy the data.
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    for name, levels in (("linear uint8", 255), ("linear uint16", 65535)):
        q = np.round((v[good] - lo) / (hi - lo) * levels)
        bad = float(np.max(np.abs(q / levels * (hi - lo) + lo - v[good]) / v[good]))
        check(f"{name} would lose the small values", bad > 0.3,
              f"max rel err {bad * 100:.1f}%")

    c = np.where(rng.random(5000) < 0.1, np.nan, rng.uniform(0.0, 5.0, 5000))
    buf, meta = W.encode_linear_u8(c, 0.0, W.CLEARANCE_CLAMP)
    back = W.decode_linear_u8(buf, meta)
    inside = np.isfinite(c) & (c <= W.CLEARANCE_CLAMP)
    close("linear_u8 within one level",
          float(np.abs(back[inside] - c[inside]).max()),
          W.CLEARANCE_CLAMP / (W.LEVELS - 1))
    check("values above the clamp saturate, never wrap",
          bool(np.all(back[np.isfinite(c) & (c > W.CLEARANCE_CLAMP)]
                      >= W.CLEARANCE_CLAMP - 1e-9)))

    codes = np.where(rng.random(4000) < 0.2, np.nan, rng.integers(0, 3, 4000))
    buf, meta = W.encode_codes_u8(codes)
    q = np.frombuffer(buf, np.uint8)
    check("codes survive exactly", np.array_equal(
        q[np.isfinite(codes)], codes[np.isfinite(codes)].astype(np.uint8)))
    check("codes declare a NEAREST sampler", meta["filter"] == "nearest",
          "interpolating a region code invents regions")


def i_time_payload() -> None:
    """[i] a Laplace run through time, as a reader decodes it."""
    print("\n[i] the web payload in time: modes and tables")
    from mbem import TriMesh, laplace, webexport as W
    rng = np.random.default_rng(20261116)

    # A stack with the structure of a real one: a few relaxation times inside
    # the window, amplitudes falling, and noise at the level a compressed
    # solve leaves. u~(s) = a0 + sum_j a_j s / (s + 1/tau_j).
    s = np.array(laplace.s_grid(0.1, 100.0))
    rows = np.concatenate([[np.inf], s])
    taus, amp = (0.4, 2.5, 17.0, 60.0), (1.0, 0.3, 0.1, 0.03, 0.01)
    fields = rng.normal(size=(len(amp), 600, 3)) * np.array(amp)[:, None, None]
    with np.errstate(invalid="ignore"):
        basis = np.column_stack([np.ones(rows.size)] + [
            1.0 / (1.0 + 1.0 / (rows * tau)) for tau in taus])
    stack = np.tensordot(basis, fields, axes=1)
    stack += 1.0e-8 * rng.normal(size=stack.shape)

    modes, C = W.encode_modes(stack)
    check("the stack is a few modes, stored as float32",
          modes.dtype == np.float32 and modes.shape[0] < stack.shape[0]
          and modes.shape[1:] == stack.shape[1:],
          f"{modes.shape[0]} of {stack.shape[0]}")
    flat = modes.reshape(modes.shape[0], -1).astype(float)
    close("modes are orthonormal",
          float(np.abs(flat @ flat.T - np.eye(flat.shape[0])).max()), 1.0e-6)

    periods = (5.0, 20.0)
    tables = W.time_tables(s, C, periods=periods)
    pr = laplace.Prony(s)
    elastic = np.eye(1 + s.size)[0]
    check("the times stay inside the fit's window",
          all(tb["at"][-1] <= laplace.window(s)[1] * (1.0 + 1e-12)
              and tb["at"][tb["at"] > 0.0][0] >= laplace.window(s)[0]
              * (1.0 - 1e-12)
              for tb in tables.values() if tb["axis"] == "time"))
    worst = {}
    for name, tb in tables.items():
        at = tb["at"]
        if name in ("total", "since"):
            w = np.vstack([elastic, pr.weights(at[1:])])
            w = w - elastic if name == "since" else w
            check(f"{name}: t = 0 is the elastic state", at[0] == 0.0)
        elif name == "rate":
            w = pr.rate_weights(at)
        else:
            w = pr.locked_rate_weights(at * tb["period"], tb["period"])
        want = np.tensordot(w, stack, axes=1)
        got = W.decode_modes(modes, tb["rows"])
        worst[name] = float(np.abs(got - want).max() / np.abs(want).max())
        close(f"decode == Prony: {name}", worst[name], TOL_WEB_MODES)
    check("a table per period asked for",
          {f"locked:{p:g}" for p in periods} <= set(tables))

    # What would look reasonable and is not: quantise each STATE as the other
    # payload does, and let the page apply the weights.
    lo, hi = stack.min(), stack.max()
    q = np.round((stack - lo) / (hi - lo) * 255.0) / 255.0 * (hi - lo) + lo
    t = tables["since"]["at"][1:]
    want = np.tensordot(pr.weights(t) - elastic, stack, axes=1)
    bad = float(np.abs(np.tensordot(pr.weights(t) - elastic, q, axes=1)
                       - want).max() / np.abs(want).max())
    check("states quantised to uint8 and combined would lose it", bad > 0.3,
          f"max err {bad * 100:.0f}% of the peak, weights sum to "
          f"{np.abs(pr.weights(t)).sum(axis=1).max():.0f}")

    v, tri, fault = _cut_sheet()
    cen = v[tri].mean(axis=1)
    jump = np.sign(cen[:, 0])
    on = (np.abs(v[:, 0]) < 1e-12) & (np.abs(v[:, 1]) <= 0.5 + 1e-12)

    nodes, tri2, A = W.node_sheet(v, tri, [fault])
    check("a node on the fault is split, and only there",
          nodes.shape[0] == v.shape[0] + int(on.sum()),
          f"{v.shape[0]} -> {nodes.shape[0]} nodes")
    check("the sheet's triangles are the mesh's",
          np.array_equal(nodes[tri2], v[tri]))
    close("a constant stays constant",
          float(np.abs(A @ np.ones(tri.shape[0]) - 1.0).max()), 1.0e-14)
    val = A @ jump
    split = (np.abs(nodes[:, 0]) < 1e-12) & (np.abs(nodes[:, 1]) <= 0.5 + 1e-12)
    check("across the fault each side keeps its own value",
          bool(np.all(np.abs(np.abs(val[split]) - 1.0) < 1e-14))
          and int((val[split] > 0).sum()) == int((val[split] < 0).sum()))
    _n1, _t1, A1 = W.node_sheet(v, tri)
    check("without the fault the jump is averaged away",
          float(np.abs(A1 @ jump)[on].max()) < 0.5)
    # the same sheet as two pieces that share no vertices along x = 0.5
    left = cen[:, 0] < 0.5
    order = np.r_[np.flatnonzero(left), np.flatnonzero(~left)]
    piece = np.concatenate([tri[left], tri[~left] + v.shape[0]])
    n3, _t3, A3 = W.node_sheet(np.vstack([v, v]), piece, [fault])
    smooth = cen[:, 1] + 2.0 * jump          # any field, the jump included
    check("pieces that only touch are one surface",
          n3.shape[0] == nodes.shape[0]
          and np.allclose(np.sort(A3 @ smooth[order]), np.sort(A @ smooth),
                          atol=1e-12),
          f"{2 * v.shape[0]} vertices -> {n3.shape[0]} nodes")
    seg = W.boundary_segments(fault.vertices, fault.triangles)
    check("a fault's outline is its four edges", seg.shape == (4, 2, 3))


def j_near_a_face() -> None:
    """[j] the readout near a face: a continuous displacement, the traction
    and the solve untouched."""
    print("\n[j] the readout NEAR A FACE")
    from mbem import TriMesh
    from mbem.recover import NearField, recovery_operator

    # -- the recovery, on a sheet a fault cuts ---------------------------------
    v, tri, fault = _cut_sheet()
    mesh = TriMesh(v, tri)
    tv = v[tri]
    cen = tv.mean(axis=1)
    # the P2 nodes, element-major: v1 v2 v3 m12 m23 m31
    x = np.concatenate([tv, 0.5 * (tv + tv[:, [1, 2, 0]])], axis=1)
    side = np.repeat(np.sign(cen[:, 0]), 6)
    x = x.reshape(-1, 3)

    def quad(y):
        return (1.0 + 0.3 * y[:, 0] - 0.2 * y[:, 1] + 0.5 * y[:, 0] ** 2
                - 0.4 * y[:, 0] * y[:, 1] + 0.7 * y[:, 1] ** 2)

    R = recovery_operator(mesh, [fault])
    check("one row per P2 node of every element", R.shape == (6 * len(tri),
                                                             len(tri)))
    close("a constant stays constant",
          float(np.abs(R @ np.ones(len(tri)) - 1.0).max()), 1.0e-12)
    close("a quadratic field comes back exactly",
          float(np.abs(R @ quad(cen) - quad(x)).max()), TOL_RECOVER)
    # two rings from a node this far inside the trace never reach its ends
    on = (np.abs(x[:, 0]) < 1e-12) & (np.abs(x[:, 1]) <= 0.25 + 1e-12)
    jumped = R @ (quad(cen) + 2.0 * np.sign(cen[:, 0]))
    close("on the fault each side keeps its own field: nothing is fitted "
          "across", float(np.abs(jumped - quad(x) - 2.0 * side)[on].max()),
          TOL_RECOVER)
    plain = recovery_operator(mesh) @ (quad(cen) + 2.0 * np.sign(cen[:, 0]))
    check("without the fault the jump is fitted away",
          float(np.abs(plain - quad(x) - 2.0 * side)[on].min()) > 0.5,
          f"{float(np.abs(plain - quad(x) - 2.0 * side)[on].min()):.2f} of 2")
    wavy = R @ np.sin(3.0 * cen[:, 0] + 2.0 * cen[:, 1])
    _u, inv = np.unique(np.round(np.column_stack([x, side]) * 1e9)
                        .astype(np.int64), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    lo = np.full(inv.max() + 1, np.inf)
    hi = np.full(inv.max() + 1, -np.inf)
    np.minimum.at(lo, inv, wavy)
    np.maximum.at(hi, inv, wavy)
    off = np.ones(inv.max() + 1, bool)
    off[inv[np.abs(x[:, 0]) < 1e-12]] = False       # the line x = 0 apart
    check("continuous: elements that share a node read one value there",
          float((hi - lo)[off].max()) == 0.0, f"{(hi - lo)[off].max():.1e}")

    # -- the box, read near its free top ----------------------------------------
    model, region, sol = _free_top()
    mat = region.material
    eps, h = EPS_BOX, BOX_EDGE
    near = NearField(model)
    check("every boundary patch of order 0 is recovered",
          all(near.recovers(q) for q in region.patches))
    sig_ex = mat.lam * np.trace(STRAIN_A) * np.eye(3) \
        + mat.mu * (STRAIN_A + STRAIN_A.T)
    g = np.arange(-25.0, 25.0 + 1e-9, 2.5)
    X, Y = np.meshgrid(g, g, indexing="ij")

    def plane(depth):
        return np.column_stack([X.ravel(), Y.ravel(),
                                np.full(X.size, -depth)])

    def miss(pts, near=None, eps=eps):
        """(stress rms, stress worst, displacement worst) against the exact
        field, the stress relative to C:A and u to the plane's peak."""
        s = evaluate_stress(model, region, sol, pts, eps, warn_near=False,
                            near=near)
        u = evaluate_displacement(model, region, sol, pts, eps,
                                  warn_near=False, near=near)
        es = np.linalg.norm((s - sig_ex).reshape(-1, 9), axis=1) \
            / np.linalg.norm(sig_ex)
        ue = pts @ STRAIN_A.T
        eu = np.linalg.norm(u - ue, axis=1) / np.abs(ue).max()
        return float(np.sqrt((es ** 2).mean())), float(es.max()), \
            float(eu.max())

    for d in NEAR_DEPTHS:
        _rms, worst, worst_u = miss(plane(d * h), near)
        close(f"{d:.1f} h below the top: sigma == C:A", worst, TOL_NEAR_S)
        close(f"{d:.1f} h below the top: u == A x", worst_u, TOL_NEAR_U)
    at = plane(NEAR_DEPTHS[0] * h)
    as_now = miss(at)
    check("the element-constant readout MISSES there", as_now[0] > MISS_NEAR_S,
          f"rms {as_now[0]:.3f}, worst {as_now[1]:.3f}")
    narrow = miss(at, eps=defaults.READOUT_EPS_FACTOR * eps)
    check("the narrowed width WITHOUT the recovery is worse than neither",
          narrow[0] > as_now[0], f"rms {narrow[0]:.3f} > {as_now[0]:.3f}")

    # -- the bookkeeping ----------------------------------------------------------
    # with a radius that cannot reach it, a point is read as before, bitwise
    mid = np.array([[2.0, -3.0, -0.5 * BOX_H]])
    short = NearField(model, radius_h=0.5)
    same = all(np.array_equal(f(model, region, sol, mid, eps, warn_near=False),
                              f(model, region, sol, mid, eps, warn_near=False,
                                near=short))
               for f in (evaluate_displacement, evaluate_stress))
    check("a pair beyond the radius is read as before (bitwise)", same)
    moved = relerr(evaluate_stress(model, region, sol, mid, eps,
                                   warn_near=False, near=near),
                   evaluate_stress(model, region, sol, mid, eps,
                                   warn_near=False))
    close("two elements from every face the readout barely moves", moved,
          1.0e-2)

    grid = V.make_grid(model, 8.0)
    code = V.classify(model, grid)
    inside = np.flatnonzero(code > 0)
    one = V.sample(model, sol, eps, grid, code, chunk=10 ** 9, near=near)
    many = V.sample(model, sol, eps, grid, code, chunk=137, near=near)
    for key in ("u", "sigma", "eigenstress"):
        check(f"{key:12s} chunked == unchunked with the near readout "
              f"(bitwise)", np.array_equal(one[key][inside],
                                           many[key][inside]))
    # where the eigenstress of the NARROWED layer is large: one width deep.
    # Held as elastic == total + eigenstress, relative to the stress: the
    # difference total - elastic is the weak side ([d]) and weaker here.
    thin = plane(defaults.READOUT_EPS_FACTOR * eps)
    el = evaluate_stress(model, region, sol, thin, eps, True, False,
                         near=near)
    tot = evaluate_stress(model, region, sol, thin, eps, False, False,
                          near=near)
    el2, eig = evaluate_stress(model, region, sol, thin, eps, True, False,
                               parts=True, near=near)
    check("parts=True leaves the near field bitwise unchanged",
          np.array_equal(el, el2))
    close("the eigenstress returned is the one removed, near pairs included",
          relerr(el, tot + eig), 1.0e-12)
    check("... and it is the narrowed layer's: larger than the stress there",
          float(np.abs(eig).max()) > float(np.abs(el).max()),
          f"{np.abs(eig).max():.2e} against {np.abs(el).max():.2e}")

    # a point in what is left of the mollified layer is read from just inside
    gap = defaults.READOUT_STANDOFF_EPS * defaults.READOUT_EPS_FACTOR * eps
    skin = plane(0.01 * gap)
    out = V.stand_off(model, region, skin, eps, near)
    check("a point in the layer is moved to the stand-off, along the normal",
          np.allclose(out[:, 2], -gap, atol=1e-12)
          and np.array_equal(out[:, :2], skin[:, :2]),
          f"{gap:.3f} km below the top")
    deep = plane(2.0 * gap)
    check("a point outside the layer is not moved (bitwise)",
          np.array_equal(V.stand_off(model, region, deep, eps, near), deep))
    sk = V.Grid(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),
                dims=(skin.shape[0], 1, 1), points=skin)
    f = V.sample(model, sol, eps, sk, np.ones(skin.shape[0], np.int8),
                 near=near)
    es = np.linalg.norm((f["sigma"] - sig_ex).reshape(-1, 9), axis=1) \
        / np.linalg.norm(sig_ex)
    close("sampled ON the top: sigma == C:A", float(es.max()), TOL_NEAR_S)
    f0 = V.sample(model, sol, eps, sk, np.ones(skin.shape[0], np.int8))
    es0 = np.linalg.norm((f0["sigma"] - sig_ex).reshape(-1, 9), axis=1) \
        / np.linalg.norm(sig_ex)
    check("... where the plain sampler reads half of it",
          float(np.median(es0)) > 0.3, f"median {np.median(es0):.2f}")


def h_material_state(model, region, eps, asm) -> None:
    """[h] a state's solution is read with the state's moduli."""
    print("\n[h] MATERIAL STATE: densities and coefficients from ONE state")
    base = region.material
    mat = ElasticMaterial(mu=STATE_MU,
                          lam=2.0 * STATE_MU * STATE_NU / (1.0 - 2.0 * STATE_NU))
    sol = asm.rebuild_for_materials({region.name: mat}).solve()
    state = model.with_materials({region.name: mat})

    grid = V.make_grid(model, 5.0)
    code = V.classify(model, grid)                 # geometry: the base model's
    c_h, _, _ = V.clearance(model, grid, eps)
    sel = np.flatnonzero((code > 0) & (c_h > 2.0 * defaults.NEAR_BOUNDARY_H_RATIO))
    pts, n = grid.points[sel], sel.size
    u_ex = pts @ STRAIN_A.T
    sig_ex = mat.lam * np.trace(STRAIN_A) * np.eye(3) + 2.0 * mat.mu * STRAIN_A
    e_ex = 0.5 * (STRAIN_A + STRAIN_A.T)

    f = V.sample(state, sol, eps, grid, code)
    close("state readout: u == A x", relerr(f["u"][sel], u_ex), TOL_MANU_U)
    close("state readout: sigma == C_state:A",
          relerr(f["sigma"][sel], np.broadcast_to(sig_ex, (n, 3, 3))),
          TOL_MANU_S)
    close("state readout: strain == sym(A)  [Hooke at the state's moduli]",
          relerr(f["strain"][sel], np.broadcast_to(e_ex, (n, 3, 3))),
          TOL_MANU_E)

    # The tripwire. Without it the three checks above would pass on any
    # material that happened to give A x, and the clause could not fail.
    bad = V.sample(model, sol, eps, grid, code)
    miss_u, miss_s = (relerr(bad["u"][sel], u_ex),
                      relerr(bad["sigma"][sel],
                             np.broadcast_to(sig_ex, (n, 3, 3))))
    check("the BASE model reads that same solution wrong (u)",
          miss_u > MISS_STATE, f"{miss_u:.2f} > {MISS_STATE}")
    check("the BASE model reads that same solution wrong (sigma)",
          miss_s > MISS_STATE, f"{miss_s:.2f} > {MISS_STATE}")
    check("with_materials left the base model's material alone",
          region.material is base and model.regions[0] is region)

    # What `mbem sample` does with a stored run: resolve the state's model
    # from the run's own spec. Given the JSON form a resolved dump holds.
    resolved = {"spec": {"states": [
        {"label": "base", "materials": {}},
        {"label": "soft state", "materials": {
            region.name: {"mu": mat.mu, "lam": mat.lam}}}]}}
    m_base = cli._state_model(resolved, model, "base").regions[0].material
    m_soft = cli._state_model(resolved, model, "soft-state").regions[0].material
    check("stored state -> its own moduli (found by file slug)",
          (m_soft.mu, m_soft.lam) == (mat.mu, mat.lam)
          and (m_base.mu, m_base.lam) == (base.mu, base.lam))
    try:
        cli._state_model(resolved, model, "undeclared")
        refused = False
    except KeyError:
        refused = True
    check("a state the spec does not declare is refused, not read as base",
          refused)


def main() -> bool:
    t0 = time.time()
    print("=" * 76)
    print("Volume sampling: the same field, on a grid, with its own provenance")
    print("=" * 76)
    model, region, sol, eps, asm = _manufactured()
    print(f"  manufactured box: {sum(p.mesh.n_triangles for p in region.patches)}"
          f" triangles, nu={NU}, eps/h={EPS_BOX / BOX_EDGE:.3f}")
    a_chunking(model, region, sol, eps)
    b_classification()
    c_manufactured(model, region, sol, eps)
    d_eigenstress(model, region, sol, eps)
    e_vti_roundtrip(model, region, sol, eps)
    f_clearance_is_the_solver_s(model, region, sol, eps)
    g_webexport()
    h_material_state(model, region, eps, asm)
    i_time_payload()
    j_near_a_face()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: volume sampling "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
