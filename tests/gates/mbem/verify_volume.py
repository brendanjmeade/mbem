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
     unchunked call.
  b  region classification decides which material and which patch set a point
     is evaluated with. Ask for the host field inside the inclusion and the
     answer is wrong, plausible, and unremarked by anything in the evaluator.
     Gated against the solid-angle closure identity, against the analytic
     cylinder of the topo_inclusion model, and -- because the classifier reads
     the WARPED top mesh -- against points above and below a topographic hill.
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
from mbem import ElasticMaterial, defaults, volume as V, vti
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


def _manufactured():
    """Closed box, u = A x on every face. Returns (model, region, sol, eps)."""
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
    sol = AssembledDense(generate_system(model), EPS_BOX, "direct",
                         jump="calibrated").solve()
    return model, region, sol, EPS_BOX


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
    c_h, _ = V.clearance(model, grid, eps)
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
    c_h, c_eps = V.clearance(model, grid, eps)
    f = V.sample(model, sol, eps, grid, code)
    arrays = V.as_vti_arrays(grid, f, code, c_h, c_eps)
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
    c_h, c_eps = V.clearance(model, grid, eps)
    # Recompute exactly as evaluate._warn_near_boundary does: min over
    # BOUNDARY patches of d / element_size.
    ref = np.full(grid.points.shape[0], np.inf)
    for p in region.patches:
        d, idx = distance_to_mesh(grid.points, p.mesh)
        ref = np.minimum(ref, d / kb.element_sizes(p.mesh)[idx])
    # The sampler also includes faults (this model has none), so on a
    # fault-free model the two must agree exactly.
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


def main() -> bool:
    t0 = time.time()
    print("=" * 76)
    print("Volume sampling: the same field, on a grid, with its own provenance")
    print("=" * 76)
    model, region, sol, eps = _manufactured()
    print(f"  manufactured box: {sum(p.mesh.n_triangles for p in region.patches)}"
          f" triangles, nu={NU}, eps/h={EPS_BOX / BOX_EDGE:.3f}")
    a_chunking(model, region, sol, eps)
    b_classification()
    c_manufactured(model, region, sol, eps)
    d_eigenstress(model, region, sol, eps)
    e_vti_roundtrip(model, region, sol, eps)
    f_clearance_is_the_solver_s(model, region, sol, eps)
    g_webexport()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: volume sampling "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
