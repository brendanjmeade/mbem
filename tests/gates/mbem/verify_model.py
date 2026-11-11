"""Verify RegionModel.validate() on faults, and mesh validity.

Checks (PASS/FAIL):
  a. a non-FAULT patch in Region.faults is rejected
  b. the same fault object in two regions is rejected
  c. a fault named like a boundary patch is rejected
  d. a fault outside its region, or lying ON its boundary, is rejected
  d''. a patch mesh with a collapsed element is rejected
  e. RegionModel.orientation() still refuses a fault of another region
  f. the selfcheck model and mbem.cases.fault_box.build_model() validate
  g. a refinement ring keeps its vertices off the other constrained
     segments of the PSLG, and the refined topo_inclusion geometry
     passes the collapsed-element guard
  h. a region with NO boundary patch is the full space: it validates with a
     fault (either kind, anywhere), point_in_region is True everywhere with
     the closure sum 4 pi, is_anchored() is False and sigma_table() empty; it
     is refused without a fault; and clause d still rejects a fault outside
     a bounded region afterwards (the empty-boundary branch loosened nothing)
  i. with_materials() is a material STATE of the same geometry: the named
     region gets the new material, every Patch and fault object is shared
     (an interface is identified by ``is``), the orientation table is the
     same, the base model is untouched, and an unknown region is refused
  j. Gravity is a condition on the boundary, stated once
     (RegionModel.buoyancy): -rho g n (x) up on the named free surface, the
     buoyancy of the CONTRAST on an interface and only for its second region
     (whichever is listed second), nothing on a side, a base, a fault or
     between equal densities; one array per (region, patch) in every row
     block of the system; up radial about a centre; a contrast heavy side up
     is reported; a material state keeps it; and a density for a region the
     model lacks, a missing one, a surface that is not a free patch, and
     deflation under gravity are all refused
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import inclusion_mesh                                               # noqa: E402
import mollified_bem as mb                                          # noqa: E402
from local_box_mesh import (_concatenate_meshes, make_rectangular_patch,
                            make_vertical_panel)                    # noqa: E402
from mbem.model import (BCType, Gravity, Patch, Region, RegionModel,  # noqa: E402
                        generate_system)

MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)
XY = (-1.0, 1.0)


def _walls(name, zr):
    panels = [make_vertical_panel("x", +1.0, XY, zr, 2, 2, +1),
              make_vertical_panel("x", -1.0, XY, zr, 2, 2, -1),
              make_vertical_panel("y", +1.0, XY, zr, 2, 2, +1),
              make_vertical_panel("y", -1.0, XY, zr, 2, 2, -1)]
    return Patch(name, _concatenate_meshes(panels), BCType.FREE_TRACTION)


def _fault(name, z0, z1, x=0.0):
    m = make_vertical_panel("x", x, (-0.4, 0.4), (z0, z1), 2, 2, +1)
    return Patch(name, m, BCType.FAULT, value=np.array([0.0, 1e-3, 0.0]))


def _two_layer(faults1, faults2):
    """Two stacked unit boxes, z in [-1, 0] and [-2, -1], sharing an interface."""
    top = Patch("top", make_rectangular_patch(XY, XY, 0.0, 2, 2), BCType.FREE_TRACTION)
    mid = Patch("mid", make_rectangular_patch(XY, XY, -1.0, 2, 2), BCType.INTERFACE)
    base = Patch("base", make_rectangular_patch(XY, XY, -2.0, 2, 2, normal_up=False),
                 BCType.PRESCRIBED_DISPLACEMENT)
    r1 = Region("upper", MAT, [top, _walls("w1", (-1.0, 0.0)), mid],
                probe_point=np.array([0.1, 0.05, -0.5]), faults=faults1)
    r2 = Region("lower", MAT, [mid, _walls("w2", (-2.0, -1.0)), base],
                probe_point=np.array([0.1, 0.05, -1.5]), faults=faults2)
    return RegionModel([r1, r2])


def _collapsed_patch():
    """A patch whose mesh carries one zero-area element -- what a PSLG
    failure hands the solver. It has no frame for the kernels, no h for
    eps="auto", and a collocation point on top of its neighbour's, which
    collapses its cluster's bounding box so that the compressed backend
    admits (and compresses) a block whose elements touch."""
    m = make_rectangular_patch(XY, XY, 0.0, 2, 2)
    tris = np.vstack([m.triangles, m.triangles[0, [0, 1, 1]]])
    return Patch("collapsed", mb.TriMesh(vertices=m.vertices, triangles=tris),
                 BCType.FREE_TRACTION)


def _rejects(title, build, fragment):
    try:
        build()
    except ValueError as e:
        ok = fragment in str(e)
        print(f"  [{'ok' if ok else 'FAIL'}] {title}: {str(e)[:90]}")
        return ok
    print(f"  [FAIL] {title}: no ValueError raised")
    return False


def main() -> bool:
    ok = True
    ok &= _rejects("a. non-FAULT bc in faults",
                   lambda: _two_layer([Patch("plate", _fault("f", -0.8, -0.2).mesh,
                                             BCType.PRESCRIBED_DISPLACEMENT)], []),
                   "FAULT patches only")
    shared = _fault("f", -1.5, -0.5)
    ok &= _rejects("b. same fault object in two regions",
                   lambda: _two_layer([shared], [shared]), "exactly 1 region")
    ok &= _rejects("c. fault named like a patch",
                   lambda: _two_layer([_fault("top", -0.8, -0.2)], []),
                   "unique model-wide")
    ok &= _rejects("c'. two faults with one name",
                   lambda: _two_layer([_fault("f", -0.8, -0.2)], [_fault("f", -1.8, -1.2)]),
                   "unique model-wide")
    ok &= _rejects("d. fault outside its region (in the other layer)",
                   lambda: _two_layer([_fault("f", -1.8, -1.2)], []), "not inside")
    ok &= _rejects("d'. fault ON the region boundary (2 pi)",
                   lambda: _two_layer([_fault("f", -0.8, -0.2, x=1.0)], []),
                   "split it at the interface")
    ok &= _rejects("d''. a collapsed element in a patch mesh",
                   _collapsed_patch, "degenerate")
    m = _two_layer([_fault("f1", -0.8, -0.2)], [_fault("f2", -1.8, -1.2)])
    ok &= _rejects("e. orientation() of a fault of another region",
                   lambda: m.orientation(m.regions[0], m.regions[1].faults[0]),
                   "not a fault of")
    print(f"  [ok] two-layer model with one fault per layer validates; "
          f"sigma(fault) = {m.orientation(m.regions[0], m.regions[0].faults[0])}")

    from mbem import selfcheck
    selfcheck._build_model(MAT.lam)
    print("  [ok] selfcheck model validates")
    from mbem.cases import fault_box as _fault_box
    meshes = _fault_box.build_fault_box(half_x=100.0, z_bottom=-50.0,
                                        fault_half_len=25.0, fault_depth=10.0,
                                        edge_fault=5.0, edge_near=20.0,
                                        edge_far=40.0, edge_side=40.0,
                                        near_field_radius=60.0)
    _fault_box.build_model(meshes, 1e-3, MAT)
    print("  [ok] mbem.cases.fault_box.build_model validates")
    ok &= _ring_clears_the_trace()
    ok &= _full_space()
    ok &= _material_state()
    ok &= _gravity()
    # Printed HERE rather than in __main__, so a caller that imports this gate
    # and calls main() sees the verdict too -- every other gate prints its own.
    print(("PASS" if ok else "FAIL") + ": RegionModel fault validation")
    return bool(ok)


def _full_space() -> bool:
    """h. An empty patch list is the whole space."""
    ok = True
    far = _fault("far", -0.8, -0.2, x=1e3)
    for kind, f in (("FAULT", _fault("f", -0.8, -0.2)),
                    ("FAULT_TRACTION", Patch("ft", _fault("g", -0.8, -0.2).mesh,
                                             BCType.FAULT_TRACTION,
                                             value=np.array([0.0, 1e-3, 0.0]))),
                    ("far FAULT", far)):
        m = RegionModel([Region("space", MAT, [], np.zeros(3), faults=[f])])
        r = m.regions[0]
        pts = np.array([[0.0, 0.0, -0.5], [3e3, -2e3, 1e3]])
        inside, total = m.point_in_region(r, pts, closure=True)
        good = (r.is_full_space and inside.all() and np.allclose(total, 4.0 * np.pi)
                and not m.is_anchored() and m.sigma_table() == {}
                and m.orientation(r, f) == -1)
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] h. full-space region with a {kind} fault "
              f"validates; inside everywhere, unanchored, no sigma")
    ok &= _rejects("h'. full-space region with no fault",
                   lambda: RegionModel([Region("space", MAT, [], np.zeros(3))]),
                   "nothing to solve")
    ok &= _rejects("h''. clause d after h: fault outside a bounded region still rejected",
                   lambda: _two_layer([_fault("f", -1.8, -1.2)], []), "not inside")
    return bool(ok)


def _material_state() -> bool:
    """i. A material state shares the geometry and replaces only moduli."""
    base = _two_layer([_fault("f1", -0.8, -0.2)], [_fault("f2", -1.8, -1.2)])
    other = mb.ElasticMaterial(mu=7.0, lam=19.0)
    st = base.with_materials({"lower": other})
    up0, lo0 = base.regions
    up1, lo1 = st.regions
    shared = all(p is q for a, b in ((up0, up1), (lo0, lo1))
                 for p, q in zip(a.patches + a.faults, b.patches + b.faults))
    good = (lo1.material is other and up1.material is MAT
            and lo0.material is MAT                       # the base is left alone
            and shared and len(lo1.patches) == len(lo0.patches)
            and st.sigma_table() == base.sigma_table()
            and st.orientation(lo1, lo1.faults[0])
            == base.orientation(lo0, lo0.faults[0])
            and [r.name for r in st.interface_regions(lo1.patches[0])]
            == ["upper", "lower"])
    print(f"  [{'ok' if good else 'FAIL'}] i. with_materials: new moduli on "
          f"the named region, geometry and sigma shared, base untouched")
    try:
        base.with_materials({"mantle": other})
        refused = False
    except KeyError as e:
        refused = "unknown region" in str(e)
    print(f"  [{'ok' if refused else 'FAIL'}] i'. with_materials refuses a "
          f"region the model does not have")
    return bool(good and refused)


def _gravity() -> bool:
    """j. Buoyancy: where it is, its sign, and what is refused."""
    from mbem import defaults
    from mbem.backends.dense import require_anchor_or_deflate

    ok = True

    def say(good, text):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'ok' if good else 'FAIL'}] j. {text}")

    base = _two_layer([_fault("f1", -0.8, -0.2)], [])
    rho_u, rho_l, g = 2700.0, 3300.0, 9.8
    w_u, w_l = (r * g * defaults.WEIGHT_GPA_PER_KM for r in (rho_u, rho_l))
    grav = Gravity(density={"upper": rho_u, "lower": rho_l}, surface="top")
    m = RegionModel(base.regions, gravity=grav)
    up, lo = m.regions
    top, walls, mid = up.patches
    zz = np.zeros((3, 3))
    zz[2, 2] = 1.0

    def is_(B, patch, w):
        n = patch.mesh.normals_and_areas()[0]            # the STORED normal
        want = -w * np.einsum("ni,j->nij", n, [0.0, 0.0, 1.0])
        return B is not None and B.shape == (patch.n_nodes, 3, 3) \
            and np.allclose(B, want, rtol=1e-14, atol=0.0)

    say(is_(m.buoyancy(up, top), top, w_u) and grav.weight("upper") == w_u
        and abs(w_u - 0.02646) < 1e-12,
        f"free surface: B = -rho g n (x) up, rho g = {w_u:.5f} GPa/km")
    say(m.buoyancy(up, mid) is None and is_(m.buoyancy(lo, mid), mid, w_l - w_u),
        "interface: the first region sees t, the second t + the contrast")
    flipped = RegionModel([lo, up], gravity=grav)
    say(flipped.buoyancy(lo, mid) is None
        and is_(flipped.buoyancy(up, mid), mid, w_u - w_l),
        "... whichever region is listed second")
    say(all(m.buoyancy(r, q) is None for r, q in
            ((up, walls), (lo, lo.patches[1]), (lo, lo.patches[2]),
             (up, up.faults[0]))),
        "no term on a side, a base or a fault")
    same = RegionModel(base.regions, gravity=Gravity(
        density={"upper": rho_u, "lower": rho_u}, surface="top"))
    say(same.buoyancy(same.regions[1], mid) is None,
        "equal densities: nothing on the interface")
    terms = [t for t in generate_system(m).terms if t.col_op is not None]
    per = {}
    for t in terms:
        per.setdefault((t.region.name, t.source_patch.name), []).append(t.col_op)
    say(sorted(per) == [("lower", "mid"), ("upper", "top")]
        and all(len(v) == 3 and all(b is v[0] for b in v) for v in per.values())
        and all(t.col.kind == "u" for t in terms),
        "the system: one B per (region, patch), on the u columns of every row")
    radial = Gravity(density={"a": 1.0}, center=(0.0, 0.0, 1.0))
    say(np.allclose(radial.up_at([[0.0, 0.0, 3.0], [2.0, 0.0, 1.0]]),
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        and np.allclose(Gravity(density={"a": 1.0}, up=(0, 0, 5)).up, (0, 0, 1)),
        "up: one unit vector, or radial about a centre")
    heavy_up = RegionModel(base.regions, gravity=Gravity(
        density={"upper": rho_l, "lower": rho_u}, surface="top"))
    say(m.unstable_contrasts() == [] and heavy_up.unstable_contrasts() == ["mid"]
        and RegionModel(base.regions, gravity=Gravity(
            density={"upper": rho_u, "lower": rho_l},
            surface="w1")).unstable_contrasts() == [],
        "a contrast heavy side up is reported; a vertical face is neither")
    st = m.with_materials({"lower": mb.ElasticMaterial(mu=7.0, lam=19.0)})
    say(st.gravity is grav and is_(st.buoyancy(st.regions[1], mid), mid, w_l - w_u),
        "a material state keeps its gravity")
    for title, build, fragment in (
            ("a density for a region the model lacks",
             lambda: RegionModel(base.regions, gravity=Gravity(
                 density={"upper": 1.0, "lower": 1.0, "mantle": 1.0})),
             "every region"),
            ("a region without a density",
             lambda: RegionModel(base.regions, gravity=Gravity(
                 density={"upper": 1.0})), "every region"),
            ("a surface that is clamped",
             lambda: RegionModel(base.regions, gravity=Gravity(
                 density={"upper": 1.0, "lower": 1.0}, surface="base")),
             "not a"),
            ("a negative density",
             lambda: Gravity(density={"upper": -1.0}), "densities"),
            ("deflate under gravity",
             lambda: require_anchor_or_deflate(generate_system(m),
                                               "calibrated", True),
             "buoyancy DETERMINES")):
        ok &= _rejects(f"j. refused: {title}", build, fragment)
    return bool(ok)


def _ring_clears_the_trace() -> bool:
    """g. A refinement ring must not put a vertex ON another constrained
    segment: Triangle splits the segment there and the split collapses
    (``inclusion_mesh.RING_SEGMENT_CLEARANCE``). The numbers are the
    topo_inclusion bump ring at the mesh scale where its vertex count is
    a multiple of four, so the unrotated ring lands one vertex exactly on
    the axis-aligned fault trace; the second clause meshes that whole
    geometry and hands every patch to the collapsed-element guard."""
    center, R, edge = (0.0, -50.0), 90.0, 3.0
    trace = np.array([[0.0, -100.0], [0.0, 100.0]])
    seg = np.array([[0, 1]])
    plain, _n = inclusion_mesh._circle_boundary(center, R, edge)
    rotated, n = inclusion_mesh._ring_points(center, R, edge, trace, seg)
    need = inclusion_mesh.RING_SEGMENT_CLEARANCE * (2.0 * np.pi * R / n)
    d0 = float(inclusion_mesh._min_dist_to_segments(plain, trace, seg).min())
    d1 = float(inclusion_mesh._min_dist_to_segments(rotated, trace, seg).min())
    ok = d0 < need <= d1
    print(f"  [{'ok' if ok else 'FAIL'}] g. refinement ring off the fault "
          f"trace: unrotated {d0:.1e} km, rotated {d1:.3f} km, "
          f"need {need:.3f} km")

    from mbem.cases.topo_inclusion import build as topo_build
    m, _flat, top, _ff, fault, _s, _b = topo_build(scale=3.0)
    for name, mesh in dict(m, host_top=top, fault=fault).items():
        Patch(name, mesh, BCType.FREE_TRACTION)
    print(f"  [ok] the refined topo_inclusion geometry ({top.n_triangles} "
          f"host_top triangles) carries no collapsed element")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
