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
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import inclusion_mesh                                               # noqa: E402
import mollified_bem as mb                                          # noqa: E402
from local_box_mesh import (_concatenate_meshes, make_rectangular_patch,
                            make_vertical_panel)                    # noqa: E402
from mbem.model import BCType, Patch, Region, RegionModel            # noqa: E402

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
    # Printed HERE rather than in __main__, so a caller that imports this gate
    # and calls main() sees the verdict too -- every other gate prints its own.
    print(("PASS" if ok else "FAIL") + ": RegionModel fault validation")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
