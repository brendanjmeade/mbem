"""Verify RegionModel.validate() on faults.

Checks (PASS/FAIL):
  a. a non-FAULT patch in Region.faults is rejected
  b. the same fault object in two regions is rejected
  c. a fault named like a boundary patch is rejected
  d. a fault outside its region, or lying ON its boundary, is rejected
  e. RegionModel.orientation() still refuses a fault of another region
  f. the selfcheck model and examples/_fault_box.build_model() validate
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

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
    m = _two_layer([_fault("f1", -0.8, -0.2)], [_fault("f2", -1.8, -1.2)])
    ok &= _rejects("e. orientation() of a fault of another region",
                   lambda: m.orientation(m.regions[0], m.regions[1].faults[0]),
                   "not a fault of")
    print(f"  [ok] two-layer model with one fault per layer validates; "
          f"sigma(fault) = {m.orientation(m.regions[0], m.regions[0].faults[0])}")

    from mbem import selfcheck
    selfcheck._build_model(MAT.lam)
    print("  [ok] selfcheck model validates")
    import _fault_box
    meshes = _fault_box.build_fault_box(half_x=100.0, z_bottom=-50.0,
                                        fault_half_len=25.0, fault_depth=10.0,
                                        edge_fault=5.0, edge_near=20.0,
                                        edge_far=40.0, edge_side=40.0,
                                        near_field_radius=60.0)
    _fault_box.build_model(meshes, 1e-3, MAT)
    print("  [ok] examples/_fault_box.build_model validates")
    return bool(ok)


if __name__ == "__main__":
    passed = main()
    print(("PASS" if passed else "FAIL") + ": RegionModel fault validation")
    sys.exit(0 if passed else 1)
