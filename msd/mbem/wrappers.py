"""Legacy-signature wrappers over the region-graph core.

Each ``solve_*_v2`` accepts the same arguments (and returns the same
shapes) as its frozen legacy counterpart, but routes through
RegionModel -> generate_system -> DenseBackend. ``mode="legacy"`` uses
the legacy assembly calls (entrywise parity); ``mode="basis"`` uses the
numba material-basis path (fast, rebuildable for new materials).
"""

from __future__ import annotations

import numpy as np

from .backends import DenseBackend
from .model import BCType, Patch, Region, RegionModel, generate_system


def _slip_value(fault_mesh, fault_slip_vector, slip_magnitude) -> np.ndarray:
    v = slip_magnitude * np.asarray(fault_slip_vector, dtype=float)
    return np.broadcast_to(v, (fault_mesh.n_triangles, 3))


def build_three_region_box_model(meshes, fault_mesh, fault_slip_vector,
                                 slip_magnitude, material_1, material_2,
                                 material_3, fault_region="layer1"):
    top = Patch("top", meshes["top"], BCType.FREE_TRACTION)
    s1 = Patch("sides1", meshes["sides"][1], BCType.FREE_TRACTION)
    s2 = Patch("sides2", meshes["sides"][2], BCType.FREE_TRACTION)
    s3 = Patch("sides3", meshes["sides"][3], BCType.FREE_TRACTION)
    i1 = Patch("interface1", meshes["interfaces"][1], BCType.INTERFACE)
    i2 = Patch("interface2", meshes["interfaces"][2], BCType.INTERFACE)
    base = Patch("base", meshes["base"], BCType.PRESCRIBED_DISPLACEMENT)
    fault = Patch("fault", fault_mesh, BCType.FAULT,
                  value=_slip_value(fault_mesh, fault_slip_vector,
                                    slip_magnitude))

    top_v = meshes["top"].vertices
    cx = 0.5 * (top_v[:, 0].min() + top_v[:, 0].max()) \
        + 0.11 * (top_v[:, 0].max() - top_v[:, 0].min())
    cy = 0.5 * (top_v[:, 1].min() + top_v[:, 1].max()) \
        + 0.07 * (top_v[:, 1].max() - top_v[:, 1].min())
    z_top = float(top_v[:, 2].mean())
    z1 = float(meshes["interfaces"][1].vertices[:, 2].mean())
    z2 = float(meshes["interfaces"][2].vertices[:, 2].mean())
    z_bot = float(meshes["base"].vertices[:, 2].mean())

    flt = {name: [fault] if name == fault_region else []
           for name in ("layer1", "layer2", "layer3")}
    R1 = Region("layer1", material_1, [top, s1, i1],
                probe_point=np.array([cx, cy, 0.5 * (z_top + z1)]),
                faults=flt["layer1"])
    R2 = Region("layer2", material_2, [i1, s2, i2],
                probe_point=np.array([cx, cy, 0.5 * (z1 + z2)]),
                faults=flt["layer2"])
    R3 = Region("layer3", material_3, [i2, s3, base],
                probe_point=np.array([cx, cy, 0.5 * (z2 + z_bot)]),
                faults=flt["layer3"])
    return RegionModel([R1, R2, R3])


def build_vertical_fault_zone_model(x_range, y_range, z_bottom,
                                    zone_half_width,
                                    material_outer, material_zone,
                                    fault_mesh, fault_slip_vector,
                                    slip_magnitude,
                                    nx_outer=3, nx_zone=2, ny=4, nz=2):
    """Fault-zone slab between two host blocks — REAL vertical-interface
    topology (three regions side by side), which no legacy solver can
    express (mollified_fault_zone_showcase.py fakes it with horizontal
    same-material interfaces). The fault must lie inside |x| < zone_half_width.

    Region graph:  left block | fault-zone slab | right block,
    joined by two vertical INTERFACE planes at x = -+ zone_half_width;
    free tops and outer walls, fixed bases.
    """
    from local_box_mesh import (_concatenate_meshes,
                                make_rectangular_patch,
                                make_vertical_panel)

    x0, x1 = x_range
    a = zone_half_width
    zr = (z_bottom, 0.0)

    def top(name, xr, nx):
        return Patch(name, make_rectangular_patch(xr, y_range, 0.0, nx, ny,
                                                  normal_up=True),
                     BCType.FREE_TRACTION)

    def base(name, xr, nx):
        return Patch(name, make_rectangular_patch(xr, y_range, z_bottom,
                                                  nx, ny, normal_up=False),
                     BCType.PRESCRIBED_DISPLACEMENT)

    def ywalls(name, xr, nx):
        panels = [
            make_vertical_panel("y", y_range[1], xr, zr, nx, nz, +1),
            make_vertical_panel("y", y_range[0], xr, zr, nx, nz, -1),
        ]
        return Patch(name, _concatenate_meshes(panels), BCType.FREE_TRACTION)

    def xwall(name, x_at, sign):
        return Patch(name, make_vertical_panel("x", x_at, y_range, zr,
                                               ny, nz, sign),
                     BCType.FREE_TRACTION)

    def iface(name, x_at):
        return Patch(name, make_vertical_panel("x", x_at, y_range, zr,
                                               ny, nz, +1),
                     BCType.INTERFACE)

    fault = Patch("fault", fault_mesh, BCType.FAULT,
                  value=_slip_value(fault_mesh, fault_slip_vector,
                                    slip_magnitude))

    if_l = iface("iface_left", -a)
    if_r = iface("iface_right", +a)

    cy = 0.5 * (y_range[0] + y_range[1]) + 0.07 * (y_range[1] - y_range[0])
    zmid = 0.5 * z_bottom

    left = Region(
        "left", material_outer,
        [top("top_left", (x0, -a), nx_outer),
         xwall("wall_left", x0, -1),
         ywalls("ywalls_left", (x0, -a), nx_outer),
         if_l,
         base("base_left", (x0, -a), nx_outer)],
        probe_point=np.array([0.5 * (x0 - a), cy, zmid]))
    zone = Region(
        "zone", material_zone,
        [if_l,
         top("top_zone", (-a, a), nx_zone),
         ywalls("ywalls_zone", (-a, a), nx_zone),
         if_r,
         base("base_zone", (-a, a), nx_zone)],
        probe_point=np.array([0.31 * a, cy, zmid]),
        faults=[fault])
    right = Region(
        "right", material_outer,
        [if_r,
         top("top_right", (a, x1), nx_outer),
         xwall("wall_right", x1, +1),
         ywalls("ywalls_right", (a, x1), nx_outer),
         base("base_right", (a, x1), nx_outer)],
        probe_point=np.array([0.5 * (a + x1), cy, zmid]))

    return RegionModel([left, zone, right])


def solve_three_region_box_v2(meshes, fault_mesh, fault_normal,
                              fault_slip_vector, slip_magnitude,
                              material_1, material_2, material_3, eps,
                              mode="basis", return_assembled=False):
    """Build a three-region box model and solve it with the dense backend.
    Returns the raw slot->array solution dict (and the assembled operator
    if return_assembled)."""
    model = build_three_region_box_model(
        meshes, fault_mesh, fault_slip_vector, slip_magnitude,
        material_1, material_2, material_3)
    asm = DenseBackend(mode).assemble(generate_system(model), eps)
    sol = asm.solve()
    return (sol, asm) if return_assembled else sol
