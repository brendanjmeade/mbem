"""Worked example of building a RegionModel: the vertical fault-zone box.

Three regions side by side joined by two vertical INTERFACE planes --
a topology the frozen legacy solvers cannot express. Used by the
backend gates (verify_dense_backend, verify_hbackend).
"""

from __future__ import annotations

import numpy as np

from .model import BCType, Patch, Region, RegionModel


def _slip_value(fault_mesh, fault_slip_vector, slip_magnitude) -> np.ndarray:
    """``slip_magnitude * fault_slip_vector`` IS the Burgers vector
    b = u(+n) - u(-n) of the fault (Patch.value); no hidden sign."""
    v = slip_magnitude * np.asarray(fault_slip_vector, dtype=float)
    return np.broadcast_to(v, (fault_mesh.n_triangles, 3))


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
