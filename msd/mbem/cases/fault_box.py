"""Shared builder: a homogeneous Cartesian box with a vertical strike-slip fault.

Full-space mollified BEM: free top, traction-free sides, clamped base, and one
slip fault inside.  No half-space kernels -- the free surface is enforced by the
boundary integral equation on full-space Kelvin kernels.  Reused by the
fault-only, eps-convergence, and H-matrix demos.
"""
from __future__ import annotations

import numpy as np

from local_box_mesh_eq import (_concatenate_meshes, make_rectangular_patch_eq,
                               make_top_patch_with_fault,
                               make_vertical_fault_eq, make_vertical_panel_eq)
from mbem.model import BCType, Patch, Region, RegionModel

# On-fault elastic stress wants eps <= ~0.07 h on the fault (msd/CLAUDE.md
# rule 4); the boundary patches take "auto" (= defaults.EPS_OVER_H * h).
def build_fault_box(half_x=200.0, z_bottom=-100.0, fault_half_len=50.0,
                    fault_depth=20.0, edge_fault=8.0, edge_near=20.0,
                    edge_far=40.0, edge_side=40.0, near_field_radius=120.0):
    """Return a dict of boundary/fault meshes for a homogeneous box + fault.

    The top free surface is FAULT-ALIGNED (the surface-breaking fault trace is
    embedded as exact mesh edges via make_top_patch_with_fault, graded fine near
    the fault), so no triangle straddles the slip discontinuity -- the surface
    fields are clean across the trace."""
    x_range = (-half_x, half_x)
    y_range = (-half_x, half_x)
    fault_trace = np.column_stack([np.zeros(2),
                                   np.array([-fault_half_len, fault_half_len])])
    top = make_top_patch_with_fault(
        x_range, y_range, 0.0, fault_trace, edge_fault=edge_fault,
        edge_far=edge_far, near_field_radius=near_field_radius,
        edge_near=edge_near, normal_up=True)
    base = make_rectangular_patch_eq(x_range, y_range, z_bottom, edge_far,
                                     normal_up=False)
    panels = [
        make_vertical_panel_eq("x", x_range[1], y_range, (z_bottom, 0.0),
                               edge_side, +1),
        make_vertical_panel_eq("x", x_range[0], y_range, (z_bottom, 0.0),
                               edge_side, -1),
        make_vertical_panel_eq("y", y_range[1], x_range, (z_bottom, 0.0),
                               edge_side, +1),
        make_vertical_panel_eq("y", y_range[0], x_range, (z_bottom, 0.0),
                               edge_side, -1),
    ]
    sides = _concatenate_meshes(panels)
    fault, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=2.0 * fault_half_len,
        depth_range=(-fault_depth, 0.0), target_edge=edge_fault)
    return dict(top=top, base=base, sides=sides, fault=fault,
                n_hat=n_hat, s_hat=s_hat, x_range=x_range, z_bottom=z_bottom)


def build_model(meshes, slip_mag, material, order_top=0):
    """RegionModel for a single homogeneous region (top + sides + base) with the
    fault as an interior slip source: RIGHT-lateral slip of magnitude
    ``slip_mag``.

    ``order_top`` is the Lagrange order of the free-surface patch. A P0 top
    cannot follow the slope of the surface displacement at the trace, and the
    first element row of on-fault stress below it is +30-46 % high at every h
    and eps; a P1 top (``order_top=1``) at eps="auto" brings it to a few
    percent (gate: ``verify/verify_solved_bvp.py`` A4). The sides and base
    stay P0."""
    fault = meshes["fault"]
    s_hat = np.asarray(meshes["s_hat"], float)
    top = Patch("top", meshes["top"], BCType.FREE_TRACTION, order=order_top)
    sides = Patch("sides", meshes["sides"], BCType.FREE_TRACTION)
    base = Patch("base", meshes["base"], BCType.PRESCRIBED_DISPLACEMENT)
    # Patch.value is the Burgers vector b = u(+n) - u(-n); on this fault
    # (n = +x, s_hat = +y) right-lateral slip is b = -slip_mag * s_hat.
    fpatch = Patch("fault", fault, BCType.FAULT,
                   value=np.broadcast_to(-slip_mag * s_hat,
                                         (fault.n_triangles, 3)))
    hx = meshes["x_range"][1]
    region = Region("crust", material, [top, sides, base],
                    probe_point=np.array([0.45 * hx, 0.45 * hx,
                                          0.5 * meshes["z_bottom"]]),
                    faults=[fpatch])
    return RegionModel([region])
