"""Shared builder: an elastic plate over a substrate, a vertical strike-slip
fault in the plate.

Two regions in one closed box: ``plate`` (free top, its four sides, the
interface) holding the fault, and ``substrate`` (the interface, its four sides,
a clamped base). Give the substrate a rheology and this is the earthquake-cycle
model of Nur & Mavko and Savage & Prescott, on a FINITE box: mbem has no
half-space and no region that runs to infinity (docs/viscoelastic.md, section
7), so the box is part of the model and its size is a parameter to be laddered.

``sides`` is the condition on the PLATE's sides. ``"free"`` is the natural one
and it floats: as the substrate relaxes only its shear holds the plate against
a rigid horizontal translation, and the discrete net force of the fault, which
is small rather than zero, is divided by a shear modulus going to zero.
``"clamped"`` removes that mode at the price of holding the plate edges. The
two bracket the truth; neither is a half-space.

Also here, because the gate and the study both need them: the closed forms for
an infinitely long fault in a plate over a half-space (antiplane), which the
mid-profile of a long fault in a large box approaches.
"""
from __future__ import annotations

import numpy as np

PLATE, SUBSTRATE = "plate", "substrate"
SIDES = ("free", "clamped")


def _panels(half_x, z_range, edge):
    from local_box_mesh_eq import _concatenate_meshes, make_vertical_panel_eq

    r = (-half_x, half_x)
    return _concatenate_meshes([
        make_vertical_panel_eq("x", +half_x, r, z_range, edge, +1),
        make_vertical_panel_eq("x", -half_x, r, z_range, edge, -1),
        make_vertical_panel_eq("y", +half_x, r, z_range, edge, +1),
        make_vertical_panel_eq("y", -half_x, r, z_range, edge, -1)])


def build_layered_box(half_x=400.0, z_interface=-30.0, z_bottom=-300.0,
                      fault_half_len=150.0, fault_depth=15.0,
                      edge_fault=5.0, edge_near=20.0, edge_far=80.0,
                      edge_side=80.0, edge_interface=15.0,
                      near_field_radius=90.0):
    """Meshes of the box. The top is fault-aligned and graded as in
    ``fault_box``. The interface is graded the same way about the fault's
    projection (``edge_interface`` beneath it, ``edge_far`` away): the field
    there varies on the scale of the plate thickness under the fault and of
    the box far from it, and a uniform interface is what makes a large box
    expensive."""
    from local_box_mesh_eq import (make_rectangular_patch_eq,
                                   make_top_patch_with_fault,
                                   make_vertical_fault_eq)

    if not (z_bottom < z_interface < -fault_depth < 0.0):
        raise ValueError("need z_bottom < z_interface < -fault_depth < 0: the "
                         "fault must end inside the plate")
    r = (-half_x, half_x)
    trace = np.column_stack([np.zeros(2),
                             np.array([-fault_half_len, fault_half_len])])
    top = make_top_patch_with_fault(
        r, r, 0.0, trace, edge_fault=edge_fault, edge_far=edge_far,
        near_field_radius=near_field_radius, edge_near=edge_near,
        normal_up=True)
    interface = make_top_patch_with_fault(
        r, r, z_interface, trace, edge_fault=edge_interface, edge_far=edge_far,
        near_field_radius=near_field_radius,
        edge_near=max(edge_near, edge_interface), normal_up=True)
    base = make_rectangular_patch_eq(r, r, z_bottom, edge_far, normal_up=False)
    fault, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=2.0 * fault_half_len, depth_range=(-fault_depth, 0.0),
        target_edge=edge_fault)
    return dict(top=top, interface=interface, base=base,
                sides_upper=_panels(half_x, (z_interface, 0.0), edge_side),
                sides_lower=_panels(half_x, (z_bottom, z_interface), edge_side),
                fault=fault, n_hat=n_hat, s_hat=s_hat, half_x=half_x,
                z_interface=z_interface, z_bottom=z_bottom,
                fault_depth=fault_depth, fault_half_len=fault_half_len)


def build_model(meshes, slip_mag, mat_plate, mat_substrate, sides="free",
                order_top=0):
    """RIGHT-lateral slip ``slip_mag`` on the fault, in the plate. The
    substrate's closed boundary is ``interface``, ``sides_lower``, ``base``: a
    Laplace run needs one scalar eps on those three (docs/viscoelastic.md,
    section 3), and the runner refuses otherwise."""
    from mbem.model import BCType, Patch, Region, RegionModel

    if sides not in SIDES:
        raise ValueError(f"sides {sides!r} must be one of {SIDES}")
    fault = meshes["fault"]
    s_hat = np.asarray(meshes["s_hat"], float)
    side_bc = (BCType.FREE_TRACTION if sides == "free"
               else BCType.PRESCRIBED_DISPLACEMENT)
    interface = Patch("interface", meshes["interface"], BCType.INTERFACE)
    # Patch.value is the Burgers vector b = u(+n) - u(-n); on this fault
    # (n = +x, s_hat = +y) right-lateral slip is b = -slip_mag * s_hat.
    fpatch = Patch("fault", fault, BCType.FAULT,
                   value=np.broadcast_to(-slip_mag * s_hat,
                                         (fault.n_triangles, 3)))
    hx, zi, zb = meshes["half_x"], meshes["z_interface"], meshes["z_bottom"]
    plate = Region(
        PLATE, mat_plate,
        [Patch("top", meshes["top"], BCType.FREE_TRACTION, order=order_top),
         Patch("sides_upper", meshes["sides_upper"], side_bc), interface],
        probe_point=np.array([0.45 * hx, 0.45 * hx, 0.5 * zi]),
        faults=[fpatch])
    substrate = Region(
        SUBSTRATE, mat_substrate,
        [interface,
         Patch("sides_lower", meshes["sides_lower"], BCType.FREE_TRACTION),
         Patch("base", meshes["base"], BCType.PRESCRIBED_DISPLACEMENT)],
        probe_point=np.array([0.45 * hx, 0.45 * hx, 0.5 * (zi + zb)]))
    return RegionModel([plate, substrate])


# ---------------------------------------------------------------------
# the antiplane closed forms: an infinitely long fault, a half-space below
# ---------------------------------------------------------------------

def image_series(x, depth, thickness, mu_plate, mu_substrate,
                 rtol: float = 1e-14, n_max: int = 100000):
    """Fault-parallel surface displacement, per unit slip, at distance ``x``
    (> 0) from an infinitely long vertical strike-slip fault that cuts an
    elastic plate from the surface to ``depth``, over a half-space:

        u = (1/pi) [ atan(D/x) + sum_{n>=1} G^n ( atan((2nH + D)/x)
                                                 - atan((2nH - D)/x) ) ]
        G = (mu_plate - mu_substrate) / (mu_plate + mu_substrate)

    (Rybicki 1971). Plain arithmetic in ``mu_substrate``, so the same function
    is the Laplace-domain reference when handed ``mu(s)``. The side x > 0 moves
    with half the slip at the fault: ``u -> 1/2`` as ``x -> 0``.
    """
    x = np.asarray(x, float)
    D, H = float(depth), float(thickness)
    G = (mu_plate - mu_substrate) / (mu_plate + mu_substrate)
    total = np.arctan(D / x)
    gn = 1.0
    for n in range(1, n_max + 1):
        gn = gn * G
        term = gn * (np.arctan((2 * n * H + D) / x)
                     - np.arctan((2 * n * H - D) / x))
        total = total + term
        if np.max(np.abs(term)) <= rtol * np.max(np.abs(total)):
            break
    return total / np.pi


def maxwell_series(x, t, depth, thickness, mu_plate, mu_substrate, tau,
                   rtol: float = 1e-12, n_max: int = 5000):
    """The same after a unit step of slip at t = 0, the half-space a Maxwell
    solid of relaxation time ``tau`` (Nur & Mavko 1974; Savage & Prescott
    1978): ``(n_t, n_x)``.

    ``G(s)`` has ONE pole, at ``-1/tau_R`` with ``tau_R = tau (mu_plate +
    mu_substrate) / mu_plate``, so ``G^n`` has it n-fold and its step response
    is a sum of regularized incomplete gamma functions ``P(k, t/tau_R)``.
    """
    from scipy.special import comb, gammainc

    x = np.atleast_1d(np.asarray(x, float))
    t = np.atleast_1d(np.asarray(t, float))
    D, H = float(depth), float(thickness)
    G0 = (mu_plate - mu_substrate) / (mu_plate + mu_substrate)   # unrelaxed
    w = t / (tau * (mu_plate + mu_substrate) / mu_plate)
    total = np.broadcast_to(np.arctan(D / x), (t.size, x.size)).copy()
    # G(s)^n = sum_k C(n,k) G0^(n-k) (1-G0)^k [beta/(s+beta)]^k, step response
    # of [beta/(s+beta)]^k is P(k, beta t), with P(0, .) = 1
    for n in range(1, n_max + 1):
        k = np.arange(0, n + 1)
        coef = comb(n, k) * G0 ** (n - k) * (1.0 - G0) ** k
        P = np.ones((t.size, n + 1))
        P[:, 1:] = gammainc(k[None, 1:], w[:, None])
        gn = P @ coef                                             # (n_t,)
        geom = (np.arctan((2 * n * H + D) / x)
                - np.arctan((2 * n * H - D) / x))
        term = gn[:, None] * geom[None, :]
        total += term
        if np.max(np.abs(term)) <= rtol * np.max(np.abs(total)):
            break
    return total / np.pi
