"""Shared builder: an elastic plate over a substrate, a fault from the
surface down -- vertical and strike-slip in the plate unless told otherwise.

Two regions in one closed box: ``plate`` (free top, its four sides, the
interface) holding the fault, and ``substrate`` (the interface, its four sides,
a clamped base). Give the substrate a rheology and this is the earthquake-cycle
model of Nur & Mavko and Savage & Prescott, on a FINITE box: mbem has no
half-space and no region that runs to infinity (docs/viscoelastic.md, section
8), so the box is part of the model and its size is a parameter to be laddered.

A fault deeper than the plate is thick goes THROUGH the interface, and what
lies below it is a second fault, in the substrate: a megathrust whose lower
reach slips inside the region that relaxes.

``sides`` is the condition on the PLATE's sides. ``"free"`` is the natural one
and it floats: as the substrate relaxes only its shear holds the plate against
a rigid horizontal translation, and the discrete net force of the fault, which
is small rather than zero, is divided by a shear modulus going to zero.
``"clamped"`` removes that mode at the price of holding the plate edges. The
two bracket the truth; neither is a half-space.

Under gravity (``density``) the plate's free top is the free surface.

Also here, because the gate and the study both need them: the references for
an infinitely long fault in a plate over a half-space, which the mid-profile
of a long fault in a large box approaches. For strike slip they are closed
forms (antiplane). For dip slip there is none once the substrate differs or
gravity acts, and the reference is the plane-strain problem solved per
wavenumber (``dip_slip_profile``).
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


def _tilt(fault, dip):
    """The vertical fault panel turned about its trace to ``dip`` degrees,
    dipping toward +x, and its normal: ``(sin, 0, cos)``, up and into the
    hanging wall. A rotation, so the triangulation and its orientation are
    the vertical panel's own."""
    from mollified_bem import TriMesh

    d = np.deg2rad(dip)
    R = np.array([[np.sin(d), 0.0, -np.cos(d)],
                  [0.0, 1.0, 0.0],
                  [np.cos(d), 0.0, np.sin(d)]])
    return (TriMesh(vertices=fault.vertices @ R.T, triangles=fault.triangles),
            R[:, 0].copy())


def build_layered_box(half_x=400.0, z_interface=-30.0, z_bottom=-300.0,
                      fault_half_len=150.0, fault_depth=15.0,
                      edge_fault=5.0, edge_near=20.0, edge_far=80.0,
                      edge_side=80.0, edge_interface=15.0,
                      near_field_radius=90.0, dip=90.0):
    """Meshes of the box. The top is fault-aligned and graded as in
    ``fault_box``. The interface is graded the same way about the fault's
    projection (``edge_interface`` beneath it, ``edge_far`` away): the field
    there varies on the scale of the plate thickness under the fault and of
    the box far from it, and a uniform interface is what makes a large box
    expensive.

    The fault breaks the surface along x = 0 and reaches the DEPTH
    ``fault_depth``, dipping toward +x at ``dip`` degrees (90: vertical), so
    +x is the hanging wall and the stored normal points into it.

    Deeper than the plate is thick, it goes THROUGH the interface. The part
    below is a second mesh, ``fault_lower``, on the same plane, which belongs
    to the substrate; and the interface is then aligned on the line where the
    fault cuts it, as the top is on the trace, because the displacement of
    the interface jumps there by the slip."""
    from local_box_mesh_eq import (make_rectangular_patch_eq,
                                   make_top_patch_with_fault,
                                   make_vertical_fault_eq)

    if not 0.0 < dip <= 90.0:
        raise ValueError(f"dip {dip} must be in (0, 90] degrees")
    if not (z_bottom < z_interface < 0.0 and 0.0 < fault_depth < -z_bottom):
        raise ValueError("need z_bottom < z_interface < 0 and the fault to "
                         "end above the base")
    sin_dip = np.sin(np.deg2rad(dip))
    below = fault_depth + z_interface           # how far past the interface
    if 0.0 < below < 0.5 * edge_fault * sin_dip:
        raise ValueError(f"the fault ends {below:g} km below the interface, "
                         f"less than half a fault element: end it at the "
                         f"interface or take it deeper")
    r = (-half_x, half_x)
    trace = np.column_stack([np.zeros(2),
                             np.array([-fault_half_len, fault_half_len])])
    top = make_top_patch_with_fault(
        r, r, 0.0, trace, edge_fault=edge_fault, edge_far=edge_far,
        near_field_radius=near_field_radius, edge_near=edge_near,
        normal_up=True)
    # where the fault meets the interface, or under the trace if it does not
    cut = trace if below < 0.0 else trace + np.array(
        [-z_interface / np.tan(np.deg2rad(dip)) if dip != 90.0 else 0.0, 0.0])
    interface = make_top_patch_with_fault(
        r, r, z_interface, cut, edge_fault=edge_interface, edge_far=edge_far,
        near_field_radius=near_field_radius,
        edge_near=max(edge_near, edge_interface), normal_up=True)
    base = make_rectangular_patch_eq(r, r, z_bottom, edge_far, normal_up=False)
    reach = (-fault_depth, z_interface)         # the depths the two parts end at

    def panel(z_lo, z_hi):
        mesh, n_hat, s_hat = make_vertical_fault_eq(
            strike_length=2.0 * fault_half_len,
            depth_range=(z_lo / sin_dip, z_hi / sin_dip), target_edge=edge_fault)
        if dip != 90.0:
            mesh, n_hat = _tilt(mesh, dip)
        return mesh, n_hat, s_hat

    fault, n_hat, s_hat = panel(max(reach), 0.0)
    lower = dict(fault_lower=panel(*reach)[0]) if below > 0.0 else {}
    return dict(top=top, interface=interface, base=base,
                sides_upper=_panels(half_x, (z_interface, 0.0), edge_side),
                sides_lower=_panels(half_x, (z_bottom, z_interface), edge_side),
                fault=fault, n_hat=n_hat, s_hat=s_hat, half_x=half_x,
                z_interface=z_interface, z_bottom=z_bottom,
                fault_depth=fault_depth, fault_half_len=fault_half_len,
                dip=dip, **lower)


def burgers(slip_mag, rake, n_hat, s_hat):
    """The Burgers vector ``b = u(+n) - u(-n)`` of slip ``slip_mag`` at
    ``rake`` degrees (Aki & Richards): the motion of the +n side -- the
    hanging wall -- in the fault plane, measured from the strike direction
    toward up-dip. 180 is right-lateral, 0 left-lateral, 90 a thrust, -90 a
    normal fault. Rounded so that the quarter turns are exact."""
    c, s = (round(float(f(np.deg2rad(rake))), 15) for f in (np.cos, np.sin))
    s_hat = np.asarray(s_hat, float)
    return slip_mag * (c * s_hat + s * np.cross(np.asarray(n_hat, float), s_hat))


def build_model(meshes, slip_mag, mat_plate, mat_substrate, sides="free",
                order_top=0, density=None, g=None, rake=180.0):
    """Slip ``slip_mag`` on the fault at ``rake`` degrees (``burgers``):
    RIGHT-lateral unless told otherwise. It is in the plate, and where the
    meshes carry ``fault_lower`` the same slip continues on it in the
    substrate. The
    substrate's closed boundary is ``interface``, ``sides_lower``, ``base``: a
    Laplace run needs one scalar eps on those three (docs/viscoelastic.md,
    section 3), and the runner refuses otherwise.

    ``density = (plate, substrate)`` in kg/m^3 turns gravity on
    (``model.Gravity``): buoyancy on ``top``, the free surface, and on the
    interface if the two differ. The sides and the base are cuts through
    material that continues and carry none -- which also leaves the
    substrate's free sides a reservoir, so that relaxed it holds no pressure,
    as a half-space would not."""
    from mbem.model import BCType, Gravity, Patch, Region, RegionModel

    if sides not in SIDES:
        raise ValueError(f"sides {sides!r} must be one of {SIDES}")
    side_bc = (BCType.FREE_TRACTION if sides == "free"
               else BCType.PRESCRIBED_DISPLACEMENT)
    interface = Patch("interface", meshes["interface"], BCType.INTERFACE)
    # Patch.value is the Burgers vector b = u(+n) - u(-n); on the vertical
    # fault (n = +x, s_hat = +y) right-lateral slip is b = -slip_mag * s_hat.
    b = burgers(slip_mag, rake, meshes["n_hat"], meshes["s_hat"])

    def slipping(name):
        return Patch(name, meshes[name], BCType.FAULT, value=np.broadcast_to(
            b, (meshes[name].n_triangles, 3)))

    hx, zi, zb = meshes["half_x"], meshes["z_interface"], meshes["z_bottom"]
    plate = Region(
        PLATE, mat_plate,
        [Patch("top", meshes["top"], BCType.FREE_TRACTION, order=order_top),
         Patch("sides_upper", meshes["sides_upper"], side_bc), interface],
        probe_point=np.array([0.45 * hx, 0.45 * hx, 0.5 * zi]),
        faults=[slipping("fault")])
    substrate = Region(
        SUBSTRATE, mat_substrate,
        [interface,
         Patch("sides_lower", meshes["sides_lower"], BCType.FREE_TRACTION),
         Patch("base", meshes["base"], BCType.PRESCRIBED_DISPLACEMENT)],
        probe_point=np.array([0.45 * hx, 0.45 * hx, 0.5 * (zi + zb)]),
        faults=[slipping("fault_lower")] if "fault_lower" in meshes else [])
    gravity = None
    if density is not None:
        gravity = Gravity(density={PLATE: density[0], SUBSTRATE: density[1]},
                          surface=("top",),
                          **({} if g is None else {"g": g}))
    return RegionModel([plate, substrate], gravity=gravity)


def two_mesh_limit(coarse, fine, ratio):
    """Richardson's limit of a result whose error is FIRST ORDER in an element
    size, from two meshes with that size in ``ratio`` (coarse over fine):
    ``(ratio fine - coarse) / (ratio - 1)``, both at the same points.

    Here because a plate that BENDS needs it. Its relaxed vertical motion is
    short by an amount first order in the P0 elements of its two faces and in
    the interface's eps (docs/viscoelastic.md, section 7), so a pair must
    have all three in ONE ratio, and the limit then stands where a mesh
    several times finer would. It removes that error and no other: not the
    box, and not the fault's own band at the trace."""
    if not ratio > 1.0:
        raise ValueError(f"ratio {ratio} must exceed 1: coarse over fine")
    return (ratio * np.asarray(fine) - np.asarray(coarse)) / (ratio - 1.0)


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


# ---------------------------------------------------------------------
# the plane-strain reference: an infinitely long DIPPING fault
# ---------------------------------------------------------------------
#
# With exp(ikx) dependence and u_x = U, u_z = iW, tau_xz = T, tau_zz = iS,
# plane-strain elasticity is a real first-order system in z for (U, W, T, S),
# whose four solutions in a uniform layer are (1, kz) x exp(+-kz). Slip is an
# eigenstrain E = sym(b (x) n) on the fault, which enters that system as a
# JUMP of (U, W, T, S) at the source depth and nothing else. Buoyancy is the
# condition of docs/viscoelastic.md section 7 on the surface and on the
# interface. So a wavenumber costs one small linear solve, and everything is
# plain arithmetic in the substrate's moduli: real for a state, complex on an
# inversion contour. A fault that goes through the interface is two sources,
# each in the moduli of the layer it lies in, the lower one entering on the
# substrate's side of the interface conditions.

def _modes(sign, kz, mu, lam):
    """The two solutions ~ exp(sign k z) at local height ``kz = k (z - z0)``:
    ``(..., 4, 2)``, rows ``(U, W, T/k, S/k)``. ``z0`` is where the caller
    wants the exponential to be 1, so nothing here ever grows."""
    kz = np.asarray(kz, float)
    kappa = (lam + 3.0 * mu) / (lam + mu)
    e = np.exp(sign * kz)
    one, zero = np.ones_like(kz), np.zeros_like(kz)
    out = []
    # U = (a + b kz) e, W = (c + d kz) e: a potential gradient, and the
    # Papkovich-Neuber solution z grad(psi) - kappa psi z^
    for a, b, c, d in ((one, zero, -sign * one, zero),
                       (zero, one, kappa * one, -sign * one)):
        U, W = (a + b * kz) * e, (c + d * kz) * e
        dU = (sign * (a + b * kz) + b) * e
        dW = (sign * (c + d * kz) + d) * e
        out.append(np.stack([U, W, mu * (dU - W),
                             (lam + 2.0 * mu) * dW + lam * U], axis=-1))
    return np.stack(out, axis=-1)


def _layer_system(k, thickness, plate, substrate, weights, z_bottom):
    """``(A, top, bottom)``: the equations of a plate over a substrate at
    wavenumbers ``k`` for the plate's four and the substrate's two (a
    half-space) or four (a clamped base at ``z_bottom``) amplitudes, and the
    plate's modes at the surface and at the interface.

    Rows: at the surface ``T = 0`` and ``S + w_p W = 0``; at the interface
    ``U``, ``W``, ``T`` continuous and ``S_s - S_p + (w_s - w_p) W = 0``; at a
    base ``U = W = 0``. Traction rows are divided by ``mu_p k``.
    """
    (mu1, lam1), (mu2, lam2), (w1, w2) = plate, substrate, weights
    k = np.asarray(k, float)
    kh, zero = k * thickness, np.zeros(k.size)
    top = np.concatenate([_modes(+1, zero, mu1, lam1),
                          _modes(-1, kh, mu1, lam1)], axis=-1)
    bot = np.concatenate([_modes(+1, -kh, mu1, lam1),
                          _modes(-1, zero, mu1, lam1)], axis=-1)
    sub = _modes(+1, zero, mu2, lam2)
    if z_bottom is not None:
        kd = k * (-thickness - z_bottom)
        sub = np.concatenate([sub, _modes(-1, kd, mu2, lam2)], axis=-1)
        base = np.concatenate([_modes(+1, -kd, mu2, lam2),
                               _modes(-1, zero, mu2, lam2)], axis=-1)
    n = 4 + sub.shape[-1]
    A = np.zeros((k.size, n, n), complex)
    g1, g2 = (w1 / (mu1 * k))[:, None], ((w2 - w1) / (mu1 * k))[:, None]
    A[:, 0, :4] = top[:, 2] / mu1
    A[:, 1, :4] = top[:, 3] / mu1 + g1 * top[:, 1]
    A[:, 2, :4], A[:, 2, 4:] = bot[:, 0], -sub[:, 0]
    A[:, 3, :4], A[:, 3, 4:] = bot[:, 1], -sub[:, 1]
    A[:, 4, :4], A[:, 4, 4:] = bot[:, 2] / mu1, -sub[:, 2] / mu1
    A[:, 5, :4], A[:, 5, 4:] = g2 * bot[:, 1] - bot[:, 3] / mu1, sub[:, 3] / mu1
    if z_bottom is not None:
        A[:, 6, 4:], A[:, 7, 4:] = base[:, 0], base[:, 1]
    return A, top, bot


def _slip_field(k, down_dip, dip, material, z_above, z_below=None,
                n_source: int = 128):
    """The field, in an UNBOUNDED ``material``, of unit thrust slip on the
    stretch ``down_dip = (near, far)`` of the fault (distance from the trace),
    at the height ``z_above`` over it and ``z_below`` under it: ``(nk, 4)``
    each. It knows nothing of the other layer or of gravity."""
    mu, lam = material
    d = np.deg2rad(dip)
    n_hat = np.array([np.sin(d), np.cos(d)])
    b = np.array([-np.cos(d), np.sin(d)])          # the hanging wall, up dip
    exx, ezz = b[0] * n_hat[0], b[1] * n_hat[1]
    exz = 0.5 * (b[0] * n_hat[1] + b[1] * n_hat[0])
    # the jump (above minus below) of (U, W, T/k, S/k) at a source point
    jump = np.array([2.0 * exz,
                     -1j * (ezz + lam * exx / (lam + 2.0 * mu)),
                     4j * mu * (lam + mu) / (lam + 2.0 * mu) * exx, 0.0])
    here = np.concatenate([_modes(-1, 0.0, mu, lam),
                           -_modes(+1, 0.0, mu, lam)], axis=-1)
    amp = np.linalg.solve(here, jump)              # decaying up, decaying down
    node, weight = np.polynomial.legendre.leggauss(n_source)
    half = 0.5 * (down_dip[1] - down_dip[0])
    xi = down_dip[0] + half * (1.0 + node)         # distance down dip
    xs, zs = xi * np.cos(d), -xi * np.sin(d)
    phase = (half * weight)[None, :] * np.exp(-1j * k[:, None] * xs[None, :])

    def at(z, sign, a):
        return np.einsum("ks,ksij,j->ki", phase, _modes(
            sign, k[:, None] * (z - zs[None, :]), mu, lam), a)

    return (at(z_above, -1, amp[:2]),
            None if z_below is None else at(z_below, +1, amp[2:]))


def _dip_slip_source(k, depth, thickness, dip, plate):
    """The slip IN THE PLATE, at the surface and at the interface. A stack
    over ``s`` computes it once: the plate does not relax."""
    return _slip_field(k, (0.0, min(depth, thickness) / np.sin(np.deg2rad(dip))),
                       dip, plate, 0.0, -thickness)


def _dip_slip_deep(k, depth, thickness, dip, substrate, z_bottom):
    """The slip BELOW THE INTERFACE, in the substrate's moduli, at the
    interface and at the base; ``None`` for a fault that ends in the plate."""
    if depth <= thickness:
        return None
    sin_dip = np.sin(np.deg2rad(dip))
    return _slip_field(k, (thickness / sin_dip, depth / sin_dip), dip,
                       substrate, -thickness, z_bottom)


def _dip_slip_solve(k, source, thickness, plate, substrate, weights, z_bottom,
                    deep=None):
    """``(u_x^, u_z^)`` at the surface: the amplitudes that make the source
    fields satisfy the conditions of ``_layer_system``. ``source`` is the
    slip in the plate, ``deep`` the slip in the substrate."""
    (mu1, _lam1), (w1, w2), (y_top, y_bot) = plate, weights, source
    A, top, _ = _layer_system(k, thickness, plate, substrate, weights, z_bottom)
    rhs = np.zeros(A.shape[:2], complex)
    g1, g2 = w1 / (mu1 * k), (w2 - w1) / (mu1 * k)
    rhs[:, 0] = -y_top[:, 2] / mu1
    rhs[:, 1] = -(y_top[:, 3] / mu1 + g1 * y_top[:, 1])
    rhs[:, 2], rhs[:, 3] = -y_bot[:, 0], -y_bot[:, 1]
    rhs[:, 4] = -y_bot[:, 2] / mu1
    rhs[:, 5] = y_bot[:, 3] / mu1 - g2 * y_bot[:, 1]
    if deep is not None:
        y_int, y_base = deep
        rhs[:, 2] += y_int[:, 0]
        rhs[:, 3] += y_int[:, 1]
        rhs[:, 4] += y_int[:, 2] / mu1
        rhs[:, 5] -= y_int[:, 3] / mu1
        if z_bottom is not None:
            rhs[:, 6], rhs[:, 7] = -y_base[:, 0], -y_base[:, 1]
    c = np.linalg.solve(A, rhs[..., None])[..., 0]
    y = np.einsum("kij,kj->ki", top, c[:, :4]) + y_top
    return y[:, 0], 1j * y[:, 1]


def dip_slip_spectrum(k, depth, thickness, dip, plate, substrate,
                      weights=(0.0, 0.0), z_bottom=None):
    """``(u_x^, u_z^)`` at the surface and at wavenumbers ``k > 0``, per unit
    THRUST slip on a fault from the surface at x = 0 down to ``depth``,
    dipping toward +x at ``dip`` degrees: the transform over x of what
    ``dip_slip_profile`` returns."""
    k = np.asarray(k, float)
    return _dip_slip_solve(
        k, _dip_slip_source(k, depth, thickness, dip, plate), thickness,
        plate, substrate, weights, z_bottom,
        _dip_slip_deep(k, depth, thickness, dip, substrate, z_bottom))


def half_space_dip_slip(x, depth, dip):
    """``(u_x, u_z)`` at the surface of a uniform half-space without gravity,
    per unit thrust slip on the fault of ``dip_slip_spectrum`` (Freund &
    Barnett 1976). It does not depend on Poisson's ratio."""
    x = np.asarray(x, float)
    d = np.deg2rad(dip)
    zeta = (x - depth / np.tan(d)) / depth
    turn = np.arctan(zeta) - 0.5 * np.pi * np.sign(x)
    return ((np.cos(d) * turn
             + (np.sin(d) - zeta * np.cos(d)) / (1.0 + zeta ** 2)) / np.pi,
            -(np.sin(d) * turn
              + (np.cos(d) + zeta * np.sin(d)) / (1.0 + zeta ** 2)) / np.pi)


def _wavenumbers(x_max, k_max, k_min=1.0e-7, growth=0.25, order=8):
    """Gauss panels on ``[k_min, k_max]``: geometric from ``k_min``, where
    gravity and a deep base put structure, then no wider than a fraction of
    the fastest oscillation ``exp(i k x_max)``."""
    edges = [k_min]
    while edges[-1] < k_max:
        edges.append(edges[-1] + min(growth * edges[-1], 1.5 / x_max))
    edges = np.array(edges)
    node, weight = np.polynomial.legendre.leggauss(order)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    return ((mid[:, None] + half[:, None] * node).ravel(),
            (half[:, None] * weight).ravel())


def dip_slip_profile(x, depth, thickness, dip, plate, substrate,
                     weights=(0.0, 0.0), z_bottom=None, k_max: float = 4.0):
    """``(u_x, u_z)`` at the surface points ``x`` (either sign), per unit
    THRUST slip on an infinitely long fault that cuts an elastic plate from
    the surface at x = 0 down to ``depth``, dipping toward +x at ``dip``
    degrees, over a substrate: a half-space, or a layer clamped at
    ``z_bottom``. Deeper than the plate is thick, the fault goes on through
    the interface with the same slip, in the substrate.

    ``plate`` and ``substrate`` are ``(mu, lam)``, ``weights`` their
    ``rho g`` (GPa/km): buoyancy on the free surface and on the interface, as
    ``model.Gravity`` applies it. Handed ``(mu(s), lam(s))`` for the substrate
    this is the Laplace-domain reference; those may be complex. A LIST of
    substrates returns the stack, ``(n, n_x)`` each, for the price of one.

    Computed as the half-space closed form plus the transform of the
    DIFFERENCE from it, which is smooth in x: the slip at the trace, which
    would make the transform converge like 1/k, is common to both.
    """
    x = np.atleast_1d(np.asarray(x, float))
    many = np.ndim(substrate[0]) > 0 or isinstance(substrate[0], (tuple, list))
    k, wk = _wavenumbers(max(float(np.abs(x).max()), thickness), k_max)
    phase = np.exp(1j * np.outer(x, k)) * wk
    source = _dip_slip_source(k, depth, thickness, dip, plate)
    # the half-space of the closed form has no interface for the fault to go
    # through: all of it in a plate thick enough to hold it
    whole = max(depth, thickness)
    free = _dip_slip_solve(k, _dip_slip_source(k, depth, whole, dip, plate),
                           whole, plate, plate, (0.0, 0.0), None)

    def half(sub):
        got = _dip_slip_solve(
            k, source, thickness, plate, sub, weights, z_bottom,
            _dip_slip_deep(k, depth, thickness, dip, sub, z_bottom))
        return [phase @ (a - b) for a, b in zip(got, free)]

    def one(sub):
        # u^(-k) = conj(u^(k)) at the CONJUGATE moduli, so real moduli need
        # one half-line and complex ones two
        plus = half(sub)
        if np.iscomplexobj(np.asarray(sub)):
            minus = half(tuple(np.conj(m) for m in sub))
            return [(p + np.conj(m)) / (2.0 * np.pi)
                    for p, m in zip(plus, minus)]
        return [p.real / np.pi for p in plus]

    closed = half_space_dip_slip(x, depth, dip)
    if not many:
        return tuple(c + dl for c, dl in zip(closed, one(substrate)))
    rows = [one(sub) for sub in substrate]
    return tuple(c + np.array([r[i] for r in rows])
                 for i, c in enumerate(closed))
