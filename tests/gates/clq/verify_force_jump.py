"""Physics of the force (single-layer) element: jump, continuity, equilibrium.

Every check here is convention-free -- none of them consults another kernel
implementation, so together they pin the force element's sign, its absolute
normalisation and its units.

(a) TRACTION JUMP.  Across the mollified layer the traction jumps by the
    density, smeared by the blob's fault-normal marginal:

        t(x + z0 n) - t(x - z0 n) = -f(z0/eps) * density(x),
        f(t) = t (2 t^2 + 3) / (2 (1 + t^2)^1.5)      (f(3) = 0.99612),

    the SAME profile as the displacement jump of a slip element
    (verify_jump.py) -- both are the marginal of the same blob.  On a finite
    triangle the residual is the finite-size deficit ~ z0 / (distance to the
    nearest edge); doubling L must shrink it, and at L = 100 it is below 1e-3.

(b) DISPLACEMENT CONTINUITY.  The single layer itself is continuous: this is
    the clq-level statement of why it carries NO 1/2 I free term in a BEM
    (src/mbem/model/equations.py puts the jump on the double layer alone).

(c) GLOBAL EQUILIBRIUM.  For any closed surface S enclosing the element,
    ``int_S sigma . n dS = -int_T f dS``, which follows from
    ``div sigma + phi_eps * f = 0`` and the blob's unit mass.  Zero free
    parameters, no oracle.

(d) FAR-FIELD POINT FORCE.  At distance D the element looks like a point force
    ``F = int_T f dS`` at the centroid: order 2 for P0 (the area centroid kills
    the dipole) and order 1 for P1/P2 (it does not).  The rates are gated, not
    the values.
"""
from __future__ import annotations

import numpy as np

from _common import MU, Report, relmax
import clq
from clq.frame import local_frame
from clq.moments import gauss_triangle
from clq.pointwise import kelvin_G
from clq.shape import shape_functions, interpolate

NUS = (0.25, 0.30, 0.35, 0.45)
EPS = 1.0e-3
Z0 = 3.0 * EPS
R_REL = 0.12


def f_profile(t: float) -> float:
    return t * (2.0 * t * t + 3.0) / (2.0 * (1.0 + t * t) ** 1.5)


def nodal_force(order, seed=3):
    rng = np.random.default_rng(seed)
    K = (order + 1) * (order + 2) // 2
    return rng.normal(size=(K, 3))


def interior_points(tri, n=6):
    """Centroid plus a ring, all well inside the triangle."""
    fr = local_frame(tri)
    c = tri.mean(0)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = R_REL * fr.L
    return np.vstack([c[None, :],
                      c + r * (np.cos(th)[:, None] * fr.e1 + np.sin(th)[:, None] * fr.e2)])


def shape_integrals(tri, order, n_gauss=20):
    """int_T N_k dS (K,) -- the nodal weights of the total force."""
    fr = local_frame(tri)
    x1, x2, w = gauss_triangle(n_gauss)
    y = ((1 - x1 - x2)[:, None] * tri[0] + x1[:, None] * tri[1] + x2[:, None] * tri[2])
    return (w * 2.0 * fr.area) @ shape_functions(tri, order, y)


def check_traction_jump(rep):
    for L, tol, label in ((1.0, 1.5e-2, "L=1"), (2.0, 1.5e-2, "L=2"),
                          (100.0, 1.0e-3, "L=100")):
        tri = clq.equilateral(L)
        fr = local_frame(tri)
        pts = interior_points(tri)
        for order in (0, 1, 2):
            dens = nodal_force(order)
            scale = np.max(np.linalg.norm(dens, axis=1))
            for nu in NUS:
                sp = clq.force_stress(pts + Z0 * fr.nhat, tri, dens, MU, nu, EPS)
                sm = clq.force_stress(pts - Z0 * fr.nhat, tri, dens, MU, nu, EPS)
                jump = (sp - sm) @ fr.nhat
                want = -f_profile(Z0 / EPS) * interpolate(tri, dens, pts)
                err = np.max(np.abs(jump - want)) / scale
                rep.check(f"[{label}] P{order} nu={nu}: traction jump = -f(z0/eps) f",
                          err, tol)


def check_displacement_continuity(rep):
    tri = clq.equilateral(10.0)
    fr = local_frame(tri)
    pts = interior_points(tri)
    for order in (0, 1, 2):
        dens = nodal_force(order)
        up = clq.force_displacement(pts + Z0 * fr.nhat, tri, dens, MU, 0.3, EPS)
        um = clq.force_displacement(pts - Z0 * fr.nhat, tri, dens, MU, 0.3, EPS)
        rep.check(f"P{order}: displacement continuous across the layer",
                  np.max(np.abs(up - um)) / np.max(np.abs(up)), 1.0e-3)


def sphere_rule(radius, center, n_theta=60, n_phi=120):
    ct, wt = np.polynomial.legendre.leggauss(n_theta)      # cos(theta) in [-1,1]
    st = np.sqrt(1.0 - ct ** 2)
    ph = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    wp = 2.0 * np.pi / n_phi
    nx = np.outer(st, np.cos(ph)).ravel()
    ny = np.outer(st, np.sin(ph)).ravel()
    nz = np.repeat(ct, n_phi)
    nrm = np.stack([nx, ny, nz], axis=1)
    w = np.repeat(wt, n_phi) * wp * radius ** 2
    return center + radius * nrm, nrm, w


def check_equilibrium(rep):
    """int_S sigma.n dS = -int_T f dS for a sphere enclosing the element."""
    tri = clq.equilateral(1.0)
    c = tri.mean(0)
    for order in (0, 1, 2):
        dens = nodal_force(order)
        Ftot = shape_integrals(tri, order) @ dens          # (3,)
        for radius, tol in ((3.0, 1.0e-6), (8.0, 1.0e-6)):
            pts, nrm, w = sphere_rule(radius, c)
            sig = clq.force_stress(pts, tri, dens, MU, 0.3, 0.05)
            net = np.einsum("q,qij,qj->i", w, sig, nrm)
            rep.check(f"P{order}: closed-surface equilibrium, R = {radius} L",
                      np.max(np.abs(net + Ftot)) / np.max(np.abs(Ftot)), tol)


def check_far_field(rep):
    """u -> G^eps(x - centroid) . F, order 2 for P0 and order 1 for P1/P2."""
    tri = clq.equilateral(1.0)
    c = tri.mean(0)
    direction = np.array([0.37, -0.52, 0.77])
    direction /= np.linalg.norm(direction)
    for order in (0, 1, 2):
        dens = nodal_force(order)
        Ftot = shape_integrals(tri, order) @ dens
        errs = []
        for D in (10.0, 20.0, 40.0, 80.0):
            x = (c + D * direction)[None, :]
            u = clq.force_displacement(x, tri, dens, MU, 0.3, 0.05)
            u_pt = kelvin_G(x - c, MU, 0.3, 0.05)[0] @ Ftot
            errs.append(np.max(np.abs(u[0] - u_pt)) / np.max(np.abs(u_pt)))
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        expect = 4.0 if order == 0 else 2.0            # doubling D
        ok = all(0.7 * expect < r < 1.4 * expect for r in ratios)
        rep.check_bool(f"P{order}: point-force limit, order {2 if order == 0 else 1}",
                       ok, extra=" ".join(f"{e:.2e}" for e in errs))


def main():
    rep = Report("force element: jump, continuity, equilibrium, far field")
    check_traction_jump(rep)
    check_displacement_continuity(rep)
    check_equilibrium(rep)
    check_far_field(rep)
    rep.finish()


if __name__ == "__main__":
    main()
