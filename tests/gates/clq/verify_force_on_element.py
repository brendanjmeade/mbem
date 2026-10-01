"""Force (single-layer) kernel at eps = 0, including observers ON the element.

The Kelvin single layer is weakly singular: ``int_T N_k / r dS`` converges for
an observer in the plane of the triangle, unlike every dislocation kernel
(whose ``I_5``, ``I_7`` diverge like ``h^-3``, ``h^-5``).  ``clq`` therefore
lifts its ``eps = 0`` on-plane guard for ``want=("G",)`` alone, via a degree
floor (:func:`clq.moments.h0_floor`): at ``h = 0`` the seeds decouple,

    I_1 = E_1,    I_{-1} = E_{-1}/3,    I_3 = undefined,

and only the ``a+b >= 2`` slots of the ``n = 3`` table are needed, because
``z = 0`` kills every out-of-plane slot of the rank-2 lift.

Checks
------
(a) on-element P0/P1/P2 against an angle-parameterised polar oracle (the
    ``r dr`` Jacobian cancels the 1/r singularity analytically; a product
    Gauss rule on the triangle does NOT converge here).  Three triangles: a
    generic tilted one, an axis-aligned one -- where every vertex and edge
    midpoint gives ``d_perp == 0`` bit-exactly, i.e. the rho = 0 primitives --
    and a sliver (h/L = 0.02).  Observers: a barycentric grid plus all six P2
    nodes plus the centroid, so vertices and edge points are included.
(b) the eps -> 0 ladder: the mollified on-element value converges to the
    eps = 0 one at FIRST order, with the predicted coefficient
    ``I_1(eps) = E_1 - Omega_2D eps + O(eps^2)`` where ``Omega_2D`` is the
    in-plane angle subtended by the triangle: 2 pi inside, pi on an open edge,
    the interior angle at a vertex, 0 outside.
(c) no free term: the off-plane path converges to the h = 0 value from BOTH
    sides at first order in |z| (the single layer is continuous; it is its
    gradient that jumps).  This needs no oracle at all.
(d) the internal identity ``M_3^(2,0) + M_3^(0,2) = M_1^(0,0)`` at h = 0 (the
    vertical identity with ``h^2 I_3 = 0``), and finiteness at every node.
(e) the guards still hold: any slip kernel, or a mixed ``want``, still raises
    on the plane at eps = 0, and ``far_field="quadrature"`` refuses.
"""
from __future__ import annotations

import numpy as np

from _common import MU, Report, relmax
import clq
from clq.frame import local_frame
from clq.moments import MomentTable, kernel_degrees, h0_floor
from clq.shape import shape_functions, nodes, barycentric_grid

NU = 0.3
TRIS = {
    "tilted": np.array([[0.37, -0.81, 0.44], [1.92, 0.11, -0.63], [-0.25, 1.57, 1.22]]),
    "axis": np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.1, 0.0]]),
    "sliver": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.02, 0.0]]),
}
TOL_POLAR = {"tilted": 1.0e-12, "axis": 1.0e-12, "sliver": 1.0e-10}


def polar_force_G(x2, tri2d, shape_fn, e1, e2, mu, nu, nphi=400, nt=12):
    """int_T N_k(y) G^0(x - y) dS for an observer IN the plane, by wedge
    integration with the radial factor done analytically (r dr / r = dr)."""
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    gt, gw = np.polynomial.legendre.leggauss(nt)
    gt, gw = 0.5 * (gt + 1.0), 0.5 * gw
    gp, gwp = np.polynomial.legendre.leggauss(nphi)
    K = shape_fn(np.zeros((1, 2))).shape[1]
    out = np.zeros((K, 3, 3))
    for e in range(3):
        pa, pb = tri2d[e] - x2, tri2d[(e + 1) % 3] - x2
        if pa[0] * pb[1] - pa[1] * pb[0] == 0.0:
            continue                                   # observer on this edge line
        ta, tb = np.arctan2(pa[1], pa[0]), np.arctan2(pb[1], pb[0])
        dth = (tb - ta + np.pi) % (2 * np.pi) - np.pi
        phi = ta + 0.5 * dth * (gp + 1.0)
        wphi = 0.5 * dth * gwp
        c, s = np.cos(phi), np.sin(phi)
        nx, ny = pa[1] - pb[1], pb[0] - pa[0]
        nn = np.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        dperp = pa[0] * nx + pa[1] * ny
        if dperp == 0.0:
            continue                                   # zero-area wedge
        Rmax = dperp / (c * nx + s * ny)
        for it in range(nt):
            r = gt[it] * Rmax
            w = gw[it] * Rmax * wphi * r                # r dr dphi
            y2 = x2 + np.stack([r * c, r * s], axis=1)
            d = -(np.outer(r * c, e1) + np.outer(r * s, e2))    # x - y, global
            G = ((c34 / r)[:, None, None] * np.eye(3)[None]
                 + np.einsum("ni,nj,n->nij", d, d, 1.0 / r ** 3))
            out += C1 * np.einsum("n,nk,nij->kij", w, shape_fn(y2), G)
    return out


def on_element_points(tri):
    return np.vstack([barycentric_grid(4) @ tri, nodes(tri, 2), tri.mean(0)[None, :]])


def check_polar(rep):
    for name, tri in TRIS.items():
        fr = local_frame(tri)
        pts = on_element_points(tri)
        tri2d = np.stack([fr.to_plane(v[None])[1][0] for v in tri])
        for order in (0, 1, 2):
            got = clq.influence(pts, tri, MU, NU, 0.0, order=order, want=("G",)).G
            rep.check_bool(f"[{name}] P{order} finite on the element",
                           bool(np.all(np.isfinite(got))))
            shp = (lambda y2, o=order, t=tri, f=fr:
                   shape_functions(t, o, f.from_plane(y2)))
            worst = 0.0
            for i, x in enumerate(pts):
                ref = polar_force_G(fr.to_plane(x[None])[1][0], tri2d, shp,
                                    fr.e1, fr.e2, MU, NU)
                worst = max(worst, relmax(got[i], ref))
            rep.check(f"[{name}] P{order} vs polar oracle ({len(pts)} obs)",
                      worst, TOL_POLAR[name])


def omega_2d(tri, x):
    """In-plane angle subtended by the triangle at an in-plane point: 2 pi
    strictly inside, pi on an open edge, the interior angle at a vertex, the
    subtended wedge outside.  The polygon is oriented counter-clockwise first
    so that a collinear-opposite edge contributes +pi with the right sign."""
    fr = local_frame(tri)
    x2 = fr.to_plane(x[None])[1][0]
    p = np.stack([fr.to_plane(v[None])[1][0] for v in tri])
    area2 = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
             - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
    if area2 < 0.0:
        p = p[::-1]
    q = p - x2
    tot = 0.0
    for e in range(3):
        a, b = q[e], q[(e + 1) % 3]
        if np.hypot(*a) == 0.0 or np.hypot(*b) == 0.0:
            continue                      # observer at a vertex: edge drops out
        cross = a[0] * b[1] - a[1] * b[0]
        if cross == 0.0 and a @ b < 0.0:
            continue                      # observer ON this edge: zero-area wedge
        tot += np.arctan2(cross, a @ b)
    return abs(tot)


def check_eps_ladder(rep):
    """I_1(eps) = E_1 - Omega_2D eps + O(eps^2), and G(eps) -> G(0) at order 1."""
    tri = TRIS["tilted"]
    fr = local_frame(tri)
    v1, v2, v3 = tri
    spots = {"interior": tri.mean(0),
             "edge midpoint": 0.5 * (v2 + v3),
             "vertex": v1,
             "exterior": tri.mean(0) + 3.0 * (v1 - tri.mean(0))}
    need = kernel_degrees(0, ("G",))
    for label, x in spots.items():
        x = np.asarray(x, float)[None, :]
        tab0 = MomentTable(fr, x, 0.0, need, floor=h0_floor(need))
        E1 = tab0.M[1][0, 0, 0]
        eps = 1.0e-6
        tab = MomentTable(fr, x, eps, need)
        coeff = (E1 - tab.M[1][0, 0, 0]) / eps
        rep.check(f"I_1 first-order coefficient, {label}",
                  abs(coeff - omega_2d(tri, x[0])) / max(omega_2d(tri, x[0]), 1.0),
                  2.0e-3, extra=f"{coeff:.5f} vs Omega_2D {omega_2d(tri, x[0]):.5f}")
    # first-order convergence of the kernel itself
    x = tri.mean(0)[None, :]
    G0 = clq.influence(x, tri, MU, NU, 0.0, order=2, want=("G",)).G
    errs = []
    for eps in (1e-2, 1e-3, 1e-4):
        Ge = clq.influence(x, tri, MU, NU, eps, order=2, want=("G",)).G
        errs.append(relmax(Ge, G0))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    rep.check_bool("G(eps) -> G(0), first order (ratio ~ 10 per decade)",
                   all(8.0 < r < 12.0 for r in ratios),
                   extra=" ".join(f"{e:.2e}" for e in errs))


def check_no_free_term(rep):
    """The off-plane path converges to the h = 0 value from BOTH sides: the
    single layer is continuous across the element (no 1/2 I free term)."""
    tri = TRIS["tilted"]
    fr = local_frame(tri)
    x0 = tri.mean(0)
    G0 = clq.influence(x0[None], tri, MU, NU, 0.0, order=2, want=("G",)).G
    L = fr.L
    for sgn, side in ((+1.0, "+z"), (-1.0, "-z")):
        errs = []
        for q in (1e-3, 1e-4, 1e-5):
            x = x0 + sgn * q * L * fr.nhat
            Gz = clq.influence(x[None], tri, MU, NU, 0.0, order=2, want=("G",)).G
            errs.append(relmax(Gz, G0))
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        rep.check_bool(f"off-plane -> on-plane from {side}, first order",
                       all(6.0 < r < 14.0 for r in ratios),
                       extra=" ".join(f"{e:.2e}" for e in errs))
        rep.check(f"|G({side}) - G(0)| at |z| = 1e-5 L", errs[-1], 2.0e-4)
    # the two sides approach each other at first order: the single layer is
    # continuous (it is its GRADIENT that jumps), so the difference is O(|z|),
    # not small at any fixed z -- gate the rate, not the value.
    jumps = []
    for q in (1e-3, 1e-4, 1e-5):
        Gp = clq.influence((x0 + q * L * fr.nhat)[None], tri, MU, NU, 0.0,
                           order=2, want=("G",)).G
        Gm = clq.influence((x0 - q * L * fr.nhat)[None], tri, MU, NU, 0.0,
                           order=2, want=("G",)).G
        jumps.append(relmax(Gp, Gm))
    ratios = [jumps[i] / jumps[i + 1] for i in range(len(jumps) - 1)]
    rep.check_bool("|G(+z) - G(-z)| -> 0 at first order (continuity)",
                   all(6.0 < r < 14.0 for r in ratios),
                   extra=" ".join(f"{j:.2e}" for j in jumps))


def check_internal_identity(rep):
    """At h = 0 the vertical identity reads M_3^(2,0) + M_3^(0,2) = M_1^(0,0)
    (the h^2 I_3 coupling term vanishes)."""
    need = kernel_degrees(0, ("G",))
    for name, tri in TRIS.items():
        fr = local_frame(tri)
        pts = on_element_points(tri)
        tab = MomentTable(fr, pts, 0.0, need, floor=h0_floor(need))
        lhs = tab.M[3][:, 2, 0] + tab.M[3][:, 0, 2]
        rhs = tab.M[1][:, 0, 0]
        rep.check(f"[{name}] M_3^(2,0) + M_3^(0,2) == M_1^(0,0) at h = 0",
                  relmax(lhs, rhs), 1.0e-12)


def check_mixed_batch(rep):
    """A batch that MIXES on-plane and off-plane observers.

    The eps = 0 relaxation applies a degree floor, which leaves the sub-floor
    table slots NaN; that is correct on-plane (z is exactly 0, so `lift` masks
    them) and WRONG off-plane, where a nonzero z^c would multiply the NaN.
    Building one floored table for the whole near-set therefore returned NaN
    for every off-plane row of a mixed batch -- silently, with no warning and
    no raise.  Every check here was written after that bug, so it exists only
    to keep it fixed: each row of a mixed batch must equal the value the same
    point gets on its own, BITWISE.
    """
    for name in ("tilted", "axis", "sliver"):
        tri = TRIS[name]
        nh = local_frame(tri).nhat
        c = tri.mean(0)
        on = [c, 0.5 * (tri[0] + tri[1]), tri[2]]            # centroid, midpoint, vertex
        off = [c + d * nh for d in (0.05, 0.37, -0.22, 2.5)]
        f = np.array([0.3, -0.7, 0.5])

        pts = []
        for i in range(max(len(on), len(off))):              # interleaved, not blocked
            if i < len(on):
                pts.append(on[i])
            if i < len(off):
                pts.append(off[i])
        pts = np.array(pts)

        batch = clq.force_displacement(pts, tri, f, MU, NU, 0.0)
        solo = np.array([clq.force_displacement(x, tri, f, MU, NU, 0.0) for x in pts])

        rep.check_bool(f"{name}: mixed on/off-plane batch is finite",
                       bool(np.all(np.isfinite(batch))))
        rep.check_bool(f"{name}: mixed batch == one-at-a-time, bitwise",
                       bool(np.array_equal(batch, solo)))
        # and the ordering must not matter
        perm = np.array([5, 0, 6, 2, 1, 4, 3])[:len(pts)]
        rep.check_bool(f"{name}: mixed batch is order-independent",
                       bool(np.array_equal(
                           clq.force_displacement(pts[perm], tri, f, MU, NU, 0.0),
                           solo[perm])))


def check_guards(rep):
    tri = TRIS["tilted"]
    x = tri.mean(0)[None, :]

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    rep.check_bool("slip displacement still raises at eps = 0 on-plane",
                   raises(lambda: clq.displacement(x, tri, [1.0, 0, 0], MU, NU, 0.0)))
    rep.check_bool("slip stress still raises at eps = 0 on-plane",
                   raises(lambda: clq.stress(x, tri, [1.0, 0, 0], MU, NU, 0.0)))
    for w in (("G", "H"), ("G", "U"), ("G", "S"), ("G", "E")):
        rep.check_bool(f"mixed want {w} raises at eps = 0 on-plane",
                       raises(lambda w=w: clq.influence(x, tri, MU, NU, 0.0,
                                                        order=0, want=w)))
    rep.check_bool("far_field='quadrature' refuses an on-element eps = 0 observer",
                   raises(lambda: clq.influence(x, tri, MU, NU, 0.0, order=0,
                                                want=("G",), far_field="quadrature")))
    rep.check_bool("eps = 0 OFF the plane still works for the slip kernels",
                   np.all(np.isfinite(clq.displacement(tri.mean(0) + local_frame(tri).nhat,
                                                       tri, [1.0, 0, 0], MU, NU, 0.0))))


def main():
    rep = Report("force kernel at eps = 0, on the element")
    check_polar(rep)
    check_eps_ladder(rep)
    check_no_free_term(rep)
    check_internal_identity(rep)
    check_mixed_batch(rep)
    check_guards(rep)
    rep.finish()


if __name__ == "__main__":
    main()
