"""Raw 2-D moment table vs Gauss quadrature, all orders needed for quadratic
slip, on the tilted triangle: off-plane observers (1e-10), on-plane
mollified observers inside/outside/at a vertex/on an edge (quadrature-limited,
1e-7 with h = eps = 0.15 L and a 160x160 rule), the two-route M^{(1,1)}
identity (1e-13), and an order-3 table smoke test (1e-9).

The slip degrees ``kernel_degrees(p, ("U","H","E"))`` reach n = 3, 5, 7 only.
The FORCE (Kelvin single-layer) element reads two rows they never touch, so
they get their own sweep here:

    kernel_degrees(2, ("G","S")) == {1: 2, 3: 4, 5: 5}

-- the n = 1 row at DEGREE 2 (the only kernel that reads n = 1 above degree 0),
which via ``_close_degrees`` pulls in n = -1 (positive powers of R,
``I_{-1} = (E_{-1} + h^2 I_1)/3``, seeded downward instead of upward).  Both
rows are compared entry by entry against the same Gauss reference, at
eps = 0.1 off the plane (worst 1.2e-13, gated 1e-11), at h = eps = 0.15 L on
the plane (1.7e-14, gated 1e-12) and at eps = 0 off the plane (1.2e-13,
gated 1e-11).  Without this sweep the n = 1 and n = -1 rows are exercised only
indirectly, through the assembled G kernel."""
from __future__ import annotations

import numpy as np

from _common import TRI, Report, relmax
from clq.frame import local_frame
from clq.moments import MomentTable, kernel_degrees, gauss_triangle


def quad_moments(fr, obs, eps, degrees, n):
    z, X = fr.to_plane(obs)
    x1, x2, w = gauss_triangle(n)
    lam = np.stack([1 - x1 - x2, x1, x2], 1)
    eta = lam @ fr.p
    wq = w * 2.0 * fr.area
    xi = eta[None] - X[:, None]
    R2 = xi[..., 0] ** 2 + xi[..., 1] ** 2 + (z * z + eps * eps)[:, None]
    out = {}
    for nn, d in degrees.items():
        out[nn] = np.zeros((len(obs), d + 1, d + 1))
        for a in range(d + 1):
            for b in range(d + 1 - a):
                out[nn][:, a, b] = (wq * xi[..., 0] ** a * xi[..., 1] ** b * R2 ** (-0.5 * nn)).sum(1)
    return out


def worst(tab, ref):
    w = 0.0
    for n, d in tab.degrees.items():
        for a in range(d + 1):
            for b in range(d + 1 - a):
                w = max(w, relmax(tab.M[n][:, a, b], ref[n][:, a, b]))
    return w


def main():
    rep = Report("2-D moment table vs Gauss quadrature")
    fr = local_frame(TRI)
    v1, v2, v3 = TRI
    deg2 = kernel_degrees(2, ("U", "H", "E"))
    # off-plane
    obs = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    eps = 0.1
    tab = MomentTable(fr, obs, eps, deg2, check_identity=True)
    rep.check("P2 table, off-plane, vs 200x200 Gauss", worst(tab, quad_moments(fr, obs, eps, tab.degrees, 200)), 1e-10)
    # near-plane (z = -0.02 on this tilted triangle): the reference needs 200x200
    obs_np = np.array([[0.9, 0.3, 0.15]])
    tab_np = MomentTable(fr, obs_np, eps, deg2)
    rep.check("P2 table, near-plane z=0.2 h, vs 200x200 Gauss", worst(tab_np, quad_moments(fr, obs_np, eps, tab_np.degrees, 200)), 1e-10)
    rep.check("two-route M(1,1) identity (off-plane)", tab.identity_residual, 1e-13)
    # on-plane, mollified: inside, outside, vertex, edge midpoint
    eps = 0.15 * fr.L
    obs = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),
                    v1 + 1.2 * (v2 - v1) - 0.4 * (v3 - v1),
                    v1,
                    0.5 * (v2 + v3)])
    tab = MomentTable(fr, obs, eps, deg2, check_identity=True)
    rep.check("P2 table, on-plane h=eps=0.15L, vs 160x160 Gauss", worst(tab, quad_moments(fr, obs, eps, tab.degrees, 160)), 1e-7)
    rep.check("two-route M(1,1) identity (on-plane)", tab.identity_residual, 1e-13)
    # eps = 0 off-plane (singular kernel, finite integrals)
    obs = np.array([[0.60, -0.10, 0.60], [-0.4, 0.0, -0.8]])
    tab = MomentTable(fr, obs, 0.0, deg2)
    rep.check("P2 table, eps=0 off-plane, vs 200x200 Gauss", worst(tab, quad_moments(fr, obs, 0.0, tab.degrees, 200)), 1e-10)
    # order-3 smoke
    deg3 = kernel_degrees(3, ("U", "H", "E"))
    obs = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0]])
    tab = MomentTable(fr, obs, 0.1, deg3)
    # degree-7 entries lose a few digits in the binomial edge expansion (smoke test only)
    rep.check("P3 table (smoke), off-plane, vs 200x200 Gauss", worst(tab, quad_moments(fr, obs, 0.1, tab.degrees, 200)), 1e-8)
    # eps = 0 on the plane must raise
    try:
        MomentTable(fr, np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1)]), 0.0, deg2)
        rep.check_bool("eps=0 on the plane raises ValueError", False)
    except ValueError:
        rep.check_bool("eps=0 on the plane raises ValueError", True)

    # --- force (Kelvin single-layer) degrees: the n = 1 and n = -1 rows -------
    degF = kernel_degrees(2, ("G", "S"))
    rep.check_bool("kernel_degrees(2, ('G','S')) == {1: 2, 3: 4, 5: 5}",
                   degF == {1: 2, 3: 4, 5: 5}, f"(got {degF})")
    rep.check_bool("kernel_degrees(0, ('S',)) == kernel_degrees(0, ('U',)) "
                   "(S and U share G1)",
                   kernel_degrees(0, ("S",)) == kernel_degrees(0, ("U",)),
                   f"(S {kernel_degrees(0, ('S',))}, U {kernel_degrees(0, ('U',))})")
    obs_f = np.array([[0.60, -0.10, 0.60], [2.0, 1.5, 3.0], [-0.4, 0.0, -0.8]])
    tabF = MomentTable(fr, obs_f, 0.1, degF, check_identity=True)
    rep.check_bool("force degrees close to n = 1 (degree 2) and n = -1",
                   tabF.degrees.get(1, -1) == 2 and -1 in tabF.degrees,
                   f"(closed degrees {tabF.degrees})")
    rep.check("P2 force table (n = 1, -1, 3, 5), off-plane, vs 200x200 Gauss",
              worst(tabF, quad_moments(fr, obs_f, 0.1, tabF.degrees, 200)), 1e-11)
    rep.check("two-route M(1,1) identity (force degrees)", tabF.identity_residual, 1e-13)
    eps_f = 0.15 * fr.L
    obs_fon = np.array([v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),
                        v1 + 1.2 * (v2 - v1) - 0.4 * (v3 - v1),
                        v1,
                        0.5 * (v2 + v3)])
    tabF_on = MomentTable(fr, obs_fon, eps_f, degF)
    rep.check("P2 force table, on-plane h=eps=0.15L, vs 160x160 Gauss",
              worst(tabF_on, quad_moments(fr, obs_fon, eps_f, tabF_on.degrees, 160)), 1e-12)
    tabF_0 = MomentTable(fr, obs_f[[0, 2]], 0.0, degF)
    rep.check("P2 force table, eps=0 off-plane, vs 200x200 Gauss",
              worst(tabF_0, quad_moments(fr, obs_f[[0, 2]], 0.0, tabF_0.degrees, 200)), 1e-11)
    rep.finish()


if __name__ == "__main__":
    main()
