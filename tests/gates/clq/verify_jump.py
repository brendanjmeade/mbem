"""Displacement-jump recovery of the slip (convention-free, nu sweep).

For an infinite mollified plane the displacement jump measured between the
two points ``x +- z0 nhat`` is ``f(z0/eps) * slip(x)`` with

    f(t) = t (2 t^2 + 3) / (2 (1 + t^2)^1.5)        (f(3) = 0.99612),

independent of nu and of every index-pairing convention.  On a finite
equilateral triangle (L = 1, eps = L/1000, z0 = 3 eps) at interior points
(centroid + 6 points on a circle of radius 0.12 L) the jump must reproduce
``f(3) * interpolate(tri, slip, x)`` for p = 0, 1, 2 and nu in
{0.25, 0.30, 0.35, 0.45} to 1.5e-2 of max|slip| (max nodal slip vector
magnitude) -- the residual is the finite-size deficit
~ z0 / (distance to the nearest edge) ~ 1%.  Doubling L with the same eps and
z0 (interior points scaled with L) must shrink that deficit (check_bool per
(p, nu)); at L = 100 it must be below 1e-3 (the deficit is O(z0 / L): it
halves exactly from L = 1 to 2 and reads 1e-4 at L = 100).

A wrong Lame contraction (msd's pre-fix lam/mu swap on the displacement
kernel) would read ~2.3 instead of ~1 at nu = 0.35, so the nu sweep is the
tripwire.
"""
from __future__ import annotations

import numpy as np

from _common import MU, Report
import clq
import sys


EPS = 1.0e-3
Z0 = 3.0 * EPS
R_REL = 0.12
NUS = (0.25, 0.30, 0.35, 0.45)


def f_profile(t: float) -> float:
    return t * (2.0 * t * t + 3.0) / (2.0 * (1.0 + t * t) ** 1.5)


def interior_points(tri, radius: float) -> np.ndarray:
    """Centroid plus 6 points on a circle of the given radius about it (in-plane)."""
    fr = clq.local_frame(tri)
    ang = np.arange(6) * (np.pi / 3.0)
    X = np.vstack([[0.0, 0.0], radius * np.stack([np.cos(ang), np.sin(ang)], 1)])
    return fr.from_plane(X, 0.0)


def nodal_slips() -> dict[int, np.ndarray]:
    out = {0: np.array([[0.7, -0.4, 0.5]])}
    for p in (1, 2):
        rng = np.random.default_rng(1)
        out[p] = rng.standard_normal((clq.n_nodes(p), 3))
    return out


def jump_error(tri, slip, pts, nu):
    """(max|jump - expected| / max|slip|, recovery ratio) at the points."""
    n = clq.unit_normal(tri)
    u_plus = clq.displacement(pts + Z0 * n, tri, slip, MU, nu, EPS, far_field="analytic")
    u_minus = clq.displacement(pts - Z0 * n, tri, slip, MU, nu, EPS, far_field="analytic")
    jump = u_plus - u_minus
    expected = f_profile(Z0 / EPS) * clq.interpolate(tri, slip, pts)
    err = np.max(np.abs(jump - expected)) / np.max(np.linalg.norm(slip, axis=1))
    ratio = np.sum(jump * expected) / np.sum(expected * expected)
    return err, ratio


def main():
    rep = Report("displacement-jump recovery of the slip (L = 1 gate, L = 2 deficit shrinks)")
    print(f"  eps = {EPS:g}, z0 = {Z0:g} = 3 eps, f(3) = {f_profile(3.0):.5f}")
    slips = nodal_slips()
    tris = {L: clq.equilateral(L) for L in (1.0, 2.0)}
    pts = {L: interior_points(tris[L], R_REL * L) for L in (1.0, 2.0)}
    assert np.all(clq.inside(tris[1.0], pts[1.0], margin=0.05))
    print(f"  {'p':>2s} {'nu':>5s} {'err(L=1)':>10s} {'ratio(L=1)':>11s} "
          f"{'err(L=2)':>10s} {'ratio(L=2)':>11s} {'err2/err1':>10s}")
    worst = 0.0
    for p in (0, 1, 2):
        slip = slips[p]
        for nu in NUS:
            e1, r1 = jump_error(tris[1.0], slip, pts[1.0], nu)
            e2, r2 = jump_error(tris[2.0], slip, pts[2.0], nu)
            worst = max(worst, e1)
            print(f"  {p:2d} {nu:5.2f} {e1:10.3e} {r1:11.5f} {e2:10.3e} {r2:11.5f} {e2 / e1:10.3f}")
            rep.check(f"p={p} nu={nu:.2f}: max|jump - f(3) slip| / max|slip| (L=1)", e1, 1.5e-2,
                      f"(recovery ratio {r1:.4f})")
            rep.check_bool(f"p={p} nu={nu:.2f}: deficit shrinks with L (L=2 < L=1)", e2 < e1,
                           f"(L=2 {e2:.2e} vs L=1 {e1:.2e})")
    print(f"  worst L=1 error: {worst:.3e}")
    # large triangle: the deficit is O(z0 / L) and must be gone at L = 100
    tri_big = clq.equilateral(100.0)
    pts_big = interior_points(tri_big, R_REL * 100.0)
    for p in (0, 1, 2):
        e_big, r_big = jump_error(tri_big, slips[p], pts_big, 0.45)
        rep.check(f"p={p} nu=0.45: jump error at L=100 (deficit O(z0/L))", e_big, 1e-3,
                  f"(recovery ratio {r_big:.5f})")
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
