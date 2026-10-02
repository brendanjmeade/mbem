"""Small exact-geometry helpers shared by the model layer and the evaluators."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def _solid_angle_rows(verts, tris, points, out):
    """van Oosterom signed solid angle of every triangle, summed per point.

    The ONE statement of the formula: ``model/core`` reads containment off it
    and the volume sampler classifies a grid with it, so a second copy is the
    one that would disagree. ``prange`` is over the POINTS and the per-triangle
    work is O(1) scratch, so this is O(1) memory at any point count -- the
    vectorised numpy form needs (N_pts, N_tri, 3) temporaries, which is 54 GB
    for a 500k-point grid over a 4.5k-triangle surface.

    ``parallel=True``, so main thread only (rule 9); there is no ``_serial``
    variant because nothing threads this yet.
    """
    for n in prange(points.shape[0]):
        px, py, pz = points[n, 0], points[n, 1], points[n, 2]
        total = 0.0
        for t in range(tris.shape[0]):
            ax = verts[tris[t, 0], 0] - px
            ay = verts[tris[t, 0], 1] - py
            az = verts[tris[t, 0], 2] - pz
            bx = verts[tris[t, 1], 0] - px
            by = verts[tris[t, 1], 1] - py
            bz = verts[tris[t, 1], 2] - pz
            cx = verts[tris[t, 2], 0] - px
            cy = verts[tris[t, 2], 1] - py
            cz = verts[tris[t, 2], 2] - pz
            ra = np.sqrt(ax * ax + ay * ay + az * az)
            rb = np.sqrt(bx * bx + by * by + bz * bz)
            rc = np.sqrt(cx * cx + cy * cy + cz * cz)
            # numer = a . (b x c)
            numer = (ax * (by * cz - bz * cy)
                     + ay * (bz * cx - bx * cz)
                     + az * (bx * cy - by * cx))
            denom = (ra * rb * rc
                     + rc * (ax * bx + ay * by + az * bz)
                     + ra * (bx * cx + by * cy + bz * cz)
                     + rb * (ax * cx + ay * cy + az * cz))
            total += 2.0 * np.arctan2(numer, denom)
        out[n] = total


def solid_angle_batch(mesh, points) -> np.ndarray:
    """``(N,)`` signed solid angle subtended by ``mesh`` at each point."""
    points = np.ascontiguousarray(np.asarray(points, float).reshape(-1, 3))
    out = np.zeros(points.shape[0])
    _solid_angle_rows(np.ascontiguousarray(np.asarray(mesh.vertices, float)),
                      np.ascontiguousarray(np.asarray(mesh.triangles)),
                      points, out)
    return out


def point_triangle_distance(p, a, b, c):
    """Exact distance from points ``p`` to triangles ``(a, b, c)``, all
    broadcastable (..., 3): the closest point is the plane projection
    clamped to the triangle by its Voronoi regions (vertex / edge /
    interior), selected by the edge-direction dot products.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        s = va + vb + vc
        q = a + ab * (vb / s)[..., None] + ac * (vc / s)[..., None]
        e = (d1 / (d1 - d3))[..., None]
        q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
                     a + ab * e, q)
        e = (d2 / (d2 - d6))[..., None]
        q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
                     a + ac * e, q)
        e = ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[..., None]
        q = np.where(((va <= 0) & (d4 >= d3) & (d5 >= d6))[..., None],
                     b + (c - b) * e, q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, q)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, q)
    return np.linalg.norm(p - q, axis=-1)


def distance_to_mesh(points, mesh, k: int = 32):
    """Exact distance from each point to a triangulated mesh, searched over
    the ``k`` triangles whose centroids are nearest (a cKDTree query), and
    the index of the closest of them: ``(d, idx)``, each ``(N,)``.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, float).reshape(-1, 3)
    tv = np.asarray(mesh.vertices, float)[np.asarray(mesh.triangles)]
    k = min(k, tv.shape[0])
    _, idx = cKDTree(mesh.centroids()).query(points, k=k)
    idx = np.asarray(idx).reshape(points.shape[0], k)
    d = point_triangle_distance(points[:, None, :], tv[idx, 0], tv[idx, 1],
                                tv[idx, 2])
    j = d.argmin(axis=1)
    return d[np.arange(points.shape[0]), j], idx[np.arange(points.shape[0]), j]
