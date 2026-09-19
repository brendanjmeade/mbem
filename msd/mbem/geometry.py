"""Small exact-geometry helpers shared by the model layer and the evaluators."""
from __future__ import annotations

import numpy as np


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
