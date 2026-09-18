"""Closed and open triangulated surfaces for gates, studies and examples.

Nothing here is a mesh *generator* in the production sense (no grading, no
geometry input, no quality control) -- these are the three shapes the solver
gates need and nothing more:

* :func:`icosphere` -- a closed, outward-oriented, near-uniform sphere.  The
  canonical closed boundary: no edges, so a free-traction model on it isolates
  the collocation/free-term question from the ``rho^(-1/3)`` corner behaviour
  that ``../fbem/FINDINGS.md`` sec.1 measured on a polyhedron.
* :func:`box` -- a closed, outward-oriented rectangular box.  The polyhedron
  case, with the 270-degree exterior wedges that stalled the force element.
* :func:`rectangle` -- an OPEN patch (a fault / crack), oriented by its
  ``normal`` argument.

Every returned triangle obeys clq's orientation rule: the unit normal is
``(v2 - v1) x (v3 - v1)`` normalised, so "outward-oriented" means that vector
points away from the enclosed volume.  :func:`enclosed_volume` is positive
exactly then, and :func:`is_closed` checks that the surface is a closed
oriented manifold (every directed edge matched once by its reverse).

Both are what :class:`ddbem.model.Model` uses to decide whether the rigid-body
jump calibration is even defined -- it is a closed-surface row-sum identity, so
an open or inconsistently oriented boundary must not be calibrated.
"""
from __future__ import annotations

import numpy as np

from . import defaults


# ---------------------------------------------------------------------------
# connectivity helpers
# ---------------------------------------------------------------------------

def tri_verts(vertices, triangles) -> np.ndarray:
    """``(n_tri, 3, 3)`` vertex coordinates from a (V, 3) / (T, 3) mesh pair."""
    v = np.asarray(vertices, float)
    t = np.asarray(triangles, np.intp)
    return np.ascontiguousarray(v[t])


def weld(tri_v, rel_tol: float = defaults.WELD_REL_TOL):
    """Recover ``(vertices, triangles)`` from ``(n_tri, 3, 3)`` coordinates.

    Vertices closer than ``rel_tol`` times the model diagonal are merged.  This
    is how a model assembled from bare coordinate arrays (several patches, no
    shared index array) gets the connectivity that :func:`is_closed` and
    :func:`ddbem.layout.continuous` need.  Rounding, not a tree query: the
    meshes here are built from exact shared vertices, so a quantisation is
    enough and is deterministic.
    """
    t = np.asarray(tri_v, float).reshape(-1, 3)
    if t.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.intp)
    span = float(np.max(t.max(axis=0) - t.min(axis=0)))
    scale = span if span > 0 else 1.0
    q = np.round(t / (rel_tol * scale)).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    verts = t[first]
    return verts, np.asarray(inv, np.intp).reshape(-1, 3)


def is_closed(tri_v, rel_tol: float = defaults.WELD_REL_TOL) -> bool:
    """True if the triangles form a closed, consistently oriented manifold.

    Consistency is checked on DIRECTED edges: a closed oriented surface has
    every directed edge (a, b) appearing exactly once, matched by exactly one
    (b, a).  A flipped triangle therefore fails even though the unoriented edge
    counts still pair up -- which is the point, because a flipped patch silently
    breaks the rigid-body calibration rather than the geometry.
    """
    _, tris = weld(tri_v, rel_tol)
    if tris.shape[0] == 0:
        return False
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    fwd = {(int(a), int(b)) for a, b in e}
    if len(fwd) != e.shape[0]:
        return False                                   # repeated directed edge
    return all((b, a) in fwd for (a, b) in fwd)


def enclosed_volume(tri_v) -> float:
    """Signed volume ``(1/6) sum v1 . (v2 x v3)``; positive for outward normals."""
    t = np.asarray(tri_v, float)
    return float(np.sum(np.einsum("ti,ti->t", t[:, 0],
                                  np.cross(t[:, 1], t[:, 2]))) / 6.0)


def flip(tri_v) -> np.ndarray:
    """Reverse every triangle's orientation (swap vertices 2 and 3)."""
    t = np.asarray(tri_v, float)
    return np.ascontiguousarray(t[:, [0, 2, 1], :])


# ---------------------------------------------------------------------------
# closed surfaces
# ---------------------------------------------------------------------------

_ICO_T = (1.0 + np.sqrt(5.0)) / 2.0

_ICO_V = np.array([
    [-1, _ICO_T, 0], [1, _ICO_T, 0], [-1, -_ICO_T, 0], [1, -_ICO_T, 0],
    [0, -1, _ICO_T], [0, 1, _ICO_T], [0, -1, -_ICO_T], [0, 1, -_ICO_T],
    [_ICO_T, 0, -1], [_ICO_T, 0, 1], [-_ICO_T, 0, -1], [-_ICO_T, 0, 1],
], dtype=float)

_ICO_F = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
], dtype=np.intp)


def icosphere(n_sub: int = 1, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    """Subdivided icosahedron, outward-oriented.  ``20 * 4**n_sub`` triangles.

    Returns ``(vertices, triangles)``; call :func:`tri_verts` for coordinates.
    """
    v = list(_ICO_V / np.linalg.norm(_ICO_V[0]))
    f = _ICO_F.copy()
    for _ in range(int(n_sub)):
        mid: dict = {}
        new = []

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                m = 0.5 * (v[a] + v[b])
                v.append(m / np.linalg.norm(m))
                mid[key] = len(v) - 1
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.asarray(new, np.intp)
    verts = np.asarray(v, float) * float(radius) + np.asarray(center, float)
    return np.ascontiguousarray(verts), np.ascontiguousarray(f)


def box(lengths=(1.0, 1.0, 1.0), n=(1, 1, 1), center=(0.0, 0.0, 0.0)):
    """Closed, outward-oriented rectangular box, each face an ``n_i x n_j`` grid.

    ``12 * prod`` -ish triangles; the exact count is
    ``4 * (n0 n1 + n1 n2 + n2 n0)``.  Vertices are shared between faces (the
    result welds), so the box is a genuine closed manifold, edges included.
    """
    L = np.asarray(lengths, float)
    nn = np.asarray(n, int)
    if nn.size == 1:
        nn = np.full(3, int(nn))
    c = np.asarray(center, float)
    faces = []
    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3
        na, nb = int(nn[a]), int(nn[b])
        ua = np.linspace(-0.5, 0.5, na + 1) * L[a]
        ub = np.linspace(-0.5, 0.5, nb + 1) * L[b]
        for side in (-1, +1):
            P = np.zeros((na + 1, nb + 1, 3))
            P[..., a] = ua[:, None]
            P[..., b] = ub[None, :]
            P[..., axis] = side * 0.5 * L[axis]
            for i in range(na):
                for j in range(nb):
                    p00, p10 = P[i, j], P[i + 1, j]
                    p11, p01 = P[i + 1, j + 1], P[i, j + 1]
                    # (+a, +b, +axis) is right-handed, so for side=+1 the
                    # ccw order in (a, b) already gives the +axis normal.
                    if side > 0:
                        faces += [[p00, p10, p11], [p00, p11, p01]]
                    else:
                        faces += [[p00, p11, p10], [p00, p01, p11]]
    t = np.asarray(faces, float) + c
    return weld(t)


# ---------------------------------------------------------------------------
# open surfaces
# ---------------------------------------------------------------------------

def rectangle(corner, edge_a, edge_b, na: int = 1, nb: int = 1):
    """Open rectangular patch, ``2 na nb`` triangles, normal ``edge_a x edge_b``.

    ``corner`` is one corner; ``edge_a``/``edge_b`` are the two full edge
    vectors.  Used for faults.
    """
    o = np.asarray(corner, float)
    ea = np.asarray(edge_a, float)
    eb = np.asarray(edge_b, float)
    s = np.linspace(0.0, 1.0, int(na) + 1)
    t = np.linspace(0.0, 1.0, int(nb) + 1)
    P = o[None, None, :] + s[:, None, None] * ea[None, None, :] \
        + t[None, :, None] * eb[None, None, :]
    faces = []
    for i in range(int(na)):
        for j in range(int(nb)):
            faces += [[P[i, j], P[i + 1, j], P[i + 1, j + 1]],
                      [P[i, j], P[i + 1, j + 1], P[i, j + 1]]]
    return weld(np.asarray(faces, float))
