"""Shared-node (continuous) layout of a triangulated patch.

A continuous Lagrange density stores ONE value per shared node: the welded
mesh vertices, then one node per edge (its midpoint). Element s reads its
six values through ``ids[s]`` in clq's node order (v1 v2 v3 m12 m23 m31), so
the element-major density the kernels take is ``values[ids].reshape(-1, 3)``
and two elements sharing an edge interpolate the same values along it: the
density is continuous by construction, with no constraint equation and no
extra row. Where a discontinuous layout radiates a 1/d field from every
inter-element jump, a continuous one leaves only the kink in the gradient, a
log singularity the mollified kernel bounds; that is what makes collocating
AT the shared nodes admissible (eps > 0) and the on-fault stress readout
convergent in eps instead of floored (1.2 eps/a on every mesh, against a P0
centroid floor of 1.4-2.6 % that no refinement moves).

The table is order-generic (vertices, then edge nodes) but only P2 is
offered by ``Patch``: continuous P1 measured 15-22 % too stiff on the penny
crack and first order in h.

Welding is by ROUNDED COORDINATES, never by index: a mesh concatenated from
panels repeats its seam vertices, and a density that stayed discontinuous
across the seam would defeat the layout silently. Nodes on an edge with a
single incident element are ``boundary`` (a crack front, or a trace); an
edge with more than two is refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kernels.basis import n_nodes

#: Welding tolerance as a fraction of the shortest mesh edge: two vertices
#: closer than this are one node. Meshes from one Triangle call are shared
#: by index already (the weld is then the identity); panel seams coincide to
#: round-off; anything in between is a mesher fault, not a layout question.
WELD_TOL_OVER_H = 1e-9


@dataclass(eq=False)
class SharedNodes:
    coords: np.ndarray        # (n, 3): welded vertices, then edge nodes
    ids: np.ndarray           # (N_tri, K) int64, clq node order per element
    normals: np.ndarray       # (n, 3) unit: area-weighted mean of the incident element normals
    boundary: np.ndarray      # (n,) bool: on an edge with one incident element
    areas: np.ndarray         # (N_tri,)
    vertex_id: np.ndarray     # (n_mesh_vertices,) -> welded vertex node
    n_vertices: int           # nodes [0, n_vertices) are vertices, the rest edge nodes
    order: int

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    def expand(self, values) -> np.ndarray:
        """Nodal values (n, 3) -> the element-major density (K N_tri, 3)."""
        v = np.asarray(values, float)
        if v.shape != (self.n, 3):
            raise ValueError(f"nodal values must be ({self.n}, 3), got {v.shape}")
        return np.ascontiguousarray(v[self.ids].reshape(-1, 3))

    def integrate(self, values) -> np.ndarray:
        """Integral over the patch of a nodal field (n, 3) -> (3,): the
        midpoint rule, A_T / 3 times the three edge-node values, exact for
        quadratics (P2); the vertex rule at P1; the centroid value at P0."""
        v = np.asarray(values, float)
        if v.shape != (self.n, 3):
            raise ValueError(f"nodal values must be ({self.n}, 3), got {v.shape}")
        if self.order == 2:
            per = v[self.ids[:, 3:6]].sum(axis=1) / 3.0
        elif self.order == 1:
            per = v[self.ids[:, 0:3]].sum(axis=1) / 3.0
        else:
            per = v[self.ids[:, 0]]
        return (self.areas[:, None] * per).sum(axis=0)


def build_shared_nodes(mesh, order: int = 2) -> SharedNodes:
    """The shared-node table of ``mesh`` (TriMesh-compatible) at ``order``."""
    K = n_nodes(order)
    verts = np.asarray(mesh.vertices, float)
    tris = np.asarray(mesh.triangles, np.int64)
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.shape[0] == 0:
        raise ValueError("a shared-node layout needs an (N_tri, 3) triangle table")
    tv = verts[tris]
    edge_len = np.linalg.norm(tv[:, [1, 2, 0], :] - tv, axis=2)
    h_min = float(edge_len.min())
    if not h_min > 0.0:
        raise ValueError("a shared-node layout needs non-degenerate triangles")

    # 1. weld the vertices by rounded coordinates
    key = np.round(verts / (WELD_TOL_OVER_H * h_min)).astype(np.int64)
    _uniq, first, inverse = np.unique(key, axis=0, return_index=True,
                                      return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    # keep the welded vertices in first-appearance order, so a mesh that is
    # already shared by index gets the identity weld (coords[:n_v] == vertices)
    perm = np.argsort(first, kind="stable")
    rank = np.empty(perm.size, np.int64)
    rank[perm] = np.arange(perm.size)
    vertex_id = rank[inverse]
    v_coords = verts[first[perm]]
    n_v = int(v_coords.shape[0])
    tw = vertex_id[tris]
    if np.any(tw[:, 0] == tw[:, 1]) or np.any(tw[:, 1] == tw[:, 2]) \
            or np.any(tw[:, 2] == tw[:, 0]):
        raise ValueError("welding collapsed an element: two vertices of one "
                         "triangle coincide within the weld tolerance")

    # 2. one node per edge, in the element's own edge order 12, 23, 31
    e = np.stack([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]], axis=1)   # (N, 3, 2)
    e_sorted = np.sort(e.reshape(-1, 2), axis=1)
    e_uniq, edge_id, count = np.unique(e_sorted, axis=0, return_inverse=True,
                                       return_counts=True)
    edge_id = np.asarray(edge_id).reshape(tris.shape[0], 3)
    if count.max() > 2:
        k = int(np.argmax(count))
        raise ValueError(f"non-manifold mesh: edge {tuple(e_uniq[k])} has "
                         f"{int(count[k])} incident elements")
    if count.min() == 1 and count.max() == 1 and n_v == 3 * tris.shape[0]:
        raise ValueError("nothing is shared: a triangle soup cannot carry a "
                         "continuous density (weld or re-mesh it)")
    mid = 0.5 * (v_coords[e_uniq[:, 0]] + v_coords[e_uniq[:, 1]])
    coords = np.vstack([v_coords, mid])

    if order == 2:
        ids = np.concatenate([tw, n_v + edge_id], axis=1)
    elif order == 1:
        ids = tw.copy()
    else:
        ids = np.arange(tris.shape[0], dtype=np.int64)[:, None]
        coords = np.ascontiguousarray(tv.mean(axis=1))
    if ids.shape[1] != K:
        raise AssertionError("node table does not match n_nodes(order)")

    # 3. node normals: area-weighted mean of the incident element normals
    normals_e, areas = mesh.normals_and_areas()
    normals_e = np.asarray(normals_e, float)
    areas = np.asarray(areas, float)
    acc = np.zeros((coords.shape[0], 3))
    np.add.at(acc, ids.reshape(-1), np.repeat(areas[:, None] * normals_e, K, axis=0))
    length = np.linalg.norm(acc, axis=1)
    if np.any(length <= 0.0):
        raise ValueError("a shared node has no net normal: incident elements "
                         "wound in opposite senses")
    normals = acc / length[:, None]

    # 4. boundary nodes: the vertices and the node of every single-count edge
    boundary = np.zeros(coords.shape[0], bool)
    if order >= 1:
        b_edges = np.flatnonzero(count == 1)
        boundary[e_uniq[b_edges].reshape(-1)] = True
        if order == 2:
            boundary[n_v + b_edges] = True
    else:
        boundary[np.flatnonzero(np.any(count[edge_id] == 1, axis=1))] = True

    return SharedNodes(coords=np.ascontiguousarray(coords),
                       ids=np.ascontiguousarray(ids.astype(np.int64)),
                       normals=np.ascontiguousarray(normals),
                       boundary=boundary, areas=np.ascontiguousarray(areas),
                       vertex_id=vertex_id, n_vertices=n_v, order=int(order))
