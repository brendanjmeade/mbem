"""Material-basis assembly API, eps resolution, Lagrange element nodes.

The mollified Kelvin U/T influence matrices decompose exactly as

    M(material) = sum_k c_k(mu, lam) * B_k

with GEOMETRY-ONLY basis matrices B_k (3 for the U kernel, 6 for the T
kernel; eps^2 is baked into the relevant B_k). Assemble the basis once
per (field mesh, source mesh, eps) and recombine for every region
material. Materials are real; both backends allocate real accumulators.
The source density of every block is Lagrange nodal of the source patch's
``order`` (P0/P1/P2); the field side is any point set.

BINDING RULES (from the approved plan's cross-review):
  * coefficients are computed from (mu, lam) directly — NEVER via a
    1/(1-2nu) intermediate (it amplifies catastrophically at the fluid
    limit nu -> 1/2);
  * eps is an (N_src,) per-source-element array everywhere; a scalar is
    promoted to a constant array (== legacy global-eps behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .. import defaults
from . import tri_nodal as tn
from .tri_nodal import n_nodes                                     # noqa: F401


# ---------------------------------------------------------------------
# Material coefficient functions
# ---------------------------------------------------------------------

def _mu_lam(material=None, mu=None, lam=None):
    if material is not None:
        return material.mu, material.lam
    if mu is None or lam is None:
        raise ValueError("provide either material or (mu, lam)")
    return mu, lam


def u_coeffs(mu, lam) -> np.ndarray:
    """Coefficients (3,) for the U-kernel basis [I1*d, eps^2*I3*d, T2[3]]."""
    nu = lam / (2.0 * (lam + mu))
    one_minus_nu = (lam + 2.0 * mu) / (2.0 * (lam + mu))
    C1 = 1.0 / (16.0 * np.pi * mu * one_minus_nu)
    c34 = 3.0 - 4.0 * nu
    g1 = c34 * C1
    g2 = 1.0 / (8.0 * np.pi * mu)          # = 2*(1-nu)*C1, the (1-nu) cancels
    g3 = C1
    return np.array([g1, g2, g3])


def t_coeffs(mu, lam) -> np.ndarray:
    """Coefficients (6,) for the T-kernel basis [N1, N2, N3, R1, R2, R3].

    N[P]_ij = n_j tr(P_i..) carries lam*C1; R[P]_ij = n_m P_ijm + n_k P_ikj
    carries mu*C1 (traction-operator pairing, see tri_kernels).
    """
    nu = lam / (2.0 * (lam + mu))
    one_minus_nu = (lam + 2.0 * mu) / (2.0 * (lam + mu))
    C1 = 1.0 / (16.0 * np.pi * mu * one_minus_nu)
    c34 = 3.0 - 4.0 * nu
    cL = lam * C1                           # N (trace) blocks
    cM = mu * C1                            # R blocks; = 1/(16*pi*(1-nu)), mu cancels
    blob = 6.0 * one_minus_nu
    return np.array([c34 * cL, -cL, blob * cL,
                     c34 * cM, -cM, blob * cM])


# ---------------------------------------------------------------------
# Mesh-input normalization
# ---------------------------------------------------------------------

def _source_arrays(mesh_source):
    tri_verts = mesh_source.vertices[mesh_source.triangles]
    normals, _ = mesh_source.normals_and_areas()
    return np.ascontiguousarray(tri_verts), np.ascontiguousarray(normals)


def _field_points(mesh_or_points):
    if hasattr(mesh_or_points, "centroids"):
        return np.ascontiguousarray(mesh_or_points.centroids())
    return np.ascontiguousarray(np.asarray(mesh_or_points, dtype=float))


class MeshArrays:
    """ASSEMBLY-SCOPED memo of mesh-derived arrays.

    TriMesh recomputes centroids()/normals_and_areas() on every call, and
    one assembly requests the same mesh's arrays once per block it appears
    in (~2x the patch count). This memo computes them once per mesh, and a
    patch's collocation points once per patch.

    Scope rule (why this is NOT a module-level cache): TriMesh is an
    unhashable dataclass keyed here by id(), and
    ``TriMesh.ensure_outward_normals`` mutates ``triangles`` in place --
    a cache that outlives one assembly could serve stale arrays after id
    reuse or in-place mutation. Create one per build and drop it (the
    ``tree_cache`` pattern in la/hop.py).
    """

    def __init__(self):
        self._src: dict[int, tuple] = {}
        self._pts: dict[int, np.ndarray] = {}
        self._coll: dict[int, np.ndarray] = {}

    def source_arrays(self, mesh_source):
        key = id(mesh_source)
        v = self._src.get(key)
        if v is None:
            v = _source_arrays(mesh_source)
            self._src[key] = v
        return v

    def field_points(self, mesh_or_points):
        if not hasattr(mesh_or_points, "centroids"):
            return _field_points(mesh_or_points)
        key = id(mesh_or_points)
        v = self._pts.get(key)
        if v is None:
            v = _field_points(mesh_or_points)
            self._pts[key] = v
        return v

    def collocation_points(self, patch):
        """``patch.collocation_points()`` (n_dofs, 3), once per patch."""
        key = id(patch)
        v = self._coll.get(key)
        if v is None:
            v = np.ascontiguousarray(patch.collocation_points())
            self._coll[key] = v
        return v

    def collocation_normals(self, patch):
        """``patch.collocation_normals()`` (n_dofs, 3), once per patch."""
        key = ("n", id(patch))
        v = self._coll.get(key)
        if v is None:
            v = np.ascontiguousarray(patch.collocation_normals())
            self._coll[key] = v
        return v


# ---------------------------------------------------------------------
# Lagrange element nodes and shape functions, in tri_nodal's node order
# (p = 0 centroid; p = 1 v1, v2, v3; p = 2 v1, v2, v3, m12, m23, m31), so
# node k of element s is density row K * s + k and column 3 (K s + k) + j.
# ---------------------------------------------------------------------

def lagrange_nodes(order: int, shrink: float = 0.0) -> np.ndarray:
    """Barycentric coordinates (K, 3) of the nodes; ``shrink`` = t pulls each
    toward the centroid, ``lam = (1 - t) lam_node + t / 3`` (a collocation
    point; the basis itself never moves)."""
    K = n_nodes(order)
    if order == 0:
        lam = np.full((1, 3), 1.0 / 3.0)
    elif order == 1:
        lam = np.eye(3)
    else:
        lam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    t = float(shrink)
    if t:
        lam = (1.0 - t) * lam + t / 3.0
    assert lam.shape == (K, 3)
    return lam


def lagrange_shape(order: int, lam) -> np.ndarray:
    """Shape functions N_k (M, K) at barycentric points ``lam`` (M, 3)."""
    lam = np.atleast_2d(np.asarray(lam, float))
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    if n_nodes(order) == 1:
        return np.ones((lam.shape[0], 1))
    if order == 1:
        return np.stack([l1, l2, l3], axis=1)
    return np.stack([l1 * (2.0 * l1 - 1.0), l2 * (2.0 * l2 - 1.0),
                     l3 * (2.0 * l3 - 1.0),
                     4.0 * l1 * l2, 4.0 * l2 * l3, 4.0 * l3 * l1], axis=1)


def as_eps_array(eps, n_source: int) -> np.ndarray:
    """Promote scalar eps to (N_src,); validate length and positivity.

    Every entry must be finite and > 0: the kernels only see eps^2, so a
    negative eps runs silently as |eps| while the eigenstress correction
    (which keys on eps <= 0) drops out, and eps = 0 hits the unfloored
    I3 = -Omega/h division inside prange, where numba cannot raise.
    """
    arr = np.asarray(eps, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n_source, float(arr))
    elif arr.shape != (n_source,):
        raise ValueError(f"eps array shape {arr.shape} != ({n_source},)")
    bad = ~(np.isfinite(arr) & (arr > 0.0))
    if bad.any():
        raise ValueError(f"eps must be finite and > 0: {int(bad.sum())} bad "
                         f"of {arr.size}, min = {arr.min()}")
    return np.ascontiguousarray(arr)


def element_sizes(mesh) -> np.ndarray:
    """Per-triangle size h_j: mean edge length (N_src,)."""
    tv = np.asarray(mesh.vertices, float)[np.asarray(mesh.triangles)]
    e = np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 1],
                  tv[:, 0] - tv[:, 2]])
    return np.linalg.norm(e, axis=2).mean(axis=0)


def resolve_eps(eps, mesh) -> np.ndarray:
    """Resolve an eps SPEC for one source mesh to an (N_src,) array.

    Accepted specs: a scalar (constant eps -- the legacy behaviour), an
    (N_src,) array, or the string ``"auto"``, which sets each element's
    mollification width from its own size, ``eps_j = EPS_OVER_H * h_j``
    (h_j = mean edge length). "auto" keeps eps/h fixed under mesh
    grading and h-refinement -- coarse far-field panels are not
    under-mollified and fine fault-zone panels are not over-mollified.
    Per-patch dicts are split by ``resolve_patch_eps``.
    """
    if isinstance(eps, str):
        if eps != "auto":
            raise ValueError(f"unknown eps spec {eps!r} (only 'auto')")
        return np.ascontiguousarray(
            defaults.EPS_OVER_H * element_sizes(mesh))
    return as_eps_array(eps, mesh.n_triangles)


def resolve_patch_eps(spec, patch) -> np.ndarray:
    """Resolve the eps spec of one patch or fault to its (N_src,) array.

    A dict is keyed by patch name and must name every patch it is used
    for; anything else is the spec of every patch (``resolve_eps``). On a
    fault of either kind, ``"auto"`` is one value, ``FAULT_EPS_OVER_H *
    min(h)`` (see ``defaults``), not a per-element width -- a shared node's
    row sees several elements, and a uniform slip smeared unequally is a
    spurious source.
    """
    from ..model.core import FAULT_KINDS

    if isinstance(spec, dict):
        if patch.name not in spec:
            raise ValueError(f"eps dict has no entry for patch "
                             f"{patch.name!r} (keys: {sorted(spec)})")
        spec = spec[patch.name]
    if isinstance(spec, str) and spec == "auto" and patch.bc in FAULT_KINDS:
        h = element_sizes(patch.mesh)
        return np.full(h.size, defaults.FAULT_EPS_OVER_H * float(h.min()))
    return resolve_eps(spec, patch.mesh)


# ---------------------------------------------------------------------
# Shared-node columns: the kernels write one column per (element, node);
# a continuous layout owns one column per SHARED node and adds the columns
# of coincident element nodes into it (``ids`` (N_s, K) -> shared id).
# ---------------------------------------------------------------------

def identity_ids(order: int, n_source: int) -> np.ndarray:
    """The element-major (discontinuous) layout as an ids table (N_s, K):
    ids[s, k] = K s + k, the kernels' own column order."""
    K = n_nodes(order)
    return np.ascontiguousarray(np.arange(K * n_source, dtype=np.int64).reshape(n_source, K))


@njit(cache=True, parallel=True)
def _condense_columns(blk, ids, K, out):
    n_rows = blk.shape[0]
    N_s = ids.shape[0]
    for r in prange(n_rows):
        for s in range(N_s):
            for k in range(K):
                ci = 3 * (K * s + k)
                co = 3 * ids[s, k]
                for j in range(3):
                    out[r, co + j] += blk[r, ci + j]


def condense_columns(blk, ids, n_cols: int, out=None) -> np.ndarray:
    """Element-major columns (R, 3 K N_s) -> shared columns (R, 3 n_cols): the
    K node columns of every element are ADDED into their shared node's column
    (``ids`` (N_s, K)); ``out`` accumulates in place when given."""
    blk = np.ascontiguousarray(np.asarray(blk, float))
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    N_s, K = ids.shape
    if blk.shape[1] != 3 * K * N_s:
        raise ValueError(f"block has {blk.shape[1]} columns, ids imply {3 * K * N_s}")
    if out is None:
        out = np.zeros((blk.shape[0], 3 * int(n_cols)))
    elif out.shape != (blk.shape[0], 3 * int(n_cols)):
        raise ValueError(f"out shape {out.shape} != {(blk.shape[0], 3 * int(n_cols))}")
    _condense_columns(blk, ids, K, out)
    return out


# ---------------------------------------------------------------------
# Basis containers
# ---------------------------------------------------------------------

@dataclass
class UBasis:
    """Geometry-only U-kernel basis stack, shape (3, 3*N_f, 3*K*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = u_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


@dataclass
class TBasis:
    """Geometry-only T-kernel basis stack, shape (6, 3*N_f, 3*K*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = t_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


# ---------------------------------------------------------------------
# Assembly entry points. ``order`` is the source density's Lagrange order
# (0, 1, 2; columns 3*(K*s + k) + j, K = n_nodes(order)); order 0 is
# tri_kernels' P0 code, higher orders tri_nodal's.
# ---------------------------------------------------------------------

def assemble_u_basis(mesh_field, mesh_source, eps,
                     arrays: MeshArrays | None = None, order: int = 0) -> UBasis:
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, _ = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return UBasis(tn.u_basis_matrices(x_field, tri_verts, eps_arr, order))


def assemble_t_basis(mesh_field, mesh_source, eps,
                     arrays: MeshArrays | None = None, order: int = 0) -> TBasis:
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, normals = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return TBasis(tn.t_basis_matrices(x_field, tri_verts, normals, eps_arr, order))


def _condensed(kernel_call, x_field, tri_verts, normals, eps_arr, order, ids, n_cols):
    """``kernel_call`` on chunks of ASSEMBLY_SOURCE_CHUNK source elements,
    each chunk's element-major columns condensed into the shared columns
    before the next is assembled (``defaults`` has the transient budget)."""
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    N_s = tri_verts.shape[0]
    if ids.shape != (N_s, n_nodes(order)):
        raise ValueError(f"ids shape {ids.shape} != ({N_s}, {n_nodes(order)})")
    out = np.zeros((3 * x_field.shape[0], 3 * int(n_cols)))
    step = int(defaults.ASSEMBLY_SOURCE_CHUNK)
    for c0 in range(0, N_s, step):
        c1 = min(N_s, c0 + step)
        blk = kernel_call(x_field, tri_verts[c0:c1], None if normals is None
                          else normals[c0:c1], eps_arr[c0:c1], order)
        condense_columns(blk, ids[c0:c1], n_cols, out=out)
    return out


def assemble_u_matrix(mesh_field, mesh_source, material, eps,
                      arrays: MeshArrays | None = None, order: int = 0,
                      ids=None, n_cols: int | None = None) -> np.ndarray:
    """One-shot real-material U matrix (coefficients applied in-loop):
    (3 N_f, 3 K N_s), or (3 N_f, 3 n_cols) condensed onto shared-node
    columns when ``ids`` (N_s, K) is given."""
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, _ = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    g1, g2, g3 = u_coeffs(material.mu, material.lam)
    if ids is None:
        return tn.u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3, order)
    return _condensed(lambda x, tv, _n, e, p: tn.u_matrix_direct(x, tv, e, g1, g2, g3, p),
                      x_field, tri_verts, None, eps_arr, order, ids, n_cols)


def assemble_t_matrix(mesh_field, mesh_source, material, eps,
                      arrays: MeshArrays | None = None, order: int = 0,
                      ids=None, n_cols: int | None = None) -> np.ndarray:
    """One-shot real-material T matrix (coefficients applied in-loop):
    (3 N_f, 3 K N_s), or (3 N_f, 3 n_cols) condensed onto shared-node
    columns when ``ids`` (N_s, K) is given."""
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, normals = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    c = t_coeffs(material.mu, material.lam)
    if ids is None:
        return tn.t_matrix_direct(x_field, tri_verts, normals, eps_arr, *c, order)
    return _condensed(lambda x, tv, n, e, p: tn.t_matrix_direct(x, tv, n, e, *c, p),
                      x_field, tri_verts, normals, eps_arr, order, ids, n_cols)


def assemble_traction_matrix(x_field, n_field, mesh_source, material, eps,
                             arrays: MeshArrays | None = None, order: int = 0,
                             kind: str = "slip", ids=None,
                             n_cols: int | None = None) -> np.ndarray:
    """Traction rows (3 N_f, 3 n_cols) at the points ``x_field`` with normals
    ``n_field`` of a nodal density on ``mesh_source``: ``kind="slip"`` the
    ELASTIC traction of a displacement jump (eigenstress removed, the rows of
    a traction-BC fault), ``"force"`` the Kelvin single layer's; ``"eigen"`` and
    ``"total"`` (the eigenstress alone, and left in) exist for the gates'
    tripwires. Columns are
    the kernels' element-major layout unless ``ids`` (N_s, K) maps them onto
    ``n_cols`` shared nodes; the scatter happens inside the driver."""
    a = arrays if arrays is not None else MeshArrays()
    tri_verts, normals = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    if ids is None:
        ids = identity_ids(order, tri_verts.shape[0])
        n_cols = ids.size
    return tn.traction_matrix(np.ascontiguousarray(np.asarray(x_field, float)),
                              np.ascontiguousarray(np.asarray(n_field, float)),
                              tri_verts, normals, eps_arr, ids, int(n_cols),
                              material.mu, material.lam, order, kind)
