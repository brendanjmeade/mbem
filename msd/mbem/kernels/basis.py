"""Material-basis assembly API.

The mollified Kelvin U/T influence matrices decompose exactly as

    M(material) = sum_k c_k(mu, lam) * B_k

with GEOMETRY-ONLY basis matrices B_k (3 for the U kernel, 6 for the T
kernel; eps^2 is baked into the relevant B_k). Assemble the basis once
per (field mesh, source mesh, eps) and recombine for every region
material. Materials are real; both backends allocate real accumulators.

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

from .. import defaults
from . import tri_kernels as tk


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
    in (~2x the patch count). This memo computes them once per mesh.

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
    for; anything else is the spec of every patch (``resolve_eps``).
    """
    if isinstance(spec, dict):
        if patch.name not in spec:
            raise ValueError(f"eps dict has no entry for patch "
                             f"{patch.name!r} (keys: {sorted(spec)})")
        spec = spec[patch.name]
    return resolve_eps(spec, patch.mesh)


# ---------------------------------------------------------------------
# Basis containers
# ---------------------------------------------------------------------

@dataclass
class UBasis:
    """Geometry-only U-kernel basis stack, shape (3, 3*N_f, 3*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = u_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


@dataclass
class TBasis:
    """Geometry-only T-kernel basis stack, shape (6, 3*N_f, 3*N_s)."""
    stack: np.ndarray

    def combine(self, material=None, mu=None, lam=None) -> np.ndarray:
        c = t_coeffs(*_mu_lam(material, mu, lam))
        return np.tensordot(c, self.stack, axes=1)

    def nbytes(self) -> int:
        return self.stack.nbytes


# ---------------------------------------------------------------------
# Assembly entry points
# ---------------------------------------------------------------------

def assemble_u_basis(mesh_field, mesh_source, eps,
                     arrays: MeshArrays | None = None) -> UBasis:
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, _ = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return UBasis(tk.u_basis_matrices(x_field, tri_verts, eps_arr))


def assemble_t_basis(mesh_field, mesh_source, eps,
                     arrays: MeshArrays | None = None) -> TBasis:
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, normals = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    return TBasis(tk.t_basis_matrices(x_field, tri_verts, normals, eps_arr))


def assemble_u_matrix(mesh_field, mesh_source, material, eps,
                      arrays: MeshArrays | None = None) -> np.ndarray:
    """One-shot real-material U matrix (coefficients applied in-loop)."""
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, _ = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    g1, g2, g3 = u_coeffs(material.mu, material.lam)
    return tk.u_matrix_direct(x_field, tri_verts, eps_arr, g1, g2, g3)


def assemble_t_matrix(mesh_field, mesh_source, material, eps,
                      arrays: MeshArrays | None = None) -> np.ndarray:
    """One-shot real-material T matrix (coefficients applied in-loop)."""
    a = arrays if arrays is not None else MeshArrays()
    x_field = a.field_points(mesh_field)
    tri_verts, normals = a.source_arrays(mesh_source)
    eps_arr = as_eps_array(eps, tri_verts.shape[0])
    c = t_coeffs(material.mu, material.lam)
    return tk.t_matrix_direct(x_field, tri_verts, normals, eps_arr, *c)
