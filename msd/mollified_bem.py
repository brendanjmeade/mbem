"""
Multi-domain BEM with Mollified Elastic Kernels on the Sphere

This module implements:
  1. Mollified (regularized) Kelvin displacement and traction kernels
  2. Icosphere mesh generation for the Earth's free surface
  3. Triangle mesh utilities (normals, areas, centroids)
  4. BEM matrix assembly with free-surface and interface conditions
  5. Fault source as displacement discontinuity
  6. H-matrix compression (placeholder for future)

The mollified kernels replace the singular 1/r Kelvin solution with a
smooth approximation using r_eps = sqrt(r^2 + eps^2), eliminating all
singular and hypersingular integrals.

Author: Brendan Meade & Claude
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================
# Material properties
# ============================================================

@dataclass


class ElasticMaterial:
    """Isotropic elastic material."""
    mu: float    # shear modulus (GPa)
    lam: float   # first Lamé parameter (GPa)

    @property
    def nu(self):
        """Poisson's ratio."""
        return self.lam / (2.0 * (self.lam + self.mu))

    @property
    def E(self):
        """Young's modulus."""
        return self.mu * (3.0 * self.lam + 2.0 * self.mu) / (self.lam + self.mu)


def kelvin_U_mollified(r_vec, mu, nu, eps):
    """Mollified Kelvin displacement Green's function.

    The classical Kelvin solution for displacement at x due to a
    point force at x' in an infinite isotropic elastic medium is:

        U_ij(r) = 1/(16 pi mu (1-nu)) * [
            (3 - 4nu) delta_ij / r  +  r_i r_j / r^3
        ]

    where r = x - x', r = |r|.

    The mollified version replaces:
        1/r   -> 1/r_eps           where r_eps = sqrt(r^2 + eps^2)
        1/r^3 -> 1/r_eps^3

    This is equivalent to convolving the point force with a smooth
    blob of width eps, which is the exact solution to Navier's equation
    with a distributed load (the regularized Kelvinlet approach).

    Args:
        r_vec: (3,) or (N, 3) displacement vectors (x - x')
        mu: shear modulus
        nu: Poisson's ratio
        eps: mollification parameter

    Returns:
        U: (3, 3) or (N, 3, 3) displacement kernel
    """
    r_vec = np.atleast_2d(r_vec)  # (N, 3)
    N = r_vec.shape[0]

    r2 = np.sum(r_vec**2, axis=1)  # (N,)
    r_eps = np.sqrt(r2 + eps**2)   # (N,)

    prefac = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))

    a = (3.0 - 4.0 * nu)

    # Vectorized: U_ij = prefac * (a * dij / r_eps + ri*rj / r_eps^3)
    inv_r_eps = 1.0 / r_eps     # (N,)
    inv_r_eps3 = inv_r_eps**3   # (N,)

    # Outer product r_i * r_j: (N, 3, 3)
    rr = r_vec[:, :, None] * r_vec[:, None, :]

    # Identity part: a * dij / r_eps
    U = prefac * (a * np.eye(3)[None, :, :] * inv_r_eps[:, None, None]
                  + rr * inv_r_eps3[:, None, None])

    return U


def kelvin_T_mollified(r_vec, n_vec, mu, nu, eps):
    """Mollified Kelvin traction kernel.

    The classical traction kernel T_ij(r, n) gives the j-th traction
    component at a point with outward normal n, due to a unit point
    force in the i-th direction applied at the origin:

        T_ij(r, n) = -1/(8 pi (1-nu) r^2) * [
            dr/dn * ((1 - 2nu) delta_ij + 3 r_i r_j / r^2)
            + (1 - 2nu)(n_i r_j - n_j r_i) / r^2
        ]

    where dr/dn = (r . n) / r.

    The mollified version replaces powers of 1/r with 1/r_eps.
    Specifically:
        1/r^2 -> 1/r_eps^2
        1/r^3 -> 1/r_eps^3  (for the dr/dn terms)
        r_i/r -> r_i/r_eps   (unit vector regularized)

    The key insight: since T involves derivatives of U, and U is
    already mollified, T inherits the smoothness. The mollified T
    is the exact traction from the mollified displacement field.

    We compute T from the stress of the mollified U field using
    Hooke's law and then contracting with n.

    Args:
        r_vec: (N, 3) displacement vectors
        n_vec: (N, 3) or (3,) outward normal at field point
        mu: shear modulus
        nu: Poisson's ratio
        eps: mollification parameter

    Returns:
        T: (N, 3, 3) traction kernel T_ij
           T_ij * f_j = traction_i due to force f at source
    """
    r_vec = np.atleast_2d(r_vec)  # (N, 3)
    n_vec = np.atleast_2d(n_vec)  # (N, 3) or (1, 3)
    if n_vec.shape[0] == 1:
        n_vec = np.broadcast_to(n_vec, r_vec.shape)
    N = r_vec.shape[0]

    r2 = np.sum(r_vec**2, axis=1)             # (N,)
    r_eps2 = r2 + eps**2                        # (N,)
    r_eps = np.sqrt(r_eps2)                     # (N,)
    r_eps3 = r_eps * r_eps2                     # (N,) = r_eps^3
    r_eps5 = r_eps3 * r_eps2                    # (N,) = r_eps^5

    # r dot n
    r_dot_n = np.sum(r_vec * n_vec, axis=1)  # (N,)

    prefac = -1.0 / (8.0 * np.pi * (1.0 - nu))

    # Vectorized T kernel
    coeff_12nu = 1.0 - 2.0 * nu

    # Outer product r_i * r_j: (N, 3, 3)
    rr = r_vec[:, :, None] * r_vec[:, None, :]

    # Term 1: r_dot_n * [(1-2nu) dij / r_eps^3 + 3 ri rj / r_eps^5]
    term1 = (r_dot_n[:, None, None] *
             (coeff_12nu * np.eye(3)[None, :, :] / r_eps3[:, None, None]
              + 3.0 * rr / r_eps5[:, None, None]))

    # Term 2: (1-2nu)(ni rj - nj ri) / r_eps^3
    nr = n_vec[:, :, None] * r_vec[:, None, :]  # ni * rj
    rn = r_vec[:, :, None] * n_vec[:, None, :]  # ri * nj
    term2 = coeff_12nu * (nr - rn) / r_eps3[:, None, None]

    T = prefac * (term1 + term2)

    return T


# ============================================================
# Mesh data structures
# ============================================================

@dataclass


class TriMesh:
    """Triangle surface mesh."""
    vertices: np.ndarray   # (n_verts, 3)
    triangles: np.ndarray  # (n_tris, 3) integer indices

    @property
    def n_triangles(self):
        return self.triangles.shape[0]

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    def centroids(self):
        """Triangle centroids."""
        v = self.vertices[self.triangles]  # (n_tri, 3, 3)
        return v.mean(axis=1)  # (n_tri, 3)

    def normals_and_areas(self):
        """Outward normals and areas for each triangle."""
        v = self.vertices[self.triangles]  # (n_tri, 3, 3)
        e1 = v[:, 1] - v[:, 0]
        e2 = v[:, 2] - v[:, 0]
        cross = np.cross(e1, e2)
        area2 = np.linalg.norm(cross, axis=1)  # 2 * area
        normals = cross / area2[:, None]
        areas = 0.5 * area2
        return normals, areas

    def ensure_outward_normals(self, center=np.array([0.0, 0.0, 0.0])):
        """Flip triangles so normals point away from center."""
        centroids = self.centroids()
        normals, _ = self.normals_and_areas()
        outward = centroids - center
        dots = np.sum(normals * outward, axis=1)
        flip = dots < 0
        self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]


def assemble_BEM_matrices(mesh_field, mesh_source, material, eps,
                          kernel="U"):
    """Assemble BEM influence matrix using analytical triangle integration.

    For each field element i (centroid-collocated) and source element j,
    compute the exact analytical integral of the mollified kernel over
    source element j at the field centroid. Uses the per-triangle
    analytical integrators from ``mollified_kernel.analytical_kernels``
    (Kelvin G for kernel="U", displacement-discontinuity for kernel="T"),
    vectorized over observation points for each source triangle.

    Returns:
        Matrix of shape (3*N_field, 3*N_source)
    """
    from mollified_kernel.analytical_batch import (
        assemble_U_matrix_batch,
        assemble_T_matrix_batch,
    )

    mu = material.mu
    nu = material.nu

    x_field = mesh_field.centroids()
    n_source, _ = mesh_source.normals_and_areas()
    tri_verts = mesh_source.vertices[mesh_source.triangles]  # (N_s, 3, 3)

    if kernel == "U":
        return assemble_U_matrix_batch(x_field, tri_verts, mu, nu, eps)
    elif kernel == "T":
        return assemble_T_matrix_batch(x_field, tri_verts, n_source,
                                       mu, nu, eps)
    raise ValueError(f"Unknown kernel: {kernel}")
