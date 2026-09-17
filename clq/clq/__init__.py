"""clq -- closed-form mollified dislocation kernels for constant, linear and
quadratic slip on an arbitrary flat triangle (full space, Cortez
regularisation R = sqrt(r^2 + eps^2)).

    import clq
    u     = clq.displacement(obs, tri, slip, mu, nu, eps)   # (N, 3)
    sigma = clq.stress(obs, tri, slip, mu, nu, eps)         # (N, 3, 3), elastic
    inf   = clq.influence(obs, tri, mu, nu, eps, order=2)   # nodal tensors

``slip`` has shape (1, 3), (3, 3) or (6, 3) for constant / linear / quadratic
slip (nodal values at ``clq.nodes(tri, order)``); a (3,) vector is constant
slip, and higher Lagrange orders (10, 15, ... rows) run as well.
"""
from .api import Influence, influence, displacement, stress, eigenstress, traction
from .frame import equilateral, inside, local_frame, unit_normal, barycentric
from .shape import (nodes, nodal_values, shape_functions, interpolate,
                    barycentric_grid, triangle_grid, grid_triangles, n_nodes,
                    order_from_count)

__version__ = "0.1.0"
__all__ = [
    "Influence", "influence", "displacement", "stress", "eigenstress", "traction",
    "equilateral", "inside", "local_frame", "unit_normal", "barycentric",
    "nodes", "nodal_values", "shape_functions", "interpolate",
    "barycentric_grid", "triangle_grid", "grid_triangles", "n_nodes", "order_from_count",
]
