"""Element nodes, collocation points, and the eps/h budget.

The eps/h part is not cosmetic.  ``fbem/FINDINGS.md`` sec.2 measured that the
mollification width is a FLOOR on the resolvable structure of the unknown, not
merely a regularisation: at eps = 0.3 h no collocation point of a P1/P2 element
is even one mollification length clear of the element boundary, and
p-refinement bought nothing there; holding eps fixed in absolute terms
inverted the result and P1 beat P0 by 1.37x.  So any p-refinement study must
report eps/h and the node clearance, and sweep them rather than fixing one
value.  :func:`eps_report` computes both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import defaults
from ._clq import CLQ as clq
from .layout import n_nodes


def _as_tris(tri_verts):
    t = np.asarray(tri_verts, float)
    if t.ndim == 2 and t.shape == (3, 3):
        t = t[None]
    if t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError(f"tri_verts must be (N_tri, 3, 3), got {t.shape}")
    return t


def barycentric_nodes(order: int, shrink: float = defaults.COLLOCATION_SHRINK):
    """Barycentric coordinates ``(K, 3)`` of the element nodes.

    Node order follows clq exactly: ``p = 0`` centroid; ``p = 1`` the three
    vertices; ``p = 2`` vertices then edge midpoints 12, 23, 31.  ``shrink``
    pulls every node a fraction of the way toward the centroid (a P1/P2 node
    otherwise sits ON the element boundary, which is where the mollified
    traction operator is least trustworthy -- see the module docstring).
    """
    K = n_nodes(order)
    if order == 0:
        lam = np.full((1, 3), 1.0 / 3.0)
    elif order == 1:
        lam = np.eye(3)
    else:
        lam = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                        [.5, .5, 0.], [0., .5, .5], [.5, 0., .5]])
    assert lam.shape == (K, 3)
    s = float(shrink)
    if s:
        lam = (1.0 - s) * lam + s / 3.0
    return lam


def shape_at(order: int, lam) -> np.ndarray:
    """Lagrange shape functions ``(M, K)`` at barycentric coordinates ``(M, 3)``.

    clq's node order (``p = 1`` vertices; ``p = 2`` vertices then midpoints 12,
    23, 31), written straight from the barycentrics so the SOLVER's free term
    does not go through ``clq.shape_functions`` (which projects Cartesian points
    into the element frame).  ``verify/verify_solver.py`` gates the two against
    each other, so this cannot drift.

    This is the object the collocation free term actually needs: for a nodal
    density the jump/free term is the MATRIX of shape-function values at the
    collocation point, ``N_k(x_c)``, not a scalar 1/2.  At ``shrink = 0`` the
    collocation point IS node ``k_c`` and the matrix is the identity, which is
    why P0 and un-shrunk higher order look alike -- and why they stop looking
    alike the moment a node is pulled in.
    """
    lam = np.atleast_2d(np.asarray(lam, float))
    if lam.shape[-1] != 3:
        raise ValueError(f"lam must be (M, 3), got {lam.shape}")
    l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
    if order == 0:
        return np.ones((lam.shape[0], 1))
    if order == 1:
        return np.stack([l1, l2, l3], axis=1)
    if order == 2:
        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                         4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)
    raise ValueError(f"order must be 0, 1 or 2 (got {order})")


def collocation_shape_matrix(order: int, shrink: float) -> np.ndarray:
    """``(K, K)`` matrix ``N[k_c, k] = N_k(collocation point k_c)``.

    The identity at ``shrink = 0``.  This is what spreads a row's free term over
    the collocation element's own nodes.
    """
    return shape_at(order, barycentric_nodes(order, shrink))


def element_nodes(tri_verts, order: int, shrink: float = 0.0) -> np.ndarray:
    """Node coordinates ``(N_tri, K, 3)``, in clq's node order.

    Cross-checked against ``clq.nodes`` (which takes one triangle) in
    ``verify/verify_kernels.py``; this version is vectorised over elements.
    """
    tri = _as_tris(tri_verts)
    lam = barycentric_nodes(order, shrink)                # (K, 3)
    return np.einsum("kv,tvc->tkc", lam, tri)


def collocation_points(tri_verts, order: int,
                       shrink: float = defaults.COLLOCATION_SHRINK):
    """Flattened ``(N_tri * K, 3)`` collocation points in column order.

    Row ``K * s + k`` matches slip columns ``3 * (K s + k) + (0, 1, 2)`` of the
    discontinuous influence matrices, so a square DD collocation system is
    ``traction_matrix(collocation_points(...), normals_repeated, ...)``.
    """
    return element_nodes(tri_verts, order, shrink).reshape(-1, 3)


def element_normals(tri_verts) -> np.ndarray:
    """Unit normals ``(N_tri, 3)``, ``(v2-v1) x (v3-v1)`` normalised (clq's rule)."""
    tri = _as_tris(tri_verts)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def element_areas(tri_verts) -> np.ndarray:
    tri = _as_tris(tri_verts)
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)


def element_h(tri_verts, measure: str = "mean_edge") -> np.ndarray:
    """Element size ``(N_tri,)``.

    ``measure``: "mean_edge" (msd's ``eps="auto"`` convention), "max_edge"
    (the conservative one -- fbem measured that grading inflates ``h_max`` 2.3x
    at fixed dof, which is how an apparent rate gain in h can be an artifact),
    or "sqrt_area".
    """
    tri = _as_tris(tri_verts)
    e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                  np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                  np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)
    if measure == "mean_edge":
        return e.mean(axis=1)
    if measure == "max_edge":
        return e.max(axis=1)
    if measure == "sqrt_area":
        return np.sqrt(element_areas(tri))
    raise ValueError(f"unknown measure {measure!r}")


def node_clearance(tri_verts, order: int,
                   shrink: float = defaults.COLLOCATION_SHRINK) -> np.ndarray:
    """In-plane distance ``(N_tri, K)`` from each node to the element boundary.

    For barycentric coordinates ``lam`` the distance to the edge opposite
    vertex ``i`` is ``lam_i * a_i`` with ``a_i = 2 A / |edge opposite i|`` the
    altitude, so the clearance is ``min_i lam_i a_i``.  A P1 vertex node and a
    P2 midpoint node both have clearance exactly 0 at ``shrink = 0``.
    """
    tri = _as_tris(tri_verts)
    area = element_areas(tri)
    opp = np.stack([np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),   # opposite v1
                    np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),   # opposite v2
                    np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)],  # opposite v3
                   axis=1)
    alt = 2.0 * area[:, None] / opp                                   # (N_tri, 3)
    lam = barycentric_nodes(order, shrink)                            # (K, 3)
    return np.min(lam[None, :, :] * alt[:, None, :], axis=2)


@dataclass
class EpsBudget:
    order: int
    h_mean: float
    h_min: float
    h_max: float
    eps_min: float
    eps_max: float
    eps_over_h_mean: float
    eps_over_h_min: float
    eps_over_h_max: float
    clearance_over_eps_min: float
    clearance_over_eps_median: float
    shrink: float

    def __str__(self) -> str:
        return (f"P{self.order}  h = {self.h_min:.4g}..{self.h_max:.4g} "
                f"(mean {self.h_mean:.4g})   eps = {self.eps_min:.4g}..{self.eps_max:.4g}\n"
                f"      eps/h = {self.eps_over_h_min:.3f}..{self.eps_over_h_max:.3f} "
                f"(mean {self.eps_over_h_mean:.3f})\n"
                f"      node clearance / eps: min {self.clearance_over_eps_min:.3f}, "
                f"median {self.clearance_over_eps_median:.3f}  (shrink {self.shrink:g})")


def eps_report(tri_verts, eps, order: int,
               shrink: float = defaults.COLLOCATION_SHRINK,
               measure: str = "mean_edge") -> EpsBudget:
    """The mollification budget of a mesh: eps/h and node clearance in eps.

    ``clearance / eps`` is the number the fbem study turned on: below ~1 the
    collocation point is inside the element's own smeared edge, the analytic
    half-jump has no room, and p-refinement stalls regardless of the basis.
    """
    tri = _as_tris(tri_verts)
    h = element_h(tri, measure)
    e = np.asarray(eps, float)
    if e.ndim == 0:
        e = np.full(tri.shape[0], float(e))
    clr = node_clearance(tri, order, shrink)                # (N_tri, K)
    ratio = clr / e[:, None]
    eh = e / h
    return EpsBudget(order=int(order), h_mean=float(h.mean()), h_min=float(h.min()),
                     h_max=float(h.max()), eps_min=float(e.min()), eps_max=float(e.max()),
                     eps_over_h_mean=float(eh.mean()), eps_over_h_min=float(eh.min()),
                     eps_over_h_max=float(eh.max()),
                     clearance_over_eps_min=float(ratio.min()),
                     clearance_over_eps_median=float(np.median(ratio)),
                     shrink=float(shrink))


def shrink_for_clearance(tri_verts, order: int, eps,
                         target: float = defaults.TARGET_CLEARANCE_EPS,
                         shrink_max: float = defaults.SHRINK_MAX) -> float:
    """Collocation pull-in ``t`` that puts a node ``target`` eps clear of the edge.

    MEASURED MOTIVATION (``bench/sweep_collocation.py``, 80-triangle icosphere,
    hypersingular rows, P1): at ``eps/h = 0.15`` the error has a sharp minimum at
    ``t = 0.6``, which is exactly where the node clearance first reaches
    ``1.03 eps`` -- and it gets 4.5x and 8x WORSE at t = 0.7 and 0.8.  At
    ``eps/h = 0.3`` no ``t <= 0.8`` reaches 1 eps of clearance (the best is
    0.69 eps) and the error is still falling at the end of the sweep.  Both
    observations say the same thing, and it is ``../fbem/FINDINGS.md`` sec.2
    stated as a rule you can act on: **put the collocation point about one
    mollification length clear of the element boundary, and no further.**

    The clearance is (piecewise) linear in ``t``, so this is a grid search on
    the MEDIAN element's minimum node clearance -- median, not minimum, so one
    sliver in a mesh does not drive the whole patch.  Returns 0.0 for P0 (the
    centroid does not move) and ``shrink_max`` when the target is unreachable.

    NOT a default: ``Model`` uses the fixed
    ``defaults.COLLOCATION_SHRINK_BY_ORDER``.  Pass the result as
    ``Model(shrink=...)`` to use it, and sweep around it.
    """
    if int(order) == 0:
        return 0.0
    tri = _as_tris(tri_verts)
    e = np.asarray(eps, float)
    e = np.full(tri.shape[0], float(e)) if e.ndim == 0 else e
    ts = np.linspace(0.0, float(shrink_max), 201)
    ratio = np.array([float(np.median(node_clearance(tri, order, t).min(axis=1) / e))
                      for t in ts])
    if ratio[-1] <= target:
        return float(shrink_max)
    return float(ts[int(np.argmin(np.abs(ratio - target)))])


def eps_auto(tri_verts, eps_over_h: float = defaults.EPS_OVER_H,
             measure: str = "mean_edge") -> np.ndarray:
    """Per-element ``eps = eps_over_h * h`` (msd's ``eps="auto"``).

    Keeps eps/h fixed under grading and h-refinement -- which is exactly what
    you must NOT do when asking whether p-refinement pays (the answer inverts
    when eps is held fixed in absolute terms instead).  Sweep both.
    """
    return float(eps_over_h) * element_h(tri_verts, measure)


def unit_normal(tri):
    """clq's single-triangle normal, re-exported so callers need not import clq."""
    return clq.unit_normal(tri)
