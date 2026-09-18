"""Nodal layout: the map from (element, local node) to a global slip unknown.

WHY THIS IS A SEPARATE LAYER
----------------------------
Every influence matrix in :mod:`ddbem.assemble` is built DISCONTINUOUSLY --
one independent slip vector per (element, local node), ``3 * K * N_tri``
columns, with column ``3 * (K * s + k) + j`` carrying slip component ``j`` at
local node ``k`` of element ``s``.  That is the unambiguous layout: no two
elements share an unknown, so there is nothing to decide about material or
geometric edges, and the assembly is a pure loop over source triangles.

Continuity is a SEPARATE, LATER decision, and it is the one that actually pays
for higher order.  On a closed triangulated surface Euler gives
``#vertices ~ #triangles / 2`` and ``#edges ~ 3 #triangles / 2``, so per
triangle:

    layout                    slip unknowns per triangle
    ----------------------    --------------------------
    P0                        3          (1 node)
    P1 discontinuous          9          (3 nodes)
    P1 continuous             ~1.5       (V/T ~ 1/2)
    P2 discontinuous          18         (6 nodes)
    P2 continuous             ~6         ((V+E)/T ~ 2)

So continuous P1 is ~2x CHEAPER than P0 in unknowns, and discontinuous P1 is
3x more expensive.  The unknown-count win of higher-order DD is entirely a
continuity win.

But continuity is WRONG across:
  * a sharp geometric edge (the slip/displacement discontinuity field of two
    non-coplanar patches has no reason to be continuous in Cartesian
    components, and the traction operator is not even defined there);
  * a material interface;
  * a crack/fault TIP, where the physical DD must go to zero and a continuous
    basis that is shared with a non-cracked neighbour cannot express that;
  * the junction of two patches carrying different boundary conditions.

That is a real modelling decision, not a data-structure detail, which is why
this module exposes the scatter map explicitly instead of hiding it inside the
assembler.  :func:`discontinuous` is what Stage 1 uses and what every gate in
``verify/`` checks.  :func:`continuous` is the seam: it builds the
vertex/edge numbering and the scatter, and :meth:`NodalLayout.condense` applies
it to an already-assembled discontinuous matrix by summing columns.  Nothing in
``ddbem`` calls :func:`continuous` by default.

FORTRAN ORDERING
----------------
``condense`` is the one place where the F-ordering of the assembled matrix is
NOT free: it sums columns, so it must build a new buffer.  It does the sum on
the source-major (C-contiguous) transpose and returns ``.T``, so the result is
F-ordered again and ``scipy.linalg.lu_factor(overwrite_a=True)`` still factors
in place -- but at the cost of one extra full copy of the matrix.  See
:func:`ddbem.assemble.displacement_matrix` for the same trick applied for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def n_nodes(order: int) -> int:
    """Number of Lagrange nodes on a triangle of order ``order`` (1, 3, 6)."""
    order = int(order)
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2 (got {order})")
    return (order + 1) * (order + 2) // 2 if order else 1


@dataclass(frozen=True)
class NodalLayout:
    """Map from (element, local node) to a global slip node.

    Attributes
    ----------
    order : int
        0, 1 or 2 (P0 / P1 / P2).
    n_tri : int
        Number of source triangles.
    n_local : int
        Nodes per element, ``K`` (1, 3, 6).
    n_global : int
        Number of distinct global slip nodes.
    scatter : (n_tri, n_local) int array
        ``scatter[s, k]`` is the global node index of local node ``k`` of
        element ``s``.  For :func:`discontinuous` this is
        ``arange(n_tri * K).reshape(n_tri, K)``.
    continuous : bool
        True if any global node is shared by more than one element.
    """

    order: int
    n_tri: int
    n_local: int
    n_global: int
    scatter: np.ndarray
    continuous: bool

    @property
    def n_dof(self) -> int:
        """Number of scalar slip unknowns, ``3 * n_global``."""
        return 3 * self.n_global

    @property
    def n_dof_discontinuous(self) -> int:
        return 3 * self.n_local * self.n_tri

    def column_index(self, s: int, k: int, j: int) -> int:
        """Global column of slip component ``j`` at local node ``k`` of element ``s``."""
        return 3 * int(self.scatter[s, k]) + int(j)

    def _flat_columns(self) -> np.ndarray:
        """(3*K*n_tri,) global column for every discontinuous column, in order."""
        return (3 * self.scatter[:, :, None] + np.arange(3)[None, None, :]).reshape(-1)

    def condense(self, A_disc: np.ndarray) -> np.ndarray:
        """Sum the discontinuous columns of ``A_disc`` onto the global nodes.

        ``A_disc`` has ``3 * K * n_tri`` columns in the layout documented at the
        top of this module.  The result has :attr:`n_dof` columns and is
        F-ordered (see the module docstring on the copy this costs).  For a
        discontinuous layout this is a permutation-free identity and the matrix
        is returned unchanged.
        """
        A_disc = np.asarray(A_disc)
        if A_disc.shape[1] != self.n_dof_discontinuous:
            raise ValueError(f"expected {self.n_dof_discontinuous} columns, "
                             f"got {A_disc.shape[1]}")
        if not self.continuous and self.n_global == self.n_local * self.n_tri:
            return A_disc
        cols = self._flat_columns()
        src = np.ascontiguousarray(A_disc.T)              # (3*K*n_tri, n_rows)
        out = np.zeros((self.n_dof, src.shape[1]), dtype=src.dtype)
        np.add.at(out, cols, src)
        return out.T                                      # F-ordered again


def discontinuous(n_tri: int, order: int) -> NodalLayout:
    """Per-element unknowns: ``3 * K * n_tri`` of them, nothing shared."""
    K = n_nodes(order)
    n_tri = int(n_tri)
    scatter = np.arange(n_tri * K, dtype=np.intp).reshape(n_tri, K)
    return NodalLayout(order=int(order), n_tri=n_tri, n_local=K,
                       n_global=n_tri * K, scatter=scatter, continuous=False)


def continuous(triangles, order: int) -> NodalLayout:
    """Shared-node unknowns from a mesh connectivity array.

    ``triangles`` is the (n_tri, 3) vertex-index array of the source mesh.
    Global node numbering, matching clq's local node order
    (vertices, then edge midpoints 12, 23, 31):

        p = 0 : one node per element (identical to :func:`discontinuous`)
        p = 1 : one node per mesh VERTEX
        p = 2 : mesh vertices first, then one node per mesh EDGE
                (an edge is the sorted vertex pair, so the two elements that
                share it agree)

    THIS IS A SEAM, NOT A DEFAULT.  Read the module docstring before using it:
    sharing a node across a sharp geometric edge, a material interface or a
    crack tip is a modelling error, and this function cannot tell.
    """
    tri = np.asarray(triangles, dtype=np.intp)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"triangles must be (n_tri, 3), got {tri.shape}")
    n_tri = tri.shape[0]
    K = n_nodes(order)
    if order == 0:
        return discontinuous(n_tri, 0)
    n_vert = int(tri.max()) + 1
    if order == 1:
        scatter = tri.copy()
        return NodalLayout(order=1, n_tri=n_tri, n_local=K, n_global=n_vert,
                           scatter=scatter, continuous=True)
    # order 2: vertices, then edges 12, 23, 31
    pairs = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)  # (n_tri,3,2)
    key = np.sort(pairs, axis=2).reshape(-1, 2)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    scatter = np.empty((n_tri, 6), dtype=np.intp)
    scatter[:, :3] = tri
    scatter[:, 3:] = n_vert + inv.reshape(n_tri, 3)
    return NodalLayout(order=2, n_tri=n_tri, n_local=K,
                       n_global=n_vert + uniq.shape[0],
                       scatter=scatter, continuous=True)
