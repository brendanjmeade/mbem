"""Higher-order (P0 / P1 / P2) nodal force-density influence matrices.

The constant-density assembler in :mod:`assembly` puts ONE force density vector
(force per unit AREA) on each boundary triangle.  This module puts a Lagrange
nodal density of order ``p`` on each triangle instead, built directly on
``clq.influence``, which returns the nodal influence tensors in closed form:

    G[n, k, i, j]      force density component j at node k  ->  displacement i
    S[n, k, i, j, c]   force density component c at node k  ->  stress ij

with ``K = n_nodes(p) = 1, 3, 6`` nodes for ``p = 0, 1, 2`` at
``clq.nodes(tri, p)`` -- the centroid for p = 0, the vertices for p = 1, the
vertices then the edge midpoints 12, 23, 31 for p = 2.

DISCONTINUOUS (per-element) density
-----------------------------------
Every element owns its own nodes: element ``e`` owns unknowns

    3*K*e + 3*k + c        k = 0 .. K-1 local node,  c = 0, 1, 2 component

so the unknown count is ``3*K*N_tri`` and there is no connectivity, no shared
node numbering and no continuity constraint between neighbouring triangles.
For ``p = 0`` that is exactly :mod:`assembly`'s layout (``3*e + c``), so the
matrices reduce to it entry for entry -- gated in
``verify/verify_ho_assembly.py``.

**The unknown count is what the comparison must be made against, not h.**
P1 triples and P2 sextuples the number of unknowns on the same mesh, and the
dense solve is cubic in that, so a convergence plot of error against ``h`` will
flatter the higher orders.  Error against UNKNOWN COUNT (equivalently against
matrix dimension, or against solve time) is the honest axis; report both.

Cost
----
``clq`` is pure numpy and evaluates one source triangle against all field
points per call, so a dense N_tri x N_tri operator costs N_tri python-level
calls.  Measured on this machine 2026-09-17 by ``verify/verify_ho_assembly.py``
(closed box mesh, field points = centroids, eps = 0.3 h, serial), seconds for
the traction matrix and the matrix size:

    N_tri      p=0             p=1             p=2
      512    10.7 s 0.02 GB   16.5 s 0.05 GB   25.5 s 0.11 GB
     2048    61.0 s 0.28 GB   95.7 s 0.84 GB  156.9 s 1.69 GB

i.e. order p costs about (1, 1.55, 2.5) x the P0 time on the same mesh, and
P0 here is itself ~50x the numba assembler in :mod:`assembly`.  Keep meshes
small: at N_tri = 4608 the P2 matrix alone is ~8.6 GB and the LU another copy.

``workers > 1`` forks a process pool over source triangles and returns a
BITWISE-identical matrix (gated).  Same meshes, ``workers=8`` on this 16-core
machine, seconds:

    N_tri      p=0     p=1     p=2
      512     1.8     2.7     4.1
     2048    10.1    15.7    23.4

-- a 6.1-6.7x speed-up, so the higher-order study is affordable.  It is off by
default because forking a process that already holds a threaded BLAS is not
something this module should do behind the caller's back.

Memory
------
Returned matrices are FORTRAN-ordered, like :func:`assembly.traction_matrix`
(and unlike :func:`assembly.displacement_matrix`), so a column block is
contiguous both to fill and to hand to ``scipy.linalg.lu_factor(A,
overwrite_a=True)``.  A dense P2 traction matrix at N_tri = 2048 is
6144 x 36864 = 1.8 GB; size the mesh before you call.

Collocation (a note for whoever writes the rows, NOT implemented here)
---------------------------------------------------------------------
With a constant density the free term in R1/R2 is a clean ``q_e/2`` because the
collocation point is the centroid and the density there IS the unknown.  With a
nodal density it is ``(1/2) sum_k N_k(x_c) q_{e,k}``: the jump acts on the
density AT the collocation point, so the identity block is the matrix of shape
function values, not ``I``.  Two consequences worth knowing before the next
stage picks its collocation points:

* collocating at the nodes themselves makes that block exactly ``I`` -- but the
  P1/P2 nodes are vertices and edge midpoints, which sit ON element edges and,
  on a polyhedron, on the very edges where L2 says this formulation already
  stalls.  The free term there is not 1/2.
* the usual fix for a discontinuous basis is to collocate at SHRUNK nodes,
  pulled toward the centroid in barycentric coordinates (``lam -> (1-t) lam +
  t/3``, t ~ 0.1-0.2).  Then the free term is 1/2 on a flat element and the
  identity block is the dense ``K x K`` matrix ``N_k(x_c)``, which
  ``clq.shape_functions(tri, order, x_c)`` returns.

Nothing here modifies ``assembly.py``, ``model.py``, or anything in ``clq/``,
``msd/``, ``moss/`` or ``medt_paper/``.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

CLQ = str(pathlib.Path(__file__).resolve().parent.parent / "clq")
if CLQ not in sys.path:
    sys.path.insert(0, CLQ)

import clq                                                        # noqa: E402

# Voigt row order, matching assembly.stress_matrix: xx, yy, zz, xy, xz, yz.
_VI = np.array([0, 1, 2, 0, 0, 1])
_VJ = np.array([0, 1, 2, 1, 2, 2])

# Mirror msd's _tri_frame: a triangle whose |e1 x e2| falls below this is
# treated as degenerate and contributes a ZERO column block, exactly as
# assembly.py's OK mask does, rather than raising from inside clq.
DEGENERATE_TWICE_AREA = 1e-30

_KIND_ROWS = {"G": 3, "T": 3, "V": 6}
_KIND_WANT = {"G": ("G",), "T": ("S",), "V": ("S",)}


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------

def n_nodes(order: int) -> int:
    """Nodes per element: 1, 3, 6 for order 0, 1, 2."""
    return clq.n_nodes(int(order))


def n_unknowns(n_tri: int, order: int) -> int:
    """Length of the discontinuous nodal density vector: ``3 * K * n_tri``."""
    return 3 * n_nodes(order) * int(n_tri)


def element_nodes(tri_verts, order: int) -> np.ndarray:
    """(N_tri, K, 3) node coordinates, in the column order of the matrices.

    Row ``e``, node ``k`` owns unknowns ``3*K*e + 3*k + (0, 1, 2)``.
    """
    tri_verts = np.ascontiguousarray(tri_verts, float).reshape(-1, 3, 3)
    p = int(order)
    return np.stack([clq.nodes(t, p) for t in tri_verts])


def uniform_density(q, n_tri: int, order: int) -> np.ndarray:
    """The nodal vector of a spatially CONSTANT density ``q`` (3,).

    Every node of every element carries the same vector, which by partition of
    unity is the same physical density a P0 unknown of ``q`` represents.  This
    is the vector the constant-density consistency gate contracts with.
    """
    q = np.asarray(q, float).reshape(3)
    return np.tile(q, n_nodes(order) * int(n_tri))


def _check_order(order):
    p = int(order)
    if p < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    return p


def _eps_array(eps, n_tri):
    e = np.asarray(eps, float)
    if e.ndim == 0:
        e = np.full(n_tri, float(e))
    e = np.ascontiguousarray(e.ravel(), float)
    if e.shape != (n_tri,):
        raise ValueError(f"eps must be a scalar or ({n_tri},), got {e.shape}")
    if np.any(e < 0.0):
        raise ValueError("eps must be >= 0")
    return e


def _check_nodes(nodes, tri_verts, order):
    """Fail loudly if the caller's node array disagrees with ours."""
    if nodes is None:
        return
    want = element_nodes(tri_verts, order)
    got = np.asarray(nodes, float)
    if got.shape != want.shape:
        raise ValueError(f"nodes must have shape {want.shape}, got {got.shape}")
    scale = max(1.0, float(np.abs(want).max()))
    d = float(np.abs(got - want).max()) / scale
    if d > 1e-10:
        raise ValueError(
            "the supplied nodes disagree with clq.nodes(tri, order) by "
            f"{d:.2e} (relative); the unknown ordering would not match the "
            "assembled matrix")


# ---------------------------------------------------------------------------
# one source triangle -> one column block
# ---------------------------------------------------------------------------

def _twice_area(tri):
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    return float(np.linalg.norm(np.cross(e1, e2)))


def _source_block(kind, x_field, n_field, tri, eps_s, mu, nu, order, far_field):
    """(rows, 3K) influence of one source triangle's K nodes."""
    n_f = x_field.shape[0]
    K = n_nodes(order)
    nrow = _KIND_ROWS[kind] * n_f
    if _twice_area(tri) < DEGENERATE_TWICE_AREA:
        return np.zeros((nrow, 3 * K))
    inf = clq.influence(x_field, tri, mu, nu, eps_s, order=order,
                        want=_KIND_WANT[kind], far_field=far_field)
    if kind == "G":
        # G[n, k, i, j] -> row 3n+i, col 3k+j
        return np.ascontiguousarray(inf.G.transpose(0, 2, 1, 3)).reshape(nrow, 3 * K)
    if kind == "T":
        # t_i = S[n, k, i, j, c] n_field[n, j]  -> row 3n+i, col 3k+c
        T = np.einsum("nkijc,nj->nikc", inf.S, n_field)
        return np.ascontiguousarray(T).reshape(nrow, 3 * K)
    # kind == "V": Voigt stress rows, S[n, k, i, j, c] -> row 6n+v, col 3k+c
    Sv = inf.S[:, :, _VI, _VJ, :]                      # (n_f, K, 6, 3)
    return np.ascontiguousarray(Sv.transpose(0, 2, 1, 3)).reshape(nrow, 3 * K)


# ---------------------------------------------------------------------------
# process-pool plumbing (optional; off by default)
# ---------------------------------------------------------------------------

_SHARED: dict = {}


def _pool_init(payload):
    _SHARED.clear()
    _SHARED.update(payload)


def _pool_job(bounds):
    a, b = bounds                                  # b > a always (see _assemble)
    s = _SHARED
    blocks = [_source_block(s["kind"], s["x_field"], s["n_field"],
                            s["tri_verts"][i], float(s["eps"][i]), s["mu"],
                            s["nu"], s["order"], s["far_field"])
              for i in range(a, b)]
    return a, np.hstack(blocks)


def _assemble(kind, x_field, n_field, tri_verts, eps, mu, nu, order,
              nodes=None, *, far_field="hybrid", workers=1, progress=False):
    order = _check_order(order)
    x_field = np.ascontiguousarray(x_field, float).reshape(-1, 3)
    tri_verts = np.ascontiguousarray(tri_verts, float).reshape(-1, 3, 3)
    n_f, n_s = x_field.shape[0], tri_verts.shape[0]
    if kind == "T":
        n_field = np.ascontiguousarray(n_field, float).reshape(-1, 3)
        if n_field.shape[0] != n_f:
            raise ValueError(f"n_field has {n_field.shape[0]} rows, x_field {n_f}")
    else:
        n_field = None
    eps = _eps_array(eps, n_s)
    _check_nodes(nodes, tri_verts, order)
    K = n_nodes(order)
    out = np.empty((_KIND_ROWS[kind] * n_f, 3 * K * n_s), order="F")

    t0 = time.perf_counter()
    if int(workers) > 1 and n_s > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        W = min(int(workers), n_s)
        ctx = mp.get_context("fork" if os.name == "posix" else "spawn")
        # many small chunks: each returned block is bounded, so the parent's
        # peak memory stays close to the matrix itself.
        nchunk = max(W, min(n_s, 8 * W))
        edges = np.linspace(0, n_s, nchunk + 1).astype(int)
        bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        payload = dict(kind=kind, x_field=x_field, n_field=n_field,
                       tri_verts=tri_verts, eps=eps, mu=float(mu), nu=float(nu),
                       order=order, far_field=far_field)
        done = 0
        with ProcessPoolExecutor(max_workers=W, mp_context=ctx,
                                 initializer=_pool_init,
                                 initargs=(payload,)) as ex:
            for a, blk in ex.map(_pool_job, bounds, chunksize=1):
                out[:, 3 * K * a:3 * K * a + blk.shape[1]] = blk
                done += blk.shape[1] // (3 * K)
                if progress:
                    print(f"    {kind} {done}/{n_s} sources "
                          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        for s in range(n_s):
            out[:, 3 * K * s:3 * K * (s + 1)] = _source_block(
                kind, x_field, n_field, tri_verts[s], float(eps[s]), mu, nu,
                order, far_field)
            if progress and (s + 1) % 64 == 0:
                print(f"    {kind} {s+1}/{n_s} sources "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# public assemblers
# ---------------------------------------------------------------------------

def traction_matrix_ho(x_field, n_field, tri_verts, eps, mu, nu, order,
                       nodes=None, *, far_field="hybrid", workers=1,
                       progress=False):
    """(3 N_f, 3 K N_tri) traction influence of a nodal force density.

    Row ``3*f + i`` is component ``i`` of the traction ``sigma . n_field[f]``
    at ``x_field[f]``; column ``3*K*e + 3*k + c`` is a unit force density
    component ``c`` at node ``k`` of element ``e``.  This is the ``B`` block of
    the force-element rows -- the ON-surface (average) value, with no free
    term: the single layer's jump is carried by the ``q/2`` written explicitly
    in ``model.py``'s R1 and R2.

    ``eps`` is a scalar or a per-source-element array, as in
    :func:`assembly.traction_matrix`.  ``order`` is 0, 1 or 2.  ``nodes``, if
    given, must be the ``(N_tri, K, 3)`` array from :func:`element_nodes`; it
    is checked, not used, so a caller whose unknown labelling came from a
    different node convention fails here instead of silently solving the wrong
    system.  The result is Fortran-ordered.
    """
    return _assemble("T", x_field, n_field, tri_verts, eps, mu, nu, order,
                     nodes, far_field=far_field, workers=workers,
                     progress=progress)


def displacement_matrix_ho(x_field, tri_verts, eps, mu, nu, order, nodes=None,
                           *, far_field="hybrid", workers=1, progress=False):
    """(3 N_f, 3 K N_tri) displacement influence of a nodal force density.

    Row ``3*f + i``, column ``3*K*e + 3*k + c``; the ``G`` block of R3.  Same
    conventions and same ``nodes`` contract as :func:`traction_matrix_ho`.
    ``eps = 0`` is allowed here and only here -- the single layer is weakly
    singular on its own element -- because clq permits it for ``want=("G",)``.
    """
    return _assemble("G", x_field, None, tri_verts, eps, mu, nu, order,
                     nodes, far_field=far_field, workers=workers,
                     progress=progress)


def stress_matrix_ho(x_field, tri_verts, eps, mu, nu, order, nodes=None, *,
                     far_field="hybrid", workers=1, progress=False):
    """(6 N_f, 3 K N_tri) Voigt stress influence (xx, yy, zz, xy, xz, yz).

    Row ``6*f + v``, column ``3*K*e + 3*k + c``; the Voigt row order matches
    :func:`assembly.stress_matrix`.
    """
    return _assemble("V", x_field, None, tri_verts, eps, mu, nu, order,
                     nodes, far_field=far_field, workers=workers,
                     progress=progress)
