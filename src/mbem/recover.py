"""The readout near a face: a continuous displacement recovered from the
element-constant one the solve returns.

The representation formula integrates the boundary displacement as a double
layer. A value per element steps at every element edge, each step is a small
dislocation line, and an observation point closer than about half an element
resolves them one by one: 19 % in stress at 0.1 h below a face and 3 % at
0.5 h, with the EXACT value on every element and no solve at all. So no
solver that returns a value per element removes it, and none is changed here.
What changes is what the near pairs are read from:

* **the displacement is made continuous.** ``recovery_operator`` puts a value
  at every vertex and edge midpoint (a P2 layout, one value per SHARED node):
  the value there of the quadratic that best fits the elements two rings
  around it. Not across a fault, where the displacement really jumps
  (``shared_nodes.split_nodes``), and not across a crease.
* **the traction is left as solved.** Its steps are a weak (log) feature and
  it carries real concentrations -- where a fault cuts an interface, each
  region's traction has a spike -- so smoothing it moves the force it carries
  and with it the whole interior (7-9 % on the megathrust).
* **the mollification is narrowed** on the recovered patches, which thins
  the layer where the field falls to half its value on the face. Safe only
  with the continuous density.

A patch takes part if it is a boundary patch of order 0, whatever gives its
displacement (a solved slot, a prescribed datum, a window's far field).
Faults are untouched: a fault's width is the model's own.
"""
from __future__ import annotations

import numpy as np

from . import defaults
from .model.core import FAULT_KINDS
from .model.shared_nodes import split_nodes


def _fit(x, hint, ptr, cols, cen, nrm, order):
    """Weights of the fitted value at each point ``x`` (n, 3) over its
    elements (CSR ``ptr``/``cols``): ``(rows, cols, vals, done)``.

    The fit is in the tangent plane of ``hint`` (a unit normal per point),
    over the elements that face its way. ``order`` 2 fits a quadratic, 1 a
    plane, 0 takes the mean; a point whose elements are too few for the
    order asked, or whose fit does not return a constant, is not ``done``.
    """
    n = x.shape[0]
    count = np.diff(ptr)
    row = np.repeat(np.arange(n), count)
    # a crease is not fitted across
    keep = np.einsum("ij,ij->i", nrm[cols], hint[row]) > defaults.READOUT_CREASE_COS
    few = np.bincount(row, weights=keep, minlength=n) < 3
    keep |= few[row]
    row, cols = row[keep], cols[keep]
    count = np.bincount(row, minlength=n)
    need = (1, defaults.READOUT_MIN_LINEAR, defaults.READOUT_MIN_QUADRATIC)[order]
    done = count >= need

    a = np.cross(hint, [1.0, 0.0, 0.0])
    alt = np.linalg.norm(a, axis=1) < 0.3
    a[alt] = np.cross(hint[alt], [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(hint, a)
    d = cen[cols] - x[row]
    p = np.einsum("ij,ij->i", d, a[row])
    q = np.einsum("ij,ij->i", d, b[row])

    start = np.concatenate([[0], np.cumsum(count)])
    out_r, out_c, out_v = [], [], []
    for k in np.unique(count[done]):
        sel = np.flatnonzero(done & (count == k))
        at = start[sel][:, None] + np.arange(k)[None, :]          # (m, k)
        pk, qk = p[at], q[at]
        if order == 0:
            w = np.full(at.shape, 1.0 / k)
        else:
            s = np.maximum(np.abs(pk).max(axis=1), np.abs(qk).max(axis=1))
            s = np.where(s > 0.0, s, 1.0)[:, None]
            pk, qk = pk / s, qk / s
            cols_g = [np.ones_like(pk), pk, qk]
            if order == 2:
                cols_g += [pk * pk, pk * qk, qk * qk]
            w = np.linalg.pinv(np.stack(cols_g, axis=2), rcond=1e-8)[:, 0, :]
            # a fit that does not return a constant has no claim on the node
            bad = np.abs(w.sum(axis=1) - 1.0) > 1e-8
            done[sel[bad]] = False
            sel, at, w = sel[~bad], at[~bad], w[~bad]
        out_r.append(np.repeat(sel, k))
        out_c.append(cols[at].ravel())
        out_v.append(w.ravel())
    cat = (lambda parts, dt: np.concatenate(parts) if parts else np.zeros(0, dt))
    return cat(out_r, np.int64), cat(out_c, np.int64), cat(out_v, float), done


def recovery_operator(mesh, faults=()):
    """Sparse ``(6 N, N)``: a value per element -> the element-major nodal
    values of a continuous P2 density on ``mesh`` (the kernels' node order,
    v1 v2 v3 m12 m23 m31). Rows sum to one; a quadratic field is returned
    exactly wherever two rings of elements determine one.

    One value per shared node, so the density is continuous across every
    edge, except across ``faults`` (``TriMesh``), whose trace splits the
    nodes on it by side.
    """
    from scipy import sparse

    V = np.asarray(mesh.vertices, float)
    T = np.asarray(mesh.triangles, np.int64)
    N = T.shape[0]
    nodes, idx = split_nodes(V, T, faults)
    M = nodes.shape[0]
    cen = V[T].mean(axis=1)
    nrm, _ = mesh.normals_and_areas()
    nrm = np.asarray(nrm, float)

    # one node per edge of the split surface, in the element's edge order
    ea, eb = idx, idx[:, [1, 2, 0]]
    key = np.minimum(ea, eb) * np.int64(M) + np.maximum(ea, eb)
    _u, first, edge = np.unique(key.ravel(), return_index=True,
                                return_inverse=True)
    edge = np.asarray(edge).reshape(N, 3)
    E = first.size
    fe, fk = np.divmod(first, 3)                     # an element and its edge
    mid = 0.5 * (V[T[fe, fk]] + V[T[fe, (fk + 1) % 3]])
    x = np.vstack([nodes, mid])                      # the M + E fit points

    # a normal for every fit point: that of one element that owns it
    owner = np.empty(M + E, np.int64)
    owner[idx.ravel()[::-1]] = np.repeat(np.arange(N), 3)[::-1]
    owner[M:] = fe
    hint = nrm[owner]

    e2n = sparse.csr_matrix((np.ones(3 * N), (np.repeat(np.arange(N), 3),
                                              idx.ravel())), shape=(N, M))
    n2e = e2n.T.tocsr()
    e2e = (e2n @ n2e).tocsr()                        # elements sharing a node
    pick = sparse.csr_matrix(
        (np.ones(2 * E), (np.repeat(np.arange(E), 2),
                          np.column_stack([ea.ravel()[first],
                                           eb.ravel()[first]]).ravel())),
        shape=(E, M))
    one = sparse.vstack([n2e, pick @ n2e]).tocsr()   # fit point -> its elements

    rows, cols, vals = [], [], []
    todo = np.arange(M + E)
    # two rings of elements; more where two do not determine the fit (a
    # corner of the patch that one element owns), and only then a lower order
    for order, rings in ((2, 2), (2, 3), (2, 4), (1, 2), (1, 3), (0, 1)):
        if todo.size == 0:
            break
        reach = one[todo]
        for _ in range(rings - 1):
            reach = reach @ e2e
        reach = reach.tocsr()
        reach.sort_indices()
        r, c, v, done = _fit(x[todo], hint[todo], reach.indptr, reach.indices,
                             cen, nrm, order)
        rows.append(todo[r]); cols.append(c); vals.append(v)
        todo = todo[~done]
    if todo.size:
        raise ValueError(f"{todo.size} node(s) of the surface belong to no "
                         f"element")
    W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                  np.concatenate(cols))),
                          shape=(M + E, N))
    ids = np.concatenate([idx, M + edge], axis=1)    # (N, 6)
    return W[ids.ravel()].tocsr()


class NearField:
    """What ``evaluate_displacement`` / ``evaluate_stress`` / ``volume.sample``
    take as ``near=``: the recovery operator of every boundary patch of
    order 0, built once for a model's geometry.

    Patches are identified by ``is``, and a state's model
    (``RegionModel.with_materials``) shares its base's, so one of these
    serves every state of a run.
    """

    def __init__(self, model, eps_factor: float = None, radius_h: float = None,
                 standoff_eps: float = None):
        pick = lambda v, d: d if v is None else float(v)
        self.eps_factor = pick(eps_factor, defaults.READOUT_EPS_FACTOR)
        self.radius_h = pick(radius_h, defaults.READOUT_NEAR_RADIUS_H)
        self.standoff_eps = pick(standoff_eps, defaults.READOUT_STANDOFF_EPS)
        faults = [f.mesh for r in model.regions for f in r.faults]
        self._ops = {}
        self._patches = []
        for r in model.regions:
            for p in r.patches:
                if id(p) in self._ops or p.order != 0 or p.bc in FAULT_KINDS:
                    continue
                self._ops[id(p)] = recovery_operator(p.mesh, faults)
                self._patches.append(p)          # keeps the ids alive

    def recovers(self, patch) -> bool:
        return id(patch) in self._ops

    def recover(self, patch, values) -> np.ndarray:
        """Element values (N, 3) of ``patch`` -> the P2 nodal density (6 N, 3)."""
        return np.ascontiguousarray(
            self._ops[id(patch)] @ np.asarray(values, float))
