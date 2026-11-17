"""A fault's own traction kernel, compressed, at every material at once.

The stepper (``mbem.stepper``) applies, at every stage of every step, the
traction a slip puts on the fault that carries it. Dense that is n^2 per
state of the body, which is what bounds a stepped fault to some 1e4 unknowns.
It is a kernel sum, though, and two things about it are exact:

* **Two matrices serve every material.** The elastic traction of a slip
  (eigenstress removed, rule 4) is ``mu`` times a function affine in
  ``g = 1 / (1 - nu)`` (docs/viscoelastic.md section 2.3), so with ``H_lo``
  and ``H_hi`` the kernel at unit shear modulus and the two ends of a range
  of ``g``, the kernel at ANY ``(mu, lam)`` with ``g`` in that range is

      mu ( w H_lo + (1 - w) H_hi ),      w = (g_hi - g) / (g_hi - g_lo),

  a convex combination, so nothing cancels. Every state of a Laplace run is
  then the same two products with two numbers changed.
* **It is low rank between clusters.** It falls as the cube of distance, and
  blocks between well-separated clusters of a fault have rank 0 to 2 at 1e-4.

So the two matrices are stored in flat H form: near blocks dense, the rest
cross-approximated (``la.aca.compress_block``, which already compresses a
STACK of material-free matrices block by block and certifies each) and laid
out for one numba product (``la.flatview.FlatView``). Nothing of the size
n^2 is ever formed.

P0 faults only: a row is an element's centroid and a column the same
element's value, which is what a cluster tree of elements means. Faults of
ONE region: slips in different regions do not see each other except through
a boundary, which is not this operator's business.

THE BUILD IS A POOL OF PROCESSES. A cross approximation draws one row or
one column of a block at a time, and the traction kernel costs 5 microseconds
an element pair, so the far blocks are the build: 86 s of 87 at 7,771
elements, one block after another. The kernel is a ``parallel=True`` numba
function, which must not be called from Python threads (rule 9), so the
blocks go to PROCESSES, each with numba and BLAS on one thread; a block's
random lines are seeded by its own index, so the factors are the same
however many processes there were, or none, to the rounding of a
factorization on one thread or several. The pool is spawned,
not forked (a fork after numba's threads have started is not safe): the
usual rule then holds, that whatever script starts a build has its work
under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import os
import time

import numpy as np

from .. import defaults
from ..kernels import basis as kb
from ..kernels import tri_nodal as tn
from . import aca
from .cluster import build_cluster_tree, build_partition
from .flatview import FlatView, storage_dtype


def lam_of(mu: float, g: float) -> float:
    """``lam`` of the material with shear modulus ``mu`` and
    ``g = 1/(1 - nu) = 2 (lam + mu) / (lam + 2 mu)``, 1 <= g < 2."""
    return 2.0 * mu * (g - 1.0) / (2.0 - g)


def g_of(mu: float, lam: float) -> float:
    return 2.0 * (lam + mu) / (lam + 2.0 * mu)


def _stack(geo, rows, cols) -> np.ndarray:
    """``(B, 3 nr, 3 nc)``: the block of each stored matrix, exact, from a
    kernel's geometry ``geo = (x, n_c, tri_verts, normals, eps, lams)``."""
    x, n_c, tv, nrm, eps, lams = geo
    rows, cols = np.asarray(rows), np.asarray(cols)
    ids = np.arange(cols.size, dtype=np.int64).reshape(-1, 1)
    xr = np.ascontiguousarray(x[rows])
    nr = np.ascontiguousarray(n_c[rows])
    tc = np.ascontiguousarray(tv[cols])
    nc = np.ascontiguousarray(nrm[cols])
    ec = np.ascontiguousarray(eps[cols])
    return np.stack([tn.traction_matrix(xr, nr, tc, nc, ec, ids, cols.size,
                                        1.0, lam, 0, "slip") for lam in lams])


def _compress(geo, i, rows, cols, tol, seed, dtype):
    """One admissible block: ``(i, factors or None, retried, error, rank)``,
    ``factors`` the ``(U, V)`` of each stored matrix at the storage dtype.
    The block's random lines come from ``(seed, i)`` alone."""
    B = len(geo[5])
    res = aca.compress_block(
        aca.BlockEvalCache(lambda r, c: _stack(geo, r, c), rows, cols), B,
        tol=tol, rng=np.random.default_rng([seed, i]), exact_payload=False)
    if res.payload is None:                 # not low rank at this tolerance
        return i, None, res.retried, 0.0, 0
    unit = np.eye(B)
    facs = [tuple(np.ascontiguousarray(m, dtype=dtype)
                  for m in res.payload.combine(unit[b], tol))
            for b in range(B)]
    return i, facs, res.retried, res.max_err, res.payload.rank


_GEO = None                                 # a worker's copy of the geometry


def _worker_init(geo) -> None:
    import numba
    from threadpoolctl import threadpool_limits

    global _GEO
    numba.set_num_threads(1)
    threadpool_limits(1)
    _GEO = geo


def _worker(task):
    i, rows, cols, tol, seed, dtype = task
    return _compress(_GEO, i, rows, cols, tol, seed, dtype)


class FaultKernel:
    """Slip -> elastic traction on the centroids of ``faults`` (P0 patches of
    one region, in order), Cartesian, ``(3 n,) -> (3 n,)``, for every
    material whose ``g`` lies in ``g_range``.

    ``eps`` is the run's (``kb.resolve_patch_eps``). ``apply(x_lo, x_hi)`` is
    ``H_lo @ x_lo + H_hi @ x_hi``; ``weights(mu, lam)`` are the two numbers
    that make it the kernel at a material. A degenerate ``g_range`` (one
    material) stores one matrix. The sign is the system's: the block
    ``A[b_f, b_g]`` of ``model/equations.py``, ``+Ht``.

    ``processes``: how many build the far blocks (module docstring); None is
    ``defaults.FAULT_KERNEL_PROCESSES`` or the machine's cores if fewer, and
    a build of few blocks stays in this one. The result is the same.
    """

    def __init__(self, faults, eps, g_range, tol: float = None,
                 eta: float = None, min_leaf: int = None, seed: int = 0,
                 processes: int = None):
        t0 = time.perf_counter()
        faults = list(faults)
        for f in faults:
            if f.order != 0 or f.continuous:
                raise NotImplementedError(
                    f"FaultKernel: fault '{f.name}' is order {f.order} "
                    f"{f.layout}; the compressed kernel is P0 (a cluster "
                    f"tree of elements)")
        tol = defaults.BLOCK_COMPRESSION_TOL if tol is None else float(tol)
        eta = defaults.ADMISSIBILITY_ETA if eta is None else float(eta)
        min_leaf = defaults.CLUSTER_MIN_LEAF if min_leaf is None else int(min_leaf)
        arrays = kb.MeshArrays()
        parts = [arrays.source_arrays(f.mesh) for f in faults]
        self._tv = np.ascontiguousarray(np.concatenate([p[0] for p in parts]))
        self._nrm = np.ascontiguousarray(np.concatenate([p[1] for p in parts]))
        self._eps = np.concatenate([kb.resolve_patch_eps(eps, f) for f in faults])
        self._x = np.ascontiguousarray(np.concatenate(
            [f.collocation_points() for f in faults]))
        self._n_c = np.ascontiguousarray(np.concatenate(
            [f.collocation_normals() for f in faults]))
        self.n = int(self._x.shape[0])
        self.shape = (3 * self.n, 3 * self.n)
        g_lo, g_hi = float(g_range[0]), float(g_range[1])
        if not (1.0 <= g_lo <= g_hi < 2.0):
            raise ValueError(f"g_range {g_range}: need 1 <= g_lo <= g_hi < 2")
        self.g = (g_lo, g_hi) if g_hi - g_lo > 1.0e-12 else (g_lo,)
        self._lam = tuple(lam_of(1.0, g) for g in self.g)
        self.tol = tol
        B = len(self.g)

        tree = build_cluster_tree(self._x, min_leaf)
        part = build_partition(tree, tree, eta)
        dtype = storage_dtype(tol)
        geo = (self._x, self._n_c, self._tv, self._nrm, self._eps, self._lam)
        tasks = [(i, rows, cols, tol, seed, dtype)
                 for i, (rows, cols) in enumerate(part.admissible)]
        if processes is None:
            processes = min(defaults.FAULT_KERNEL_PROCESSES,
                            os.cpu_count() or 1)
            if len(tasks) < defaults.FAULT_KERNEL_POOL_MIN_BLOCKS:
                processes = 1
        if processes > 1 and tasks:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(
                    max_workers=processes,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_worker_init, initargs=(geo,)) as pool:
                done = list(pool.map(_worker, tasks, chunksize=max(
                    1, len(tasks) // (16 * processes))))
        else:
            done = [_compress(geo, *task) for task in tasks]
        lr = [[] for _ in range(B)]
        dense = list(part.dense)
        stats = dict(blocks=len(tasks), exact=0, retried=0, rank=[],
                     max_err=0.0)
        for i, facs, retried, err, rank in done:
            rows, cols = part.admissible[i]
            stats["retried"] += int(retried)
            if facs is None:
                stats["exact"] += 1
                dense.append((rows, cols))
                continue
            stats["max_err"] = max(stats["max_err"], err)
            stats["rank"].append(rank)
            for b in range(B):
                lr[b].append((rows, cols, *facs[b]))
        d_ptr = np.zeros(len(dense) + 1, dtype=np.int64)
        for i, (rows, cols) in enumerate(dense):
            d_ptr[i + 1] = d_ptr[i] + 9 * len(rows) * len(cols)
        D = [np.empty(d_ptr[-1], dtype=dtype) for _ in range(B)]
        for i, (rows, cols) in enumerate(dense):
            blk = self.stack(rows, cols)
            for b in range(B):
                D[b][d_ptr[i]:d_ptr[i + 1]] = blk[b].ravel()
        self.views = [FlatView(self.shape, lr[b], dense, D[b], d_ptr,
                               dtype=dtype) for b in range(B)]
        rank = np.array(stats["rank"] or [0])
        self.report = {
            "elements": self.n, "matrices": B, "tol": tol,
            "admissible_blocks": stats["blocks"],
            "dense_blocks": len(dense), "exact_blocks": stats["exact"],
            "retried_blocks": stats["retried"],
            "rank_median": float(np.median(rank)), "rank_max": int(rank.max()),
            "certified_error": float(stats["max_err"]),
            "bytes": int(sum(v.nbytes() for v in self.views)),
            "bytes_per_unknown": float(sum(v.nbytes() for v in self.views)
                                       / (3 * self.n)),
            "dense_bytes": int(B * 8 * (3 * self.n) ** 2),
            "processes": int(processes),
            "build_s": time.perf_counter() - t0}

    # -- the kernel itself ----------------------------------------------

    def stack(self, rows, cols) -> np.ndarray:
        """``(B, 3 nr, 3 nc)``: the block of each stored matrix, exact."""
        return _stack((self._x, self._n_c, self._tv, self._nrm, self._eps,
                       self._lam), rows, cols)

    def weights(self, mu: float, lam: float) -> np.ndarray:
        """``(B,)``: the kernel at ``(mu, lam)`` is ``sum_b weights[b] H_b``."""
        if len(self.g) == 1:
            return np.array([float(mu)])
        g = g_of(mu, lam)
        lo, hi = self.g
        if not (lo - 1.0e-9 <= g <= hi + 1.0e-9):
            raise ValueError(f"g = {g:.6g} of (mu, lam) = ({mu:g}, {lam:g}) "
                             f"is outside this kernel's range {self.g}")
        w = (hi - g) / (hi - lo)
        return float(mu) * np.array([w, 1.0 - w])

    def apply(self, *x) -> np.ndarray:
        """``sum_b H_b @ x[b]``, each ``x[b]`` a Cartesian ``(3 n,)``."""
        out = self.views[0].matvec(x[0])
        for v, xb in zip(self.views[1:], x[1:]):
            out += v.matvec(xb)
        return out

    def dense(self, mu: float, lam: float) -> np.ndarray:
        """The kernel at a material, exact and dense: for a gate."""
        every = np.arange(self.n)
        return np.tensordot(self.weights(mu, lam), self.stack(every, every),
                            axes=1)
