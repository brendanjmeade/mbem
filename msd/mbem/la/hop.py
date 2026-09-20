"""Compressed pair operators with material-basis storage.

A ``PairCompressed`` represents ALL basis matrices of one
(field mesh, source mesh, kernel) pair in block-compressed form:
admissible blocks as per-basis low-rank factors (see :mod:`.aca`) or,
when not low rank at the tolerance, as exact dense stacks; near-field
leaf blocks as exact dense basis stacks. A material enters only through
its coefficient vector: the per-material view combines the factors
(with one QR+SVD re-truncation per low-rank block), certifies every
recombined block on its exact rows and columns (a block that fails is
applied exactly in that view), and is cached, so the SAME object serves
every region material. Coefficients are real (both backends allocate
real accumulators).

Threads: block compression and view recombination run with BLAS pinned
to one thread (``threadpoolctl``): the per-block work is many small
QR/SVD calls whose multi-threaded BLAS only oversubscribes the machine.
Kernel work runs on a Python thread pool in the serial nogil variant
(it scales; the parallel prange kernels are called from the main thread
only), while the BLAS calls themselves stay serial -- OpenBLAS
serializes concurrent callers on its buffer lock, and a pool of
recompressions is slower than one thread (measured).
"""

from __future__ import annotations

import os
import threading as _threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from .. import defaults
from ..kernels import KERNEL_T, kernel_n_basis
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from .aca import (BasisLR, BlockEvalCache, certify_combined, compress_block,
                  recompress)
from .cluster import build_cluster_tree, build_partition


def _dof_idx(elems: np.ndarray) -> np.ndarray:
    return (3 * elems[:, None] + np.arange(3)[None, :]).ravel()


def require_order0(model) -> None:
    """The compressed path (PairCompressed, HBackend, DisplacementEvaluator)
    is P0-only: one centroid per element on both sides of every pair. Raise
    at construction on any patch or fault of ``model`` with order > 0."""
    for r in model.regions:
        for p in list(r.patches) + list(r.faults):
            if p.order:
                raise NotImplementedError(
                    "compressed backend: order > 0 patches are not supported; "
                    f"use DenseBackend (patch '{p.name}' has order {p.order})")


class _BasisEval:
    """Subset evaluators for one (field, source, kernel) pair."""

    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr,
                 arrays: kb.MeshArrays | None = None):
        a = arrays if arrays is not None else kb.MeshArrays()
        self.x_field = a.field_points(field_mesh)
        tri_verts, normals = a.source_arrays(source_mesh)
        self.tri_verts = tri_verts
        self.normals = normals
        self.eps = eps_arr
        self.kernel = kernel
        self.n_basis = kernel_n_basis(kernel)      # raises on a bad tag

    def stack(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """(B, 3*len(rows), 3*len(cols)) exact basis sub-stack."""
        xf = self.x_field[rows]
        tv = self.tri_verts[cols]
        ee = self.eps[cols]
        if self.kernel == KERNEL_T:
            return tk.t_basis_matrices(xf, tv, self.normals[cols], ee)
        return tk.u_basis_matrices(xf, tv, ee)

    def stack_serial(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """As ``stack`` but serial + nogil -- safe (and scalable) from
        Python worker threads; the parallel=True kernels crash numba's
        workqueue threading layer under concurrent calls."""
        xf = self.x_field[rows]
        tv = self.tri_verts[cols]
        ee = self.eps[cols]
        if self.kernel == KERNEL_T:
            return tk.t_basis_matrices_serial(xf, tv, self.normals[cols], ee)
        return tk.u_basis_matrices_serial(xf, tv, ee)


class PairCompressed:
    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr,
                 tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK,
                 tree_cache: dict | None = None,
                 arrays: kb.MeshArrays | None = None,
                 n_workers: int | None = None,
                 storage: str = "basis",
                 combine_for=None):
        """``storage="basis"`` (default) keeps the GEOMETRY-ONLY per-basis
        factors/stacks: any material recombines for free, at B-fold
        storage (B = 3 for U, 6 for T). ``storage="combined"`` builds the
        views of the coefficient vectors in ``combine_for`` and then
        DROPS the per-basis payloads -- 1x storage, the memory mode for
        large models; a material NOT in ``combine_for`` (e.g. a later
        rebuild_for_materials) triggers a transient re-compression of
        this pair. Prefer "basis" for material sweeps when it fits.

        Every view is certified for its own material when it is combined
        (``aca.certify_combined``), whether or not the material was
        declared here, so a view depends on the factors and the material
        alone: a transient re-compression reproduces a fresh build.
        """
        if storage not in ("basis", "combined"):
            raise ValueError(storage)
        arrays = arrays if arrays is not None else kb.MeshArrays()
        self.eval = _BasisEval(field_mesh, source_mesh, kernel, eps_arr,
                               arrays=arrays)
        self.n_field = field_mesh.n_triangles
        self.n_source = source_mesh.n_triangles
        self.n_basis = self.eval.n_basis
        self.shape = (3 * self.n_field, 3 * self.n_source)
        self.tol = tol
        self.storage = storage
        self._n_workers = (n_workers if n_workers is not None
                           else min(16, os.cpu_count() or 1))

        tree_cache = {} if tree_cache is None else tree_cache

        def _tree(mesh):
            key = id(mesh)
            t = tree_cache.get(key)
            if t is None:
                t = build_cluster_tree(arrays.field_points(mesh), min_leaf)
                tree_cache[key] = t
            return t

        self._part = build_partition(_tree(field_mesh), _tree(source_mesh),
                                     eta=eta, max_admissible=max_admissible)
        self.n_lowrank = len(self._part.admissible)     # admissible blocks
        self.n_dense = len(self._part.dense)
        self.n_fallback = 0        # per-basis certificate failures at build
        self.n_capped = 0          # admissible blocks dense at the rank cap
        self.n_view_fallback = 0   # combined-certificate failures, worst view
        self.max_verified_err = 0.0

        self.blocks = self._compress_all()

        # Per-material views: LRU-bounded by count and bytes (``_evict``).
        # Unbounded caching leaked one full recombined operator PER
        # MATERIAL across a material sweep.
        self._views: OrderedDict[bytes, list] = OrderedDict()
        self._view_nbytes: dict[bytes, int] = {}

        if storage == "combined":
            for c in (combine_for or []):
                self._view(c)
            self.blocks = None       # drop the B-fold basis payloads

    def _compress_all(self) -> list:
        """Compress every partition block; returns the basis-form block
        list [(row_dofs, col_dofs, BasisLR | ndarray)] and records the
        rank-cap, fallback and certified-error statistics.

        Compression is parallel ACROSS blocks for LARGE blocks only
        (Python threads; their kernel work runs in serial nogil numba,
        so threads scale and cannot trip the workqueue threading
        layer). Small blocks are Python-overhead(GIL)-bound -- threads
        only add contention (measured 0.6x) -- so they compress inline.
        Each block draws from its OWN rng seeded by (12345, block
        index): factors are deterministic and independent of both
        scheduling order and the size routing.
        """
        part = self._part
        tol = self.tol
        n_workers = self._n_workers
        fallback_gate = _threading.Semaphore(
            defaults.ACA_FALLBACK_CONCURRENCY)
        min_side = defaults.ACA_PARALLEL_MIN_SIDE

        def _compress(i, rows, cols):
            cache = BlockEvalCache(self.eval.stack_serial, rows, cols)
            rng = np.random.default_rng((12345, i))
            return compress_block(cache, self.n_basis, tol=tol, rng=rng,
                                  fallback_gate=fallback_gate)

        results: dict = {}
        big = [(i, rows, cols)
               for i, (rows, cols) in enumerate(part.admissible)
               if min(len(rows), len(cols)) >= min_side]
        with threadpool_limits(limits=1):
            if n_workers > 1 and len(big) > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futs = {ex.submit(_compress, i, rows, cols): i
                            for i, rows, cols in big}
                    for f, i in futs.items():
                        results[i] = f.result()
            else:
                for i, rows, cols in big:
                    results[i] = _compress(i, rows, cols)
            for i, (rows, cols) in enumerate(part.admissible):
                if i not in results:
                    results[i] = _compress(i, rows, cols)

        blocks = []
        self.n_fallback = 0
        self.n_capped = 0
        for i, (rows, cols) in enumerate(part.admissible):
            res = results[i]
            self.n_fallback += 1 if res.fallback else 0
            self.n_capped += 1 if res.capped else 0
            self.max_verified_err = max(self.max_verified_err, res.max_err)
            blocks.append((_dof_idx(rows), _dof_idx(cols), res.payload))
        for rows, cols in part.dense:
            stack = self.eval.stack(rows, cols)      # prange kernels
            blocks.append((_dof_idx(rows), _dof_idx(cols), stack))
        return blocks

    # -- material views -----------------------------------------------

    def _combine(self, blocks: list, c: np.ndarray):
        """The per-material view of ``blocks`` for coefficients ``c``:
        ([(rdofs, cdofs, A, V)], max certified error, blocks applied
        exactly), A @ V.T for a recombined low-rank block, (A, None) for
        a dense one. Each recombined block is certified on its exact
        rows and columns (``aca.certify_combined``); above
        ``ACA_CERTIFY_FACTOR`` x tol the view holds the exact combined
        block instead.

        The recompressions run SERIALLY at one BLAS thread: that is the
        whole gain (1.1 ms per block against 8-10 ms under BLAS
        oversubscription), and a thread pool loses it again -- OpenBLAS
        serializes concurrent callers on its buffer lock (measured 9x
        slower than one thread). Only the certificate's kernel
        evaluations (serial nogil kernels) run on the pool, in chunks
        whose exact row/column stacks total at most the bytes of one
        exact MAX_ADMISSIBLE_BLOCK stack.
        """
        n_basis = self.n_basis
        tol = self.tol
        limit = defaults.ACA_CERTIFY_FACTOR * tol
        ev = self.eval
        chunk_max = 8 * n_basis * (3 * defaults.MAX_ADMISSIBLE_BLOCK) ** 2

        def _elems(rdofs, cdofs):
            return rdofs[0::3] // 3, cdofs[0::3] // 3

        def _cert(blk):
            rdofs, cdofs, payload = blk
            er, ec = _elems(rdofs, cdofs)
            return (ev.stack_serial(er[payload.cert_rows], ec),
                    ev.stack_serial(er, ec[payload.cert_cols]))

        def _cert_bytes(blk):
            rdofs, cdofs, payload = blk
            return 8 * n_basis * 3 * len(payload.cert_rows) * (len(rdofs)
                                                               + len(cdofs))

        view: list = [None] * len(blocks)
        err = 0.0
        n_exact = 0
        lr = [i for i, blk in enumerate(blocks) if isinstance(blk[2], BasisLR)]
        pool = (ThreadPoolExecutor(max_workers=self._n_workers)
                if self._n_workers > 1 and len(lr) > 1 else None)
        with threadpool_limits(limits=1):
            for i, (rdofs, cdofs, payload) in enumerate(blocks):
                if not isinstance(payload, BasisLR):
                    view[i] = (rdofs, cdofs,
                               np.tensordot(c, payload, axes=1), None)
            try:
                start = 0
                while start < len(lr):
                    stop, nbytes = start, 0
                    while stop < len(lr) and (
                            stop == start
                            or nbytes + _cert_bytes(blocks[lr[stop]])
                            <= chunk_max):
                        nbytes += _cert_bytes(blocks[lr[stop]])
                        stop += 1
                    idx = lr[start:stop]
                    chunk = [blocks[i] for i in idx]
                    stacks = (list(pool.map(_cert, chunk)) if pool
                              else [_cert(blk) for blk in chunk])
                    for i, (rows_exact, cols_exact) in zip(idx, stacks):
                        rdofs, cdofs, payload = blocks[i]
                        U_cat = np.hstack([c[b] * payload.U[b]
                                           for b in range(n_basis)])
                        V_cat = np.hstack(payload.V)
                        U, V = recompress(U_cat, V_cat, tol)
                        e = certify_combined(U, V, c, rows_exact, cols_exact,
                                             payload.cert_rows,
                                             payload.cert_cols)
                        if e <= limit:
                            view[i] = (rdofs, cdofs, U, V)
                            err = max(err, e)       # error as APPLIED
                        else:
                            er, ec = _elems(rdofs, cdofs)
                            view[i] = (rdofs, cdofs, np.tensordot(
                                c, ev.stack(er, ec), axes=1), None)
                            n_exact += 1
                    start = stop
            finally:
                if pool is not None:
                    pool.shutdown()
        return view, err, n_exact

    def _evict(self) -> None:
        """LRU bound on the view cache by count (``HOP_VIEW_CACHE_MAX``)
        and bytes (``HOP_VIEW_CACHE_MAX_BYTES``); the
        ``HOP_VIEW_CACHE_MIN_KEEP`` most recent views -- one solve's
        working set on this pair -- are never evicted by the bytes bound."""
        total = sum(self._view_nbytes.values())
        while (len(self._views) > defaults.HOP_VIEW_CACHE_MAX
               or (len(self._views) > defaults.HOP_VIEW_CACHE_MIN_KEEP
                   and total > defaults.HOP_VIEW_CACHE_MAX_BYTES)):
            key, _ = self._views.popitem(last=False)
            total -= self._view_nbytes.pop(key)

    def _view(self, coeffs: np.ndarray) -> list:
        c = np.asarray(coeffs, dtype=float)
        key = c.tobytes()
        view = self._views.get(key)
        if view is not None:
            self._views.move_to_end(key)
            return view
        blocks = self.blocks
        if blocks is None:
            # storage="combined" and a material we have no view for:
            # transiently re-compress this pair from geometry, combine,
            # then drop the basis payloads again.
            blocks = self._compress_all()
        view, err, n_exact = self._combine(blocks, c)
        self.max_verified_err = max(self.max_verified_err, err)
        self.n_view_fallback = max(self.n_view_fallback, n_exact)
        self._views[key] = view
        self._view_nbytes[key] = sum(
            A.nbytes + (V.nbytes if V is not None else 0)
            for _, _, A, V in view)
        self._evict()
        return view

    def warm_views(self, coeffs_list) -> None:
        """Build the views for every coefficient vector in one pass.

        In storage="combined" mode each UNSEEN material would otherwise
        trigger its own transient re-compression; warming batches them
        over a single compression.
        """
        missing = [np.asarray(c, dtype=float) for c in coeffs_list
                   if np.asarray(c, dtype=float).tobytes() not in self._views]
        if not missing:
            return
        if self.blocks is None:
            self.blocks = self._compress_all()
            for c in missing:
                self._view(c)
            self.blocks = None
        else:
            for c in missing:
                self._view(c)

    # -- operations -----------------------------------------------------

    def matvec(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.shape[0])
        for rdofs, cdofs, A, V in self._view(coeffs):
            if V is None:
                y[rdofs] += A @ x[cdofs]
            else:
                y[rdofs] += A @ (V.T @ x[cdofs])
        return y

    def to_dense(self, coeffs: np.ndarray) -> np.ndarray:
        M = np.zeros(self.shape)
        for rdofs, cdofs, A, V in self._view(coeffs):
            if V is None:
                M[np.ix_(rdofs, cdofs)] += A
            else:
                M[np.ix_(rdofs, cdofs)] += A @ V.T
        return M

    # -- stats ------------------------------------------------------

    def nbytes(self) -> int:
        if self.blocks is None:            # storage="combined"
            return sum(self._view_nbytes.values())
        total = 0
        for _, _, payload in self.blocks:
            total += payload.nbytes() if isinstance(payload, BasisLR) \
                else payload.nbytes
        return total

    def dense_equivalent_bytes(self) -> int:
        """Bytes of ONE dense material matrix (what a solver would hold)."""
        return self.shape[0] * self.shape[1] * 8

    def view_nbytes(self, coeffs: np.ndarray) -> int:
        """Bytes of the per-material working set (recombined view)."""
        self._view(coeffs)
        return self._view_nbytes[np.asarray(coeffs, dtype=float).tobytes()]

    def summary(self) -> str:
        combined = self.blocks is None
        head = f"PairCompressed {self.shape}" + (
            f" [combined, {len(self._views)} material view(s)]" if combined
            else f" x{self.n_basis} bases")
        adm = (f"{self.n_lowrank} admissible blocks ({self.n_capped} dense "
               f"at the rank cap, {self.n_fallback} fallback at build, "
               f"up to {self.n_view_fallback} exact per view; max certified "
               f"err {self.max_verified_err:.1e}")
        if not combined:
            ranks = [r for _, _, p in self.blocks if isinstance(p, BasisLR)
                     for r in p.ranks]
            adm += f", avg basis rank {np.mean(ranks) if ranks else 0:.1f}"
        adm += ")"
        dense_mb = self.dense_equivalent_bytes() / 1e6
        tail = (f"{self.n_dense} dense leaf, {self.nbytes()/1e6:.1f} MB vs "
                + (f"{dense_mb:.1f} MB dense" if combined
                   else f"{self.n_basis} x {dense_mb:.1f} MB dense-basis"))
        return f"{head}: {adm}, {tail}"
