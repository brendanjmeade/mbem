"""Compressed pair operators with material-basis storage.

A ``PairCompressed`` represents ALL basis matrices of one
(field mesh, source mesh, kernel) pair in block-compressed form:
admissible blocks as one shared subspace per block (``aca.SharedLR``:
``A_b = Qu cores[b] Qv^T``) or, when not low rank at the tolerance, as
their indices alone. A material enters only through its coefficient
vector: the per-material view sums the cores and truncates that k x k
sum (no factor of a block's own size is factorized per material),
certifies every recombined block on its exact rows and columns (a block
that fails is applied exactly in that view), evaluates every exact block
-- those and the partition's near-field leaves -- from the kernels in one
parallel pass, and lays the result out as a :class:`.flatview.FlatView`
whose single numba kernel applies the whole pair. Views are cached, so
the SAME object serves every region material. Coefficients are real (both
backends allocate real accumulators).

The NEAR FIELD is never stored per basis: it is part of the view, at 1x
instead of Bx, and a second material re-evaluates it from the kernels
(kernel work only) instead of recombining B stacks.

Threads: block compression and view recombination run with BLAS pinned
to one thread (``threadpoolctl``): the per-block work is many small
QR/SVD calls whose multi-threaded BLAS only oversubscribes the machine.
Every block's ACA runs on a Python thread pool as one nogil numba kernel
(``aca_numba``); its shared subspace and every material recombination
are numpy and run on the MAIN thread, the ACA's chunk at a time, because
concurrent numpy LAPACK only contends on OpenBLAS's buffer lock (0.6x of
one thread, measured). The leaf kernel and the flat matvec are prange
kernels called from the main thread only (rule 8).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from .. import defaults
from ..kernels import KERNEL_T, kernel_n_basis
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from .aca import BlockEvalCache, certify_combined, compress_block
from .aca_numba import KERNEL_FLAG, block_factors
from .cluster import build_cluster_tree, build_partition
from .flatview import FlatView, dense_leaves, storage_dtype


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
        """(B, 3*len(rows), 3*len(cols)) exact basis sub-stack (prange
        kernels, main thread). The operator path combines a block's
        coefficients inside the leaf kernel instead and never builds a
        basis stack; this is the gate's reference (``view_reference``)."""
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

    def aca_fn(self, rows: np.ndarray, cols: np.ndarray):
        """This block's ACA + certificate stage as ONE nogil numba kernel
        over its own geometry (``aca_numba.block_factors``) -- what
        ``aca.compress_block`` runs in place of its Python loop, and why
        every block, not only the large ones, belongs on the pool."""
        xf = np.ascontiguousarray(self.x_field[rows])
        tv = np.ascontiguousarray(self.tri_verts[cols])
        nm = np.ascontiguousarray(self.normals[cols])
        ee = np.ascontiguousarray(self.eps[cols])
        flag = KERNEL_FLAG[self.kernel]

        def run(tol, srows, scols, cert_rows, cert_cols, max_rank):
            return block_factors(xf, tv, nm, ee, flag, tol, srows, scols,
                                 cert_rows, cert_cols, max_rank)

        return run


class PairCompressed:
    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr,
                 tol: float = defaults.BLOCK_COMPRESSION_TOL,
                 min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                 eta: float = defaults.ADMISSIBILITY_ETA,
                 max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK,
                 min_aca: int = defaults.ACA_MIN_BLOCK,
                 tree_cache: dict | None = None,
                 arrays: kb.MeshArrays | None = None,
                 n_workers: int | None = None,
                 storage: str = "basis",
                 combine_for=None,
                 precision=None):
        """``storage="basis"`` (default) keeps the GEOMETRY-ONLY shared
        subspace of the admissible blocks: any material recombines
        without re-compressing, at the JOINT rank of the B bases rather
        than their sum (0.52x over a 31k-unknown model, so ~4x the
        low-rank factors of one material view instead of B = 3 for U,
        6 for T; the near field is per material either way, never per
        basis, and it dominates -- the whole pair set of that model is
        1.16x its combined-storage size, against 1.6x unfolded).
        ``storage="combined"`` builds the views of ``combine_for``
        and then DROPS the payloads -- 1x storage, the memory mode for
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
        # Storage precision of every view of this pair, from its own block
        # tolerance (la/flatview.storage_dtype); ``precision`` overrides it
        # for the gates that need float64 to compare bitwise against a
        # float64 reference. Arithmetic stays float64 either way.
        self.storage_dtype = (storage_dtype(tol) if precision is None
                              else np.dtype(precision).type)
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
                                     eta=eta, max_admissible=max_admissible,
                                     min_aca=min_aca)
        self.n_lowrank = len(self._part.admissible)     # admissible blocks
        self.n_dense = len(self._part.dense)
        self.n_fallback = 0        # certificate failures AFTER the retry
        self.n_retry = 0           # blocks whose first certificate failed
        self.n_capped = 0          # admissible blocks dense at the rank cap
        self.n_view_fallback = 0   # combined-certificate failures, worst view
        self.max_verified_err = 0.0

        self.blocks = self._compress_all()

        # Per-material views: LRU-bounded by count and bytes (``_evict``).
        # Unbounded caching leaked one full recombined operator PER
        # MATERIAL across a material sweep.
        self._views: OrderedDict[bytes, FlatView] = OrderedDict()
        self._view_nbytes: dict[bytes, int] = {}

        if storage == "combined":
            for c in (combine_for or []):
                self._view(c)
            self.blocks = None       # drop the geometry-only payloads

    def _compress_all(self) -> list:
        """Compress every ADMISSIBLE block; returns the basis-form block
        list [(rows, cols, SharedLR | None)] (element indices; ``None``
        = not low rank at this tolerance, applied exactly) and records
        the rank-cap, retry, fallback and certified-error statistics.

        The near field is not touched here and holds no per-basis stack:
        dense leaves, and the admissible blocks applied exactly, are
        evaluated per material into the flat view (``_dense_leaves``).

        EVERY block compresses on the Python thread pool: its ACA,
        certificate and per-basis recompression are one nogil numba
        kernel (``aca_numba``), so the threads scale (no GIL between
        pivots) and cannot trip numba's threading layer, and a small
        block is no longer Python-overhead-bound. BLAS stays at one
        thread inside the pool. Each block draws from its OWN rng seeded
        by (12345, block index): factors are deterministic and
        independent of scheduling order.
        """
        part = self._part
        tol = self.tol
        n_workers = self._n_workers

        def _compress(i, rows, cols):
            cache = BlockEvalCache(self.eval.stack_serial, rows, cols,
                                   aca_fn=self.eval.aca_fn(rows, cols))
            rng = np.random.default_rng((12345, i))
            return compress_block(cache, self.n_basis, tol=tol, rng=rng,
                                  exact_payload=False, fold=False)

        blocks = []
        self.n_fallback = 0
        self.n_retry = 0
        self.n_capped = 0
        # One CHUNK of blocks compresses on the pool, then that chunk's
        # factors are folded into their shared subspaces
        # (``aca.PendingLR.fold``) -- ON THE POOL as well, now that the fold
        # is one nogil kernel (``la/fold_numba``). It used to be main-thread
        # because a pool of the NUMPY fold measured 0.6x of one thread,
        # recorded here as OpenBLAS's buffer lock; that was the GIL, held by
        # the wrapper around the (K, K) Gram matrices rather than by the QR.
        # Chunking rather than one pass at the end is what keeps the
        # UNFOLDED per-basis factors -- the B-fold form the fold exists to
        # shrink -- down to one chunk's worth of the pair at any moment.
        chunk = max(4 * n_workers, 1)
        with threadpool_limits(limits=1):
            for start in range(0, len(part.admissible), chunk):
                items = list(enumerate(part.admissible))[start:start + chunk]
                if n_workers > 1 and len(items) > 1:
                    with ThreadPoolExecutor(max_workers=n_workers) as ex:
                        futs = [ex.submit(_compress, i, rows, cols)
                                for i, (rows, cols) in items]
                        res_list = [f.result() for f in futs]
                        folds = list(ex.map(
                            lambda r: (r.payload.fold(tol)
                                       if r.payload is not None else None),
                            res_list))
                else:
                    res_list = [_compress(i, rows, cols) for i, (rows, cols)
                                in items]
                    folds = [r.payload.fold(tol) if r.payload is not None
                             else None for r in res_list]
                for ((_i, (rows, cols)), res, folded) in zip(items, res_list,
                                                             folds):
                    self.n_fallback += 1 if res.fallback else 0
                    self.n_retry += 1 if res.retried else 0
                    self.n_capped += 1 if res.capped else 0
                    self.max_verified_err = max(self.max_verified_err,
                                                res.max_err)
                    blocks.append((rows, cols, folded))
        return blocks

    # -- material views -----------------------------------------------

    def _recombine(self, blocks: list, c: np.ndarray):
        """Material recombination of the ADMISSIBLE blocks for
        coefficients ``c``: ([(rows, cols, U, V)] low rank,
        [(rows, cols)] applied exactly, max certified error, certificate
        failures). Each recombined block is certified on its exact rows
        and columns (``aca.certify_combined``); above
        ``ACA_CERTIFY_FACTOR`` x tol it joins the exact list, where the
        blocks the build already found not low rank (payload ``None``)
        start.

        A block costs a k x k core sum and a k x k SVD here
        (``aca.SharedLR.combine``) -- the shared subspace the build
        folded its bases into is what the material enters through, and
        nothing of the block's own size is factorized again. The loop is
        SERIAL at one BLAS thread: what is left of it is small dense
        algebra, a thread pool loses it to OpenBLAS's buffer lock
        (measured 9x slower than one thread), and a nogil numba pool
        leaves numba's prange kernels segfaulting on the next call. Only
        the certificate's kernel evaluations (serial nogil kernels) run
        on the pool, in chunks whose exact row/column stacks total at
        most the bytes of one exact MAX_ADMISSIBLE_BLOCK stack.
        """
        n_basis = self.n_basis
        tol = self.tol
        limit = defaults.ACA_CERTIFY_FACTOR * tol
        ev = self.eval
        chunk_max = 8 * n_basis * (3 * defaults.MAX_ADMISSIBLE_BLOCK) ** 2

        def _cert(blk):
            rows, cols, payload = blk
            return (ev.stack_serial(rows[payload.cert_rows], cols),
                    ev.stack_serial(rows, cols[payload.cert_cols]))

        def _cert_bytes(blk):
            rows, cols, payload = blk
            return 8 * n_basis * 9 * len(payload.cert_rows) * (len(rows)
                                                               + len(cols))

        low: dict = {}
        exact = [i for i, blk in enumerate(blocks) if blk[2] is None]
        err = 0.0
        n_exact = 0
        lr = [i for i, blk in enumerate(blocks) if blk[2] is not None]
        pool = (ThreadPoolExecutor(max_workers=self._n_workers)
                if self._n_workers > 1 and len(lr) > 1 else None)
        with threadpool_limits(limits=1):
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
                        rows, cols, payload = blocks[i]
                        U, V = payload.combine(c, tol)
                        # Certify the factors AS STORED: round through the
                        # storage precision and back, so the certificate
                        # sees the rounding the view will apply while the
                        # product it forms still accumulates in float64,
                        # exactly as the matvec does. Certifying before the
                        # rounding would bound a block never applied.
                        if self.storage_dtype != np.float64:
                            U = U.astype(self.storage_dtype).astype(np.float64)
                            V = V.astype(self.storage_dtype).astype(np.float64)
                        e = certify_combined(U, V, c, rows_exact, cols_exact,
                                             payload.cert_rows,
                                             payload.cert_cols)
                        if e <= limit:
                            low[i] = (rows, cols, U, V)
                            err = max(err, e)       # error as APPLIED
                        else:
                            exact.append(i)
                            n_exact += 1
                    start = stop
            finally:
                if pool is not None:
                    pool.shutdown()
        # Partition order on both lists: the view's block order, and with
        # it the accumulation order of its matvec, is a function of the
        # partition alone -- not of the chunking or the thread count.
        lr_blocks = [low[i] for i in sorted(low)]
        exact_blocks = [(blocks[i][0], blocks[i][1]) for i in sorted(exact)]
        return lr_blocks, exact_blocks, err, n_exact

    def _dense_leaves(self, blocks: list, c: np.ndarray):
        """``(D_flat, d_ptr)``: every block of ``blocks`` (element index
        arrays) evaluated from the kernels for coefficients ``c`` in one
        parallel pass (``flatview.dense_leaves``, main thread).

        This is what the near field costs per material -- its kernel work
        and nothing else (measured 0.16 s for the 1,148 leaves of a
        10k-element T self-pair, ~12 s at 1M unknowns) -- instead of B
        stored basis stacks recombined per material."""
        ev = self.eval
        return dense_leaves(ev.kernel, ev.x_field, ev.tri_verts, ev.normals,
                            ev.eps, blocks, c, dtype=self.storage_dtype)

    def _combine(self, blocks: list, c: np.ndarray):
        """The per-material ``FlatView`` of ``blocks`` for coefficients
        ``c``, with the max certified error and the number of admissible
        blocks applied exactly."""
        lr_blocks, exact_blocks, err, n_exact = self._recombine(blocks, c)
        dense_blocks = exact_blocks + list(self._part.dense)
        D_flat, d_ptr = self._dense_leaves(dense_blocks, c)
        return (FlatView(self.shape, lr_blocks, dense_blocks, D_flat, d_ptr,
                         dtype=self.storage_dtype),
                err, n_exact)

    def view_reference(self, coeffs: np.ndarray) -> list:
        """The pre-flat form of a material view, block by block:
        ``[(rdofs, cdofs, A, V)]`` -- ``A @ V.T`` low rank, ``(A, None)``
        exact -- in the flat view's own block order, with every exact
        block combined from its per-basis stack in numpy instead of by
        the batched leaf kernel.

        The reference the flat view is gated against
        (``verify_hbackend``); it needs the per-basis payloads, so it is
        for storage="basis" pairs, and no solve path calls it."""
        c = np.asarray(coeffs, dtype=float)
        blocks = self.blocks if self.blocks is not None else self._compress_all()
        lr_blocks, exact_blocks, _, _ = self._recombine(blocks, c)
        view = [(_dof_idx(rows), _dof_idx(cols), U, V)
                for rows, cols, U, V in lr_blocks]
        for rows, cols in exact_blocks + list(self._part.dense):
            view.append((_dof_idx(rows), _dof_idx(cols),
                         np.tensordot(c, self.eval.stack(rows, cols), axes=1),
                         None))
        return view

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

    def _view(self, coeffs: np.ndarray) -> FlatView:
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
        self._view_nbytes[key] = view.nbytes()
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
        """The material view applied to ``x`` by ONE kernel over every
        block (``FlatView.matvec``). Main thread only (rule 8)."""
        return self._view(coeffs).matvec(x)

    def to_dense(self, coeffs: np.ndarray) -> np.ndarray:
        return self._view(coeffs).to_dense()

    # -- stats ------------------------------------------------------

    def nbytes(self) -> int:
        """Resident bytes: every cached material view (its recombined
        factors AND its near field -- the near field is per material,
        never per basis) plus, in storage="basis", the shared
        subspaces."""
        total = sum(self._view_nbytes.values())
        if self.blocks is not None:
            total += sum(p.nbytes() for _, _, p in self.blocks
                         if p is not None)
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
               f"at the rank cap, {self.n_retry} retried and "
               f"{self.n_fallback} exact at build, up to "
               f"{self.n_view_fallback} exact per view; max certified "
               f"err {self.max_verified_err:.1e}")
        if not combined:
            held = [p for _, _, p in self.blocks if p is not None]
            ranks = [p.rank for p in held]
            summed = [sum(p.basis_ranks) for p in held]
            adm += (f", avg shared rank {np.mean(ranks) if ranks else 0:.1f} "
                    f"of {np.mean(summed) if summed else 0:.1f} summed")
        adm += ")"
        dense_mb = self.dense_equivalent_bytes() / 1e6
        tail = (f"{self.n_dense} dense leaf (per view), "
                f"{self.nbytes()/1e6:.1f} MB held vs "
                + (f"{dense_mb:.1f} MB dense" if combined
                   else f"{self.n_basis} x {dense_mb:.1f} MB dense-basis"))
        return f"{head}: {adm}, {tail}"
