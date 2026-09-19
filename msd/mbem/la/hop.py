"""Compressed pair operators with material-basis storage.

A ``PairCompressed`` represents ALL basis matrices of one
(field mesh, source mesh, kernel) pair in block-compressed form:
admissible blocks as per-basis low-rank factors (see :mod:`.aca`),
near-field leaf blocks as exact dense basis stacks. A material enters
only through its coefficient vector: the per-material view combines the
factors (with one QR+SVD re-truncation per low-rank block) and is
cached, so the SAME object serves every region material. Coefficients
are real (both backends allocate real accumulators).
"""

from __future__ import annotations

import threading as _threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import defaults
from ..kernels import KERNEL_T, kernel_n_basis
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from .aca import BasisLR, BlockEvalCache, compress_block, recompress
from .cluster import build_cluster_tree, build_partition


def _dof_idx(elems: np.ndarray) -> np.ndarray:
    return (3 * elems[:, None] + np.arange(3)[None, :]).ravel()


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
        storage (B = 3 for U, 6 for T). ``storage="combined"`` combines
        the factors for the coefficient vectors in ``combine_for`` and
        then DROPS the per-basis payloads -- 1x storage, the memory mode
        for very large models; a material NOT in ``combine_for`` (e.g. a
        later rebuild_for_materials) triggers a transient re-compression
        of this pair. Prefer "basis" for material sweeps when it fits.
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
        self._n_workers = n_workers

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
        self.n_lowrank = len(self._part.admissible)
        self.n_dense = len(self._part.dense)
        self.n_fallback = 0

        self.blocks = self._compress_all()

        # Per-material views: LRU-bounded. Unbounded caching leaked one
        # full recombined operator PER MATERIAL across a material sweep;
        # the LRU keeps the working set of a solve (every distinct
        # coefficient vector the terms + preconditioner touch) while old
        # sweep materials age out.
        self._views: OrderedDict[bytes, list] = OrderedDict()

        if storage == "combined":
            for c in (combine_for or []):
                self._view(np.asarray(c))
            self.blocks = None       # drop the B-fold basis payloads

    def _compress_all(self) -> list:
        """Compress every partition block; returns the basis-form block
        list [(row_dofs, col_dofs, BasisLR | ndarray)].

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
        if n_workers is None:
            import os
            n_workers = min(16, os.cpu_count() or 1)
        fallback_gate = _threading.Semaphore(
            defaults.ACA_FALLBACK_CONCURRENCY)
        min_side = defaults.ACA_PARALLEL_MIN_SIDE

        def _compress(i, rows, cols):
            cache = BlockEvalCache(self.eval.stack_serial, rows, cols)
            rng = np.random.default_rng((12345, i))
            return compress_block(cache, self.n_basis, tol=tol, rng=rng,
                                  fallback_gate=fallback_gate)

        lrs: dict[int, BasisLR] = {}
        big = [(i, rows, cols)
               for i, (rows, cols) in enumerate(part.admissible)
               if min(len(rows), len(cols)) >= min_side]
        if n_workers > 1 and len(big) > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_compress, i, rows, cols): i
                        for i, rows, cols in big}
                for f, i in futs.items():
                    lrs[i] = f.result()
        else:
            for i, rows, cols in big:
                lrs[i] = _compress(i, rows, cols)
        for i, (rows, cols) in enumerate(part.admissible):
            if i not in lrs:
                lrs[i] = _compress(i, rows, cols)

        blocks = []
        self.n_fallback = 0
        for i, (rows, cols) in enumerate(part.admissible):
            lr = lrs[i]
            self.n_fallback += 1 if lr.fallback else 0
            blocks.append((_dof_idx(rows), _dof_idx(cols), lr))
        for rows, cols in part.dense:
            stack = self.eval.stack(rows, cols)      # prange kernels
            blocks.append((_dof_idx(rows), _dof_idx(cols), stack))
        return blocks

    # -- material views -----------------------------------------------

    def _view(self, coeffs: np.ndarray) -> list:
        c = np.asarray(coeffs)
        key = c.tobytes()
        view = self._views.get(key)
        if view is not None:
            self._views.move_to_end(key)
            return view
        blocks = self.blocks
        transient = blocks is None
        if transient:
            # storage="combined" and a material we have no view for:
            # transiently re-compress this pair from geometry, combine,
            # then drop the basis payloads again.
            blocks = self._compress_all()
        view = []
        for rdofs, cdofs, payload in blocks:
            if isinstance(payload, BasisLR):
                U_cat = np.hstack([c[b] * payload.U[b]
                                   for b in range(self.n_basis)])
                V_cat = np.hstack(payload.V)
                U, V = recompress(U_cat, V_cat, self.tol)
                view.append((rdofs, cdofs, U, V))
            else:
                M_eff = np.tensordot(c, payload, axes=1)
                view.append((rdofs, cdofs, M_eff, None))
        self._views[key] = view
        while len(self._views) > defaults.HOP_VIEW_CACHE_MAX:
            self._views.popitem(last=False)
        return view

    def warm_views(self, coeffs_list) -> None:
        """Build the views for every coefficient vector in one pass.

        In storage="combined" mode each UNSEEN material would otherwise
        trigger its own transient re-compression; warming batches them
        over a single compression.
        """
        missing = [np.asarray(c) for c in coeffs_list
                   if np.asarray(c).tobytes() not in self._views]
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
            total = 0
            for view in self._views.values():
                for _, _, A, V in view:
                    total += A.nbytes + (V.nbytes if V is not None else 0)
            return total
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
        total = 0
        for _, _, A, V in self._view(coeffs):
            total += A.nbytes + (V.nbytes if V is not None else 0)
        return total

    def summary(self) -> str:
        if self.blocks is None:            # storage="combined"
            return (f"PairCompressed {self.shape} [combined, "
                    f"{len(self._views)} material view(s)]: "
                    f"{self.n_lowrank} low-rank blocks "
                    f"({self.n_fallback} w/ fallback), "
                    f"{self.n_dense} dense leaf, "
                    f"{self.nbytes()/1e6:.1f} MB vs "
                    f"{self.dense_equivalent_bytes()/1e6:.1f} MB dense")
        ranks = [r for _, _, p in self.blocks if isinstance(p, BasisLR)
                 for r in p.ranks]
        return (f"PairCompressed {self.shape} x{self.n_basis} bases: "
                f"{self.n_lowrank} low-rank blocks "
                f"({self.n_fallback} w/ fallback, "
                f"avg basis rank {np.mean(ranks) if ranks else 0:.1f}), "
                f"{self.n_dense} dense leaf, "
                f"{self.nbytes()/1e6:.1f} MB vs {self.n_basis} x "
                f"{self.dense_equivalent_bytes()/1e6:.1f} MB dense-basis")
