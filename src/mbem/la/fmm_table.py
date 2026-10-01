"""The shared M2L table: one block per transfer OFFSET, level folded out.

WHAT IS SHARED, AND WHY IT IS O(1). On a translation-invariant lattice
(``domain="cube"`` or ``"canonical"``) two same-level boxes at integer
offset ``d`` have M2L nodes ``c_a + H u_t`` and ``c_b + H u_s`` with ``H``
the half-width and ``u`` the one fixed Chebyshev lattice, so

    x_t - y_s = H (u_t - u_s - 2 d)

depends on the pair only through ``d`` and ``H``. And ``H`` folds out: each
far pass is HOMOGENEOUS, so the block at half-width ``H`` is the block at
``H = 1`` times ``H**-degree``, exactly, a power of two between levels.
What is left to store is one block per distinct offset, and V entries are
same-level with adjacent parents, so the offset lies in ``[-3, 3]^3`` minus
``[-1, 1]^3``: **316 blocks, whatever the mesh and whatever N**
(``verify_octree`` gates that ceiling; ``Octree.transfer_offset`` is the key).

THE DEGREE IS NOT THE RADIAL POWER FOR T, and getting it wrong is a silent
factor of two per level of reuse. Every T term carries one extra factor of
``d`` in the numerator -- ``a1 wk d_i tr(q)``, ``a2 wk (q + q^T)_im d_m``,
``a3 wk2 d_i d_j q_jm d_m`` are all ``r**-(power-1)`` -- while both U terms
are ``r**-power``. ``verify_fmm`` pins the fold with ``np.array_equal``, not
a tolerance, because a tolerance would accept the factor of two.

WHY A TABLE AT ALL, GIVEN ``fmm_numba``. Not for the arithmetic: 28.3 Mflop
per V pair against 34.1 matrix-free, 1.2x. For the RATE, by turning the sum
into a GEMM -- but only when the GEMM is wide. Measured per V pair at p = 8,
the table costs 0.499 / 0.182 / 0.064 / 0.061 ms at m = 1 / 8 / 114 / 512
against the matrix-free kernel's 0.399, so it LOSES below m ~ 8. That is why
this is keyed on the offset and applied key-major over one traversal of the
whole operator (``FarGroups``): per pair the batch is 1.60 and the table
would be a pessimisation, while over a group it is a V-weighted 404 / 1,717
/ 2,292 at the three measured scales.

MEMORY is the reason for the cap. A T key at p = 8 is
``2 (3p^3)(9p^3) x 8 B`` = 108 MiB, so the full 316 is 33.3 GiB at float64 --
affordable at the 4M target, not while a gate runs. Blocks are therefore
built on demand under ``FMM_M2L_TABLE_MAX_BYTES`` and anything that does not
fit falls back to the matrix-free kernel, which is exact and merely slower.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from numba import njit, prange

from .. import defaults
from ..kernels import KERNEL_U


def pass_degree(kernel: str, power: int) -> int:
    """Homogeneity degree of one far pass: ``K(s g) = s**-degree K(g)``.

    ``power`` is the pass's radial power. U's two terms are both
    ``r**-power``; every T term carries one more factor of ``d`` upstairs,
    so T is ``r**-(power-1)``. Using the power for T rescales a reused block
    by exactly 2x too much per level.
    """
    return int(power) if kernel == KERNEL_U else int(power) - 1


@njit(cache=True, parallel=True)
def _u_block(u, off, power, b1, b2, out):
    """U block at unit half-width, ``out[(t,i), (s,c)]``."""
    p3 = u.shape[0]
    for t in prange(p3):
        for s in range(p3):
            g0 = u[t, 0] - u[s, 0] - 2.0 * off[0]
            g1 = u[t, 1] - u[s, 1] - 2.0 * off[1]
            g2 = u[t, 2] - u[s, 2] - 2.0 * off[2]
            inv = 1.0 / np.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
            inv2 = inv * inv
            wk = inv
            for _ in range((power - 1) // 2):
                wk = wk * inv2
            w1 = b1 * wk
            w2 = b2 * wk * inv2
            g = (g0, g1, g2)
            for i in range(3):
                r = 3 * t + i
                for c in range(3):
                    v = w2 * g[i] * g[c]
                    if i == c:
                        v += w1
                    out[r, 3 * s + c] = v


@njit(cache=True, parallel=True)
def _t_block(u, off, power, a1, a2, a3, out):
    """T block at unit half-width, ``out[(t,i), (s, 3m+n)]``."""
    p3 = u.shape[0]
    for t in prange(p3):
        for s in range(p3):
            g0 = u[t, 0] - u[s, 0] - 2.0 * off[0]
            g1 = u[t, 1] - u[s, 1] - 2.0 * off[1]
            g2 = u[t, 2] - u[s, 2] - 2.0 * off[2]
            inv = 1.0 / np.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
            inv2 = inv * inv
            wk = inv
            for _ in range((power - 1) // 2):
                wk = wk * inv2
            w1 = a1 * wk
            w2 = a2 * wk
            w3 = a3 * wk * inv2
            g = (g0, g1, g2)
            for i in range(3):
                r = 3 * t + i
                for m in range(3):
                    for n in range(3):
                        # a1 d_i tr(q) + a2 (q + q^T)_im d_m + a3 d_i d_m d_n
                        v = w3 * g[i] * g[m] * g[n]
                        if m == n:
                            v += w1 * g[i]
                        if i == m:
                            v += w2 * g[n]
                        if i == n:
                            v += w2 * g[m]
                        out[r, 9 * s + 3 * m + n] = v


class M2LTable:
    """Blocks for one (kernel, coefficient vector, order), keyed by offset.

    ``block(offset)`` returns the list of per-pass blocks at UNIT half-width,
    or None when the cap is reached -- the caller then evaluates that entry
    matrix-free, which is exact.
    """

    def __init__(self, kernel: str, p: int, params: list, nodes: np.ndarray,
                 max_bytes: int | None = None, rank_tol: float | None = None):
        self.kernel = kernel
        self.p = int(p)
        self.params = params
        self.nc = 3 if kernel == KERNEL_U else 9
        # The one lattice every box's M2L nodes are a translate and scaling
        # of, in units of the half-width: nodes of the box at the origin.
        self.u = np.ascontiguousarray(nodes, dtype=float)
        self.degree = [pass_degree(kernel, prm[0]) for prm in params]
        self.max_bytes = int(defaults.FMM_M2L_TABLE_MAX_BYTES
                             if max_bytes is None else max_bytes)
        self.rank_tol = float(defaults.FMM_M2L_RANK_TOL
                              if rank_tol is None else rank_tol)
        self.sketch = int(defaults.FMM_M2L_SKETCH)
        self._blocks: OrderedDict = OrderedDict()
        self._bytes = 0
        self.ranks: list = []
        self.hits = self.misses = 0

    def _factor(self, B):
        """``(A, Bt)`` with ``A @ Bt ~ B`` to ``rank_tol``, or ``(B, None)``
        when truncation is off.

        Through the Gram rather than a full SVD: the rank is 2-5 % of the
        block, so ``eigh(B B^T)`` at (3p^3, 3p^3) is several times cheaper
        than an SVD of (3p^3, nc p^3) and this is paid once per offset.
        """
        if self.rank_tol <= 0.0:
            return B, None
        nr, nc = B.shape
        # A RANDOMIZED range finder, not a Gram eigendecomposition. The rank
        # is a few per cent, so an exact (3p^3, 3p^3) `eigh` spends ~9 n^3 to
        # find ~100 directions and is the whole build: measured, it made the
        # table take longer to FACTOR than the operator took to run. The
        # sketch below is all GEMM and ~6x fewer flops at much better rate.
        # One subspace iteration, because the spectrum decays fast but not
        # instantly. The seed is FIXED so a table is reproducible: the gates
        # compare a compressed apply against a dense one entrywise.
        sk = min(nr, self.sketch)
        rs = np.random.default_rng(0)
        Q, _ = np.linalg.qr(B @ rs.standard_normal((nc, sk)))
        Q, _ = np.linalg.qr(B @ (B.T @ Q))
        Bp = Q.T @ B                                     # (sk, nc)
        ev, W = np.linalg.eigh(Bp @ Bp.T)                # (sk, sk), cheap
        ev = np.clip(ev[::-1], 0.0, None)
        W = W[:, ::-1]
        s = np.sqrt(ev)
        r = max(int((s > self.rank_tol * max(s[0], 1e-300)).sum()), 1)
        if r >= sk or r * (nr + nc) >= nr * nc:
            return B, None      # sketch too narrow, or no saving: stay dense
        A = Q @ W[:, :r]
        return np.ascontiguousarray(A), np.ascontiguousarray(A.T @ B)

    def _build(self, off) -> list:
        p3 = self.p ** 3
        out = []
        for prm in self.params:
            B = np.empty((3 * p3, self.nc * p3))
            o = np.asarray(off, dtype=float)
            if self.kernel == KERNEL_U:
                _u_block(self.u, o, int(prm[0]), prm[1], prm[2], B)
            else:
                _t_block(self.u, o, int(prm[0]), prm[1], prm[2], prm[3], B)
            out.append(self._factor(B))
        return out

    def block(self, off) -> list | None:
        got = self._blocks.get(off)
        if got is not None:
            self.hits += 1
            self._blocks.move_to_end(off)
            return got
        if self._bytes >= self.max_bytes:
            self.misses += 1
            return None
        got = self._build(off)              # size is known only once factored
        need = sum(a.nbytes + (0 if b is None else b.nbytes) for a, b in got)
        if self._bytes + need > self.max_bytes and self._blocks:
            self.misses += 1
            return None
        self._blocks[off] = got
        self._bytes += need
        self.ranks.append(got[0][0].shape[1] if got[0][1] is not None
                          else min(got[0][0].shape))
        self.misses += 1
        return got

    def summary(self) -> str:
        r = np.array(self.ranks) if self.ranks else np.zeros(1)
        full = min(3 * self.p ** 3, self.nc * self.p ** 3)
        return (f"M2LTable[{self.kernel}] p={self.p}: {len(self._blocks)} "
                f"offsets, {self._bytes / 2**20:.0f} MiB, rank "
                f"{r.min():.0f}-{r.max():.0f} (med {np.median(r):.0f}) of "
                f"{full}, {self.hits} hits / {self.misses} builds")


# ---------------------------------------------------------------------
# the table is shared across TREES, and bounded
# ---------------------------------------------------------------------

_CACHE: OrderedDict = OrderedDict()


def shared_table(kernel: str, p: int, params: list, nodes: np.ndarray):
    """One ``M2LTable`` per (kernel, order, material, lattice), process-wide.

    WHY ACROSS TREES. A block is the kernel on the unit lattice at one
    transfer offset, so it depends on the model through nothing at all --
    the lattice is the canonical Chebyshev grid (measured bitwise equal
    between two trees over different surfaces of one geometry) and the
    half-width folds out as a power of two. Held per tree it was rebuilt per
    tree: the four-state topography example paid the whole construction
    twice, ~177 s at 31k unknowns, for two trees that need identical blocks.

    WHY BOUNDED, which is the part that is not an optimisation. The key
    carries the MATERIAL, so a material sweep mints a new table per novel
    material and nothing evicted the superseded ones: measured 4 -> 5 -> 6
    -> 7 tables and 3.34 -> 3.99 GB over three rebuilds at 31k, and at the
    4M target a T table reaches the ``FMM_M2L_TABLE_MAX_BYTES`` cap, so a
    22-material sweep would add tens of GB of dead blocks.
    ``FMM_M2L_TABLE_MAX_BYTES`` bounds one table and never bounded their
    number; this does, by LRU over whole tables.

    The LATTICE IS IN THE KEY rather than checked after the fact: a tree
    whose lattice differed would otherwise either be handed the wrong blocks
    or silently evict the right ones. A caller with no shared lattice at all
    (``domain="extent"``) must not reach here -- ``PairFMM._unit_lattice``
    returns None and the caller stays matrix-free.
    """
    u = np.ascontiguousarray(nodes, dtype=float)
    key = (kernel, int(p),
           tuple(tuple(float(v) for v in prm) for prm in params),
           u.shape, u.tobytes())
    table = _CACHE.get(key)
    if table is None:
        table = M2LTable(kernel, p, params, u)
        _CACHE[key] = table
    _CACHE.move_to_end(key)
    while len(_CACHE) > defaults.FMM_M2L_CACHE_MAX_TABLES:
        _CACHE.popitem(last=False)
    return table


def cache_stats() -> dict:
    """``{tables, bytes, blocks}`` resident in the shared cache."""
    return {"tables": len(_CACHE),
            "bytes": sum(t._bytes for t in _CACHE.values()),
            "blocks": sum(len(t._blocks) for t in _CACHE.values())}


def clear_cache() -> None:
    """Drop every shared table. For gates that measure a build, and for a
    caller that is done with one geometry and wants the bytes back."""
    _CACHE.clear()
