"""Adaptive octree and FMM interaction lists over ALL elements at once.

The cluster tree of ``la/cluster.py`` is per (field mesh, source mesh) pair,
binary, and split on the principal axis. That is right for an H-matrix, whose
low-rank blocks are fitted per block, and wrong for a fast multipole method,
whose whole economy is that ONE M2L table is shared by every box pair with the
same relative offset at a level. A shared table needs one tree, cubic boxes and
a level structure, which is what this module builds.

Measured on ``topo_inclusion``, which is why the details below are what they
are:

* Globality alone buys almost nothing on the near field (1.04-1.22x against
  per-patch trees at fixed leaf and admissibility); the CUBIC BOX SHAPE buys
  2.2-2.7x. The reason to be global is the shared M2L table, not the bytes.
* Element size grades 79:1 at scale 1 and 214:1 at scale 3, driven by the FINE
  end, so no single level holds every element and the tree must be adaptive.
* A box is therefore often occupied AND internal: an element too large to
  descend is PINNED where it is and interacts directly with its own box's whole
  subtree. That mixed-level adjacency is what makes the W and X lists nonempty,
  and it survives even on an ungraded mesh (W+X is 0.83x U with the size
  constraint removed) because the fault and inclusion are refined against a
  coarse host.

Placement and the enlarged box. An element goes to the finest level whose box
edge is at least its own size, and within that level to the box holding its
centroid. That leaves it PROTRUDING from its box by up to 0.53 box edges, and
Chebyshev interpolation is invalid for a source outside its box at all. Demanding
strict containment instead pins 52 % of elements and costs 284 KiB/unknown, and
demanding a 2x margin makes the placement rule rather than ``ncrit`` set the
near-field floor. So each box carries, besides its nominal cube, the bounding
box of everything it actually holds (``Box.lo``/``hi`` against
``Box.cube_lo``/``cube_hi``): the M2L table stays on the nominal cubes, which
keeps it translation-invariant, and the enlargement is absorbed into P2M/L2P.
Lists are formed on the nominal cubes and then repaired, because a protruding
element can destroy the separation a list entry assumed.

Lists follow the standard adaptive-FMM definitions, with adjacency meaning that
two closed cubes touch (face, edge, corner or containment) even at different
levels:

    U(B)  leaves adjacent to leaf B, B itself included -- direct evaluation
    V(B)  children of B's parent's neighbours not adjacent to B -- M2L
    W(B)  descendants of B's neighbours, not adjacent to B, whose parent is
          adjacent to B -- M2P
    X(B)  the transpose of W -- P2L

W and X are not optional here: they are 0.75-1.2x the U list, and evaluating
them directly instead costs 1.8-2.2x on near-field bytes and more on flops,
a W record carrying 147 element pairs against a U record's 55.6.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import defaults

# Integer box coordinates are packed three-to-an-int. 21 bits each caps the
# tree at level 21, far past the ~10 levels a 1e6-element surface reaches.
_SHIFT = 21
_MASK = (1 << _SHIFT) - 1


def _key(i, j, k) -> int:
    return (int(i) << (2 * _SHIFT)) | (int(j) << _SHIFT) | int(k)


def _unkey(q: int) -> tuple:
    return (q >> (2 * _SHIFT), (q >> _SHIFT) & _MASK, q & _MASK)


def cubes_adjacent(la: int, qa: int, lb: int, qb: int) -> bool:
    """Do the closed cubes touch -- face, edge, corner or containment?

    Levels may differ: the coarser box is expanded to the finer level's
    integer grid and the test is then per axis.
    """
    if la > lb:
        la, qa, lb, qb = lb, qb, la, qa
    d = lb - la
    ai, aj, ak = _unkey(qa)
    bi, bj, bk = _unkey(qb)
    for a, b in ((ai, bi), (aj, bj), (ak, bk)):
        lo = a << d
        hi = lo + (1 << d) - 1
        if lo > b + 1 or b > hi + 1:
            return False
    return True


class Octree:
    """Adaptive octree over element centroids with size-constrained placement.

    ``sizes`` is each element's own extent (its longest edge for a triangle).
    An element may live no deeper than the level whose cube edge is at least
    ``placement_safety`` times that, so it is never much larger than the box
    holding it. A box subdivides while it holds more than ``ncrit`` elements
    AND at least one of them may descend -- on the TOTAL count, not on the
    descendable count, or a box of 100 elements of which 20 can descend would
    stay a leaf and hand 80 of them to the near field.
    """

    def __init__(self, centroids: np.ndarray, sizes: np.ndarray,
                 verts: np.ndarray | None = None,
                 ncrit: int = defaults.OCTREE_NCRIT,
                 placement_safety: float = defaults.OCTREE_PLACEMENT_SAFETY,
                 level_cap: int = defaults.OCTREE_LEVEL_CAP):
        centroids = np.ascontiguousarray(centroids, dtype=float)
        sizes = np.ascontiguousarray(sizes, dtype=float)
        n = centroids.shape[0]
        if sizes.shape[0] != n:
            raise ValueError("sizes must be one per element")
        self.n = n
        self.ncrit = int(ncrit)
        self.level_cap = int(level_cap)

        pts = centroids if verts is None else np.asarray(verts).reshape(-1, 3)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        edge = float((hi - lo).max())
        # A degenerate (single point, or planar) cloud still needs a cube.
        edge = edge if edge > 0.0 else 1.0
        self.root_edge = edge * (1.0 + 1e-9)
        self.origin = 0.5 * (lo + hi) - 0.5 * self.root_edge

        u = (centroids - self.origin) / self.root_edge          # in [0, 1)
        cap = 1 << self.level_cap
        self._fine = np.clip((u * cap).astype(np.int64), 0, cap - 1)

        with np.errstate(divide="ignore", invalid="ignore"):
            deepest = np.floor(np.log2(
                self.root_edge / (placement_safety
                                  * np.maximum(sizes, 1e-300))))
        self.max_level_of = np.clip(deepest, 0, self.level_cap).astype(np.int64)

        self._build()

    # -- construction --------------------------------------------------

    def _build(self) -> None:
        cap = self.level_cap
        reach = [dict() for _ in range(cap + 2)]
        reach[0][_key(0, 0, 0)] = np.arange(self.n, dtype=np.int64)
        self.resident = [dict() for _ in range(cap + 1)]   # pinned or leaf
        self.subtree_n = [dict() for _ in range(cap + 1)]  # |elements at+below|
        self.max_level = 0

        for level in range(cap + 1):
            nxt = reach[level + 1] if level < cap else None
            for q, held in reach[level].items():
                self.subtree_n[level][q] = held.size
                self.max_level = max(self.max_level, level)
                descends = self.max_level_of[held] > level
                if (held.size > self.ncrit and descends.any() and level < cap):
                    self.resident[level][q] = held[~descends]
                    self._scatter(held[descends], level + 1, nxt)
                else:
                    self.resident[level][q] = held
            reach[level] = None

        self._index()

    def _scatter(self, elems: np.ndarray, level: int, out: dict) -> None:
        shift = self.level_cap - level
        code = (((self._fine[elems, 0] >> shift) << (2 * _SHIFT))
                | ((self._fine[elems, 1] >> shift) << _SHIFT)
                | (self._fine[elems, 2] >> shift))
        order = np.argsort(code, kind="stable")
        code = code[order]
        elems = elems[order]
        cut = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
        for a, b in zip(cut, np.r_[cut[1:], code.size]):
            out[int(code[a])] = elems[a:b]

    def _index(self) -> None:
        """Flat box registry, parent links, and each box's true extent."""
        self.boxes: list = []                     # (level, key)
        self.box_id = [dict() for _ in range(self.max_level + 1)]
        for level in range(self.max_level + 1):
            for q in self.subtree_n[level]:
                self.box_id[level][q] = len(self.boxes)
                self.boxes.append((level, q))
        nb = len(self.boxes)
        self.level = np.array([b[0] for b in self.boxes], dtype=np.int64)
        self.n_subtree = np.array(
            [self.subtree_n[L][q] for (L, q) in self.boxes], dtype=np.int64)
        self.n_resident = np.array(
            [self.resident[L][q].size for (L, q) in self.boxes],
            dtype=np.int64)
        self.parent = np.full(nb, -1, dtype=np.int64)
        for bi, (L, q) in enumerate(self.boxes):
            if L:
                i, j, k = _unkey(q)
                self.parent[bi] = self.box_id[L - 1][_key(i >> 1, j >> 1,
                                                          k >> 1)]
        self.children = defaultdict(list)
        for bi in range(nb):
            p = self.parent[bi]
            if p >= 0:
                self.children[int(p)].append(bi)
        # A box is a LEAF when nothing descends from it. Because an oversized
        # element is pinned rather than pushed down, a box can be internal and
        # still hold residents; those residents are what W and X exist for.
        self.is_leaf = np.array([len(self.children[bi]) == 0
                                 for bi in range(nb)])
        self.occupied = np.flatnonzero(self.n_resident > 0)

    def elements_of(self, bi: int) -> np.ndarray:
        """Elements resident in box ``bi`` (pinned there, or its leaf set)."""
        L, q = self.boxes[bi]
        return self.resident[L][q]

    def cube(self, bi: int) -> tuple:
        """``(lo, hi)`` of the box's NOMINAL cube -- what M2L is stated on."""
        L, q = self.boxes[bi]
        i, j, k = _unkey(q)
        h = self.root_edge / (1 << L)
        lo = self.origin + np.array([i, j, k], dtype=float) * h
        return lo, lo + h

    def protrusion(self, verts: np.ndarray) -> np.ndarray:
        """Per element, how far it sticks out of its own box, in box edges.

        Zero means contained. This is the diagnostic that condemns naive
        placement: at ``placement_safety`` 1 the maximum is ~0.53, which is why
        interpolation must use the enlarged extent rather than the cube.
        """
        verts = np.asarray(verts)
        level = np.zeros(self.n, dtype=np.int64)
        for L in range(self.max_level + 1):
            for q, held in self.resident[L].items():
                level[held] = L
        shift = self.level_cap - level
        coord = self._fine >> shift[:, None]
        h = self.root_edge / (2.0 ** level)
        box_lo = self.origin + coord * h[:, None]
        box_hi = box_lo + h[:, None]
        out = np.maximum(box_lo[:, None, :] - verts, verts - box_hi[:, None, :])
        return np.maximum(out.max(axis=(1, 2)), 0.0) / h

    def extents(self, verts: np.ndarray) -> tuple:
        """``(lo, hi)`` per box of everything it holds, subtree included.

        The ENLARGED interpolation domain. Keeping the M2L table on the nominal
        cubes preserves translation invariance; P2M and L2P work on these.
        """
        verts = np.asarray(verts)
        nb = len(self.boxes)
        lo = np.full((nb, 3), np.inf)
        hi = np.full((nb, 3), -np.inf)
        for bi in range(nb):
            held = self.elements_of(bi)
            if held.size:
                v = verts[held].reshape(-1, 3)
                lo[bi] = v.min(axis=0)
                hi[bi] = v.max(axis=0)
        for bi in range(nb - 1, -1, -1):          # boxes are level-ordered
            p = self.parent[bi]
            if p >= 0 and np.isfinite(lo[bi]).all():
                lo[p] = np.minimum(lo[p], lo[bi])
                hi[p] = np.maximum(hi[p], hi[bi])
        return lo, hi

    def summary(self) -> str:
        occ = self.n_resident[self.occupied]
        return (f"{self.n} elements, {len(self.boxes)} boxes, depth "
                f"{self.max_level}, {self.occupied.size} occupied "
                f"(mean {occ.mean():.1f}, max {occ.max()}), ncrit {self.ncrit}")


class _Neighbourhood:
    """Boxes at level >= level(B) whose cube touches B's.

    Bucketed by the coarser box's key so the search is over the 27 cells
    around B at each level rather than over every box.
    """

    def __init__(self, tree: Octree):
        self.tree = tree
        self.bucket = [dict() for _ in range(tree.max_level + 1)]
        for deep in range(tree.max_level + 1):
            for coarse in range(deep + 1):
                d = deep - coarse
                table = defaultdict(list)
                for q in tree.subtree_n[deep]:
                    i, j, k = _unkey(q)
                    table[_key(i >> d, j >> d, k >> d)].append(
                        tree.box_id[deep][q])
                self.bucket[deep][coarse] = dict(table)

    def at_or_below(self, bi: int) -> list:
        t = self.tree
        L, q = t.boxes[bi]
        i, j, k = _unkey(q)
        span = 1 << L
        out = []
        for deep in range(L, t.max_level + 1):
            table = self.bucket[deep][L]
            for di in (-1, 0, 1):
                ii = i + di
                if ii < 0 or ii >= span:
                    continue
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if jj < 0 or jj >= span:
                        continue
                    for dk in (-1, 0, 1):
                        kk = k + dk
                        if kk < 0 or kk >= span:
                            continue
                        cand = table.get(_key(ii, jj, kk))
                        if not cand:
                            continue
                        if deep == L:
                            out.extend(cand)
                        else:
                            out.extend(c for c in cand
                                       if cubes_adjacent(L, q, deep,
                                                         t.boxes[c][1]))
        return out


class InteractionLists:
    """U, V, W, X over an :class:`Octree`, from a dual-tree traversal.

    Derived rather than enumerated. Writing the four lists out box by box --
    colleagues, children of the parent's colleagues, descendants that have
    separated -- is how the textbook states them and it double-counts the
    moment a box can be occupied AND internal, because the descendants of two
    adjacent boxes on the same ancestor chain are reachable by more than one
    path. Here one traversal covers every ORDERED element pair exactly once by
    construction, and the four lists are what it emits:

        descend(A, B) covers subtree(A) x subtree(B)
          not adjacent                -> V: one M2L, A's local from B's multipole
          both leaves                 -> U: direct
          otherwise, split exactly
              res(A) x res(B)         -> U
              res(A) x subtree(b)     -> recurse, emitting W when it separates
              subtree(a) x res(B)     -> recurse, emitting X when it separates
              subtree(a) x subtree(b) -> recurse

    W is therefore "a box's own residents against a source subtree that has
    separated from them" (M2P) and X its transpose (P2L), which is exactly
    what pinned oversized elements need: they cannot descend, so their
    interaction with a neighbour's subtree separates at a level they are not
    at. ``verify_octree`` gates the resulting identity U + V + 2W = N^2.
    """

    def __init__(self, tree: Octree):
        self.tree = tree
        self.U = defaultdict(list)
        self.V = defaultdict(list)
        self.W = defaultdict(list)
        self.X = defaultdict(list)
        root = tree.box_id[0][_key(0, 0, 0)]
        self._descend(root, root)

    def _adjacent(self, a: int, b: int) -> bool:
        la, qa = self.tree.boxes[a]
        lb, qb = self.tree.boxes[b]
        return cubes_adjacent(la, qa, lb, qb)

    def _descend(self, a: int, b: int) -> None:
        """subtree(a) x subtree(b)."""
        t = self.tree
        if not t.n_subtree[a] or not t.n_subtree[b]:
            return
        if not self._adjacent(a, b):
            self.V[a].append(b)
            return
        a_leaf = not t.children[a]
        b_leaf = not t.children[b]
        if a_leaf and b_leaf:
            if t.n_resident[a] and t.n_resident[b]:
                self.U[a].append(b)
            return
        if a_leaf:
            self._descend_res_sub(a, b)
            return
        if b_leaf:
            self._descend_sub_res(a, b)
            return
        if t.n_resident[a] and t.n_resident[b]:
            self.U[a].append(b)
        for child in t.children[b]:
            self._descend_res_sub(a, child)
        for child in t.children[a]:
            self._descend_sub_res(child, b)
        for ca in t.children[a]:
            for cb in t.children[b]:
                self._descend(ca, cb)

    def _descend_res_sub(self, a: int, b: int) -> None:
        """res(a) x subtree(b) -- a's pinned elements against b's subtree."""
        t = self.tree
        if not t.n_resident[a] or not t.n_subtree[b]:
            return
        if not self._adjacent(a, b):
            self.W[a].append(b)
            return
        if t.n_resident[b]:
            self.U[a].append(b)
        for child in t.children[b]:
            self._descend_res_sub(a, child)

    def _descend_sub_res(self, a: int, b: int) -> None:
        """subtree(a) x res(b) -- the transpose direction."""
        t = self.tree
        if not t.n_subtree[a] or not t.n_resident[b]:
            return
        if not self._adjacent(a, b):
            self.X[a].append(b)
            return
        if t.n_resident[a]:
            self.U[a].append(b)
        for child in t.children[a]:
            self._descend_sub_res(child, b)

    def counts(self) -> dict:
        return {"U": sum(len(v) for v in self.U.values()),
                "V": sum(len(v) for v in self.V.values()),
                "W": sum(len(v) for v in self.W.values()),
                "X": sum(len(v) for v in self.X.values())}
