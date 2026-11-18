"""A direct solver for a compressed boundary: block low-rank LU.

The compressed backend holds a boundary operator it can only APPLY, and
solves with it by iteration, one right-hand side at a time. A body that is
solved for a thousand right-hand sides at every state (what its boundary
returns of a fault's slip, ``la.coupling``) or whose inverse is itself the
operator wanted (a slip on an interface, whose stiffness is a Schur
complement of the whole body) needs the factors instead.

The unknowns are cut into CLUSTERS: the nodes of each patch's own cluster
tree at ``leaf`` elements, a cluster holding every slot of its patch there
(an interface's displacement and traction together, so a diagonal block is
the local transmission problem, which is well posed where a first-kind
block alone is not). A pair of clusters is FAR when their boxes are
admissible and near otherwise; a far block is ``X Y'`` at a rank the
tolerance sets, a near one dense. That is the flat format of the operator
itself, one level coarser, and the blocks are read off the operator's own
compressed views (``from_assembled``): no kernel is evaluated again.

The factorization is block LU, pivoting inside a diagonal block only
(measured sufficient on the megathrust's body at every state: the error of
a solve follows the tolerance down to 1e-8). It is LEFT-LOOKING, because
that is what lets it run on matrix products of useful size. A far block's
update is a sum over every earlier pivot of products of low-rank blocks,
``X_im (Y_im' X_mk) Y_mk'``: one small product per triple of clusters, and
there are p^3 / 3 triples. Done one at a time they are the whole cost.
Here the factors of a row of L and of a column of U are stored side by
side, the middle factors ``G_m = Y_m' X_m`` are formed once per pivot by
one product, and a block is compressed from a sketch of its update (the
range of ``T Omega``, then ``Q' T``), so that what is left per triple is a
product of two matrices of the size of a RANK.

``factor(stop)`` may stop early: ``schur`` is then the complement of the
clusters eliminated, in the same block format.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import lu_factor, qr, solve_triangular, svd

from .. import defaults

FAR, NEAR = 1, 2


def _cut(node, leaf: int) -> list:
    if node.is_leaf or node.n <= leaf:
        return [node]
    return _cut(node.left, leaf) + _cut(node.right, leaf)


class Clusters:
    """The clusters of a layout (module docstring): for cluster ``g`` its
    unknowns ``dofs[g]`` (slot-major, elements ascending), its box and its
    patch; for a patch, the cluster and the position in it of each element."""

    def __init__(self, layout, trees: dict, leaf: int):
        self.slots: dict = {}
        self.slot_index: dict = {}
        for s in layout.slots:
            mine = self.slots.setdefault(id(s.patch), [])
            self.slot_index[(id(s.patch), s.kind)] = len(mine)
            mine.append(s)
        self.dofs, self.lo, self.hi, self.n_elem, self.patch = [], [], [], [], []
        self.cluster_of, self.position = {}, {}
        for pid, slots in self.slots.items():
            patch = slots[0].patch
            n = patch.mesh.n_triangles
            cl = np.empty(n, dtype=np.int64)
            pos = np.empty(n, dtype=np.int64)
            for node in _cut(trees[id(patch.mesh)], leaf):
                e = np.sort(node.indices)
                cl[e] = len(self.dofs)
                pos[e] = np.arange(e.size)
                self.dofs.append(np.concatenate([
                    (s.offset + 3 * e[:, None] + np.arange(3)[None, :]).ravel()
                    for s in slots]))
                self.lo.append(node.lo)
                self.hi.append(node.hi)
                self.n_elem.append(e.size)
                self.patch.append(patch)
            self.cluster_of[pid], self.position[pid] = cl, pos
        self.p = len(self.dofs)
        self.sizes = [d.size for d in self.dofs]
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)

    def far(self, eta: float) -> np.ndarray:
        """(p, p) bool: the admissible pairs, by the boxes' rule of
        ``la.cluster.is_admissible``."""
        gap = np.maximum(0.0, np.maximum(
            self.lo[:, None, :] - self.hi[None, :, :],
            self.lo[None, :, :] - self.hi[:, None, :]))
        dist = np.linalg.norm(gap, axis=2)
        diam = np.linalg.norm(self.hi - self.lo, axis=1)
        out = np.minimum(diam[:, None], diam[None, :]) < eta * dist
        np.fill_diagonal(out, False)
        return out

    def local(self, patch, slot, elems: np.ndarray) -> np.ndarray:
        """The unknowns of ``slot`` at ``elems`` (one cluster's), numbered
        inside that cluster."""
        pid = id(patch)
        q = self.slot_index[(pid, slot.kind)]
        g = self.cluster_of[pid][elems[0]]
        base = 3 * q * self.n_elem[g]
        return (base + 3 * self.position[pid][elems][:, None]
                + np.arange(3)[None, :]).ravel()


def _groups(ids: np.ndarray):
    if (ids == ids[0]).all():
        return [(int(ids[0]), None)]
    return [(int(u), np.nonzero(ids == u)[0]) for u in np.unique(ids)]


def _rows(sel) -> np.ndarray:
    return (3 * sel[:, None] + np.arange(3)[None, :]).ravel()


def recompress(X: np.ndarray, Y: np.ndarray, tol: float):
    """``X Y'`` at the rank ``tol`` of its largest singular value keeps."""
    if X.shape[1] == 0:
        return X, Y
    Qx, Rx = qr(X, mode="economic", check_finite=False)
    Qy, Ry = qr(Y, mode="economic", check_finite=False)
    U, s, Vt = svd(Rx @ Ry.T, full_matrices=False, check_finite=False)
    r = int((s > tol * s[0]).sum()) if s.size and s[0] > 0.0 else 0
    return Qx @ (U[:, :r] * s[:r]), Qy @ Vt[:r].T


def lowrank(D: np.ndarray, tol: float, rng, guess: int = 16):
    """A dense block as ``X Y'`` at ``tol`` of its largest singular value,
    from a sketch of its range: what a small leaf inside a far block
    costs, a full SVD being fifteen times that."""
    k = min(D.shape)
    s = min(k, guess + 8)
    while s < k:
        Q = qr(D @ rng.standard_normal((D.shape[1], s)), mode="economic",
               check_finite=False)[0]
        U, sv, Vt = svd(Q.T @ D, full_matrices=False, check_finite=False)
        r = int((sv > tol * sv[0]).sum()) if sv[0] > 0.0 else 0
        if r <= s - 4:
            return Q @ (U[:, :r] * sv[:r]), Vt[:r].T
        s = min(k, 2 * s)
    U, sv, Vt = svd(D, full_matrices=False, check_finite=False)
    r = int((sv > tol * sv[0]).sum()) if sv.size and sv[0] > 0.0 else 0
    return U[:, :r] * sv[:r], Vt[:r].T


def blocks_of(asm, cl: Clusters, tol: float, eta: float, seed: int = 0):
    """``(kind, blocks)`` of an assembled compressed operator on ``cl``:
    every term's view cut along the clusters, the pieces of a far pair
    joined and recompressed, those of a near one summed dense, and the
    collocation diagonal added. A block with no term is absent (zero)."""
    from ..kernels import kernel_coeffs
    from ..model.equations import term_diagonal

    far = cl.far(eta)
    rng = np.random.default_rng(seed)
    pieces: dict = {}
    for term in asm.system.terms:
        q, p = term.field_patch, term.source_patch
        if term.row.patch is not q or term.col.patch is not p:
            raise ValueError(f"term {term.row.name} <- {term.col.name}: its "
                             f"slots are not its patches'")
        view = asm.pair_for(term)._view(
            kernel_coeffs(term.kernel, asm.materials[term.region.name]))
        clq, clp = cl.cluster_of[id(q)], cl.cluster_of[id(p)]
        op = term.col_op
        for rdofs, cdofs, A, V in view.blocks():
            re, ce = rdofs[::3] // 3, cdofs[::3] // 3
            for a, rsel in _groups(clq[re]):
                ea = re if rsel is None else re[rsel]
                lrows = cl.local(q, term.row, ea)
                Aa = A if rsel is None else A[_rows(rsel)]
                for b, csel in _groups(clp[ce]):
                    eb = ce if csel is None else ce[csel]
                    lcols = cl.local(p, term.col, eb)
                    if V is None:
                        blk = np.asarray(
                            Aa if csel is None else Aa[:, _rows(csel)],
                            dtype=float) * term.scale
                        if op is not None:
                            blk = np.einsum(
                                "rnk,nkj->rnj",
                                blk.reshape(blk.shape[0], -1, 3),
                                op[eb]).reshape(blk.shape[0], -1)
                        piece = (lrows, lcols, blk, None)
                    else:
                        Vb = np.asarray(V if csel is None else V[_rows(csel)],
                                        dtype=float)
                        if op is not None:
                            Vb = np.einsum(
                                "nkj,nkr->njr", op[eb],
                                Vb.reshape(-1, 3, Vb.shape[1])
                            ).reshape(-1, Vb.shape[1])
                        piece = (lrows, lcols,
                                 np.asarray(Aa, dtype=float) * term.scale, Vb)
                    pieces.setdefault((a, b), []).append(piece)

    blocks: dict = {}
    kind = np.where(far, FAR, NEAR).astype(np.int8)
    for key in list(pieces):
        a, b = key
        plist = pieces.pop(key)
        m, n = cl.sizes[a], cl.sizes[b]
        if kind[a, b] == FAR:
            cols = []
            for lrows, lcols, A, V in plist:
                if V is None:              # a leaf too small for the ACA
                    A, V = lowrank(A, 0.1 * tol, rng)
                cols.append((lrows, lcols, A, V))
            R = sum(c[2].shape[1] for c in cols)
            X, Y, at = np.zeros((m, R)), np.zeros((n, R)), 0
            for lrows, lcols, A, V in cols:
                r = A.shape[1]
                X[lrows, at:at + r] = A
                Y[lcols, at:at + r] = V
                at += r
            X, Y = recompress(X, Y, tol)
            if 2 * X.shape[1] * (m + n) <= m * n:
                blocks[key] = (X, Y)
                continue
            kind[a, b] = NEAR
            blocks[key] = X @ Y.T
            continue
        D = np.zeros((m, n))
        for lrows, lcols, A, V in plist:
            D[np.ix_(lrows, lcols)] += A if V is None else A @ V.T
        blocks[key] = D
    for term in asm.system.terms:
        Dg = term_diagonal(term, asm.calib)
        if Dg is None:
            continue
        q = term.field_patch
        for g in np.unique(cl.cluster_of[id(q)]):
            e = np.nonzero(cl.cluster_of[id(q)] == g)[0]
            lr = cl.local(q, term.row, e).reshape(-1, 3)
            lc = cl.local(q, term.col, e).reshape(-1, 3)
            blk = blocks.setdefault((g, g), np.zeros((cl.sizes[g],) * 2))
            for i in range(3):
                for j in range(3):
                    blk[lr[:, i], lc[:, j]] += Dg[e, i, j]
    return kind, blocks


class _Grow:
    """Columns appended to a ``(c, R)`` array as a row of L or a column of
    U is finalized, pivot by pivot; ``a[:, :n]`` is what is there."""

    def __init__(self, c: int, capacity: int):
        self.a = np.empty((c, max(capacity, 1)), order="F")
        self.n = 0

    def append(self, X: np.ndarray) -> tuple:
        r = X.shape[1]
        if self.n + r > self.a.shape[1]:
            new = np.empty((self.a.shape[0],
                            max(self.n + r, int(1.5 * self.a.shape[1]) + 8)),
                           order="F")
            new[:, :self.n] = self.a[:, :self.n]
            self.a = new
        self.a[:, self.n:self.n + r] = X
        self.n += r
        return self.n - r, self.n


class _Index:
    """The same for a run of integers ``start .. start + r``."""

    def __init__(self):
        self.a = np.empty(64, dtype=np.int64)
        self.n = 0

    def append(self, start: int, r: int) -> None:
        if self.n + r > self.a.size:
            new = np.empty(max(self.n + r, 2 * self.a.size), dtype=np.int64)
            new[:self.n] = self.a[:self.n]
            self.a = new
        self.a[self.n:self.n + r] = np.arange(start, start + r)
        self.n += r


class BlockLU:
    """Block low-rank LU of ``blocks`` on clusters of ``sizes`` (module
    docstring). ``kind[i, j]`` is FAR or NEAR; a far block is ``(X, Y)``
    and a near one dense, an absent one zero."""

    def __init__(self, sizes, kind: np.ndarray, blocks: dict, tol: float,
                 oversample: int = defaults.BLR_OVERSAMPLE, seed: int = 0,
                 log=None):
        self.c = [int(s) for s in sizes]
        self.p = len(self.c)
        self.kind = kind
        self.A = blocks
        self.tol = float(tol)
        self.oversample = int(oversample)
        self.rng = np.random.default_rng(seed)
        self.log = log
        p = self.p
        rank0 = np.zeros((p, p), dtype=np.int64)
        for (i, j), blk in blocks.items():
            if isinstance(blk, tuple):
                rank0[i, j] = blk[0].shape[1]
        self.rank0 = rank0
        low = np.tril(rank0, -1).sum(axis=1)
        up = np.triu(rank0, 1).sum(axis=0)
        self.XL = [_Grow(self.c[i], int(1.2 * low[i]) + 32) for i in range(p)]
        self.YU = [_Grow(self.c[j], int(1.2 * up[j]) + 32) for j in range(p)]
        self.YL = [None] * p        # per pivot m: Y of L_im, i > m, side by side
        self.XU = [None] * p        # per pivot m: X of U_mj, j > m
        self.G = [None] * p         # YL[m]' XU[m]
        z = np.zeros((p, p), dtype=np.int64)
        self.l0, self.l1 = z.copy(), z.copy()      # L_im in XL[i]
        self.lc0, self.lc1 = z.copy(), z.copy()    # L_im in YL[m]: [i, m]
        self.u0, self.u1 = z.copy(), z.copy()      # U_mj in XU[m]: [m, j]
        self.uc0, self.uc1 = z.copy(), z.copy()    # U_mj in YU[j]: [m, j]
        # where a row's (a column's) factors sit in the pivots' own
        # stacks, YL (XU) one after another: what gathers a sketch
        self.iL = [_Index() for _ in range(p)]
        self.iU = [_Index() for _ in range(p)]
        self.offL, self.offU = [0], [0]
        self.D: dict = {}           # dense factor blocks; (k, k) is its LU
        # a pivot's near blocks side by side, L below it and U beside it
        # (D holds views of them), and where each block sits
        self.DL, self.DU = [None] * p, [None] * p
        self.dl_at = [dict() for _ in range(p)]
        self.du_at = [dict() for _ in range(p)]
        self.perm = [None] * p
        self.stop = 0
        self.seconds = {"near": 0.0, "far": 0.0, "pivot": 0.0}
        self.resketched = 0

    # -- the updates ------------------------------------------------------

    def _near(self, i: int, j: int, limit: int) -> np.ndarray:
        """``A_ij - sum_{m < limit} L_im U_mj`` of a near block, dense."""
        blk = self.A.pop((i, j), None)
        T = (np.zeros((self.c[i], self.c[j])) if blk is None
             else np.array(blk, dtype=float, order="F"))
        kind, YUj = self.kind, self.YU[j]
        C = None
        for m in range(limit):
            kl, ku = kind[i, m], kind[m, j]
            if ku == FAR:
                c0, c1 = self.uc0[m, j], self.uc1[m, j]
                if c1 == c0:
                    continue
                u0, u1 = self.u0[m, j], self.u1[m, j]
                if kl == FAR:
                    a, b = self.l0[i, m], self.l1[i, m]
                    if b == a:
                        continue
                    piece = self.XL[i].a[:, a:b] @ self.G[m][
                        self.lc0[i, m]:self.lc1[i, m], u0:u1]
                else:
                    piece = self.D[(i, m)] @ self.XU[m][:, u0:u1]
                if C is None:
                    C = np.zeros((self.c[i], YUj.n), order="F")
                C[:, c0:c1] = piece
            else:
                Dmj = self.D[(m, j)]
                if kl == FAR:
                    a, b = self.l0[i, m], self.l1[i, m]
                    if b == a:
                        continue
                    T -= self.XL[i].a[:, a:b] @ (
                        self.YL[m][:, self.lc0[i, m]:self.lc1[i, m]].T @ Dmj)
                else:
                    T -= self.D[(i, m)] @ Dmj
        if C is not None:
            T -= C @ YUj.a[:, :C.shape[1]].T
        return T

    def _far(self, limit: int, col: int, targets: list, side: str,
             boost: int = 1) -> dict:
        """The far blocks ``(i, col)``, ``i`` in ``targets``, less their
        updates from the pivots below ``limit``, each as ``(X, Y)``. With
        ``side="row"`` everything is read transposed: the blocks are
        ``(col, i)`` and the pair returned is still ``(X, Y)`` of the block
        as it stands in the matrix.

        What a pivot ``m`` contributes to the sketch of EVERY target is one
        product, ``G_m`` (or the pivot's own factors) on the sketch of its
        block in this column; a target then gathers its rows of them
        (``_Index``). Only the projection ``Q' T`` is taken a pivot at a
        time, in products of the size of two ranks."""
        if not targets:
            return {}
        kind, c = self.kind, self.c
        if side == "col":
            base = lambda i: self.A.pop((i, col), None)             # noqa: E731
            kl = kind[:, :limit]                 # [i, m]: the kind of L_im
            ku = kind[:limit, col]
            Lrow, idx, off = self.XL, self.iL, self.offL
            la, lb = self.l0, self.l1            # [i, m] in Lrow[i]
            Lcol = self.YL
            ca, cb = self.lc0, self.lc1          # [i, m] in Lcol[m]
            Ucol = self.YU[col]
            ua, ub = self.uc0[:limit, col], self.uc1[:limit, col]
            Urow = self.XU
            xa, xb = self.u0[:limit, col], self.u1[:limit, col]
            Gcol = lambda m: self.G[m][:, xa[m]:xb[m]]              # noqa: E731
            DL = lambda i, m: self.D[(i, m)]                        # noqa: E731
            DU = lambda m: self.D[(m, col)]                         # noqa: E731
            tall, at_of = (lambda m: self.DL[m]), self.dl_at        # noqa: E731
        else:
            def base(i):
                blk = self.A.pop((col, i), None)
                return None if blk is None else (blk[1], blk[0])
            kl = kind[:limit, :].T
            ku = kind[col, :limit]
            Lrow, idx, off = self.YU, self.iU, self.offU
            la, lb = self.uc0.T, self.uc1.T
            Lcol = self.XU
            ca, cb = self.u0.T, self.u1.T
            Ucol = self.XL[col]
            ua, ub = self.l0[col, :limit], self.l1[col, :limit]
            Urow = self.YL
            xa, xb = self.lc0[col, :limit], self.lc1[col, :limit]
            Gcol = lambda m: self.G[m][xa[m]:xb[m], :].T            # noqa: E731
            DL = lambda i, m: self.D[(m, i)].T                      # noqa: E731
            DU = lambda m: self.D[(col, m)].T                       # noqa: E731
            tall, at_of = (lambda m: self.DU[m].T), self.du_at      # noqa: E731

        bases = {i: base(i) for i in targets}
        r0 = {i: (0 if b is None else b[0].shape[1]) for i, b in bases.items()}
        want = {i: min(boost * (max(r0[i], 8) + self.oversample),
                       c[i], c[col]) for i in targets}
        s_max = max(want.values())
        Om = self.rng.standard_normal((c[col], s_max))
        nU = int(ub.max()) if limit > 0 else 0
        P = Ucol.a[:, :nU].T @ Om                  # every far U_m,col at once
        far_m = [m for m in range(limit) if ku[m] == FAR and ub[m] > ua[m]]
        near_m = [m for m in range(limit) if ku[m] == NEAR]
        zs = {m: DU(m) @ Om for m in near_m}       # U_m,col Omega, dense U
        Vall = np.zeros((off[limit], s_max))       # Y_im' U_m,col Omega, all i
        GC = {}
        for m in far_m:
            GC[m] = Gcol(m)
            Vall[off[m]:off[m + 1]] = GC[m] @ P[ua[m]:ub[m]]
        for m in near_m:
            Vall[off[m]:off[m + 1]] = Lcol[m].T @ zs[m]
        dxs = {}

        def dx(m):
            """Every near L_im, i below m, on the left factor of U_m,col:
            one product for all of them."""
            if m not in dxs:
                dxs[m] = tall(m) @ Urow[m][:, xa[m]:xb[m]]
            return dxs[m]

        out, again = {}, []
        far_set = np.zeros(limit, dtype=bool)
        far_set[far_m] = True
        for i in targets:
            s = want[i]
            b0 = bases[i]
            nL = Lrow[i].n if limit > 0 else 0
            Li = Lrow[i].a[:, :nL]
            kli = kl[i]
            dense_l = np.flatnonzero(kli == NEAR).tolist()
            S = (np.zeros((c[i], s)) if b0 is None or r0[i] == 0
                 else b0[0] @ (b0[1].T @ Om[:, :s]))
            for m in dense_l:
                if ku[m] == NEAR:
                    S -= DL(i, m) @ zs[m][:, :s]
                elif ub[m] > ua[m]:
                    a, b = at_of[m][i]
                    S -= dx(m)[a:b] @ P[ua[m]:ub[m], :s]
            if nL:
                S -= Li @ Vall[idx[i].a[:nL], :s]
            Q = qr(S, mode="economic", check_finite=False)[0]
            B = (np.zeros((Q.shape[1], c[col])) if b0 is None or r0[i] == 0
                 else (Q.T @ b0[0]) @ b0[1].T)
            W = Q.T @ Li
            H = np.zeros((Q.shape[1], nU))
            lai, lbi, cai, cbi = la[i], lb[i], ca[i], cb[i]
            lr_l = np.flatnonzero((kli == FAR) & (lbi[:limit] > lai[:limit]))
            for m in lr_l[far_set[lr_l]].tolist():
                H[:, ua[m]:ub[m]] = W[:, lai[m]:lbi[m]] @ GC[m][cai[m]:cbi[m]]
            for m in lr_l[~far_set[lr_l]].tolist():
                if ku[m] == NEAR:
                    B -= (W[:, lai[m]:lbi[m]]
                          @ Lcol[m][:, cai[m]:cbi[m]].T) @ DU(m)
            for m in dense_l:
                if ku[m] == NEAR:
                    B -= (Q.T @ DL(i, m)) @ DU(m)
                elif ub[m] > ua[m]:
                    a, b = at_of[m][i]
                    H[:, ua[m]:ub[m]] = Q.T @ dx(m)[a:b]
            B -= H @ Ucol.a[:, :nU].T
            Ub, sv, Vt = svd(B, full_matrices=False, check_finite=False)
            r = int((sv > self.tol * sv[0]).sum()) if sv[0] > 0.0 else 0
            if r > s - self.oversample // 2 and s < min(c[i], c[col]):
                again.append(i)
                continue
            X, Y = Q @ (Ub[:, :r] * sv[:r]), Vt[:r].T
            out[i] = (X, Y) if side == "col" else (Y, X)
        if again:
            self.resketched += len(again)
            for i in again:                 # put the bases back for a retry
                b0 = bases[i]
                if b0 is not None:
                    key = (i, col) if side == "col" else (col, i)
                    self.A[key] = b0 if side == "col" else (b0[1], b0[0])
            out.update(self._far(limit, col, again, side, boost=2 * boost))
        return out

    # -- factorization ----------------------------------------------------

    def factor(self, stop: int | None = None):
        """Eliminate the first ``stop`` clusters (all of them by default)."""
        p, kind, c = self.p, self.kind, self.c
        stop = p if stop is None else int(stop)
        t_last = time.time()
        for k in range(stop):
            t0 = time.time()
            lu, piv = lu_factor(self._near(k, k, k), overwrite_a=True,
                                check_finite=False)
            perm = np.arange(c[k])
            for a, b in enumerate(piv):
                perm[a], perm[b] = perm[b], perm[a]
            self.D[(k, k)], self.perm[k] = lu, perm
            self.seconds["pivot"] += time.time() - t0
            t0 = time.time()
            beside = [j for j in range(k + 1, p) if kind[k, j] == NEAR]
            below = [j for j in range(k + 1, p) if kind[j, k] == NEAR]
            if beside:
                U = solve_triangular(
                    lu, np.hstack([self._near(k, j, k) for j in beside])[perm],
                    lower=True, unit_diagonal=True, check_finite=False)
                at = 0
                for j in beside:
                    self.du_at[k][j] = (at, at + c[j])
                    self.D[(k, j)] = U[:, at:at + c[j]]
                    at += c[j]
                self.DU[k] = U
            if below:
                L = solve_triangular(
                    lu, np.vstack([self._near(j, k, k) for j in below]).T,
                    trans="T", lower=False, check_finite=False).T
                at = 0
                for j in below:
                    self.dl_at[k][j] = (at, at + c[j])
                    self.D[(j, k)] = L[at:at + c[j]]
                    at += c[j]
                self.DL[k] = L
            self.seconds["near"] += time.time() - t0
            t0 = time.time()
            rows = [i for i in range(k + 1, p) if kind[i, k] == FAR]
            cols = [j for j in range(k + 1, p) if kind[k, j] == FAR]
            Lk = self._far(k, k, rows, "col")
            Uk = self._far(k, k, cols, "row")
            ys, at = [], 0
            for i in rows:
                X, Y = Lk[i]
                Y = solve_triangular(lu, Y, trans="T", lower=False,
                                     check_finite=False)
                self.l0[i, k], self.l1[i, k] = self.XL[i].append(X)
                self.lc0[i, k], self.lc1[i, k] = at, at + Y.shape[1]
                self.iL[i].append(self.offL[k] + at, Y.shape[1])
                at += Y.shape[1]
                ys.append(Y)
            self.YL[k] = (np.hstack(ys) if ys else np.zeros((c[k], 0)))
            xs, at = [], 0
            for j in cols:
                X, Y = Uk[j]
                X = solve_triangular(lu, X[perm], lower=True,
                                     unit_diagonal=True, check_finite=False)
                self.uc0[k, j], self.uc1[k, j] = self.YU[j].append(Y)
                self.u0[k, j], self.u1[k, j] = at, at + X.shape[1]
                self.iU[j].append(self.offU[k] + at, X.shape[1])
                at += X.shape[1]
                xs.append(X)
            self.XU[k] = (np.hstack(xs) if xs else np.zeros((c[k], 0)))
            self.G[k] = self.YL[k].T @ self.XU[k]
            self.offL.append(self.offL[-1] + self.YL[k].shape[1])
            self.offU.append(self.offU[-1] + self.XU[k].shape[1])
            self.seconds["far"] += time.time() - t0
            if self.log is not None and time.time() - t_last > 30.0:
                t_last = time.time()
                self.log(f"      pivot {k + 1} of {stop}")
        self.stop = stop
        return self

    def schur(self):
        """``(kind, blocks)`` of the complement of the clusters eliminated,
        on the clusters that are left (numbered from 0)."""
        s, p, kind = self.stop, self.p, self.kind
        blocks = {}
        for j in range(s, p):
            for i in range(s, p):
                if kind[i, j] == NEAR:
                    blocks[(i - s, j - s)] = self._near(i, j, s)
            far = self._far(s, j, [i for i in range(s, p)
                                   if kind[i, j] == FAR], "col")
            for i, blk in far.items():
                blocks[(i - s, j - s)] = blk
        return kind[s:, s:].copy(), blocks

    # -- solves -------------------------------------------------------------

    def solve(self, b: np.ndarray, dofs: list) -> np.ndarray:
        """``A^-1 b`` for one or many right-hand sides (columns), the
        unknowns of cluster ``g`` at ``dofs[g]``."""
        if self.stop != self.p:
            raise ValueError("a partial factorization does not solve")
        one = b.ndim == 1
        b = b[:, None] if one else b
        p, kind = self.p, self.kind
        x = [np.array(b[d], dtype=float, order="F") for d in dofs]
        for k in range(p):                       # L y = P b
            acc = x[k]
            V = np.zeros((self.XL[k].n, acc.shape[1]))
            for m in range(k):
                if kind[k, m] == FAR:
                    a, bb = self.l0[k, m], self.l1[k, m]
                    if bb > a:
                        V[a:bb] = self.YL[m][
                            :, self.lc0[k, m]:self.lc1[k, m]].T @ x[m]
                else:
                    acc -= self.D[(k, m)] @ x[m]
            acc -= self.XL[k].a[:, :self.XL[k].n] @ V
            x[k] = solve_triangular(self.D[(k, k)], acc[self.perm[k]],
                                    lower=True, unit_diagonal=True,
                                    check_finite=False)
        for k in range(p - 1, -1, -1):           # U x = y
            acc = x[k]
            V = np.zeros((self.XU[k].shape[1], acc.shape[1]))
            for j in range(k + 1, p):
                if kind[k, j] == FAR:
                    a, bb = self.u0[k, j], self.u1[k, j]
                    if bb > a:
                        V[a:bb] = self.YU[j].a[
                            :, self.uc0[k, j]:self.uc1[k, j]].T @ x[j]
                else:
                    acc -= self.D[(k, j)] @ x[j]
            acc -= self.XU[k] @ V
            x[k] = solve_triangular(self.D[(k, k)], acc, lower=False,
                                    check_finite=False)
        out = np.empty((b.shape[0], b.shape[1]))
        for d, v in zip(dofs, x):
            out[d] = v
        return out[:, 0] if one else out

    def nbytes(self) -> int:
        tot = sum(g.a[:, :g.n].nbytes for g in self.XL + self.YU)
        tot += sum(a.nbytes for a in self.YL + self.XU + self.G
                   if a is not None)
        tot += sum(self.D[(k, k)].nbytes for k in range(self.stop))
        return int(tot + sum(a.nbytes for a in self.DL + self.DU
                             if a is not None))

    def ranks(self) -> np.ndarray:
        r = np.concatenate([(self.l1 - self.l0)[np.tril(self.kind == FAR, -1)],
                            (self.u1 - self.u0)[np.triu(self.kind == FAR, 1)]])
        return r


def from_assembled(asm, leaf: int = None, tol: float = None,
                   eta: float = None, clusters: Clusters = None, log=None):
    """``(BlockLU, Clusters)`` of an assembled compressed operator
    (``backends.hmat.AssembledH``), not yet factored. ``clusters`` are a
    geometry's, so every state of a body shares one set."""
    leaf = defaults.BLR_LEAF if leaf is None else int(leaf)
    tol = defaults.BLR_TOL if tol is None else float(tol)
    eta = defaults.ADMISSIBILITY_ETA if eta is None else float(eta)
    cl = (Clusters(asm.layout, asm._tree_cache, leaf) if clusters is None
          else clusters)
    kind, blocks = blocks_of(asm, cl, tol, eta)
    return BlockLU(cl.sizes, kind, blocks, tol, log=log), cl
