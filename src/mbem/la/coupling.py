"""What a body's boundary adds to the stiffness of a fault inside it, in low rank.

A solved fault in a body with a boundary sees its own slip twice: directly,
through the traction kernel of the region holding it (``la.faultkernel``),
and through the boundary, which the slip loads and which loads the fault
back. Reduced to the fault (``AssembledDense.condense``) the second is

    R = - A_fb  A_bb^-1  A_bf ,

``A_bf`` the columns of a unit slip in the boundary's rows, ``A_bb`` the
body's own operator and ``A_fb`` the rows of the fault's traction on the
boundary's unknowns. As a matrix on the fault ``R`` is full, and it is not
small (4 to 10 % of the stiffness of a megathrust's creeping reach). But it
is a product THROUGH THE BOUNDARY, and the boundary has a mesh:

* ``A_bf`` is the displacement of a slip at the boundary's collocation
  points, a smooth function of where the slip is except at the points a few
  elements from the fault. Its numerical rank is set by the boundary's mesh
  and does not grow as the fault is refined under it.
* ``A_fb`` likewise: the traction on the fault of the boundary's densities.

So with ``W`` an orthonormal basis of the slips the boundary can see and
``P`` one of the tractions it can return,

    R_k = P  M_k  W' ,        M_k = - (P' A_fb,k) A_bb,k^-1 (A_bf,k W) ,

and a state of the body costs its factorization and ONE SOLVE PER COLUMN OF
``W``, not per slip unknown; a stage of the stepper costs two thin products.
Nothing of the size (fault)^2 or (boundary) x (fault) is ever stored.

The bases are found by a randomized range finder: the blocks are swept once,
a chunk of the fault's elements at a time, against a random matrix, and once
more to project them, and both are then cut at ``tol`` of their largest
singular value; if the random vectors did not hold them with some to spare
there are twice as many and the sweep is made again. Every block of a
region is its geometry times ``mu^p`` times
a function affine in ``g = 1/(1 - nu)`` (docs/viscoelastic.md section 2.3:
``p = 0`` for a slip's displacement and a force's traction, ``1`` for a
slip's elastic traction), so the two ends of the run's range of ``g`` are
every state, as they are for the fault's own kernel.

Faults of ONE region, P0, inside it: a slip ON an interface (rule 21) has no
kernel of its own and all of its stiffness is the body's.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .. import defaults
from ..kernels import KERNEL_GT, KERNEL_HT, KERNEL_T
from ..kernels import basis as kb
from ..kernels import tri_nodal as tn
from .faultkernel import g_of, lam_of


def _columns(blk: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """``(R, 3 n)`` Cartesian columns of ``n`` nodes -> ``(R, 2 n)`` along
    the two rows of each node's ``frame`` ``(n, 2, 3)``. Six passes of
    array arithmetic: an ``einsum`` here is one thread and was the build."""
    b = blk.reshape(blk.shape[0], frame.shape[0], 3)
    out = np.empty((blk.shape[0], frame.shape[0], 2))
    for i in range(2):
        out[:, :, i] = (b[:, :, 0] * frame[:, i, 0] + b[:, :, 1] * frame[:, i, 1]
                        + b[:, :, 2] * frame[:, i, 2])
    return out.reshape(blk.shape[0], -1)


def _triangular(A: np.ndarray):
    """``(s, vt)`` of a tall ``A``: its singular values and right singular
    vectors, from the small triangular factor of its QR."""
    _u, s, vt = np.linalg.svd(np.linalg.qr(A, mode="r"))
    return s, vt


class Coupling:
    """The boundary's rows and columns of a region's solved faults.

    ``system`` is the model's (with the faults), ``layout`` the layout of the
    SAME model without them -- the body's own unknowns, the order ``state``
    answers in -- and ``coords`` the faults' ``FaultCoordinates`` in the
    order of the kernel. ``rhs`` holds the system's right-hand-side terms
    state by state (``equations.rhs_terms_at``): the prescribed data's
    traction on the faults is read with the walls' data of each state.

    ``tol`` is where the bases are cut and ``rank`` the random vectors the
    search starts with. After the build: ``P`` and ``W`` ``(n, r)``,
    orthonormal, ``state``, and ``spectra``, the singular values the cut
    was made in.
    """

    def __init__(self, system, layout, coords, eps, g_range, mu_ref,
                 rhs, tol: float = None, rank: int = None, seed: int = 0,
                 log=None):
        t0 = time.perf_counter()
        self.tol = defaults.COUPLING_TOL if tol is None else float(tol)
        rank = defaults.COUPLING_RANK if rank is None else int(rank)
        self.coords = tuple(coords)
        self.n = int(self.coords[-1].stop)
        self.N = int(layout.n_unknowns)
        self.g = (float(min(g_range)), float(max(g_range)))
        if self.g[1] - self.g[0] < 1.0e-9:
            self.g = self.g[:1]
        self.mu_ref = float(mu_ref)
        solved = [f for _r, f in system.traction_faults()]
        faults = {id(f): c for f, c in zip(solved, self.coords)}
        self._by_fault = {id(c): f for f, c in zip(solved, self.coords)}
        slots = {id(system.layout.slot(f, "b")) for f in solved}
        self._arrays = kb.MeshArrays()
        self._eps = eps

        def body(slot):
            return layout.slot(slot.patch, slot.kind)

        # the three kinds of term a solved fault has with the boundary
        self._slip, self._trac, self._data = [], [], []
        for t in system.terms:
            col, row = id(t.col) in slots, id(t.row) in slots
            if col and row:
                if t.scale != 1.0 or t.kernel != KERNEL_HT:
                    raise ValueError("a solved fault's own traction is "
                                     "+1 x the slip kernel; this system "
                                     "says otherwise")
            elif col:
                if t.kernel != KERNEL_T or t.col_op is not None:
                    raise NotImplementedError(f"slip column of kernel {t.kernel}")
                self._slip.append((t, faults[id(t.source_patch)], body(t.row)))
            elif row:
                if t.kernel not in (KERNEL_HT, KERNEL_GT):
                    raise NotImplementedError(f"fault row of kernel {t.kernel}")
                self._trac.append((t, faults[id(t.field_patch)], body(t.col)))
        for i, rt in enumerate(system.rhs_terms):
            if id(rt.row) in slots:
                vec = np.stack([np.asarray(r[i].vector, float) for r in rhs], 1)
                if np.all(vec == vec[:, :1]):
                    vec, which = vec[:, :1], np.zeros(len(rhs), int)
                else:
                    which = np.arange(len(rhs))
                self._data.append((rt, faults[id(rt.field_patch)], vec, which))
        self._datum = np.zeros(self.n)
        for rv in system.rhs_values:
            if id(rv.row) in slots:
                c = faults[id(rv.row.patch)]
                self._datum[c.start:c.stop] += c.coordinates(
                    np.asarray(rv.vector, float).reshape(-1, 3))

        G = len(self.g)
        self._rv = np.zeros((G, 2, self.n, len(rhs)))     # [g][mu power] data
        r, data = min(rank, self.n), True
        while True:
            W0, P0, B, E = self._sweep(r, seed, data)
            data = False
            # cut both at tol of their largest singular value: the small
            # triangular factor has the singular values and the side wanted
            sw, vt = _triangular(B.reshape(G * self.N, r))
            scale = np.r_[np.full(G, self.mu_ref), np.ones(G)][:, None, None]
            sp, u = _triangular(
                (scale * E).transpose(1, 0, 2).reshape(r, 2 * G * self.N).T)
            rw = max(1, int((sw > self.tol * sw[0]).sum())) if sw[0] > 0.0 else 1
            rp = max(1, int((sp > self.tol * sp[0]).sum())) if sp[0] > 0.0 else 1
            # held with vectors to spare, or the fault itself is spanned
            if r >= self.n or max(rw, rp) <= r - defaults.COUPLING_SPARE:
                break
            r = min(2 * r, self.n)
            if log is not None:
                log(f"boundary coupling: {r // 2} random vectors do not "
                    f"hold it at {self.tol:g}; again with {r}")
        u = u.T
        # the two spectra, for whoever asks what another tol would keep
        self.spectra = (sw / sw[0] if sw[0] > 0.0 else sw,
                        sp / sp[0] if sp[0] > 0.0 else sp)
        self.W = W0 @ vt[:rw].T
        self.P = P0 @ u[:, :rp]
        self._B = B @ vt[:rw].T
        self._E = np.matmul(u[:, :rp].T, E)
        self._arrays = None
        self.report = {
            "slip_rank": rw, "traction_rank": rp, "sketch": r,
            "tol": self.tol, "boundary_unknowns": self.N,
            "slip_unknowns": self.n, "build_s": time.perf_counter() - t0,
            "bytes": int(self.W.nbytes + self.P.nbytes + self._B.nbytes
                         + self._E.nbytes)}
        if log is not None:
            log(f"boundary coupling: {self.n} slip unknowns through "
                f"{self.N} of the boundary's; rank {rw} (slip) and {rp} "
                f"(traction) at {self.tol:g}, built in "
                f"{self.report['build_s']:.0f} s")

    # -- the sweep --------------------------------------------------------

    def _sweep(self, r: int, seed: int, data: bool):
        """Two passes over the blocks with ``r`` random vectors: the two
        ranges ``(n, r)``, orthonormal, and the blocks in them -- the slip's
        boundary rows ``B`` ``(G, N, r)`` and the boundary's traction rows
        ``E`` ``(2 G, r, N)``, the slip kernel's at each ``g`` and then the
        force kernel's. With ``data`` the first pass also puts the
        prescribed data's traction on the faults in ``self._rv``."""
        G = len(self.g)
        rng = np.random.default_rng(seed)
        om_w = rng.standard_normal((G, self.N, r))
        om_p = rng.standard_normal((2 * G, self.N, r))
        Yw, Yp = np.zeros((self.n, r)), np.zeros((self.n, r))
        for c, sel, at in self._chunks():
            for x in range(G):
                for (t, _c, row), blk in self._slip_blocks(c, sel, x):
                    Yw[at] += blk.T @ om_w[x, row.offset:row.stop]
                for (t, _c, col), blk in self._trac_blocks(c, sel, x):
                    if t.kernel == KERNEL_HT:
                        Yp[at] += self.mu_ref * (blk @ om_p[x, col.offset:col.stop])
                    else:
                        Yp[at] += blk @ om_p[G + x, col.offset:col.stop]
                if not data:
                    continue
                for (rt, _c, vec, which), blk in self._data_blocks(c, sel, x):
                    p = 1 if rt.kernel == KERNEL_HT else 0
                    self._rv[x, p][at] += (blk @ vec)[:, which]
        del om_w, om_p
        W0, P0 = np.linalg.qr(Yw)[0], np.linalg.qr(Yp)[0]
        del Yw, Yp
        B = np.zeros((G, self.N, r))
        E = np.zeros((2 * G, r, self.N))
        for c, sel, at in self._chunks():
            for x in range(G):
                for (t, _c, row), blk in self._slip_blocks(c, sel, x):
                    B[x, row.offset:row.stop] += blk @ W0[at]
                for (t, _c, col), blk in self._trac_blocks(c, sel, x):
                    k = x if t.kernel == KERNEL_HT else G + x
                    E[k, :, col.offset:col.stop] += P0[at].T @ blk
        return W0, P0, B, E

    def _chunks(self):
        """``(coords, elements, slice of coordinates)``: the faults' free
        elements, a chunk at a time."""
        step = max(1, int(defaults.COUPLING_CHUNK))
        for c in self.coords:
            for i in range(0, c.free.size, step):
                j = min(c.free.size, i + step)
                yield c, slice(i, j), slice(c.start + 2 * i, c.start + 2 * j)

    def _material(self, x):
        return 1.0, lam_of(1.0, self.g[x])

    def _slip_blocks(self, c, sel, x):
        """Each boundary row block of a unit slip of a chunk, in
        coordinates: ``(3 N_q, 2 n_chunk)``, at unit shear modulus."""
        f = self._by_fault[id(c)]
        tv, nrm = self._arrays.source_arrays(f.mesh)
        e = kb.resolve_patch_eps(self._eps, f)
        el, frame = c.free[sel], c.frame[sel]
        coef = kb.t_coeffs(*self._material(x))
        for item in self._slip:
            t, tc, _row = item
            if tc is not c:
                continue
            x_c = self._arrays.collocation_points(t.field_patch)
            blk = tn.t_matrix_direct(
                x_c, np.ascontiguousarray(tv[el]),
                np.ascontiguousarray(nrm[el]), np.ascontiguousarray(e[el]),
                *coef, 0)
            yield item, t.scale * _columns(blk, frame)

    def _rows(self, c, sel, kernel, source, x):
        """The traction rows ``(2 n_chunk, 3 n_dofs)`` of a chunk of a
        fault's nodes on a nodal density of ``source``."""
        f = self._by_fault[id(c)]
        el, frame = c.free[sel], c.frame[sel]
        x_c = np.ascontiguousarray(self._arrays.collocation_points(f)[el])
        n_c = np.ascontiguousarray(self._arrays.collocation_normals(f)[el])
        mu, lam = self._material(x)
        blk = kb.assemble_traction_matrix(
            x_c, n_c, source.mesh, SimpleNamespace(mu=mu, lam=lam),
            kb.resolve_patch_eps(self._eps, source), arrays=self._arrays,
            order=source.order,
            kind="slip" if kernel == KERNEL_HT else "force",
            ids=source.column_ids(), n_cols=source.n_dofs)
        return _columns(blk.T, frame).T

    def _trac_blocks(self, c, sel, x):
        for item in self._trac:
            t, tc, _col = item
            if tc is c:
                yield item, t.scale * t.col_block(
                    self._rows(c, sel, t.kernel, t.source_patch, x))

    def _data_blocks(self, c, sel, x):
        for item in self._data:
            rt, tc, _vec, _which = item
            if tc is c:
                yield item, rt.scale * self._rows(c, sel, rt.kernel,
                                                  rt.source_patch, x)

    # -- a state ----------------------------------------------------------

    def _weights(self, mu, lam) -> np.ndarray:
        if len(self.g) == 1:
            return np.ones(1)
        g = g_of(mu, lam)
        lo, hi = self.g
        if not (lo - 1.0e-9 <= g <= hi + 1.0e-9):
            raise ValueError(f"g = {g:.6g} is outside this coupling's range "
                             f"{self.g}")
        w = (hi - g) / (hi - lo)
        return np.array([w, 1.0 - w])

    def state(self, mu: float, lam: float, k: int):
        """``(B, E, held)`` at the faults' region's ``(mu, lam)``, state
        ``k`` of the data: ``A_bf W`` ``(N, r_w)``, ``P' A_fb`` ``(r_p,
        N)``, and ``(n,)`` the right-hand side of the faults' own rows --
        the traction the prescribed data put on them, with the sign the
        system's ``b`` has."""
        w = self._weights(mu, lam)
        G = len(self.g)
        B = np.tensordot(w, self._B, axes=1)
        E = (float(mu) * np.tensordot(w, self._E[:G], axes=1)
             + np.tensordot(w, self._E[G:], axes=1))
        held = self._datum + np.tensordot(
            w, float(mu) * self._rv[:, 1, :, k] + self._rv[:, 0, :, k], axes=1)
        return B, E, held
