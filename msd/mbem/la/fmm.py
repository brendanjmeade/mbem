"""Reference Chebyshev black-box FMM (bbFMM) for one compressed pair.

``PairFMM`` is interface-compatible with ``la/hop.PairCompressed``: the same
``matvec(coeffs, x)``, the same DOF layout (3 per source element,
``3*elem + component``) and the same meaning of ``coeffs`` -- the material
coefficient vector of the kernel's geometry-only basis
(``kernels/basis.u_coeffs``/``t_coeffs``) -- so ``backends/hmat.AssembledH``
drops it in with no change above it.

SCOPE. This is the REFERENCE implementation: correctness first, performance
later. Pure numpy, no numba in the far field, no blocked GEMM, no M2L
compression and no shared M2L table; every M2L is evaluated where it is
used. Everything built after it will be gated against it, so it is written
to be obviously right rather than fast -- on a model small enough to hold a
dense reference it costs MORE than the dense operator it approximates, and
that is the intended trade.

WHAT IS APPROXIMATED, AND WHAT IS NOT. Only the kernel is interpolated; a
source triangle is never collapsed to a point:

    U list  res(a) x res(b)   exact, ``tri_kernels.{u,t}_matrix_direct``
    V list  sub(a) x sub(b)   M2L, interpolated on both sides
    W list  res(a) x sub(b)   M2P, interpolated on the source side
    X list  sub(a) x res(b)   P2L, exact source integrals AT the local
                              nodes, interpolated on the target side

P2M integrates the Chebyshev weight over the source triangle exactly (a
Duffy-mapped Gauss rule of the degree that weight has), so the far field
carries no one-point-quadrature error. Collapsing an element to its centroid
instead costs O((h/R)^2), which at a V pair's separation (R ~ 2 box edges,
h up to one box edge) is percent-level and no order p would cure it; for the
same reason X integrates its sources with the analytic triangle kernels.

eps IS A WEIGHT, NOT A KERNEL. Mollification expands as
``K(eps) = K0 + eps_j^2 K1 + O(eps^4)`` with both terms translation-invariant
and K1 homogeneous of K0's degree minus 2, so the far field is TWO passes of
one machinery: pass 0 carries the source charge, pass 1 the same charge
times ``eps_j^2`` and a kernel that differs only in a radial power and a
coefficient triple (``_far_params``). Neither is a special case of the
other. One pass alone drops the eps^2 term; a per-leaf scalar eps is not an
option (15.8x spread of eps inside one leaf), which is why eps stays a
per-source-element weight. The U and X lists are exact in eps regardless.

THE INTERPOLATION DOMAIN, and what it costs. An element protrudes from its
box by up to 0.53 box edges under the placement rule
(``OCTREE_PLACEMENT_SAFETY``), so a source can lie outside its own box and
Chebyshev interpolation there is extrapolation. The default domain is
therefore the box's ENLARGED extent (``Octree.extents``), per role: the
source domain is the bounding box of the triangles the box holds, the target
domain that of the collocation points it holds (always inside the nominal
cube, targets being points). Containment then holds at every level and every
order, which the gate pins as max|xhat| <= 1.

What that costs is the shared M2L table: with a domain per box the kernel
between two node sets is not a function of their offset, so nothing is
shared -- the economy a production bbFMM exists for. Four domains are
offered, and only the extent is NOT shareable:

    domain="extent"                per-box bounding box; contains every
                                   source; nothing shared
    domain="cube", inflate=f       the nominal cube scaled by one factor f,
                                   the same at every box and level, so the
                                   node lattice stays translation-invariant
                                   and the M2L table stays shared; f large
                                   enough contains the protrusion too
    domain="canonical"             per-box extents for P2M/L2P, plus a
                                   per-box change of basis onto the nominal
                                   cube lattice around M2L alone

``inflate`` is one factor or ``(f_source, f_target)``. Targets are
collocation POINTS and are always inside their own cube, so only the source
domain ever has to grow; splitting the two pays the accuracy for the
protrusion on one side instead of both. What limits f is not accuracy in
the abstract but the V list: two non-adjacent boxes are 2 edges apart
centre to centre and their node lattices span +-f/2 edges, so the M2L
separation is ``(2 - f_s/2 - f_t/2) * cos(pi/2p)`` edges and vanishes near
f = 2 -- the interpolation domain would then reach the singularity it is
interpolating across.

``domain="canonical"`` keeps containment exact where the geometry enters
(P2M, L2P, W and X are all on the extents) and moves only M2L onto the
shared cube lattice, through two separable p^3 x p^3 transforms per box. It
does not buy anything WHERE THE PROBLEM IS: the extent lattice reproduces
every polynomial of degree < p per axis and a cube Chebyshev weight IS one,
so extent-P2M composed with the transform is cube-P2M identically --
measured at 3e-15 on a pair with no W and no X, at every p through 12. Its
M2L is the cube's, so it inherits the cube's stall. What it keeps is the
extent's W and X margins, which is not nothing -- 2.6x better than the cube
on the fault-zone OPERATOR, where W and X are 144 box pairs each -- but it
is still 4.8x worse than the extent and still over ``FMM_OPERATOR_PARITY``.
The gate pins the identity so it is not rediscovered as a third option.

The default stays on the extents because containment is what holds at any
protrusion and any p, and because the choice is the far field's to make once
the shared table is built. ONE PAIR AT ONE p DOES NOT SETTLE WHICH IS MORE
ACCURATE, and taken at face value it points the wrong way. The cube wins the
fault-zone pair ``verify_fmm`` [g] prints, by 5x at p = 8. On the OPERATOR
that pair belongs to it is 6-30x worse and over ``FMM_OPERATOR_PARITY``, and
on topo_inclusion -- 0.53 box edges of protrusion against the fault zone's
0.33 -- its convergence in p STALLS while the extent's does not
(``interface_side <- host_top``, far-isolated at p = 4, 6, 8: cube 5.2e-3,
7.5e-4, 4.1e-4 against the extent's 6.6e-3, 4.7e-4, 3.5e-5). The stall is
M2L, not P2L: replacing the X list by its exact value moves the cube's error
by 1-3 % from p = 6 up, so what fails is the P2M EXTRAPOLATION of a
protruding source into the multipole -- the thing the extent exists to
prevent, arriving at the protrusion the trunk's target topology has.

THE TREE IS PER PAIR OR SHARED. With no ``geom``, ``PairFMM`` builds one
``Octree`` over the union of its own field points and source elements: the
self-contained thing to do, and what the pair-level gates use. On a model of
many small patches that union has no far field at all (measured: every pair
of the fault-zone model at refine 1 is 100 % U list), so an operator-level
number needs ONE tree over every mesh of the model, which :class:`FmmTree`
is -- each pair restricts the shared U/V/W/X to its own two element ranges,
and the near/far partition becomes a property of the model rather than of
the pair. Which of the two was used is part of any number reported from here.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .. import defaults
from ..kernels import KERNEL_T, KERNEL_U, kernel_n_basis
from ..kernels import basis as kb
from ..kernels import tri_kernels as tk
from .octree import InteractionLists, Octree


# ---------------------------------------------------------------------
# Chebyshev interpolation (Fong-Darve form)
# ---------------------------------------------------------------------

def _cheb_nodes(p: int) -> np.ndarray:
    """The p Chebyshev points of the first kind on [-1, 1]."""
    return np.cos(np.pi * (2.0 * np.arange(p) + 1.0) / (2.0 * p))


def _cheb_T(x: np.ndarray, p: int) -> np.ndarray:
    """(len(x), p) Chebyshev polynomials T_0..T_{p-1}, by recurrence."""
    x = np.asarray(x, dtype=float).ravel()
    T = np.empty((x.size, p))
    T[:, 0] = 1.0
    if p > 1:
        T[:, 1] = x
    for n in range(2, p):
        T[:, n] = 2.0 * x * T[:, n - 1] - T[:, n - 2]
    return T


def _cheb_weights(xhat: np.ndarray, p: int) -> np.ndarray:
    """Interpolation weights ``S(x, x_k)``, shape (len(x), p).

    ``S(x, x_k) = (1 + 2 sum_{n=1}^{p-1} T_n(x_k) T_n(x)) / p`` is the
    Lagrange weight of node k at Chebyshev points of the first kind, so
    ``sum_k S(x, x_k) f(x_k)`` reproduces every polynomial of degree < p.
    Nothing clamps ``x`` to [-1, 1]: a point outside its own box has to
    show up as the extrapolation it is (``_Stencil.max_xhat``).
    """
    T = _cheb_T(xhat, p)
    Tk = _cheb_T(_cheb_nodes(p), p)
    return (1.0 + 2.0 * (T[:, 1:] @ Tk[:, 1:].T)) / p


def _weights3(points: np.ndarray, center: np.ndarray, half: np.ndarray,
              p: int) -> tuple:
    """``(S3, max|xhat|)`` with S3 (len(points), p^3) tensor-product weights.

    Node index is ``(a*p + b)*p + c`` over the x, y, z Chebyshev grids,
    which is C order on (p, p, p) -- what ``_separable`` reshapes to.
    """
    xh = (np.asarray(points, dtype=float) - center) / half
    S = [_cheb_weights(xh[:, d], p) for d in range(3)]
    S3 = (S[0][:, :, None, None] * S[1][:, None, :, None]
          * S[2][:, None, None, :])
    return S3.reshape(xh.shape[0], -1), float(np.abs(xh).max())


def _separable(A3: list, C: np.ndarray, contract: int) -> np.ndarray:
    """Apply a separable operator to ``C`` of shape (p, p, p, m).

    Both interpolation domains are axis-aligned boxes, so M2M and L2L
    factor over the axes: three (p, p) matrices and an O(p^4) apply
    instead of one p^3 x p^3 matrix and an O(p^6) one. ``contract=0``
    sums over each ``A``'s first index (M2M, child nodes anterpolated
    onto the parent grid), ``contract=1`` over its second (L2L, the
    parent's interpolant evaluated at the child's nodes).
    """
    out = C
    for axis, A in enumerate(A3):
        out = np.moveaxis(np.tensordot(A, out, axes=([contract], [axis])),
                          0, axis)
    return out


# ---------------------------------------------------------------------
# Source-triangle quadrature for P2M
# ---------------------------------------------------------------------

def _p2m_quad_order(p: int) -> int:
    """Gauss points per direction that integrate the P2M weight EXACTLY.

    The weight is degree p-1 per axis, so degree 3(p-1) in the point; the
    Duffy map to the unit square adds one degree in u through its
    Jacobian, and n Gauss points are exact through degree 2n-1.
    """
    return int(np.ceil((3 * p - 1) / 2))


def _duffy_rule(n: int) -> tuple:
    """``(bary, w)``: an n x n Duffy-mapped rule on the reference triangle.

    ``bary`` is (n^2, 3) barycentric, ``w`` (n^2,) sums to 1, so
    ``area * sum_q w_q g(y_q)`` is the integral of g over the triangle.
    """
    t, wt = np.polynomial.legendre.leggauss(n)
    s = 0.5 * (t + 1.0)
    ws = 0.5 * wt
    u = s[:, None] * np.ones(n)[None, :]
    v = np.ones(n)[:, None] * s[None, :]
    w = 2.0 * (ws[:, None] * u) * ws[None, :]
    bary = np.stack([1.0 - u, u * (1.0 - v), u * v], axis=-1)
    return bary.reshape(-1, 3), w.reshape(-1)


# ---------------------------------------------------------------------
# Point-pair far-field kernels: the two terms of K = K0 + eps^2 K1
# ---------------------------------------------------------------------
#
# Both are the point limit of the geometry basis of kernels/tri_kernels --
# every triangle moment replaced by its one-point value at d = x - y, R_eps
# by r -- expanded in t = eps^2 about t = 0 and regrouped by material
# coefficient. The element's own weight is carried by the P2M integral.
#
#   U (KERNEL_U, coefficients g1, g2, g3; charge = the traction DOF f):
#       B1 = delta/R_eps, B2 = eps^2 delta/R_eps^3, B3 = d d/R_eps^3
#     K0 = g1 f_i / r            + g3 d_i (d.f) / r^3
#     K1 = (g2 - g1/2) f_i / r^3 - (3/2) g3 d_i (d.f) / r^5
#
#   T (KERNEL_T, coefficients c1..c6; charge M_jm = b_j n_m, which takes
#   the source normal out of the kernel and leaves it a function of d):
#       P1 = delta_ik V3_m, P2 = delta_im V3_k + delta_km V3_i - 3 T3[5],
#       P3 = eps^2 delta_ik V5_m; N[P] b = (n.b) tr_i(P) and
#       R[P] b = P_ijm S_jm with S = M + M^T, so with tr(M) = n.b
#     K0 = a1 d_i tr(M)/r^3 + a2 (S d)_i/r^3 + a3 d_i (d.M.d)/r^5
#          a1 = c1 + c2 + 2 c5,  a2 = c4 + c5,  a3 = -6 c5
#     K1 = a1 d_i tr(M)/r^5 + a2 (S d)_i/r^5 + a3 d_i (d.M.d)/r^7
#          a1 = -3(c1 - c2)/2 + c3 - 3 c5,  a2 = -3(c4 + c5)/2 + c6,
#          a3 = 15 c5
#
# A pass is therefore fully described by (radial power, coefficients) and
# ONE evaluator runs both; pass 1 differs only in its charge weight eps^2.
# The lam/mu pairing lives entirely in c1..c6 (kernels/basis.t_coeffs), so
# a swap there is invisible at nu = 1/4 here as everywhere else.


def _far_params(kernel: str, coeffs: np.ndarray) -> list:
    """``[(power, coefficients...)]``, one entry per eps pass."""
    c = np.asarray(coeffs, dtype=float)
    if c.shape != (kernel_n_basis(kernel),):
        raise ValueError(f"coeffs shape {c.shape} != "
                         f"({kernel_n_basis(kernel)},) for kernel {kernel!r}")
    if kernel == KERNEL_U:
        g1, g2, g3 = c
        return [(1, g1, g3), (3, g2 - 0.5 * g1, -1.5 * g3)]
    c1, c2, c3, c4, c5, c6 = c
    return [(3, c1 + c2 + 2.0 * c5, c4 + c5, -6.0 * c5),
            (5, -1.5 * (c1 - c2) + c3 - 3.0 * c5,
             -1.5 * (c4 + c5) + c6, 15.0 * c5)]


def _far_apply(kernel: str, xt: np.ndarray, ys: np.ndarray,
               charges: list, params: list) -> np.ndarray:
    """Field at targets ``xt`` (nt, 3) from point charges at ``ys`` (ns, 3).

    ``charges[i]`` is pass i's charge -- (ns, 3, k) for U, (ns, 3, 3, k)
    for T -- and ``params[i]`` its ``(power, coefficients...)``. The
    passes share one geometry pass: ``d`` and the radial weights are
    formed once. Returns (nt, 3, k).
    """
    d = xt[:, None, :] - ys[None, :, :]
    inv = 1.0 / np.sqrt(np.einsum("tsi,tsi->ts", d, d))
    inv2 = inv * inv

    def radial(k):                      # 1/r^k, k odd
        out = inv
        for _ in range((k - 1) // 2):
            out = out * inv2
        return out

    out = np.zeros((xt.shape[0], 3, charges[0].shape[-1]))
    for prm, q in zip(params, charges):
        wk = radial(prm[0])
        wk2 = wk * inv2
        if kernel == KERNEL_U:
            _, b1, b2 = prm
            out += b1 * np.tensordot(wk, q, axes=([1], [0]))
            dq = np.einsum("tsj,sjk->tsk", d, q)
            out += b2 * np.einsum("ts,tsi,tsk->tik", wk2, d, dq)
        else:
            _, a1, a2, a3 = prm
            tr = q[:, 0, 0] + q[:, 1, 1] + q[:, 2, 2]              # (ns, k)
            out += a1 * np.einsum("ts,tsi,sk->tik", wk, d, tr)
            S = q + np.swapaxes(q, 1, 2)
            out += a2 * np.einsum("ts,simk,tsm->tik", wk, S, d)
            dmd = np.einsum("tsj,sjmk,tsm->tsk", d, q, d)
            out += a3 * np.einsum("ts,tsi,tsk->tik", wk2, d, dmd)
    return out


# ---------------------------------------------------------------------
# Shared geometry: one octree over a set of meshes
# ---------------------------------------------------------------------

class _Stencil:
    """Everything an :class:`FmmTree` needs at one interpolation order p.

    Kept per p, not per tree: the U and T kernels run at different orders
    (``FMM_ORDER_U``, ``FMM_ORDER_T``) over the same geometry.
    """

    def __init__(self, geom: "FmmTree", p: int):
        self.p = p
        tree = geom.tree
        nb = len(tree.boxes)
        p3 = p ** 3
        z = _cheb_nodes(p)
        # One number per PASS, because they fail for different reasons: a
        # source protrudes, a collocation point never does, and a child's
        # nodes leave its parent only through the half-width floor.
        self.xhat = {"p2m": 0.0, "l2p": 0.0, "m2m": 0.0, "l2l": 0.0}
        # Node index layout, matching _weights3's C order on (p, p, p).
        grid = np.stack(np.meshgrid(z, z, z, indexing="ij"),
                        axis=-1).reshape(p3, 3)

        def lattice(dom):
            center, half = dom
            return center[:, None, :] + half[:, None, :] * grid[None, :, :]

        self.src_nodes = lattice(geom.src_dom)
        self.tgt_nodes = lattice(geom.tgt_dom)

        # P2M: the Chebyshev weight integrated over each source triangle.
        # L2P: the weight at each collocation point.
        bary, wq = _duffy_rule(_p2m_quad_order(p))
        self.p2m = np.zeros((tree.n, p3))
        self.l2p = np.zeros((tree.n, p3))
        for bi in range(nb):
            held = tree.elements_of(bi)
            if not held.size:
                continue
            c_s, h_s = geom.src_dom[0][bi], geom.src_dom[1][bi]
            for e in held:
                S3, over = _weights3(bary @ geom.verts[e], c_s, h_s, p)
                self.p2m[e] = geom.areas[e] * (wq[:, None] * S3).sum(axis=0)
                self._seen("p2m", over)
            S3, over = _weights3(geom.centroids[held], geom.tgt_dom[0][bi],
                                 geom.tgt_dom[1][bi], p)
            self.l2p[held] = S3
            self._seen("l2p", over)

        # M2M / L2L: the child's nodes expressed in the parent's domain.
        self.m2m = [None] * nb
        self.l2l = [None] * nb
        for bi in range(nb):
            pj = int(tree.parent[bi])
            if pj < 0:
                continue
            self.m2m[bi] = self._axis_maps(geom.src_dom, bi, geom.src_dom, pj,
                                           z, p, "m2m")
            self.l2l[bi] = self._axis_maps(geom.tgt_dom, bi, geom.tgt_dom, pj,
                                           z, p, "l2l")

        # M2L on the shared cube lattice: the per-box change of basis into
        # it and back. Only the canonical domain has one; the others run
        # M2L on the same nodes P2M and L2P already use.
        self.src_m2l_nodes = self.src_nodes
        self.tgt_m2l_nodes = self.tgt_nodes
        self.e2c_src = self.e2c_tgt = None
        if geom.cube_dom is not None:
            self.xhat["m2c"] = self.xhat["c2l"] = 0.0
            self.src_m2l_nodes = lattice(geom.cube_dom)
            self.tgt_m2l_nodes = self.src_m2l_nodes
            self.e2c_src = [self._axis_maps(geom.src_dom, bi, geom.cube_dom,
                                            bi, z, p, "m2c")
                            for bi in range(nb)]
            self.e2c_tgt = [self._axis_maps(geom.tgt_dom, bi, geom.cube_dom,
                                            bi, z, p, "c2l")
                            for bi in range(nb)]

    def _seen(self, pass_: str, value: float) -> None:
        self.xhat[pass_] = max(self.xhat[pass_], float(value))

    @property
    def max_xhat(self) -> float:
        """The worst |xhat| over every pass -- what the gates pin."""
        return max(self.xhat.values())

    def _axis_maps(self, dom, bi: int, into, ci: int, z, p: int,
                   pass_: str) -> list:
        """Box ``bi``'s nodes in ``dom``, evaluated in box ``ci``'s ``into``
        domain: three (p, p) axis matrices, ``_separable``'s operands."""
        center, half = dom
        c_in, h_in = into
        out = []
        for d in range(3):
            xh = (center[bi, d] + half[bi, d] * z - c_in[ci, d]) / h_in[ci, d]
            self._seen(pass_, np.abs(xh).max())
            out.append(_cheb_weights(xh, p))
        return out


class FmmTree:
    """One adaptive octree over a set of meshes, with the FMM's geometry.

    Element indices are the concatenation of the meshes in the order given
    (``range_of``), so a pair restricts the shared interaction lists to its
    own two ranges. Tree, lists, interpolation domains and every Chebyshev
    operator are material-, kernel- and eps-free, so one instance serves
    every pair of a model -- which is what makes the far-field partition a
    property of the model and not of the pair.
    """

    def __init__(self, meshes, ncrit: int = defaults.OCTREE_NCRIT,
                 placement_safety: float = defaults.OCTREE_PLACEMENT_SAFETY,
                 domain: str = "extent", inflate=1.0,
                 arrays: kb.MeshArrays | None = None):
        if domain not in ("extent", "cube", "canonical"):
            raise ValueError(f"unknown interpolation domain {domain!r}")
        f = ((float(inflate), float(inflate)) if np.isscalar(inflate)
             else tuple(float(v) for v in inflate))
        if len(f) != 2 or min(f) < 1.0:
            raise ValueError("inflate is one factor >= 1, or (source, target)")
        a = arrays if arrays is not None else kb.MeshArrays()
        self.domain = domain
        self.inflate = f
        self._range: dict = {}
        cen, siz, verts, areas, off = [], [], [], [], 0
        for m in meshes:
            if id(m) in self._range:
                continue
            tv, _ = a.source_arrays(m)
            _, ar = m.normals_and_areas()
            e = np.stack([np.linalg.norm(tv[:, 1] - tv[:, 0], axis=1),
                          np.linalg.norm(tv[:, 2] - tv[:, 1], axis=1),
                          np.linalg.norm(tv[:, 0] - tv[:, 2], axis=1)], axis=1)
            self._range[id(m)] = (off, off + m.n_triangles)
            off += m.n_triangles
            cen.append(a.field_points(m))
            siz.append(e.max(axis=1))
            verts.append(tv)
            areas.append(ar)
        self.centroids = np.ascontiguousarray(np.vstack(cen))
        self.verts = np.ascontiguousarray(np.vstack(verts))
        self.areas = np.concatenate(areas)
        self.tree = Octree(self.centroids, np.concatenate(siz), self.verts,
                           ncrit=ncrit, placement_safety=placement_safety)
        self.lists = InteractionLists(self.tree)
        self._cube = np.array([np.concatenate(self.tree.cube(b))
                               for b in range(len(self.tree.boxes))])
        self.src_dom = self._domains(self.verts, f[0])
        self.tgt_dom = self._domains(self.centroids, f[1])
        # The lattice M2L is stated on when it is not the role's own:
        # translation-invariant, hence shared, hence the canonical frame.
        # Always the nominal cube -- ``inflate`` scales the extents P2M and
        # L2P work on, never the frame the shared table is stated in.
        self.cube_dom = (self._cube_domain(1.0) if domain == "canonical"
                         else None)
        self._stencils: dict = {}

    def _cube_domain(self, factor: float) -> tuple:
        cube = self._cube
        return 0.5 * (cube[:, :3] + cube[:, 3:]), \
            factor * 0.5 * (cube[:, 3:] - cube[:, :3])

    def _domains(self, geo: np.ndarray, factor: float) -> tuple:
        """``(center, half)`` per box for one role, scaled by ``factor``.

        ``domain="extent"`` is the bounding box of what the box holds --
        the only domain a protruding source is inside; ``"cube"`` is the
        nominal cube, on which the M2L table is shared and P2M
        extrapolates; ``"canonical"`` is the extent, the cube entering
        only through :attr:`cube_dom`. Scaling by ONE factor per role
        preserves whatever translation invariance the base domain had,
        because the cubes at a level are congruent.

        Every half-width is floored at a fraction of the box's own cube
        edge (``FMM_MIN_HALF_OVER_EDGE``). A flat patch gives a box zero
        extent across its plane, and a zero (or near-zero) width would
        divide rounding in the quadrature points by itself; flooring only
        ENLARGES a domain, so every point it must contain is still inside.
        The floor is relative to the cube because an absolute one would
        put a child's nodes outside its parent's domain, which is where
        M2M and L2L evaluate them. Even the relative floor moves them
        slightly out when BOTH boxes are floored and their extent centres
        differ -- 1.0005 on topo_inclusion, where 423 of 783 boxes hold a
        planar patch and would otherwise divide 0 by 0 -- so the floor is
        load-bearing, and small enough that what it costs is 5e-4 of a
        half-width. Raising it to 1e-3 costs 0.44 (``verify_fmm`` [g]).
        """
        if self.domain == "cube":
            return self._cube_domain(factor)
        lo, hi = self.tree.extents(geo)
        floor = defaults.FMM_MIN_HALF_OVER_EDGE * (self._cube[:, 3]
                                                   - self._cube[:, 0])
        half = np.maximum(0.5 * (hi - lo), floor[:, None])
        return 0.5 * (lo + hi), factor * half

    def containment_factors(self) -> tuple:
        """``(f_source, f_target)``: the smallest uniform cube inflation
        that puts every source VERTEX, and every collocation point, inside
        its own box's domain -- what ``inflate`` has to be for P2M and L2P
        to interpolate rather than extrapolate. ``f_target`` is 1 by
        construction, a collocation point being placed by its own box.

        Read it with the price: the V-list node separation is
        ``(2 - (f_source + f_target) / 2) cos(pi/2p)`` box edges, so a
        factor near 2 leaves the two lattices touching.
        """
        cen = 0.5 * (self._cube[:, :3] + self._cube[:, 3:])
        half = 0.5 * (self._cube[:, 3:] - self._cube[:, :3])
        f = [0.0, 0.0]
        for bi in range(len(self.tree.boxes)):
            held = self.tree.elements_of(bi)
            if not held.size:
                continue
            for i, geo in enumerate((self.verts[held].reshape(-1, 3),
                                     self.centroids[held])):
                f[i] = max(f[i], float(np.abs((geo - cen[bi])
                                              / half[bi]).max()))
        return tuple(f)

    def range_of(self, mesh) -> tuple:
        """``(start, stop)`` of one mesh's elements in the tree's indexing."""
        try:
            return self._range[id(mesh)]
        except KeyError:
            raise KeyError("mesh is not in this FmmTree") from None

    def stencil(self, p: int) -> _Stencil:
        st = self._stencils.get(p)
        if st is None:
            st = _Stencil(self, p)
            self._stencils[p] = st
        return st

    def summary(self) -> str:
        c = self.lists.counts()
        f = (f" inflate {self.inflate[0]:g}/{self.inflate[1]:g}"
             if self.inflate != (1.0, 1.0) else "")
        return (f"FmmTree {self.domain!r}{f}: {self.tree.summary()}; "
                f"U {c['U']} V {c['V']} W {c['W']} X {c['X']} box pairs")


# ---------------------------------------------------------------------
# The pair operator
# ---------------------------------------------------------------------

def _order_for(kernel: str) -> int:
    return defaults.FMM_ORDER_T if kernel == KERNEL_T else defaults.FMM_ORDER_U


class PairFMM:
    """bbFMM operator for one (field mesh, source mesh, kernel) pair.

    ``matvec(coeffs, x)`` matches :class:`~mbem.la.hop.PairCompressed` term
    for term: ``coeffs`` is the kernel's material coefficient vector, ``x``
    the source DOF vector (3 per source element, ``3*elem + component``),
    the result 3 per field element at the field mesh's collocation points.
    P0 only -- one collocation point and one constant density per element --
    like every compressed path here.

    To run an operator on it, build one PairFMM per pair key of the system
    (``(id(field_patch), id(source_patch), kernel)``) on one shared
    :class:`FmmTree` and hand the dict to ``AssembledH(..., _shared=(pairs,
    {}))``: everything above the pair -- slot offsets, the sigma sign, the
    per-region material through ``kernel_coeffs``, the collocation diagonal
    -- is untouched. ``storage="basis"``, because the "combined" path warms
    views this class does not have.
    """

    def __init__(self, field_mesh, source_mesh, kernel: str, eps_arr,
                 p: int | None = None, geom: FmmTree | None = None,
                 eps_terms: int = defaults.FMM_EPS_TERMS,
                 ncrit: int = defaults.OCTREE_NCRIT,
                 domain: str = "extent", inflate=1.0,
                 arrays: kb.MeshArrays | None = None):
        self.kernel = kernel
        self.n_basis = kernel_n_basis(kernel)          # raises on a bad tag
        if eps_terms not in (1, 2):
            raise ValueError("eps_terms is 1 (K0) or 2 (K0 + eps^2 K1)")
        self.eps_terms = int(eps_terms)
        self.p = int(p) if p is not None else _order_for(kernel)
        self.n_field = field_mesh.n_triangles
        self.n_source = source_mesh.n_triangles
        self.shape = (3 * self.n_field, 3 * self.n_source)
        self.eps = kb.as_eps_array(eps_arr, self.n_source)
        arrays = arrays if arrays is not None else kb.MeshArrays()
        self.geom = geom if geom is not None else FmmTree(
            [field_mesh, source_mesh], ncrit=ncrit, domain=domain,
            inflate=inflate, arrays=arrays)
        self.st = self.geom.stencil(self.p)

        f0, f1 = self.geom.range_of(field_mesh)
        s0, s1 = self.geom.range_of(source_mesh)
        self._f0, self._s0 = f0, s0
        self.x_field = arrays.field_points(field_mesh)
        self.tri_verts, self.normals = arrays.source_arrays(source_mesh)

        tree = self.geom.tree
        nb = len(tree.boxes)
        empty = np.empty(0, dtype=np.int64)
        self._tgt_res = [empty] * nb
        self._src_res = [empty] * nb
        tgt_sub = np.zeros(nb, dtype=bool)
        src_sub = np.zeros(nb, dtype=bool)
        for bi in range(nb):
            held = tree.elements_of(bi)
            if not held.size:
                continue
            self._tgt_res[bi] = np.sort(held[(held >= f0) & (held < f1)]) - f0
            self._src_res[bi] = np.sort(held[(held >= s0) & (held < s1)]) - s0
            tgt_sub[bi] = self._tgt_res[bi].size > 0
            src_sub[bi] = self._src_res[bi].size > 0
        for bi in range(nb - 1, -1, -1):             # boxes are level-ordered
            pj = int(tree.parent[bi])
            if pj >= 0:
                tgt_sub[pj] |= tgt_sub[bi]
                src_sub[pj] |= src_sub[bi]
        self._tgt_sub, self._src_sub = tgt_sub, src_sub

        lists = self.geom.lists
        self._near = self._group(lists.U, False, False)
        self._m2l = self._group(lists.V, True, True)
        self._m2p = self._group(lists.W, False, True)
        self._p2l = self._group(lists.X, True, False)
        self._counts = {"U": sum(n for _a, _r, _c, n in self._near),
                        "V": sum(n for _a, _r, _c, n in self._m2l),
                        "W": sum(n for _a, _r, _c, n in self._m2p),
                        "X": sum(n for _a, _r, _c, n in self._p2l)}
        # Per coefficient vector, exactly as PairCompressed caches views:
        # the geometry above is material-free, the U list is not.
        self._near_cache: OrderedDict = OrderedDict()

    # -- list restriction ------------------------------------------------

    def _group(self, lst, sub_target: bool, sub_source: bool) -> list:
        """One interaction list restricted to this pair's two roles.

        Returns ``[(a, rows, cols_or_boxes, n_entries)]`` grouped by target
        box: ``rows`` are a's resident field elements (None when the list
        serves a's whole subtree through a local expansion), and the third
        item is the source boxes (multipole side) or the concatenated
        resident source elements (point side). Sorted throughout, so the
        accumulation order is a function of the tree alone.
        """
        out = []
        for a in sorted(lst):
            if sub_target:
                if not self._tgt_sub[a]:
                    continue
                rows = None
            else:
                rows = self._tgt_res[a]
                if not rows.size:
                    continue
            if sub_source:
                bs = [b for b in sorted(lst[a]) if self._src_sub[b]]
                if bs:
                    out.append((a, rows, bs, len(bs)))
            else:
                cols = [self._src_res[b] for b in sorted(lst[a])
                        if self._src_res[b].size]
                if cols:
                    out.append((a, rows, np.concatenate(cols), len(cols)))
        return out

    def counts(self) -> dict:
        """Nonempty list entries this pair uses, after role restriction."""
        return dict(self._counts)

    def near_mask(self) -> np.ndarray:
        """(n_field, n_source) True where the pair is evaluated EXACTLY.

        The U list only. W and X are M2P and P2L -- approximate -- so a
        far-field metric that counted them as near would mis-attribute
        their error to the near field.
        """
        mask = np.zeros((self.n_field, self.n_source), dtype=bool)
        for _a, rows, cols, _n in self._near:
            mask[np.ix_(rows, cols)] = True
        return mask

    # -- pieces ----------------------------------------------------------

    def _exact(self, x_field: np.ndarray, cols: np.ndarray,
               c: np.ndarray) -> np.ndarray:
        """The analytic triangle kernel of an arbitrary point set against a
        subset of the source elements, each at its own eps."""
        xf = np.ascontiguousarray(x_field)
        tv = np.ascontiguousarray(self.tri_verts[cols])
        ee = np.ascontiguousarray(self.eps[cols])
        if self.kernel == KERNEL_T:
            nm = np.ascontiguousarray(self.normals[cols])
            return tk.t_matrix_direct(xf, tv, nm, ee, *c)
        return tk.u_matrix_direct(xf, tv, ee, *c)

    def _near_blocks(self, c: np.ndarray) -> list:
        key = c.tobytes()
        blocks = self._near_cache.get(key)
        if blocks is None:
            blocks = [(rows, cols, self._exact(self.x_field[rows], cols, c))
                      for _a, rows, cols, _n in self._near]
            self._near_cache[key] = blocks
            while len(self._near_cache) > defaults.FMM_NEAR_CACHE_MAX:
                self._near_cache.popitem(last=False)
        return blocks

    def _charges(self, dens: np.ndarray) -> np.ndarray:
        """Source charges of every eps pass, stacked: (n_source, nc, T*k).

        The U kernel's charge is the traction DOF itself; the T kernel's is
        ``b_j n_m``. Pass 1 is the same charge times ``eps_j^2``: that, and
        the pass's own kernel, is all of the mollification in the far
        field. Stacked along the right-hand-side axis because P2M and M2M
        are linear, so both passes ride ONE upward traversal.
        """
        q = (dens if self.kernel == KERNEL_U else
             (dens[:, :, None, :] * self.normals[:, None, :, None]
              ).reshape(self.n_source, 9, -1))
        terms = [q] + ([q * (self.eps ** 2)[:, None, None]]
                       if self.eps_terms > 1 else [])
        return np.concatenate(terms, axis=-1)

    def _split(self, M: np.ndarray, k: int) -> list:
        """One box's stacked multipole back into one array per pass, in the
        shape ``_far_apply`` wants."""
        out = []
        for t in range(self.eps_terms):
            q = M[:, :, t * k:(t + 1) * k]
            out.append(q if self.kernel == KERNEL_U
                       else q.reshape(q.shape[0], 3, 3, k))
        return out

    def _canonical(self, boxes: list, which: str, only=None) -> list:
        """One box list carried between its own domain and the cube lattice.

        ``which="e2c_src"`` anterpolates a multipole onto the cube nodes
        (the same contraction M2M uses), ``"e2c_tgt"`` evaluates a local
        expansion held on the cube nodes at the box's own target nodes
        (the contraction L2L uses). Returns ``boxes`` itself when the
        domain needs no transform, which is every domain but canonical.
        """
        maps = getattr(self.st, which)
        if maps is None:
            return boxes
        contract = 0 if which == "e2c_src" else 1
        out = [None] * len(boxes)
        for bi in (range(len(boxes)) if only is None else only):
            val = boxes[bi]
            if val is None:
                continue
            C = val.reshape((self.p,) * 3 + (-1,))
            out[bi] = _separable(maps[bi], C, contract).reshape(val.shape)
        return out

    def _upward(self, charge: np.ndarray) -> list:
        """Multipole per box: P2M of its residents, then M2M of its
        children. Boxes are level-ordered, so descending index order is
        leaves first."""
        tree = self.geom.tree
        p3 = self.p ** 3
        nc, kk = charge.shape[1], charge.shape[2]
        M = [None] * len(tree.boxes)
        for bi in range(len(tree.boxes) - 1, -1, -1):
            if not self._src_sub[bi]:
                continue
            acc = np.zeros((p3, nc, kk))
            cols = self._src_res[bi]
            if cols.size:
                w = self.st.p2m[cols + self._s0]                # (ns, p^3)
                acc += np.tensordot(w.T, charge[cols], axes=([1], [0]))
            for ch in tree.children[bi]:
                if M[ch] is None:
                    continue
                C = M[ch].reshape((self.p,) * 3 + (nc * kk,))
                acc += _separable(self.st.m2m[ch], C, 0).reshape(p3, nc, kk)
            M[bi] = acc
        return M

    def _downward(self, M: list, params: list, c: np.ndarray, X: np.ndarray,
                  k: int) -> list:
        """Local expansions: M2L over V, P2L over X, then L2L down.

        A local expansion holds the FIELD VALUES at the box's target nodes,
        which is what makes bbFMM black-box: M2L, P2L and L2L all add or
        interpolate values, and L2P is one more interpolation.

        Under ``domain="canonical"`` M2L alone runs on the shared cube
        lattice: each multipole it reads is carried there once per
        traversal, each local it writes carried back once. P2L, L2L and
        L2P stay on the extents, so the geometry still only ever meets a
        domain that contains it.
        """
        tree = self.geom.tree
        p3 = self.p ** 3
        Lx = [None] * len(tree.boxes)
        step = max(1, int(defaults.FMM_MAX_POINT_PAIRS // max(p3 * p3, 1)))

        used = {b for _a, _r, bs, _n in self._m2l for b in bs}
        Mc = self._canonical(M, "e2c_src", used)
        Lc = [None] * len(tree.boxes) if Mc is not M else Lx
        for a, _rows, bs, _n in self._m2l:
            xt = self.st.tgt_m2l_nodes[a]
            for i0 in range(0, len(bs), step):
                chunk = bs[i0:i0 + step]
                ys = np.concatenate([self.st.src_m2l_nodes[b] for b in chunk])
                qs = self._split(np.concatenate([Mc[b] for b in chunk]), k)
                val = _far_apply(self.kernel, xt, ys, qs, params)
                Lc[a] = val if Lc[a] is None else Lc[a] + val
        if Lc is not Lx:
            for a, val in enumerate(self._canonical(Lc, "e2c_tgt")):
                Lx[a] = val

        for a, _rows, cols, _n in self._p2l:
            block = self._exact(self.st.tgt_nodes[a], cols, c)
            val = (block @ _dof_rows(X, cols)).reshape(p3, 3, k)
            Lx[a] = val if Lx[a] is None else Lx[a] + val

        for bi in range(len(tree.boxes)):            # parents before children
            if Lx[bi] is None:
                continue
            C = Lx[bi].reshape((self.p,) * 3 + (3 * k,))
            for ch in tree.children[bi]:
                if not self._tgt_sub[ch]:
                    continue
                val = _separable(self.st.l2l[ch], C, 1).reshape(p3, 3, k)
                Lx[ch] = val if Lx[ch] is None else Lx[ch] + val
        return Lx

    # -- operations ------------------------------------------------------

    def matvec(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``y = A(coeffs) x``. ``x`` is (3 n_source,) or (3 n_source, k);
        several right-hand sides share one traversal and one M2L geometry,
        which is the only concession to speed here."""
        c = np.asarray(coeffs, dtype=float)
        params = _far_params(self.kernel, c)[:self.eps_terms]   # validates c
        X = np.asarray(x, dtype=float)
        vector = X.ndim == 1
        X = X.reshape(self.shape[1], -1)
        k = X.shape[1]
        y = np.zeros((self.n_field, 3, k))

        for rows, cols, block in self._near_blocks(c):
            y[rows] += (block @ _dof_rows(X, cols)).reshape(rows.size, 3, k)

        M = self._upward(self._charges(X.reshape(self.n_source, 3, k)))

        for a, rows, bs, _n in self._m2p:
            ys = np.concatenate([self.st.src_nodes[b] for b in bs])
            qs = self._split(np.concatenate([M[b] for b in bs]), k)
            y[rows] += _far_apply(self.kernel, self.x_field[rows], ys, qs,
                                  params)

        for bi, val in enumerate(self._downward(M, params, c, X, k)):
            rows = self._tgt_res[bi]
            if val is None or not rows.size:
                continue
            y[rows] += np.tensordot(self.st.l2p[rows + self._f0], val,
                                    axes=([1], [0]))

        y = y.reshape(self.shape[0], k)
        return y[:, 0] if vector else y

    def to_dense(self, coeffs: np.ndarray) -> np.ndarray:
        """The operator as a dense matrix -- for the gates, never a solve.

        Each column chunk is a full FMM traversal, so this costs O(N)
        matvecs and belongs on small pairs only.
        """
        n = self.shape[1]
        out = np.empty(self.shape)
        step = max(1, int(defaults.FMM_DENSE_COLUMN_CHUNK))
        for i0 in range(0, n, step):
            cols = np.arange(i0, min(i0 + step, n))
            E = np.zeros((n, cols.size))
            E[cols, np.arange(cols.size)] = 1.0
            out[:, i0:i0 + cols.size] = self.matvec(coeffs, E)
        return out

    def summary(self) -> str:
        c = self._counts
        return (f"PairFMM {self.shape} [{self.kernel}] p={self.p} "
                f"terms={self.eps_terms}: U {c['U']} V {c['V']} W {c['W']} "
                f"X {c['X']} entries, max|xhat| {self.st.max_xhat:.3f}")


def _dof_rows(X: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The rows of ``X`` belonging to source elements ``cols`` (3 per
    element, ``3*elem + component`` -- hop's layout, unchanged)."""
    return X[(3 * cols[:, None] + np.arange(3)[None, :]).ravel()]
