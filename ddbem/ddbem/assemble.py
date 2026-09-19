"""Influence matrices for displacement-discontinuity (DD) elements, P0/P1/P2.

Three matrices, all linear in the nodal slip vector ``s`` laid out by
:mod:`ddbem.layout`:

    displacement_matrix(x_field, tri_verts, eps, mu, nu, order)      -> (3 N_f, n_dof)
    traction_matrix(x_field, n_field, tri_verts, eps, mu, nu, order) -> (3 N_f, n_dof)
    stress_matrix(x_field, tri_verts, eps, mu, nu, order)            -> (6 N_f, n_dof)

Each is a loop over source triangles around :func:`clq.influence`, which is
closed form on one flat triangle for constant / linear / quadratic nodal slip
with the Cortez blob ``R = sqrt(r^2 + eps^2)``.  Everything is numpy-vectorised
over the observation points with one triangle per call, exactly as clq is, so a
numba port of the outer loop stays mechanical.

Sign and pairing conventions (inherited from clq, do not "fix" them)
--------------------------------------------------------------------
* Slip sign ``Delta u = u(+nhat) - u(-nhat)``; ``nhat`` is
  ``(v2-v1) x (v3-v1)`` normalised, taken from the vertex order.  There is
  deliberately no separate normal argument, so the element orientation lives in
  ``tri_verts`` alone.
* The slip -> displacement contraction is the traction operator applied to the
  Kelvin solution, slip and normal in C's FIRST index pair:
  ``U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]``.
  This is INVISIBLE at nu = 1/4 (lam = mu) and a swapped form survived in this
  tree for months, so ``verify/verify_kernels.py`` runs every parity check at
  nu in {0.25, 0.30, 0.45}.

DECISION 1 -- ELASTIC vs TOTAL STRESS.  ``stress_matrix`` and
``traction_matrix`` return the ELASTIC stress by default.
A mollified slip is an anelastic (eigen-) strain: clq's ``H`` is the TOTAL
stress ``C:(eps_el + eps*)`` of the smeared slip, whose eigenstress part
``C:eps*`` peaks at ``(3/4) mu s / eps`` on the element and diverges as
eps -> 0.  ``../BACKLOG.md`` is the tree-wide policy: every stress
presented as elastic must have ``C:eps*`` subtracted.  Here the subtraction
uses clq's EXACT finite-triangle weights ``E[n, k]``,

    C:eps*(x) = sum_k E[n,k] [ lam (s_k . nhat) I + mu (s_k nhat^T + nhat s_k^T) ],
    E[n,k]    = int_T N_k(y) phi_eps(x - y) dS(y),   phi_eps = 15 eps^4 / (8 pi R^7),

i.e. the blob-weighted shape-function integral over the actual triangle -- NOT
the point/marginal approximation of ``msd/anelastic.py``
(``rho_eps = (3/4) eps^4 / (d^2 + eps^2)^(5/2)`` evaluated at the distance to
the nearest triangle), which is the infinite-plane limit and is wrong by O(1)
within ~eps of an element edge.  ``verify_kernels.py`` gates both halves: the
exactness of ``E`` against an independent quadrature, and that the two
formulations genuinely differ near an edge (so the gate could not pass with the
approximation swapped in).  Pass ``subtract_eigenstress=False`` for kernel
diagnostics only, and say so in the figure.

DECISION 2 -- NODAL LAYOUT.  DISCONTINUOUS by default: ``3 * K * N_tri``
unknowns, column ``3 * (K * s + k) + j``.  This is the unambiguous layout and
the one every gate checks.  Continuity -- which is where the unknown-count win
of higher-order DD actually comes from (continuous P1 is ~1.5 unknowns per
triangle against P0's 3) -- is a separate modelling decision, wrong across a
sharp geometric edge, a material interface or a crack tip, and it lives in
:mod:`ddbem.layout` as an explicit scatter map.  Pass ``layout=`` to apply one.

DECISION 3 -- FORTRAN ORDERING.  The matrices are filled [source, field]
C-contiguous and returned transposed, so the returned array is F-ordered and
``scipy.linalg.lu_factor(overwrite_a=True)`` factors a square collocation
system in place with no copy.  This is free here: clq returns field-major
blocks, so ONE strided copy per triangle is needed either way, and writing into
a contiguous row slab of the source-major buffer is the cheaper of the two.
It is NOT free in :meth:`ddbem.layout.NodalLayout.condense` (continuous
layouts), which must sum columns into a fresh buffer -- one extra full copy,
documented there.

DEGENERATE TRIANGLES RAISE.  msd's numba assemblers silently ``continue`` past
a zero-area source triangle, leaving its columns at zero; clq raises
``ValueError("degenerate triangle (zero area)")`` and ddbem does not catch it.
A silently-zeroed column is a singular collocation row later, which is a worse
place to find out.

``eps`` is a scalar or a per-source-element ``(N_tri,)`` array (msd's
convention).  fbem/FINDINGS.md sec.2 is the reason it is worth reporting
rather than just setting: eps is a FLOOR on the resolvable structure of the
unknown, so p-refinement pays only when eps/h is small enough that the extra
nodes are more than one mollification length apart.  Use
:func:`ddbem.mesh.eps_report` before trusting a p-refinement study.
"""
from __future__ import annotations

import numpy as np

from . import defaults, layout as _layout
from ._clq import CLQ as clq


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------

def _as_points(x, name="x_field"):
    x = np.ascontiguousarray(np.asarray(x, float))
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3), got {x.shape}")
    return x


def _as_tris(tri_verts):
    t = np.ascontiguousarray(np.asarray(tri_verts, float))
    if t.ndim == 2 and t.shape == (3, 3):
        t = t[None]
    if t.ndim != 3 or t.shape[1:] != (3, 3):
        raise ValueError(f"tri_verts must be (N_tri, 3, 3), got {t.shape}")
    return t


def _as_eps(eps, n_tri):
    e = np.asarray(eps, float)
    if e.ndim == 0:
        e = np.full(n_tri, float(e))
    elif e.shape != (n_tri,):
        raise ValueError(f"eps must be a scalar or ({n_tri},), got {e.shape}")
    if np.any(e < 0.0):
        raise ValueError("eps must be >= 0")
    return e


def _check_layout(lay, n_tri, order):
    if lay is None:
        return _layout.discontinuous(n_tri, order)
    if lay.n_tri != n_tri:
        raise ValueError(f"layout has n_tri={lay.n_tri}, mesh has {n_tri}")
    if lay.order != order:
        raise ValueError(f"layout has order={lay.order}, called with {order}")
    return lay


def _lame(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def eigen_column_tensor(nhat, mu, nu):
    """``sig0[m, l, j]``: the eigenstress ``C:eps*`` per unit ``E`` weight and
    unit slip component ``j`` on a facet with unit normal ``nhat``:

        sig0[m,l,j] = lam n_j delta_ml + mu (delta_mj n_l + n_m delta_lj).

    Identical to clq's ``lam (s.nhat) I + mu (s nhat^T + nhat s^T)`` with
    ``s = e_j``.
    """
    lam = _lame(mu, nu)
    n = np.asarray(nhat, float)
    eye = np.eye(3)
    return (lam * np.einsum("ml,j->mlj", eye, n)
            + mu * (np.einsum("mj,l->mlj", eye, n) + np.einsum("m,lj->mlj", n, eye)))


# ---------------------------------------------------------------------------
# core loop
# ---------------------------------------------------------------------------

def _assemble(x_field, tri_verts, eps, mu, nu, order, n_rows_per_obs,
              block_fn, far_field):
    """Fill a source-major C-contiguous buffer and return it (not transposed).

    ``block_fn(inf, nhat, mu, nu) -> (N_f, K, n_rows_per_obs, 3)``, the last two
    axes being (row-of-this-obs, slip component); ``block_fn.want`` is the clq
    ``want`` tuple it needs.  The buffer is ``(3 K N_tri, n_rows_per_obs N_f)``
    with row ``3 (K s + k) + j``, so each source triangle writes ONE contiguous
    row slab and the caller's ``.T`` is F-ordered.
    """
    n_tri = tri_verts.shape[0]
    n_f = x_field.shape[0]
    K = _layout.n_nodes(order)
    out = np.empty((3 * K * n_tri, n_rows_per_obs * n_f), dtype=float)
    for s in range(n_tri):
        tri = tri_verts[s]
        nhat = clq.unit_normal(tri)
        inf = clq.influence(x_field, tri, mu, nu, eps[s], order=order,
                            far_field=far_field, want=block_fn.want)
        blk = block_fn(inf, nhat, mu, nu)            # (N_f, K, R, 3)
        # -> rows (k, j), cols (n, r)
        out[3 * K * s:3 * K * (s + 1)] = np.ascontiguousarray(
            blk.transpose(1, 3, 0, 2)).reshape(3 * K, n_rows_per_obs * n_f)
    return out


class _Block:
    """Small callable carrying the clq ``want`` tuple it needs."""

    def __init__(self, want, fn):
        self.want = want
        self._fn = fn

    def __call__(self, inf, nhat, mu, nu):
        return self._fn(inf, nhat, mu, nu)


def _u_block(inf, nhat, mu, nu):
    return inf.U                                     # (N, K, 3, 3) = [n,k,i,j]


def _elastic_H(inf, nhat, mu, nu, subtract):
    H = inf.H                                        # (N, K, 3, 3, 3) = [n,k,m,l,j]
    if not subtract:
        return H
    sig0 = eigen_column_tensor(nhat, mu, nu)         # (3,3,3) = [m,l,j]
    return H - inf.E[:, :, None, None, None] * sig0[None, None]


# ---------------------------------------------------------------------------
# public matrices
# ---------------------------------------------------------------------------

def displacement_matrix(x_field, tri_verts, eps, mu, nu, order=0, *,
                        layout=None, far_field=defaults.FAR_FIELD):
    """Slip -> displacement influence matrix, ``(3 N_f, n_dof)``, F-ordered.

    Row ``3 n + i`` is displacement component ``i`` at ``x_field[n]``; column
    ``3 * (K s + k) + j`` is slip component ``j`` at local node ``k`` of source
    triangle ``s`` (discontinuous layout; pass ``layout=`` to condense).

    No free term and no jump is applied: this is the pure single-element
    representation formula, valid at field points OFF the source element.  A
    displacement evaluated ON an element is finite for eps > 0 but is the blob
    average across the discontinuity, not either side of it.
    """
    tri_verts = _as_tris(tri_verts)
    x_field = _as_points(x_field)
    eps = _as_eps(eps, tri_verts.shape[0])
    lay = _check_layout(layout, tri_verts.shape[0], order)
    blk = _Block(("U",), _u_block)
    A = _assemble(x_field, tri_verts, eps, mu, nu, order, 3, blk, far_field)
    return lay.condense(A.T)


def stress_matrix(x_field, tri_verts, eps, mu, nu, order=0, *,
                  subtract_eigenstress=True, layout=None,
                  far_field=defaults.FAR_FIELD):
    """Slip -> stress influence matrix in Voigt rows, ``(6 N_f, n_dof)``, F-ordered.

    Row ``6 n + v`` is Voigt component ``v`` of the stress at ``x_field[n]``,
    with ``v`` ordered ``(xx, yy, zz, yz, xz, xy)``
    (:data:`ddbem.defaults.VOIGT_PAIRS`; no factor of 2 on the shear rows).

    ELASTIC by default: ``sigma_el = sigma_total - C:eps*`` with clq's exact
    finite-triangle eigenstress weights (module docstring, DECISION 1).
    ``subtract_eigenstress=False`` returns the raw total -- kernel diagnostics
    only.
    """
    tri_verts = _as_tris(tri_verts)
    x_field = _as_points(x_field)
    eps = _as_eps(eps, tri_verts.shape[0])
    lay = _check_layout(layout, tri_verts.shape[0], order)
    want = ("H", "E") if subtract_eigenstress else ("H",)
    rows = np.array(defaults.VOIGT_PAIRS)

    def fn(inf, nhat, mu_, nu_):
        sig = _elastic_H(inf, nhat, mu_, nu_, subtract_eigenstress)
        return sig[:, :, rows[:, 0], rows[:, 1], :]          # (N, K, 6, 3)

    blk = _Block(want, fn)
    A = _assemble(x_field, tri_verts, eps, mu, nu, order, 6, blk, far_field)
    return lay.condense(A.T)


def traction_matrix(x_field, n_field, tri_verts, eps, mu, nu, order=0, *,
                    subtract_eigenstress=True, layout=None,
                    far_field=defaults.FAR_FIELD):
    """Slip -> traction influence matrix, ``(3 N_f, n_dof)``, F-ordered.

    ``t_i = sigma_ij n_field_j`` with ``n_field`` the (N_f, 3) unit normal at
    each field point (a single (3,) normal is broadcast).  This is the
    hypersingular DD operator: the collocation row of a traction boundary
    condition.

    ELASTIC by default, for the same reason as :func:`stress_matrix` -- the
    eigenstress is the anelastic part of a smeared slip and is not a traction
    the medium can carry.  It is a no-op more than ~2 eps off the element, so
    it only matters for the near-diagonal blocks, which is exactly where a DD
    collocation system is decided.

    NO free term is applied.  For a mollified kernel there is no analytic
    ``1/2`` jump to add: the traction varies smoothly across the smeared
    element over a width ~eps, and the discrete operator already contains
    whatever the collocation point sees.  fbem/FINDINGS.md sec.2 is the warning
    that goes with that: a collocation point must be at least ~1 eps clear of
    the element boundary for this to behave, and at eps = 0.3 h no P1/P2 node
    is.  Budget eps/h (:func:`ddbem.mesh.eps_report`) before reading any
    p-refinement result.
    """
    tri_verts = _as_tris(tri_verts)
    x_field = _as_points(x_field)
    eps = _as_eps(eps, tri_verts.shape[0])
    lay = _check_layout(layout, tri_verts.shape[0], order)
    nf = np.asarray(n_field, float)
    if nf.ndim == 1:
        nf = np.broadcast_to(nf.reshape(1, 3), x_field.shape)
    nf = _as_points(nf, "n_field")
    if nf.shape[0] != x_field.shape[0]:
        raise ValueError(f"n_field has {nf.shape[0]} rows, x_field has "
                         f"{x_field.shape[0]}")
    want = ("H", "E") if subtract_eigenstress else ("H",)

    def fn(inf, nhat, mu_, nu_):
        sig = _elastic_H(inf, nhat, mu_, nu_, subtract_eigenstress)
        return np.einsum("nkmlj,nl->nkmj", sig, nf)          # (N, K, 3, 3)

    blk = _Block(want, fn)
    A = _assemble(x_field, tri_verts, eps, mu, nu, order, 3, blk, far_field)
    return lay.condense(A.T)


def eigenstress_matrix(x_field, tri_verts, eps, mu, nu, order=0, *,
                       layout=None, far_field=defaults.FAR_FIELD):
    """Slip -> eigenstress ``C:eps*`` in Voigt rows, ``(6 N_f, n_dof)``.

    Exposed so ``total = elastic + eigenstress`` can be checked exactly and so a
    caller who needs the total stress can rebuild it without re-deciding the
    subtraction.  This is the EXACT finite-triangle eigenstress, not a point
    approximation (module docstring, DECISION 1).
    """
    tri_verts = _as_tris(tri_verts)
    x_field = _as_points(x_field)
    eps = _as_eps(eps, tri_verts.shape[0])
    lay = _check_layout(layout, tri_verts.shape[0], order)
    rows = np.array(defaults.VOIGT_PAIRS)

    def fn(inf, nhat, mu_, nu_):
        sig0 = eigen_column_tensor(nhat, mu_, nu_)           # (3,3,3)
        full = inf.E[:, :, None, None, None] * sig0[None, None]
        return full[:, :, rows[:, 0], rows[:, 1], :]

    blk = _Block(("E",), fn)
    A = _assemble(x_field, tri_verts, eps, mu, nu, order, 6, blk, far_field)
    return lay.condense(A.T)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def voigt_to_tensor(sig_voigt):
    """(..., 6) Voigt stress -> (..., 3, 3) symmetric tensor."""
    v = np.asarray(sig_voigt, float)
    if v.shape[-1] != 6:
        raise ValueError(f"expected a trailing axis of 6, got {v.shape}")
    out = np.empty(v.shape[:-1] + (3, 3))
    for c, (m, l) in enumerate(defaults.VOIGT_PAIRS):
        out[..., m, l] = v[..., c]
        out[..., l, m] = v[..., c]
    return out


def tensor_to_voigt(sig):
    """(..., 3, 3) stress -> (..., 6) Voigt."""
    s = np.asarray(sig, float)
    if s.shape[-2:] != (3, 3):
        raise ValueError(f"expected trailing (3, 3), got {s.shape}")
    return np.stack([s[..., m, l] for (m, l) in defaults.VOIGT_PAIRS], axis=-1)
