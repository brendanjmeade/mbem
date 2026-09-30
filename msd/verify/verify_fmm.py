#!/usr/bin/env python
"""Gate: the reference Chebyshev black-box FMM (mbem/la/fmm.py).

The far field is the one part of this operator nobody can check by eye. It
is an approximation whose error moves with the interpolation order p, with
the mollification eps and with the interaction lists, and every one of those
is SILENT when it is wrong: the answer changes, nothing raises. So every
clause here is a number against an EXACT reference -- the analytic triangle
kernels through ``kernels/basis`` for a pair, ``AssembledDense`` for the
operator -- and never against another approximation.

  a  the point-pair far kernel against the analytic triangle kernel: its
     one-point limit as the element shrinks (O((h/R)^2)), and the eps^2 term
     against the exact eps dependence as eps shrinks (O(eps^4), 16x per
     halving, against the 4x of the term it corrects). At nu = 0.30, and the
     clause also shows that the lam/mu swap this geometry hides at nu = 1/4
     is a 40 % error here.
  b  PAIR level: matvec against the exact dense pair in 2-norm and in
     max-entry, and entrywise on the matrix, at p = 6 (U) and p = 8 (T) --
     on two panels with NO near field (0 % U list), so nothing exact is
     mixed in, and on a fault-zone pair that exercises all four lists.
  c  FAR-FIELD ISOLATION: the near field is exact, so a relative error on
     A v understates the far field by whatever fraction of A v the near
     field carries. This clause splits the exact operator by the FMM's OWN
     U list (verified to machine precision), and every error below is
     reported BOTH ways: naive over ||A v|| and isolated over ||A_far v||.
     W and X are M2P and P2L -- approximate -- and stay on the far side.
  d  OPERATOR level: ``AssembledH`` with PairFMM swapped in for
     PairCompressed, against ``AssembledDense.A``, on the smallest model
     with a dense reference. jump="half" (the calibrated diagonal is built
     FROM the far field, so it would partly cancel the error it came from)
     and eps="auto" (a scalar eps of 3 puts (eps/r)^4 at 260x the target and
     would price the expansion, not the FMM).
  e  CONVERGENCE IN p, p = 4..8: the error must fall at every step. A bbFMM
     whose error does not move with p is broken in a way no single-p number
     reveals.
  f  the eps^2 PASS must matter: one pass against two, on a pair and on the
     operator.
  g  the INTERPOLATION DOMAIN: containment on the enlarged extents, its
     failure on the nominal cubes, and the three ways to have both that and
     a SHARED M2L table -- a uniform cube inflation, a source-only one, and
     a per-box change of basis onto the cube lattice. Each is priced in
     error, in the V-list node separation it spends, in the W and X margins
     it spends (those two paths evaluate at a FIELD POINT, not at a node,
     so the V separation says nothing about them), and in the number of
     M2L tables it actually shares, counted on the lattices rather than on
     the cube indices. The half-width floor is checked by removal: without
     it a flat box's P2M is NaN.

Run from msd/. PASS:/FAIL:, exit 1 on FAIL. It is slow (minutes): the
reference evaluates every M2L where it is used, in numpy.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh import make_rectangular_patch                 # noqa: E402
from mbem import defaults                                         # noqa: E402
from mbem.backends.dense import AssembledDense, translation_basis  # noqa: E402
from mbem.backends.hmat import AssembledH                         # noqa: E402
from mbem.kernels import (KERNEL_T, KERNEL_U, kernel_coeffs)      # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.kernels import tri_kernels as tk                        # noqa: E402
from mbem.la.fmm import (FarGroups, FmmTree, PairFMM, _cheb_nodes,  # noqa: E402
                         _far_apply, _far_params)
from mbem.la.fmm_table import pass_degree                         # noqa: E402
from mbem.model import generate_system                            # noqa: E402
from mbem.model.equations import (add_block_diagonal,            # noqa: E402
                                  term_diagonal)
# The fault-zone model is STATED ONCE, in verify_hbackend; this gate uses
# the same geometry at the same refinement so the two are comparable.
from verify_hbackend import _build_zone_model                     # noqa: E402

# nu = 0.30. Every kernel number here runs off nu != 1/4, where a lam/mu
# swap in the slip -> displacement pairing is invisible (msd rule 6).
MAT = mb.ElasticMaterial(mu=30.0, lam=45.0)
EPS = "auto"                      # eps_j = 0.1 h_j; see defaults.FMM_EPS_TERMS
PAIR_TOL = defaults.FMM_PAIR_PARITY
VARIANT_TOL = defaults.FMM_M2L_VARIANT_PARITY
# A truncated M2L is an approximation, so its parity limit is its own
# truncation with room for the accumulation through a whole traversal.
RANK_SLACK = 50.0
OP_TOL = defaults.FMM_OPERATOR_PARITY
NAIVE_TOL = defaults.FMM_OPERATOR_PARITY_NAIVE
ORDER = {KERNEL_U: defaults.FMM_ORDER_U, KERNEL_T: defaults.FMM_ORDER_T}
# Placement safety this gate's trees are built at, PINNED rather than read
# from OCTREE_PLACEMENT_SAFETY, because the shipping value is chosen for a
# 4M-unknown mesh and this model has 2,592 unknowns. Measured on it: safety
# 1.0 leaves U at 60.3 % with V 2588, W 758, X 437 box pairs; 1.5 empties W
# and X entirely; 2.0 cuts V to 384; and 2.5 and above leave U at 100 %, no
# far field at all, so the operator clause would divide roundoff by roundoff.
# A gate that follows the default would therefore stop exercising the very
# lists it exists to check. What the default is measured against is
# topo_inclusion at three scales, recorded beside the constant.
SAFETY = 1.0


# ---------------------------------------------------------------------
# Shared geometry (built once; these clauses are minutes, not seconds)
# ---------------------------------------------------------------------

_CACHE: dict = {}


def _panels():
    """Two parallel 256-triangle panels at 2x their own size apart.

    verify_hbackend's admissible geometry. Under one octree over the union
    the pair is 0 % U list: every element pair is far, so a far-field error
    appears at full strength with no near field to hide behind.
    """
    if "panels" not in _CACHE:
        field = make_rectangular_patch((-40.0, 40.0), (-40.0, 40.0), 0.0,
                                       16, 8, normal_up=True)
        source = make_rectangular_patch((-40.0, 40.0), (-40.0, 40.0), -160.0,
                                        16, 8, normal_up=True)
        _CACHE["panels"] = (field, source)
    return _CACHE["panels"]


def _zone_meshes(model) -> list:
    """Every distinct mesh of the model, once, in declaration order."""
    meshes, seen = [], set()
    for region in model.regions:
        for patch in list(region.patches) + list(region.faults):
            if id(patch.mesh) not in seen:
                seen.add(id(patch.mesh))
                meshes.append(patch.mesh)
    return meshes


def _zone():
    """The fault-zone model, its shared FmmTree and its exact dense operator.

    Every region is put at nu = 0.30: the wrapper's own materials are
    lam = mu, i.e. exactly the nu = 1/4 that hides a lam/mu swap.
    """
    if "zone" in _CACHE:
        return _CACHE["zone"]
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=15.0))
    for region in model.regions:
        if region.material.lam == region.material.mu:
            region.material = mb.ElasticMaterial(mu=region.material.mu,
                                                 lam=1.5 * region.material.mu)
    system = generate_system(model)
    dense = AssembledDense(system, EPS, "direct", jump="half")
    arrays = kb.MeshArrays()
    geom = FmmTree(_zone_meshes(model), arrays=arrays,
                   placement_safety=SAFETY)
    _CACHE["zone"] = (model, system, dense, geom, arrays)
    return _CACHE["zone"]


def _pair_keys(system) -> dict:
    keys = {}
    for t in system.terms:
        keys.setdefault((id(t.field_patch), id(t.source_patch), t.kernel),
                        (t.field_patch, t.source_patch, t.kernel))
    return keys


def _zone_pairs(order: dict, **kw) -> dict:
    """One PairFMM per system-term pair key, on the shared tree."""
    _model, system, _dense, geom, arrays = _zone()
    return {k: PairFMM(fp.mesh, sp.mesh, kern, kb.resolve_patch_eps(EPS, sp),
                       p=order[kern], geom=geom, arrays=arrays, **kw)
            for k, (fp, sp, kern) in _pair_keys(system).items()}


def _exact_pair(field_mesh, source_mesh, kernel, material, eps) -> np.ndarray:
    """The exact dense pair, through ``kernels/basis`` -- an assembly path
    the FMM does not use, so the reference is independent of it."""
    if kernel == KERNEL_T:
        return kb.assemble_t_matrix(field_mesh, source_mesh, material, eps)
    return kb.assemble_u_matrix(field_mesh, source_mesh, material, eps)


def _errs(y, ref) -> tuple:
    """(2-norm relative, max-entry relative) of an approximation."""
    scale = max(float(np.max(np.abs(ref))), 1e-300)
    return (float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-300)),
            float(np.max(np.abs(y - ref)) / scale))


def _block_mask(mask: np.ndarray) -> np.ndarray:
    """Element-pair mask to DOF-pair mask (3x3 per element pair)."""
    return np.kron(mask.astype(float), np.ones((3, 3)))


# ---------------------------------------------------------------------
# a. the point-pair far kernel against the analytic triangle kernel
# ---------------------------------------------------------------------

def _tri(scale: float) -> tuple:
    """A triangle scaled about its centroid; verts, normal, area, centre."""
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.0]])
    c = v.mean(axis=0)
    v = c + scale * (v - c)
    n = np.cross(v[1] - v[0], v[2] - v[0])
    area = 0.5 * float(np.linalg.norm(n))
    return v, n / np.linalg.norm(n), area, c


def _point_field(kernel, coeffs, obs, tri, nrm, area, dens, eps, terms):
    """The FMM's own far kernel at one source point, one target point."""
    q = (area * dens[None, :, None] if kernel == KERNEL_U else
         area * (dens[None, :, None, None] * nrm[None, None, :, None]))
    charges = [q] + ([q * eps ** 2] if terms > 1 else [])
    params = _far_params(kernel, coeffs)[:terms]
    return _far_apply(kernel, obs[None, :], tri.mean(axis=0)[None, :],
                      charges, params)[0, :, 0]


def _exact_field(kernel, coeffs, obs, tri, nrm, dens, eps):
    v = np.ascontiguousarray(tri[None, :, :])
    e = np.array([eps])
    if kernel == KERNEL_T:
        M = tk.t_matrix_direct(obs[None, :], v, nrm[None, :], e, *coeffs)
    else:
        M = tk.u_matrix_direct(obs[None, :], v, e, *coeffs)
    return M @ dens


def check_point_kernel() -> bool:
    """[a] The far kernel IS the triangle kernel's point limit, and its
    eps^2 term IS the mollification's."""
    obs = np.array([2.0, -1.5, 3.0])
    dens = np.array([0.3, -0.7, 0.5])
    ok = True
    for kernel in (KERNEL_U, KERNEL_T):
        c = np.asarray(kernel_coeffs(kernel, MAT))
        print(f"    [{kernel}] one-point limit at eps = 0, R = "
              f"{np.linalg.norm(obs):.2f}")
        prev, ratios = None, []
        for s in (1.0, 0.5, 0.25, 0.125):
            v, n, area, cen = _tri(s)
            h = float(np.linalg.norm(v[1] - v[0]))
            ref = _exact_field(kernel, c, obs, v, n, dens, 0.0)
            got = _point_field(kernel, c, obs, v, n, area, dens, 0.0, 1)
            e = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
            if prev is not None:
                ratios.append(prev / e)
            prev = e
            print(f"        h/R = {h/np.linalg.norm(obs - cen):.4f}: "
                  f"rel {e:.3e}"
                  + (f"   x{ratios[-1]:.1f}" if ratios else ""))
        ok &= all(r >= 3.0 for r in ratios)          # O((h/R)^2) is 4x

        # eps: the EXACT eps dependence of the triangle kernel, against the
        # eps^2 term alone. The h-quadrature error is eps-independent and
        # cancels out of the difference; what does NOT cancel is the
        # quadrature error OF the eps^2 term, an O(eps^2 (h/R)^2) cross term
        # that at h/R = 0.03 is already half of the eps^4 remainder at the
        # smallest eps. The element is shrunk until it is not, or the clause
        # would be measuring the one-point limit again.
        v, n, area, cen = _tri(0.03125)
        base = _exact_field(kernel, c, obs, v, n, dens, 0.0)
        print(f"    [{kernel}] eps expansion (residual after 1 term / after "
              f"2 terms)")
        r1 = r2 = None
        gains = []
        for eps in (0.4, 0.2, 0.1, 0.05):
            d_exact = _exact_field(kernel, c, obs, v, n, dens, eps) - base
            two = _point_field(kernel, c, obs, v, n, area, dens, eps, 2)
            one = _point_field(kernel, c, obs, v, n, area, dens, eps, 1)
            d_model = two - one
            scale = float(np.max(np.abs(base)))
            e1 = float(np.max(np.abs(d_exact))) / scale
            e2 = float(np.max(np.abs(d_exact - d_model))) / scale
            line = (f"        eps = {eps:.3f}: 1 term {e1:.3e}  "
                    f"2 terms {e2:.3e}")
            if r1 is not None:
                gains.append((r1 / e1, r2 / e2))
                line += f"   x{r1/e1:.1f} / x{r2/e2:.1f}"
            r1, r2 = e1, e2
            print(line)
        ok &= all(g1 >= 3.0 and g2 >= 10.0 for g1, g2 in gains)

    # The pairing this geometry would hide at nu = 1/4.
    v, n, area, cen = _tri(0.125)
    c = np.asarray(kernel_coeffs(KERNEL_T, MAT))
    swapped = c.copy()
    swapped[[0, 1, 2]], swapped[[3, 4, 5]] = c[[3, 4, 5]], c[[0, 1, 2]]
    ref = _exact_field(KERNEL_T, c, obs, v, n, dens, 0.0)
    bad = _point_field(KERNEL_T, swapped, obs, v, n, area, dens, 0.0, 1)
    swap_err = float(np.max(np.abs(bad - ref)) / np.max(np.abs(ref)))
    print(f"    [H] lam/mu slots swapped: rel {swap_err:.3f} "
          f"(0 at nu = 1/4 -- this clause runs at nu = "
          f"{MAT.lam / (2 * (MAT.lam + MAT.mu)):.2f})")
    ok &= swap_err > 0.1
    return ok


# ---------------------------------------------------------------------
# b / c. pair level and the far-field isolation metric
# ---------------------------------------------------------------------

def _pair_report(name, pair, exact, coeffs, rng, far=None) -> tuple:
    """matvec errors both ways, plus the entrywise matrix error."""
    x = rng.standard_normal(pair.shape[1])
    y = pair.matvec(coeffs, x)
    ref = exact @ x
    e2, emax = _errs(y, ref)
    line = (f"    {name}: matvec 2-norm {e2:.3e}  max-entry {emax:.3e}")
    if far is not None:
        scale = float(np.max(np.abs(far @ x)))
        line += f"  far-isolated {np.max(np.abs(y - ref)) / scale:.3e}"
    print(line)
    return e2, emax


def _panel_lattice(p: int) -> np.ndarray:
    """The Chebyshev lattice at unit half-width, in the node order
    ``_Stencil`` uses -- the one every box's M2L nodes translate."""
    c = _cheb_nodes(p)
    return np.stack(np.meshgrid(c, c, c, indexing="ij"),
                    axis=-1).reshape(-1, 3)


def check_pair() -> bool:
    """[b] matvec and the matrix itself against the exact dense pair."""
    field, source = _panels()
    eps = kb.resolve_eps(EPS, source)
    rng = np.random.default_rng(0)
    ok = True
    for kernel in (KERNEL_U, KERNEL_T):
        c = np.asarray(kernel_coeffs(kernel, MAT))
        exact = _exact_pair(field, source, kernel, MAT, eps)
        t0 = time.time()
        pair = PairFMM(field, source, kernel, eps, p=ORDER[kernel])
        print(f"    {pair.summary()}")
        e2, emax = _pair_report(f"[{kernel}] panels p={ORDER[kernel]}", pair,
                                exact, c, rng)
        # Entrywise on the matrix, over a spread of source elements: the
        # per-block work was measured in relative Frobenius while max-entry
        # is 1.3-3.0e-4 at p = 8, so both belong on the record. A column
        # slice rather than the whole matrix because to_dense is one FMM
        # traversal per chunk -- the U pair below pays for the whole thing
        # once, to keep that API gated too.
        sel = np.arange(0, source.n_triangles,
                        max(1, source.n_triangles // 16))[:16]
        cols = (3 * sel[:, None] + np.arange(3)[None, :]).ravel()
        E = np.zeros((pair.shape[1], cols.size))
        E[cols, np.arange(cols.size)] = 1.0
        f2, fmax = _errs(pair.matvec(c, E), exact[:, cols])
        print(f"    [{kernel}] matrix over {cols.size} columns: Frobenius "
              f"{f2:.3e}  max-entry {fmax:.3e}   ({time.time() - t0:.1f}s)")
        frac = 100 * pair.near_mask().mean()
        print(f"    [{kernel}] U list covers {frac:.1f} % of element "
              f"pairs -- nothing exact is mixed in")
        ok &= max(e2, emax, f2, fmax) < PAIR_TOL
        # Every other M2L evaluator is gated HERE, against the reference one
        # and not against the exact kernel: the same arithmetic in a different
        # summation order, so the limit is roundoff. Against PAIR_TOL a
        # variant could regress by four orders and still pass.
        xv = rng.standard_normal(pair.shape[1])
        ref_v = pair.matvec(c, xv)
        # The numba kernel is a REARRANGEMENT of the reference and is held to
        # roundoff. The table is too when it is dense -- but compressed it is
        # an APPROXIMATION, so it gets its own limit, against the dense table
        # rather than against the reference: that isolates the truncation
        # from the rearrangement, and holding it at VARIANT_TOL would simply
        # assert that the compression does nothing.
        held = defaults.FMM_M2L_RANK_TOL
        try:
            for variant, tol, ref_kind in (
                    ("numba", VARIANT_TOL, "reference"),
                    ("table", VARIANT_TOL, "reference"),      # rank_tol 0
                    ("table", defaults.FMM_M2L_RANK_TOL * RANK_SLACK,
                     "dense table")):
                t0 = time.time()
                # "table" needs a translation-invariant lattice, which the
                # default extent domain does not have -- it would silently
                # fall back and gate nothing, so it is named here.
                kw = {} if variant == "numba" else {"domain": "canonical"}
                defaults.FMM_M2L_RANK_TOL = (held if ref_kind == "dense table"
                                             else 0.0)
                var = PairFMM(field, source, kernel, eps, p=ORDER[kernel],
                              m2l=variant, **kw)
                got = var.matvec(c, xv)
                if ref_kind == "reference":
                    base = ref_v if not kw else PairFMM(
                        field, source, kernel, eps,
                        p=ORDER[kernel], **kw).matvec(c, xv)
                else:
                    defaults.FMM_M2L_RANK_TOL = 0.0
                    base = PairFMM(field, source, kernel, eps,
                                   p=ORDER[kernel], m2l="table",
                                   **kw).matvec(c, xv)
                v2, vmax = _errs(got, base)
                rk = var._table.ranks if getattr(var, "_table", None) else []
                tag = f"m2l={variant}" + (" (rank)" if ref_kind ==
                                          "dense table" else "")
                print(f"    [{kernel}] {tag} vs the {ref_kind}: 2-norm "
                      f"{v2:.3e}  max-entry {vmax:.3e}"
                      + (f"  rank {min(rk)}-{max(rk)} of {3*ORDER[kernel]**3}"
                         if rk and ref_kind == "dense table" else "")
                      + f"   ({time.time() - t0:.1f}s)")
                ok &= max(v2, vmax) < tol
        finally:
            defaults.FMM_M2L_RANK_TOL = held

        # THE LEVEL FOLD, which is what makes the table O(1) in N: halving
        # every length multiplies a pass by exactly 2**degree, and the
        # degree is the radial power for U but power - 1 for T, every T term
        # carrying one more factor of d upstairs. Gated with array_equal
        # because the ratio is a power of two and there is nothing to round
        # -- and because using the power for T is a factor of 2 per level of
        # reuse, which a tolerance on the far field would swallow.
        prm = _far_params(kernel, c)
        u = _panel_lattice(ORDER[kernel])
        d = np.array([2.0, 0.0, 0.0])
        nc = 3 if kernel == KERNEL_U else 9
        q = rng.standard_normal((u.shape[0],) + ((3,) if nc == 3 else (3, 3))
                                + (1,))
        exact = True
        for p_i in prm:
            deg = pass_degree(kernel, p_i[0])
            one = _far_apply(kernel, u, u + 2 * d, [q], [p_i])
            half = _far_apply(kernel, 0.5 * u, 0.5 * (u + 2 * d), [q], [p_i])
            exact &= np.array_equal(half, (2.0 ** deg) * one)
        print(f"    [{kernel}] level fold exact at degree "
              f"{[pass_degree(kernel, p_i[0]) for p_i in prm]}: {exact}")
        ok &= exact
    t0 = time.time()
    pair = PairFMM(field, source, KERNEL_U, eps, p=ORDER[KERNEL_U])
    c = np.asarray(kernel_coeffs(KERNEL_U, MAT))
    d2, dmax = _errs(pair.to_dense(c),
                     _exact_pair(field, source, KERNEL_U, MAT, eps))
    print(f"    [G] to_dense, whole matrix: Frobenius {d2:.3e}  max-entry "
          f"{dmax:.3e}   ({time.time() - t0:.1f}s)")
    ok &= max(d2, dmax) < PAIR_TOL

    # The geometry is material-free and a pair serves several materials
    # (4 of the fault-zone model's 91 pair keys are used at two, an
    # interface patch being in two regions): a second material must leave
    # the first bitwise unchanged, or a sweep measures build order.
    other = np.asarray(kernel_coeffs(KERNEL_U,
                                     mb.ElasticMaterial(mu=10.0, lam=6.0)))
    x = np.random.default_rng(6).standard_normal(pair.shape[1])
    y1 = pair.matvec(c, x)
    pair.matvec(other, x)
    same = np.array_equal(pair.matvec(c, x), y1)
    print(f"    [G] a second material leaves the first bitwise: {same}")
    return ok and same


def _worst_zone_pair(pairs, which: str = "WX"):
    """The pair with the most entries of one list.

    W (M2P) and X (P2L) are the mixed-level lists -- a pinned oversized
    element against a neighbour's subtree -- which is what the ADAPTIVE
    tree exists for and what a uniform one never tests. They are global
    transposes of each other, so one pair rarely carries both; the gate
    takes the worst of each.
    """
    key = max(pairs, key=lambda k: sum(pairs[k].counts()[w] for w in which))
    return key, pairs[key]


def _term_material(system, model, key):
    names = {r.name: r.material for r in model.regions}
    for t in system.terms:
        if (id(t.field_patch), id(t.source_patch), t.kernel) == key:
            return names[t.region.name]
    raise KeyError(key)


def check_far_isolation() -> bool:
    """[c] Split the exact pair by the FMM's own U list, and show what the
    naive metric hides."""
    model, system, _dense, _geom, _arrays = _zone()
    pairs = _zone_pairs(ORDER)
    keys = _pair_keys(system)
    ok = True
    seen = set()
    for which in ("W", "X"):
        key, pair = _worst_zone_pair(pairs, which)
        if key in seen:
            continue
        seen.add(key)
        fp, sp, kernel = keys[key]
        mat = _term_material(system, model, key)
        c = np.asarray(kernel_coeffs(kernel, mat))
        exact = _exact_pair(fp.mesh, sp.mesh, kernel, mat,
                            kb.resolve_patch_eps(EPS, sp))
        print(f"    most {which}: {fp.name} <- {sp.name} [{kernel}]: "
              f"{pair.summary()}")

        near = _block_mask(pair.near_mask())
        A_U, A_far = exact * near, exact * (1.0 - near)
        split = float(np.max(np.abs(A_U + A_far - exact)))
        print(f"        exact = A_U + A_far to {split:.1e}; U list is "
              f"{100 * pair.near_mask().mean():.1f} % of element pairs, "
              f"A_far max {np.max(np.abs(A_far)) / np.max(np.abs(exact)):.3f} "
              f"of A max")

        rng = np.random.default_rng(1)
        ok &= split == 0.0
        for name, v in (("gaussian", rng.standard_normal(pair.shape[1])),
                        ("constant", np.tile([1.0, 0.0, 0.0], pair.n_source))):
            ref, far = exact @ v, A_far @ v
            e = float(np.max(np.abs(pair.matvec(c, v) - ref)))
            naive = e / np.max(np.abs(ref))
            iso = e / np.max(np.abs(far))
            print(f"        {name:8s}: ||A_far v||/||A v|| "
                  f"{np.max(np.abs(far)) / np.max(np.abs(ref)):.3f}   "
                  f"naive {naive:.3e}   far-isolated {iso:.3e}   "
                  f"(x{iso / naive:.1f})")
            ok &= iso < PAIR_TOL
        # The M2L variant again, on the list this pair is chosen FOR: the
        # far evaluator serves M2P as well as M2L, and clause [b]'s panels
        # have no W or X, so that call site is gated only here.
        var = PairFMM(fp.mesh, sp.mesh, kernel,
                      kb.resolve_patch_eps(EPS, sp), p=ORDER[kernel],
                      geom=pair.geom, m2l="numba")
        xv = np.random.default_rng(2).standard_normal(pair.shape[1])
        v2, vmax = _errs(var.matvec(c, xv), pair.matvec(c, xv))
        print(f"        m2l=numba vs the reference evaluator (W {pair.counts()['W']}"
              f" X {pair.counts()['X']}): 2-norm {v2:.3e}  max-entry {vmax:.3e}")
        ok &= max(v2, vmax) < VARIANT_TOL
    return ok


# ---------------------------------------------------------------------
# d. operator level
# ---------------------------------------------------------------------

def _far_operator(system, pairs, materials) -> np.ndarray:
    """The exact operator restricted to the element pairs the FMM does NOT
    evaluate exactly, term by term. Its complement plus the collocation
    diagonal must reproduce ``AssembledDense.A`` exactly, which is checked
    where it is used.

    THE DENOMINATOR MUST COME FROM ONE FIXED PARTITION. It is tempting to
    take each configuration's own near/far split, and that is wrong: the
    split is what a change under test MOVES. Demoting three X entries at
    260,598 unknowns -- 540 element pairs of 57 million -- drops
    ``||A_far v||`` by 36 %, so a fix that leaves the absolute error alone
    reads as a 1.6x REGRESSION, and two configurations with the same
    absolute error (2.9048e-05 against 2.8999e-05) land either side of the
    limit because their denominators differ by 49 %. Callers therefore
    compute this ONCE, from the reference configuration, and reuse it for
    every variant they compare. Tuning an FMM against a denominator that
    moves when you touch the operator is how a threshold wandered
    1.45 -> 1.90 -> 1.65 over three measurements of the same quantity.
    """
    n = system.layout.n_unknowns
    A_far = np.zeros((n, n))
    A_near = np.zeros((n, n))
    for term in system.terms:
        pair = pairs[(id(term.field_patch), id(term.source_patch),
                      term.kernel)]
        mat = materials[term.region.name]
        exact = _exact_pair(term.field_patch.mesh, term.source_patch.mesh,
                            term.kernel, mat,
                            kb.resolve_patch_eps(EPS, term.source_patch))
        near = _block_mask(pair.near_mask())
        r0, c0 = term.row.offset, term.col.offset
        A_far[r0:term.row.stop, c0:term.col.stop] += \
            term.scale * exact * (1.0 - near)
        A_near[r0:term.row.stop, c0:term.col.stop] += term.scale * exact * near
        D = term_diagonal(term, None)
        if D is not None:
            add_block_diagonal(A_near, r0, c0, D,
                               term.field_patch.collocation_shape())
    return A_far, A_near


def _operator(order: dict, grouped: bool = False, **kw):
    model, system, _dense, geom, arrays = _zone()
    pairs = _zone_pairs(order, **kw)
    groups = None
    if grouped:
        groups = FarGroups(system, {r.name: r.material for r in model.regions},
                           geom, order, EPS, arrays, **kw)
    hasm = AssembledH(system, EPS, {}, False, jump="half", storage="basis",
                      _shared=(pairs, {}), _groups=groups)
    return pairs, hasm


def _test_vectors(system, rng) -> list:
    """Seeded Gaussians plus the unit translation. The far field is smooth
    and low rank, so a Gaussian's energy sits largely where the far blocks
    cancel it (~sqrt(n) instead of n) while a constant field sums
    coherently: the translation is the cancellation-sensitive case and the
    one the calibrated operator annihilates."""
    n = system.layout.n_unknowns
    Z = translation_basis(system.layout)
    return [("gaussian 0", rng.standard_normal(n)),
            ("gaussian 1", rng.standard_normal(n)),
            ("translation", Z[:, 0] / np.max(np.abs(Z[:, 0])))]


def check_operator() -> bool:
    """[d] AssembledH with PairFMM in place of PairCompressed, against the
    exact dense operator."""
    model, system, dense, geom, _arrays = _zone()
    materials = {r.name: r.material for r in model.regions}
    print(f"    {system.layout.n_unknowns} unknowns, "
          f"{len(_pair_keys(system))} pairs, {geom.summary()}")

    ok = True
    ref_far = None
    for order in ({KERNEL_U: 4, KERNEL_T: 4}, ORDER):
        pairs, hasm = _operator(order)
        if ref_far is None:
            ref_far, near = _far_operator(system, pairs, materials)
            split = float(np.max(np.abs(ref_far + near - dense.A)))
            print(f"    A_far + A_U + diagonal = AssembledDense.A to "
                  f"{split:.1e};  A_far max "
                  f"{np.max(np.abs(ref_far)) / np.max(np.abs(dense.A)):.3f} "
                  f"of A max")
            ok &= split < 1e-12 * np.max(np.abs(dense.A))
            tot = {k: sum(p.counts()[k] for p in pairs.values())
                   for k in "UVWX"}
            print(f"    list entries over all pairs: {tot}")
        tag = f"p = {order[KERNEL_U]} (U) / {order[KERNEL_T]} (T)"
        for name, v in _test_vectors(system, np.random.default_rng(2)):
            t0 = time.time()
            y = hasm.matvec(v)
            ref = dense.A @ v
            e = float(np.max(np.abs(y - ref)))
            e2 = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
            naive = e / float(np.max(np.abs(ref)))
            iso = e / float(np.max(np.abs(ref_far @ v)))
            print(f"    {tag}  {name:12s}: naive {naive:.3e}  far-isolated "
                  f"{iso:.3e}  2-norm {e2:.3e}   ({time.time() - t0:.0f}s)")
            if order == ORDER:
                # Two limits, because neither alone is safe. The isolated
                # one is the meaningful accuracy statement but its
                # denominator is a property of the PARTITION; the naive one
                # is weaker but is a pure ratio of the operator to itself,
                # so no change to the near/far split can move it. A variant
                # that passes by shrinking its own far field fails here.
                ok &= iso < OP_TOL
                ok &= naive < NAIVE_TOL

    # The grouped traversal is a REARRANGEMENT of the per-term sum, not an
    # approximation: one traversal per (region, kernel) with sigma folded
    # into the source, which is exact only because a region couples its
    # patches completely. So it is gated against the per-term operator at
    # roundoff -- against the dense operator it would pass while silently
    # dropping a whole patch pair.
    t0 = time.time()
    _gp, hg = _operator(ORDER, grouped=True)
    _pp, hp = _operator(ORDER)
    worst = 0.0
    for name, v in _test_vectors(system, np.random.default_rng(2)):
        a, b = hg.matvec(v), hp.matvec(v)
        worst = max(worst, float(np.max(np.abs(a - b)))
                    / max(float(np.max(np.abs(b))), 1e-300))
    print(f"    grouped ({len(hg._groups.groups)} traversals) vs per-term "
          f"({len(_pair_keys(system))} pairs): {worst:.3e}   "
          f"({time.time() - t0:.0f}s)")

    # The PRECONDITIONER's strictly-lower couplings, grouped against the
    # per-term loop they replace. Gated because the failure is SILENT: an
    # applier that scatters to the wrong key subtracts nothing, the sweep
    # quietly degrades to block-Jacobi, and the solve still converges --
    # just slower. That is exactly what happened (37 iterations became 52,
    # because `local_offset` is keyed by SLOT name and the group was
    # indexed by PATCH name), and no residual or error limit would show it.
    from mbem.la.preconditioner import BlockGaussSeidel

    t0 = time.time()
    # PINNED OFF for the equivalence check: with the near-only coupling on,
    # the grouped applier deliberately computes something DIFFERENT from the
    # per-term loop (a preconditioner, not the operator), so comparing them
    # would be comparing two things that are not meant to agree. The flag's
    # own effect is checked below.
    held_near = defaults.PRECOND_LOWER_NEAR_ONLY
    defaults.PRECOND_LOWER_NEAR_ONLY = False
    try:
        Mg = BlockGaussSeidel(hg, grouping="patch")  # grouped lower couplings
        Mp = BlockGaussSeidel(hp, grouping="patch")  # the per-term loop
        grouped_sbs = len(Mg._lower)
        r = np.random.default_rng(5).standard_normal(system.layout.n_unknowns)
        zg, zp = Mg(r), Mp(r)
        rel = float(np.max(np.abs(zg - zp))
                    / max(float(np.max(np.abs(zp))), 1e-300))
        # The sweep must also be doing SOMETHING lower-triangular, or the two
        # agree trivially: compare against the same preconditioner with every
        # lower coupling dropped, which is what the silent bug produced.
        keep = [sb.lower_terms for sb in Mp.sbs]
        for sb in Mp.sbs:
            sb.lower_terms = []
        zj = Mp(r)
        for sb, lt in zip(Mp.sbs, keep):
            sb.lower_terms = lt
        jac = float(np.max(np.abs(zj - zp))
                    / max(float(np.max(np.abs(zp))), 1e-300))
        # And the near-only coupling must be a REAL approximation: between
        # the full coupling and none at all. Equal to either would mean the
        # flag does nothing, or drops everything.
        defaults.PRECOND_LOWER_NEAR_ONLY = True
        Mn = BlockGaussSeidel(hg, grouping="patch")
        zn = Mn(r)
        d_full = float(np.max(np.abs(zn - zg))
                       / max(float(np.max(np.abs(zg))), 1e-300))
        d_none = float(np.max(np.abs(zn - zj))
                       / max(float(np.max(np.abs(zj))), 1e-300))
    finally:
        defaults.PRECOND_LOWER_NEAR_ONLY = held_near
    print(f"    preconditioner: grouped lower couplings on {grouped_sbs} of "
          f"{len(Mg.sbs)} super-blocks vs the per-term loop {rel:.3e}; "
          f"none differs {jac:.3e}; near-only sits between "
          f"({d_full:.3e} from full, {d_none:.3e} from none)   "
          f"({time.time() - t0:.0f}s)")
    ok &= (rel < VARIANT_TOL and grouped_sbs > 0 and jac > 1e-6
           and d_full > 1e-9 and d_none > 1e-9)
    return ok and worst < VARIANT_TOL


# ---------------------------------------------------------------------
# e / f / g. p, the eps pass, and the interpolation domain
# ---------------------------------------------------------------------

def check_p_convergence() -> bool:
    """[e] The error must fall at every step of p = 4..8."""
    field, source = _panels()
    eps = kb.resolve_eps(EPS, source)
    ok = True
    for kernel in (KERNEL_U, KERNEL_T):
        c = np.asarray(kernel_coeffs(kernel, MAT))
        exact = _exact_pair(field, source, kernel, MAT, eps)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(3 * source.n_triangles)
        ref = exact @ x
        errs = []
        for p in range(4, 9):
            pair = PairFMM(field, source, kernel, eps, p=p)
            errs.append(_errs(pair.matvec(c, x), ref))
        line = "  ".join(f"p={p}: {e[0]:.2e}"
                         for p, e in zip(range(4, 9), errs))
        print(f"    [{kernel}] 2-norm   {line}")
        print(f"    [{kernel}] max      " + "  ".join(
            f"p={p}: {e[1]:.2e}" for p, e in zip(range(4, 9), errs)))
        falls = all(errs[i + 1][0] < errs[i][0] for i in range(len(errs) - 1))
        gain = errs[0][0] / errs[-1][0]
        print(f"    [{kernel}] monotone: {falls}; p=4 -> p=8 gain "
              f"{gain:.3g}x (need {defaults.FMM_P_CONVERGENCE_GAIN:g})")
        ok &= falls and gain >= defaults.FMM_P_CONVERGENCE_GAIN
    return ok


def check_eps_terms() -> bool:
    """[f] The eps^2 pass must reduce the error, on a pair and end to end.

    How much it can reduce it is bounded by where the INTERPOLATION error
    sits. On the two panels the interpolation error is 1e-9 and the eps^2
    term is the whole of the remaining error, so the pass buys orders. End
    to end at eps="auto" the eps^2 term is ~(eps/r)^2 = 2.5e-3 of the far
    field and the far field is 4.5 % of the operator, i.e. ~1e-4 of ||A v||
    -- above the interpolation error at p = 6 and BELOW it at p = 4, where
    the comparison therefore cannot show anything and does not.
    """
    field, source = _panels()
    eps = kb.resolve_eps(EPS, source)
    rng = np.random.default_rng(4)
    x = rng.standard_normal(3 * source.n_triangles)
    ok = True
    for kernel in (KERNEL_U, KERNEL_T):
        c = np.asarray(kernel_coeffs(kernel, MAT))
        ref = _exact_pair(field, source, kernel, MAT, eps) @ x
        e = {}
        for terms in (1, 2):
            pair = PairFMM(field, source, kernel, eps, p=ORDER[kernel],
                           eps_terms=terms)
            e[terms] = _errs(pair.matvec(c, x), ref)[0]
        print(f"    [{kernel}] panels eps/h = {defaults.EPS_OVER_H}: "
              f"1 pass {e[1]:.3e}  2 passes {e[2]:.3e}   "
              f"gain x{e[1] / e[2]:.3g}")
        ok &= e[1] / e[2] >= defaults.FMM_EPS_TERM_GAIN

    _model, system, dense, _geom, _arrays = _zone()
    v = _test_vectors(system, np.random.default_rng(2))[0][1]
    ref = dense.A @ v
    gains = {}
    for p in (4, 6):
        e = {}
        for terms in (1, 2):
            _pairs, hasm = _operator({KERNEL_U: p, KERNEL_T: p},
                                     eps_terms=terms)
            e[terms] = float(np.max(np.abs(hasm.matvec(v) - ref))
                             / np.max(np.abs(ref)))
        gains[p] = e[1] / e[2]
        print(f"    operator p = {p}, gaussian: 1 pass {e[1]:.3e}  2 passes "
              f"{e[2]:.3e}   gain x{gains[p]:.3g}")
    print(f"    at p = 4 the interpolation error is above the eps^2 term, so "
          f"the pass cannot show there and does not (x{gains[4]:.2f})")
    return ok and gains[6] >= defaults.FMM_EPS_TERM_GAIN_OPERATOR


def _m2l_tables(gm) -> tuple:
    """``(distinct M2L tables, V box pairs)`` on ``gm``'s own M2L lattices.

    Two V pairs share ONE table iff their target and source node lattices
    coincide up to a single translation -- iff the two half-widths and the
    centre offset agree, which is what the key quantizes. Counting
    (level, offset) instead counts CUBES and returns the same number for
    every domain, the per-box extents included, where nothing is shared.
    """
    R = gm.tree.root_edge
    sdom = tdom = gm.cube_dom
    if sdom is None:
        sdom, tdom = gm.src_dom, gm.tgt_dom
    keys, n = set(), 0
    for a, lst in gm.lists.V.items():
        for b in lst:
            keys.add((tuple(np.round(tdom[1][a] / R, 9)),
                      tuple(np.round(sdom[1][b] / R, 9)),
                      tuple(np.round((sdom[0][b] - tdom[0][a]) / R, 9))))
            n += 1
    return len(keys), n


def _v_separation(gm, p: int) -> float:
    """The closest V-list NODE pair, in box edges.

    What every domain inflation spends: the interpolation error is bounded
    through the kernel's analyticity BETWEEN the two node sets, and the
    eps^2 expansion through (eps/r)^4 at the same r. Separable -- the
    lattices are tensor products, so the minimum of the squared distance
    is the sum of the per-axis minima.
    """
    R = gm.tree.root_edge
    sdom = tdom = gm.cube_dom
    if sdom is None:
        sdom, tdom = gm.src_dom, gm.tgt_dom
    z = _cheb_nodes(p)
    worst = np.inf
    for a, lst in gm.lists.V.items():
        edge = R / (1 << int(gm.tree.level[a]))
        for b in lst:
            s = 0.0
            for d in range(3):
                u = tdom[0][a][d] + tdom[1][a][d] * z
                v = sdom[0][b][d] + sdom[1][b][d] * z
                s += float(np.abs(u[:, None] - v[None, :]).min()) ** 2
            worst = min(worst, np.sqrt(s) / edge)
    return float(worst)


def _wx_margin(gm) -> tuple:
    """``(W margin, X margin)``: how far OUTSIDE the other side's domain the
    point-evaluated paths evaluate, in units of that domain's half-width.

    V is node against node and ``_v_separation`` covers it. W and X are not:
    W evaluates a MULTIPOLE at a field point, which converges only outside
    the source box's own domain, and X builds a local expansion from exact
    source integrals and then interpolates it over the target box's domain,
    which converges only if the source is outside THAT. Both margins must
    exceed 1, and neither is implied by max|xhat|, which is about a box's
    own contents. A domain inflation spends them: the X margin is the
    target domain's to lose, so a source-only inflation leaves it alone.

    The X margin is read off the DOMAIN-BLIND list -- the entries emitted
    plus the entries ``FMM_X_MARGIN`` refused -- because it is a statement
    about what the domain does to the geometry, and the rule's own job is to
    keep the bad ones out of the list. Measuring the surviving entries would
    report the threshold back at every domain that has any.
    """
    tree = gm.tree
    sc, sh = gm.src_dom
    tc, th = gm.tgt_dom
    w = x = np.inf
    for a, lst in gm.lists.W.items():
        pts = gm.centroids[tree.elements_of(a)]
        if pts.size:
            for b in lst:
                w = min(w, float(np.abs((pts - sc[b]) / sh[b]
                                        ).max(axis=1).min()))
    for a in set(gm.lists.X) | set(gm.lists.X_demoted):
        for b in gm.lists.X.get(a, []) + gm.lists.X_demoted.get(a, []):
            held = tree.elements_of(b)
            if held.size:
                v = gm.verts[held].reshape(-1, 3)
                x = min(x, float(np.abs((v - tc[a]) / th[a]
                                        ).max(axis=1).min()))
    return w, x


def _floor_is_load_bearing(meshes, arrays) -> tuple:
    """``(flat boxes, elements whose P2M would be non-finite)`` with the
    half-width floor removed.

    A box holding one planar patch has ZERO extent across that plane, so
    without ``FMM_MIN_HALF_OVER_EDGE`` the P2M weight divides 0 by 0 and the
    multipole is NaN -- silently, since nothing raises. The floor is not a
    tidiness measure and this is the clause that says so.
    """
    keep = defaults.FMM_MIN_HALF_OVER_EDGE
    defaults.FMM_MIN_HALF_OVER_EDGE = 0.0
    try:
        gm = FmmTree(meshes, arrays=arrays, placement_safety=SAFETY)
        flat = int((gm.src_dom[1] == 0.0).any(axis=1).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            p2m = gm.stencil(4).p2m
        return flat, int((~np.isfinite(p2m)).any(axis=1).sum())
    finally:
        defaults.FMM_MIN_HALF_OVER_EDGE = keep


def check_domain() -> bool:
    """[g] The interpolation domain, and the three ways to have BOTH
    containment and a shared M2L table.

    An element protrudes up to 0.53 box edges, so the nominal cube -- the
    only lattice an M2L table can be shared on -- extrapolates at P2M,
    while the per-box extent contains every source and shares nothing.
    Each candidate buys the missing half back somewhere, and each is priced
    here on the same geometry against the same exact pair:

      inflate=f           the cube scaled by one factor at every box and
                          level, so the lattice stays translation-invariant
      inflate=(f, 1)      the same on the SOURCE side only; a target is a
                          collocation point and never protrudes
      domain="canonical"  per-box extents for P2M/L2P/W/X, the cube lattice
                          for M2L alone, with a change of basis between

    What an inflation spends is the V-list node separation: two
    non-adjacent boxes are 2 edges apart and their lattices span +-f/2 of
    an edge, so f near 2 leaves them touching and BOTH the interpolation
    and the eps^2 expansion lose their small parameter. The clause reports
    the separation beside the error for that reason.

    It also spends the W and X MARGINS, which max|xhat| does not see: W
    evaluates a multipole at a field point and X interpolates a local
    expansion built from exact source integrals, so each needs the other
    side's geometry OUTSIDE its own domain. The X margin belongs to the
    TARGET domain, which is why a source-only inflation leaves it exactly
    where the plain cube puts it. On this model both margins stay above 1
    at every factor; on ``topo_inclusion`` they do not, and that is where
    the uniform inflation dies -- measured there, the plain cube already
    has 2 of 2509 X records holding a source inside the target's own
    domain, the uniform inflation at the containment factor has 51, and
    the extent has none with a 1.21 margin to spare. This clause cannot
    reach that model (31,098 unknowns, no dense reference) and does not
    pretend to; what it pins is the mechanism and its direction.

    The half-width FLOOR is checked by removing it: a box holding one
    planar patch has zero extent across the plane, and without the floor
    its P2M is NaN rather than an error.

    The canonical transform is pinned as an IDENTITY, not as an error: the
    extent lattice reproduces every polynomial of degree < p per axis and a
    cube Chebyshev weight is one, so extent-P2M composed with the transform
    IS cube-P2M. Measured on the two panels, which have no W and no X, so
    the two configurations differ nowhere else.

    The single-p error does not settle the choice and can point the wrong
    way: a tight extent is a SMALLER domain than the cube wherever nothing
    protrudes, and larger only where something does, so at low p the cube
    can win on the pairs it extrapolates least. What settles it is the
    trend in p, because extrapolation amplifies like rho^p while
    interpolation converges like rho^-p.

    NOR DOES ONE PAIR SETTLE IT, and this clause prints one. Clause [d]
    rerun on the cube measures 5.3e-4 / 4.8e-4 / 1.9e-4 far-isolated
    against the extent's 4.3e-5 / 8.0e-5 / 6.3e-6 -- two of the three over
    FMM_OPERATOR_PARITY. The cube is 5x better at p = 8 on the pair
    printed here and 6-30x worse on the operator that pair belongs to.
    Read the ratio line as a statement about ONE pair of one model.
    """
    model, system, _dense, geom, _arrays = _zone()
    keys = _pair_keys(system)
    key, _worst = _worst_zone_pair(_zone_pairs(ORDER), "WX")
    fp, sp, kernel = keys[key]
    mat = _term_material(system, model, key)
    c = np.asarray(kernel_coeffs(kernel, mat))
    eps = kb.resolve_patch_eps(EPS, sp)
    exact = _exact_pair(fp.mesh, sp.mesh, kernel, mat, eps)
    ref = exact @ np.random.default_rng(5).standard_normal(exact.shape[1])
    x = np.random.default_rng(5).standard_normal(exact.shape[1])
    f_src, f_tgt = geom.containment_factors()
    f = float(np.ceil(100 * f_src) / 100)
    print(f"    {fp.name} <- {sp.name} [{kernel}], protrusion up to "
          f"{geom.tree.protrusion(geom.verts).max():.2f} box edges; "
          f"containment needs cube inflation {f_src:.3f} on the source, "
          f"{f_tgt:.3f} on the target")

    meshes = _zone_meshes(model)
    trees = {"extent": geom,
             "cube": FmmTree(meshes, domain="cube",
                             placement_safety=SAFETY),
             f"cube f={f:g}": FmmTree(meshes, domain="cube", inflate=f,
                                      placement_safety=SAFETY),
             f"src f={f:g}": FmmTree(meshes, domain="cube",
                                     inflate=(f, 1.0),
                                     placement_safety=SAFETY),
             "canonical": FmmTree(meshes, domain="canonical",
                                  placement_safety=SAFETY)}
    flat, nan_rows = _floor_is_load_bearing(meshes, _arrays)
    print(f"    half-width floor {defaults.FMM_MIN_HALF_OVER_EDGE:g} of the "
          f"cube edge: without it {flat} boxes get a zero source half-width "
          f"and {nan_rows} elements a non-finite P2M row (topo_inclusion: "
          f"423 boxes, 4678 of 8116 elements)")

    out, margins = {}, {}
    for domain, gm in trees.items():
        row = []
        for p in range(4, 9):
            pair = PairFMM(fp.mesh, sp.mesh, kernel, eps, p=p, geom=gm)
            row.append((_errs(pair.matvec(c, x), ref)[0], pair.st.xhat))
        out[domain] = row
        tab, n_v = _m2l_tables(gm)
        margins[domain] = _wx_margin(gm)
        x8 = row[-1][1]
        print(f"    {domain:11s} max|xhat| P2M {x8['p2m']:.3f} L2P "
              f"{x8['l2p']:.3f}" + (f" M2C {x8['m2c']:.3f}" if "m2c" in x8
                                    else " " * 10)
              + f"  V sep {_v_separation(gm, 8):.3f} edges  M2L tables "
              f"{tab}/{n_v}  W/X margin {margins[domain][0]:.3f}/"
              f"{margins[domain][1]:.3f}")
        print(f"    {'':11s} " + "  ".join(
            f"p={p}: {e:.2e}" for p, (e, _h) in zip(range(4, 9), row)))
    ratio = [c / e for (e, _1), (c, _2) in zip(out["extent"], out["cube"])]
    print("    cube over extent: " + "  ".join(
        f"p={p}: x{r:.2f}" for p, r in zip(range(4, 9), ratio)))

    # The canonical domain is the cube's own operator wherever the two can
    # differ only through M2L -- on the panels, which are all V list.
    field, source = _panels()
    e0 = kb.resolve_eps(EPS, source)
    v = np.random.default_rng(8).standard_normal(3 * source.n_triangles)
    y = {}
    for domain in ("cube", "canonical"):
        pr = PairFMM(field, source, KERNEL_U, e0, p=6, domain=domain)
        y[domain] = pr.matvec(np.asarray(kernel_coeffs(KERNEL_U, MAT)), v)
        wx = pr.counts()["W"] + pr.counts()["X"]
    same = float(np.linalg.norm(y["canonical"] - y["cube"])
                 / np.linalg.norm(y["cube"]))
    print(f"    canonical vs cube on the panels (W+X = {wx}): {same:.2e} -- "
          f"the transform reproduces the cube lattice exactly, so it moves "
          f"the extrapolation rather than removing it")

    shared = {d: _m2l_tables(g)[0] for d, g in trees.items()}
    n_v = _m2l_tables(geom)[1]
    return (all(h["p2m"] <= 1.0 + 1e-12 and h["l2p"] <= 1.0 + 1e-12
                for _e, h in out["extent"])
            and all(h["p2m"] > 1.0 for _e, h in out["cube"])
            and all(h["p2m"] <= 1.0 + 1e-12 and h["l2p"] <= 1.0 + 1e-12
                    for _e, h in out[f"src f={f:g}"])
            and all(h["m2c"] > 1.0 for _e, h in out["canonical"])
            and same < 1e-12
            and shared["extent"] > 0.9 * n_v
            and len({shared[d] for d in trees if d != "extent"}) == 1
            and (_v_separation(trees["cube"], 8)
                 > _v_separation(trees[f"src f={f:g}"], 8)
                 > _v_separation(trees[f"cube f={f:g}"], 8))
            and nan_rows > 0
            and min(margins["extent"]) > 1.0
            # the X margin is the TARGET domain's: a uniform inflation
            # spends it, a source-only one leaves it at the cube's value.
            and (margins["extent"][1] > margins["cube"][1]
                 > margins[f"cube f={f:g}"][1])
            and margins[f"src f={f:g}"][1] == margins["cube"][1])


CHECKS = [
    ("point kernel vs the analytic triangle kernel", check_point_kernel),
    ("pair level: matvec and matrix vs the exact pair", check_pair),
    ("far-field isolation on a four-list pair", check_far_isolation),
    ("operator level vs AssembledDense", check_operator),
    ("convergence in p", check_p_convergence),
    ("the eps^2 pass", check_eps_terms),
    ("interpolation domain", check_domain),
]


def main() -> int:
    failed = []
    for name, fn in CHECKS:
        print(f"\n[{name}]")
        t0 = time.time()
        try:
            ok = fn()
        except Exception as exc:                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok = False
            print(f"    raised {exc!r}")
        print(f"    -> {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
        if not ok:
            failed.append(name)
    if failed:
        print("\nFAIL: " + "; ".join(failed))
        return 1
    print(f"\nPASS: reference bbFMM, {len(CHECKS)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
