"""Gate for ddbem's DD influence matrices (P0/P1/P2).

Five parts, in the order of the Stage-1 spec.

1. P0 PARITY WITH msd, ENTRYWISE.  ``order=0`` must reproduce msd's frozen P0
   DD assembly.  Four independent msd entry points are used, so the check does
   not rest on one code path:
     * ``mollified_kernel/analytical_batch.py::assemble_T_matrix_batch``
       -- the legacy (3N_f, 3N_s) slip -> displacement assembly that
       ``mollified_bem.assemble_BEM_matrices(kernel="T")`` calls;
     * ``mbem/kernels/tri_kernels.py::t_matrix_direct`` with
       ``mbem.kernels.basis.t_coeffs(mu, lam)`` -- the numba in-loop assembly
       the DenseBackend ``mode="direct"`` path uses;
     * ``mollified_kernel/analytical_kernels.py::analytical_dd_displacement``
       and ``::analytical_stress_kernel`` -- the frozen scalar oracles;
     * ``mbem/kernels/tri_kernels.py::dd_stress_contract`` driven with one-hot
       nodal densities to recover the slip -> stress matrix column by column
       (the mbem stress path is matrix-free, so this is the only way to reach
       it entrywise).
   The measure is BLOCK-relative (``_common.block_relmax``): entries inside a
   (field, source) block pass through zero, so a per-entry relative measure is
   meaningless; each 3x3 / 6x3 block is normalised by its own largest entry.
   Tolerance 1e-11 as specified.  MEASURED WORST, and the honest margin:
   5e-14..2e-13 at nu = 0.25 and 0.30, but 2.9e-12 at nu = 0.45 -- only 3.4x
   under the tolerance.  That worst block is the on-plane observer
   [1.6, -0.3, 0] against the SLIVER triangle at 1.4 L, whose own scale is
   4.1e-3 against a global 51, so the block-relative measure is amplifying the
   last bits of two different closed forms that both lose digits there
   (nu = 0.45 makes c34 = 3 - 4 nu = 1.2, so the cancellation is worse).  The
   whole-matrix relative difference is 3.6e-15.  If this number ever moves by
   orders of magnitude the kernels have changed; if it drifts to 8e-12 the
   sliver observer, not the tolerance, is what should be revisited.
   ``far_field="analytic"`` is used on BOTH sides, because the msd oracles are
   closed form everywhere while ddbem defaults to clq's hybrid producer -- with
   the test observers inside ~4 L of the elements the two agree to rounding, but
   comparing hybrid against analytic would be comparing two different (both
   correct) quadratures.  Every OTHER gate in this file runs the shipped
   default producer.

2. nu != 1/4.  Every parity check runs at nu = 0.25, 0.30 AND 0.45.  The
   lam/mu pairing bug in the slip -> displacement contraction is EXACTLY
   invisible at nu = 1/4 (lam = mu) and survived months in this tree.  A
   tripwire re-derives the swapped form from msd's ``integrate_DG`` and asserts
   it differs by > 1e-2 at nu = 0.30 and coincides at nu = 0.25, so the sweep
   is demonstrably discriminating and not just three copies of one number.

3. PARTITION OF UNITY, and node ORDER pinned separately.  With every nodal slip
   equal, P1 and P2 must reproduce P0 with that slip (sum_k N_k = 1).  That
   check is invariant under any PERMUTATION of the nodes, so it cannot pin the
   node order on its own; the order is pinned separately by comparing each
   individual node's column block against a quadrature that uses an
   INDEPENDENT barycentric basis (``_common.shape_independent``: textbook
   signed-area barycentrics and the textbook P2 polynomials, not clq's
   monomial machinery).  Swapping any two nodes fails that check.

4. EIGENSTRESS.  (a) ``stress_matrix(subtract_eigenstress=True)`` equals
   ``stress_matrix(False) - eigenstress_matrix(...)`` to 0.0 exactly -- the
   subtraction is a single expression, not a re-derivation.  (b) the
   eigenstress is the EXACT finite-triangle one: ``E[n,k]`` is gated against an
   independent quadrature of ``N_k(y) * phi_eps(x-y)`` over the real triangle
   with the Cortez blob ``phi_eps = 15 eps^4 / (8 pi R^7)``.  (c) it is NOT the
   point/marginal approximation of ``msd/anelastic.py``
   (``rho = 0.75 eps^4 / (d^2+eps^2)^2.5`` at the distance to the nearest
   triangle, the infinite-plane limit): the two are gated to AGREE deep inside
   a large element (both correct there) and to DISAGREE by a large factor near
   an element edge.  Without (c) the quadrature gate in (b) could not be
   trusted to discriminate, because an approximate subtraction was a real bug
   in this tree's history.

5. QUADRATURE for a non-constant nodal slip at off-element field points.
   Displacement and total stress from the assembled matrices, contracted with
   an asymmetric nodal slip, against a Duffy-collapsed Gauss rule over the
   triangle using msd's frozen POINT kernels (``kelvin_dG_pointwise``,
   ``dd_stress_kernel``) and the independent barycentric basis.  The gate
   tolerance is set above the measured quadrature residual (the n_quad/2 vs
   n_quad difference), which is printed, so a loosened tolerance would be
   visible.
"""
from __future__ import annotations

import sys

import numpy as np

from _common import (MU, NU_SWEEP, Report, block_relmax, relmax,
                     gauss_triangle, shape_independent,
                     msd_analytical, msd_batch, msd_points, msd_anelastic,
                     msd_mbem, test_triangles, test_observers)

import ddbem
from ddbem import defaults

EPS = 0.15
# The msd oracles are closed form everywhere, so the PARITY gate runs clq's
# "analytic" producer on both sides; every other gate runs the SHIPPED default
# (clq's hybrid producer), which is what production code will use.
FF_PARITY = "analytic"
FF = defaults.FAR_FIELD
TOL_PARITY = 1e-11                  # spec; observed ~4e-15
TOL_IDENTITY = 1e-12                # exact identities through different tables
N_QUAD = 32                         # point-kernel quadrature (Duffy-collapsed)
N_QUAD_COARSE = 24                  # convergence probe for the same
N_QUAD_BLOB = 240                   # blob quadrature (1/R^7 is sharply peaked)
N_QUAD_BLOB_COARSE = 160            # convergence probe for the same


# ===========================================================================
# 1 + 2.  P0 parity against msd, at nu = 0.25 / 0.30 / 0.45
# ===========================================================================

def swapped_pre_fix_U(ak, obs, v1, v2, v3, normal, mu, nu, eps):
    """msd's PRE-2026-09-04 contraction (lam and mu swapped on the first two
    terms).  Rebuilt here from the frozen ``integrate_DG`` so the tripwire does
    not depend on any surviving copy of the bug."""
    G1 = ak.integrate_DG(v1, v2, v3, obs, mu, nu, eps)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    U = np.zeros((3, 3))
    for i in range(3):
        tr = sum(G1[i, m, m] for m in range(3))
        for j in range(3):
            U[i, j] = -(lam * sum(normal[m] * G1[i, j, m] for m in range(3))
                        + mu * normal[j] * tr
                        + mu * sum(normal[k] * G1[i, k, j] for k in range(3)))
    return U


def part_parity(rep):
    tri = test_triangles()
    obs = test_observers()
    n_s, n_f = tri.shape[0], obs.shape[0]
    nrm = np.ascontiguousarray(ddbem.element_normals(tri))
    eps_arr = np.full(n_s, EPS)
    ak = msd_analytical()
    ab = msd_batch()
    tk, t_coeffs = msd_mbem()
    v = tri

    for nu in NU_SWEEP:
        lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)

        # ---- slip -> displacement -----------------------------------------
        A = ddbem.displacement_matrix(obs, tri, EPS, MU, nu, 0, far_field=FF_PARITY)
        B_batch = ab.assemble_T_matrix_batch(obs, tri, nrm, MU, nu, EPS)
        rep.check(f"nu={nu}: U vs msd assemble_T_matrix_batch",
                  block_relmax(A, B_batch, n_f, n_s, 3, 3), TOL_PARITY)

        B_numba = tk.t_matrix_direct(obs, tri, nrm, eps_arr, *t_coeffs(MU, lam))
        rep.check(f"nu={nu}: U vs mbem t_matrix_direct",
                  block_relmax(A, B_numba, n_f, n_s, 3, 3), TOL_PARITY)

        B_scal = np.zeros_like(A)
        for s in range(n_s):
            for f in range(n_f):
                B_scal[3 * f:3 * f + 3, 3 * s:3 * s + 3] = ak.analytical_dd_displacement(
                    obs[f], v[s, 0], v[s, 1], v[s, 2], nrm[s], MU, nu, EPS)
        rep.check(f"nu={nu}: U vs msd analytical_dd_displacement (scalar)",
                  block_relmax(A, B_scal, n_f, n_s, 3, 3), TOL_PARITY)

        # ---- slip -> stress (TOTAL: msd's kernels carry no subtraction) ----
        S = ddbem.stress_matrix(obs, tri, EPS, MU, nu, 0,
                                subtract_eigenstress=False, far_field=FF_PARITY)
        S_scal = np.zeros_like(S)
        for s in range(n_s):
            for f in range(n_f):
                H = ak.analytical_stress_kernel(obs[f], v[s, 0], v[s, 1], v[s, 2],
                                                nrm[s], MU, nu, EPS)
                S_scal[6 * f:6 * f + 6, 3 * s:3 * s + 3] = ddbem.tensor_to_voigt(
                    np.moveaxis(H, 2, 0)).T
        rep.check(f"nu={nu}: H vs msd analytical_stress_kernel (scalar)",
                  block_relmax(S, S_scal, n_f, n_s, 6, 3), TOL_PARITY)

        S_numba = np.zeros_like(S)
        for s in range(n_s):
            for j in range(3):
                d = np.zeros((n_s, 3))
                d[s, j] = 1.0
                sig = tk.dd_stress_contract(obs, tri, nrm, eps_arr,
                                            np.ascontiguousarray(d), MU, nu)
                S_numba[:, 3 * s + j] = ddbem.tensor_to_voigt(sig).reshape(-1)
        rep.check(f"nu={nu}: H vs mbem dd_stress_contract (one-hot columns)",
                  block_relmax(S, S_numba, n_f, n_s, 6, 3), TOL_PARITY)

        # ---- slip -> traction ---------------------------------------------
        n_field = np.ascontiguousarray(
            np.tile(np.array([[0.3, -0.5, 0.81240384]]), (n_f, 1)))
        n_field /= np.linalg.norm(n_field, axis=1, keepdims=True)
        T = ddbem.traction_matrix(obs, n_field, tri, EPS, MU, nu, 0,
                                  subtract_eigenstress=False, far_field=FF_PARITY)
        sig_ref = ddbem.voigt_to_tensor(S_numba.reshape(n_f, 6, -1).transpose(0, 2, 1))
        T_ref = np.einsum("ncml,nl->ncm", sig_ref, n_field).transpose(0, 2, 1
                                                                     ).reshape(3 * n_f, -1)
        rep.check(f"nu={nu}: t = sigma.n vs mbem dd_stress_contract",
                  block_relmax(T, T_ref, n_f, n_s, 3, 3), TOL_PARITY)

        # ---- tripwire: the swapped pairing --------------------------------
        U_swap = np.zeros_like(A)
        for s in range(n_s):
            for f in range(n_f):
                U_swap[3 * f:3 * f + 3, 3 * s:3 * s + 3] = swapped_pre_fix_U(
                    ak, obs[f], v[s, 0], v[s, 1], v[s, 2], nrm[s], MU, nu, EPS)
        d = block_relmax(A, U_swap, n_f, n_s, 3, 3)
        if nu == 0.25:
            rep.check(f"nu={nu}: swapped pairing coincides (lam == mu)", d, 1e-12)
        else:
            rep.check_bool(f"nu={nu}: TRIPWIRE swapped pairing differs > 1e-2",
                           d > 1e-2, f"(block-rel diff {d:.2e})")


# ===========================================================================
# 3.  Partition of unity, and node order
# ===========================================================================

def part_partition(rep):
    tri = test_triangles()
    obs = test_observers()
    n_s, n_f = tri.shape[0], obs.shape[0]
    n_field = np.ascontiguousarray(np.tile(np.array([[0.0, 0.0, 1.0]]), (n_f, 1)))
    s0 = np.array([0.7, -0.4, 0.25])

    for nu in NU_SWEEP:
        A0 = ddbem.displacement_matrix(obs, tri, EPS, MU, nu, 0, far_field=FF)
        S0 = ddbem.stress_matrix(obs, tri, EPS, MU, nu, 0, far_field=FF)
        T0 = ddbem.traction_matrix(obs, n_field, tri, EPS, MU, nu, 0, far_field=FF)
        E0 = ddbem.eigenstress_matrix(obs, tri, EPS, MU, nu, 0, far_field=FF)
        x0 = np.tile(s0, n_s)
        for p in (1, 2):
            K = ddbem.n_nodes(p)
            xp = np.tile(s0, n_s * K)
            for name, M0, Mp in (
                    ("U", A0, ddbem.displacement_matrix(obs, tri, EPS, MU, nu, p,
                                                        far_field=FF)),
                    ("sigma_el", S0, ddbem.stress_matrix(obs, tri, EPS, MU, nu, p,
                                                         far_field=FF)),
                    ("t", T0, ddbem.traction_matrix(obs, n_field, tri, EPS, MU, nu, p,
                                                    far_field=FF)),
                    ("C:eps*", E0, ddbem.eigenstress_matrix(obs, tri, EPS, MU, nu, p,
                                                            far_field=FF))):
                rep.check(f"nu={nu}: P{p} partition of unity, {name}",
                          relmax(Mp @ xp, M0 @ x0), TOL_IDENTITY)


def quad_reference(tri, order, obs, mu, nu, eps, n_quad, pts_mod, ak_mod,
                   want=("U", "H")):
    """Independent nodal influence by quadrature of msd's frozen POINT kernels
    times the independent barycentric shape functions."""
    y, w = gauss_triangle(tri, n_quad)
    Nq = shape_independent(tri, order, y)                    # (Q, K)
    nhat = ddbem.mesh.unit_normal(tri)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    K = Nq.shape[1]
    N = obs.shape[0]
    out = {}
    if "U" in want:
        out["U"] = np.zeros((N, K, 3, 3))
    if "H" in want:
        out["H"] = np.zeros((N, K, 3, 3, 3))
    for q in range(y.shape[0]):
        wk = w[q] * Nq[q]
        for n in range(N):
            d = obs[n] - y[q]
            if "U" in want:
                DG = ak_mod.kelvin_dG_pointwise(d, mu, nu, eps)
                Uq = np.zeros((3, 3))
                for i in range(3):
                    tr = DG[i, 0, 0] + DG[i, 1, 1] + DG[i, 2, 2]
                    for j in range(3):
                        Uq[i, j] = -(mu * (nhat @ DG[i, j])
                                     + lam * nhat[j] * tr
                                     + mu * (nhat @ DG[i, :, j]))
                out["U"][n] += wk[:, None, None] * Uq
            if "H" in want:
                Hq = pts_mod.dd_stress_kernel(obs[n], y[q], nhat, mu, nu, eps)
                out["H"][n] += wk[:, None, None, None] * Hq
    return out


#   Field points for the point-kernel quadrature gates: OFF the element, at
#   3.5-6 eps standoff from its plane, on both sides and inside/outside the
#   footprint.  The standoff matters: the Duffy-collapsed rule has to resolve a
#   feature of width max(eps, standoff) across an element of size ~1, and at a
#   0.15-eps standoff even n_quad = 32 is only good to ~5e-3 (measured).  Here
#   n_quad = 32 is at rounding and n_quad = 24 is within 5e-14 of it.
QUAD_TRI = 0
QUAD_OBS = np.array([[0.40, 0.30, 0.80],      # +0.78 above the plane
                     [-0.50, 0.40, -0.90],    # -0.92 below, outside the footprint
                     [1.60, 1.40, 0.70],      # +0.62, well outside
                     [0.45, 0.30, 0.55]])     # +0.53, over the interior


def part_node_order(rep):
    """Pin the NODE ORDER against the independent barycentric basis."""
    tri = test_triangles()[QUAD_TRI]
    obs = QUAD_OBS
    pts = msd_points()
    ak = msd_analytical()
    nu = 0.30
    for p in (0, 1, 2):
        K = ddbem.n_nodes(p)
        ref = quad_reference(tri, p, obs, MU, nu, EPS, N_QUAD, pts, ak)
        ref_c = quad_reference(tri, p, obs, MU, nu, EPS, N_QUAD_COARSE, pts, ak)
        resid = max(relmax(ref_c["U"], ref["U"]), relmax(ref_c["H"], ref["H"]))
        tol = max(10.0 * resid, 1e-11)
        A = ddbem.displacement_matrix(obs, tri[None], EPS, MU, nu, p, far_field=FF)
        S = ddbem.stress_matrix(obs, tri[None], EPS, MU, nu, p,
                                subtract_eigenstress=False, far_field=FF)
        # column block of node k, component j:  [n,k,i,j] -> row 3n+i, col 3k+j
        Aq = ref["U"].transpose(0, 2, 1, 3).reshape(3 * obs.shape[0], 3 * K)
        # [n,k,m,l,j] -> [n,k,j,m,l] -> Voigt [n,k,j,v] -> row 6n+v, col 3k+j
        Sq = ddbem.tensor_to_voigt(np.moveaxis(ref["H"], 4, 2))
        Sq = Sq.transpose(0, 3, 1, 2).reshape(6 * obs.shape[0], 3 * K)
        rep.check(f"P{p}: per-node U vs independent-basis quadrature",
                  relmax(A, Aq), tol, f"[quad resid {resid:.1e}]")
        rep.check(f"P{p}: per-node H vs independent-basis quadrature",
                  relmax(S, Sq), tol, f"[quad resid {resid:.1e}]")
        if p > 0:
            # a node swap must break it -- otherwise the check pins nothing
            perm = np.arange(K)
            perm[0], perm[1] = perm[1], perm[0]
            Aperm = A.reshape(-1, K, 3)[:, perm].reshape(A.shape)
            rep.check_bool(f"P{p}: TRIPWIRE node swap 0<->1 breaks it",
                           relmax(Aperm, Aq) > 1e-3,
                           f"(rel {relmax(Aperm, Aq):.2e})")


# ===========================================================================
# 4.  Eigenstress
# ===========================================================================

def part_eigenstress(rep):
    tri = test_triangles()
    obs = test_observers()
    n_s, n_f = tri.shape[0], obs.shape[0]

    # (a) elastic == total - C:eps*, exactly
    for nu in NU_SWEEP:
        for p in (0, 1, 2):
            el = ddbem.stress_matrix(obs, tri, EPS, MU, nu, p, far_field=FF)
            tot = ddbem.stress_matrix(obs, tri, EPS, MU, nu, p,
                                      subtract_eigenstress=False, far_field=FF)
            eig = ddbem.eigenstress_matrix(obs, tri, EPS, MU, nu, p, far_field=FF)
            rep.check_bool(f"nu={nu} P{p}: elastic == total - C:eps* exactly",
                           np.array_equal(el, tot - eig),
                           f"(max |diff| {np.abs(el - (tot - eig)).max():.1e})")

    # (b) E is the EXACT finite-triangle blob-weighted shape integral.
    #     The observers sit where the eigenstress actually matters: ON the
    #     element plane, at 1 and 3 eps above it, next to an edge, and at 8 eps
    #     where it must already be negligible.
    t1 = tri[0]
    nu = 0.30
    nhat = ddbem.mesh.unit_normal(t1)
    cen = t1.mean(axis=0)
    obs_e = np.array([cen, cen + EPS * nhat, cen + 3.0 * EPS * nhat,
                      0.5 * (t1[0] + t1[1]) + 0.3 * EPS * nhat,
                      cen + 8.0 * EPS * nhat])
    for p in (0, 1, 2):
        K = ddbem.n_nodes(p)
        Eq = np.zeros((obs_e.shape[0], K))
        Eq_c = np.zeros_like(Eq)
        for n_q, store in ((N_QUAD_BLOB, Eq), (N_QUAD_BLOB_COARSE, Eq_c)):
            y, w = gauss_triangle(t1, n_q)
            Nq = shape_independent(t1, p, y)
            R2 = np.sum((obs_e[:, None, :] - y[None]) ** 2, axis=2) + EPS ** 2
            phi = 15.0 * EPS ** 4 / (8.0 * np.pi * R2 ** 3.5)
            store[:] = (phi * w[None, :]) @ Nq
        resid = relmax(Eq_c, Eq)
        sig0 = ddbem.eigen_column_tensor(nhat, MU, nu)
        ref = np.einsum("nk,mlj->nkmlj", Eq, sig0)
        ref = ddbem.tensor_to_voigt(np.moveaxis(ref, 4, 2))      # (N,K,3,6)
        ref = ref.transpose(0, 3, 1, 2).reshape(6 * obs_e.shape[0], 3 * K)
        got = ddbem.eigenstress_matrix(obs_e, t1[None], EPS, MU, nu, p, far_field=FF)
        tol = max(20.0 * resid, 1e-11)
        rep.check(f"P{p}: C:eps* vs exact finite-triangle blob quadrature",
                  relmax(got, ref), tol, f"[quad resid {resid:.1e}]")

    # (c) exact != the msd point/marginal approximation near an edge,
    #     but == deep inside a LARGE element (the infinite-plane limit)
    an = msd_anelastic()
    L = 40.0 * EPS
    big = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [0.5 * L, 0.866 * L, 0.0]])
    slip = np.array([1.0, 0.0, 0.0])
    cen = big.mean(axis=0)
    deep = np.array([cen + np.array([0.0, 0.0, z]) for z in (0.0, EPS, 2.0 * EPS)])
    edge = np.array([[0.5 * L, 0.0, 0.0],                      # on an edge
                     [0.5 * L, 0.3 * EPS, 0.0],                # just inside it
                     big[0] + np.array([0.0, 0.0, 0.0])])      # at a vertex
    for tag, pts_, lo, hi in (("deep interior", deep, None, 1e-4),
                              ("edge / vertex", edge, 0.25, None)):
        Em = ddbem.eigenstress_matrix(pts_, big[None], EPS, MU, nu, 0, far_field=FF)
        got = ddbem.voigt_to_tensor((Em @ slip).reshape(-1, 6))
        ref = an.eigenstress_at_points(pts_, big[None], slip, MU, nu, EPS)
        r = relmax(got, ref)
        if hi is not None:
            rep.check(f"C:eps* == msd point/marginal, {tag} of a 40-eps element",
                      r, hi)
        else:
            rep.check_bool(f"C:eps* != msd point/marginal, {tag} (> {lo:.0%})",
                           r > lo, f"(rel diff {r:.2f} -- the approximation is "
                                   f"the infinite-plane limit)")


# ===========================================================================
# 5.  Quadrature with a non-constant nodal slip, off-element field points
# ===========================================================================

def part_quadrature(rep):
    tri = test_triangles()[QUAD_TRI]
    obs = QUAD_OBS
    pts = msd_points()
    ak = msd_analytical()
    rng = np.random.default_rng(7)
    for nu in NU_SWEEP:
        for p in (1, 2):
            K = ddbem.n_nodes(p)
            slip = rng.normal(size=(K, 3))               # asymmetric, non-constant
            ref = quad_reference(tri, p, obs, MU, nu, EPS, N_QUAD, pts, ak)
            ref_c = quad_reference(tri, p, obs, MU, nu, EPS, N_QUAD_COARSE, pts, ak)
            u_ref = np.einsum("nkij,kj->ni", ref["U"], slip)
            s_ref = np.einsum("nkmlj,kj->nml", ref["H"], slip)
            u_res = relmax(np.einsum("nkij,kj->ni", ref_c["U"], slip), u_ref)
            s_res = relmax(np.einsum("nkmlj,kj->nml", ref_c["H"], slip), s_ref)
            A = ddbem.displacement_matrix(obs, tri[None], EPS, MU, nu, p, far_field=FF)
            S = ddbem.stress_matrix(obs, tri[None], EPS, MU, nu, p,
                                    subtract_eigenstress=False, far_field=FF)
            u_got = (A @ slip.reshape(-1)).reshape(-1, 3)
            s_got = ddbem.voigt_to_tensor((S @ slip.reshape(-1)).reshape(-1, 6))
            rep.check(f"nu={nu} P{p}: U(non-constant slip) vs point-kernel quadrature",
                      relmax(u_got, u_ref), max(10.0 * u_res, 1e-11),
                      f"[quad resid {u_res:.1e}]")
            rep.check(f"nu={nu} P{p}: H(non-constant slip) vs point-kernel quadrature",
                      relmax(s_got, s_ref), max(10.0 * s_res, 1e-11),
                      f"[quad resid {s_res:.1e}]")


def main():
    rep = Report("ddbem DD influence matrices, P0/P1/P2")
    part_parity(rep)
    part_partition(rep)
    part_node_order(rep)
    part_eigenstress(rep)
    part_quadrature(rep)
    ok = rep.finish()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
