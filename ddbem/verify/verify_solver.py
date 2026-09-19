"""Gate: collocation rows, boundary conditions, free term, and the solve.

    python verify/verify_solver.py          # from the ddbem root

WHAT IS PINNED HERE, AND WHY EACH CHECK IS NOT VACUOUS
======================================================

1. CONVENTIONS.  ``ddbem.shape_at`` (the barycentric shape functions the free
   term is built from) against ``clq.shape_functions`` (which projects
   Cartesian points into the element frame -- a completely different code
   path), with a node-swap tripwire.  Closure and orientation of the shapes.
   The three refusals the design depends on (INTERFACE, calibration on an open
   boundary, exterior rows with a non-traction-free patch).

2. THE FREE TERM IS A ROW-SUM IDENTITY.  A rigid translation of the body is the
   DD density ``q = -sigma c``, and every row's exact value on it is known
   (``u_int = c``, ``u_ext = 0``, ``t = 0``).  Three separate facts:

   (a) the measured row sum of a DISPLACEMENT row is exactly ``phi(x_c) I``, a
       SCALAR times the identity -- the mollified smoothed indicator at the
       collocation point.  The classical free term is the assertion
       ``phi = 1/2``; measured over a sphere and a box it is 0.29-0.47, so the
       analytic jump is wrong by up to 42 % (the low end is a box corner, where
       the surface is not flat) before anything else happens.  That the row sum
       is ISOTROPIC to 1e-13 is a structural fact no tolerance can fake.
   (b) with ``jump="calibrated"`` the assembled operator annihilates the rigid
       mode EXACTLY, at P0, P1 and P2, on a sphere and a box, at three nu.
       ``jump="half"`` misses it by the amount in (a).
       CAUTION, and it is gated as such: a rigid translation CANNOT tell the
       shape-function free term ``N_k(x_c)`` from the identity, because
       ``sum_k N_k = 1``.  So the patch test alone does NOT validate the
       higher-order free term; the difference is measured separately on a solve
       whose density varies inside an element.
   (c) the HYPERSINGULAR row needs no calibration at all: its raw row sum is
       already machine zero.  That is not luck -- it is the exact
       finite-triangle eigenstress subtraction (``../BACKLOG.md``),
       which makes the elastic stress of a uniform DD on a closed surface
       vanish pointwise.  Tripwire: with ``subtract_eigenstress=False`` the same
       row sum is 4.4e15 times larger.

3. PATCH TEST.  An end-to-end solve whose exact answer is a rigid translation:
   prescribe ``u = c`` on a closed sphere, recover ``q = -sigma c``.  Exact with
   calibration, ~2-20 % wrong with the analytic half -- so the gate is measuring
   the calibration and not the linear solver.

4. KNOWN EXACT SOLUTION.  A Kelvin point force OUTSIDE the body
   (``_exact.py`` for why that is the only exact solution a boundary-only
   representation can be asked to reproduce), run BOTH as a Dirichlet problem
   (displacement rows) and as a Neumann problem (hypersingular rows), at
   nu = 0.25, 0.30 and 0.45 -- the lam/mu pairing bug is invisible at 1/4.  The
   gate is on the RATIO of errors between two mesh levels (it must fall), not
   only on an absolute number, because a ratio cannot be passed by loosening a
   tolerance.  The reference's own stress formula is checked against a central
   difference of its own displacement first, so a transcription error there
   cannot masquerade as a solver result.  The gated quantity is the interior
   DISPLACEMENT; the interior stress is printed but not gated on monotonicity,
   because it measurably is not monotone at P0 on this ladder (see the comment
   at the check).

5. MIXED BOUNDARY CONDITIONS in one system (half the sphere Dirichlet, half
   traction-free) against the same exact solution.

6. THE TWO FREE-TRACTION ROW TYPES AGREE.  The hypersingular row ``t = 0`` and
   the second-kind row ``u_ext = 0`` are different discretisations of the same
   physics; they are solved on the same fault-in-a-sphere problem and their
   interior fields must agree, and agree BETTER on the finer mesh.  The same
   case gates the boundary-trace readout: on a traction-free body the exterior
   field vanishes, so ``trace_displacement == -sigma q``, which the exterior row
   must satisfy to machine precision (it IS the equation) and the traction row
   only to discretisation error (that residual is reported).

P0 parity against msd's solver is a separate script,
``verify/verify_msd_parity.py`` (it needs numba).

TOLERANCES.  Every tolerance below is printed next to the value it gates.  The
structural checks (isotropy, exactness of the calibration, machine-zero
hypersingular row sum) sit at 1e-11..1e-13 and are machine-precision
statements.  The physics checks are gated on RATIOS -- an error that must fall
under refinement -- plus a ceiling set a few times above the measured value on
the coarse mesh; the measured numbers are in the README table, so a loosened
ceiling would be visible there.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import MU, NU_SWEEP, Report                        # noqa: E402
from _exact import (best_fit_rigid, kelvin_displacement,        # noqa: E402
                    kelvin_stress, kelvin_stress_fd, kelvin_traction)

import ddbem                                                    # noqa: E402
from ddbem._clq import CLQ as clq                               # noqa: E402

X0 = np.array([3.0, 1.2, -0.7])
FORCE = np.array([1.0, -0.5, 0.8])
# Observation points are kept INSIDE |x| <= 0.3 of the unit sphere's centre.
# The representation is mesh-limited within ~1 h of the boundary (msd says the
# same in ``mbem/evaluate.py::_warn_near_boundary``: the c = 1/2 -> 1 boundary
# jump smears over h, NOT over eps, and shrinking eps makes it worse), and the
# coarsest sphere here has h = 1.05.  With points at |x| <= 0.5 the interior
# STRESS error is measurably non-monotone in h; the displacement is not.
OBS = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1], [-0.15, 0.2, 0.1]])
ORDERS = (0, 1, 2)


def _voigt(s):
    return np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2],
                     s[:, 1, 2], s[:, 0, 2], s[:, 0, 1]], axis=1)


def _sphere(nsub=1, radius=1.0):
    v, t = ddbem.icosphere(nsub, radius=radius)
    return ddbem.tri_verts(v, t)


def _rigid_mode(model, system, translation):
    """The DD density of a rigid translation: ``q = -sigma c`` at every node."""
    q = np.zeros(system.n_dof)
    for p in model.boundary:
        o = model.order_of(p)
        blk = np.broadcast_to(-float(p.orientation) * np.asarray(translation, float),
                              (p.n_tri * ddbem.n_nodes(o), 3))
        q[system.col_slice[p.name]] = blk.reshape(-1)
    return q


def _identity_free_term(tv, eps, order, problem):
    """Errors ``(with N_k(x_c), with the identity)`` for the same solve.

    Substitutes the identity for the collocation shape matrix -- the mistake a
    P0 code makes when it grows a P1 path -- by swapping the symbol
    :mod:`ddbem.model` resolved at import, and re-runs the same solve.
    ``problem="rigid"`` returns the ratio for a constant prescribed
    displacement (which by the partition of unity must be 1 exactly);
    ``problem="kelvin"`` for the exterior point force (whose density varies
    inside an element, so it can see the difference).
    """
    import ddbem.model as dm
    nu = 0.30
    if problem == "rigid":
        val, ref = np.array([0.03, -0.02, 0.05]), None
    else:
        val = lambda x: kelvin_displacement(x, X0, FORCE, MU, nu)   # noqa: E731
        ref = kelvin_displacement(OBS, X0, FORCE, MU, nu)
    out = []
    good = dm.collocation_shape_matrix
    try:
        for fn in (good, lambda o, s: np.eye(ddbem.n_nodes(o))):
            dm.collocation_shape_matrix = fn
            p = ddbem.Patch("s", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                            value=val, eps=eps)
            sol = ddbem.Model([p], MU, nu, order=order, jump="calibrated").solve()
            if ref is None:
                out.append(float(np.max(np.abs(sol.q.reshape(-1, 3) + val))
                                 / np.max(np.abs(val))))
            else:
                out.append(float(np.max(np.abs(sol.displacement(OBS) - ref))
                                 / np.max(np.abs(ref))))
    finally:
        dm.collocation_shape_matrix = good
    return out[0], out[1]


# ---------------------------------------------------------------------------

def gate_conventions(rep: Report) -> None:
    tri = np.array([[0.37, -0.81, 0.44], [1.92, 0.11, -0.63], [-0.25, 1.57, 1.22]])
    worst = 0.0
    for order in ORDERS:
        for shrink in (0.0, 0.35, 0.7):
            lam = ddbem.mesh.barycentric_nodes(order, shrink)
            pts = np.einsum("kv,vc->kc", lam, tri)
            worst = max(worst, float(np.max(np.abs(
                clq.shape_functions(tri, order, pts) - ddbem.shape_at(order, lam)))))
    rep.check("shape_at == clq.shape_functions (P0/P1/P2, 3 shrinks)", worst, 1e-13)

    lam = ddbem.mesh.barycentric_nodes(2, 0.35)
    swapped = ddbem.shape_at(2, lam)[:, [1, 0, 2, 3, 4, 5]]
    rep.check_bool("  tripwire: a 0<->1 node swap breaks it",
                   float(np.max(np.abs(swapped - ddbem.shape_at(2, lam)))) > 0.1,
                   f"moves it by {float(np.max(np.abs(swapped - ddbem.shape_at(2, lam)))):.3f}")

    for order in ORDERS:
        K = ddbem.n_nodes(order)
        N0 = ddbem.collocation_shape_matrix(order, 0.0)
        rep.check(f"P{order} collocation shape matrix is I at shrink = 0",
                  float(np.max(np.abs(N0 - np.eye(K)))), 1e-14)
        Ns = ddbem.collocation_shape_matrix(order, 0.35)
        rep.check(f"P{order} shrunk shape matrix rows sum to 1 (partition of unity)",
                  float(np.max(np.abs(Ns.sum(axis=1) - 1.0))), 1e-14)
        if order:
            rep.check_bool(f"P{order}   and is NOT the identity (free term is a matrix)",
                           float(np.max(np.abs(Ns - np.eye(K)))) > 0.1,
                           f"max |N - I| = {float(np.max(np.abs(Ns - np.eye(K)))):.3f}")

    sph = _sphere(1)
    bx = ddbem.tri_verts(*ddbem.box((2.0, 1.5, 1.0), (2, 2, 2)))
    rep.check_bool("icosphere is a closed oriented manifold", ddbem.shapes.is_closed(sph))
    rep.check_bool("box is a closed oriented manifold", ddbem.shapes.is_closed(bx))
    rep.check("box encloses the right volume",
              abs(ddbem.shapes.enclosed_volume(bx) - 3.0) / 3.0, 1e-13)
    broken = sph.copy()
    broken[5] = broken[5][[0, 2, 1]]
    rep.check_bool("  tripwire: one flipped triangle breaks closure",
                   not ddbem.shapes.is_closed(broken))
    rep.check_bool("open patch is not closed",
                   not ddbem.shapes.is_closed(ddbem.tri_verts(
                       *ddbem.rectangle([0, 0, 0], [1, 0, 0], [0, 1, 0], 2, 2))))

    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    eps = 0.3 * float(ddbem.element_h(sph).mean())
    rep.check_bool("BCType.INTERFACE raises NotImplementedError", raises(
        lambda: ddbem.Model([ddbem.Patch("i", sph, ddbem.BCType.INTERFACE, eps=eps)],
                            MU, 0.25), NotImplementedError))
    rep.check_bool("jump='calibrated' on an OPEN boundary raises", raises(
        lambda: ddbem.Model([ddbem.Patch("f", ddbem.tri_verts(*ddbem.rectangle(
            [0, 0, 0], [1, 0, 0], [0, 1, 0], 2, 2)), ddbem.BCType.FREE_TRACTION,
            eps=0.1)], MU, 0.25, jump="calibrated"), ValueError))
    rep.check_bool("inward-oriented boundary raises", raises(
        lambda: ddbem.Model([ddbem.Patch("s", ddbem.shapes.flip(sph),
                                         ddbem.BCType.FREE_TRACTION, eps=eps)],
                            MU, 0.25), ValueError))
    rep.check_bool("neumann_row='exterior' with a Dirichlet patch raises", raises(
        lambda: ddbem.Model([ddbem.Patch("s", sph,
                                         ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                                         value=[1, 0, 0], eps=eps)],
                            MU, 0.25, neumann_row="exterior"), ValueError))


def gate_rowsum(rep: Report) -> None:
    """The free term is a row-sum identity -- structurally, at every order."""
    meshes = [("sphere", _sphere(0)),
              ("box", ddbem.tri_verts(*ddbem.box((2.0, 1.5, 1.0), (1, 1, 1))))]
    c = np.array([0.03, -0.02, 0.05])
    phi_lo, phi_hi = 1.0, 0.0
    for mname, tv in meshes:
        eps = 0.3 * float(ddbem.element_h(tv).mean())
        for nu in NU_SWEEP:
            for order in ORDERS:
                # --- displacement row: R is exactly phi(x_c) * I
                pd = ddbem.Patch("s", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                                 value=c, eps=eps)
                md = ddbem.Model([pd], MU, nu, order=order, jump="calibrated")
                sd = md.assemble()
                phi = np.einsum("cii->c", sd.rowsum) / 3.0
                dev = float(np.max(np.abs(sd.rowsum - phi[:, None, None] * np.eye(3))))
                rep.check(f"{mname} nu={nu} P{order}: displacement row sum is phi*I",
                          dev, 1e-13)
                phi_lo, phi_hi = min(phi_lo, phi.min()), max(phi_hi, phi.max())

                q = _rigid_mode(md, sd, c)
                exact = np.broadcast_to(c, (sd.n_dof // 3, 3)).reshape(-1)
                rep.check(f"{mname} nu={nu} P{order}: calibrated u_int row exact on rigid",
                          float(np.max(np.abs(sd.A @ q - exact))) / float(np.max(np.abs(c))),
                          1e-12)

                # --- hypersingular row: raw sum already machine zero
                pt = ddbem.Patch("s", tv, ddbem.BCType.FREE_TRACTION, eps=eps)
                mt = ddbem.Model([pt], MU, nu, order=order, jump="half")
                st = mt.assemble()
                # tolerance 1e-12 on a quantity whose worst measured value is
                # 7.1e-14 (nu = 0.45, P2, where the closed forms lose the most
                # digits): a 14x margin on a machine-precision statement.  It is
                # NOT 1e-13, which the nu = 0.45 rows clear by only 1.4x.
                rep.check(f"{mname} nu={nu} P{order}: hypersingular row sum is 0",
                          float(np.max(np.abs(st.rowsum))) / float(np.max(np.abs(st.A))),
                          1e-12)
                rep.check(f"{mname} nu={nu} P{order}: traction row exact on rigid",
                          float(np.max(np.abs(st.A @ _rigid_mode(mt, st, c))))
                          / float(np.max(np.abs(st.A)) * np.max(np.abs(c))), 1e-12)
    rep.note(f"measured phi (the discrete smoothed indicator at the collocation "
             f"point) = {phi_lo:.4f}..{phi_hi:.4f}; the classical free term asserts "
             f"phi = 1/2, so it is off by {100 * max(abs(phi_lo - .5), abs(phi_hi - .5)) / .5:.1f} %")

    # -- tripwire: the eigenstress subtraction is WHY the traction row sum is 0
    tv = _sphere(0)
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    pts = ddbem.collocation_points(tv, 0, 0.0)
    nrm = ddbem.element_normals(tv)
    A_tot = ddbem.traction_matrix(pts, nrm, tv, eps, MU, 0.30, 0,
                                  subtract_eigenstress=False)
    A_el = ddbem.traction_matrix(pts, nrm, tv, eps, MU, 0.30, 0)
    sgn = -np.ones(A_tot.shape[1])
    r_tot = max(float(np.max(np.abs(A_tot[:, j::3] @ sgn[j::3]))) for j in range(3))
    r_el = max(float(np.max(np.abs(A_el[:, j::3] @ sgn[j::3]))) for j in range(3))
    rep.check_bool("  tripwire: without the eigenstress subtraction it is NOT zero",
                   r_tot / max(r_el, 1e-300) > 1e10,
                   f"total-stress row sum {r_tot:.3e} vs elastic {r_el:.3e} "
                   f"({r_tot / max(r_el, 1e-300):.1e}x)")

    # -- the free term is the shape MATRIX N_k(x_c), not the identity.
    # A rigid translation CANNOT see the difference (sum_k N_k = 1, so both
    # forms give the same answer on a constant density) -- so this is measured
    # on a solve whose density varies inside an element, the Kelvin problem.
    tv = _sphere(1)
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    g, bad = _identity_free_term(tv, eps, 1, "rigid")
    rep.check("  a constant density cannot distinguish N_k(x_c) from I",
              max(g, bad), 1e-11,
              "(partition of unity) -- so the check below uses a varying one")
    for order in (1, 2):
        g, bad = _identity_free_term(tv, eps, order, "kelvin")
        rep.check_bool(f"  P{order}: substituting I for N_k(x_c) changes the answer",
                       abs(bad / g - 1.0) > 0.05,
                       f"{g:.3e} -> {bad:.3e}  (x{bad / g:.3f})")
    rep.note("N_k(x_c) is the derived form -- the free term multiplies the "
             "density AT the collocation point, which is sum_k N_k(x_c) q_k.  "
             "It is a no-op at shrink = 0 (N = I) and for traction rows (whose "
             "calibrated free term is identically zero, see above); on the "
             "Dirichlet rows at the default shrink = 0.5 the identity is 1.12x "
             "worse at P1 and 0.91x at P2 on this one problem, i.e. the "
             "difference is real but not a one-sided accuracy claim.")


def gate_patch_test(rep: Report) -> None:
    """End-to-end: a constant displacement field must be reproduced exactly."""
    tv = _sphere(1)
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    c = np.array([0.03, -0.02, 0.05])
    half_err = {}
    for nu in NU_SWEEP:
        for order in ORDERS:
            for jump in ("calibrated", "half"):
                p = ddbem.Patch("s", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                                value=c, eps=eps)
                m = ddbem.Model([p], MU, nu, order=order, jump=jump)
                sol = m.solve()
                e = float(np.max(np.abs(sol.q.reshape(-1, 3) + c))) / float(np.max(np.abs(c)))
                if jump == "calibrated":
                    rep.check(f"nu={nu} P{order}: rigid translation recovered exactly",
                              e, 1e-11, f"cond {sol.cond:.1e}")
                else:
                    half_err[(nu, order)] = e
    worst_half = max(half_err.values())
    rep.check_bool("  tripwire: jump='half' does NOT reproduce it",
                   worst_half > 1e-3,
                   f"half-jump error {min(half_err.values()):.2e}..{worst_half:.2e}")


def _solve_kelvin(tv, order, nu, eps, bc, jump="calibrated"):
    K = ddbem.n_nodes(order)
    if bc == "dirichlet":
        p = ddbem.Patch("s", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                        value=lambda x: kelvin_displacement(x, X0, FORCE, MU, nu),
                        eps=eps)
    else:
        p = ddbem.Patch("s", tv, ddbem.BCType.FREE_TRACTION,
                        value=lambda x, n: kelvin_traction(x, n, X0, FORCE, MU, nu),
                        eps=eps)
    m = ddbem.Model([p], MU, nu, order=order, jump=jump)
    sol = m.solve()
    u, s = sol.displacement(OBS), sol.stress(OBS)
    ue = kelvin_displacement(OBS, X0, FORCE, MU, nu)
    se = _voigt(kelvin_stress(OBS, X0, FORCE, MU, nu))
    eu = (np.max(np.abs(u - ue)) if bc == "dirichlet"
          else np.max(np.abs(best_fit_rigid(u - ue, OBS)))) / np.max(np.abs(ue))
    es = float(np.max(np.abs(s - se)) / np.max(np.abs(se)))
    return float(eu), es, sol


def gate_exact_solution(rep: Report) -> None:
    """A Kelvin point force outside the body, Dirichlet rows and traction rows."""
    xt = np.array([[0.2, 0.3, 0.1], [-0.5, 0.4, 0.9], [0.8, -0.2, -0.6]])
    for nu in NU_SWEEP:
        a = kelvin_stress(xt, X0, FORCE, MU, nu)
        b = kelvin_stress_fd(xt, X0, FORCE, MU, nu)
        rep.check(f"reference: kelvin_stress == d/dx kelvin_displacement (nu={nu})",
                  float(np.max(np.abs(a - b)) / np.max(np.abs(a))), 1e-7)

    coarse, fine = _sphere(0), _sphere(1)
    hc = float(ddbem.element_h(coarse).mean())
    hf = float(ddbem.element_h(fine).mean())
    for bc in ("dirichlet", "neumann"):
        for order in ORDERS:
            euc, esc, _ = _solve_kelvin(coarse, order, 0.30, 0.3 * hc, bc)
            euf, esf, sol = _solve_kelvin(fine, order, 0.30, 0.3 * hf, bc)
            # DISPLACEMENT is the gated quantity for both row types (for the
            # Neumann problem after removing the rigid motion it is defined only
            # up to).  The interior STRESS is reported, not gated on
            # monotonicity: measured on this ladder it is non-monotone at P0
            # (9.1e-2 -> 9.1e-2 -> 6.2e-2 over three levels), because a stress
            # read from the representation is mesh-limited near the boundary in
            # a way the displacement is not.
            rep.check_bool(f"{bc} P{order}: u error falls under refinement",
                           euf < euc,
                           f"{euc:.3e} -> {euf:.3e}  "
                           f"(rate {np.log(euf / euc) / np.log(hf / hc):+.2f} in h)")
            rep.check(f"{bc} P{order}: u error on the 80-triangle sphere",
                      euf, 0.05, f"sigma {esf:.3e}, cond {sol.cond:.1e}")
    for nu in (0.25, 0.45):
        for bc in ("dirichlet", "neumann"):
            eu, es, _ = _solve_kelvin(fine, 0, nu, 0.3 * hf, bc)
            rep.check(f"{bc} P0 nu={nu}: u error on the 80-triangle sphere",
                      eu, 0.05, f"sigma {es:.3e}")


def gate_clearance_rule(rep: Report) -> None:
    """One eps of node clearance is where the hypersingular error bottoms out.

    ``bench/sweep_collocation.py`` measured, at eps/h = 0.15 and P1, a sharp
    minimum at shrink = 0.6 (9.86e-5) that is 6.7x better than shrink = 0.5
    (6.60e-4) and 4.5x/8.1x better than 0.7/0.8 -- and 0.6 is exactly where the
    node clearance first reaches 1.03 eps.  ``shrink_for_clearance`` turns that
    into a rule; this gate checks the rule hits its target AND that the target
    is the right one, by solving at the rule's shrink and at the fixed default.
    """
    tv = _sphere(1)
    h = float(ddbem.element_h(tv).mean())
    rep.check_bool("shrink_for_clearance is 0 at P0 (the centroid does not move)",
                   ddbem.shrink_for_clearance(tv, 0, 0.15 * h) == 0.0)
    for order in (1, 2):
        t = ddbem.shrink_for_clearance(tv, order, 0.15 * h)
        clr = float(np.median(ddbem.node_clearance(tv, order, t).min(axis=1))
                    / (0.15 * h))
        rep.check(f"P{order} eps/h=0.15: rule hits 1 eps of clearance",
                  abs(clr - 1.0), 0.02, f"shrink {t:.3f} -> {clr:.3f} eps")
        t3 = ddbem.shrink_for_clearance(tv, order, 0.3 * h)
        clr3 = float(np.median(ddbem.node_clearance(tv, order, t3).min(axis=1))
                     / (0.3 * h))
        rep.check_bool(f"P{order} eps/h=0.30: target unreachable, capped at "
                       f"SHRINK_MAX", t3 == ddbem.defaults.SHRINK_MAX and clr3 < 1.0,
                       f"shrink {t3:.2f} -> {clr3:.3f} eps (< 1)")

    t = ddbem.shrink_for_clearance(tv, 1, 0.15 * h)
    errs = {}
    for name, sh in (("rule", t), ("default 0.5", 0.5)):
        p = ddbem.Patch("s", tv, ddbem.BCType.FREE_TRACTION,
                        value=lambda x, n: kelvin_traction(x, n, X0, FORCE, MU, 0.30),
                        eps=0.15 * h)
        sol = ddbem.Model([p], MU, 0.30, order=1, shrink=sh,
                          jump="calibrated").solve()
        ue = kelvin_displacement(OBS, X0, FORCE, MU, 0.30)
        errs[name] = float(np.max(np.abs(best_fit_rigid(
            sol.displacement(OBS) - ue, OBS))) / np.max(np.abs(ue)))
    rep.check_bool("P1 hypersingular: the 1-eps rule beats the fixed default",
                   errs["rule"] < errs["default 0.5"],
                   f"{errs['default 0.5']:.3e} (t=0.5) -> {errs['rule']:.3e} "
                   f"(t={t:.2f}), {errs['default 0.5'] / errs['rule']:.1f}x")


def gate_mixed_bc(rep: Report) -> None:
    """Dirichlet and traction rows in ONE system, same exact solution."""
    tv = _sphere(1)
    h = float(ddbem.element_h(tv).mean())
    eps = 0.3 * h
    z = tv.mean(axis=1)[:, 2]
    top, bot = tv[z > 0], tv[z <= 0]
    for nu in (0.30, 0.45):
        pd = ddbem.Patch("top", top, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                         value=lambda x: kelvin_displacement(x, X0, FORCE, MU, nu),
                         eps=eps)
        pn = ddbem.Patch("bot", bot, ddbem.BCType.FREE_TRACTION,
                         value=lambda x, n: kelvin_traction(x, n, X0, FORCE, MU, nu),
                         eps=eps)
        m = ddbem.Model([pd, pn], MU, nu, order=0, jump="calibrated")
        sol = m.solve()
        u = sol.displacement(OBS)
        ue = kelvin_displacement(OBS, X0, FORCE, MU, nu)
        e = float(np.max(np.abs(u - ue)) / np.max(np.abs(ue)))
        rep.check(f"mixed Dirichlet/traction model, nu={nu}", e, 0.05,
                  f"{top.shape[0]} + {bot.shape[0]} triangles, cond {sol.cond:.1e}")


def gate_row_types_agree(rep: Report) -> None:
    """t = 0 and u_ext = 0 are two rows for the same physics: they must agree."""
    fv, ft = ddbem.rectangle([-0.4, -0.3, 0.0], [0.8, 0, 0], [0, 0.6, 0], 2, 2)
    ftv = ddbem.tri_verts(fv, ft)
    epsf = 0.3 * float(ddbem.element_h(ftv).mean())
    slip = np.array([0.001, 0.0, 0.0])
    obs = np.array([[0.0, 0.0, 0.9], [0.6, 0.5, -0.7], [-0.8, 0.3, 0.4]])
    prev = None
    for nsub in (0, 1):
        tv = _sphere(nsub, radius=2.0)
        eps = 0.3 * float(ddbem.element_h(tv).mean())
        fields, traces = {}, {}
        for row in ("traction", "exterior"):
            pb = ddbem.Patch("s", tv, ddbem.BCType.FREE_TRACTION, eps=eps)
            pf = ddbem.Patch("f", ftv, ddbem.BCType.FAULT, value=-slip, eps=epsf)
            m = ddbem.Model([pb, pf], MU, 0.30, order=0, jump="calibrated",
                            neumann_row=row)
            sol = m.solve(constrain="rigid")
            fields[row] = sol.stress(obs)
            # u_interior = -sigma q on a traction-free body (the exterior field
            # vanishes).  Under the EXTERIOR row that is the equation itself, so
            # the trace machinery must reproduce it to the accuracy with which
            # the row is satisfied.  The 6-mode border does NOT satisfy the rows
            # exactly -- rotations are only NEAR-null, so the bordered solve
            # leaves a constraint force Z lam behind -- which is why the exact
            # statement is made with the 3 translations, which calibration makes
            # an exact null space.
            for cons in ("rigid", "translations"):
                s2 = m.solve(sol.system, constrain=cons)
                traces[(row, cons)] = float(
                    np.max(np.abs(s2.trace_displacement("s")
                                  + s2.q.reshape(-1, 3))) / np.max(np.abs(s2.q)))
        rep.check(f"trace_displacement == -sigma q (exterior row, translations, "
                  f"{tv.shape[0]} triangles)",
                  traces[("exterior", "translations")], 1e-11)
        rep.note(f"same identity, exterior row with the 6-mode border: "
                 f"{traces[('exterior', 'rigid')]:.3e} (the rotation constraint "
                 f"force); under the TRACTION row: "
                 f"{traces[('traction', 'translations')]:.3e} -- that residual "
                 f"IS the discrete exterior field the hypersingular row does "
                 f"not set to zero")
        d = (np.max(np.abs(fields["traction"] - fields["exterior"]))
             / np.max(np.abs(fields["exterior"])))
        rep.check(f"traction row vs exterior row, {tv.shape[0]}-triangle sphere",
                  float(d), 0.30)
        if prev is not None:
            rep.check_bool("  and they agree BETTER on the finer mesh", d < prev,
                           f"{prev:.3e} -> {d:.3e}")
        prev = d


def main() -> int:
    rep = Report("ddbem solver: rows, boundary conditions, free term, solve")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_conventions(rep)
        gate_rowsum(rep)
        gate_patch_test(rep)
        gate_exact_solution(rep)
        gate_clearance_rule(rep)
        gate_mixed_bc(rep)
        gate_row_types_agree(rep)
    return 0 if rep.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
