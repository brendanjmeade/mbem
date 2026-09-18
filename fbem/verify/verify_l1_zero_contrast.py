"""Gate L1 -- zero contrast (alpha = 1) must be exactly inert.

Run from the fbem root with
    /Users/meade/micromamba/bin/python verify/verify_l1_zero_contrast.py

WHAT IS TESTED
--------------
A small closed box (20 x 20 x 20 km, 4x4 quads per face = 192 triangles:
top + 4 sides FREE, base FIXED) containing a small closed box-shaped
"inclusion" surface (6 x 6 x 6 km, 2x2 quads per face = 48 triangles,
row type INTERFACE, normals out of the inclusion).  N = 240 triangles,
720 unknowns.  Driven by ONE slip triangle (1 m of y-slip on a vertical
patch at x = +6.5 km) through ``model.fault_source``, i.e. the same RHS
path the production driver uses.  mu = 30, nu = 0.25, eps = 1.5 km.

L1.1 STRUCTURAL   max|A[R2 rows, :] - I| must be 0.0 BITWISE.
                  No tolerance: at a = 1 the row scale is -(1-a) = -0.0 and
                  the diagonal increment is (1+a)/2 = 1.0, so every entry is
                  +-0.0 or exactly 1.0.  Anything else is a floating-point
                  ordering defect in ForceElementModel.assemble.

L1.2 SOLUTION     max|q| on the INTERFACE rows must be 0.0 BITWISE.
                  Justification for demanding bitwise rather than a backward
                  error: the interface rows are exactly e_j, so during LU with
                  partial pivoting the multiplier of every other row in those
                  columns multiplies a row that is exactly e_j -- the update is
                  exact, and the substitution gives q_j = b_j / 1.0 = 0.0
                  exactly.  The script ALSO prints the LU backward error
                  |A q - b|_inf / (|A|_inf |q|_inf) so the weaker claim is
                  available if the strong one ever fails.

L1.3 PHYSICS      The alpha = 1 solve must reproduce the HOMOGENEOUS BOX --
                  the identical problem with the interface patches DELETED
                  from the model entirely (not the fault-only free-space
                  field; the box walls still carry densities, and this script
                  prints how large they are so the test is visibly not
                  vacuous).  Compared three ways, all relative to the
                  respective max of the homogeneous quantity:
                    (a) the 576 outer force densities q,
                    (b) u at the 32 top-face centroids (with the (eps/4)M
                        self-bias correction, as the driver reports it),
                    (c) u at 25 independent interior observation points that
                        are not centroids of anything.
                  Tolerance 1e-12 RELATIVE.  Reason: the two systems are
                  algebraically identical on the outer block (the interface
                  columns are eliminated by exactly-e_j pivot rows), so the
                  only admissible difference is the row-equilibration scale
                  of the FIXED rows, s = 1/max|G row|, whose max could in
                  principle land on an interface column in the big model and
                  on an outer column in the small one.  The script checks
                  whether s is bitwise identical and reports it; when it is,
                  the expected answer is 0.0 and 1e-12 is 4 orders of
                  magnitude of slack over anything LU can produce here.

L1.4 CONTINUITY   u evaluated at all 48 interface centroids +- d n, for
                  d = 0.1, 0.01, 0.001 km.  A single layer has continuous
                  displacement, so D(d) := u(+d) - u(-d) = 2d du/dn + O(d^3)
                  with NO constant term.  Two gates:
                    (a) max|D| falls by 10 +- 0.5 across each decade of d.
                        A jump J would make D -> J and drive these ratios to 1.
                        NOTE: at alpha = 1 this is a NULL TEST and cannot fail.
                        The interface density is exactly zero, so the probes sit
                        in a source-free region, and for eps > 0 the Cortez
                        single layer is C-infinity anyway, so D(d) = O(d) holds
                        for any density and any smooth kernel.  It is retained
                        as a sanity print, not as a discriminator; L1.1-L1.3
                        and L1.5 carry the gate.  (Found by the adversarial
                        re-check, which seeded 19 defects and seeded none that
                        this sub-check alone caught.)
                    (b) the Richardson extrapolation of D to d = 0,
                        J = (D(d2) d1 - D(d1) d2)/(d1 - d2) with
                        (d1, d2) = (0.01, 0.001), must satisfy
                        max|J| <= 1e-7 max|u| at the same points.
                  Tolerance basis for (b): the extrapolation's own truncation
                  is the cubic term, and the script confirms this by showing
                  that J computed from (0.1, 0.01) is ~1000x larger -- exactly
                  the d^3 law -- so what (b) measures is truncation, not a
                  jump.  Measured 9.1e-9; 1e-7 leaves ~11x margin while still
                  sitting ~7 orders below any physically meaningful jump (a
                  half-strength layer jump would be O(1) relative).

L1.5 NON-VACUITY  Everything above would also "pass" if the interface were
                  inert at EVERY alpha (a decoupling bug).  So the script also
                  solves at alpha = 0.1 and at alpha = 1 -+ 1e-3, 1e-6 and
                  requires: |q_int| != 0 at alpha = 0.1 and a >= 1% change in
                  the top-surface displacement; and |q_int| proportional to
                  |1 - alpha| near alpha = 1 (the 1e-3 / 1e-6 pair must differ
                  by 1000 +- 5%, matching SPEC.md L1's linearity item).

L1.6 ROW FORMULAS At alpha = 1 the R2 coefficients are degenerate: any
                  formula with (1+a)/2 -> 1 and (1-a) -> 0 there passes L1.1.
                  So the assembled rows are also re-derived from scratch and
                  compared bitwise at alpha = 0.1, 1.0 and 7.0, and with
                  row_block = 7 (deliberately not a multiple of 3) to exercise
                  ForceElementModel.assemble's blocked row-scaling loop:
                    R1  B + (1/2) I ,       b = -t_F
                    R2  -(1-a) B + ((1+a)/2) I ,  b = (1-a) t_F
                    R3  s (G - (eps/4) M) ,  b = -s u_F ,  s = 1/max|G row|
                  Tolerance 0.0 bitwise -- it is the same arithmetic in a
                  different order, so anything else is an indexing bug.

Also recorded (not gated): cond_2 and cond_1 of the alpha = 1, alpha = 0.1 and
homogeneous systems on the same mesh.

WHAT L1 CANNOT SEE (do not over-read a PASS)
  * The SIGN of the B block.  At alpha = 1 the B term is multiplied by zero,
    and in L1.6 B is taken from the model's own traction_block, so a global
    sign error in B survives every check here.  SPEC.md pins it on an
    ellipsoid in L3; the sphere cannot do it either.
  * The free-term value 1/2 on R1, and the fault sign.  Both are common to the
    alpha = 1 and homogeneous solves and cancel in L1.3.
  * Anything about accuracy: every number here is an algebraic identity, not a
    comparison with a known solution.

NOTHING IN fbem/ IS EDITED OR MONKEYPATCHED BY THIS SCRIPT.
"""
from __future__ import annotations

import os
import sys

import numpy as np

FBEM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FBEM not in sys.path:
    sys.path.insert(0, FBEM)

import model as fem                                              # noqa: E402

GATE = "L1"
MU, NU, EPS = 30.0, 0.25, 1.5
ALPHA_REF = 0.1
RTOL = 1.0e-12                     # L1.3, see docstring
np.set_printoptions(precision=4)


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------
def grid(origin, uvec, vvec, nu_, nv):
    """Vertices/triangles of a parallelogram cut into nu_ x nv quads.

    Winding is irrelevant -- model.Patch re-orients every normal geometrically.
    """
    o, uv, vv = (np.asarray(a, float) for a in (origin, uvec, vvec))
    V = np.array([o + (i / nu_) * uv + (j / nv) * vv
                  for j in range(nv + 1) for i in range(nu_ + 1)])
    ix = lambda i, j: j * (nu_ + 1) + i
    T = []
    for j in range(nv):
        for i in range(nu_):
            T.append([ix(i, j), ix(i + 1, j), ix(i + 1, j + 1)])
            T.append([ix(i, j), ix(i + 1, j + 1), ix(i, j + 1)])
    return V, np.array(T, int)


def box_patches(prefix, cx, cy, z_lo, z_hi, half, n_side, n_z, row):
    """Six patches of an axis-aligned closed box, normals outward."""
    L, D = 2.0 * half, z_hi - z_lo
    p = []
    v, t = grid((cx - half, cy - half, z_hi), (L, 0, 0), (0, L, 0), n_side, n_side)
    p.append(fem.Patch(f"{prefix}_top", v, t, row, ("up",)))
    v, t = grid((cx - half, cy - half, z_lo), (L, 0, 0), (0, L, 0), n_side, n_side)
    p.append(fem.Patch(f"{prefix}_bot", v, t, row, ("down",)))
    for nm, o, uv in (("xm", (cx - half, cy - half, z_lo), (0, L, 0)),
                      ("xp", (cx + half, cy - half, z_lo), (0, L, 0)),
                      ("ym", (cx - half, cy - half, z_lo), (L, 0, 0)),
                      ("yp", (cx - half, cy + half, z_lo), (L, 0, 0))):
        v, t = grid(o, uv, (0, 0, D), n_side, n_z)
        p.append(fem.Patch(f"{prefix}_{nm}", v, t, row, ("radial", (cx, cy))))
    return p


def build_patches():
    """Outer box first (so the homogeneous model is a leading sub-model)."""
    outer = box_patches("host", 0.0, 0.0, -20.0, 0.0, 10.0, 4, 4, fem.FREE)
    outer[1] = fem.Patch("host_bot", *grid((-10, -10, -20), (20, 0, 0),
                                           (0, 20, 0), 4, 4), fem.FIXED, ("down",))
    inner = box_patches("inc", 0.0, 0.0, -13.0, -7.0, 3.0, 2, 2, fem.INTERFACE)
    return outer, inner


FAULT_TV = np.array([[[6.5, -2.0, -12.0],
                      [6.5, 2.0, -12.0],
                      [6.5, 0.0, -8.0]]])
SLIP = 0.001 * np.array([0.0, 1.0, 0.0])          # 1 m of y-slip


# --------------------------------------------------------------------------
def rel(a, b):
    """max|a - b| / max|b|, with max|b| reported by the caller."""
    d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    s = float(np.abs(np.asarray(b)).max())
    return d, d / s if s > 0 else np.inf


def main():
    outer, inner = build_patches()
    het = fem.ForceElementModel(outer + inner, MU, NU, EPS, 1.0)
    hom = fem.ForceElementModel(outer, MU, NU, EPS, 1.0)
    n_out = hom.N
    assert np.array_equal(het.centroids[:n_out], hom.centroids)
    assert np.array_equal(het.normals[:n_out], hom.normals)

    inter = het.rows(fem.INTERFACE)
    free = het.rows(fem.FREE)
    fixed = het.rows(fem.FIXED)
    d_int = fem.ForceElementModel._dof(inter)
    d_out = fem.ForceElementModel._dof(np.arange(n_out))

    print("=" * 74)
    print(f"GATE {GATE}   zero contrast (alpha = 1) must be exactly inert")
    print("=" * 74)
    print(f"  mesh          outer {n_out} tri ({len(free)} free, {len(fixed)} "
          f"fixed) + interface {len(inter)} tri  ->  N = {het.N}, "
          f"{3*het.N} unknowns")
    print(f"  material      mu = {MU}, nu = {NU}, eps = {EPS} km")
    print(f"  normals       flipped per patch: "
          f"{[p.flipped for p in outer + inner]}")
    ctr = np.array([0.0, 0.0, -10.0])          # both boxes share this centre
    dots = np.einsum("mi,mi->m", het.normals, het.centroids - ctr)
    assert (dots > 0).all(), "mesh normals are not outward"
    print(f"                min dot(n, c - centre) = {dots.min():.4f} > 0, "
          f"i.e. out of the solid / out of the inclusion everywhere")
    print(f"  source        1 slip triangle at x = 6.5 km, slip = "
          f"{1e3*np.linalg.norm(SLIP):.0f} m in +y")

    # ---- right-hand side (shared: hom is a leading slice of het) ----------
    u_F, t_F = fem.fault_source(het.centroids, het.normals, FAULT_TV, SLIP,
                                MU, NU, EPS, verbose=False)
    print(f"  fault RHS     |u_F|max {np.abs(u_F).max():.4e} km, "
          f"|t_F|max {np.abs(t_F).max():.4e} GPa")

    # ---- one traction block, reused (assemble CONSUMES its B) ------------
    B_het = het.traction_block(verbose=False)
    B_hom = hom.traction_block(verbose=False)
    assert np.array_equal(B_het[np.ix_(d_out, d_out)], B_hom), \
        "B block is not consistent between the two models"

    A1, b1 = het.assemble(t_F, u_F, verbose=False,
                          B=B_het.copy(order="F"))
    A1_keep = A1.copy()
    s_het = het.fixed_scale

    # ================= 1. STRUCTURAL ==================================
    print("\n--- L1.1 structural: interface rows are the identity --------")
    R2 = A1[np.ix_(d_int, np.arange(3 * het.N))]
    I_blk = np.zeros_like(R2)
    I_blk[np.arange(len(d_int)), d_int] = 1.0
    dev = float(np.abs(R2 - I_blk).max())
    ok_struct = (dev == 0.0)
    b_int = float(np.abs(b1[d_int]).max())
    ok_rhs = (b_int == 0.0)
    print(f"  max|A[R2 rows, :] - I|            {dev:.3e}   "
          f"(bitwise zero: {dev == 0.0})")
    print(f"  max|b[R2 rows]|                   {b_int:.3e}   "
          f"(bitwise zero: {b_int == 0.0})")
    print(f"  interface cols in FREE/FIXED rows {np.abs(A1[np.ix_(d_out, d_int)]).max():.3e}"
          f"   (nonzero => the test is not vacuous)")

    # ================= 2. SOLUTION ====================================
    print("\n--- L1.2 solution: interface densities vanish ----------------")
    q1 = het.solve(verbose=False)
    r = A1_keep @ q1.ravel() - b1
    bwd = float(np.abs(r).max()) / (float(np.abs(A1_keep).max())
                                    * float(np.abs(q1).max()))
    qi = float(np.abs(q1[inter]).max())
    qo = float(np.abs(q1[:n_out]).max())
    ok_qint = (qi == 0.0)
    print(f"  max|q| on interface               {qi:.3e}   "
          f"(bitwise zero: {qi == 0.0})")
    print(f"  max|q| on outer patches           {qo:.3e}   "
          f"(nonzero => walls do carry densities)")
    print(f"  LU backward error |Aq-b|/|A||q|   {bwd:.3e}")

    # ================= 3. PHYSICS =====================================
    print("\n--- L1.3 physics: equals the homogeneous box -----------------")
    A0, b0 = hom.assemble(t_F[:n_out], u_F[:n_out], verbose=False,
                          B=B_hom.copy(order="F"))
    s_hom = hom.fixed_scale
    same_s = bool(np.array_equal(s_het, s_hom))
    A0_keep = A0.copy()
    q0 = hom.solve(verbose=False)
    print(f"  FIXED row scales bitwise equal    {same_s}"
          + ("" if same_s else "   <-- expect O(1e-16) differences below"))
    dA = float(np.abs(A1_keep[np.ix_(d_out, d_out)] - A0_keep).max())
    db = float(np.abs(b1[d_out] - b0).max())
    print(f"  max|A_a1[out,out] - A_hom|        {dA:.3e}   "
          f"max|b_a1[out] - b_hom| {db:.3e}")
    dq, rq = rel(q1[:n_out], q0)
    print(f"  max|q_a1 - q_hom|                 {dq:.3e}  "
          f"rel {rq:.3e}   (|q_hom|max {np.abs(q0).max():.4e})")

    top = np.arange(het.slices["host_top"].start, het.slices["host_top"].stop)
    obs_t = het.centroids[top]
    uF_t, _ = fem.fault_source(obs_t, het.normals[top], FAULT_TV, SLIP,
                               MU, NU, EPS, verbose=False)
    u1 = uF_t + het.displacement(obs_t, self_elems=top, correct_eps=True)
    u0 = uF_t + hom.displacement(obs_t, self_elems=top, correct_eps=True)
    du, ru = rel(u1, u0)
    print(f"  max|u_a1 - u_hom| top centroids   {du:.3e}  "
          f"rel {ru:.3e}   (|u_hom|max {np.abs(u0).max():.4e} km)")

    g = np.linspace(-6.0, 6.0, 5)
    obs_i = np.array([[x, y, -3.0] for x in g for y in g])
    uF_i, _ = fem.fault_source(obs_i, np.tile([0, 0, 1.0], (len(obs_i), 1)),
                               FAULT_TV, SLIP, MU, NU, EPS, verbose=False)
    v1 = uF_i + het.displacement(obs_i)
    v0 = uF_i + hom.displacement(obs_i)
    dv, rv = rel(v1, v0)
    print(f"  max|u_a1 - u_hom| 25 interior pts {dv:.3e}  "
          f"rel {rv:.3e}   (|u_hom|max {np.abs(v0).max():.4e} km)")
    ok_phys = max(rq, ru, rv) <= RTOL

    # ================= 4. INTERIOR CONTINUITY =========================
    print("\n--- L1.4 continuity across the inert interface ---------------")
    c, n = het.centroids[inter], het.normals[inter]
    uu = lambda p: (het.displacement(p) + fem.fault_source(
        p, n, FAULT_TV, SLIP, MU, NU, EPS, verbose=False)[0])
    ds = (0.1, 0.01, 0.001)
    D, scale = {}, 0.0
    for d in ds:
        up, um = uu(c + d * n), uu(c - d * n)
        D[d] = up - um
        scale = max(scale, float(np.abs(0.5 * (up + um)).max()))
        print(f"  d = {d:6.4f} km   max|D(d)| = {np.abs(D[d]).max():.6e}   "
              f"({np.abs(D[d]).max()/scale:.3e} of |u|)")
    def _ratio(a, b):
        num, den = float(np.abs(D[a]).max()), float(np.abs(D[b]).max())
        return np.inf if not den > 0.0 else num / den

    ratios = [_ratio(a, b) for a, b in zip(ds[:-1], ds[1:])]
    print(f"  decade ratios of max|D|          "
          f"{ratios[0]:.4f}, {ratios[1]:.4f}   (10 => O(d), no jump; "
          f"1 => a jump)")
    rich = lambda d1, d2: (D[d2] * d1 - D[d1] * d2) / (d1 - d2)
    J, Jc = rich(0.01, 0.001), rich(0.1, 0.01)
    # A seeded/real defect can make the system singular; lu_solve then returns
    # NaN, `scale` is 0.0, and this division used to raise before the verdict
    # block ran -- so the script exited with a traceback instead of "FAIL L1".
    jrel = np.inf if not scale > 0.0 else float(np.abs(J).max()) / scale
    print(f"  extrapolated jump max|J|         {np.abs(J).max():.4e} km   "
          f"= {jrel:.3e} of |u|")
    print(f"  same from (0.1, 0.01)            {np.abs(Jc).max():.4e} km   "
          f"ratio {np.abs(Jc).max()/np.abs(J).max():.1f}  "
          f"(~1000 => pure d^3 truncation, i.e. J is not a jump)")
    ok_cont = (all(9.5 <= r <= 10.5 for r in ratios) and jrel <= 1.0e-7)

    # ================= 5. NON-VACUITY =================================
    print("\n--- L1.5 non-vacuity: the interface is not inert at other a --")

    def solve_alpha(a, keep_A=False):
        m = fem.ForceElementModel(outer + inner, MU, NU, EPS, a)
        A, _ = m.assemble(t_F, u_F, verbose=False, B=B_het.copy(order="F"))
        Ak = A.copy() if keep_A else None
        q = m.solve(verbose=False)
        return m, Ak, q

    het2, A2, q2 = solve_alpha(ALPHA_REF, keep_A=True)
    u2 = uF_t + het2.displacement(obs_t, self_elems=top, correct_eps=True)
    qi2 = float(np.abs(q2[inter]).max())
    du2 = float(np.abs(u2 - u0).max()) / float(np.abs(u0).max())
    print(f"  alpha = {ALPHA_REF}:  max|q_int| {qi2:.4e}  "
          f"(vs 0 at alpha=1);  top-u change vs hom {du2:.3%}")
    lin = {}
    for da in (1.0e-3, 1.0e-6):
        lo = float(np.abs(solve_alpha(1.0 - da)[2][inter]).max())
        hi = float(np.abs(solve_alpha(1.0 + da)[2][inter]).max())
        lin[da] = (lo, hi)
        print(f"  alpha = 1 -+ {da:.0e}:  max|q_int| = {lo:.4e} / {hi:.4e}")
    r_lo = lin[1e-3][0] / lin[1e-6][0]
    r_hi = lin[1e-3][1] / lin[1e-6][1]
    print(f"  linearity in |1-alpha| (want 1000) {r_lo:.2f} / {r_hi:.2f}")
    ok_vac = (qi2 > 0.0 and du2 > 0.01
              and 950.0 <= r_lo <= 1050.0 and 950.0 <= r_hi <= 1050.0)

    # ================= 6. ROW FORMULAS ================================
    print("\n--- L1.6 assembled rows vs the formulas, re-derived ----------")
    ok_rowf, d_free = True, fem.ForceElementModel._dof(free)
    print(f"  {'alpha':>6s} {'row_block':>10s} {'R1':>10s} {'R2':>10s} "
          f"{'R3':>10s} {'rhs':>10s}")
    for a in (0.1, 1.0, 7.0):
        for rb in (2048, 7):
            m = fem.ForceElementModel(outer + inner, MU, NU, EPS, a)
            B = m.traction_block(verbose=False)
            A, b = m.assemble(t_F, u_F, verbose=False, row_block=rb,
                              B=B.copy(order="F"))
            E1 = B[d_free, :].copy()
            E1[np.arange(len(d_free)), d_free] += 0.5
            E2 = -(1.0 - a) * B[d_int, :]
            E2[np.arange(len(d_int)), d_int] += 0.5 * (1.0 + a)
            dx = fem.ForceElementModel._dof(fixed)
            G = fem.displacement_matrix(m.centroids[fixed], m.tv, m.eps_arr,
                                        MU, NU)
            M = fem.eps_bias_operator(m.normals[fixed], MU, NU)
            for k, e in enumerate(fixed):
                G[3 * k:3 * k + 3, 3 * e:3 * e + 3] -= 0.25 * m.eps * M[k]
            sc = 1.0 / np.abs(G).max(axis=1)
            e = [float(np.abs(A[d_free, :] - E1).max()),
                 float(np.abs(A[d_int, :] - E2).max()),
                 float(np.abs(A[dx, :] - G * sc[:, None]).max()),
                 max(float(np.abs(b[d_free] + t_F[free].ravel()).max()),
                     float(np.abs(b[d_int] - (1.0 - a)
                                  * t_F[inter].ravel()).max()),
                     float(np.abs(b[dx] + u_F[fixed].ravel() * sc).max()))]
            ok_rowf &= all(v == 0.0 for v in e)
            print(f"  {a:6.2f} {rb:10d} " + " ".join(f"{v:10.2e}" for v in e))

    # ================= conditioning (recorded, not gated) =============
    print("\n--- conditioning on this mesh (recorded, not gated) ----------")
    rows = (("alpha = 1.0  (720)", A1_keep), (f"alpha = {ALPHA_REF}  (720)", A2),
            ("homogeneous  (576)", A0_keep))
    print(f"  {'system':22s} {'cond_2':>12s} {'cond_1':>12s} {'sigma_min':>12s}")
    conds = {}
    for nm, A in rows:
        sv = np.linalg.svd(A, compute_uv=False)
        conds[nm] = (sv[0] / sv[-1], np.linalg.cond(A, 1), sv[-1])
        print(f"  {nm:22s} {conds[nm][0]:12.4e} {conds[nm][1]:12.4e} "
              f"{conds[nm][2]:12.4e}")
    c1 = conds["alpha = 1.0  (720)"][0]
    c2 = conds[f"alpha = {ALPHA_REF}  (720)"][0]
    print(f"  cond_2(alpha={ALPHA_REF}) / cond_2(alpha=1) = {c2/c1:.4f}")

    # ================= verdict ========================================
    print("\n" + "-" * 74)
    for nm, ok in (("L1.1 structural  max|A_R2 - I| == 0 bitwise", ok_struct),
                   ("L1.1 structural  max|b_R2|    == 0 bitwise", ok_rhs),
                   ("L1.2 solution    max|q_int|   == 0 bitwise", ok_qint),
                   (f"L1.3 physics     rel <= {RTOL:.0e} vs hom box ", ok_phys),
                   ("L1.4 continuity  D(d) = O(d), jump <= 1e-7", ok_cont),
                   ("L1.5 non-vacuity q_int ~ |1-a|, != 0 at a=0.1", ok_vac),
                   ("L1.6 row formulas bitwise, a=0.1/1/7      ", ok_rowf)):
        print(f"  [{'ok ' if ok else 'BAD'}] {nm}")
    allok = (ok_struct and ok_rhs and ok_qint and ok_phys and ok_cont
             and ok_vac and ok_rowf)
    print("-" * 74)
    print(f"{'PASS' if allok else 'FAIL'} {GATE}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
