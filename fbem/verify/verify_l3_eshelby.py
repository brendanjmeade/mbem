"""Gate L3 -- Eshelby spherical inhomogeneity: the interface row R2.

WHAT THIS GATE PROVES
---------------------
A sphere of radius R = 1 with C_inc = alpha*C_host (same Poisson ratio, so
C_inc = alpha C_host exactly and there is no volume term) sits in an infinite
host under a uniform remote stress sigma_inf.  The ONLY patch is the interface
(row type INTERFACE, normals out of the sphere); there is no outer boundary,
because the remote field enters as the source field

    t_F(x) = sigma_inf . n(x)        (u_F is unused: there are no FIXED rows)

exactly as a fault field would.  The solved density q must reproduce Eshelby's
uniform interior field.  This is the only rung of the ladder that checks the
interface row

    R2:   ((1+a)/2) q - (1-a) B q = (1-a) t_F

against a closed-form answer -- its sign, its free term (1+a)/2, and its
(1-a) scaling.  Structural controls (section 7 of the output) mutate each of
those in turn and must miss the analytic answer by more than the tolerance;
section 6 also checks bitwise that `model.assemble` built exactly that row.

The representation's stress is the HOST stress everywhere, so the TRUE interior
stress is  sigma_true = alpha * (sigma_inf + S q)  with S the force->stress
block.  The gate therefore also exercises `assembly.stress_matrix` and the
`alpha *` interpretation of the representation, not just B.

WHAT IT DOES NOT PROVE
----------------------
1. A sphere discriminates the SIGN of the B block only weakly.  Measured here:
   flipping B's sign gives 1.0e-1 / 9.2e-2 at 320/1280 triangles against
   2.5e-2 / 1.3e-2 correct -- caught (it does not converge), but by ~5x, not by
   the ~40x of every other control.  Pin the B sign on an ellipsoid (L3b).
2. The Eshelby inclusion has ZERO power to detect a missing volume term: the
   interior strain is uniform, so grad(div u) = 0 identically for any nu pair.
   Never cite this gate for that.
3. Nothing here touches R1 (free surface), R3 (Dirichlet), the fault source,
   the (eps/4) displacement-bias correction, or the outer (1/2)I + B block,
   whose conditioning is the real risk (see SPEC section 5).
4. The icosphere is INSCRIBED in the sphere (centroid radius 0.983 / 0.995 at
   320 / 1280).  That does not bias the interior comparison -- Eshelby's
   interior field is independent of the radius -- but it does mean the surface
   is a polyhedron, and the residual error below is dominated by that faceting.

TOLERANCES (chosen before the run, not loosened afterwards)
-----------------------------------------------------------
  <= 3.0e-2 relative at 320 triangles, <= 1.5e-2 at 1280, every load case and
  every alpha, and the error must DECREASE with refinement.  These are the
  task's / SPEC's numbers, from three independent prior measurements of the
  same benchmark (1.8-2.6e-2 -> 0.9-1.1e-2).  They are tight: the worst case
  here (alpha = 0.1) lands at 2.49e-2 and 1.33e-2, i.e. 17% and 11% of margin.

  eps = 0.5 * (mean edge length), the SAME ratio at both densities (eps/R =
  0.15 and 0.075), a little below the production model's eps/h ~ 0.65-0.9 on
  its fine patches.  Section 8 of the output sweeps it so the choice is visible:
  eps/h = 0.25 and 0.50 pass at 320 triangles (2.62e-2, 2.49e-2), 0.75 and 0.90
  do not (3.15e-2, 3.59e-2) -- but at those ratios the sample shell at 0.55 R is
  only 2.0 and 1.7 eps clear of the layer, so part of that is the mollified
  field itself, not the formulation.  The gate is NOT tuned: the answer moves by
  5% over a 2x change in eps.

  Interior sample points: 25 points with |x| <= 0.55 R, i.e. a standoff of at
  least 0.45 R = 3.0 eps (320 tri) / 6.0 eps (1280 tri) from the layer.

  THE QUOTED ERROR DEPENDS ON THAT SAMPLING RADIUS, and section 4b reports the
  dependence rather than hiding it.  The two load cases fail differently:
    - hydrostatic: the error is a uniform BIAS in the concentration factor,
      flat in radius (2.48e-2 at 0.2 R vs 2.49e-2 at 0.55 R, 320 tri).  This is
      the gate's binding constraint and it is a genuine O(h) consistency error.
    - shear: the error is dominated by the NON-UNIFORMITY that the facets
      induce near the surface -- 2.8e-3 at 0.2 R against 2.5e-2 at 0.55 R
      (320 tri).  Sampling only deep inside would make the shear case look ~9x
      better; 0.55 R is deliberately the harder choice, since Eshelby's
      interior field is uniform all the way to the interface.

Run:  /Users/meade/micromamba/bin/python verify/verify_l3_eshelby.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

FBEM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FBEM not in sys.path:
    sys.path.insert(0, FBEM)

from model import Patch, ForceElementModel, INTERFACE          # noqa: E402
from assembly import traction_matrix, stress_matrix, lame      # noqa: E402

MU, NU = 30.0, 0.25
LAM = lame(MU, NU)
EPS_RATIO = 0.5
R = 1.0
ALPHAS = (0.1, 0.5, 2.0)
TOL = {320: 3.0e-2, 1280: 1.5e-2}
GATE = "L3"

HYD = np.eye(3)                                                # probes lam/trace
SHR = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # probes mu
LOADS = (("hydrostatic", HYD), ("simple shear", SHR))


# ---------------------------------------------------------------------------
# 1. analytic Eshelby (inlined so this script is self-contained; both routes
#    are cross-checked against each other at run time)
# ---------------------------------------------------------------------------
def _bulk(mu, nu):
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    return lam, lam + 2.0 * mu / 3.0


def interior_strain(eps0, mu0, nu0, mu1, nu1):
    """Uniform strain inside a spherical inhomogeneity, concentration form."""
    eps0 = np.asarray(eps0, float)
    _, K0 = _bulk(mu0, nu0)
    _, K1 = _bulk(mu1, nu1)
    a = 3.0 * K0 / (3.0 * K0 + 4.0 * mu0)
    b = 6.0 * (K0 + 2.0 * mu0) / (5.0 * (3.0 * K0 + 4.0 * mu0))
    tr = np.trace(eps0)
    dev0 = eps0 - tr / 3.0 * np.eye(3)
    return (dev0 / (1.0 + b * (mu1 / mu0 - 1.0))
            + (tr / (1.0 + a * (K1 / K0 - 1.0))) / 3.0 * np.eye(3))


def _mandel_basis():
    r2 = np.sqrt(2.0) / 2.0
    B = np.zeros((6, 3, 3))
    B[0, 0, 0] = B[1, 1, 1] = B[2, 2, 2] = 1.0
    B[3, 1, 2] = B[3, 2, 1] = r2
    B[4, 0, 2] = B[4, 2, 0] = r2
    B[5, 0, 1] = B[5, 1, 0] = r2
    return B


def interior_strain_via_tensor(eps0, mu0, nu0, mu1, nu1):
    """Same quantity from eps1 = [I + S:C0^-1:(C1-C0)]^-1 : eps0, in MANDEL
    notation (the naive 9x9 flatten is singular: it annihilates antisymmetric
    tensors).  Independent route, used only to cross-check the formula above."""
    B = _mandel_basis()
    nu = nu0
    ca = (7.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
    cb = (5.0 * nu - 1.0) / (15.0 * (1.0 - nu))
    cc = (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
    S4 = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == j == k == l:
                        S4[i, j, k, l] = ca
                    elif i == j and k == l:
                        S4[i, j, k, l] = cb
                    elif (i == k and j == l) or (i == l and j == k):
                        S4[i, j, k, l] = cc
    S6 = np.array([[np.einsum("ij,ijkl,kl->", B[i], S4, B[j]) for j in range(6)]
                   for i in range(6)])

    def stiff6(mu, nu):
        lam, _ = _bulk(mu, nu)
        one = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return lam * np.outer(one, one) + 2.0 * mu * np.eye(6)

    C0, C1 = stiff6(mu0, nu0), stiff6(mu1, nu1)
    A = np.eye(6) + S6 @ np.linalg.solve(C0, C1 - C0)
    v = np.linalg.solve(A, np.einsum("kij,ij->k", B, np.asarray(eps0, float)))
    return np.einsum("k,kij->ij", v, B)


# ---------------------------------------------------------------------------
# 2. geometry and small tensor helpers
# ---------------------------------------------------------------------------
def icosphere(level, radius=1.0):
    """Per-triangle vertices (M,3,3) of a level-`level` icosphere, outward
    winding enforced geometrically (cross . centroid > 0)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            v += [[0.0, s1, s2 * phi], [s1, s2 * phi, 0.0], [s2 * phi, 0.0, s1]]
    v = np.array(v, float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    emin = d[d > 1e-9].min()
    faces = [[i, j, k] for i in range(12) for j in range(i + 1, 12)
             for k in range(j + 1, 12)
             if abs(d[i, j] - emin) < 1e-6 and abs(d[i, k] - emin) < 1e-6
             and abs(d[j, k] - emin) < 1e-6]
    tv = v[np.array(faces)]
    for _ in range(level):
        a, b, c = tv[:, 0], tv[:, 1], tv[:, 2]
        ab, bc, ca = a + b, b + c, c + a
        for m in (ab, bc, ca):
            m /= np.linalg.norm(m, axis=1)[:, None]
        tv = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                             np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
    cr = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    flip = np.einsum("mi,mi->m", cr, tv.mean(axis=1)) < 0
    tv[flip] = tv[flip][:, ::-1]
    return np.ascontiguousarray(radius * tv)


def mean_edge(tv):
    return np.linalg.norm(tv - tv[:, [1, 2, 0]], axis=2).mean()


_VOIGT = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def voigt_to_tensor(v):
    S = np.zeros((len(v), 3, 3))
    for k, (i, j) in enumerate(_VOIGT):
        S[:, i, j] = v[:, k]
        S[:, j, i] = v[:, k]
    return S


def strain_of(sig):
    tr = np.trace(sig, axis1=1, axis2=2)
    return (sig - (LAM * tr / (2 * MU + 3 * LAM))[:, None, None] * np.eye(3)) / (2 * MU)


def stress_of(e):
    tr = np.trace(e, axis1=1, axis2=2)
    return 2 * MU * e + LAM * tr[:, None, None] * np.eye(3)


def sample_points(rmax=0.55, n=24):
    """Deterministic interior points: a Fibonacci-spiral direction set at four
    radii, plus the centre.  Deliberately NOT aligned with the icosahedral
    symmetry of the mesh, so the scatter is a fair error estimate."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = np.pi * (1.0 + np.sqrt(5.0)) * k
    dirs = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    rad = np.tile(np.array([0.15, 0.30, 0.45, 0.55]), n // 4 + 1)[:n] * rmax / 0.55
    return np.vstack([np.zeros(3), dirs * rad[:, None]])


# ---------------------------------------------------------------------------
# 3. one solve
# ---------------------------------------------------------------------------
def build(tv):
    verts = tv.reshape(-1, 3)
    tris = np.arange(len(verts)).reshape(-1, 3)
    return Patch("sphere", verts, tris, INTERFACE, ("radial", (0.0, 0.0)))


def solve_model(patch, eps, alpha, sig_inf, Bblock):
    m = ForceElementModel([patch], MU, NU, eps, alpha)
    t_F = np.einsum("ij,mj->mi", sig_inf, m.normals)
    m.assemble(t_F, np.zeros((m.N, 3)), verbose=False,
               B=np.asfortranarray(Bblock.copy()))
    cond = np.linalg.cond(m.A)
    m.solve(verbose=False)
    return m, cond


def evaluate(m, alpha, sig_inf, x):
    """True interior stress at x, plus the exact answer."""
    S = stress_matrix(x, m.tv, m.eps_arr, MU, NU)
    sig0 = voigt_to_tensor((S @ m.q.ravel()).reshape(-1, 6)) + sig_inf[None]
    sig_true = alpha * stress_of(strain_of(sig0))
    e_inf = strain_of(sig_inf[None])[0]
    e_ex = interior_strain(e_inf, MU, NU, alpha * MU, NU)
    sig_ex = alpha * stress_of(e_ex[None])[0]
    err = np.linalg.norm(sig_true - sig_ex[None], axis=(1, 2)) / np.linalg.norm(sig_ex)
    mean = sig_true.mean(axis=0)
    scatter = (np.linalg.norm(sig_true - mean[None], axis=(1, 2)).max()
               / np.linalg.norm(mean))
    return dict(err_max=err.max(), err_mean=err.mean(), scatter=scatter,
                sig_mean=mean, sig_ex=sig_ex, sig0=sig0, e_ex=e_ex)


def radial_profile(m, alpha, sig_inf, radii=(0.2, 0.35, 0.5, 0.55, 0.7, 0.8)):
    """err_max on a 24-direction shell at each radius: separates a uniform bias
    in the concentration factor from near-surface non-uniformity."""
    k = np.arange(24) + 0.5
    z = 1.0 - 2.0 * k / 24
    rr = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    th = np.pi * (1.0 + np.sqrt(5.0)) * k
    d = np.stack([rr * np.cos(th), rr * np.sin(th), z], axis=1)
    return [(r, evaluate(m, alpha, sig_inf, d * r * R)["err_max"]) for r in radii]


def concentration(sig, sig_inf):
    """A scalar concentration factor: the trace ratio for a hydrostatic load,
    the xy ratio for the shear load."""
    if abs(np.trace(sig_inf)) > 1e-12:
        return np.trace(sig) / np.trace(sig_inf)
    return sig[0, 1] / sig_inf[0, 1]


# ---------------------------------------------------------------------------
# 4. mutations (structural controls), assembled by hand from the same B
# ---------------------------------------------------------------------------
def solve_raw(tv, eps, sig_inf, normals, Bc, cF, cB, cR, x, alpha_true):
    A = cF * np.eye(3 * len(tv)) + cB * Bc
    t_F = np.einsum("ij,mj->mi", sig_inf, normals)
    q = np.linalg.solve(A, cR * t_F.ravel()).reshape(-1, 3)
    S = stress_matrix(x, tv, np.full(len(tv), eps), MU, NU)
    sig0 = voigt_to_tensor((S @ q.ravel()).reshape(-1, 6)) + sig_inf[None]
    sig_true = alpha_true * stress_of(strain_of(sig0))
    e_ex = interior_strain(strain_of(sig_inf[None])[0], MU, NU, alpha_true * MU, NU)
    sig_ex = alpha_true * stress_of(e_ex[None])[0]
    return (np.linalg.norm(sig_true - sig_ex[None], axis=(1, 2))
            / np.linalg.norm(sig_ex)).max()


MUTANTS = (
    ("free term sign flipped   -(1+a)/2", lambda a: (-0.5 * (1 + a), -(1 - a), (1 - a))),
    ("free term 1/2  (no alpha)        ", lambda a: (0.5, -(1 - a), (1 - a))),
    ("free term (1-a)/2                ", lambda a: (0.5 * (1 - a), -(1 - a), (1 - a))),
    ("RHS sign flipped                 ", lambda a: (0.5 * (1 + a), -(1 - a), -(1 - a))),
    ("(1-a) dropped from B             ", lambda a: (0.5 * (1 + a), -1.0, (1 - a))),
    ("alpha -> 1/alpha                 ", lambda a: (0.5 * (1 + 1 / a), -(1 - 1 / a), (1 - 1 / a))),
    ("B sign flipped   +(1-a)B         ", lambda a: (0.5 * (1 + a), +(1 - a), (1 - a))),
)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    ok = True
    print(f"gate {GATE}: Eshelby spherical inhomogeneity, interface row R2 only")
    print(f"  host mu={MU} nu={NU} (lam={LAM});  C_inc = alpha C_host;  R={R}")
    print(f"  eps = {EPS_RATIO} * mean edge, same ratio at both densities")

    # -- 0. the analytic reference, two independent routes -------------------
    xchk = 0.0
    for alpha in ALPHAS:
        for _, s in LOADS:
            e0 = strain_of(s[None])[0]
            d = interior_strain(e0, MU, NU, alpha * MU, NU) \
                - interior_strain_via_tensor(e0, MU, NU, alpha * MU, NU)
            xchk = max(xchk, np.abs(d).max())
    print(f"\n0. analytic cross-check (concentration form vs Mandel S-tensor "
          f"form): max|diff| = {xchk:.2e}")
    ok &= xchk < 1e-14

    # -- 1. the solves --------------------------------------------------------
    x = sample_points()
    print(f"\n1. interior field ({len(x)} sample points, |x| <= "
          f"{np.linalg.norm(x, axis=1).max():.2f} R)")
    print("   tri   alpha  load           eps      cond    err_max    err_mean"
          "   scatter")
    res = {}
    for level, ntri in ((1, 80), (2, 320), (3, 1280)):
        tv = icosphere(level, R)
        assert len(tv) == ntri
        h = mean_edge(tv)
        eps = EPS_RATIO * h
        p = build(tv)
        chat = p.centroids / np.linalg.norm(p.centroids, axis=1)[:, None]
        dotn = np.einsum("mi,mi->m", p.normals, chat).min()
        B = traction_matrix(p.centroids, p.normals, p.tv, np.full(ntri, eps), MU, NU)
        for alpha in ALPHAS:
            for name, s in LOADS:
                m, cond = solve_model(p, eps, alpha, s, B)
                r = evaluate(m, alpha, s, x)
                r.update(cond=cond, eps=eps, h=h, q=m.q, normals=m.normals,
                         dotn=dotn, ntri=ntri)
                if alpha == 0.1 and ntri in (320, 1280):
                    r["profile"] = radial_profile(m, alpha, s)
                res[(ntri, alpha, name)] = r
                print(f"  {ntri:5d} {alpha:5.2f}  {name:13s} {eps:6.4f} "
                      f"{cond:7.3f}  {r['err_max']:.3e}  {r['err_mean']:.3e}"
                      f"  {r['scatter']:.2e}")
        if ntri == 320:
            tv320, eps320, p320, B320 = tv, eps, p, B
        elif ntri == 1280:
            tv1280, eps1280, p1280, B1280 = tv, eps, p, B
        else:
            del B

    # -- 2. gate: accuracy + convergence -------------------------------------
    print("\n2. gate  (<= 3.0e-2 at 320 tri, <= 1.5e-2 at 1280, error decreasing"
          " 320 -> 1280)")
    print("   the 80-tri column is context only: it is too coarse to gate an "
          "asymptotic rate on")
    print("   alpha  load            80 tri     320 tri    1280 tri   "
          "rate 320/1280  verdict")
    for alpha in ALPHAS:
        for name, _ in LOADS:
            e80 = res[(80, alpha, name)]["err_max"]
            e320 = res[(320, alpha, name)]["err_max"]
            e1280 = res[(1280, alpha, name)]["err_max"]
            good = (e320 <= TOL[320] and e1280 <= TOL[1280] and e1280 < e320)
            ok &= good
            print(f"  {alpha:5.2f}  {name:13s} {e80:.3e}  {e320:.3e}  "
                  f"{e1280:.3e}     {e320/e1280:5.2f}      "
                  f"{'ok' if good else 'FAIL'}"
                  f"{'' if e80 > e320 else '  (80 tri not monotone)'}")

    # -- 3. direction: soft softer, stiff stiffer ----------------------------
    print("\n3. direction of the concentration (soft alpha<1 MUST be < 1, "
          "stiff alpha>1 MUST be > 1)")
    print("   alpha  load           computed   exact     verdict   "
          "(1280 triangles)")
    for alpha in ALPHAS:
        for name, s in LOADS:
            r = res[(1280, alpha, name)]
            c = concentration(r["sig_mean"], s)
            cx = concentration(r["sig_ex"], s)
            good = (c < 1.0) == (alpha < 1.0) and abs(c - 1.0) > 1e-3
            ok &= good
            print(f"  {alpha:5.2f}  {name:13s} {c:8.5f}  {cx:8.5f}   "
                  f"{'ok' if good else 'FAIL'}       "
                  f"{'lower' if c < 1 else 'higher'} than remote")

    # -- 4. uniformity (needs no analytic reference) -------------------------
    print("\n4. interior uniformity: max scatter over the sample points, "
          "relative to the mean")
    print("   (Eshelby's interior field is EXACTLY uniform, so this is a pure "
          "discretisation estimate)")
    for alpha in ALPHAS:
        for name, _ in LOADS:
            s320 = res[(320, alpha, name)]["scatter"]
            s1280 = res[(1280, alpha, name)]["scatter"]
            print(f"  alpha={alpha:4.2f} {name:13s} 320: {s320:.2e}   "
                  f"1280: {s1280:.2e}   ratio {s320/max(s1280,1e-300):5.2f}")

    # -- 4b. where the error lives (alpha = 0.1) -----------------------------
    print("\n4b. err_max on a 24-point shell vs its radius, alpha = 0.1")
    print("    (a flat profile is a bias in the concentration factor; a rising "
          "one is facet-induced\n     non-uniformity near the interface.  The "
          "gate samples |x| <= 0.55 R.)")
    for ntri in (320, 1280):
        for name, _ in LOADS:
            prof = res[(ntri, 0.1, name)]["profile"]
            cells = "  ".join(f"{r:.2f}:{e:.2e}" for r, e in prof)
            print(f"   {ntri:5d} {name:13s} " + cells)

    # -- 5. the density itself, against the exact polarization ---------------
    print("\n5. surface density vs the exact polarization  q = (1-a) C0:eps_in . n")
    for alpha in (0.1, 2.0):
        line = []
        for ntri in (320, 1280):
            r = res[(ntri, alpha, "simple shear")]
            sig0_in = stress_of(r["e_ex"][None])[0]
            q_ex = (1.0 - alpha) * np.einsum("ij,mj->mi", sig0_in, r["normals"])
            rel = (np.linalg.norm(r["q"] - q_ex, axis=1).max()
                   / np.abs(q_ex).max())
            line.append(f"{ntri}: {rel:.3e}")
        print(f"  alpha={alpha:4.2f} shear   max relative defect   " + "   ".join(line))

    # -- 6. the row that model.assemble actually built -----------------------
    print("\n6. structure of the assembled interface row (320 tri, alpha=0.1)")
    m = ForceElementModel([p320], MU, NU, eps320, 0.1)
    t_F = np.einsum("ij,mj->mi", HYD, m.normals)
    Am, bm = m.assemble(t_F, np.zeros((m.N, 3)), verbose=False,
                        B=np.asfortranarray(B320.copy()))
    dA = np.abs(Am - (0.5 * 1.1 * np.eye(3 * m.N) - 0.9 * B320)).max()
    db = np.abs(bm - 0.9 * t_F.ravel()).max()
    print(f"   max|A - ((1+a)/2 I - (1-a) B)| = {dA:.3e}      "
          f"max|b - (1-a) t_F| = {db:.3e}")
    print(f"   normals: min dot(n, r_hat) = {res[(320,0.1,'hydrostatic')]['dotn']:.6f}"
          f"   (Patch flipped {p320.flipped} of {p320.n})")
    ok &= dA <= 1e-15 and db <= 1e-15 and res[(320, 0.1, 'hydrostatic')]['dotn'] > 0.999
    del Am, bm, m

    # -- 7. structural controls: every mutation must miss ---------------------
    print("\n7. structural controls (320 tri): each mutation of R2 must exceed "
          f"the {TOL[320]:.1e} tolerance")
    Bc = np.ascontiguousarray(B320)
    base = {a: max(res[(320, a, n)]["err_max"] for n, _ in LOADS) for a in ALPHAS}
    print(f"   {'':34s}  " + "   ".join(f"a={a:<5.2f}" for a in ALPHAS))
    print(f"   {'row as assembled by model.py':34s}  "
          + "   ".join(f"{base[a]:.2e}" for a in ALPHAS) + "   (the gate above)")
    controls = list(MUTANTS)
    # the normal flip needs B rebuilt on the flipped field normal
    Bf = np.ascontiguousarray(traction_matrix(p320.centroids, -p320.normals,
                                              p320.tv, np.full(len(tv320), eps320),
                                              MU, NU))
    for name, f in controls + [("interface normal flipped         ", None)]:
        cell, worst = [], 0.0
        for alpha in ALPHAS:
            w = 0.0
            for _, s in LOADS:
                if f is None:
                    e = solve_raw(tv320, eps320, s, -p320.normals, Bf,
                                  0.5 * (1 + alpha), -(1 - alpha), (1 - alpha),
                                  x, alpha)
                else:
                    cF, cB, cR = f(alpha)
                    e = solve_raw(tv320, eps320, s, p320.normals, Bc,
                                  cF, cB, cR, x, alpha)
                w = max(w, e)
            cell.append(f"{w:.2e}")
            worst = max(worst, w)
        good = worst > TOL[320]
        ok &= good
        print(f"   {name:34s}  " + "   ".join(cell) +
              f"   {'caught' if good else 'MISSED -- gate is blind to this'}"
              f" ({worst/max(base.values()):5.1f}x)")
    del Bf, Bc
    # the B-sign control again at 1280: a wrong operator does not converge
    Bc2 = np.ascontiguousarray(B1280)
    flip1280 = max(solve_raw(tv1280, eps1280, s, p1280.normals, Bc2,
                             0.5 * (1 + 0.1), +(1 - 0.1), (1 - 0.1), x, 0.1)
                   for _, s in LOADS)
    print(f"   {'B sign flipped, at 1280 tri (a=0.1)':34s}  {flip1280:.2e}"
          f"                         vs {res[(1280,0.1,'simple shear')]['err_max']:.2e}"
          f" correct: the wrong operator does not converge")
    ok &= flip1280 > TOL[1280]
    del Bc2, B1280
    print("   NOTE the two weak controls, and which alpha carries them:")
    print("     '(1-a) dropped from B' is only 1.4x the correct error at "
          "alpha = 0.1 (the coefficient\n       moves 0.9 -> 1.0); it is caught "
          "at alpha = 2.0, where dropping (1-a) also flips the sign.")
    print("     'B sign flipped' is caught at alpha = 0.1 (1.0e-1, and it does "
          "NOT converge: 9.2e-2 at\n       1280) but would be MISSED at "
          "alpha = 0.5 alone (2.5e-2 < 3.0e-2).  A sphere is a weak\n       "
          "discriminator of the B sign -- pin it on an ellipsoid.")

    # -- 8. eps robustness: the gate is not tuned to one eps -----------------
    print("\n8. eps robustness at 320 triangles (worst case over alpha and load)")
    for ratio in (0.25, 0.5, 0.75, 0.9):
        eps = ratio * mean_edge(tv320)
        pr = build(tv320)
        Br = np.ascontiguousarray(traction_matrix(pr.centroids, pr.normals, pr.tv,
                                                  np.full(len(tv320), eps), MU, NU))
        worst = max(solve_raw(tv320, eps, s, pr.normals, Br, 0.5 * (1 + a),
                              -(1 - a), (1 - a), x, a)
                    for a in ALPHAS for _, s in LOADS)
        print(f"   eps/h = {ratio:4.2f}  (eps/R = {eps/R:5.3f}, sample points "
              f"{(1-0.55)/eps:4.1f} eps clear)   worst "
              f"err = {worst:.3e}   {'<=' if worst <= TOL[320] else '> '} "
              f"{TOL[320]:.1e}"
              + ("   <-- used by this gate" if ratio == EPS_RATIO else ""))
        del Br

    print(f"\n  ({time.time()-t_start:.0f}s)")
    print(f"{'PASS' if ok else 'FAIL'} {GATE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
