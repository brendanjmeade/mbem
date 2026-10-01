"""Point kernels in ``clq.pointwise`` vs sympy.

The mollified Kelvin solution is built symbolically from

    G_ij = C1 [ (3-4nu) d_ij/R + x_i x_j/R^3 + 2(1-nu) eps^2 d_ij/R^3 ],
    R^2 = |x|^2 + eps^2,   C1 = 1/(16 pi mu (1-nu)),

with symbols x1, x2, x3, mu, nu, eps, and every reference below is evaluated
from the symbolic expressions with 30-digit mpmath arithmetic.

  (a) kelvin_G / kelvin_dG / kelvin_d2G vs the symbolic G and its first and
      second derivatives at 3 random points, nu = 0.3, 1e-12 relative;
  (b) Cauchy-Navier residual  mu lap G_ij + (lam+mu) d_i (d_k G_kj) + d_ij phi
      with lam = 2 mu nu/(1-2nu), phi = 15 eps^4/(8 pi R^7), for nu in
      {0, 0.25, 0.3, 0.49}: 30-digit evaluation at random points (1e-12 of phi),
      an exact-rational sympy evaluation (identically 0), and the same residual
      assembled from clq's numeric kelvin_d2G and blob (1e-11, cancellation
      limited) -- this ties the eigenstress blob to the kernel;
  (c) dd_displacement_point == -C_{jmpq} n_m dG_ip/dx_q with the explicit
      isotropic C_{jmpq} = lam d_jm d_pq + mu (d_jp d_mq + d_jq d_mp), slip j
      and normal m in the FIRST pair, random unit normal, nu = 0.3, 1e-12;
      tripwire: the alternative pairing -C_{jqmp} n_m dG_ip/dx_q (slip with
      the derivative pair) differs by > 1e-2 at nu = 0.3, and coincides at
      nu = 1/4 (lam = mu, C fully symmetric) which is why that value cannot
      discriminate the two;
  (d) dd_stress_point == C_{mlab} d/dx_b (dd_displacement_point)_{aj} by sympy
      differentiation of the symbolic displacement point kernel (stress =
      C : sym grad u), nu = 0.3, 1e-12; plus stress symmetry.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import mpmath

from _common import MU, Report, relmax
from clq import pointwise as pw

# --- symbols ----------------------------------------------------------------
x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
mu_s, nu_s, eps_s = sp.symbols("mu nu eps", positive=True)
n1, n2, n3 = sp.symbols("n1 n2 n3", real=True)
XS = (x1, x2, x3)
NS = (n1, n2, n3)
ARGS = (x1, x2, x3, mu_s, nu_s, eps_s)
ARGS_N = (x1, x2, x3, n1, n2, n3, mu_s, nu_s, eps_s)


def delta(i, j):
    return 1 if i == j else 0


def symbolic_kernels():
    """G, dG/dx_m, d2G/dx_s dx_q, phi and lam as symbolic nested lists."""
    R = sp.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2 + eps_s ** 2)
    C1 = 1 / (16 * sp.pi * mu_s * (1 - nu_s))
    G = [[C1 * ((3 - 4 * nu_s) * delta(i, j) / R + XS[i] * XS[j] / R ** 3
                + 2 * (1 - nu_s) * eps_s ** 2 * delta(i, j) / R ** 3)
          for j in range(3)] for i in range(3)]
    DG = [[[sp.diff(G[i][j], XS[m]) for m in range(3)] for j in range(3)] for i in range(3)]
    D2 = [[[[sp.diff(DG[i][j][s], XS[q]) for q in range(3)] for s in range(3)]
           for j in range(3)] for i in range(3)]
    phi = 15 * eps_s ** 4 / (8 * sp.pi * R ** 7)
    lam = 2 * mu_s * nu_s / (1 - 2 * nu_s)
    return G, DG, D2, phi, lam


def iso_C_sym(lam, mu):
    """C_{jmpq} = lam d_jm d_pq + mu (d_jp d_mq + d_jq d_mp), symbolic entries."""
    return {(j, m, p, q): lam * delta(j, m) * delta(p, q) + mu * (delta(j, p) * delta(m, q) + delta(j, q) * delta(m, p))
            for j in range(3) for m in range(3) for p in range(3) for q in range(3)}


def iso_C_num(lam, mu):
    d = np.eye(3)
    return (lam * np.einsum("jm,pq->jmpq", d, d)
            + mu * (np.einsum("jp,mq->jmpq", d, d) + np.einsum("jq,mp->jmpq", d, d)))


def mpf_args(x, extra):
    """Exact mpf copies of the double inputs (so the reference sees the same numbers)."""
    return [mpmath.mpf(float(v)) for v in x] + [mpmath.mpf(float(v)) for v in extra]


def main():
    rep = Report("clq.pointwise vs sympy (mollified Kelvin point kernels)")
    mpmath.mp.dps = 30
    G, DG, D2, phi, lam_s = symbolic_kernels()
    f_G = sp.lambdify(ARGS, G, modules="mpmath")
    f_DG = sp.lambdify(ARGS, DG, modules="mpmath")
    f_D2 = sp.lambdify(ARGS, D2, modules="mpmath")
    f_phi = sp.lambdify(ARGS, phi, modules="mpmath")

    rng = np.random.default_rng(20260904)
    eps = 0.4
    pts = rng.uniform(-1.0, 1.0, (3, 3))        # R/eps between ~1 and ~4.5: blob and tails
    nhat = rng.standard_normal(3)
    nhat /= np.linalg.norm(nhat)

    # --- (a) G, dG, d2G vs symbolic -----------------------------------------
    for mu in (MU, 2.5):
        nu = 0.3
        Gs = np.array([f_G(*mpf_args(p, (mu, nu, eps))) for p in pts], dtype=float)
        DGs = np.array([f_DG(*mpf_args(p, (mu, nu, eps))) for p in pts], dtype=float)
        D2s = np.array([f_D2(*mpf_args(p, (mu, nu, eps))) for p in pts], dtype=float)
        rep.check(f"(a) mu={mu}: kelvin_G vs symbolic G", relmax(pw.kelvin_G(pts, mu, nu, eps), Gs), 1e-12)
        rep.check(f"(a) mu={mu}: kelvin_dG vs symbolic dG/dx_m", relmax(pw.kelvin_dG(pts, mu, nu, eps), DGs), 1e-12)
        rep.check(f"(a) mu={mu}: kelvin_d2G vs symbolic d2G/dx_s dx_q", relmax(pw.kelvin_d2G(pts, mu, nu, eps), D2s), 1e-12)

    # --- (b) Cauchy-Navier residual L G = -phi I ------------------------------
    res_sym = [[mu_s * sum(D2[i][j][s][s] for s in range(3))
                + (lam_s + mu_s) * sum(D2[k][j][i][k] for k in range(3))
                + delta(i, j) * phi for j in range(3)] for i in range(3)]
    f_res = sp.lambdify(ARGS, res_sym, modules="mpmath")
    rat_pt = {x1: sp.Rational(1, 3), x2: sp.Rational(-2, 5), x3: sp.Rational(7, 4),
              eps_s: sp.Rational(1, 2), mu_s: sp.Rational(3, 2)}
    nus_exact = {0.0: sp.Integer(0), 0.25: sp.Rational(1, 4), 0.3: sp.Rational(3, 10), 0.49: sp.Rational(49, 100)}
    pts_b = rng.uniform(-1.0, 1.0, (4, 3))
    for nu in (0.0, 0.25, 0.3, 0.49):
        mu = MU
        lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
        # 30-digit evaluation of the symbolic operator, relative to phi at each point
        worst = 0.0
        for p in pts_b:
            a = mpf_args(p, (mu, nu, eps))
            r = np.array(f_res(*a), dtype=float)
            worst = max(worst, np.abs(r).max() / float(f_phi(*a)))
        rep.check(f"(b) nu={nu}: |mu lap G + (lam+mu) grad div G + phi I| / phi (30 digits)", worst, 1e-12)
        # exact rationals: the residual must simplify to 0 identically
        exact = [[sp.simplify(res_sym[i][j].subs(rat_pt).subs(nu_s, nus_exact[nu])) for j in range(3)] for i in range(3)]
        rep.check_bool(f"(b) nu={nu}: exact-rational residual is identically 0",
                       all(e == 0 for row in exact for e in row), f"(residual {exact[0][0]}, {exact[0][1]})")
        # the same residual from clq's numeric second derivatives and blob
        D2n = pw.kelvin_d2G(pts_b, mu, nu, eps)
        phin = pw.blob(pts_b, eps)
        resn = (mu * np.einsum("nijss->nij", D2n) + (lam + mu) * np.einsum("nkjik->nij", D2n)
                + np.eye(3)[None] * phin[:, None, None])
        rep.check(f"(b) nu={nu}: same residual from clq kelvin_d2G and blob (double)",
                  (np.abs(resn).max(axis=(1, 2)) / phin).max(), 1e-11)

    # --- (c) displacement pairing --------------------------------------------
    nu = 0.3
    mu = MU
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    C = iso_C_num(lam, mu)
    DGs = np.array([f_DG(*mpf_args(p, (mu, nu, eps))) for p in pts], dtype=float)
    U_ref = -np.einsum("jmpq,m,nipq->nij", C, nhat, DGs)        # slip j, normal m in the first pair
    U_alt = -np.einsum("jqmp,m,nipq->nij", C, nhat, DGs)        # slip j with the derivative index q
    U_clq = pw.dd_displacement_point(pts, nhat, mu, nu, eps)
    rep.check("(c) nu=0.3: dd_displacement_point vs -C_{jmpq} n_m dG_ip/dx_q", relmax(U_clq, U_ref), 1e-12)
    d_alt = relmax(U_alt, U_ref)
    rep.check_bool("(c) nu=0.3: tripwire, pairing -C_{jqmp} n_m dG_ip/dx_q differs by > 1e-2",
                   d_alt > 1e-2, f"(rel diff {d_alt:.2e})")
    C_q = iso_C_num(mu, mu)                                       # nu = 1/4: lam = mu
    DGq = np.array([f_DG(*mpf_args(p, (mu, 0.25, eps))) for p in pts], dtype=float)
    rep.check("(c) nu=0.25: both pairings coincide (lam = mu)",
              relmax(-np.einsum("jqmp,m,nipq->nij", C_q, nhat, DGq), -np.einsum("jmpq,m,nipq->nij", C_q, nhat, DGq)), 1e-12)
    rep.check("(c) nu=0.25: dd_displacement_point vs -C_{jmpq} n_m dG_ip/dx_q",
              relmax(pw.dd_displacement_point(pts, nhat, mu, 0.25, eps), -np.einsum("jmpq,m,nipq->nij", C_q, nhat, DGq)), 1e-12)

    # --- (d) stress = C : grad u by sympy differentiation of U ----------------
    Cs = iso_C_sym(lam_s, mu_s)
    U_sym = [[-sum(Cs[(j, m, p, q)] * NS[m] * DG[i][p][q]
                   for m in range(3) for p in range(3) for q in range(3) if Cs[(j, m, p, q)] != 0)
              for j in range(3)] for i in range(3)]
    dU = [[[sp.diff(U_sym[a][j], XS[b]) for b in range(3)] for j in range(3)] for a in range(3)]
    S_sym = [[[sum(Cs[(m, l, a, b)] * dU[a][j][b] for a in range(3) for b in range(3) if Cs[(m, l, a, b)] != 0)
               for j in range(3)] for l in range(3)] for m in range(3)]
    f_U = sp.lambdify(ARGS_N, U_sym, modules="mpmath")
    f_S = sp.lambdify(ARGS_N, S_sym, modules="mpmath")
    U_sym_n = np.array([f_U(*mpf_args(p, (*nhat, mu, nu, eps))) for p in pts], dtype=float)
    S_ref = np.array([f_S(*mpf_args(p, (*nhat, mu, nu, eps))) for p in pts], dtype=float)
    S_clq = pw.dd_stress_point(pts, nhat, mu, nu, eps)
    rep.check("(d) nu=0.3: symbolic U (sanity) vs dd_displacement_point", relmax(U_sym_n, U_clq), 1e-12)
    rep.check("(d) nu=0.3: dd_stress_point vs C_{mlab} d_b U_aj (sympy)", relmax(S_clq, S_ref), 1e-12)
    rep.check("(d) nu=0.3: dd_stress_point symmetric in (m, l)", relmax(S_clq, np.swapaxes(S_clq, 1, 2)), 1e-13)
    # and at a second nu (lam != mu, nearly incompressible) to exercise the lam terms
    nu2 = 0.45
    S_ref2 = np.array([f_S(*mpf_args(p, (*nhat, mu, nu2, eps))) for p in pts], dtype=float)
    rep.check("(d) nu=0.45: dd_stress_point vs C_{mlab} d_b U_aj (sympy)",
              relmax(pw.dd_stress_point(pts, nhat, mu, nu2, eps), S_ref2), 1e-12)
    rep.finish()


if __name__ == "__main__":
    main()
