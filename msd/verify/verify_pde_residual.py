"""
PDE-residual sanity check for the regularized Kelvin Green's function.

Goal: confirm that the Galerkin / Cortez-convolved form
    G^eps_ij = C1 [(3-4nu) delta_ij/R_eps + d_i d_j/R_eps^3 + 2(1-nu) eps^2 delta_ij/R_eps^3]
satisfies the regularized Cauchy-Navier equation
    L_ij G^eps_jk(x) = -delta_ik * phi^(C)(R)
where L_ij = mu nabla^2 delta_ij + (lambda+mu) d_i d_j and
phi^(C)(R) = 15 eps^4 / (8 pi R_eps^7) is the Cortez (2001) blob.

Compares against the codebase form (no eps^2 term) and the manuscript form
(coefficient 1 instead of 2(1-nu)). Only the Galerkin form should give
machine-precision residuals.

Uses sympy for exact symbolic differentiation, then evaluates at fixed (x, mu, nu, eps).
"""

import sympy as sp


def main():
    # Symbols
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    mu, nu, eps = sp.symbols("mu nu eps", positive=True, real=True)

    d = [x1, x2, x3]
    R2 = sum(di * di for di in d)
    Re = sp.sqrt(R2 + eps**2)
    Re3 = Re**3
    Re7 = Re**7

    C1 = 1 / (16 * sp.pi * mu * (1 - nu))
    lam = 2 * mu * nu / (1 - 2 * nu)

    def kron(i, j):
        return 1 if i == j else 0

    # Three candidate forms
    def G_codebase(i, j):
        return C1 * ((3 - 4 * nu) * kron(i, j) / Re + d[i] * d[j] / Re3)

    def G_manuscript(i, j):
        return G_codebase(i, j) + C1 * eps**2 * kron(i, j) / Re3

    def G_galerkin(i, j):
        return G_codebase(i, j) + C1 * 2 * (1 - nu) * eps**2 * kron(i, j) / Re3

    # Cortez blob phi^(C)(R) = 15 eps^4 / (8 pi R_eps^7)
    phi_C = 15 * eps**4 / (8 * sp.pi * Re7)

    # Cauchy-Navier residual: L_ij G_jk(x) + delta_ik phi^(C)(R)
    # L_ij = mu d_p d_p delta_ij + (lambda+mu) d_i d_j
    def residual(G_func, i, k):
        # Sum over j: L_ij G_jk
        # = mu nabla^2 G_ik + (lambda+mu) sum_j d_i d_j G_jk
        term_lap = sum(sp.diff(G_func(i, k), d[p], 2) for p in range(3))
        term_div = sum(
            sp.diff(sp.diff(G_func(j, k), d[i]), d[j]) for j in range(3)
        )
        return mu * term_lap + (lam + mu) * term_div + kron(i, k) * phi_C

    # Numerical evaluation points (avoid nu=1/2 since lambda blows up there;
    # nu=0.49 captures the near-incompressible limit safely)
    test_points = [
        ("nu=0.0", {x1: 1.3, x2: 0.7, x3: -0.5, mu: 1.0, nu: sp.Rational(0), eps: sp.Rational(3, 10)}),
        ("nu=1/4", {x1: 1.3, x2: 0.7, x3: -0.5, mu: 1.0, nu: sp.Rational(1, 4), eps: sp.Rational(3, 10)}),
        ("nu=1/3", {x1: 1.3, x2: 0.7, x3: -0.5, mu: 1.0, nu: sp.Rational(1, 3), eps: sp.Rational(3, 10)}),
        ("nu=0.49", {x1: 1.3, x2: 0.7, x3: -0.5, mu: 1.0, nu: sp.Rational(49, 100), eps: sp.Rational(3, 10)}),
    ]

    print("=" * 78)
    print("PDE residual:  L_ij G^eps_jk(x) + delta_ik phi^(C)(R)")
    print("Should be 0 when G^eps satisfies the regularized Cauchy-Navier PDE")
    print("=" * 78)

    for label, G_func in [
        ("CODEBASE (no eps^2 term) ", G_codebase),
        ("MANUSCRIPT (coef 1)      ", G_manuscript),
        ("GALERKIN  (coef 2(1-nu)) ", G_galerkin),
    ]:
        print(f"\n--- {label} ---")
        for sub_label, sub_dict in test_points:
            max_abs = 0.0
            for k in range(3):
                for i in range(3):
                    res = residual(G_func, i, k)
                    val = float(res.subs(sub_dict).evalf())
                    if abs(val) > max_abs:
                        max_abs = abs(val)
            print(f"  {sub_label}: max |residual| = {max_abs:.3e}")

    print("\n" + "=" * 78)
    print("Expected: GALERKIN ~ machine precision at every nu;")
    print("          MANUSCRIPT (coef 1) only at nu near 1/2;")
    print("          CODEBASE never.")
    print("=" * 78)


if __name__ == "__main__":
    main()
