"""
Pointwise verification: kelvin_d2G implementation == sympy second derivative
of the Galerkin/Cortez-convolved G^eps.
"""

import numpy as np
import sympy as sp

from moss_kernel.mollified_elastic_kernels import kelvin_d2G


def main():
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    mu_s, nu_s, eps_s = sp.symbols("mu nu eps", positive=True, real=True)

    d_sym = [x1, x2, x3]
    R2 = sum(di * di for di in d_sym)
    Re = sp.sqrt(R2 + eps_s**2)
    C1_sym = 1 / (16 * sp.pi * mu_s * (1 - nu_s))

    def G_galerkin(i, j):
        delta = 1 if i == j else 0
        return C1_sym * (
            (3 - 4 * nu_s) * delta / Re
            + d_sym[i] * d_sym[j] / Re**3
            + 2 * (1 - nu_s) * eps_s**2 * delta / Re**3
        )

    # Test point and parameters
    d_num = np.array([1.3, 0.7, -0.5])
    mu_num = 1.0
    nu_num = 0.25
    eps_num = 0.3

    sub = {x1: d_num[0], x2: d_num[1], x3: d_num[2],
           mu_s: mu_num, nu_s: nu_num, eps_s: eps_num}

    # Numerical via implementation
    D2G_num = kelvin_d2G(d_num, mu_num, nu_num, eps_num)

    # Symbolic via sympy and evaluate
    print("Computing symbolic d^2 G_rp / d d_s d d_q ...")
    D2G_sym = np.zeros((3, 3, 3, 3))
    for r in range(3):
        for p in range(3):
            G_rp = G_galerkin(r, p)
            for s in range(3):
                for q in range(3):
                    expr = sp.diff(sp.diff(G_rp, d_sym[s]), d_sym[q])
                    D2G_sym[r, p, s, q] = float(expr.subs(sub).evalf())

    diff = D2G_num - D2G_sym
    max_abs = np.max(np.abs(diff))
    max_ref = np.max(np.abs(D2G_sym))
    print(f"\nmax |D2G_num - D2G_sym| = {max_abs:.3e}")
    print(f"max |D2G_sym|           = {max_ref:.3e}")
    print(f"relative                = {max_abs/max_ref:.3e}")
    if max_abs / max_ref < 1e-12:
        print("\nPASS: implementation matches Galerkin/Cortez form to machine precision")
    else:
        print("\nFAIL: implementation does NOT match symbolic form")


if __name__ == "__main__":
    main()
