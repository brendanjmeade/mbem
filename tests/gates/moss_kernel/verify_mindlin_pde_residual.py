"""
verify_mindlin_pde_residual.py
==============================

Numerical verification harness for mollified Mindlin half-space Green's
function in mindlin_kernels.py.

Two residuals are measured:

  1. PDE residual in the bulk: L_ij G^{eps,HS}_{jk}(x, x') + delta_ik phi_eps(R1)
     should be ~0 in z < 0, modulo image-blob leakage and any approximations
     in the correction term.

  2. Boundary-traction residual at z = 0: T_3k(x_3=0) should be ~0 (exactly
     zero in classical Mindlin; expected to be O(eps^2) in our current
     "literal R -> R_eps" mollification of the correction term).

Both are computed via finite-difference of mindlin_G at multiple test
points; tighter symbolic verification can be added later.

Run:
    python mollified_kernel/verify_mindlin_pde_residual.py
"""

import os
import sys

# Allow running from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from moss_kernel.mindlin_kernels import mindlin_G


# ============================================================
# Helpers
# ============================================================


def _navier_apply_fd(G_func, obs, source, mu, nu, eps, k, h=1e-3):
    """
    Compute (L u)_i for u_i = G[i, k] from finite differences.

    L u_i = mu * laplacian(u_i) + (lambda + mu) * d_i (div u)
          = mu * sum_p d^2 u_i / dx_p^2  +  (lambda + mu) * sum_j d^2 u_j / (dx_i dx_j)

    Returns a length-3 vector.
    """
    lam = 2 * mu * nu / (1 - 2 * nu)
    coef_div = lam + mu

    obs = np.asarray(obs, dtype=float)

    # Evaluate G[:, k] (displacement field from a unit force in direction k)
    # at a small stencil of points to estimate Laplacian and divergence-gradient

    def u(pt):
        G = G_func(pt, source, mu, nu, eps)
        return G[:, k]

    # Build 2nd derivatives tensor: d^2 u_i / (dx_p dx_q)
    d2u = np.zeros((3, 3, 3))  # [i, p, q]
    for p in range(3):
        e_p = np.zeros(3)
        e_p[p] = h
        for q in range(3):
            e_q = np.zeros(3)
            e_q[q] = h
            if p == q:
                u_pp = u(obs + e_p)
                u_0 = u(obs)
                u_mm = u(obs - e_p)
                d2u[:, p, q] = (u_pp - 2 * u_0 + u_mm) / (h * h)
            else:
                u_pp = u(obs + e_p + e_q)
                u_pm = u(obs + e_p - e_q)
                u_mp = u(obs - e_p + e_q)
                u_mm = u(obs - e_p - e_q)
                d2u[:, p, q] = (u_pp - u_pm - u_mp + u_mm) / (4 * h * h)

    # Laplacian of u_i = sum_p d2u[i, p, p]
    lap = np.array([sum(d2u[i, p, p] for p in range(3)) for i in range(3)])
    # div u = sum_j d u_j / dx_j -- we need first derivatives. Compute via first-order FD.
    # But d_i(div u) = sum_j d^2 u_j / (dx_i dx_j) = sum_j d2u[j, i, j]
    grad_div = np.array([sum(d2u[j, i, j] for j in range(3)) for i in range(3)])

    return mu * lap + coef_div * grad_div


def _phi_blob(R1, eps):
    """Cortez blob phi^(C)(R) = 15 eps^4 / (8 pi R_eps^7)"""
    Re = np.sqrt(R1 * R1 + eps * eps)
    return 15 * eps**4 / (8 * np.pi * Re**7)


def _stress_fd(G_func, obs, source, mu, nu, eps, k, h=1e-3):
    """
    Compute stress sigma_ij at obs due to unit force in direction k.

    sigma_ij = lambda delta_ij div(u) + mu (d u_i/d x_j + d u_j / d x_i)
    """
    lam = 2 * mu * nu / (1 - 2 * nu)
    obs = np.asarray(obs, dtype=float)

    def u(pt):
        G = G_func(pt, source, mu, nu, eps)
        return G[:, k]

    # First derivatives: du[i, j] = d u_i / d x_j
    du = np.zeros((3, 3))
    for j in range(3):
        e_j = np.zeros(3)
        e_j[j] = h
        u_p = u(obs + e_j)
        u_m = u(obs - e_j)
        du[:, j] = (u_p - u_m) / (2 * h)

    div_u = du[0, 0] + du[1, 1] + du[2, 2]
    sig = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            sig[i, j] = lam * (1.0 if i == j else 0.0) * div_u + mu * (du[i, j] + du[j, i])
    return sig


def _traction_z(G_func, obs, source, mu, nu, eps, k, h=1e-3):
    """Surface traction T_3k = sigma_{3,:}[k] at obs from unit force in direction k."""
    sig = _stress_fd(G_func, obs, source, mu, nu, eps, k, h=h)
    return sig[2, :]


# ============================================================
# Phase 1: PDE residual in bulk
# ============================================================


def check_pde_residual():
    print("=" * 78)
    print("PDE residual in z < 0: L G^{eps,HS}(x, x') + delta phi_eps(R1) ~ 0")
    print("=" * 78)
    print("(numerical via finite-difference; exact value should be zero modulo")
    print(" image-blob leakage and any correction-term mollification gaps)\n")

    mu = 1.0
    nu_values = [0.0, 0.25, 0.49]
    eps_values = [0.5, 0.2, 0.1, 0.05]

    rng = np.random.default_rng(123)
    n_pts = 5
    obs_pts = []
    sources = []
    for _ in range(n_pts):
        # Random obs in bulk
        obs = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, -0.3)])
        # Random source deeper than obs, well-buried
        z_src = obs[2] + rng.uniform(-2, -0.3)  # source deeper than obs
        src = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), z_src])
        obs_pts.append(obs)
        sources.append(src)

    print(f"  {'nu':>6s}  {'eps':>6s}  {'max |L G + del phi|':>22s}  "
          f"{'max |G|':>10s}  {'rel resid':>10s}")
    print("  " + "-" * 70)

    for nu in nu_values:
        for eps in eps_values:
            max_resid = 0.0
            max_G = 0.0
            for obs, src in zip(obs_pts, sources):
                d1 = obs - src
                R1 = float(np.linalg.norm(d1))
                phi = _phi_blob(R1, eps)
                # FD stencil size — needs to be larger when eps is large, smaller otherwise
                h_fd = max(1e-3, 0.02 * eps)
                # Skip if obs too close to source for our FD stencil (avoid eps-scale numerics)
                if R1 < 5 * h_fd:
                    continue
                for k in range(3):
                    Lu = _navier_apply_fd(mindlin_G, obs, src, mu, nu, eps, k, h=h_fd)
                    for i in range(3):
                        target = (1.0 if i == k else 0.0) * phi
                        resid = Lu[i] + target
                        if abs(resid) > max_resid:
                            max_resid = abs(resid)
                G = mindlin_G(obs, src, mu, nu, eps)
                Gnorm = float(np.max(np.abs(G)))
                if Gnorm > max_G:
                    max_G = Gnorm
            rel = max_resid / max_G if max_G > 0 else float("nan")
            print(f"  {nu:6.2f}  {eps:6.3f}  {max_resid:22.3e}  {max_G:10.3e}  {rel:10.3e}")
    print()


# ============================================================
# Phase 2: Boundary traction at z = 0
# ============================================================


def check_boundary_traction():
    print("=" * 78)
    print("Boundary traction at z = 0: T_3k(x_1, x_2, 0) ~ 0")
    print("=" * 78)
    print("(numerical via finite-difference; classical Mindlin gives exact zero;")
    print(" current mollification expected to give O(eps^2) residual)\n")

    mu = 1.0
    nu_values = [0.0, 0.25, 0.49]
    eps_values = [0.5, 0.2, 0.1, 0.05]

    rng = np.random.default_rng(456)
    n_pts = 5
    # Surface obs (z = 0), but slightly below for numerical safety with the FD stencil
    # Sources strictly buried
    obs_pts_surface = []
    sources = []
    for _ in range(n_pts):
        obs = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])
        src = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, -0.8)])
        obs_pts_surface.append(obs)
        sources.append(src)

    print(f"  {'nu':>6s}  {'eps':>6s}  {'max |T_3k(z=0)|':>18s}  "
          f"{'max bulk stress':>16s}  {'rel resid':>10s}")
    print("  " + "-" * 70)

    for nu in nu_values:
        for eps in eps_values:
            max_T = 0.0
            max_sig_bulk = 0.0
            for obs, src in zip(obs_pts_surface, sources):
                # FD stencil: must stay in half-space (z <= 0) -- offset obs slightly below surface
                h_fd = max(1e-3, 0.02 * eps)
                obs_eval = obs.copy()
                obs_eval[2] = -h_fd  # one stencil away from surface
                for k in range(3):
                    T = _traction_z(mindlin_G, obs_eval, src, mu, nu, eps, k, h=h_fd)
                    for i in range(3):
                        if abs(T[i]) > max_T:
                            max_T = abs(T[i])
                # Bulk stress reference: evaluate stress at a typical deep point
                obs_deep = src + np.array([0.0, 0.0, 0.5])  # half a unit above source
                if obs_deep[2] > -h_fd:
                    obs_deep[2] = -h_fd
                for k in range(3):
                    sig = _stress_fd(mindlin_G, obs_deep, src, mu, nu, eps, k, h=h_fd)
                    snorm = float(np.max(np.abs(sig)))
                    if snorm > max_sig_bulk:
                        max_sig_bulk = snorm
            rel = max_T / max_sig_bulk if max_sig_bulk > 0 else float("nan")
            print(f"  {nu:6.2f}  {eps:6.3f}  {max_T:18.3e}  {max_sig_bulk:16.3e}  {rel:10.3e}")
    print()


# ============================================================
# Phase 3: convergence as eps -> 0
# ============================================================


def check_eps_convergence():
    """
    Fix a (source, obs) pair away from the boundary, sweep eps, and verify
    G^{eps,HS} converges as eps -> 0 with second-order rate.
    """
    print("=" * 78)
    print("Convergence as eps -> 0 (fixed source/obs, away from boundary)")
    print("=" * 78)
    print("Expected: max|G(eps) - G(eps_small)| ~ eps^2\n")

    mu = 1.0
    nu = 0.25
    obs = np.array([1.5, 0.7, -0.8])
    source = np.array([0.0, 0.0, -2.0])

    eps_ref = 0.005
    G_ref = mindlin_G(obs, source, mu, nu, eps_ref)
    print(f"  Reference G at eps = {eps_ref}:")
    print(f"  obs    = {obs}")
    print(f"  source = {source}")
    print()
    print(f"  {'eps':>8s}  {'max |G(eps) - G(eps_ref)|':>28s}  {'order':>6s}")
    print("  " + "-" * 50)
    prev_err, prev_eps, orders = None, None, []
    for eps in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
        G = mindlin_G(obs, source, mu, nu, eps)
        err = float(np.max(np.abs(G - G_ref)))
        if prev_err is not None and prev_err > 1e-15 and err > 1e-15:
            order = np.log(prev_err / err) / np.log(prev_eps / eps)
            orders.append(order)
        else:
            order = float("nan")
        print(f"  {eps:8.3f}  {err:28.3e}  {order:6.2f}")
        prev_err, prev_eps = err, eps
    print()
    # The gate: second-order convergence in eps on every rung of the ladder.
    ok = bool(orders) and all(o >= 1.8 for o in orders)
    print(f"{'PASS' if ok else 'FAIL'}: eps-convergence order "
          f"min {min(orders) if orders else float('nan'):.2f} (>= 1.8 on every rung)")


# ============================================================
# Main
# ============================================================


def main():
    check_pde_residual()
    check_boundary_traction()
    check_eps_convergence()


if __name__ == "__main__":
    main()
