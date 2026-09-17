"""FD-Hooke consistency of the closed-form kernels (the lambda/mu-swap gate).

For p in {0, 1, 2}, nu in {0.25, 0.30}, mu = 1, eps = 0.2 L, generic nodal
slips, at three observers (two off-plane at |z| = 0.3 L, one ON the plane
inside TRI):

  * grad u by Richardson-extrapolated central differences of
    ``clq.displacement`` (h0 = 1e-3 eps and h0/2, (4 D(h/2) - D(h)) / 3),
    strain = sym(grad u), sigma_hooke = lam tr(strain) I + 2 mu strain,
    compared with ``clq.stress(..., subtract_eigenstress=False)`` (the TOTAL
    stress ``C : sym(grad u)`` of the mollified dislocation), 1e-7 relative.
    At nu = 1/4 (lam == mu) a lambda/mu swap in the displacement contraction
    is invisible; at nu = 0.3 it is not -- both must pass.
  * tripwire that the test discriminates total from elastic stress: on the
    plane the FD-Hooke stress differs from ``clq.stress`` (elastic, default)
    by > 1e-2.
  * demonstration (constant slip, Gauss quadrature of the point kernel
    ``clq.pointwise.kelvin_dG``): with the correct traction-operator pairing
    the same FD-Hooke pipeline reproduces clq's total stress (1e-6,
    quadrature-limited); with lam and mu swapped on the first two contraction
    terms the mismatch exceeds 1e-2 at nu = 0.3 while staying below 1e-6 at
    nu = 0.25.
"""
from __future__ import annotations

import numpy as np

from _common import TRI, MU, Report, relmax
import clq
from clq.frame import local_frame
from clq.moments import gauss_triangle
from clq.shape import shape_functions, n_nodes
from clq import pointwise as pw

FD_REL_STEP = 1e-3        # h0 = FD_REL_STEP * eps
EPS_OVER_L = 0.2
TOL_FD = 1e-7
TOL_QUAD = 1e-6
TRIPWIRE = 1e-2
N_GAUSS = 40


def lame_lambda(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def hooke(grad_u, mu, nu):
    """sigma = lam tr(e) I + 2 mu e with e = sym(grad u), grad_u[n, i, m] = du_i/dx_m."""
    e = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
    tr = np.einsum("nii->n", e)
    return lame_lambda(mu, nu) * tr[:, None, None] * np.eye(3)[None] + 2.0 * mu * e


def fd_gradient(disp, obs, h0):
    """Richardson-extrapolated central-difference gradient (N, 3, 3) of the
    vector field ``disp(points (M,3)) -> (M,3)``: grad[n, i, m] = du_i/dx_m."""
    obs = np.asarray(obs, float).reshape(-1, 3)
    N = obs.shape[0]
    eye = np.eye(3)
    pts = []
    for h in (h0, 0.5 * h0):
        for m in range(3):
            for sgn in (+1.0, -1.0):
                pts.append(obs + sgn * h * eye[m])
    pts = np.concatenate(pts, axis=0)                     # (12 N, 3)
    u = np.asarray(disp(pts), float).reshape(2, 3, 2, N, 3)   # (step, m, sign, n, i)
    D = np.empty((2, N, 3, 3))
    for s, h in enumerate((h0, 0.5 * h0)):
        for m in range(3):
            D[s, :, :, m] = (u[s, m, 0] - u[s, m, 1]) / (2.0 * h)
    return (4.0 * D[1] - D[0]) / 3.0


def quad_displacement(obs, tri, slip, order, mu, nu, eps, swapped=False, n_gauss=N_GAUSS):
    """Displacement (M, 3) by Gauss quadrature of the point kernel built from
    ``kelvin_dG``; ``swapped=True`` puts lam on the normal-derivative term and
    mu on the divergence term (msd's pre-fix contraction)."""
    fr = local_frame(tri)
    obs = np.asarray(obs, float).reshape(-1, 3)
    x1, x2, w = gauss_triangle(n_gauss)
    y = (1 - x1 - x2)[:, None] * fr.v[0] + x1[:, None] * fr.v[1] + x2[:, None] * fr.v[2]
    wq = w * 2.0 * fr.area
    Nq = shape_functions(tri, order, y)                    # (Q, K)
    sq = Nq @ slip                                          # (Q, 3) slip at the Gauss points
    lam = lame_lambda(mu, nu)
    n = fr.nhat
    M, Q = obs.shape[0], y.shape[0]
    d = (obs[:, None, :] - y[None, :, :]).reshape(-1, 3)
    DG = pw.kelvin_dG(d, mu, nu, eps).reshape(M, Q, 3, 3, 3)   # [o, q, i, j, k] = dG_ij/dx_k
    c1, c2 = (lam, mu) if swapped else (mu, lam)
    t1 = c1 * np.einsum("k,oqijk->oqij", n, DG)             # n_m dG_ij/dx_m
    t2 = c2 * np.einsum("j,oqikk->oqij", n, DG)             # n_j dG_im/dx_m
    t3 = mu * np.einsum("k,oqikj->oqij", n, DG)             # n_m dG_im/dx_j
    U = -(t1 + t2 + t3)                                     # (M, Q, 3, 3)
    return np.einsum("oqij,q,qj->oi", U, wq, sq)


def main():
    rep = Report("FD-Hooke consistency: C:sym(grad u) vs total stress kernel")
    fr = local_frame(TRI)
    L = fr.L
    eps = EPS_OVER_L * L
    h0 = FD_REL_STEP * eps
    v1, v2, v3 = TRI
    obs = np.array([
        fr.centroid + 0.3 * L * fr.nhat + 0.15 * L * fr.e1 - 0.10 * L * fr.e2,   # off-plane, +side
        fr.centroid - 0.3 * L * fr.nhat - 0.20 * L * fr.e1 + 0.25 * L * fr.e2,   # off-plane, -side
        v1 + 0.3 * (v2 - v1) + 0.3 * (v3 - v1),                                  # on-plane, inside
    ])
    z, _ = fr.to_plane(obs)
    print(f"  L = {L:.4f}, eps = {eps:.4f}, h0 = {h0:.2e}, obs z/L = "
          + ", ".join(f"{zz / L:+.3f}" for zz in z))
    rng = np.random.default_rng(20260904)
    slips = {p: rng.standard_normal((n_nodes(p), 3)) for p in (0, 1, 2)}

    worst = 0.0
    for nu in (0.25, 0.30):
        for p in (0, 1, 2):
            slip = slips[p]
            disp = lambda pts: clq.displacement(pts, TRI, slip, MU, nu, eps)
            grad = fd_gradient(disp, obs, h0)
            sig_fd = hooke(grad, MU, nu)
            sig_tot = clq.stress(obs, TRI, slip, MU, nu, eps, subtract_eigenstress=False)
            for n_obs, label in enumerate(("off-plane +z", "off-plane -z", "on-plane")):
                d = relmax(sig_fd[n_obs], sig_tot[n_obs])
                worst = max(worst, d)
                rep.check(f"nu={nu:.2f} p={p}: FD-Hooke vs total stress, {label}", d, TOL_FD)
            if nu == 0.30:
                sig_el = clq.stress(obs, TRI, slip, MU, nu, eps)
                d_el = relmax(sig_fd[2], sig_el[2])
                rep.check_bool(f"nu={nu:.2f} p={p}: tripwire, FD-Hooke vs ELASTIC on-plane > {TRIPWIRE:.0e}",
                               d_el > TRIPWIRE, f"(rel diff {d_el:.2e})")
    print(f"  worst FD-Hooke vs total stress over all cases: {worst:.3e}")

    # --- pairing demonstration with the point-kernel quadrature route -------
    slip0 = slips[0]
    for nu in (0.25, 0.30):
        sig_tot = clq.stress(obs, TRI, slip0, MU, nu, eps, subtract_eigenstress=False)
        for swapped in (False, True):
            disp = lambda pts: quad_displacement(pts, TRI, slip0, 0, MU, nu, eps, swapped=swapped)
            sig_fd = hooke(fd_gradient(disp, obs, h0), MU, nu)
            d = relmax(sig_fd, sig_tot)
            if not swapped:
                rep.check(f"nu={nu:.2f} p=0: FD-Hooke of kelvin_dG quadrature (correct pairing) vs clq",
                          d, TOL_QUAD)
            elif nu == 0.25:
                rep.check(f"nu={nu:.2f} p=0: lam/mu-swapped pairing is invisible (lam == mu)",
                          d, TOL_QUAD)
            else:
                rep.check_bool(f"nu={nu:.2f} p=0: lam/mu-swapped pairing is caught (> {TRIPWIRE:.0e})",
                               d > TRIPWIRE, f"(rel diff {d:.2e})")
    rep.finish()


if __name__ == "__main__":
    main()
