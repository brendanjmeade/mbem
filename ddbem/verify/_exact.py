"""Exact reference solutions for the solver gates -- independent of ddbem.

Only the sharp (un-mollified) Kelvin point-force solution lives here, written
from the textbook formulae, with no clq/msd/ddbem code in the path.  Its stress
is checked against a central finite difference of its own displacement inside
``verify_solver.py``, so a transcription error in either formula shows up as a
failed check rather than a silently wrong "exact" answer.

WHY A POINT FORCE OUTSIDE THE BODY IS THE RIGHT EXACT SOLUTION HERE
-------------------------------------------------------------------
A ddbem model represents the interior field by a displacement discontinuity
over the boundary, so the exact solution it must reproduce has to be a
homogeneous solution of the Lame equation throughout the body.  A point force
placed OUTSIDE gives exactly that, on ANY closed surface and with no series
truncation:

* prescribe ``u_bar = u*`` on the boundary -> the interior field must be ``u*``
  (unique), so the displacement rows are gated directly;
* prescribe ``t_bar = sigma* . n`` -> the interior field must be ``u*`` up to a
  rigid motion (the interior Neumann problem), and ``sigma*`` exactly, so the
  hypersingular rows are gated on stress.

Both data sets are automatically self-equilibrated (no force inside), which is
the solvability condition for the all-Neumann case.  A point force INSIDE would
not work: the field is then not a homogeneous solution in the body and no
boundary-only representation can produce it.
"""
from __future__ import annotations

import numpy as np


def kelvin_displacement(x, x0, force, mu, nu):
    """Sharp Kelvin displacement ``(N, 3)`` of a point force at ``x0``.

        u_i = F_j / (16 pi mu (1 - nu)) * [ (3 - 4 nu) delta_ij / r
                                            + r_i r_j / r^3 ]
    """
    x = np.atleast_2d(np.asarray(x, float))
    r = x - np.asarray(x0, float)[None, :]
    R = np.linalg.norm(r, axis=1)
    F = np.asarray(force, float)
    c = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    iso = (3.0 - 4.0 * nu) / R
    return c * (iso[:, None] * F[None, :]
                + (r @ F)[:, None] * r / (R ** 3)[:, None])


def kelvin_stress(x, x0, force, mu, nu):
    """Sharp Kelvin stress ``(N, 3, 3)`` of a point force at ``x0``.

        sigma_ij = -F_k / (8 pi (1-nu) r^2) [ (1-2nu)(d_ik n_j + d_jk n_i
                                                      - d_ij n_k)
                                              + 3 n_i n_j n_k ]
    """
    x = np.atleast_2d(np.asarray(x, float))
    r = x - np.asarray(x0, float)[None, :]
    R = np.linalg.norm(r, axis=1)
    n = r / R[:, None]
    F = np.asarray(force, float)
    nF = n @ F
    eye = np.eye(3)
    c = -1.0 / (8.0 * np.pi * (1.0 - nu) * R ** 2)
    term = ((1.0 - 2.0 * nu)
            * (np.einsum("i,nj->nij", F, n) + np.einsum("j,ni->nij", F, n)
               - nF[:, None, None] * eye[None])
            + 3.0 * nF[:, None, None] * np.einsum("ni,nj->nij", n, n))
    return c[:, None, None] * term


def kelvin_traction(x, normals, x0, force, mu, nu):
    """``t_i = sigma_ij n_j`` for the sharp Kelvin field, ``(N, 3)``."""
    s = kelvin_stress(x, x0, force, mu, nu)
    nf = np.asarray(normals, float)
    if nf.ndim == 1:
        nf = np.broadcast_to(nf, (s.shape[0], 3))
    return np.einsum("nij,nj->ni", s, nf)


def kelvin_stress_fd(x, x0, force, mu, nu, step=1e-6):
    """Stress from a CENTRAL DIFFERENCE of :func:`kelvin_displacement` + Hooke.

    The independent check on :func:`kelvin_stress`: different code path, same
    field.  ``step`` is absolute; the gate uses points at O(1) distance.
    """
    x = np.atleast_2d(np.asarray(x, float))
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    grad = np.empty((x.shape[0], 3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = step
        up = kelvin_displacement(x + d, x0, force, mu, nu)
        um = kelvin_displacement(x - d, x0, force, mu, nu)
        grad[:, :, j] = (up - um) / (2.0 * step)
    e = 0.5 * (grad + grad.transpose(0, 2, 1))
    tr = np.einsum("nii->n", e)
    return 2.0 * mu * e + lam * tr[:, None, None] * np.eye(3)[None]


def best_fit_rigid(u, x):
    """Remove the best-fit rigid motion ``a + b x x`` from ``u`` (least squares).

    The interior Neumann problem fixes the displacement only up to a rigid
    motion, so a displacement comparison there is meaningless without this.
    Stress comparisons do not need it.
    """
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    n = x.shape[0]
    M = np.zeros((3 * n, 6))
    for k in range(3):
        M[k::3, k] = 1.0
        e = np.zeros(3)
        e[k] = 1.0
        M[:, 3 + k] = np.cross(e[None, :], x).reshape(-1)
    c, *_ = np.linalg.lstsq(M, u.reshape(-1), rcond=None)
    return (u.reshape(-1) - M @ c).reshape(-1, 3)
