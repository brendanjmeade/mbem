"""Dense influence-matrix assembly for force (Kelvin single-layer) elements.

One constant force density per flat triangle, collocated at centroids.  The
pair kernels come from msd's numba code (loaded by file path, never edited):
``_kelvin_stress_pair`` for force -> stress and ``u_matrix_direct`` for
force -> displacement.  clq is the oracle these are gated against.

MEMORY: ``traction_matrix`` and ``stress_matrix`` fill a C-contiguous array
indexed [source, field] and return its TRANSPOSE, which is a Fortran-ordered
view of the matrix the solver wants -- ``scipy.linalg.lu_factor(A,
overwrite_a=True)`` then factors in place instead of silently copying.  Do not
call ``np.asfortranarray`` on those; that would undo the point.

``displacement_matrix`` is NOT one of them: it returns msd's
``tk.u_matrix_direct`` output unchanged, which is plain C-ordered.  Its only
caller builds the 222 Dirichlet rows, where order is irrelevant.  If you ever
hand G to an in-place LAPACK call, transpose-fill it first or accept the copy.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
from numba import njit, prange

MSD = str(pathlib.Path(__file__).resolve().parent.parent / "msd")
if MSD not in sys.path:
    sys.path.insert(0, MSD)

from mbem.kernels import tri_kernels as tk          # noqa: E402
from mbem.kernels.basis import u_coeffs             # noqa: E402

_kelvin_stress_pair = tk._kelvin_stress_pair
_precompute_frames = tk._precompute_frames


def lame(mu, nu):
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


@njit(cache=True, parallel=True)
def _traction_T(x_field, n_field, tri_verts, eps_arr, mu, nu, lam):
    """[3*N_s, 3*N_f] traction influence, TRANSPOSED (see module docstring)."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    outT = np.zeros((3 * N_s, 3 * N_f))
    for s in prange(N_s):
        if not OK[s]:
            continue
        M3 = np.zeros((3, 3)); M5 = np.zeros((4, 4)); M7 = np.zeros((5, 5))
        bas = np.empty((3, 3)); T35 = np.empty((3, 3, 3)); S = np.empty((3, 3, 3))
        for f in range(N_f):
            _kelvin_stress_pair(tri_verts[s], EX[s], EY[s], NH[s], x_field[f],
                                eps_arr[s], mu, nu, lam, S, M3, M5, M7, bas, T35)
            n0 = n_field[f, 0]; n1 = n_field[f, 1]; n2 = n_field[f, 2]
            for i in range(3):
                for c in range(3):
                    outT[3 * s + c, 3 * f + i] = (S[i, 0, c] * n0 + S[i, 1, c] * n1
                                                  + S[i, 2, c] * n2)
    return outT


def traction_matrix(x_field, n_field, tri_verts, eps_arr, mu, nu):
    """(3 N_f, 3 N_s) Fortran-ordered: traction at x_field on normals n_field
    per unit constant force density on each source triangle."""
    A = _traction_T(np.ascontiguousarray(x_field, float),
                    np.ascontiguousarray(n_field, float),
                    np.ascontiguousarray(tri_verts, float),
                    np.ascontiguousarray(eps_arr, float),
                    float(mu), float(nu), lame(mu, nu)).T
    assert A.flags.f_contiguous
    return A


@njit(cache=True, parallel=True)
def _stress_T(x_field, tri_verts, eps_arr, mu, nu, lam):
    """[3*N_s, 6*N_f] Voigt stress influence, transposed."""
    N_f = x_field.shape[0]
    N_s = tri_verts.shape[0]
    EX, EY, NH, OK = _precompute_frames(tri_verts)
    outT = np.zeros((3 * N_s, 6 * N_f))
    vi = np.array([0, 1, 2, 0, 0, 1]); vj = np.array([0, 1, 2, 1, 2, 2])
    for s in prange(N_s):
        if not OK[s]:
            continue
        M3 = np.zeros((3, 3)); M5 = np.zeros((4, 4)); M7 = np.zeros((5, 5))
        bas = np.empty((3, 3)); T35 = np.empty((3, 3, 3)); S = np.empty((3, 3, 3))
        for f in range(N_f):
            _kelvin_stress_pair(tri_verts[s], EX[s], EY[s], NH[s], x_field[f],
                                eps_arr[s], mu, nu, lam, S, M3, M5, M7, bas, T35)
            for v in range(6):
                for c in range(3):
                    outT[3 * s + c, 6 * f + v] = S[vi[v], vj[v], c]
    return outT


def stress_matrix(x_field, tri_verts, eps_arr, mu, nu):
    """(6 N_f, 3 N_s) Voigt-row stress influence (xx, yy, zz, xy, xz, yz)."""
    return _stress_T(np.ascontiguousarray(x_field, float),
                     np.ascontiguousarray(tri_verts, float),
                     np.ascontiguousarray(eps_arr, float),
                     float(mu), float(nu), lame(mu, nu)).T


def displacement_matrix(x_field, tri_verts, eps_arr, mu, nu):
    """(3 N_f, 3 N_s) force density -> displacement (msd's U kernel = clq's G)."""
    g = u_coeffs(float(mu), lame(mu, nu))
    return tk.u_matrix_direct(np.ascontiguousarray(x_field, float),
                              np.ascontiguousarray(tri_verts, float),
                              np.ascontiguousarray(eps_arr, float), *g)


def apply_displacement(x_field, tri_verts, eps_arr, mu, nu, q, chunk=4000,
                      verbose=False):
    """u = G q at x_field, evaluated in row chunks so G is never held whole."""
    x_field = np.ascontiguousarray(x_field, float)
    out = np.empty((x_field.shape[0], 3))
    qf = np.ascontiguousarray(q, float).ravel()
    for a in range(0, x_field.shape[0], chunk):
        b = min(a + chunk, x_field.shape[0])
        G = displacement_matrix(x_field[a:b], tri_verts, eps_arr, mu, nu)
        out[a:b] = (G @ qf).reshape(-1, 3)
        if verbose:
            print(f"    u {b}/{x_field.shape[0]}", flush=True)
        del G
    return out


def eps_bias_operator(normals, mu, nu):
    """Per-element 3x3 blocks M with M q = g, the O(eps) on-surface bias
    direction: u_on,eps = u_sharp + (eps/4) g,  g = -q_t/mu - q_n n/(lam+2mu).

    (The Cortez kernel is G*phi_eps, so the layer field is u_sharp * phi_eps;
    u_sharp has a (g/2)|z| kink and the blob's 1-D marginal has <|z|> = eps/2.)
    """
    n = np.asarray(normals, float)
    lam = lame(mu, nu)
    eye = np.eye(3)[None, :, :]
    nn = np.einsum("mi,mj->mij", n, n)
    return -(eye - nn) / mu - nn / (lam + 2.0 * mu)
