"""Analytic reference: isotropic spherical inhomogeneity in an infinite
isotropic matrix under a uniform remote strain (Eshelby 1957).

The interior strain is UNIFORM and splits into hydrostatic and deviatoric
parts with different concentration factors:

    eps1_kk  = eps0_kk / (1 + alpha (K1/K0 - 1)),   alpha = 3 K0 / (3 K0 + 4 mu0)
    e1_ij    = e0_ij   / (1 + beta  (mu1/mu0 - 1)), beta  = 6 (K0 + 2 mu0)
                                                           / (5 (3 K0 + 4 mu0))

(e = deviatoric part).  Limits: k = 1 -> identity; rigid (k -> inf) -> 0;
void (k -> 0) -> the well-known amplification.
"""
import numpy as np


def moduli(mu, nu):
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    K = lam + 2.0 * mu / 3.0
    return lam, K


def interior_strain(eps0, mu0, nu0, mu1, nu1):
    """Uniform strain inside a spherical inhomogeneity (3,3)."""
    eps0 = np.asarray(eps0, float)
    _, K0 = moduli(mu0, nu0)
    _, K1 = moduli(mu1, nu1)
    alpha = 3.0 * K0 / (3.0 * K0 + 4.0 * mu0)
    beta = 6.0 * (K0 + 2.0 * mu0) / (5.0 * (3.0 * K0 + 4.0 * mu0))
    tr = np.trace(eps0)
    dev0 = eps0 - tr / 3.0 * np.eye(3)
    tr1 = tr / (1.0 + alpha * (K1 / K0 - 1.0))
    dev1 = dev0 / (1.0 + beta * (mu1 / mu0 - 1.0))
    return dev1 + tr1 / 3.0 * np.eye(3)


def eshelby_tensor_sphere(nu):
    """S_ijkl for a sphere (Mura eq. 11.16)."""
    a = (7.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
    b = (5.0 * nu - 1.0) / (15.0 * (1.0 - nu))
    c = (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu))
    S = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    t = 0.0
                    if i == j == k == l:
                        t = a
                    elif i == j and k == l:
                        t = b
                    elif (i == k and j == l) or (i == l and j == k):
                        t = c
                    S[i, j, k, l] = t
    return S


def _mandel_basis():
    """Orthonormal basis of symmetric 3x3 tensors (Mandel)."""
    r2 = np.sqrt(2.0) / 2.0
    B = np.zeros((6, 3, 3))
    B[0, 0, 0] = B[1, 1, 1] = B[2, 2, 2] = 1.0
    B[3, 1, 2] = B[3, 2, 1] = r2
    B[4, 0, 2] = B[4, 2, 0] = r2
    B[5, 0, 1] = B[5, 1, 0] = r2
    return B


def _as6(T, B):
    return np.einsum("kij,ij->k", B, T)


def _from6(v, B):
    return np.einsum("k,kij->ij", v, B)


def interior_strain_via_tensor(eps0, mu0, nu0, mu1, nu1):
    """Same quantity from eps1 = [I + S:C0^-1:(C1-C0)]^-1 : eps0, built in
    Mandel notation (the 9x9 form is singular: it annihilates antisymmetric
    tensors).  An independent route, used to cross-check `interior_strain`."""
    B = _mandel_basis()

    def stiff6(mu, nu):
        lam, _ = moduli(mu, nu)
        one = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return lam * np.outer(one, one) + 2.0 * mu * np.eye(6)

    S4 = eshelby_tensor_sphere(nu0)
    S6 = np.array([[np.einsum("ij,ijkl,kl->", B[i], S4, B[j]) for j in range(6)]
                   for i in range(6)])
    C0, C1 = stiff6(mu0, nu0), stiff6(mu1, nu1)
    A = np.eye(6) + S6 @ np.linalg.solve(C0, C1 - C0)
    return _from6(np.linalg.solve(A, _as6(np.asarray(eps0, float), B)), B)
