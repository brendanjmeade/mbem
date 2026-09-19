"""Shared exact-solution harness for the verify scripts: a point force
OUTSIDE the unit sphere, so the sharp Kelvin field is a homogeneous elastic
solution inside, and the sphere BVP (Dirichlet: Kelvin u prescribed;
Neumann: Kelvin traction prescribed) must reproduce it.

Used by ``verify_boundary_eigenstress.py`` and ``verify_eps_auto.py``.
Self-contained (own icosphere, own Kelvin formulas) so the msd gates do not
depend on ``ddbem``.
"""
from __future__ import annotations

import numpy as np

import mollified_bem as mb
from mbem.backends.dense import DenseBackend
from mbem.kernels import basis as kb
from mbem.model import BCType, Patch, Region, RegionModel, generate_system

MU = 1.0
X0 = np.array([1.7, 0.9, -1.3])        # point force outside the unit sphere
FORCE = np.array([0.3, -0.7, 0.5])
N_SHELL = 60

# ------------------------------------------------------------ geometry --
_ICO_T = (1.0 + 5.0 ** 0.5) / 2.0
_ICO_V = np.array([
    [-1, _ICO_T, 0], [1, _ICO_T, 0], [-1, -_ICO_T, 0], [1, -_ICO_T, 0],
    [0, -1, _ICO_T], [0, 1, _ICO_T], [0, -1, -_ICO_T], [0, 1, -_ICO_T],
    [_ICO_T, 0, -1], [_ICO_T, 0, 1], [-_ICO_T, 0, -1], [-_ICO_T, 0, 1]], float)
_ICO_F = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.intp)


def icosphere(level):
    """Outward-oriented unit icosphere with 20 * 4**level triangles."""
    v = list(_ICO_V / np.linalg.norm(_ICO_V[0]))
    f = _ICO_F.copy()
    for _ in range(level):
        mid = {}
        new = []

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in mid:
                m = 0.5 * (v[a] + v[b])
                v.append(m / np.linalg.norm(m))
                mid[key] = len(v) - 1
            return mid[key]

        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.asarray(new, np.intp)
    return mb.TriMesh(vertices=np.ascontiguousarray(np.asarray(v, float)),
                      triangles=np.ascontiguousarray(f))


def square_fault(half=0.3, n=2):
    """A vertical square fault in the plane x = 0, n_hat = +x, 2 n^2 tris."""
    ys = np.linspace(-half, half, n + 1)
    zs = np.linspace(-half, half, n + 1)
    verts, tris = [], []
    for z in zs:
        for y in ys:
            verts.append([0.0, y, z])
    for k in range(n):
        for j in range(n):
            a = k * (n + 1) + j
            b, c, d = a + 1, a + n + 1, a + n + 2
            tris += [[a, b, d], [a, d, c]]      # ccw seen from +x
    return mb.TriMesh(vertices=np.ascontiguousarray(np.asarray(verts, float)),
                      triangles=np.ascontiguousarray(np.asarray(tris, np.intp)))


def best_fit_rigid(du, x):
    """Remove the least-squares rigid motion (translation + rotation, 6
    parameters) from a displacement-difference field ``du`` at points ``x``.
    A Neumann sphere is determined only up to a rigid motion: the deflated
    calibrated solve pins the translation, and the rotation -- an approximate
    null vector of the mollified operator -- comes out arbitrary (measured
    ~20 % of max|u| on the 1280-triangle sphere), so both must be removed
    before comparing with the exact field."""
    x = np.asarray(x, float)
    A = np.zeros((3 * len(x), 6))
    b = np.asarray(du, float).ravel()
    for i, p in enumerate(x):
        A[3 * i:3 * i + 3, :3] = np.eye(3)
        A[3 * i:3 * i + 3, 3:] = np.array([[0, p[2], -p[1]],
                                           [-p[2], 0, p[0]],
                                           [p[1], -p[0], 0]])
    return (b - A @ np.linalg.lstsq(A, b, rcond=None)[0]).reshape(-1, 3)


def shell(r, seed=1):
    """N_SHELL random directions at radius r."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N_SHELL, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return r * v


# ------------------------------------------------------- exact Kelvin --
def kelvin_u(x, nu):
    r = np.atleast_2d(x) - X0
    R = np.linalg.norm(r, axis=1)
    c = 1.0 / (16.0 * np.pi * MU * (1.0 - nu))
    return c * (((3.0 - 4.0 * nu) / R)[:, None] * FORCE
                + (r @ FORCE)[:, None] * r / (R ** 3)[:, None])


def kelvin_sigma(x, nu):
    r = np.atleast_2d(x) - X0
    R = np.linalg.norm(r, axis=1)
    n = r / R[:, None]
    nF = n @ FORCE
    eye = np.eye(3)
    c = -1.0 / (8.0 * np.pi * (1.0 - nu) * R ** 2)
    term = ((1.0 - 2.0 * nu)
            * (np.einsum("i,nj->nij", FORCE, n) + np.einsum("j,ni->nij", FORCE, n)
               - nF[:, None, None] * eye[None])
            + 3.0 * nF[:, None, None] * np.einsum("ni,nj->nij", n, n))
    return c[:, None, None] * term


def material(nu):
    return mb.ElasticMaterial(mu=MU, lam=2.0 * MU * nu / (1.0 - 2.0 * nu))


# ------------------------------------------------------------- models --
def sphere_model(level, bc, nu, eps_over_h=0.3, value=None, fault=None,
                 slip=None, jump=None, eps=None, return_asm=False):
    """Kelvin BVP on the icosphere; returns (model, region, sol, eps_spec, h)
    (+ the AssembledDense when ``return_asm``).

    ``eps``: an explicit eps spec for the sphere patch (scalar, array or
    "auto"); by default ``eps_over_h * h``. ``jump``: defaults to "half" for
    Dirichlet and "calibrated" (+ deflate) for Neumann -- the choices the
    near-boundary ladders were measured with.
    """
    mesh = icosphere(level)
    h = float(kb.element_sizes(mesh).mean())
    c = mesh.centroids()
    nrm, _ = mesh.normals_and_areas()
    if bc == "dirichlet":
        val = kelvin_u(c, nu) if value is None else np.broadcast_to(value, c.shape)
        patch = Patch("sphere", mesh, BCType.PRESCRIBED_DISPLACEMENT, value=val)
        jump = "half" if jump is None else jump
        deflate = False
    else:
        val = np.einsum("nij,nj->ni", kelvin_sigma(c, nu), nrm)
        patch = Patch("sphere", mesh, BCType.FREE_TRACTION, value=val)
        jump = "calibrated" if jump is None else jump
        deflate = jump == "calibrated"
    faults = []
    eps_spec = {"sphere": eps_over_h * h if eps is None else eps}
    if fault is not None:
        faults = [Patch("fault", fault, BCType.FAULT, value=slip)]
        eps_spec["fault"] = eps_over_h * float(kb.element_sizes(fault).mean())
    region = Region("body", material(nu), [patch], np.zeros(3), faults=faults)
    model = RegionModel([region])
    asm = DenseBackend(mode="direct", jump=jump, deflate=deflate).assemble(
        generate_system(model), eps_spec)
    sol = asm.solve()
    if return_asm:
        return model, region, sol, eps_spec, h, asm
    return model, region, sol, eps_spec, h
