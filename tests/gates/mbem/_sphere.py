"""Shared exact-solution harness for the verify scripts: a point force
OUTSIDE the unit sphere, so the sharp Kelvin field is a homogeneous elastic
solution inside, and the sphere BVP (Dirichlet: Kelvin u prescribed;
Neumann: Kelvin traction prescribed) must reproduce it.

Used by ``verify_boundary_eigenstress.py``, ``verify_eps_auto.py`` and
``verify_nodal_solve.py``.
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
                 slip=None, jump=None, eps=None, return_asm=False, order=0,
                 fault_order=0):
    """Kelvin BVP on the icosphere; returns (model, region, sol, eps_spec, h)
    (+ the AssembledDense when ``return_asm``).

    ``eps``: an explicit eps spec for the sphere patch (scalar, array or
    "auto"); by default ``eps_over_h * h``. ``jump``: defaults to "half" for
    Dirichlet and "calibrated" (+ deflate) for Neumann -- the choices the
    near-boundary ladders were measured with. ``order``: the sphere patch's
    Lagrange order; the Kelvin data are callables sampled at its nodes (the
    displacement ``u(x)``, the traction ``sigma(x) . n`` with the element
    normal ``Patch`` hands a two-argument callable); ``value`` overrides
    either datum as ``Patch.value``.
    """
    mesh = icosphere(level)
    h = float(kb.element_sizes(mesh).mean())
    if bc == "dirichlet":
        val = (lambda x: kelvin_u(x, nu)) if value is None else value
        patch = Patch("sphere", mesh, BCType.PRESCRIBED_DISPLACEMENT,
                      value=val, order=order)
        jump = "half" if jump is None else jump
        deflate = False
    else:
        val = ((lambda x, n: np.einsum("nij,nj->ni", kelvin_sigma(x, nu), n))
               if value is None else value)
        patch = Patch("sphere", mesh, BCType.FREE_TRACTION, value=val,
                      order=order)
        jump = "calibrated" if jump is None else jump
        deflate = jump == "calibrated"
    faults = []
    eps_spec = {"sphere": eps_over_h * h if eps is None else eps}
    if fault is not None:
        faults = [Patch("fault", fault, BCType.FAULT, value=slip,
                        order=fault_order)]
        eps_spec["fault"] = eps_over_h * float(kb.element_sizes(fault).mean())
    region = Region("body", material(nu), [patch], np.zeros(3), faults=faults)
    model = RegionModel([region])
    asm = DenseBackend(mode="direct", jump=jump, deflate=deflate).assemble(
        generate_system(model), eps_spec)
    sol = asm.solve()
    if return_asm:
        return model, region, sol, eps_spec, h, asm
    return model, region, sol, eps_spec, h


# ------------------------------------- a layered sphere under buoyancy --
# The exact reference for gravity (``mbem.model.Gravity``, docs/viscoelastic.md
# section 7): concentric elastic shells on a rigid core, loaded on the outer
# surface by a normal traction sigma P_n(cos theta), with RADIAL buoyancy on
# the surface and on every density interface. A finite body, a closed surface,
# an exact solution -- which a flat earth under gravity does not have. Used by
# ``verify_solved_bvp`` (elastic) and ``verify_viscoelastic`` (per s, and on a
# complex contour, so the arithmetic here is generic: floats or mpmath).

def sphere_mesh(level, radius):
    """The icosphere scaled to ``radius``."""
    unit = icosphere(level)
    return mb.TriMesh(
        vertices=np.ascontiguousarray(np.asarray(unit.vertices, float) * radius),
        triangles=np.ascontiguousarray(np.asarray(unit.triangles)))


def _p2(c):
    return 0.5 * (3.0 * c * c - 1.0)


def degree2_load(sigma):
    """``Patch.value`` for the normal load ``sigma P_2(cos theta) r_hat``."""
    def load(x, normals):
        r = np.linalg.norm(x, axis=1)
        return sigma * _p2(x[:, 2] / r)[:, None] * (x / r[:, None])
    return load


def degree2_amplitude(nodes, u) -> float:
    """``U`` of ``u . r_hat = U P_2(cos theta)``, by least squares over the
    nodes of a sphere patch."""
    r = np.linalg.norm(nodes, axis=1)
    p = _p2(nodes[:, 2] / r)
    return float((np.einsum("ni,ni->n", u, nodes / r[:, None]) * p).sum()
                 / (p * p).sum())


def spheroidal(n, mu, lam, r):
    """The four degree-n spheroidal solutions of the elastic equations at
    radius ``r``, as rows ``[U, V, T_r, T_theta]`` by columns (regular A, B;
    irregular A, B), for ``u = U Y r_hat + V grad_1 Y`` and the traction on
    the sphere ``T_r Y r_hat + T_theta grad_1 Y``:

        A:  u = grad(omega),                      omega = r^m Y,  m = n, -n-1
        B:  u = r^2 grad(omega) + alpha x omega,
            alpha = -2 (m lam + (3m+1) mu) / ((m+3) lam + (m+5) mu)

        T_r = (lam + 2 mu) U' + lam (2 U - n(n+1) V) / r
        T_theta = mu (V' - V / r + U / r)

    (Love's solution; each field checked symbolically against the Navier
    operator when this was written, and pinned in the gate by Kelvin's
    stiffness of the incompressible sphere, which uses all four formulas.)
    """
    rows = [[], [], [], []]
    for m in (n, -n - 1):
        al = -2 * (m * lam + (3 * m + 1) * mu) / ((m + 3) * lam + (m + 5) * mu)
        for U, dU, V, dV in (
                (m * r ** (m - 1), m * (m - 1) * r ** (m - 2),
                 r ** (m - 1), (m - 1) * r ** (m - 2)),
                ((m + al) * r ** (m + 1), (m + al) * (m + 1) * r ** m,
                 r ** (m + 1), (m + 1) * r ** m)):
            for row, v in zip(rows, (
                    U, V,
                    (lam + 2 * mu) * dU + lam * (2 * U - n * (n + 1) * V) / r,
                    mu * (dV - V / r + U / r))):
                row.append(v)
    return rows


def _solve(M, rhs):
    if all(isinstance(v, (int, float, np.floating)) for row in M for v in row):
        return list(np.linalg.solve(np.array(M, float), np.array(rhs, float)))
    import mpmath as mp
    return list(mp.lu_solve(mp.matrix(M), mp.matrix(rhs)))


def solid_sphere_stiffness(n, mu, lam, a):
    """``sigma / U(a)`` of a SOLID sphere under the normal load
    ``sigma Y_n`` with no shear traction: the two regular solutions."""
    U, V, Tr, Tt = (row[:2] for row in spheroidal(n, mu, lam, a))
    c = _solve([Tt, Tr], [0.0, 1.0])
    return 1.0 / (U[0] * c[0] + U[1] * c[1])


def _shell_coefficients(n, layers, radii, weights, sigma):
    """The 4 N coefficients of ``shells_on_core`` (its docstring)."""
    N = len(layers)
    M, rhs = [], []

    def row(k, vals, sign=1):
        """An equation row carrying ``vals`` in layer k's four columns."""
        out = [0.0] * (4 * N)
        out[4 * k:4 * k + 4] = [sign * v for v in vals]
        return out

    def add(a, b=None):
        M.append(a if b is None else [x + y for x, y in zip(a, b)])
        rhs.append(0.0)

    U, V, Tr, Tt = spheroidal(n, *layers[0], radii[0])
    add(row(0, U))
    add(row(0, V))
    for k in range(N - 1):
        lo = spheroidal(n, *layers[k], radii[k + 1])
        up = spheroidal(n, *layers[k + 1], radii[k + 1])
        add(row(k, lo[0]), row(k + 1, up[0], -1))
        add(row(k, lo[1]), row(k + 1, up[1], -1))
        add(row(k, lo[3]), row(k + 1, up[3], -1))
        add(row(k, [t + weights[k] * u for t, u in zip(lo[2], lo[0])]),
            row(k + 1, [t + weights[k + 1] * u for t, u in zip(up[2], up[0])],
                -1))
    U, V, Tr, Tt = spheroidal(n, *layers[-1], radii[-1])
    add(row(N - 1, Tt))
    add(row(N - 1, [t + weights[-1] * u for t, u in zip(Tr, U)]))
    rhs[-1] = sigma
    return _solve(M, rhs)


def shells_on_core(n, layers, radii, weights, sigma):
    """Radial surface/interface displacement amplitudes ``[U(r_1), ...,
    U(r_N)]`` of N welded shells on a rigid core of radius ``r_0``.

    ``layers[k] = (mu, lam)`` and ``weights[k] = rho g`` of the shell between
    ``radii[k]`` and ``radii[k + 1]``. The conditions are those of
    ``Gravity``: the MATERIAL traction is continuous, and it is the traction
    of the elastic field plus ``rho g U`` along the normal, so

        core        U = V = 0
        interface   U, V, T_theta continuous;  [T_r + rho g U] continuous
        surface     T_theta = 0;  T_r + rho g U = sigma
    """
    c = _shell_coefficients(n, layers, radii, weights, sigma)
    out = []
    for k in range(len(layers)):
        Uk = spheroidal(n, *layers[k], radii[k + 1])[0]
        out.append(sum(u * x for u, x in zip(Uk, c[4 * k:4 * k + 4])))
    return out


def shell_displacement(layers, radii, weights, sigma, k, x):
    """The exact degree-2 displacement (N, 3) at points ``x`` inside shell
    ``k`` of ``shells_on_core``: ``U(r) P_2 r_hat + V(r) dP_2/dtheta
    theta_hat``."""
    c = _shell_coefficients(2, layers, radii, weights, sigma)[4 * k:4 * k + 4]
    x = np.asarray(x, float)
    r = np.linalg.norm(x, axis=1)
    ct = x[:, 2] / r
    st = np.sqrt(1.0 - ct * ct)
    ph = np.arctan2(x[:, 1], x[:, 0])
    theta_hat = np.stack([ct * np.cos(ph), ct * np.sin(ph), -st], axis=1)
    out = np.zeros_like(x)
    for i, ri in enumerate(r):
        U, V = (sum(a * b for a, b in zip(row, c))
                for row in spheroidal(2, *layers[k], ri)[:2])
        out[i] = U * _p2(ct[i]) * x[i] / ri - 3.0 * V * ct[i] * st[i] * theta_hat[i]
    return out
