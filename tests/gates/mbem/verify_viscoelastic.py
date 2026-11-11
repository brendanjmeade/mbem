"""Gate: linear viscoelasticity -- the structure it rests on, and the Laplace
route built on it (docs/viscoelastic.md).

With an elastic bulk modulus and a viscoelastic shear modulus, a region's
operators at Laplace parameter s are its elastic operators at
(mu(s), lam(s)). That is cheap, and has boundary-only memory in the time
domain, because of how the moduli enter the mollified kernels. With
g = 1/(1 - nu):

    slip -> displacement        T = T_A + g T_B              (no mu at all)
    traction -> displacement    U = (1/mu)(U_A + g U_B)
    slip -> ELASTIC stress      S = mu (S_A + g S_B)

and every *_A, *_B is geometry and eps only. None of this was built: it is a
property of kernels written for elasticity, so nothing but this gate keeps a
later kernel change from breaking it quietly. The clauses:

  a  the coefficients in g. ``t_coeffs`` / ``u_coeffs`` against the closed
     forms of the note, at nu != 1/4 and toward the fluid limit, where
     c1..c3 each grow like lam/mu and the combinations that multiply
     INDEPENDENT matrices stay O(1).
  b  the T basis is linearly dependent, N2 = N1 + 3 N3, at every order. This
     is the cancellation of (1 - 2 nu) between lam and the divergence of the
     kernel, mollified terms included; it is why (a) is finite.
  c  the assembled T is affine in g and independent of mu; mu U is affine in
     g. P0, P1 and P2.
  d  the ELASTIC stress over mu is affine in g, for slip with an opening
     component, at points inside the eps band, P0/P1/P2 -- and the RAW TOTAL
     stress is NOT (it carries lam tr(eps*)), which is the tripwire: rule 4,
     the eigenstress removal, is what makes the structure hold.
  e  ONE eps on a closed surface gives two exact identities on the faceted
     mesh: the rigid-translation row sum of T has no material in it (so the
     calibrated diagonal is the same at every s), and at g = 2 the single
     layer of the normal vanishes (a uniform pressure moves nothing in an
     incompressible body). Per-element eps breaks both at 1e-4 .. 1e-3, and
     the second defect is amplified by K/mu(s) in a relaxing region -- so a
     region that relaxes carries one scalar eps. The defect is checked to be
     PRESENT, because a rule nothing violates is not being tested.

Then what is built on it, the Laplace route (``mbem.rheology``,
``mbem.laplace``). Each solved clause is checked twice: PER s against a closed
form evaluated at the moduli of s, which is an elastic statement and separates
the spatial error; and IN TIME against the same closed form inverted on a
complex contour at 30 digits (``mpmath``, Talbot), which shares nothing with
the real-axis fit under test.

  f  the rheologies: s = inf is exactly the unrelaxed solid, the bulk modulus
     does not move, Maxwell's mu(s) and g(s) are their closed forms (g has ONE
     pole, at tau (3K + 4mu)/(3K)), a Burgers body sits on its transient
     plateau mu kelvin/(1 + kelvin) between its two times.
  g  the inversion, on transforms that are known exactly: poles OFF the
     collocation grid, value and rate, a creeping response (which needs the
     term linear in t, and is visibly wrong without it), and the closed-form
     sum over an infinite train of repeating steps.
  h  a dislocation in a relaxing FULL SPACE, Maxwell and Burgers: the fault
     inside a viscoelastic region. Kernel parity against the clq oracle AT
     nu(s), which reaches 0.49 where no other gate goes; then u(t) and the
     elastic stress sigma(t), which relaxes to nothing while u moves to its
     incompressible value.
  i  the Eshelby sphere of verify_solved_bvp B9 with the HOST relaxing: a
     solved two-region model whose relaxing region carries outer faces. The
     hydrostatic and the deviatoric concentration both move (two relaxation
     times), so lam is live; one scalar eps, per clause e.
  j  torsion of a two-layer cylinder, the lower layer relaxing: the only
     closed form whose relaxing region has a plate-over-substrate's boundary
     set (an interface, a free wall, a clamped base).
  k  a penny crack under a sustained stress drop in a Maxwell full space
     (FAULT_TRACTION): the slip creeps for ever, at 8 a dtau / (3 pi eta).
  l  the closed forms the plate-over-substrate study is compared with
     (``cases.layered_box``): the image series has its two limits (a
     half-space; a plate over a fluid, where every point ends at D / 2H of
     the slip), and the Maxwell time series IS the image series at mu(s),
     inverted -- one pole n-fold, equal and unequal moduli. A reference that
     is wrong is worse than none, and these two share no step.

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import sys
import time
import types

import mpmath as mp
import numpy as np

import clq
import inclusion_mesh
import mollified_bem as mb
import verify_solved_bvp as VB                # sibling: the B9 sphere fixture
from _sphere import icosphere                 # sibling; see tests/conftest.py
from mbem import laplace, rheology
from mbem.backends.dense import AssembledDense
from mbem.cases import layered_box, penny
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.kernels import basis as kb
from mbem.model import BCType, Patch, Region, RegionModel, generate_system
from mbem.volume import _strain

CHECKS: list = []

# nu = 1/4 hides a lam/mu exchange (rule 7), so it is not in the ladder. Three
# values are the least that can test "affine": two points always fit a line.
NUS = (0.10, 0.31, 0.47)
MU = 30.0

# Measured, then pinned with headroom. The identities are exact; what is
# measured is rounding, and P2's closed forms carry about a hundred times more
# of it than P0's (``verify_nodal_kernels`` pins the same kernels against clq
# at 1e-12), so the pins are per order rather than one loose number for all.
TOL_COEFF = 1.0e-13      # closed forms vs the code's coefficients; 5e-16
# N2 - N1 - 3 N3, relative, by order; measured 7e-16, 2e-15, 2e-13
TOL_BASIS = {0: 1.0e-13, 1: 1.0e-13, 2: 1.0e-11}
# slope from (nu0, nu1) vs slope from (nu0, nu2); measured 8e-15, 2e-14, 2e-12
TOL_AFFINE = {0: 1.0e-12, 1: 1.0e-12, 2: 1.0e-10}
TOL_STRESS = 1.0e-10     # the same for the stress readout; 3e-13 at every order
MISS_TOTAL = 1.0         # the raw total must FAIL the affinity by at least this
TOL_ONE_EPS = 1.0e-12    # the two closed-surface identities; 4e-15, 1.5e-15
MISS_PER_ELEMENT = 1.0e-5   # ... and what per-element eps must break them by


# ---- the Laplace route. Times are in units of the Maxwell time.
LAM = 45.0                                   # with MU = 30: nu = 0.30
MAXWELL = rheology.Maxwell(tau=1.0)
BURGERS = rheology.Burgers(tau_M=5.0, tau_K=0.4, kelvin=1.5)
WINDOW = (0.1, 10.0)                         # the times the solved clauses read

TOL_RHEOLOGY = 1.0e-13   # closed forms of mu(s), g(s), K; 2e-16
# [g] at the default 3 points per decade, poles off the grid; measured
# 3.4e-4 (value), 4.8e-3 (rate), 6.2e-5 / 2.4e-4 with the secular term,
# 1.8e-3 / 4.5e-4 for the periodic sum and its rate.
TOL_INV_VALUE = 2.0e-3
TOL_INV_RATE = 2.0e-2
TOL_INV_CYCLE = 5.0e-3
MISS_NO_SECULAR = 1.0e-2     # a creeping response fitted without c t; 6.8e-2
# [h] parity with clq at nu(s): 6e-14 (u), 8e-12 (sigma) measured, the stress
# losing digits to lam = 2 mu nu/(1 - 2 nu) inside the ORACLE as nu -> 1/2.
TOL_PARITY_U = 1.0e-11
TOL_PARITY_S = 1.0e-9
TOL_TIME_FULL_SPACE = 5.0e-3     # u(t), sigma(t) vs exact; 1e-4, 9e-4 measured
# The solved clauses. Each has a SPATIAL bias that is the elastic gate's own
# (P0 boundary densities), so what is pinned is: the bias per s, that it does
# not move with s (the correspondence adds nothing), and that the time-domain
# answer carries the same bias and no more (the inversion adds nothing).
# [i] the B9 sphere, host relaxing. Measured per s: hydrostatic 1.6e-2,
# deviatoric 7.8e-3; in time 1.5e-2 and 7.2e-3.
TOL_SPHERE_HYD, TOL_SPHERE_DEV = 3.0e-2, 2.0e-2
# [j] the torsion cylinder, 847 P0 triangles: the interface twist is 5.1-5.4 %
# low at every s, and the same in time.
TORSION_BIAS = (-0.075, -0.035)
TORSION_FLAT = 6.0e-3            # spread of that bias over s; 3e-3 measured
# [k] the penny crack at a/h = 8, continuous P2, eps "auto": moment over the
# exact one 0.9944-0.9954 at every s and in time; fitted creep rate 0.995.
PENNY_BAND = (0.990, 1.000)
PENNY_FLAT = 3.0e-3              # spread over s; 1e-3 measured
TOL_PLATE_FORMS = 1.0e-10        # series vs contour inversion; 2e-13 measured


def check(name, ok, extra="") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:62s} {extra}")
    return ok


def close(name, got, tol) -> bool:
    return check(name, got <= tol, f"{got:.2e} <= {tol:.0e}")


def above(name, got, floor) -> bool:
    return check(name, got >= floor, f"{got:.2e} >= {floor:.0e}")


def lam_of(mu, nu):
    """lam from (mu, nu): a gate's own data. Rule 8 forbids a 1/(1 - 2nu)
    INSIDE the kernels, not here."""
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


def g_of(nu):
    return 1.0 / (1.0 - nu)


def affinity(vals) -> float:
    """How far three samples at NUS are from one line in g: the slope from
    the first pair against the slope from the first and last, relative."""
    g = [g_of(nu) for nu in NUS]
    s1 = (vals[1] - vals[0]) / (g[1] - g[0])
    s2 = (vals[2] - vals[0]) / (g[2] - g[0])
    return float(np.abs(s1 - s2).max() / np.abs(s1).max())


def step_response(transform, t) -> np.ndarray:
    """The response at times ``t`` to a unit step, given ``s -> s u^(s)``.

    Inverted on a Talbot contour in the complex plane at 30 digits. Nothing
    here is shared with ``mbem.laplace``, which samples the REAL axis and fits
    exponentials: the two agree only if both are right.
    """
    with mp.workdps(30):
        return np.array([float(mp.invertlaplace(lambda s: transform(s) / s, tt,
                                                method="talbot"))
                         for tt in np.atleast_1d(t)])


def relaxed(rheo, mat, s):
    """``mat`` at Laplace parameter ``s`` under ``rheo``, as an object with
    ``.mu`` / ``.lam`` -- floats for a state, mpmath numbers on the contour."""
    mu, lam = rheology.moduli(rheo, mat.mu, mat.lam, s)
    return types.SimpleNamespace(mu=mu, lam=lam)


def elastic(m):
    return mb.ElasticMaterial(mu=float(m.mu), lam=float(m.lam))


def nu_of(m) -> float:
    return float(m.lam / (2.0 * (m.lam + m.mu)))


def times():
    return np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 13)


def states(rheo, mat):
    """The Laplace states of one relaxing material over WINDOW: the elastic
    one first, then the collocation points, as ``laplace.Prony`` orders them."""
    s = laplace.s_grid(*WINDOW)
    return s, [relaxed(rheo, mat, sv) for sv in (np.inf,) + s]


# ---------------------------------------------------------------- fixtures --

def _triangles(rng):
    """A few arbitrary triangles and a cloud of observers around them, some
    close enough to be inside the eps band."""
    v = rng.normal(size=(4, 3, 3))
    mesh = mb.TriMesh(vertices=np.ascontiguousarray(v.reshape(-1, 3)),
                      triangles=np.arange(12).reshape(4, 3))
    far = rng.normal(size=(40, 3)) * 2.0
    near = mesh.centroids() + 0.05 * rng.normal(size=(4, 3))
    return mesh, np.vstack([far, near])


def _ellipsoid():
    """A closed surface with a genuinely non-uniform mesh: an icosphere
    squashed along two axes, so "0.1 h" really varies from element to
    element."""
    unit = icosphere(3)
    v = np.asarray(unit.vertices, float) * np.array([1.0, 0.6, 1.7])
    return mb.TriMesh(vertices=np.ascontiguousarray(v),
                      triangles=np.ascontiguousarray(np.asarray(unit.triangles)))


# ------------------------------------------------------------------ clauses --

def a_coefficients() -> None:
    print("\n[a] the material coefficients, written in g = 1/(1 - nu)")
    worst_t = worst_u = 0.0
    for nu in NUS + (0.49,):
        for mu in (MU, 2.0):
            g, lam = g_of(nu), lam_of(mu, nu)
            c, u = kb.t_coeffs(mu, lam), kb.u_coeffs(mu, lam)
            # [N1, N3, R1, R2, R3] after N2 = N1 + 3 N3 is used
            reduced = np.array([c[0] + c[1], 3.0 * c[1] + c[2], c[3], c[4], c[5]])
            want_t = np.array([(g - 1.0) / 4.0, 3.0 * (g - 1.0) / 8.0,
                               (4.0 - g) / 16.0, -g / 16.0, 3.0 / 8.0]) / np.pi
            want_u = np.array([4.0 - g, 2.0, g]) / (16.0 * np.pi * mu)
            worst_t = max(worst_t, float(np.abs(reduced - want_t).max()))
            worst_u = max(worst_u, float(np.abs(u - want_u).max()
                                         / np.abs(want_u).max()))
    close("T coefficients == the closed forms in g", worst_t, TOL_COEFF)
    close("U coefficients == (4-g, 2, g)/(16 pi mu)", worst_u, TOL_COEFF)

    # toward a fluid at fixed bulk modulus, K = 50
    K, mu = 50.0, 1.0e-6
    lam = K - 2.0 * mu / 3.0
    c = kb.t_coeffs(mu, lam)
    reduced = np.array([c[0] + c[1], 3.0 * c[1] + c[2]])
    want = np.array([lam / (4.0 * np.pi * (lam + 2.0 * mu)),
                     3.0 * lam / (8.0 * np.pi * (lam + 2.0 * mu))])
    above("near the fluid limit c1..c3 are individually huge",
          float(np.abs(c[:3]).max()), 1.0e6)
    # K/mu = 5e7 digits are lost in the sum, and that is all that is lost
    close("... and what multiplies N1 and N3 is O(1) and right",
          float(np.abs(reduced - want).max() / np.abs(want).max()), 1.0e-8)


def b_basis_dependency(rng) -> None:
    print("\n[b] N2 = N1 + 3 N3: the T basis is linearly dependent")
    mesh, x = _triangles(rng)
    for order in (0, 1, 2):
        for label, eps in (("one eps", 0.3),
                           ("per-element eps", np.array([0.1, 0.3, 0.2, 0.4]))):
            B = kb.assemble_t_basis(x, mesh, eps, order=order).stack
            close(f"P{order} {label}: |N2 - N1 - 3 N3| / |N1|",
                  float(np.abs(B[1] - B[0] - 3.0 * B[2]).max()
                        / np.abs(B[0]).max()), TOL_BASIS[order])


def c_displacement_kernels(rng) -> None:
    print("\n[c] T = T_A + g T_B with no mu;  mu U = U_A + g U_B")
    mesh, x = _triangles(rng)
    for order in (0, 1, 2):
        TB = kb.assemble_t_basis(x, mesh, 0.3, order=order)
        UB = kb.assemble_u_basis(x, mesh, 0.3, order=order)
        T = [TB.combine(mu=MU, lam=lam_of(MU, nu)) for nu in NUS]
        U = [MU * UB.combine(mu=MU, lam=lam_of(MU, nu)) for nu in NUS]
        tol = TOL_AFFINE[order]
        close(f"P{order}: T is affine in g", affinity(T), tol)
        other = TB.combine(mu=3.0, lam=lam_of(3.0, NUS[1]))
        close(f"P{order}: T does not see mu",
              float(np.abs(other - T[1]).max() / np.abs(T[1]).max()), tol)
        close(f"P{order}: mu U is affine in g", affinity(U), tol)


def _slip(x):
    """A smooth slip field with shear AND opening on every triangle."""
    x = np.asarray(x, float)
    return 1.0e-3 * np.column_stack([1.0 + 0.3 * x[:, 0],
                                     0.4 - 0.2 * x[:, 1],
                                     0.7 + 0.5 * x[:, 0] * x[:, 1]])


def d_elastic_stress() -> None:
    print("\n[d] ELASTIC stress = mu (S_A + g S_B); the raw total is not")
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.2], [0.0, 1.0, 0.1]])
    mesh = mb.TriMesh(vertices=v, triangles=np.array([[0, 1, 2], [0, 2, 3]]))
    rng = np.random.default_rng(1971)             # Rizzo & Shippy
    eps = 0.08
    pts = np.vstack([rng.normal(size=(30, 3)) * 1.5 + [0.5, 0.5, 0.0],
                     [[0.4, 0.4, 0.03], [0.7, 0.5, 0.25], [0.3, 0.7, -0.05]]])

    def stress(order, mu, nu, subtract):
        mat = mb.ElasticMaterial(mu=mu, lam=lam_of(mu, nu))
        fault = Patch("f", mesh, BCType.FAULT, value=_slip, order=order)
        model = RegionModel([Region("space", mat, [], np.zeros(3),
                                    faults=[fault])])
        return evaluate_stress(model, "space", {}, pts, eps,
                               subtract_anelastic=subtract, warn_near=False)

    for order in (0, 1, 2):
        el = [stress(order, MU, nu, True) / MU for nu in NUS]
        close(f"P{order}: elastic stress / mu is affine in g",
              affinity(el), TOL_STRESS)
        other = stress(order, 3.0, NUS[1], True) / 3.0
        close(f"P{order}: ... and scales with mu",
              float(np.abs(other - el[1]).max() / np.abs(el[1]).max()),
              TOL_STRESS)
        raw = [stress(order, MU, nu, False) / MU for nu in NUS]
        above(f"P{order}: the RAW TOTAL is not (the eigenstress is why)",
              affinity(raw), MISS_TOTAL)


def e_one_eps_per_closed_surface() -> None:
    print("\n[e] one eps on a closed surface: two exact identities")
    mesh = _ellipsoid()
    normals, _ = mesh.normals_and_areas()
    h = kb.element_sizes(mesh)
    print(f"      ellipsoid, {mesh.n_triangles} triangles, h in "
          f"[{h.min():.3f}, {h.max():.3f}]")
    cen = mesh.centroids()                        # the P0 collocation points
    stokeslet = np.array([2.0, 2.0, 2.0]) / (16.0 * np.pi)   # mu U at g = 2

    def defects(eps):
        TB = kb.assemble_t_basis(cen, mesh, eps)
        UB = kb.assemble_u_basis(cen, mesh, eps)
        n = mesh.n_triangles
        rows = [TB.combine(mu=1.0, lam=lam_of(1.0, nu))
                .reshape(n, 3, n, 3).sum(axis=2) for nu in (0.05, 0.25, 0.45)]
        spread = float(max(np.abs(r - rows[1]).max() for r in rows))
        S = np.tensordot(stokeslet, UB.stack, axes=1)
        pressure = float(np.abs(S @ normals.reshape(-1)).max()
                         / np.abs(S).sum(axis=1).max())
        return spread, pressure, float(np.abs(rows[1]).max())

    spread, pressure, size = defects(0.1 * float(h.mean()))
    check("the rigid-translation row sum is the half-space value there",
          0.4 < size < 0.6, f"{size:.3f}")
    close("ONE eps: that row sum has no material in it", spread, TOL_ONE_EPS)
    close("ONE eps: single layer of the normal at g = 2 vanishes",
          pressure, TOL_ONE_EPS)
    spread, pressure, _ = defects(0.1 * h)
    above("per-element eps: the row sum moves with nu", spread,
          MISS_PER_ELEMENT)
    above("per-element eps: a uniform pressure moves the body", pressure,
          MISS_PER_ELEMENT)


def f_rheology() -> None:
    print("\n[f] the rheologies")
    K = LAM + 2.0 * MU / 3.0
    g_e = rheology.g_of(MU, LAM)
    tau2 = MAXWELL.tau * (3.0 * K + 4.0 * MU) / (3.0 * K)
    check("s = inf is exactly the unrelaxed solid (Maxwell, Burgers)",
          MAXWELL.shear_ratio(np.inf) == 1.0 == BURGERS.shear_ratio(np.inf))
    worst_k = worst_mu = worst_g = 0.0
    for s in (1e-3, 0.3, 7.0, 1e4, 0.5 + 2.0j):
        for rheo in (MAXWELL, BURGERS):
            mu, lam = rheology.moduli(rheo, MU, LAM, s)
            worst_k = max(worst_k, abs(lam + 2.0 * mu / 3.0 - K) / K)
        mu, lam = rheology.moduli(MAXWELL, MU, LAM, s)
        worst_mu = max(worst_mu, abs(mu - MU * s / (s + 1.0 / MAXWELL.tau)) / MU)
        worst_g = max(worst_g, abs(rheology.g_of(mu, lam)
                                   - (2.0 + (g_e - 2.0) * s / (s + 1.0 / tau2))))
    close("the bulk modulus does not move with s", worst_k, TOL_RHEOLOGY)
    close("Maxwell: mu(s) = mu s / (s + 1/tau)", worst_mu, TOL_RHEOLOGY)
    close("Maxwell: g(s) has one pole, at tau (3K + 4 mu)/(3K)", worst_g,
          TOL_RHEOLOGY)
    wide = rheology.Burgers(tau_M=1e7, tau_K=1e-7, kelvin=1.5)
    close("Burgers: the transient plateau is mu kelvin/(1 + kelvin)",
          abs(wide.shear_ratio(1.0) - 1.5 / 2.5), 1.0e-6)
    check("a rheology round-trips through the dict a dump holds",
          rheology.from_spec({"kind": "burgers", "tau_M": 5.0, "tau_K": 0.4,
                              "kelvin": 1.5}) == BURGERS
          and rheology.from_spec({"kind": "maxwell", "tau": 1.0}) == MAXWELL)


def g_inversion() -> None:
    print("\n[g] the inversion, on transforms known exactly")
    t_min, t_max = 0.1, 100.0
    s = laplace.s_grid(t_min, t_max)
    t = np.logspace(np.log10(t_min), np.log10(t_max), 31)
    # relaxation times deliberately NOT reciprocals of the collocation points
    taus = np.array([0.7, 3.3, 21.0])
    amps = np.array([0.8, -0.5, 0.3])
    a0, c = 0.4, 0.05
    decay = np.exp(-t[:, None] / taus)

    def stack(secular):
        def F(sv):
            return (a0 + float((amps * sv / (sv + 1.0 / taus)).sum())
                    + (c / sv if secular else 0.0))
        return np.array([a0 + amps.sum()] + [F(v) for v in s])

    for secular in (False, True):
        pr = laplace.Prony(s, secular=secular)
        u = a0 + (amps * decay).sum(axis=1) + (c * t if secular else 0.0)
        du = (-amps / taus * decay).sum(axis=1) + (c if secular else 0.0)
        tag = "creeping" if secular else "relaxing"
        close(f"{tag}: u(t), poles off the grid ({len(s)} points)",
              float(np.abs(pr.weights(t) @ stack(secular) - u).max()
                    / np.abs(u).max()), TOL_INV_VALUE)
        close(f"{tag}: t du/dt",
              float(np.abs((pr.rate_weights(t) @ stack(secular) - du) * t).max()
                    / np.abs(du * t).max()), TOL_INV_RATE)
        if secular:
            close("creeping: the fitted creep rate is c",
                  abs(float(pr.creep_rate_weights() @ stack(True)) / c - 1.0),
                  TOL_INV_VALUE)
    pr = laplace.Prony(s)
    u = a0 + (amps * decay).sum(axis=1) + c * t
    above("TRIPWIRE: a creeping response fitted without the secular term",
          float(np.abs(pr.weights(t) @ stack(True) - u).max() / np.abs(u).max()),
          MISS_NO_SECULAR)
    for period in (5.0, 50.0):
        tc = np.linspace(0.02, 1.0, 15) * period
        geo = 1.0 / (1.0 - np.exp(-period / taus))
        e = np.exp(-tc[:, None] / taus)
        cyc = (amps * (e - 1.0) * geo).sum(axis=1)
        rate = (-amps / taus * e * geo).sum(axis=1)
        close(f"a step every {period:g}: displacement since the last one",
              float(np.abs(pr.cycle_weights(tc, period) @ stack(False) - cyc).max()
                    / np.abs(cyc).max()), TOL_INV_CYCLE)
        close(f"a step every {period:g}: its rate",
              float(np.abs(pr.cycle_rate_weights(tc, period) @ stack(False)
                           - rate).max() / np.abs(rate).max()), TOL_INV_CYCLE)
    try:
        laplace.Prony(s, secular=True).cycle_weights(tc, 5.0)
        refused = False
    except ValueError:
        refused = True
    check("a creeping response has no periodic state: refused", refused)


def h_full_space() -> None:
    print("\n[h] a dislocation in a relaxing full space: the fault in a "
          "viscoelastic region")
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.9, 0.2]])
    mesh = mb.TriMesh(vertices=tri, triangles=np.array([[0, 1, 2]]))
    rng = np.random.default_rng(1978)            # Savage & Prescott
    obs = np.vstack([rng.normal(size=(24, 3)) * 1.2 + tri.mean(axis=0),
                     tri.mean(axis=0) + [[0.02, 0.01, 0.03],
                                         [0.10, -0.05, -0.02]]])
    eps = 0.1
    base = mb.ElasticMaterial(mu=MU, lam=LAM)
    t = times()
    for order, n_nodes in ((0, 1), (2, 6)):
        slip = 1.0e-3 * (rng.normal(size=(n_nodes, 3)) + [1.0, 0.3, 0.6])
        fault = Patch("f", mesh, BCType.FAULT, value=slip, order=order)
        model = RegionModel([Region("space", base, [], np.zeros(3),
                                    faults=[fault])])
        # u = u_A + g u_B and sigma = mu (S_A + g S_B), the two pieces from
        # the ORACLE at two Poisson ratios and unit mu
        n1, n2 = 0.12, 0.41
        g1, g2 = g_of(n1), g_of(n2)
        u1, u2 = (clq.displacement(obs, tri, slip, 1.0, n, eps, order=order)
                  for n in (n1, n2))
        s1, s2 = (clq.stress(obs, tri, slip, 1.0, n, eps, order=order)
                  for n in (n1, n2))
        uB, sB = (u2 - u1) / (g2 - g1), (s2 - s1) / (g2 - g1)
        uA, sA = u1 - g1 * uB, s1 - g1 * sB
        for rheo in (MAXWELL, BURGERS):
            s, mats = states(rheo, base)
            U, S, par_u, par_s = [], [], 0.0, 0.0
            for m in mats:
                st = model.with_materials({"space": elastic(m)})
                u = evaluate_displacement(st, "space", {}, obs, eps,
                                          warn_near=False)
                sg = evaluate_stress(st, "space", {}, obs, eps, warn_near=False)
                uc = clq.displacement(obs, tri, slip, m.mu, nu_of(m), eps,
                                      order=order)
                sc = clq.stress(obs, tri, slip, m.mu, nu_of(m), eps, order=order)
                par_u = max(par_u, float(np.abs(u - uc).max() / np.abs(uc).max()))
                par_s = max(par_s, float(np.abs(sg - sc).max() / np.abs(sc).max()))
                U.append(u)
                S.append(sg)
            tag = f"P{order} {rheo.kind}"
            close(f"{tag}: u == clq at nu(s), to nu = {nu_of(mats[-1]):.3f}",
                  par_u, TOL_PARITY_U)
            close(f"{tag}: elastic stress == clq at nu(s)", par_s, TOL_PARITY_S)
            W = laplace.Prony(s).weights(t)

            def of(fn):
                return step_response(lambda sv: fn(relaxed(rheo, base, sv)), t)
            g_t = of(lambda m: rheology.g_of(m.mu, m.lam))
            mu_t = of(lambda m: m.mu)
            mug_t = of(lambda m: m.mu * rheology.g_of(m.mu, m.lam))
            u_ex = uA[None] + g_t[:, None, None] * uB[None]
            s_ex = (mu_t[:, None, None, None] * sA[None]
                    + mug_t[:, None, None, None] * sB[None])
            close(f"{tag}: u(t)",
                  float(np.abs(np.tensordot(W, np.array(U), 1) - u_ex).max()
                        / np.abs(u_ex).max()), TOL_TIME_FULL_SPACE)
            close(f"{tag}: sigma(t)",
                  float(np.abs(np.tensordot(W, np.array(S), 1) - s_ex).max()
                        / np.abs(s_ex).max()), TOL_TIME_FULL_SPACE)
            if rheo is MAXWELL:
                above(f"{tag}: the stress really relaxes over the window",
                      float(np.abs(s_ex[0]).max() / np.abs(s_ex[-1]).max()),
                      100.0)


def i_sphere() -> None:
    print("\n[i] the Eshelby sphere, the HOST relaxing (solved, two regions)")
    host, inc = VB.ESH_MAT_HOST, VB.ESH_MAT_INC
    bx = VB._esh_box(VB.ESH_L, VB.ESH_EDGE)
    unit = icosphere(VB.ESH_LEVEL)
    sph = mb.TriMesh(
        vertices=np.ascontiguousarray(VB.ESH_A * np.asarray(unit.vertices, float)),
        triangles=np.ascontiguousarray(np.asarray(unit.triangles)))
    # ONE eps on the relaxing host's closed boundary: cube and sphere alike
    eps = 0.1 * float(kb.element_sizes(sph).mean())
    e0 = VB.ESH_E0
    tr0 = float(np.trace(e0)) / 3.0
    dev0 = e0 - tr0 * np.eye(3)
    iface = Patch("iface", sph, BCType.INTERFACE)
    outer = [Patch(k, bx[k], BCType.PRESCRIBED_DISPLACEMENT,
                   value=VB._exact_u(bx[k].centroids(), e0))
             for k in ("top", "base", "sides")]
    model = RegionModel([
        Region("host", host, outer + [iface],
               probe_point=np.array([0.7, 0.6, 0.5]) * VB.ESH_L),
        Region("inclusion", inc, [iface], probe_point=np.zeros(3))])
    asm = AssembledDense(generate_system(model), eps, "basis", jump="calibrated")
    q = VB._esh_interior()

    def closed(m_host):
        a_h, a_d, _, _ = VB._eshelby_sphere_factors(m_host, inc)
        return a_h, a_d

    s, mats = states(MAXWELL, host)
    got, want = [], []
    for k, m in enumerate(mats):
        state = model.with_materials({"host": elastic(m)})
        a = asm if k == 0 else asm.rebuild_for_materials({"host": elastic(m)})
        sig = evaluate_stress(state, "inclusion", a.solve(), q, eps,
                              warn_near=False)
        e = _strain(sig, inc.mu, inc.lam)
        tr3 = np.trace(e, axis1=1, axis2=2) / 3.0
        dev = np.median(e - tr3[:, None, None] * np.eye(3), axis=0)
        got.append((float(np.median(tr3) / tr0),
                    float(np.sum(dev * dev0) / np.sum(dev0 * dev0))))
        want.append(tuple(float(v) for v in closed(m)))
    got, want = np.array(got), np.array(want)
    t = times()
    W = laplace.Prony(s).weights(t)
    for col, name, tol in ((0, "hydrostatic", TOL_SPHERE_HYD),
                           (1, "deviatoric", TOL_SPHERE_DEV)):
        scale = np.abs(want[:, col]).max()
        close(f"{name} concentration, per s ({len(s) + 1} states)",
              float(np.abs(got[:, col] - want[:, col]).max() / scale), tol)
        ref = step_response(lambda sv: closed(relaxed(MAXWELL, host, sv))[col], t)
        close(f"{name} concentration, in time",
              float(np.abs(W @ got[:, col] - ref).max() / np.abs(ref).max()),
              tol)
        above(f"{name}: it moves over the window (lam is live)",
              float(abs(ref[-1] - ref[0]) / abs(ref[0])), 0.3)


def _cylinder(radius, h_up, h_lo, edge):
    """A closed two-layer cylinder: caps, the interface disc, two walls."""
    ring, _n = inclusion_mesh._circle_boundary((0.0, 0.0), radius, edge)

    def cap(z, up):
        return inclusion_mesh._disk_with_boundary(ring, edge, z, normal_up=up)

    def wall(z_top, z_bot):
        return inclusion_mesh._cylinder_side(
            ring, z_top, z_bot, max(int(round((z_top - z_bot) / edge)), 2),
            (0.0, 0.0))
    return dict(top=cap(h_up, True), interface=cap(0.0, True),
                base=cap(-h_lo, False), wall_upper=wall(h_up, 0.0),
                wall_lower=wall(0.0, -h_lo))


def j_torsion() -> None:
    print("\n[j] torsion of a two-layer cylinder, the lower layer relaxing")
    h_up, h_lo, phi0 = 0.8, 1.2, 1.0e-3
    mat_up = mb.ElasticMaterial(mu=30.0, lam=45.0)
    mat_lo = mb.ElasticMaterial(mu=12.0, lam=20.0)
    m = _cylinder(1.0, h_up, h_lo, 0.25)
    eps = 0.1 * float(np.mean([kb.element_sizes(v).mean() for v in m.values()]))

    def spin(x):
        return phi0 * np.column_stack([-x[:, 1], x[:, 0], np.zeros(len(x))])

    iface = Patch("interface", m["interface"], BCType.INTERFACE)
    model = RegionModel([
        Region("upper", mat_up,
               [Patch("top", m["top"], BCType.PRESCRIBED_DISPLACEMENT, value=spin),
                Patch("wall_upper", m["wall_upper"], BCType.FREE_TRACTION),
                iface], probe_point=np.array([0.1, 0.05, 0.5 * h_up])),
        Region("lower", mat_lo,
               [iface,
                Patch("wall_lower", m["wall_lower"], BCType.FREE_TRACTION),
                Patch("base", m["base"], BCType.PRESCRIBED_DISPLACEMENT)],
               probe_point=np.array([0.1, 0.05, -0.5 * h_lo]))])
    asm = AssembledDense(generate_system(model), eps, "basis", jump="calibrated")
    c = m["interface"].centroids()
    r2 = float((c[:, 0] ** 2 + c[:, 1] ** 2).sum())

    def twist(sol):
        u = sol["u:interface"]
        return float((c[:, 0] * u[:, 1] - c[:, 1] * u[:, 0]).sum() / r2) / phi0

    def closed(m_lo):
        """Two torsional springs in series: the interface takes the share of
        the twist the LOWER layer's compliance is of the whole."""
        return (mat_up.mu / h_up) / (mat_up.mu / h_up + m_lo.mu / h_lo)

    t = times()
    for rheo in (MAXWELL, BURGERS):
        s, mats = states(rheo, mat_lo)
        got = np.array([twist((asm if k == 0 else asm.rebuild_for_materials(
            {"lower": elastic(mm)})).solve()) for k, mm in enumerate(mats)])
        want = np.array([float(closed(mm)) for mm in mats])
        bias = got / want - 1.0
        check(f"{rheo.kind}: interface twist per s, the P0 bias",
              TORSION_BIAS[0] < bias.min() and bias.max() < TORSION_BIAS[1],
              f"{bias.min():+.4f} .. {bias.max():+.4f}")
        close(f"{rheo.kind}: ... and it does not move with s",
              float(np.ptp(bias)), TORSION_FLAT)
        ref = step_response(lambda sv: closed(relaxed(rheo, mat_lo, sv)), t)
        bt = laplace.Prony(s).weights(t) @ got / ref - 1.0
        check(f"{rheo.kind}: interface twist in time, the same bias",
              TORSION_BIAS[0] < bt.min() and bt.max() < TORSION_BIAS[1],
              f"{bt.min():+.4f} .. {bt.max():+.4f}")
        if rheo is MAXWELL:
            above("maxwell: the lower layer ends up taking nearly all the "
                  "twist", float(ref[-1]), 0.9)


def k_penny() -> None:
    print("\n[k] a penny crack under a sustained stress drop: creep")
    mu, lam = MU, 2.0 * MU * 0.30 / (1.0 - 2.0 * 0.30)
    base = mb.ElasticMaterial(mu=mu, lam=lam)
    a, delta0 = 1.0, 1.0e-3
    drop = penny.stress_drop(mu, lam, delta0, a)
    mesh = penny.penny_mesh(a=a, h=a / 8)

    def closed(m):
        """Eshelby's moment for THIS drop at moduli m: plain arithmetic, so it
        is also the transform on the contour."""
        d0 = 8.0 * a * drop / (np.pi * m.mu * (1.0 + rheology.g_of(m.mu, m.lam)))
        return penny.eshelby_moment(d0, a)

    t = times()
    for rheo in (MAXWELL, BURGERS):
        s, mats = states(rheo, base)
        got = []
        for m in mats:
            model = penny.build_model(mesh, elastic(m), drop=drop, order=2)
            sol = AssembledDense(generate_system(model), "auto", "direct",
                                 jump="calibrated").solve()
            got.append(model.regions[0].faults[0].integrate(sol["b:penny"])[0])
        got = np.array(got)
        ratio = got / np.array([float(closed(m)) for m in mats])
        check(f"{rheo.kind}: moment over Eshelby's, per s",
              PENNY_BAND[0] < ratio.min() and ratio.max() < PENNY_BAND[1],
              f"{ratio.min():.4f} .. {ratio.max():.4f}")
        close(f"{rheo.kind}: ... flat in s", float(np.ptp(ratio)), PENNY_FLAT)
        pr = laplace.Prony(s, secular=True)
        ref = step_response(lambda sv: closed(relaxed(rheo, base, sv)), t)
        rt = pr.weights(t) @ got / ref
        check(f"{rheo.kind}: moment in time, the same band",
              PENNY_BAND[0] < rt.min() and rt.max() < PENNY_BAND[1],
              f"{rt.min():.4f} .. {rt.max():.4f}")
        tau = rheo.tau if rheo is MAXWELL else rheo.tau_M
        rate = penny.eshelby_moment(8.0 * a * drop / (3.0 * np.pi * mu * tau), a)
        c = float(pr.creep_rate_weights() @ got) / rate
        check(f"{rheo.kind}: creep rate == 8 a dtau/(3 pi eta), in moment",
              PENNY_BAND[0] < c < PENNY_BAND[1], f"{c:.4f}")
        above(f"{rheo.kind}: the slip really grows over the window",
              float(ref[-1] / ref[0]), 2.0)


def l_plate_closed_forms() -> None:
    print("\n[l] the antiplane closed forms of the plate over a half-space")
    depth, thick, mu = 15.0, 30.0, 30.0
    x = np.array([5.0, 30.0, 90.0, 300.0])
    close("image series, equal moduli == the half-space, atan(D/x)/pi",
          float(np.abs(layered_box.image_series(x, depth, thick, mu, mu)
                       - np.arctan(depth / x) / np.pi).max()), 1.0e-15)
    fluid = layered_box.image_series(x[2:], depth, thick, mu, 1.0e-7 * mu,
                                     rtol=1.0e-10)
    close("image series over a fluid -> D / 2H far from the fault",
          float(np.abs(fluid - depth / (2.0 * thick)).max()), 1.0e-4)
    t = np.array([0.1, 1.0, 5.0, 30.0])
    for mu_sub in (mu, 2.0 * mu):
        series = layered_box.maxwell_series(x, t, depth, thick, mu, mu_sub, 1.0)

        def transform(sv, xx):
            """The image series at mu(s), summed at the contour's precision."""
            gam = (mu - mu_sub * MAXWELL.shear_ratio(sv)) \
                / (mu + mu_sub * MAXWELL.shear_ratio(sv))
            tot, gn = mp.atan(depth / xx), mp.mpf(1)
            for n in range(1, 4000):
                gn = gn * gam
                term = gn * (mp.atan((2 * n * thick + depth) / xx)
                             - mp.atan((2 * n * thick - depth) / xx))
                tot += term
                if abs(term) < mp.mpf(10) ** (-24) * abs(tot):
                    break
            return tot / mp.pi
        ref = np.array([step_response(lambda sv: transform(sv, mp.mpf(xx)), t)
                        for xx in x]).T
        close(f"Maxwell series == the image series at mu(s), inverted "
              f"(substrate {mu_sub / mu:g} x plate)",
              float(np.abs(series - ref).max()), TOL_PLATE_FORMS)
    above("... and it relaxes: the far field grows several-fold",
          float(series[-1, -1] / series[0, -1]), 3.0)


def main() -> bool:
    t0 = time.time()
    print("=" * 76)
    print("Viscoelasticity: {1, g} x {1, mu, 1/mu} times geometry; the Laplace route")
    print("=" * 76)
    rng = np.random.default_rng(1957)
    a_coefficients()
    b_basis_dependency(rng)
    c_displacement_kernels(rng)
    d_elastic_stress()
    e_one_eps_per_closed_surface()
    f_rheology()
    g_inversion()
    h_full_space()
    i_sphere()
    j_torsion()
    k_penny()
    l_plate_closed_forms()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: viscoelasticity, structure and Laplace route "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
