"""Gate: the structure linear viscoelasticity rests on (docs/viscoelastic.md).

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

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import mollified_bem as mb
from _sphere import icosphere                 # sibling; see tests/conftest.py
from mbem.evaluate import evaluate_stress
from mbem.kernels import basis as kb
from mbem.model import BCType, Patch, Region, RegionModel

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


def main() -> bool:
    t0 = time.time()
    print("=" * 76)
    print("Viscoelastic structure: every operator is {1, g} x {1, mu, 1/mu}")
    print("=" * 76)
    rng = np.random.default_rng(1957)
    a_coefficients()
    b_basis_dependency(rng)
    c_displacement_kernels(rng)
    d_elastic_stress()
    e_one_eps_per_closed_surface()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    print(f"  wall time: {time.time() - t0:.1f} s")
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: viscoelastic structure "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
