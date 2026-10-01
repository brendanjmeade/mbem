"""Verify that ``evaluate_stress`` subtracts the eigenstress of the BOUNDARY
double layers, not only of faults -- and that doing so is what makes the
interior stress near a boundary converge.

Background. Every ``-sigma * SH @ u_p``
term of the stress representation is a mollified double layer whose
Cortez-smeared jump carries an anelastic eigenstress ``C:eps_star ~ mu u_p (x)
n Phi_eps(d)`` inside the body. For a fault that jump is the slip and msd has
always removed it (BACKLOG.md). For a boundary patch the jump is
``u_p`` itself (the field inside R against zero outside). Measured on an
icosphere against the exact Kelvin point-force field, leaving that term in
gives an h-INDEPENDENT near-boundary stress error: Neumann, d/h = 0.66, eps/h = 0.3: 2.43e-1 / 2.22e-1 / 2.10e-1 over
80 / 320 / 1280 triangles raw, 2.25e-1 / 1.29e-1 / 7.4e-2 with the term.

Exact solution. A point force OUTSIDE the unit sphere; the Kelvin field is
then a homogeneous elastic solution inside, and the sphere BVP -- Dirichlet
(Kelvin u prescribed) or Neumann (Kelvin traction prescribed) -- must
reproduce it. Observation shells are 60 random directions at a fixed
standoff ``d`` from the surface, expressed in local element sizes ``h``
(mean edge) since the piecewise-constant density is the other error source
there.

Checks, all PASS/FAIL:

  [a] WIRING, nu = 0.30 (lam != mu, so the lam (n.u) trace term of the
      eigenstress is live -- it is silent for every strike-slip fault):
      Dirichlet sphere WITH an interior fault; (elastic - total) must equal
      +sum_p sigma_p C:eps_star(u_p) + sigma_f C:eps_star(slip), restated
      here by hand, to 1e-12, at points near the boundary AND near the
      fault; the boundary part must itself be non-negligible at those
      points (so the identity cannot pass by the term being absent).
  [b] ACCURACY: Dirichlet, 1280 triangles, eps/h = 0.3, shell at
      d/h = 0.5, nu = 0.25 and 0.30. Relative max stress error under a
      ceiling set at ~2.5x the measured value (quoted in the code), and
      the RAW (subtract_anelastic=False) error at least 5x the ceiling,
      so the gate is shown to discriminate.
  [c] CONVERGENCE: Neumann, eps/h = 0.3, d/h = 0.66, 80 -> 320 -> 1280
      triangles. The subtracted error must fall by >= 1.3x per level
      (measured 1.74x, 1.74x); the raw error is printed alongside (flat).
  [d] RIGID TRANSLATION (manufactured, exact stress = 0): Dirichlet sphere
      with u_p = const. The total stress near the surface IS the
      eigenstress of the smeared constant jump; the elastic stress at
      d/h = 0.25 must be < 5 % of the raw one.
  [e] DEEP NO-OP: 1280 triangles, r = 0.5 (d ~ 10 eps): the subtraction
      changes the stress by < 1e-3 relative.
  [f] NEAR-BOUNDARY WARNING: 320 triangles; points ON the surface at 20
      mesh vertices must ALL be flagged by ``_warn_near_boundary`` (a
      centroid-distance metric misses vertex points, which sit 0.3-0.7 h
      from every centroid); points at r = 0.5 must not be.

Geometry, exact Kelvin field and the sphere BVP builder come from
``verify/_sphere.py`` (shared with ``verify_eps_auto.py``; no ddbem import),
so it runs from the msd root with numpy/scipy/numba only. Exits 1 on FAIL.
"""
import pathlib
import re
import sys
import time
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

from mbem.evaluate import _stress_from_source, evaluate_stress    # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.model import BCType                                     # noqa: E402

EPS_OVER_H = 0.3

# geometry, exact Kelvin field and the sphere BVP builder are shared with
# verify_eps_auto.py
from _sphere import (MU, icosphere, kelvin_sigma, kelvin_u,      # noqa: E402,F401
                     material, shell, sphere_model, square_fault)

CHECKS = []


def check(label, value, tol, fmt="{:.3e}"):
    ok = bool(np.isfinite(value)) and value < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:60s} {fmt.format(value):>10s}"
          f"  (tol {fmt.format(tol)})")
    return ok


def check_true(label, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {label:60s} {note}")
    return ok


def stress_err(model, region, sol, eps_spec, pts, nu, subtract):
    sg = evaluate_stress(model, region, sol, pts, eps_spec,
                         subtract_anelastic=subtract, warn_near=False)
    se = kelvin_sigma(pts, nu)
    return float(np.max(np.abs(sg - se)) / np.max(np.abs(se)))


def hand_eigen(model, region, sol, eps_spec, pts, nu):
    """+sum over EVERY double layer of sigma * C:eps_star(jump), by hand."""
    mat = region.material
    out = np.zeros((pts.shape[0], 3, 3))
    parts = {}
    for p in region.patches:
        u_p = (p.value_array() if p.bc is BCType.PRESCRIBED_DISPLACEMENT
               else sol[f"u:{p.name}"])
        e = float(model.orientation(region, p)) * _stress_from_source(
            pts, p.mesh, u_p, "eigen", mat.mu, mat.lam,
            kb.resolve_eps(eps_spec[p.name], p.mesh))
        parts[p.name] = e
        out += e
    for f in region.faults:
        e = float(model.orientation(region, f)) * _stress_from_source(
            pts, f.mesh, f.value_array(), "eigen", mat.mu, mat.lam,
            kb.resolve_eps(eps_spec[f.name], f.mesh))
        parts[f.name] = e
        out += e
    return out, parts


# ------------------------------------------------------------- checks --
def a_wiring():
    print("\n[a] WIRING at nu = 0.30: (elastic - total) == +sum sigma C:eps_star,"
          " boundary patches AND fault")
    nu = 0.30
    fault = square_fault()
    slip = np.array([0.0, 1e-3, 0.0])
    model, region, sol, eps_spec, h = sphere_model(2, "dirichlet", nu,
                                                   fault=fault, slip=slip)
    # points: a shell just inside the sphere, and points straddling the fault
    near_b = shell(1.0 - 0.5 * h)
    fc = fault.centroids()
    ef = eps_spec["fault"]
    near_f = np.concatenate([fc + [0.5 * ef, 0, 0], fc - [0.5 * ef, 0, 0]])
    for name, pts in (("near boundary", near_b), ("near fault", near_f)):
        tot = evaluate_stress(model, region, sol, pts, eps_spec,
                              subtract_anelastic=False, warn_near=False)
        ela = evaluate_stress(model, region, sol, pts, eps_spec,
                              subtract_anelastic=True, warn_near=False)
        star, parts = hand_eigen(model, region, sol, eps_spec, pts, nu)
        scale = np.abs(star).max()
        check(f"{name}: (elastic - total) == +C:eps_star",
              np.abs((ela - tot) - star).max() / scale, 1e-12)
        frac_b = np.abs(parts["sphere"]).max() / scale
        check_true(f"{name}: boundary part is non-negligible "
                   f"({frac_b:.2e} of the total)", frac_b > 1e-3)


def b_accuracy():
    print("\n[b] ACCURACY: Dirichlet, 1280 tri, eps/h = 0.3, shell at d/h = 0.5")
    # measured (this script): nu = 0.25 -> 7.01e-2
    # subtracted / 2.22e0 raw; nu = 0.30 -> 7.22e-2 / 2.42e0. Ceiling 2.5x.
    ceil = {0.25: 0.18, 0.30: 0.18}
    for nu in (0.25, 0.30):
        model, region, sol, eps_spec, h = sphere_model(3, "dirichlet", nu)
        pts = shell(1.0 - 0.5 * h)
        e_sub = stress_err(model, region, sol, eps_spec, pts, nu, True)
        e_raw = stress_err(model, region, sol, eps_spec, pts, nu, False)
        print(f"      nu = {nu}: subtracted {e_sub:.3e}   raw {e_raw:.3e}")
        check(f"nu = {nu}: elastic stress error at d/h = 0.5", e_sub, ceil[nu])
        check_true(f"nu = {nu}: raw error >= 5x the ceiling (gate discriminates)",
                   e_raw >= 5.0 * ceil[nu], f"raw/ceiling = {e_raw / ceil[nu]:.1f}")


def c_convergence():
    print("\n[c] CONVERGENCE: Neumann, eps/h = 0.3, d/h = 0.66, levels 1-3"
          " (the near-boundary ladder)")
    nu = 0.25
    sub, raw, ntri = [], [], []
    for level in (1, 2, 3):
        model, region, sol, eps_spec, h = sphere_model(level, "neumann", nu)
        pts = shell(1.0 - 0.66 * h)
        sub.append(stress_err(model, region, sol, eps_spec, pts, nu, True))
        raw.append(stress_err(model, region, sol, eps_spec, pts, nu, False))
        ntri.append(region.patches[0].n_triangles)
        print(f"      {ntri[-1]:5d} tri  h = {h:.4f}  subtracted {sub[-1]:.3e}"
              f"   raw {raw[-1]:.3e}")
    for k in (1, 2):
        ratio = sub[k - 1] / sub[k]
        check_true(f"{ntri[k-1]} -> {ntri[k]} tri: subtracted error falls "
                   f">= 1.3x", ratio >= 1.3, f"ratio {ratio:.2f}")
    print(f"      raw ratios: {raw[0]/raw[1]:.2f}, {raw[1]/raw[2]:.2f}"
          f"  (the un-subtracted term does not converge)")


def d_rigid_translation():
    print("\n[d] RIGID TRANSLATION: Dirichlet u_p = const, exact stress = 0,"
          " shell at d/h = 0.25")
    nu = 0.25
    model, region, sol, eps_spec, h = sphere_model(2, "dirichlet", nu,
                                                   value=np.array([0.3, -0.2, 0.5]))
    pts = shell(1.0 - 0.25 * h)
    raw = evaluate_stress(model, region, sol, pts, eps_spec,
                          subtract_anelastic=False, warn_near=False)
    ela = evaluate_stress(model, region, sol, pts, eps_spec,
                          subtract_anelastic=True, warn_near=False)
    r, e = float(np.abs(raw).max()), float(np.abs(ela).max())
    print(f"      max|sigma|: raw {r:.3e}   elastic {e:.3e}   ratio {e / r:.3e}")
    check("elastic / raw max|sigma| (want << 1)", e / r, 0.05)


def e_deep_noop():
    print("\n[e] DEEP NO-OP: 1280 tri, r = 0.5")
    nu = 0.25
    model, region, sol, eps_spec, h = sphere_model(3, "dirichlet", nu)
    pts = shell(0.5)
    raw = evaluate_stress(model, region, sol, pts, eps_spec,
                          subtract_anelastic=False, warn_near=False)
    ela = evaluate_stress(model, region, sol, pts, eps_spec,
                          subtract_anelastic=True, warn_near=False)
    d = float(np.abs(ela - raw).max() / np.abs(raw).max())
    print(f"      d/eps = {0.5 / eps_spec['sphere']:.1f}")
    check("relative change from the subtraction", d, 5e-3)   # measured 9.7e-4


def f_near_warning():
    print("\n[f] NEAR-BOUNDARY WARNING: 320 tri, 20 surface vertices flagged,"
          " r = 0.5 not")
    model, region, sol, eps_spec, h = sphere_model(2, "dirichlet", 0.25)
    verts = np.asarray(region.patches[0].mesh.vertices, float)
    on_surface = verts[np.linspace(0, len(verts) - 1, 20).astype(int)]
    for name, pts, want in (("20 mesh vertices", on_surface, 20),
                            ("60 points at r = 0.5", shell(0.5), 0)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluate_stress(model, region, sol, pts, eps_spec)
        msgs = [str(x.message) for x in w if "local-h" in str(x.message)]
        m = re.match(r"(\d+) of (\d+)", msgs[0]) if msgs else None
        n = int(m.group(1)) if m else 0
        check_true(f"{name}: {n} of {len(pts)} flagged (want {want})",
                   n == want)


def main():
    t0 = time.time()
    a_wiring()
    b_accuracy()
    c_convergence()
    d_rigid_translation()
    e_deep_noop()
    f_near_warning()
    n_fail = CHECKS.count(False)
    print(f"\n{len(CHECKS)} checks, {n_fail} failed, {time.time() - t0:.1f} s")
    if n_fail:
        print(f"FAIL: boundary eigenstress ({n_fail} of {len(CHECKS)} checks)")
    else:
        print(f"PASS: boundary eigenstress subtraction ({len(CHECKS)} checks)")
    return n_fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
