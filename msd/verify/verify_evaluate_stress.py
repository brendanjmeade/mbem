"""Verify the mbem stress-evaluation path (evaluate_stress / _stress_from_source).

Four checks, each PASS/FAIL:

  1. DD-stress path vs an INDEPENDENT reference (cutde full-space classical
     TDE): off the fault, the mollified slip->stress kernel summed over the
     triangulated fault must converge to the classical dislocation stress as
     eps -> 0 (mollification error ~ eps^2). Guards the kernel choice
     (analytical_stress_kernel), the slip handling, the sign, and the
     per-triangle assembly.
  2. Force-stress path vs the repo's own quadrature reference
     (integrate_kelvin_stress_numerical): the boundary-traction term.
  3. Anelastic subtraction through the production path
     (`_stress_from_source(..., "eigen")`, the exact finite-triangle
     eigenstress that evaluate_stress subtracts): a no-op OFF the fault, but
     ON the fault it removes the divergent part so the corrected stress
     stays bounded as eps -> 0 while the raw total grows ~ 1/eps.  (The
     frozen `anelastic.py` approximation is compared as a deep-interior
     oracle in `verify/verify_eigenstress_exact.py`, not here.)
  4. Public evaluate_stress end to end on a small fault box: runs, returns
     finite symmetric tensors, (elastic - total) equals the eigenstress of
     every double layer at the fault centroids, and the on-fault elastic
     shear over the ladder eps = 4, 2, 1 keeps the SIGN physics dictates
     and stays within a band of its eps = 4 value.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from local_box_mesh_eq import make_vertical_fault_eq                # noqa: E402
from mbem.evaluate import _stress_from_source, evaluate_stress      # noqa: E402
from mbem.model import BCType                                       # noqa: E402
from mollified_kernel.analytical_kernels import (                   # noqa: E402
    integrate_kelvin_stress_numerical,
)
from tde_reference import classical_tde_stress                      # noqa: E402

MU, NU = 30.0, 0.25


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def check_dd_vs_cutde():
    fault, _n, s_hat = make_vertical_fault_eq(
        strike_length=20.0, depth_range=(-10.0, 0.0), target_edge=5.0)
    nt = fault.n_triangles
    slip = np.broadcast_to(np.asarray(s_hat, float), (nt, 3))
    obs = np.array([[3.0, 1.0, -4.0], [-2.5, -3.0, -6.0], [5.0, 0.0, -5.0]])

    ref = classical_tde_stress(obs, fault, s_hat, MU, NU)
    errs = []
    for eps in (0.5, 0.1, 0.02):
        sig = _stress_from_source(obs, fault, slip, "dd", MU, NU,
                                  np.full(nt, eps))
        errs.append((eps, _relmax(sig, ref)))
    for eps, e in errs:
        print(f"    DD vs cutde   eps={eps:6.3f}  rel diff = {e:.3e}")
    decreasing = errs[0][1] > errs[1][1] > errs[2][1]
    return errs[-1][1] < 1e-3 and decreasing


def check_force_vs_quadrature():
    # A generic triangulated patch (reuse the fault mesh as geometry) with a
    # per-triangle force density; analytic assembly vs Gauss quadrature.
    mesh, _, _ = make_vertical_fault_eq(
        strike_length=12.0, depth_range=(-6.0, 0.0), target_edge=4.0)
    verts = np.asarray(mesh.vertices, float)[np.asarray(mesh.triangles)]
    nt = verts.shape[0]
    rng = np.random.default_rng(1)
    dens = rng.normal(size=(nt, 3))
    obs = np.array([[2.0, 0.5, -3.0], [-1.5, -1.0, -2.0]])
    eps = 0.6

    sig = _stress_from_source(obs, mesh, dens, "force", MU, NU,
                              np.full(nt, eps))
    ref = np.zeros_like(sig)
    for i, x in enumerate(obs):
        for t in range(nt):
            A, B, C = verts[t]
            S = integrate_kelvin_stress_numerical(x, A, B, C, MU, NU, eps,
                                                  n_quad=24)
            ref[i] += S @ dens[t]
    err = _relmax(sig, ref)
    print(f"    force vs quadrature  rel diff = {err:.3e}")
    return err < 1e-3


def check_anelastic_subtraction():
    fault, _n, s_hat = make_vertical_fault_eq(
        strike_length=20.0, depth_range=(-10.0, 0.0), target_edge=5.0)
    nt = fault.n_triangles
    slip = np.broadcast_to(np.asarray(s_hat, float), (nt, 3))

    # OFF the fault: subtraction is negligible.
    off = np.array([[4.0, 0.0, -5.0]])
    e = 0.5
    tot_off = _stress_from_source(off, fault, slip, "dd", MU, NU, np.full(nt, e))
    star_off = _stress_from_source(off, fault, slip, "eigen", MU, NU,
                                   np.full(nt, e))
    off_ratio = np.max(np.abs(star_off)) / max(np.max(np.abs(tot_off)), 1e-30)
    print(f"    off-fault eigenstress/total = {off_ratio:.3e} (want << 1)")

    # ON the fault (centroids): raw grows ~1/eps, corrected stays bounded.
    c = fault.centroids()
    raw_pk, cor_pk = [], []
    for eps in (1.0, 0.25):
        tot = _stress_from_source(c, fault, slip, "dd", MU, NU, np.full(nt, eps))
        cor = tot - _stress_from_source(c, fault, slip, "eigen", MU, NU,
                                        np.full(nt, eps))
        raw_pk.append(np.max(np.abs(tot[:, 0, 1])))
        cor_pk.append(np.max(np.abs(cor[:, 0, 1])))
    raw_growth = raw_pk[1] / raw_pk[0]      # eps 1.0 -> 0.25 : raw ~ x4
    cor_growth = cor_pk[1] / cor_pk[0]      # corrected ~ flat
    print(f"    on-fault peak |sxy| raw  : {raw_pk[0]:.3e} -> {raw_pk[1]:.3e} "
          f"(x{raw_growth:.2f})")
    print(f"    on-fault peak |sxy| corr : {cor_pk[0]:.3e} -> {cor_pk[1]:.3e} "
          f"(x{cor_growth:.2f})")
    return (off_ratio < 1e-2 and raw_growth > 2.5 and cor_growth < 1.5
            and cor_pk[1] < raw_pk[1])


def check_evaluate_stress_endtoend():
    import mollified_bem as mb
    from _fault_box import build_fault_box, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system

    slip_mag = 0.01
    mat = mb.ElasticMaterial(mu=30.0, lam=30.0)
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0, fault_half_len=20.0,
                             fault_depth=18.0, edge_fault=3.0, edge_near=24.0,
                             edge_far=50.0, edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, slip_mag, mat)
    system = generate_system(model)
    nsrc = sum(meshes[k].n_triangles for k in ("top", "sides", "base", "fault"))
    print(f"    box: {nsrc} src triangles, {system.layout.n_unknowns} unknowns")
    region = model.regions[0]
    # A few on-fault centroids (full-centroid evaluation is needlessly slow).
    call = meshes["fault"].centroids()
    n_hat = np.asarray(meshes["n_hat"], float)
    s_hat = np.asarray(meshes["s_hat"], float)
    slip = slip_mag * s_hat
    i_c = int(np.argmin(np.linalg.norm(call - [0, 0, -9.0], axis=1)))
    sel = np.argsort(np.linalg.norm(call - [0, 0, -9.0], axis=1))[:6]
    obs = call[sel]

    # Self-consistency at one eps: stress operator applied to evaluate_stress
    # must match FINITE differencing of evaluate_displacement off the fault.
    eps = 3.0
    asm = AssembledDense(system, eps, "direct", jump="calibrated")
    sol = asm.solve()
    sig_tot = evaluate_stress(model, region, sol, obs, eps,
                              subtract_anelastic=False)
    sig_el = evaluate_stress(model, region, sol, obs, eps,
                             subtract_anelastic=True)
    # the EXACT finite-triangle eigenstress -- what evaluate_stress subtracts:
    # the fault's PLUS every boundary patch's
    # u_p, each with the patch's own sigma. Restated here by hand so the
    # check is an independent statement of the wiring, not a call back
    # into it.
    nt_f = meshes["fault"].n_triangles
    star = _stress_from_source(obs, meshes["fault"],
                               np.broadcast_to(slip, (nt_f, 3)), "eigen",
                               mat.mu, mat.nu, np.full(nt_f, eps))
    for p in region.patches:
        u_p = (p.value_array() if p.bc is BCType.PRESCRIBED_DISPLACEMENT
               else sol[f"u:{p.name}"])
        if np.any(u_p):
            star = star + float(model.orientation(region, p)) * \
                _stress_from_source(obs, p.mesh, u_p, "eigen", mat.mu, mat.nu,
                                    np.full(p.mesh.n_triangles, eps))
    finite = np.all(np.isfinite(sig_el))
    symm = float(np.max(np.abs(sig_el - np.transpose(sig_el, (0, 2, 1)))))
    # elastic = total + eigenstress (every double layer enters as
    # -sigma*Sdd@jump), so (elastic - total) must equal +eigenstress.
    sub_ok = _relmax(sig_el - sig_tot, star)
    print(f"    finite={finite}  max asym={symm:.2e}  "
          f"(elastic-total) vs eigenstress rel={sub_ok:.2e}")

    # The decisive test: on-fault ELASTIC shear must neither diverge nor
    # change sign as eps -> 0 (eigenstress removed, not doubled or negated).
    # Sign from physics: a stress drop is negative resolved on the Burgers
    # vector b = u(+n) - u(-n). Here value = u(-n) - u(+n) = +|slip| s_hat,
    # so b = -|slip| s_hat and tau = n.sigma.s must be POSITIVE at every eps
    # (cutde with b = +|slip| s_hat gives the mirror image, -9.5 MPa).
    ctr = call[i_c:i_c + 1]

    def tau_center(eps):
        a = AssembledDense(system, eps, "direct", jump="calibrated")
        sg = evaluate_stress(model, region, a.solve(), ctr, eps,
                             subtract_anelastic=True)
        return float(np.einsum("nij,i,j->n", sg, n_hat, s_hat)[0]) * 1e3
    tau = [tau_center(e) for e in (4.0, 2.0, 1.0)]
    ratio = [abs(t) / max(abs(tau[0]), 1e-30) for t in tau[1:]]
    print(f"    on-fault center elastic shear (MPa): "
          f"{tau[0]:.3f} (eps=4)  {tau[1]:.3f} (eps=2)  {tau[2]:.3f} (eps=1)"
          f"  |tau/tau(4)| = {ratio[0]:.3f}, {ratio[1]:.3f}")
    signed = all(t > 0.0 for t in tau)
    # band basis (measured): tau = 9.79, 9.91, 9.88 MPa; ratios 1.012, 1.009
    banded = all(0.85 < r < 1.1 for r in ratio)
    return (bool(finite) and symm < 1e-9 and sub_ok < 1e-9
            and signed and banded)


def main():
    checks = [
        ("DD stress vs cutde classical TDE (off-fault)", check_dd_vs_cutde),
        ("force-stress vs quadrature", check_force_vs_quadrature),
        ("anelastic subtraction (off vs on fault)", check_anelastic_subtraction),
        ("evaluate_stress end-to-end (fault box)", check_evaluate_stress_endtoend),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: evaluate_stress matches the independent references.")
    else:
        print("\nFAIL: one or more stress-evaluation checks disagree.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
