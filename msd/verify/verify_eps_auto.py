"""Verify the eps="auto" per-element mollification policy.

eps="auto" resolves each source element's mollification width from its
own size, eps_j = EPS_OVER_H * h_j (h_j = mean edge length), keeping
eps/h fixed under mesh grading and h-refinement. EPS_OVER_H = 0.1 (the
measured basis is in ``mbem/defaults.py``).

Checks (PASS/FAIL):
  1. resolve_eps unit behaviour: scalar -> constant array; explicit
     array passes through; "auto" == EPS_OVER_H * mean edge length;
     unknown strings raise; zero, negative or nan eps raise.
  2. h-REFINEMENT CONVERGENCE ORDER of the KERNEL: with eps="auto" (so
     eps ~ h), the off-fault DD stress of a uniformly slipping planar
     fault must converge to the classical (singular) TDE stress at the
     mollification order eps^2 ~ h^2. Geometry and slip are exact at every
     h, so the ONLY error is mollification -- observed order must be ~2.
  3. WIRING end to end: the fault-box BEM with eps="auto" is bit-identical
     to the same solve with the explicit per-element array
     EPS_OVER_H * h_j; a per-patch dict mixing "auto" and scalars solves
     and evaluates finite on-fault stress.
  4. ACCURACY of a SOLVED BVP with eps="auto" at the backends' DEFAULT
     jump, against the exact Kelvin point-force field on an icosphere
     ladder (80 / 320 / 1280 triangles): Dirichlet interior displacement
     and Neumann surface displacement must fall under refinement at a
     rate >= 0.8 and sit under ceilings set at ~2x the measured values
     (quoted in the code). This is the gate item 2 lacked: before it,
     nothing asserted that "auto" gave an accurate SOLUTION.
  5. GUARD: jump="half" with eps/h above defaults.HALF_JUMP_MAX_EPS_OVER_H
     warns (that combination was measured non-convergent), jump="calibrated"
     does not, and the default jump is "calibrated".

Exits 1 on FAIL.
"""
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem import defaults                                         # noqa: E402
from mbem.backends.dense import AssembledDense, DenseBackend      # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.kernels.tri_kernels import dd_stress_contract           # noqa: E402
from tde_reference import classical_tde_stress                    # noqa: E402

import _sphere as S                                               # noqa: E402

MU, NU = 30.0, 0.25


def check_resolve():
    mesh, _n, _s = make_vertical_fault_eq(
        strike_length=12.0, depth_range=(-6.0, 0.0), target_edge=3.0)
    n = mesh.n_triangles
    ok = True

    r = kb.resolve_eps(2.5, mesh)
    ok &= r.shape == (n,) and np.all(r == 2.5)

    arr = np.linspace(1.0, 2.0, n)
    ok &= np.array_equal(kb.resolve_eps(arr, mesh), arr)

    auto = kb.resolve_eps("auto", mesh)
    tv = np.asarray(mesh.vertices, float)[np.asarray(mesh.triangles)]
    h = np.stack([np.linalg.norm(tv[:, 1] - tv[:, 0], axis=1),
                  np.linalg.norm(tv[:, 2] - tv[:, 1], axis=1),
                  np.linalg.norm(tv[:, 0] - tv[:, 2], axis=1)]).mean(axis=0)
    ok &= np.allclose(auto, defaults.EPS_OVER_H * h, rtol=1e-14)
    print(f"    auto eps range: [{auto.min():.3f}, {auto.max():.3f}] "
          f"(EPS_OVER_H={defaults.EPS_OVER_H})")

    try:
        kb.resolve_eps("magic", mesh)
        ok = False
    except ValueError:
        pass

    # non-positive / non-finite eps must be refused at the choke point
    nan_arr = arr.copy(); nan_arr[n // 2] = np.nan
    zero_arr = arr.copy(); zero_arr[0] = 0.0
    for bad in (0.0, -0.3, nan_arr, zero_arr):
        try:
            kb.as_eps_array(bad, n)
            print(f"    as_eps_array accepted bad eps {np.asarray(bad).min()!r}")
            ok = False
        except ValueError:
            pass
    return ok


def check_h_convergence():
    """eps='auto' + h-refinement: off-fault stress -> classical TDE at
    order ~2 (the mollification order; geometry/slip are exact here)."""
    obs = np.array([[6.0, 2.0, -5.0], [-4.0, -3.0, -7.0], [8.0, 0.0, -3.0]])
    errs, hs = [], []
    for edge in (4.0, 2.0, 1.0):
        fault, _n, s_hat = make_vertical_fault_eq(
            strike_length=20.0, depth_range=(-10.0, 0.0), target_edge=edge)
        nt = fault.n_triangles
        slip = np.broadcast_to(np.asarray(s_hat, float), (nt, 3))
        eps_arr = kb.resolve_eps("auto", fault)
        tri_verts, normals = kb._source_arrays(fault)
        sig = dd_stress_contract(np.ascontiguousarray(obs), tri_verts,
                                 normals, eps_arr,
                                 np.ascontiguousarray(slip), MU, NU)
        ref = classical_tde_stress(obs, fault, s_hat, MU, NU)
        err = float(np.max(np.abs(sig - ref)) / np.max(np.abs(ref)))
        errs.append(err)
        hs.append(float(eps_arr.mean()))
        print(f"    h={edge:4.1f}  mean eps={eps_arr.mean():5.2f}  "
              f"rel err vs classical TDE = {err:.3e}")
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"    observed convergence order: {order:.2f} (expect ~2)")
    return order > 1.5 and errs[0] > errs[1] > errs[2]


def check_end_to_end():
    from _fault_box import build_fault_box, build_model
    from mbem.evaluate import evaluate_stress
    from mbem.model import generate_system

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mb.ElasticMaterial(mu=30.0, lam=30.0))
    system = generate_system(model)
    region = model.regions[0]

    sol_auto = AssembledDense(system, "auto", "direct",
                              jump="calibrated").solve()
    ok = all(np.all(np.isfinite(v)) for v in sol_auto.values())
    print(f"    eps='auto' solve finite: {ok}")

    # WIRING: "auto" must be exactly the explicit per-element array
    explicit = {p.name: defaults.EPS_OVER_H * kb.element_sizes(p.mesh)
                for p in region.patches}
    for f in region.faults:
        explicit[f.name] = defaults.EPS_OVER_H * kb.element_sizes(f.mesh)
    sol_exp = AssembledDense(system, explicit, "direct",
                             jump="calibrated").solve()
    same = all(np.array_equal(sol_auto[k], sol_exp[k]) for k in sol_auto)
    print(f"    'auto' == explicit EPS_OVER_H*h array, bitwise: {same}")
    ok &= same

    # a per-patch dict mixing "auto" and scalars solves, and the on-fault
    # elastic stress evaluates finite with it
    mixed = {"top": "auto", "sides": "auto", "base": "auto", "fault": 3.0}
    sol_mixed = AssembledDense(system, mixed, "direct",
                               jump="calibrated").solve()
    ok &= all(np.all(np.isfinite(v)) for v in sol_mixed.values())
    ctr = meshes["fault"].centroids()[:4]
    sig = evaluate_stress(model, region, sol_mixed, ctr, mixed)
    fin = bool(np.all(np.isfinite(sig)))
    ok &= fin
    print(f"    mixed dict solve + on-fault elastic stress finite: {fin}")
    return ok


def check_kelvin_accuracy():
    """A SOLVED BVP with eps='auto' at the default jump vs the exact Kelvin
    field, on the icosphere ladder."""
    default_jump = DenseBackend().jump
    print(f"    default jump = {default_jump!r}, EPS_OVER_H = "
          f"{defaults.EPS_OVER_H}")
    # measured at EPS_OVER_H = 0.1, calibrated jump, 80/320/1280
    # tri: Dirichlet interior u 2.94e-3 / 1.70e-3 / 9.04e-4 (rate 0.87);
    # Neumann surface u, rigid-stripped, ~9.5e-3 at 1280 tri (eps_sweep.py).
    # Ceilings ~2x the fine-mesh values.
    ceil = {"dirichlet": 2.0e-3, "neumann": 2.0e-2}
    ok = True
    for bc in ("dirichlet", "neumann"):
        errs, hs = [], []
        for level in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, region, sol, es, h = S.sphere_model(
                    level, bc, 0.25, eps="auto", jump=default_jump)
            if bc == "dirichlet":
                P = S.shell(0.5)
                from mbem.evaluate import evaluate_displacement
                u = evaluate_displacement(model, region, sol, P, es,
                                          warn_near=False)
                ue = S.kelvin_u(P, 0.25)
                err = float(np.abs(u - ue).max() / np.abs(ue).max())
                what = "interior u at r = 0.5"
            else:
                c = region.patches[0].mesh.centroids()
                ue = S.kelvin_u(c, 0.25)
                # a Neumann sphere is determined up to a RIGID MOTION; the
                # deflated solve pins the translation, the rotation is free
                du = S.best_fit_rigid(sol["u:sphere"] - ue, c)
                err = float(np.abs(du).max() / np.abs(ue).max())
                what = "surface u (rigid-motion-free)"
            errs.append(err)
            hs.append(h)
            print(f"    {bc:9s} {region.patches[0].n_triangles:5d} tri  "
                  f"h={h:.4f}  eps/h={defaults.EPS_OVER_H}  {what}: {err:.3e}")
        rate = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        good = errs[-1] < ceil[bc] and rate >= 0.8
        print(f"    {bc:9s} rate {rate:.2f} (want >= 0.8), fine error "
              f"{errs[-1]:.3e} (ceiling {ceil[bc]:.1e}) -> "
              f"{'ok' if good else 'XX'}")
        ok &= good
    return ok


def check_half_jump_guard():
    from _fault_box import build_fault_box, build_model
    from mbem.model import generate_system

    ok = DenseBackend().jump == "calibrated"
    print(f"    DenseBackend default jump is 'calibrated': {ok}")

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mb.ElasticMaterial(mu=30.0, lam=30.0))
    system = generate_system(model)
    big = {p.name: 1.25 * kb.element_sizes(p.mesh)
           for r in model.regions for p in list(r.patches) + list(r.faults)}

    def warns(jump, eps):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AssembledDense(system, eps, "direct", jump=jump)
        return any("NON-convergent" in str(x.message) for x in w)

    w_half_big = warns("half", big)
    w_half_auto = warns("half", "auto")
    w_cal_big = warns("calibrated", big)
    print(f"    half + eps/h=1.25 warns: {w_half_big};  half + 'auto' "
          f"(eps/h={defaults.EPS_OVER_H}) warns: {w_half_auto};  "
          f"calibrated + eps/h=1.25 warns: {w_cal_big}")
    return ok and w_half_big and not w_half_auto and not w_cal_big


def main():
    checks = [
        ("resolve_eps unit behaviour", check_resolve),
        ("h-refinement convergence order of the kernel (eps='auto')",
         check_h_convergence),
        ("BEM wiring with eps='auto' / mixed dict", check_end_to_end),
        ("SOLVED BVP accuracy with eps='auto' at the default jump (Kelvin)",
         check_kelvin_accuracy),
        ("jump='half' + large eps/h guard; default jump", check_half_jump_guard),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print(f"\nPASS: eps='auto' policy verified ({len(results)} checks).")
    else:
        print("\nFAIL: eps='auto' policy check failed.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
