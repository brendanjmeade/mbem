"""Verify the eps="auto" per-element mollification policy (roadmap B3).

eps="auto" resolves each source element's mollification width from its
own size, eps_j = EPS_OVER_H * h_j (h_j = mean edge length), keeping
eps/h fixed under mesh grading and h-refinement.

Checks (PASS/FAIL):
  1. resolve_eps unit behaviour: scalar -> constant array; explicit
     array passes through; "auto" == EPS_OVER_H * mean edge length;
     unknown strings raise.
  2. h-REFINEMENT CONVERGENCE ORDER (the acceptance criterion): with
     eps="auto" (so eps ~ h), the off-fault DD stress of a uniformly
     slipping planar fault must converge to the classical (singular)
     TDE stress at the mollification order eps^2 ~ h^2. The geometry is
     exactly triangulated at every h and the slip is constant, so the
     ONLY error is mollification -- observed order must be ~2.
  3. End-to-end: the fault-box BEM solves with eps="auto" globally and
     with a per-patch dict mixing "auto" and scalars; solutions finite
     and close to the scalar-eps reference.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem import defaults                                         # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.kernels.tri_kernels import dd_stress_contract           # noqa: E402
from tde_reference import classical_tde_stress                    # noqa: E402

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
    from mbem.backends.dense import AssembledDense
    from mbem.evaluate import evaluate_stress
    from mbem.model import generate_system

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mb.ElasticMaterial(mu=30.0, lam=30.0))
    system = generate_system(model)

    sol_auto = AssembledDense(system, "auto", "direct",
                              jump="calibrated").solve()
    mixed = {"top": "auto", "sides": "auto", "base": "auto", "fault": 3.0}
    sol_mixed = AssembledDense(system, mixed, "direct",
                               jump="calibrated").solve()
    ok = all(np.all(np.isfinite(v)) for v in sol_auto.values())
    ok &= all(np.all(np.isfinite(v)) for v in sol_mixed.values())
    print(f"    eps='auto' solve finite: {ok}")

    # same physics: 'auto' surface displacement close to a scalar-eps run
    sol_ref = AssembledDense(system, 3.0, "direct", jump="calibrated").solve()
    d = float(np.max(np.abs(sol_auto["u:top"] - sol_ref["u:top"])))
    s = float(np.max(np.abs(sol_ref["u:top"])))
    print(f"    'auto' vs scalar-eps surface displacement: rel {d/s:.2e} "
          f"(discretization-level difference expected)")
    ok &= d / s < 0.15

    # eigenstress guard: on-fault elastic stress with the mixed dict
    # (uniform scalar fault eps) must evaluate finite
    region = model.regions[0]
    ctr = meshes["fault"].centroids()[:4]
    sig = evaluate_stress(model, region, sol_mixed, ctr, mixed)
    ok &= bool(np.all(np.isfinite(sig)))
    print(f"    on-fault elastic stress (mixed dict) finite: "
          f"{bool(np.all(np.isfinite(sig)))}")
    return ok


def main():
    checks = [
        ("resolve_eps unit behaviour", check_resolve),
        ("h-refinement convergence order (eps='auto')", check_h_convergence),
        ("BEM end-to-end with eps='auto' / mixed dict", check_end_to_end),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: eps='auto' policy verified (order-2 h-convergence).")
    else:
        print("\nFAIL: eps='auto' policy check failed.")


if __name__ == "__main__":
    main()
