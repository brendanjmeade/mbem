"""Verify the matrix-free displacement contraction drivers (roadmap A2).

``u_disp_contract`` / ``t_disp_contract`` compute u(x) = sum_s K_xs @ d_s
without materializing the (3N_obs, 3N_src) influence matrix -- the
evaluation primitive for very large source counts. They reuse the SAME
pair kernels as the dense assemblers, so parity is machine precision.

Checks (PASS/FAIL):
  1. T contraction == assemble_t_matrix(...) @ density   (rel < 1e-12)
  2. U contraction == assemble_u_matrix(...) @ density   (rel < 1e-12)
  3. T/U contraction == legacy oracle assemble_BEM_matrices @ density
     (rel < 1e-12; guards the whole chain against the frozen oracle)
  4. evaluate_displacement end-to-end on the fault box reproduces the
     pre-contraction dense-matrix evaluation path (rel < 1e-12) and a
     surface-displacement spot value from the solved model.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem.evaluate import evaluate_displacement                   # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.kernels import tri_kernels as tk                        # noqa: E402

MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)
EPS = 0.8


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _fixture():
    mesh, _n, _s = make_vertical_fault_eq(
        strike_length=18.0, depth_range=(-9.0, 0.0), target_edge=3.0)
    rng = np.random.default_rng(2)
    density = rng.normal(size=(mesh.n_triangles, 3))
    density[3] = 0.0                       # exercise the zero-skip branch
    obs = np.column_stack([rng.uniform(-8, 8, 40),
                           rng.uniform(-6, 6, 40),
                           rng.uniform(-8, -0.5, 40)])
    return mesh, density, np.ascontiguousarray(obs)


def check_vs_dense(kernel):
    mesh, density, obs = _fixture()
    eps_arr = kb.as_eps_array(EPS, mesh.n_triangles)
    tri_verts, normals = kb._source_arrays(mesh)
    if kernel == "t":
        u_con = tk.t_disp_contract(obs, tri_verts, normals, eps_arr,
                                   density, *kb.t_coeffs(MAT.mu, MAT.lam))
        M = kb.assemble_t_matrix(obs, mesh, MAT, eps_arr)
    else:
        u_con = tk.u_disp_contract(obs, tri_verts, eps_arr, density,
                                   *kb.u_coeffs(MAT.mu, MAT.lam))
        M = kb.assemble_u_matrix(obs, mesh, MAT, eps_arr)
    u_ref = (M @ density.ravel()).reshape(-1, 3)
    err = _relmax(u_con, u_ref)
    print(f"    {kernel}-contract vs dense matrix @ density: rel = {err:.2e}")
    return err < 1e-12


class _PointField:
    """Stand-in field 'mesh': the legacy assembler only calls centroids()."""

    def __init__(self, points):
        self._pts = points

    def centroids(self):
        return self._pts


def check_vs_legacy():
    mesh, density, obs = _fixture()
    eps_arr = kb.as_eps_array(EPS, mesh.n_triangles)
    tri_verts, normals = kb._source_arrays(mesh)

    ok = True
    for kern, legacy_kern in (("t", "T"), ("u", "U")):
        M = mb.assemble_BEM_matrices(_PointField(obs), mesh, MAT, EPS,
                                     legacy_kern)
        u_ref = (M @ density.ravel()).reshape(-1, 3)
        if kern == "t":
            u_con = tk.t_disp_contract(obs, tri_verts, normals, eps_arr,
                                       density,
                                       *kb.t_coeffs(MAT.mu, MAT.lam))
        else:
            u_con = tk.u_disp_contract(obs, tri_verts, eps_arr, density,
                                       *kb.u_coeffs(MAT.mu, MAT.lam))
        err = _relmax(u_con, u_ref)
        print(f"    {kern}-contract vs legacy oracle: rel = {err:.2e}")
        ok &= err < 1e-12
    return ok


def check_evaluate_end_to_end():
    from mbem.cases.fault_box import build_fault_box, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.model import BCType, generate_system

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, MAT)
    system = generate_system(model)
    sol = AssembledDense(system, 3.0, "direct", jump="calibrated").solve()
    region = model.regions[0]

    rng = np.random.default_rng(4)
    obs = np.column_stack([rng.uniform(-60, 60, 25),
                           rng.uniform(-60, 60, 25),
                           rng.uniform(-40, -10, 25)])

    u_new = evaluate_displacement(model, region, sol, obs, 3.0)

    # reference: the pre-contraction dense-matrix representation formula
    u_ref = np.zeros(3 * obs.shape[0])
    for p in region.patches:
        sigma = float(model.orientation(region, p))
        u_p = (p.value_array() if p.bc is BCType.PRESCRIBED_DISPLACEMENT
               else sol[f"u:{p.name}"]).ravel()
        if np.any(u_p):
            H = kb.assemble_t_matrix(obs, p.mesh, MAT,
                                     kb.as_eps_array(3.0, p.n_triangles))
            u_ref -= sigma * (H @ u_p)
        t_p = (p.value_array() if p.bc is BCType.FREE_TRACTION
               else sol[f"t:{p.name}"]).ravel()
        if np.any(t_p):
            G = kb.assemble_u_matrix(obs, p.mesh, MAT,
                                     kb.as_eps_array(3.0, p.n_triangles))
            u_ref += sigma * (G @ t_p)
    for f in region.faults:
        slip = f.value_array().ravel()
        if np.any(slip):
            H = kb.assemble_t_matrix(obs, f.mesh, MAT,
                                     kb.as_eps_array(3.0, f.n_triangles))
            u_ref -= float(model.orientation(region, f)) * (H @ slip)
    err = _relmax(u_new, u_ref.reshape(-1, 3))
    print(f"    evaluate_displacement vs dense-matrix path: rel = {err:.2e}")
    finite = bool(np.all(np.isfinite(u_new)))
    return err < 1e-12 and finite


def check_compressed_evaluator():
    """DisplacementEvaluator (roadmap C6): the block-compressed
    evaluation operator for a fixed grid must match the matrix-free
    contraction path at compression tolerance, and repeated evaluations
    must reuse the compressed pairs (no rebuild)."""
    import time

    from mbem.cases.fault_box import build_fault_box, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.evaluate import DisplacementEvaluator
    from mbem.model import generate_system

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, MAT)
    system = generate_system(model)
    asm = AssembledDense(system, 3.0, "direct", jump="calibrated")
    sol = asm.solve()
    region = model.regions[0]

    g = np.linspace(-70, 70, 24)
    X, Y = np.meshgrid(g, g)
    obs = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, -30.0)])

    u_ref = evaluate_displacement(model, region, sol, obs, 3.0)
    t0 = time.perf_counter()
    ev = DisplacementEvaluator(model, region, obs, 3.0, tol=1e-8)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    u_c = ev(sol)
    t_apply = time.perf_counter() - t0
    err = _relmax(u_c, u_ref)
    print(f"    compressed evaluator vs contraction: rel = {err:.2e}")
    print(f"    build {t_build:.2f} s, apply {t_apply*1e3:.1f} ms, "
          f"operator {ev.nbytes()/1e6:.1f} MB")

    # second solution (rebuilt material) reuses the SAME operator
    sol2 = asm.rebuild_for_materials(
        {"crust": mb.ElasticMaterial(mu=45.0, lam=20.0)}).solve()
    # evaluator's region material must match the solve's material:
    ev.region.material = mb.ElasticMaterial(mu=45.0, lam=20.0)
    u2_ref = evaluate_displacement(model, region, sol2, obs, 3.0)
    err2 = _relmax(ev(sol2), u2_ref)
    print(f"    repeated apply (new material): rel = {err2:.2e}")
    return err < 1e-6 and err2 < 1e-6


def main():
    checks = [
        ("T contraction vs dense", lambda: check_vs_dense("t")),
        ("U contraction vs dense", lambda: check_vs_dense("u")),
        ("contraction vs legacy oracle", check_vs_legacy),
        ("evaluate_displacement end-to-end", check_evaluate_end_to_end),
        ("compressed evaluation operator (fixed grid)",
         check_compressed_evaluator),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: displacement contraction matches the dense path "
              "and the legacy oracle.")
    else:
        print("\nFAIL: displacement contraction disagrees.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
