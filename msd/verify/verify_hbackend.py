"""Verify the block-compressed H backend against the dense backend.

The H path (PairCompressed / ACA / BlockGaussSeidel / FGMRES) previously
had no PASS/FAIL gate -- only a demo. This is the gate that every change
to mbem/la/* or mbem/backends/hmat.py must keep green.

Three checks, each PASS/FAIL:

  1. Pair-level ACA compression on a GUARANTEED-admissible geometry
     (two parallel panels separated by ~2x their size): at least one
     low-rank block must be produced (the ACA path is exercised, not
     just dense leaves) and to_dense must match the exact basis combine
     for both kernels.
  2. End-to-end on the three-region vertical fault-zone model
     (real INTERFACE topology + fault): HBackend operator/RHS/solution
     vs AssembledDense(mode="direct", jump="half"), and the FGMRES
     report must be converged with a true relative residual below rtol.
  3. rebuild_for_materials: rebuilding with a changed zone material must
     reproduce a fresh HBackend assembly at that material (compressed
     geometry is material-independent; ACA is seeded, so this is
     deterministic).

The later checks cover the material sweep (view cache, preconditioner
reuse, warm start, Krylov recycling), parallel determinism, the
calibrated jump, combined storage, the flat view (the batched matvec and
leaf kernel against the block loop they replaced), the block-Jacobi rung,
and the convergence rate: FGMRES iterations on the fault-zone model at two sizes
~2.7x apart may grow by at most ``defaults.GMRES_ITER_GROWTH_MAX`` and
never past ``defaults.GMRES_ITER_CEILING`` (the ladder is size-independent).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh import make_rectangular_patch                 # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem import defaults                                         # noqa: E402
from mbem.backends import HBackend                                # noqa: E402
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.kernels import KERNEL_T, KERNEL_U, kernel_coeffs        # noqa: E402
from mbem.kernels import basis as kb                              # noqa: E402
from mbem.kernels import tri_kernels as tk                        # noqa: E402
from mbem.la.hop import PairCompressed                            # noqa: E402
from mbem.model import generate_system                            # noqa: E402
from mbem.wrappers import build_vertical_fault_zone_model         # noqa: E402

EPS = 3.0
TOL = defaults.BLOCK_COMPRESSION_TOL    # the operator tolerance the H path runs at
# H-vs-dense parity (defaults.H_PARITY_*): operator/RHS entries and
# solutions per slot; rebuild, combined-storage and determinism checks
# stay bitwise.
OP_PARITY = defaults.H_PARITY_OPERATOR * TOL
SOL_PARITY = defaults.H_PARITY_SOLUTION * TOL


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def check_pair_aca():
    """Two parallel 256-tri panels, 2x-size separation: admissible root."""
    field = make_rectangular_patch((-40.0, 40.0), (-40.0, 40.0), 0.0,
                                   16, 8, normal_up=True)
    source = make_rectangular_patch((-40.0, 40.0), (-40.0, 40.0), -160.0,
                                    16, 8, normal_up=True)
    mat = mb.ElasticMaterial(mu=30.0, lam=30.0)
    eps_arr = kb.as_eps_array(EPS, source.n_triangles)

    ok = True
    try:                       # a kernel tag outside KERNELS must not
        PairCompressed(field, source, "T", eps_arr)            # silently
        print("    PairCompressed accepted kernel 'T'")        # be U
        ok = False
    except ValueError:
        pass
    for kernel in (KERNEL_T, KERNEL_U):
        pc = PairCompressed(field, source, kernel, eps_arr)
        coeffs = kernel_coeffs(kernel, mat)
        # exact dense reference: basis stack combined with coefficients
        tri_verts, normals = kb._source_arrays(source)
        xf = np.ascontiguousarray(field.centroids())
        if kernel == KERNEL_T:
            stack = tk.t_basis_matrices(xf, tri_verts, normals, eps_arr)
        else:
            stack = tk.u_basis_matrices(xf, tri_verts, eps_arr)
        exact = np.tensordot(np.asarray(coeffs), stack, axes=1)

        err = _relmax(pc.to_dense(np.asarray(coeffs)), exact)
        nlr = pc.n_lowrank
        print(f"    [{kernel}] {pc.summary()}")
        print(f"    [{kernel}] to_dense vs exact combine: rel = {err:.2e}  "
              f"(low-rank blocks: {nlr})")
        ok &= (err < OP_PARITY) and (nlr >= 1)
    return ok


def _build_zone_model(mat_zone, refine: float = 1.0):
    """The three-region fault-zone box; ``refine`` multiplies every
    element count per direction (unknowns scale ~refine^2)."""
    r = float(refine)
    fault, _n_hat, s_hat = make_vertical_fault_eq(
        strike_length=16.0, depth_range=(-24.0, -6.0), target_edge=4.0 / r)
    mat_outer = mb.ElasticMaterial(mu=30.0, lam=30.0)
    model = build_vertical_fault_zone_model(
        x_range=(-60.0, 60.0), y_range=(-40.0, 40.0), z_bottom=-40.0,
        zone_half_width=12.0, material_outer=mat_outer,
        material_zone=mat_zone, fault_mesh=fault,
        fault_slip_vector=s_hat, slip_magnitude=0.01,
        nx_outer=round(4 * r), nx_zone=round(2 * r), ny=round(8 * r),
        nz=round(4 * r))
    return model


def check_end_to_end():
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0))
    system = generate_system(model)
    n = system.layout.n_unknowns
    print(f"    fault-zone model: {n} unknowns")

    dense = AssembledDense(system, EPS, "direct", jump="half")
    hasm = HBackend(jump="half").assemble(system, EPS)

    op_err = _relmax(hasm.to_dense(), dense.A)
    rhs_err = _relmax(hasm.b, dense.b)
    print(f"    operator parity : rel = {op_err:.2e}")
    print(f"    RHS parity      : rel = {rhs_err:.2e}")

    sol_d = dense.solve()
    sol_h = hasm.solve(rtol=1e-9)
    report = hasm.report
    worst = max(_relmax(sol_h[k], sol_d[k]) for k in sol_d
                if np.max(np.abs(sol_d[k])) > 0)
    print(f"    solution parity : rel = {worst:.2e}")
    print(f"    FGMRES          : converged={report.converged} "
          f"iters={report.iterations} true_relres={report.true_relres:.2e}")
    return (op_err < OP_PARITY and rhs_err < OP_PARITY and worst < SOL_PARITY
            and report.converged and report.true_relres < 1e-9)


def check_rebuild():
    mat_a = mb.ElasticMaterial(mu=10.0, lam=10.0)
    mat_b = mb.ElasticMaterial(mu=20.0, lam=15.0)
    model = _build_zone_model(mat_a)
    system = generate_system(model)

    h1 = HBackend(jump="half").assemble(system, EPS)
    h2 = h1.rebuild_for_materials({"zone": mat_b})
    sol_rebuild, rep_r = h2.solve(rtol=1e-9), h2.report

    model_b = _build_zone_model(mat_b)
    system_b = generate_system(model_b)
    fresh = HBackend(jump="half").assemble(system_b, EPS)
    sol_fresh, rep_f = fresh.solve(rtol=1e-9), fresh.report

    worst = max(_relmax(sol_rebuild[k], sol_fresh[k]) for k in sol_fresh
                if np.max(np.abs(sol_fresh[k])) > 0)
    print(f"    rebuild vs fresh solution: rel = {worst:.2e} "
          f"(iters {rep_r.iterations}/{rep_f.iterations})")
    return worst < 1e-8 and rep_r.converged and rep_f.converged


def _sweep_arm(system, mus, policy, rtol):
    """One material sweep through ONE assembly under one policy.

    The policy name names the ingredients: ``fresh`` rebuilds from an
    assembly that never solved, so every solve builds its own
    preconditioner; otherwise the rebuilds are chained and the ladder is
    kept while the material step allows. ``warm`` starts each solve from
    the last solution, ``recycle`` adds the GCRO-DR subspace. Returns
    (solutions, iterations, ladder build times, true relative residuals).
    """
    base = HBackend(jump="half").assemble(system, EPS)   # never solved
    asm = base
    sols, iters, builds, relres = [], [], [], []
    for mu in mus:
        asm = (base if "fresh" in policy else asm).rebuild_for_materials(
            {"zone": mb.ElasticMaterial(mu=float(mu), lam=float(mu))})
        sols.append(asm.solve(rtol=rtol,
                              x0="previous" if "warm" in policy else None,
                              recycle="recycle" in policy))
        rep = asm.report
        iters.append(rep.iterations)
        builds.append(rep.precond_summary["build_s"])
        relres.append(rep.true_relres)
    return sols, iters, builds, relres


def check_views_bounded():
    """A 50-sample material sweep must not grow the per-pair view cache
    beyond the LRU bound (the pre-fix behaviour cached one recombined
    operator per material forever), and a solve at a revisited material
    must still match its first solve exactly.

    The same sweep is where preconditioner reuse, the warm start and
    Krylov recycling are gated (WP5a/WP6-E2), since all three are
    properties of a SEQUENCE of solves: the ladder must be kept while
    the materials stay within ``defaults.PRECOND_REUSE_MAX_STEP`` of the
    ones it was built at and dropped beyond it; a kept ladder plus a warm
    start must not cost more than ``defaults.PRECOND_REUSE_ITER_SLACK``
    iterations against a fresh build; recycling must leave a single solve
    alone and bring the sweep's solves 2..N under
    ``defaults.GCRO_SWEEP_ITER_MAX`` -- all three to the same solution,
    since a preconditioner and a search space change the path, never the
    answer.
    """
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0))
    system = generate_system(model)
    h = HBackend(jump="half").assemble(system, EPS)
    sol_first = h.solve(rtol=1e-9)

    mus = np.linspace(8.0, 24.0, 50)
    asm = h
    for mu in mus:
        asm = asm.rebuild_for_materials(
            {"zone": mb.ElasticMaterial(mu=float(mu), lam=float(mu))})
        asm.matvec(np.ones(system.layout.n_unknowns))   # touch the views
    n_views = max(len(p._views) for p in h._pairs.values())
    bound = defaults.HOP_VIEW_CACHE_MAX
    print(f"    max cached views/pair after 50-material sweep: {n_views} "
          f"(bound {bound})")

    # revisit the original material through the SHARED pair set
    back = asm.rebuild_for_materials(
        {"zone": mb.ElasticMaterial(mu=10.0, lam=10.0)})
    sol_back = back.solve(rtol=1e-9)
    worst = max(_relmax(sol_back[k], sol_first[k]) for k in sol_first
                if np.max(np.abs(sol_first[k])) > 0)
    print(f"    revisited-material solve vs first: rel = {worst:.2e}")
    ok = n_views <= bound and worst < 1e-10

    # -- one solve: recycling must change nothing but the bookkeeping --
    single = HBackend(jump="half").assemble(system, EPS)
    sol_off = single.solve(rtol=1e-9, recycle=False)
    it_off = single.report.iterations
    sol_on = single.solve(rtol=1e-9, recycle=True)
    rep_on = single.report
    dev = max(_relmax(sol_on[k], sol_off[k]) for k in sol_off
              if np.max(np.abs(sol_off[k])) > 0)
    print(f"    single solve, recycling on vs off: {rep_on.iterations} vs "
          f"{it_off} iters, solution rel = {dev:.2e}, harvested "
          f"{rep_on.recycle_dim} vectors")
    ok &= (rep_on.iterations <= it_off + 1 and dev < defaults.SOLUTION_RTOL
           and rep_on.converged)

    # -- the sweep policies -------------------------------------------
    # 4 materials over a 1.9x span: inside the reuse window throughout,
    # so the ladder built at the first is what solves 2-4 apply.
    mus = 10.0 * np.geomspace(1.0, 1.9, 4)
    arms = {p: _sweep_arm(system, mus, p, 1e-9)
            for p in ("fresh", "reuse+warm", "reuse+warm+recycle")}
    ref = arms["fresh"][0]
    for policy, (sols, iters, builds, relres) in arms.items():
        dev = max(max(_relmax(s[k], r[k]) for k in r
                      if np.max(np.abs(r[k])) > 0)
                  for s, r in zip(sols, ref))
        kept = sum(b == builds[0] for b in builds)
        print(f"    {policy:>18}: iters {iters}, ladder kept "
              f"{kept}/{len(builds)}, max true relres {max(relres):.1e}, "
              f"vs fresh rel = {dev:.2e}")
        ok &= max(relres) < 1e-9 and dev < defaults.SOLUTION_RTOL
    # the ladder is kept across the sweep, and the warm start + stale
    # ladder cost at most PRECOND_REUSE_ITER_SLACK iterations
    for policy in ("reuse+warm", "reuse+warm+recycle"):
        iters, builds = arms[policy][1], arms[policy][2]
        ok &= all(b == builds[0] for b in builds)
        ok &= all(i <= f + defaults.PRECOND_REUSE_ITER_SLACK
                  for i, f in zip(iters, arms["fresh"][1]))
    # recycling: solves 2..N under the ceiling, and never above the
    # counts of the same sweep without it
    rec, reuse = arms["reuse+warm+recycle"][1], arms["reuse+warm"][1]
    print(f"    recycling: solves 2..N {rec[1:]} (ceiling "
          f"{defaults.GCRO_SWEEP_ITER_MAX}, without recycling {reuse[1:]})")
    ok &= max(rec[1:]) <= defaults.GCRO_SWEEP_ITER_MAX
    ok &= max(rec[1:]) <= max(reuse[1:])

    # -- and dropped beyond the reuse window ---------------------------
    far = HBackend(jump="half").assemble(system, EPS)
    a1 = far.rebuild_for_materials({"zone": mb.ElasticMaterial(mu=10.0,
                                                              lam=10.0)})
    a1.solve(rtol=1e-9)
    a2 = a1.rebuild_for_materials({"zone": mb.ElasticMaterial(mu=40.0,
                                                             lam=40.0)})
    a2.solve(rtol=1e-9)
    rebuilt = (a2.report.precond_summary["build_s"]
               != a1.report.precond_summary["build_s"])
    print(f"    4x material step rebuilds the ladder: {rebuilt} "
          f"(window {defaults.PRECOND_REUSE_MAX_STEP:.3f} in log moduli)")
    return ok and rebuilt


def check_parallel_determinism():
    """Parallel block compression must be deterministic: per-block rng
    seeding makes the factors independent of thread scheduling, so two
    builds -- and builds at different worker counts -- must agree
    BITWISE. Also times the compression stage serial vs parallel."""
    import time

    # Two well-separated 3200-tri panels with max_admissible=1024 give
    # several large admissible blocks -- enough of them to fill the pool
    # the numba ACA runs every block on.
    field = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), 0.0,
                                   40, 40, normal_up=True)   # 3200 tris
    source = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), -240.0,
                                    40, 40, normal_up=True)
    eps_arr = kb.as_eps_array(EPS, source.n_triangles)
    mat = mb.ElasticMaterial(mu=30.0, lam=30.0)
    coeffs = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
    t0 = time.perf_counter()
    pc1 = PairCompressed(field, source, KERNEL_T, eps_arr,
                         max_admissible=1024, n_workers=1)
    t_serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    pc8 = PairCompressed(field, source, KERNEL_T, eps_arr,
                         max_admissible=1024)
    t_par = time.perf_counter() - t0
    pc8b = PairCompressed(field, source, KERNEL_T, eps_arr,
                          max_admissible=1024)

    d1 = pc1.to_dense(coeffs)
    d8 = pc8.to_dense(coeffs)
    d8b = pc8b.to_dense(coeffs)
    same_workers = np.array_equal(d1, d8)
    same_repeat = np.array_equal(d8, d8b)
    print(f"    blocks: {pc8.n_lowrank} low-rank / {pc8.n_dense} dense "
          f"({pc8.n_fallback} fallback)")
    print(f"    n_workers=1 vs auto bitwise-identical: {same_workers}; "
          f"repeat-build identical: {same_repeat}")
    print(f"    compression: serial {t_serial:.2f} s, parallel {t_par:.2f} s "
          f"({t_serial / max(t_par, 1e-9):.1f}x)")
    return same_workers and same_repeat


def check_calibrated():
    """Calibrated jump in the H path (roadmap C1): the compressed
    operator with jump='calibrated' must match the calibrated dense
    backend, annihilate rigid translations on the unknown-u slots to
    compression accuracy, and survive a material rebuild."""
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0))
    system = generate_system(model)
    layout = system.layout

    dense = AssembledDense(system, EPS, "direct", jump="calibrated")
    hasm = HBackend(jump="calibrated").assemble(system, EPS)

    op_err = _relmax(hasm.to_dense(), dense.A)
    rhs_err = _relmax(hasm.b, dense.b)
    print(f"    operator parity : rel = {op_err:.2e}")
    print(f"    RHS parity      : rel = {rhs_err:.2e}")

    sol_d = dense.solve()
    sol_h = hasm.solve(rtol=1e-9)
    report = hasm.report
    worst = max(_relmax(sol_h[k], sol_d[k]) for k in sol_d
                if np.max(np.abs(sol_d[k])) > 0)
    print(f"    solution parity : rel = {worst:.2e} "
          f"(iters {report.iterations}, converged {report.converged})")

    # rigid-translation identity: A_h @ x_rigid must equal the dense
    # calibrated A @ x_rigid exactly (on this anchored model the product
    # is NOT zero -- it equals the prescribed-base H terms by design --
    # but both operators must encode the identical calibrated identity).
    x = np.zeros(layout.n_unknowns)
    for slot in layout.slots:
        if slot.kind == "u":
            x[slot.offset:slot.stop:3] = 1.0     # unit x-translation
    rig = _relmax(hasm.matvec(x), dense.A @ x)
    print(f"    A_h @ rigid_x vs dense calibrated: rel = {rig:.2e}")

    # material rebuild recomputes the calibration
    reb = hasm.rebuild_for_materials(
        {"zone": mb.ElasticMaterial(mu=20.0, lam=15.0)})
    dense_b = AssembledDense(
        generate_system(_build_zone_model(
            mb.ElasticMaterial(mu=20.0, lam=15.0))),
        EPS, "direct", jump="calibrated")
    sol_rb, rep_rb = reb.solve(rtol=1e-9), reb.report
    sol_db = dense_b.solve()
    worst_rb = max(_relmax(sol_rb[k], sol_db[k]) for k in sol_db
                   if np.max(np.abs(sol_db[k])) > 0)
    print(f"    rebuild solution parity vs dense: rel = {worst_rb:.2e}")

    return (op_err < OP_PARITY and rhs_err < OP_PARITY and worst < SOL_PARITY
            and report.converged and worst_rb < SOL_PARITY)


def check_combined_storage():
    """storage='combined' (roadmap C2): 1x memory instead of B-fold
    per-basis storage. Same seeds + identical combine path => the
    combined views must equal the basis-mode views BITWISE; a material
    rebuild (transient re-compression) must reproduce basis mode."""
    field = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), 0.0,
                                   16, 16, normal_up=True)   # 512 tris
    source = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), -240.0,
                                    16, 16, normal_up=True)
    eps_arr = kb.as_eps_array(EPS, source.n_triangles)
    mat_a = mb.ElasticMaterial(mu=30.0, lam=30.0)
    mat_b = mb.ElasticMaterial(mu=12.0, lam=18.0)
    ca = np.asarray(kb.t_coeffs(mat_a.mu, mat_a.lam))
    cb = np.asarray(kb.t_coeffs(mat_b.mu, mat_b.lam))

    basis = PairCompressed(field, source, KERNEL_T, eps_arr,
                           max_admissible=512)
    comb = PairCompressed(field, source, KERNEL_T, eps_arr,
                          max_admissible=512, storage="combined",
                          combine_for=[ca])
    ratio = basis.nbytes() / max(comb.nbytes(), 1)
    print(f"    memory: basis {basis.nbytes()/1e6:.1f} MB vs combined "
          f"{comb.nbytes()/1e6:.1f} MB ({ratio:.1f}x)")

    same_a = np.array_equal(comb.to_dense(ca), basis.to_dense(ca))
    print(f"    combined vs basis view (built material): bitwise {same_a}")

    # unseen material -> transient re-compression
    err_b = _relmax(comb.to_dense(cb), basis.to_dense(cb))
    print(f"    rebuild material: rel = {err_b:.2e}")
    still_dropped = comb.blocks is None
    print(f"    basis payloads still dropped after rebuild: {still_dropped}")

    # end-to-end: combined+calibrated backend matches dense on the model
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0))
    system = generate_system(model)
    dense = AssembledDense(system, EPS, "direct", jump="calibrated")
    hc = HBackend(jump="calibrated", storage="combined").assemble(system, EPS)
    sol_d = dense.solve()
    sol_h, rep = hc.solve(rtol=1e-9), hc.report
    worst = max(_relmax(sol_h[k], sol_d[k]) for k in sol_d
                if np.max(np.abs(sol_d[k])) > 0)
    print(f"    combined+calibrated end-to-end vs dense: rel = {worst:.2e} "
          f"(converged {rep.converged})")

    return (ratio > 2.5 and same_a and err_b == 0.0
            and still_dropped and worst < SOL_PARITY and rep.converged)


def check_bj_rung():
    """Cluster block-Jacobi preconditioner rung (roadmap C3): for
    super-blocks too large for the HODLR build, the third rung must
    still converge FGMRES to the dense answer with a bounded iteration
    penalty (its build is O(N x chunk) at any size)."""
    import time

    from mbem.la.preconditioner import BlockGaussSeidel
    from mbem.la.solver import fgmres

    sys.path.insert(0, str(ROOT / "examples"))
    from _fault_box import build_fault_box, build_model

    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=30.0, fault_depth=18.0,
                             edge_fault=2.5, edge_near=8.0, edge_far=25.0,
                             edge_side=25.0, near_field_radius=60.0)
    model = build_model(meshes, 0.01, mb.ElasticMaterial(mu=30.0, lam=30.0))
    system = generate_system(model)
    top_dofs = 3 * meshes["top"].n_triangles
    print(f"    {system.layout.n_unknowns} unknowns; "
          f"largest super-block (top) = {top_dofs} DOFs")

    hasm = HBackend(jump="half").assemble(system, EPS)
    sol_ref = AssembledDense(system, EPS, "direct", jump="half").solve()

    # max_dense=1500 forces the top super-block OFF the dense-LU rung so
    # the HODLR and block-Jacobi rungs are genuinely exercised/compared.
    results = {}
    for name, hodlr_max in (("HODLR", 10 ** 9), ("block-Jacobi", 1)):
        t0 = time.perf_counter()
        M = BlockGaussSeidel(hasm, max_dense=1500, hodlr_max=hodlr_max)
        t_build = time.perf_counter() - t0
        x, rep = fgmres(hasm.matvec, hasm.b, M=M, rtol=1e-9)
        sol = {s.name: x[s.offset:s.stop].reshape(-1, 3)
               for s in system.layout.slots}
        worst = max(float(np.max(np.abs(sol[k] - sol_ref[k]))
                          / np.max(np.abs(sol_ref[k]))) for k in sol_ref
                    if np.max(np.abs(sol_ref[k])) > 0)
        results[name] = rep.iterations
        print(f"    {name:>12}: build {t_build:5.1f} s, "
              f"{rep.iterations:3d} iters, converged {rep.converged}, "
              f"vs dense rel = {worst:.2e}")
        if not (rep.converged and worst < SOL_PARITY):
            return False
    penalty = results["block-Jacobi"] / max(results["HODLR"], 1)
    print(f"    iteration penalty BJ/HODLR: {penalty:.1f}x")
    return penalty < 4.0


def _flat_view_parity(hasm, label, with_dense: bool):
    """Per (pair, material) parity of the flat view against the block
    loop: ``(worst to_dense, worst matvec, pairs checked)``."""
    from mbem.kernels import kernel_coeffs

    seen = {}
    for term in hasm.system.terms:
        pair = hasm.pair_for(term)
        c = np.asarray(kernel_coeffs(term.kernel,
                                     hasm.materials[term.region.name]))
        seen[(id(pair), c.tobytes())] = (pair, c)

    worst_d = worst_m = 0.0
    for pair, c in seen.values():
        ref = pair.view_reference(c)
        flat = pair._view(c)
        x = np.random.default_rng(7).standard_normal(pair.shape[1])
        y_ref = np.zeros(pair.shape[0])
        M = np.zeros(pair.shape) if with_dense else None
        for rdofs, cdofs, A, V in ref:
            if V is None:
                y_ref[rdofs] += A @ x[cdofs]
            else:
                y_ref[rdofs] += A @ (V.T @ x[cdofs])
            if with_dense:
                M[np.ix_(rdofs, cdofs)] += A if V is None else A @ V.T
        worst_m = max(worst_m, _relmax(flat.matvec(x), y_ref))
        if with_dense:
            worst_d = max(worst_d, _relmax(flat.to_dense(), M))
    print(f"    {label}: {len(seen)} (pair, material) views, "
          f"matvec vs block loop rel = {worst_m:.1e}"
          + (f", to_dense rel = {worst_d:.1e}" if with_dense else ""))
    return worst_d, worst_m, len(seen)


def check_flat_view():
    """The flat view (``la/flatview.py``) is what FGMRES applies: one
    parallel numba kernel over per-(pair, material) flat buffers, with
    the near field evaluated straight into them by the batched leaf
    kernel instead of stored per basis.

    Against ``PairCompressed.view_reference`` -- the pre-flat block loop,
    whose dense blocks are combined from the per-basis stacks in numpy --
    every (pair, material) view of the fault-zone model and of
    demo_hmatrix's 10.9k-unknown inclusion model must agree within
    ``defaults.FLATVIEW_PARITY``: the two sum the same terms in a
    different order (numba loops against BLAS dot, in-loop coefficients
    against tensordot), which is round-off, not tolerance. What IS
    bitwise is the flat matvec across thread counts -- the row chunks own
    their output rows, so no reduction order can change."""
    import numba

    sys.path.insert(0, str(ROOT / "examples"))
    from assess_fig06_inclusion import build as build_inclusion
    from assess_fig06_inclusion import build_model as inclusion_model

    ok = True
    worst_d, worst_m, _ = _flat_view_parity(
        HBackend(jump="half", storage="basis").assemble(
            generate_system(_build_zone_model(
                mb.ElasticMaterial(mu=10.0, lam=10.0))), EPS),
        "fault-zone", with_dense=True)
    ok &= worst_d < defaults.FLATVIEW_PARITY
    ok &= worst_m < defaults.FLATVIEW_PARITY

    meshes, fault, _n_hat, s_hat = build_inclusion()
    model = inclusion_model(meshes, fault, s_hat,
                            mat_inc=mb.ElasticMaterial(mu=3.0, lam=3.0))
    system = generate_system(model)
    hasm = HBackend(jump="half", storage="basis").assemble(system, "auto")
    print(f"    inclusion model: {system.layout.n_unknowns} unknowns")
    _, worst_inc, _ = _flat_view_parity(hasm, "inclusion", with_dense=False)
    ok &= worst_inc < defaults.FLATVIEW_PARITY

    # Thread-count determinism of the whole compressed operator.
    x = np.random.default_rng(11).standard_normal(system.layout.n_unknowns)
    y16 = hasm.matvec(x)
    numba.set_num_threads(1)
    y1 = hasm.matvec(x)
    numba.set_num_threads(5)
    y5 = hasm.matvec(x)
    numba.set_num_threads(16)
    same = np.array_equal(y16, y1) and np.array_equal(y16, y5)
    print(f"    operator matvec bitwise at 1 / 5 / 16 numba threads: {same}")
    return ok and same


def _aca_impl_parity(pairs, label, max_blocks=24):
    """Both ACA implementations over the admissible blocks of ``pairs``,
    from the same seeds: (same outcome, same ranks, worst factor
    difference, worst error against the exact block, blocks checked)."""
    from mbem.la.aca import BlockEvalCache, compress_block

    same_kind = True
    rank_gap = 0
    worst_impl = worst_exact = 0.0
    n_checked = n_lr = 0
    for pair in pairs:
        ev = pair.eval
        for i, (rows, cols) in enumerate(pair._part.admissible):
            if n_checked >= max_blocks:
                break
            py = compress_block(
                BlockEvalCache(ev.stack_serial, rows, cols), ev.n_basis,
                tol=pair.tol, rng=np.random.default_rng((12345, i)),
                exact_payload=False)
            nb = compress_block(
                BlockEvalCache(ev.stack_serial, rows, cols,
                               aca_fn=ev.aca_fn(rows, cols)), ev.n_basis,
                tol=pair.tol, rng=np.random.default_rng((12345, i)),
                exact_payload=False)
            n_checked += 1
            same_kind &= (py.payload is None) == (nb.payload is None)
            if py.payload is None or nb.payload is None:
                continue
            n_lr += 1
            rank_gap = max(rank_gap, max(abs(a - b) for a, b in
                                         zip(py.payload.ranks,
                                             nb.payload.ranks)))
            exact = ev.stack_serial(rows, cols)
            for b in range(ev.n_basis):
                A = py.payload.U[b] @ py.payload.V[b].T
                C = nb.payload.U[b] @ nb.payload.V[b].T
                scale = max(float(np.max(np.abs(exact[b]))), 1e-300)
                worst_impl = max(worst_impl,
                                 float(np.max(np.abs(A - C))) / scale)
                worst_exact = max(worst_exact,
                                  float(np.max(np.abs(A - exact[b]))) / scale,
                                  float(np.max(np.abs(C - exact[b]))) / scale)
    print(f"    {label}: {n_checked} blocks ({n_lr} low rank), same outcome "
          f"{same_kind}, largest rank difference {rank_gap}, numba vs python "
          f"{worst_impl:.1e}, either vs exact {worst_exact:.1e}")
    return same_kind, rank_gap, worst_impl, worst_exact, n_lr


def check_aca_numba():
    """Every admissible block of the compressed operator is compressed by
    one nogil numba kernel (``la/aca_numba.py``) on the thread pool;
    ``aca.compress_block``'s Python loop stays as its reference
    implementation (``BlockEvalCache(aca_fn=None)``).

    Run from the SAME seed over the blocks of the fault-zone pairs and of
    the 10.9k inclusion model, the two must make the same decision (low
    rank or applied exactly), reach the same rank in EVERY basis -- they
    draw the same lines and pivot identically -- and approximate the
    block equally well. They are not bitwise: the recompressions call
    different LAPACK builds, and where the truncated singular values are
    nearly tied the discarded tail differs by one tolerance
    (``defaults.ACA_IMPL_PARITY``). Each path is bitwise repeatable on
    its own (``check_parallel_determinism``)."""
    sys.path.insert(0, str(ROOT / "examples"))
    from assess_fig06_inclusion import build as build_inclusion
    from assess_fig06_inclusion import build_model as inclusion_model

    tol = defaults.BLOCK_COMPRESSION_TOL
    limit = defaults.ACA_IMPL_PARITY * tol
    exact_limit = defaults.H_PARITY_OPERATOR * tol

    zone = HBackend(jump="half").assemble(
        generate_system(_build_zone_model(
            mb.ElasticMaterial(mu=10.0, lam=10.0))), EPS)
    meshes, fault, _n_hat, s_hat = build_inclusion()
    inc = HBackend(jump="half").assemble(
        generate_system(inclusion_model(
            meshes, fault, s_hat,
            mat_inc=mb.ElasticMaterial(mu=3.0, lam=3.0))), "auto")

    # A guaranteed-admissible large block (two 512-triangle panels at 2x
    # their size) next to the two models' own blocks.
    panel_field = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), 0.0,
                                         16, 16, normal_up=True)
    panel_source = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0),
                                          -240.0, 16, 16, normal_up=True)
    panels = PairCompressed(panel_field, panel_source, KERNEL_T,
                            kb.as_eps_array(EPS, panel_source.n_triangles),
                            max_admissible=512)

    ok = True
    total_lr = 0
    for label, pairs in (("fault-zone", list(zone._pairs.values())),
                         ("inclusion", list(inc._pairs.values())),
                         ("512-tri panels", [panels])):
        kind, gap, impl, ex, n_lr = _aca_impl_parity(
            [p for p in pairs if p.n_lowrank], label)
        total_lr += n_lr
        ok &= kind and gap <= 1 and impl < limit and ex < exact_limit
    return ok and total_lr > 0


def check_convergence_rate():
    """The block-Gauss-Seidel ladder is size-independent: on the fault-zone
    model at two sizes ~2.7x apart (eps = "auto", calibrated jump, the
    caller settings of the other checks) both solves must converge with a
    true residual below rtol, and iterations(large) may be at most
    GMRES_ITER_GROWTH_MAX x iterations(small) and GMRES_ITER_CEILING."""
    import time

    iters = {}
    ok = True
    for label, refine in (("small", 1.0), ("large", 1.64)):
        model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0),
                                  refine=refine)
        system = generate_system(model)
        t0 = time.perf_counter()
        hasm = HBackend().assemble(system, "auto")
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        hasm.solve()
        t_solve = time.perf_counter() - t0
        rep = hasm.report
        iters[label] = rep.iterations
        rungs = sorted({sb["rung"] for sb in rep.precond_summary["super_blocks"]})
        print(f"    {label:>5} ({system.layout.n_unknowns:5d} unknowns): "
              f"{rep.iterations:3d} iters, converged {rep.converged}, "
              f"true relres {rep.true_relres:.2e}; build {t_build:.1f} s, "
              f"solve {t_solve:.1f} s, rungs {rungs}")
        ok &= rep.converged and rep.true_relres < defaults.GMRES_RTOL
    growth = iters["large"] / max(iters["small"], 1)
    print(f"    iteration growth large/small: {growth:.2f} "
          f"(limit {defaults.GMRES_ITER_GROWTH_MAX}); ceiling "
          f"{defaults.GMRES_ITER_CEILING}")
    return (ok and growth <= defaults.GMRES_ITER_GROWTH_MAX
            and iters["large"] <= defaults.GMRES_ITER_CEILING)


def main():
    checks = [
        ("pair-level ACA", check_pair_aca),
        ("HBackend vs dense end-to-end (fault-zone model)", check_end_to_end),
        ("rebuild_for_materials parity", check_rebuild),
        ("material sweep: views, ladder reuse, recycling",
         check_views_bounded),
        ("parallel ACA determinism + speed", check_parallel_determinism),
        ("calibrated jump in the H path", check_calibrated),
        ("combined storage (1x memory) parity", check_combined_storage),
        ("flat view vs the block loop", check_flat_view),
        ("numba ACA vs the Python reference", check_aca_numba),
        ("cluster block-Jacobi preconditioner rung", check_bj_rung),
        ("convergence rate across sizes", check_convergence_rate),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: HBackend matches the dense backend.")
    else:
        print("\nFAIL: the H path disagrees with the dense backend.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
