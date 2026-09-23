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
calibrated jump and the CALIBRATION ROW SUMS it is built from (the one
part of the operator an entrywise parity does not bound, and what the
bench harness's unit-translation vector measures), combined storage, the
flat view (the batched matvec and
leaf kernel against the block loop they replaced), the block-Jacobi rung,
and the convergence rate: FGMRES iterations on the fault-zone model at two
sizes ~2.8x apart must grow no faster than N^``GMRES_ITER_GROWTH_ALPHA``
and never past ``GMRES_ITER_CEILING``, under both preconditioner
groupings -- which must also agree on the solution and on the operator
their super-blocks invert -- and on the block-Jacobi rung that carries
every super-block past the dense cap; a deliberately non-scalable
chunking runs alongside them and the criterion must reject it.
"""
import math
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
from mbem.model.equations import (COLLOCATION_JUMP,               # noqa: E402
                                  calibrated_diagonal)
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


def _exact_calibration(system, hasm):
    """``calibrated_diagonal`` with the H row-sums taken from the KERNELS
    instead of from the compressed pairs -- the calibration the exact
    operator would carry."""
    arrays = kb.MeshArrays()

    def rowsum(region, q, p):
        mat = hasm.materials[region.name]
        coeffs = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
        xq = arrays.field_points(q.mesh)
        tv, nrm = arrays.source_arrays(p.mesh)
        eps_p = hasm.eps_for(p)
        S = np.zeros((q.n_nodes, 3, 3))
        for k in range(3):
            dens = np.zeros((p.n_nodes, 3))
            dens[:, k] = 1.0
            S[:, :, k] = tk.t_disp_contract(xq, tv, nrm, eps_p, dens, *coeffs)
        return S

    return calibrated_diagonal(system, rowsum)


def _calibration_error(system, hasm) -> float:
    exact = _exact_calibration(system, hasm)
    return max(float(np.max(np.abs(hasm.calib[k] - exact[k]))) for k in exact)


def check_calibration_rowsums():
    """The calibrated diagonal is built from the COMPRESSED row sums
    (``AssembledH._calibration``), so a block tolerance lands directly in
    the operator's free term: ``C_h - C_exact`` IS the row-sum
    compression error, and it is the only part of the operator no
    entrywise parity bounds. An entrywise bound is per block and
    relative to that block; a row sum adds every block of a row, and
    what it adds up to is a quantity the exact operator holds at
    ``COLLOCATION_JUMP``, so the two can move apart. It is what the
    bench harness's unit-translation test vector measures, and what a
    compression change degrades first.

    The model is the fault-zone box refined until its H pairs actually
    carry admissible blocks -- at the gate's own size the calibration
    pairs are all dense leaves and the error is round-off, which pins
    nothing. A second arm runs the same model at a 100x looser block
    tolerance and this criterion must REJECT it, so the clause is known
    to discriminate rather than merely to be satisfied.
    """
    limit = defaults.H_PARITY_CALIBRATION * TOL * COLLOCATION_JUMP
    model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0),
                              refine=2.0)
    system = generate_system(model)
    n_lr = 0
    errs = {}
    for label, tol in (("operator tolerance", TOL),
                       ("seeded: 100x looser", 100.0 * TOL)):
        hasm = HBackend(jump="calibrated", tol=tol).assemble(system, EPS)
        if tol == TOL:
            n_lr = sum(p.n_lowrank for p in hasm._pairs.values())
        errs[label] = _calibration_error(system, hasm)
        print(f"    {label:22s} tol {tol:.0e}: max |C_h - C_exact| = "
              f"{errs[label]:.2e} ({errs[label] / limit:.2f} of the limit)")
    print(f"    {system.layout.n_unknowns} unknowns, {n_lr} admissible "
          f"blocks; limit {limit:.2e} = {defaults.H_PARITY_CALIBRATION:g} x "
          f"tol x {COLLOCATION_JUMP:g}")
    return (n_lr > 0 and errs["operator tolerance"] < limit
            and errs["seeded: 100x looser"] > limit)


def check_combined_storage():
    """storage='combined' (roadmap C2): 1x memory, against the
    geometry-only payload storage='basis' keeps so that a new material
    never re-compresses. Same seeds + identical combine path => the
    combined views must equal the basis-mode views BITWISE; a material
    rebuild (transient re-compression) must reproduce basis mode.

    The payload is bounded on BOTH sides. Below: dropping it must
    actually save memory. Above: it is one shared subspace per block, not
    B factor pairs -- measured 2.4x one material view of this pair, where
    the same payload unfolded (``ACA_JOINT_TOL_FACTOR`` at 0, joint rank
    = the summed per-basis rank) is 6.4x. A regression that undid the
    fold would land above the upper bound."""
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
    print(f"    memory: basis {basis.nbytes()/1e6:.2f} MB (shared subspace, "
          f"no view) vs combined {comb.nbytes()/1e6:.2f} MB (one view): "
          f"{ratio:.1f}x")

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

    return (1.5 < ratio < 4.0 and same_a and err_b == 0.0
            and still_dropped and worst < SOL_PARITY and rep.converged)


def check_bj_rung():
    """The preconditioner rung ladder, all three rungs on ONE partition:
    each must converge FGMRES to the dense answer, and the two
    approximate rungs must cost a bounded number of extra iterations
    against the exact dense-LU rung (block-Jacobi is the rung that
    carries every super-block past the dense cap, so its penalty is the
    one with a limit; HODLR is the non-default rung and is here so that
    it stays gated).

    Every rung is FORCED here by its own cap rather than chosen by size:
    the dense cap is otherwise resolved per machine
    (``preconditioner.dense_rung_max_dof``, tens of thousands of DOFs),
    which would put this model's every super-block on rung 1 and leave
    the other two untested.

    Grouping is pinned to "patch" whatever the default is: this is a
    check on the RUNG, and the fault box is one region, whose "region"
    super-block would be the entire matrix (one block, no sweep)."""
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

    # (max_dense, above_dense, hodlr_max, bj_chunk) per arm. 1500 as the
    # dense cap puts the top super-block on the rung under test and the
    # same 1500 as the chunk makes the block-Jacobi arm cut it the same
    # way, so the three arms invert the same partition. The HODLR arm
    # names its rung explicitly, which is the only way it is reached:
    # ``defaults.PRECOND_RUNG_ABOVE_DENSE`` sends a block past the dense
    # cap to block-Jacobi, so this check is what keeps the non-default
    # rung gated.
    results = {}
    for name, caps in (("dense LU", (10 ** 9, "block_jacobi", 0, 1500)),
                       ("HODLR", (1500, "hodlr", 10 ** 9, 1500)),
                       ("block-Jacobi", (1500, "block_jacobi", 0, 1500))):
        t0 = time.perf_counter()
        M = BlockGaussSeidel(hasm, max_dense=caps[0], above_dense=caps[1],
                             hodlr_max=caps[2], bj_chunk=caps[3],
                             grouping="patch")
        t_build = time.perf_counter() - t0
        rungs = sorted({sb["rung"] for sb in M.summary()["super_blocks"]})
        x, rep = fgmres(hasm.matvec, hasm.b, M=M, rtol=1e-9)
        sol = {s.name: x[s.offset:s.stop].reshape(-1, 3)
               for s in system.layout.slots}
        worst = max(float(np.max(np.abs(sol[k] - sol_ref[k]))
                          / np.max(np.abs(sol_ref[k]))) for k in sol_ref
                    if np.max(np.abs(sol_ref[k])) > 0)
        results[name] = rep.iterations
        print(f"    {name:>12}: build {t_build:5.1f} s, "
              f"{rep.iterations:3d} iters, converged {rep.converged}, "
              f"vs dense rel = {worst:.2e}, rungs {rungs}")
        if not (rep.converged and worst < SOL_PARITY):
            return False
    exact = max(results["dense LU"], 1)
    for name in ("HODLR", "block-Jacobi"):
        print(f"    iteration penalty {name} / dense LU: "
              f"{results[name] / exact:.1f}x")
    return results["block-Jacobi"] <= 4.0 * exact


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
                                         zip(py.payload.basis_ranks,
                                             nb.payload.basis_ranks)))
            exact = ev.stack_serial(rows, cols)
            for b in range(ev.n_basis):
                A = py.payload.basis_matrix(b)
                C = nb.payload.basis_matrix(b)
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


def check_shared_subspace():
    """The B factor pairs of a block are folded ONCE, at build, into one
    shared subspace (``aca.shared_subspace``: ``A_b = Qu cores[b]
    Qv^T``), so a material enters through a k x k core sum instead of a
    QR+SVD of the block's own (3n, sum_b k_b) factors.

    Two things must hold, and neither of them is a timing:

    * the shared-subspace view IS the per-basis recombination. The
      reference is the pre-fold path itself -- ``aca.aca_single`` per
      basis from the block's own seed, then the QR+SVD re-truncation of
      the concatenated factors -- run on the SAME factors the fold gets,
      so what is compared is the fold alone: the recombined block must
      agree within the operator tolerance, at the same rank (+-1), for
      three materials, and stay that close to the exact block.
    * a material after the first pays no BLOCK-SIZED factorization.
      ``aca.FACTORIZATIONS`` counts those apart from the k x k core
      SVDs, and a second and a third material must add ZERO of the first
      and exactly one core SVD per low-rank block.

    And, over EVERY block a real model stores, the stored form must
    reproduce the exact block. The certificate checks three rows and
    three columns of each basis inside the compression kernel; this
    checks the whole block, outside it, on the model whose blocks carry
    the largest ranks (the refined fault zone) -- which is what catches a
    factorization that returns garbage rather than an error (numba's
    ``np.linalg.qr`` does exactly that on a rank-deficient tall factor;
    ``aca_numba._gram_qr``).
    """
    from mbem.la.aca import (FACTORIZATIONS, BlockEvalCache, SharedLR,
                             aca_single, draw_lines, recompress,
                             shared_subspace)

    field = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), 0.0,
                                   16, 16, normal_up=True)   # 512 tris
    source = make_rectangular_patch((-60.0, 60.0), (-60.0, 60.0), -240.0,
                                    16, 16, normal_up=True)
    eps_arr = kb.as_eps_array(EPS, source.n_triangles)
    pair = PairCompressed(field, source, KERNEL_T, eps_arr,
                          max_admissible=256)
    ev = pair.eval
    cs = [np.asarray(kb.t_coeffs(mu, lam))
          for mu, lam in ((30.0, 30.0), (3.0, 6.0), (80.0, 20.0))]
    delta = defaults.ACA_JOINT_TOL_FACTOR * pair.tol

    worst = worst_exact = 0.0
    rank_gap = 0
    joint = summed = n_blocks = 0
    for i, (rows, cols) in enumerate(pair._part.admissible):
        if pair.blocks[i][2] is None or n_blocks >= 6:
            continue
        cache = BlockEvalCache(ev.stack_serial, rows, cols)
        srows, scols, cert_rows, cert_cols = draw_lines(
            np.random.default_rng((12345, i)), len(rows), len(cols))
        max_rank = min(max(8, int(min(len(rows), len(cols))
                                  * defaults.ACA_MAX_RANK_FRACTION)),
                       len(rows), len(cols))
        stop_exact = cache.stack_fn(cache.rows[srows], cache.cols[scols])
        uvs = [aca_single(cache, b, len(rows), len(cols), pair.tol, srows,
                          scols, stop_exact[b], max_rank)
               for b in range(ev.n_basis)]
        if any(uv is None for uv in uvs):
            continue
        n_blocks += 1
        Us = [uv[0] for uv in uvs]
        Vs = [uv[1] for uv in uvs]
        Qu, Qv, cores = shared_subspace(Us, Vs, delta)
        shared = SharedLR(Qu, Qv, cores, tuple(u.shape[1] for u in Us),
                          cert_rows, cert_cols)
        joint += shared.rank
        summed += sum(shared.basis_ranks)
        exact = ev.stack_serial(rows, cols)
        for c in cs:
            U0, V0 = recompress(
                np.hstack([c[b] * Us[b] for b in range(ev.n_basis)]),
                np.hstack(Vs), pair.tol)
            U1, V1 = shared.combine(c, pair.tol)
            ref = np.tensordot(c, exact, axes=1)
            scale = max(float(np.max(np.abs(ref))), 1e-300)
            A0, A1 = U0 @ V0.T, U1 @ V1.T
            worst = max(worst, float(np.max(np.abs(A1 - A0))) / scale)
            worst_exact = max(worst_exact,
                              float(np.max(np.abs(A1 - ref))) / scale)
            rank_gap = max(rank_gap, abs(U1.shape[1] - U0.shape[1]))
    print(f"    {n_blocks} blocks: shared-subspace view vs the per-basis "
          f"recombination {worst:.1e} (limit {TOL:.1e}), vs the exact block "
          f"{worst_exact:.1e}, rank difference {rank_gap}")
    print(f"    joint rank {joint / max(n_blocks, 1):.1f} vs summed "
          f"{summed / max(n_blocks, 1):.1f} per block "
          f"({joint / max(summed, 1):.2f}x)")
    ok = (n_blocks > 0 and worst < TOL and rank_gap <= 1
          and worst_exact < defaults.H_PARITY_OPERATOR * TOL)

    n_lr = sum(1 for _, _, p in pair.blocks if p is not None)
    pair._view(cs[0])
    before = dict(FACTORIZATIONS)
    for c in cs[1:]:
        pair._view(c)
    d_sub = FACTORIZATIONS["subspace"] - before["subspace"]
    d_core = FACTORIZATIONS["core"] - before["core"]
    print(f"    materials 2-3 over {n_lr} low-rank blocks: {d_sub} "
          f"block-sized factorizations, {d_core} k x k core SVDs")
    ok &= d_sub == 0 and d_core == 2 * n_lr

    # -- every stored block of a real model, against the exact block ---
    hasm = HBackend(storage="basis").assemble(
        generate_system(_build_zone_model(
            mb.ElasticMaterial(mu=10.0, lam=10.0), refine=1.64)), "auto")
    n_stored = 0
    worst_stored = 0.0
    max_rank = 0
    for p in hasm._pairs.values():
        ev = p.eval
        for rows, cols, payload in p.blocks:
            if payload is None:
                continue
            n_stored += 1
            max_rank = max(max_rank, max(payload.basis_ranks))
            block = ev.stack_serial(rows, cols)
            for b in range(ev.n_basis):
                scale = max(float(np.max(np.abs(block[b]))), 1e-300)
                worst_stored = max(worst_stored, float(np.max(np.abs(
                    payload.basis_matrix(b) - block[b]))) / scale)
    limit = defaults.H_PARITY_OPERATOR * TOL
    print(f"    {n_stored} stored blocks of the refined fault zone (largest "
          f"basis rank {max_rank}): worst entry vs the exact block "
          f"{worst_stored:.1e} (limit {limit:.1e})")
    return ok and n_stored > 0 and worst_stored < limit


def check_certificate_retry():
    """A failed certificate is a failure of the ACA's STOPPING RULE, not
    a verdict on the block, so ``aca.compress_block`` re-runs the ACA
    once at ``ACA_RETRY_TOL_FACTOR`` times the tolerance and only a
    second failure is applied exactly.

    Over every admissible block of the 10.9k inclusion model, from the
    build's own seeds, this gates three things:

    * the retry FIRES here, and each block it fires on really did fail
      its certificate on the first pass (the same lines, deterministic);
    * what is kept is certified: a retried block that comes back as
      factors is inside ``ACA_CERTIFY_FACTOR`` x tol, and its product is
      closer to the exact block than the first pass's was;
    * the operator path (``exact_payload=False``) never returns a
      STACK -- only certified factors or "apply this block exactly" --
      so the per-basis SVD of an exact block, B x O((3 n)^3) at one BLAS
      thread, is unreachable there whatever the block's size.
    """
    sys.path.insert(0, str(ROOT / "examples"))
    from assess_fig06_inclusion import build as build_inclusion
    from assess_fig06_inclusion import build_model as inclusion_model
    from mbem.la.aca import (BlockEvalCache, SharedLR, compress_block,
                             draw_lines)

    limit = defaults.ACA_CERTIFY_FACTOR * TOL
    meshes, fault, _n_hat, s_hat = build_inclusion()
    inc = HBackend(jump="half", storage="basis").assemble(
        generate_system(inclusion_model(
            meshes, fault, s_hat,
            mat_inc=mb.ElasticMaterial(mu=3.0, lam=3.0))), "auto")

    n_blocks = n_retry = n_exact = n_kept = 0
    worst_first = worst_kept = 0.0
    first_vs_exact = kept_vs_exact = 0.0
    ok = True
    for pair in inc._pairs.values():
        ev = pair.eval
        for i, (rows, cols) in enumerate(pair._part.admissible):
            n_blocks += 1
            res = compress_block(
                BlockEvalCache(ev.stack_serial, rows, cols,
                               aca_fn=ev.aca_fn(rows, cols)),
                ev.n_basis, tol=TOL, rng=np.random.default_rng((12345, i)),
                exact_payload=False)
            ok &= res.payload is None or isinstance(res.payload, SharedLR)
            n_exact += res.payload is None
            if not res.retried:
                continue
            n_retry += 1
            # The first pass the retry replaced: the same lines from the
            # same seed, at the plain tolerance.
            lines = draw_lines(np.random.default_rng((12345, i)),
                               len(rows), len(cols))
            max_rank = min(max(8, int(min(len(rows), len(cols))
                                      * defaults.ACA_MAX_RANK_FRACTION)),
                           len(rows), len(cols))
            Us, Vs, err0, capped0 = ev.aca_fn(rows, cols)(TOL, *lines,
                                                          max_rank)
            ok &= capped0 or err0 > limit        # it really did fail
            worst_first = max(worst_first, err0)
            if not isinstance(res.payload, SharedLR) or capped0:
                continue
            n_kept += 1
            worst_kept = max(worst_kept, res.max_err)
            ok &= res.max_err <= limit
            exact = ev.stack_serial(rows, cols)
            for b in range(ev.n_basis):
                scale = max(float(np.max(np.abs(exact[b]))), 1e-300)
                first_vs_exact = max(first_vs_exact, float(np.max(np.abs(
                    Us[b] @ Vs[b].T - exact[b]))) / scale)
                kept_vs_exact = max(kept_vs_exact, float(np.max(np.abs(
                    res.payload.basis_matrix(b) - exact[b]))) / scale)
    print(f"    {n_blocks} admissible blocks: {n_retry} retried "
          f"({n_kept} kept as factors), {n_exact} applied exactly, "
          f"0 stacks materialized")
    print(f"    retried blocks: first pass certified {worst_first:.1e} "
          f"(limit {limit:.1e}), retry {worst_kept:.1e}; vs the exact "
          f"block {first_vs_exact:.1e} -> {kept_vs_exact:.1e}")
    return ok and n_retry > 0 and kept_vs_exact < first_vs_exact


def _growth_alpha(it_small, it_large, n_small, n_large):
    """Exponent of iterations ~ N^alpha between two sizes."""
    return (math.log(it_large / max(it_small, 1))
            / math.log(n_large / n_small))


def check_convergence_rate():
    """How the preconditioned iteration count grows with the problem
    size, on the fault-zone model at two sizes ~2.8x apart (eps = "auto",
    calibrated jump, the caller settings of the other checks). Every
    solve must converge with a true residual below rtol, and the exponent
    alpha of iterations ~ N^alpha must stay under
    ``defaults.GMRES_ITER_GROWTH_ALPHA``, the count under
    ``GMRES_ITER_CEILING``.

    Four arms, because three different things set that count:

    * the two GROUPINGS at the default ladder
      (``defaults.PRECOND_GROUPING``) -- "region" inverts each region's
      whole block and iterates about half as often as "patch" -- which
      must reach the SAME solution to the solver tolerance, since a
      preconditioner changes the path, never the answer. Every
      super-block of this model fits the dense-LU rung, so these two
      measure the OPERATOR's own size-independence (alpha ~ 0.05);
    * the SCALING RUNG: cluster block-Jacobi forced on every super-block
      with the chunk cut to 1000 DOFs, so that it really subdivides at
      this model's size (see below). This is the rung the default
      ladder uses past the dense cap and the only one whose count grows,
      so it is the arm the exponent is really for;
    * a SEEDED FAILURE: the same rung at a 96-DOF chunk, a
      preconditioner that is deliberately not scalable. The check fails
      if the criterion ACCEPTS it. Without it a growth bound proves
      nothing -- it would pass just as happily at any value.

    Where the dense operator is affordable (the small size) the gate goes
    one level deeper: each super-block's diagonal AS THE LADDER EVALUATES
    IT, permuted back to the slot-concatenated layout, must reproduce the
    assembled operator's own sub-block -- exactly where that sub-block is
    near field, and to the compression tolerance where it is not. It is
    what keeps a grouping from inverting a different operator than the one
    FGMRES applies."""
    import time

    from mbem.la.preconditioner import BlockGaussSeidel
    from mbem.la.solver import fgmres

    # Chunk sizes for the two forced block-Jacobi arms. This model's
    # largest super-block is 384 DOFs small and 1092 large, so 1000 is
    # what puts the rung in the regime the production ladder runs in --
    # a chunk FIXED while the block grows -- and 96 is the degenerate
    # near-pointwise chunking that must be rejected.
    bj_chunk, seed_chunk = 1000, 96
    arms = {
        "patch": dict(grouping="patch"),
        "region": dict(grouping="region"),
        "patch/bj": dict(grouping="patch", max_dense=1, bj_chunk=bj_chunk),
        "seeded fail": dict(grouping="patch", max_dense=1,
                            bj_chunk=seed_chunk),
    }
    iters = {name: {} for name in arms}
    unknowns = {}
    ok = True
    for label, refine in (("small", 1.0), ("large", 1.64)):
        model = _build_zone_model(mb.ElasticMaterial(mu=10.0, lam=10.0),
                                  refine=refine)
        system = generate_system(model)
        unknowns[label] = system.layout.n_unknowns
        t0 = time.perf_counter()
        hasm = HBackend().assemble(system, "auto")
        t_build = time.perf_counter() - t0
        A = hasm.to_dense() if label == "small" else None
        print(f"    {label:>5} ({system.layout.n_unknowns:5d} unknowns): "
              f"operator built in {t_build:.1f} s")
        sols = {}
        for name, kw in arms.items():
            t0 = time.perf_counter()
            M = BlockGaussSeidel(hasm, **kw)
            t_ladder = time.perf_counter() - t0
            if A is not None and name in ("patch", "region"):
                worst = max(_relmax(M.diagonal_block(k),
                                    A[np.ix_(sb.global_idx, sb.global_idx)])
                            for k, sb in enumerate(M.sbs))
                print(f"       {name:>11}: super-block diagonals vs "
                      f"the assembled operator rel = {worst:.1e}")
                ok &= worst < OP_PARITY
            t0 = time.perf_counter()
            sols[name], rep = fgmres(hasm.matvec, hasm.b, M=M)
            t_solve = time.perf_counter() - t0
            iters[name][label] = rep.iterations
            rungs = sorted({sb["rung"] for sb in M.summary()["super_blocks"]})
            print(f"       {name:>11}: {len(M.sbs):2d} super-blocks, "
                  f"{rep.iterations:3d} iters, converged {rep.converged}, "
                  f"true relres {rep.true_relres:.2e}; ladder "
                  f"{t_ladder:.1f} s, solve {t_solve:.1f} s, rungs {rungs}")
            # the seeded arm is allowed to be bad, not wrong
            ok &= rep.converged and rep.true_relres < defaults.GMRES_RTOL
        dev = _relmax(sols["region"], sols["patch"])
        print(f"       region vs patch solution: rel = {dev:.2e}")
        ok &= dev < defaults.SOLUTION_RTOL
    limit = defaults.GMRES_ITER_GROWTH_ALPHA
    accepted, by_growth = {}, {}
    for name, it in iters.items():
        alpha = _growth_alpha(it["small"], it["large"],
                              unknowns["small"], unknowns["large"])
        by_growth[name] = alpha <= limit
        accepted[name] = (by_growth[name]
                          and it["large"] <= defaults.GMRES_ITER_CEILING)
        print(f"    {name:>11}: {it['small']} -> {it['large']} iterations "
              f"over {unknowns['large'] / unknowns['small']:.2f}x, "
              f"alpha = {alpha:.3f} (limit {limit}, ceiling "
              f"{defaults.GMRES_ITER_CEILING}) -> "
              f"{'accepted' if accepted[name] else 'REJECTED'}")
    ok &= all(accepted[name] for name in ("patch", "region", "patch/bj"))
    # the gate has teeth only if it rejects the seed, and it must be the
    # GROWTH criterion that does it, not the absolute ceiling
    print(f"    the seeded {seed_chunk}-DOF chunking is rejected: "
          f"{not accepted['seeded fail']} (by growth alone: "
          f"{not by_growth['seeded fail']})")
    return ok and not by_growth["seeded fail"]


def main():
    checks = [
        ("pair-level ACA", check_pair_aca),
        ("HBackend vs dense end-to-end (fault-zone model)", check_end_to_end),
        ("rebuild_for_materials parity", check_rebuild),
        ("material sweep: views, ladder reuse, recycling",
         check_views_bounded),
        ("parallel ACA determinism + speed", check_parallel_determinism),
        ("calibrated jump in the H path", check_calibrated),
        ("calibration row sums vs the exact kernels",
         check_calibration_rowsums),
        ("combined storage (1x memory) parity", check_combined_storage),
        ("flat view vs the block loop", check_flat_view),
        ("numba ACA vs the Python reference", check_aca_numba),
        ("shared subspace: fold parity and no per-material factorization",
         check_shared_subspace),
        ("certificate retry, and no exact stack in the operator path",
         check_certificate_retry),
        ("preconditioner rung ladder: dense LU / HODLR / block-Jacobi",
         check_bj_rung),
        ("convergence rate and grouping across sizes",
         check_convergence_rate),
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
