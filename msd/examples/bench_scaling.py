"""Scaling/regression bench for the mbem stack.

Times every pipeline stage at a ladder of mesh resolutions on the
fault-box model (free top, prescribed base, buried strike-slip fault):

  system    RegionModel -> generate_system
  dense_asm AssembledDense(mode="direct", jump="calibrated")  [demo config]
  dense_lu  LU factor + solve (+ cond estimate)
  h_asm     HBackend(eta=0.8, tol=1e-6) compression
  h_solve   preconditioned FGMRES (jump="half" operator)
  eval_u    evaluate_displacement on a surface grid
  eval_sig  evaluate_stress at fault centroids (elastic)

Reports wall time per stage, FGMRES iterations, compressed-operator
size, ACA low-rank/dense/fallback counts, and process peak RSS. Use
``--json out.json`` to write machine-readable results for regression
tracking. This is the baseline harness for the speed/scaling roadmap --
run it before and after any change to assembly, compression, or
evaluation.

Usage (from repo root):
    python examples/bench_scaling.py                 # 3-point ladder
    python examples/bench_scaling.py --sizes 0 1     # subset
    python examples/bench_scaling.py --json bench.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mollified_bem as mb                                        # noqa: E402
from mbem.kernels import KERNEL_T                                 # noqa: E402
from _fault_box import build_fault_box, build_model               # noqa: E402
from mbem.backends import HBackend                                # noqa: E402
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.evaluate import evaluate_displacement, evaluate_stress  # noqa: E402
from mbem.model import generate_system                            # noqa: E402

MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)
SLIP = 0.01
EPS = 3.0

# Mesh ladder: (edge_fault, edge_near, edge_far, edge_side) — each step
# refines edges by ~1.6x (~2.6x more triangles).
LADDER = [
    dict(edge_fault=4.0, edge_near=25.0, edge_far=50.0, edge_side=50.0),
    dict(edge_fault=2.5, edge_near=15.0, edge_far=32.0, edge_side=32.0),
    dict(edge_fault=1.6, edge_near=9.0, edge_far=20.0, edge_side=20.0),
]
N_EVAL_GRID = 45          # eval_u grid is N x N surface points
N_EVAL_SIG = 64           # eval_sig observation count (fault centroids)


def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return ru / 1e9 if sys.platform == "darwin" else ru * 1e3 / 1e9


def bench_size(cfg: dict) -> dict:
    rec: dict = {"cfg": cfg}

    t0 = time.perf_counter()
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=30.0, fault_depth=18.0,
                             near_field_radius=50.0, **cfg)
    model = build_model(meshes, SLIP, MAT)
    system = generate_system(model)
    rec["system_s"] = time.perf_counter() - t0
    rec["n_tris"] = int(sum(meshes[k].n_triangles
                            for k in ("top", "sides", "base", "fault")))
    rec["n_unknowns"] = int(system.layout.n_unknowns)

    # -- dense direct + calibrated (the demo configuration) ----------
    t0 = time.perf_counter()
    dense = AssembledDense(system, EPS, "direct", jump="calibrated")
    rec["dense_asm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol = dense.solve()
    rec["dense_lu_s"] = time.perf_counter() - t0
    rec["cond_estimate"] = float(dense.report.cond_estimate)

    # -- H backend ----------------------------------------------------
    t0 = time.perf_counter()
    hasm = HBackend(eta=0.8, tol=1e-6).assemble(system, EPS)
    rec["h_asm_s"] = time.perf_counter() - t0
    rec["h_nbytes_mb"] = hasm.nbytes() / 1e6
    rec["h_lowrank"] = int(sum(p.n_lowrank for p in hasm._pairs.values()))
    rec["h_dense"] = int(sum(p.n_dense for p in hasm._pairs.values()))
    rec["h_fallback"] = int(sum(p.n_fallback for p in hasm._pairs.values()))

    t0 = time.perf_counter()
    hasm.solve()
    report = hasm.report
    rec["h_solve_s"] = time.perf_counter() - t0
    rec["h_iters"] = int(report.iterations)
    rec["h_converged"] = bool(report.converged)

    # -- evaluation ----------------------------------------------------
    g = np.linspace(-80.0, 80.0, N_EVAL_GRID)
    X, Y = np.meshgrid(g, g)
    obs_u = np.column_stack([X.ravel(), Y.ravel(),
                             np.full(X.size, -30.0)])
    region = model.regions[0]
    t0 = time.perf_counter()
    evaluate_displacement(model, region, sol, obs_u, EPS)
    rec["eval_u_s"] = time.perf_counter() - t0
    rec["n_eval_u"] = int(obs_u.shape[0])

    c = meshes["fault"].centroids()
    obs_s = c[np.linspace(0, len(c) - 1, min(N_EVAL_SIG, len(c)),
                          dtype=int)]
    t0 = time.perf_counter()
    evaluate_stress(model, region, sol, obs_s, EPS)
    rec["eval_sig_s"] = time.perf_counter() - t0
    rec["n_eval_sig"] = int(obs_s.shape[0])

    rec["peak_rss_gb"] = peak_rss_gb()
    return rec


COLS = [("n_tris", "tris", "{:d}"), ("n_unknowns", "unk", "{:d}"),
        ("dense_asm_s", "dnsA(s)", "{:.1f}"), ("dense_lu_s", "LU(s)", "{:.1f}"),
        ("h_asm_s", "hA(s)", "{:.1f}"), ("h_solve_s", "hS(s)", "{:.1f}"),
        ("h_iters", "it", "{:d}"), ("h_nbytes_mb", "hMB", "{:.0f}"),
        ("h_fallback", "fbk", "{:d}"),
        ("eval_u_s", "evU(s)", "{:.1f}"), ("eval_sig_s", "evS(s)", "{:.1f}"),
        ("cond_estimate", "cond", "{:.1e}"), ("peak_rss_gb", "RSS", "{:.1f}")]


def bench_panel(n_side: int, storage: str, tol: float) -> dict:
    """Large-scale compression primitive: ONE PairCompressed between two
    parallel n_side x n_side panels (2*n_side^2 triangles each). This is
    the stage that dominates very-large-model assembly; run it at
    --panel 160 (51k tris), 224 (100k tris), ... to check feasibility.
    """
    from local_box_mesh import make_rectangular_patch
    from mbem.kernels import basis as kbm
    from mbem.la.hop import PairCompressed

    L = 400.0
    field = make_rectangular_patch((-L, L), (-L, L), 0.0,
                                   n_side, n_side, normal_up=True)
    source = make_rectangular_patch((-L, L), (-L, L), -2.0 * L,
                                    n_side, n_side, normal_up=True)
    n_tris = source.n_triangles
    eps_arr = kbm.as_eps_array(2.0 * L / n_side, n_tris)
    t0 = time.perf_counter()
    pc = PairCompressed(field, source, KERNEL_T, eps_arr, tol=tol,
                        storage=storage,
                        combine_for=[kbm.t_coeffs(30.0, 30.0)]
                        if storage == "combined" else None)
    dt = time.perf_counter() - t0
    rec = dict(panel_tris=int(n_tris), storage=storage, tol=tol,
               compress_s=dt, nbytes_mb=pc.nbytes() / 1e6,
               dense_equiv_mb=pc.dense_equivalent_bytes() / 1e6,
               lowrank=int(pc.n_lowrank), dense=int(pc.n_dense),
               fallback=int(pc.n_fallback), peak_rss_gb=peak_rss_gb())
    print(f"  panel {n_tris} tris [{storage}, tol={tol:g}]: "
          f"compress {dt:.1f} s, {rec['nbytes_mb']:.0f} MB "
          f"(dense {rec['dense_equiv_mb']:.0f} MB), "
          f"{rec['lowrank']} LR / {rec['dense']} dense "
          f"({rec['fallback']} fbk), RSS {rec['peak_rss_gb']:.1f} GB",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=list(range(len(LADDER))))
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--panel", type=int, default=None,
                    help="side count for the large-panel compression "
                         "primitive (2*panel^2 triangles per mesh); "
                         "skips the model ladder")
    ap.add_argument("--storage", choices=("basis", "combined"),
                    default="combined")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    records = []
    if args.panel:
        records.append(bench_panel(args.panel, args.storage, args.tol))
    else:
        print(" ".join(f"{h:>8}" for _, h, _ in COLS), flush=True)
        for i in args.sizes:
            rec = bench_size(LADDER[i])
            records.append(rec)
            print(" ".join(f"{fmt.format(rec[k]):>8}" for k, _, fmt in COLS),
                  flush=True)

    if args.json:
        out = pathlib.Path(args.json)
        out.write_text(json.dumps(records, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
