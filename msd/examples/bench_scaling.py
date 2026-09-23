"""Scaling / regression harness for the mbem stack.

Two model ladders, one rung per (model, scale, backend):

  --model fault_box       homogeneous box + buried strike-slip fault
                          (examples/_fault_box); every mesh edge of the
                          scale-1 configuration divided by ``scale``
                          (1 / 1.6 / 2.6 is the historical 3-point ladder)
  --model topo_inclusion  the figure-10 host + soft inclusion + topography
                          model (examples/make_topo_inclusion.build with
                          ``scale``): 4 unknowns per triangle, interfaces

Per rung a JSON record: triangles, unknowns, per-phase wall time (mesh,
system, partition, ACA, dense leaves, views, preconditioner build, matvec
median, solve), FGMRES iterations and true residual, the preconditioner
rung per super-block, low-rank ranks, ACA fallbacks, bytes (near /
low-rank / per-basis / per unknown), peak RSS, numba threads and layer,
the load average before the rung, git hash and a snapshot of ``defaults``.
Operator error is measured against the dense operator up to
``BENCH_DENSE_MAX_UNKNOWNS`` (seeded random vectors + the unit
translation) and against ``BENCH_EXACT_ROWS`` exact matrix-free rows
(``AssembledH.matvec_exact_rows``) beyond; the solution against the dense
LU per slot where dense is affordable.

Each rung runs in a child process: ``resource.getrusage`` peak RSS is
process-wide and monotone, so this is the only way to attribute it per
rung, and it gives ``--timeout`` per rung. Stage times inside
``PairCompressed`` are taken by wrapping ``la.hop``'s stage functions
from outside (``_HopPhases``), so the library carries no timers.

``--gate REV`` reruns the ladder and compares it with the ``bench-json:``
line of REV's commit message (the compact record list this script prints
at the end of every run, meant to be pasted into the commit message):
one PASS:/FAIL: line, exit 1 on any breach of the ``BENCH_*`` thresholds
in ``defaults.py``. Meshes are cached as npz under ``--mesh-cache``
(never inside the tree). ``--panel N`` is the single-pair compression
feasibility probe and skips the ladders.

Usage (from the msd root):
    python examples/bench_scaling.py                        # fault box, 1 / 1.6 / 2.6
    python examples/bench_scaling.py --model topo_inclusion --scale 1 --backend hmat dense
    python examples/bench_scaling.py --scale 1 1.6 --json after.json --gate HEAD
    python examples/bench_scaling.py --panel 160
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import mollified_bem as mb                                        # noqa: E402
from mbem import defaults                                         # noqa: E402
from mbem.backends import HBackend                                # noqa: E402
from mbem.backends.dense import AssembledDense, translation_basis # noqa: E402
from mbem.estimate import total_ram_bytes                         # noqa: E402
from mbem.kernels import KERNEL_T                                 # noqa: E402
from mbem.la import hop                                           # noqa: E402
from mbem.model import generate_system                            # noqa: E402

MODELS = ("fault_box", "topo_inclusion")
BACKENDS = ("dense", "hmat")
DEFAULT_SCALES = {"fault_box": [1.0, 1.6, 2.6], "topo_inclusion": [1.0]}
DEFAULT_MESH_CACHE = pathlib.Path("~/.cache/msd_bench_meshes").expanduser()
EPS = "auto"
JUMP = "calibrated"

# fault_box at scale 1: the historical first rung; edges are divided by scale.
FAULT_BOX = dict(half_x=100.0, z_bottom=-60.0, fault_half_len=30.0,
                 fault_depth=18.0, near_field_radius=50.0)
FAULT_BOX_EDGES = dict(edge_fault=4.0, edge_near=25.0, edge_far=50.0,
                       edge_side=50.0)
FAULT_BOX_MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)
FAULT_BOX_SLIP = 0.01


# ---------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------

def peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return ru / 1e9 if sys.platform == "darwin" else ru * 1e3 / 1e9


LOAD_POLL_S = 5.0        # how often wait_for_load re-reads os.getloadavg


def wait_for_load(tag: str) -> float:
    """The 1-minute load average, after waiting for it to fall back under
    ``BENCH_LOAD_MAX`` (at most ``BENCH_LOAD_WAIT_S``).

    A gated ladder runs its rungs back to back, and a rung's own 16
    threads leave the average well above the limit for a minute or two
    afterwards, so the limit is a thing to WAIT for, not to refuse on:
    only a load that outlasts the wait belongs to another process.
    Returns the last reading, which the caller compares with the limit.
    """
    t0 = time.perf_counter()
    load = os.getloadavg()[0]
    waited = False
    while load > defaults.BENCH_LOAD_MAX:
        if time.perf_counter() - t0 > defaults.BENCH_LOAD_WAIT_S:
            break
        if not waited:
            print(f"  waiting for load {load:.2f} to fall under "
                  f"{defaults.BENCH_LOAD_MAX:g} before {tag} "
                  f"(up to {defaults.BENCH_LOAD_WAIT_S:.0f} s)", flush=True)
            waited = True
        time.sleep(LOAD_POLL_S)
        load = os.getloadavg()[0]
    if waited:
        print(f"  load {load:.2f} after {time.perf_counter() - t0:.0f} s",
              flush=True)
    return load


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=True
                           ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."],
                               cwd=ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def numba_threads() -> dict:
    import numba
    try:
        layer = numba.threading_layer()      # resolved once a parallel kernel ran
    except ValueError:
        layer = None
    return {"threads": int(numba.get_num_threads()), "layer": layer}


def defaults_snapshot() -> dict:
    def _safe(v):
        if isinstance(v, dict):
            return {str(k): _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        return v
    return {k: _safe(v) for k, v in vars(defaults).items() if k.isupper()}


# ---------------------------------------------------------------------
# meshes (cached outside the tree)
# ---------------------------------------------------------------------

def _cache_path(cache_dir: pathlib.Path, model: str, scale: float):
    return cache_dir / f"{model}_scale{scale:g}.npz"


def _save_meshes(path, meshes: dict, arrays: dict, params: dict):
    payload = {"params": json.dumps(params, sort_keys=True)}
    for name, m in meshes.items():
        payload[f"{name}__v"] = m.vertices
        payload[f"{name}__t"] = m.triangles
    for name, a in arrays.items():
        payload[f"arr__{name}"] = np.asarray(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_meshes(path, params: dict):
    """(meshes, arrays) from the npz, or None when absent or built with
    other parameters."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["params"]) != json.dumps(params, sort_keys=True):
            return None
        meshes, arrays = {}, {}
        for key in z.files:
            if key.endswith("__v"):
                name = key[:-3]
                meshes[name] = mb.TriMesh(vertices=z[key],
                                          triangles=z[f"{name}__t"])
            elif key.startswith("arr__"):
                arrays[key[5:]] = z[key]
    return meshes, arrays


def build_meshes(model: str, scale: float, cache_dir: pathlib.Path,
                 mu_inc: float) -> tuple:
    """(RegionModel, meshes dict, mesh wall seconds, cached flag)."""
    from _fault_box import build_fault_box
    from _fault_box import build_model as fault_box_model

    if model == "fault_box":
        params = dict(FAULT_BOX, scale=scale,
                      **{k: v / scale for k, v in FAULT_BOX_EDGES.items()})
    else:
        params = dict(scale=scale)
    path = _cache_path(cache_dir, model, scale)
    hit = _load_meshes(path, params)
    t0 = time.perf_counter()
    if hit is None:
        if model == "fault_box":
            built = build_fault_box(**{k: v for k, v in params.items()
                                       if k != "scale"})
            meshes = {k: built[k] for k in ("top", "base", "sides", "fault")}
            arrays = {"n_hat": built["n_hat"], "s_hat": built["s_hat"]}
        else:
            from make_topo_inclusion import build as topo_build
            (m, _top_flat, top_topo, _fault_flat, fault_topo,
             s_hat, _bump) = topo_build(scale=scale)
            meshes = dict(m)
            meshes["host_top"] = top_topo       # the (topo, het) state
            meshes["fault"] = fault_topo
            arrays = {"s_hat": s_hat}
        _save_meshes(path, meshes, arrays, params)
    else:
        meshes, arrays = hit
    mesh_s = time.perf_counter() - t0

    if model == "fault_box":
        m = dict(meshes, s_hat=arrays["s_hat"], n_hat=arrays["n_hat"],
                 x_range=(-FAULT_BOX["half_x"], FAULT_BOX["half_x"]),
                 z_bottom=FAULT_BOX["z_bottom"])
        region_model = fault_box_model(m, FAULT_BOX_SLIP, FAULT_BOX_MAT)
    else:
        from assess_fig06_inclusion import build_model as inclusion_model
        region_model = inclusion_model(
            {k: v for k, v in meshes.items() if k != "fault"},
            meshes["fault"], arrays["s_hat"],
            mat_inc=mb.ElasticMaterial(mu=mu_inc, lam=mu_inc))
    return region_model, meshes, mesh_s, hit is not None


# ---------------------------------------------------------------------
# stage timing of PairCompressed, from outside
# ---------------------------------------------------------------------

class _HopPhases:
    """Wall time of PairCompressed's stages, accumulated by wrapping hop's
    stage functions while active: partition = cluster trees + block
    partition; dense_leaves = the batched leaf kernel, once per material
    view; aca = ``_compress_all``; views = ``_view`` minus the leaves and
    minus any transient re-compression inside it. Restores the
    originals on exit; a stage function that no longer exists is skipped."""

    def __init__(self):
        self.t = {"partition": 0.0, "aca": 0.0, "dense_leaves": 0.0,
                  "views": 0.0}
        self._orig: list = []

    def _wrap(self, owner, name, fn):
        orig = getattr(owner, name, None)
        if orig is None:
            return
        self._orig.append((owner, name, orig))
        setattr(owner, name, fn(orig))

    def __enter__(self):
        t = self.t

        def timed(key):
            def deco(orig):
                def w(*a, **k):
                    t0 = time.perf_counter()
                    try:
                        return orig(*a, **k)
                    finally:
                        t[key] += time.perf_counter() - t0
                return w
            return deco

        def minus_nested(key, nested):
            def deco(orig):
                def w(*a, **k):
                    before = sum(t[n] for n in nested)
                    t0 = time.perf_counter()
                    try:
                        return orig(*a, **k)
                    finally:
                        wall = time.perf_counter() - t0
                        t[key] += wall - (sum(t[n] for n in nested) - before)
                return w
            return deco

        self._wrap(hop, "build_cluster_tree", timed("partition"))
        self._wrap(hop, "build_partition", timed("partition"))
        self._wrap(hop.PairCompressed, "_dense_leaves", timed("dense_leaves"))
        self._wrap(hop.PairCompressed, "_compress_all",
                   minus_nested("aca", ("dense_leaves",)))
        self._wrap(hop.PairCompressed, "_view",
                   minus_nested("views", ("aca", "dense_leaves")))
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._orig):
            setattr(owner, name, orig)
        return False


# ---------------------------------------------------------------------
# operator statistics
# ---------------------------------------------------------------------

def operator_stats(hasm) -> tuple[dict, dict, int]:
    """(bytes, ranks, fallbacks) of an AssembledH from its pairs: near =
    the dense blocks and lowrank = the factors of every cached material
    view (both per material, never per basis); bases = the shared
    subspaces still held (storage="basis"), whose rank against the
    summed per-basis ranks they replace is ``joint_over_summed``. Index
    and pointer arrays (~1 % of a view) are not counted."""
    near = lowrank = bases = 0
    joint = summed = 0
    ranks: list = []
    n_lowrank = n_dense = fallbacks = capped = retried = 0
    certified = 0.0
    for pair in hasm._pairs.values():
        n_lowrank += pair.n_lowrank
        n_dense += pair.n_dense
        fallbacks += pair.n_fallback
        capped += getattr(pair, "n_capped", 0)
        retried += getattr(pair, "n_retry", 0)
        certified = max(certified, getattr(pair, "max_verified_err", 0.0))
        if pair.blocks is not None:
            for _, _, payload in pair.blocks:
                if payload is not None:
                    bases += payload.nbytes()
                    ranks.append(payload.rank)
                    joint += payload.rank
                    summed += sum(payload.basis_ranks)
        for view in pair._views.values():
            near += view.dense_nbytes()
            lowrank += view.lowrank_nbytes()
            if pair.blocks is None:
                ranks.extend(int(k) for k in view.ranks)
    n = hasm.layout.n_unknowns
    total = near + lowrank + bases
    b = {"near": near, "lowrank": lowrank, "bases": bases, "total": total,
         "per_unknown": total / n}
    r = {"mean": float(np.mean(ranks)) if ranks else 0.0,
         "max": int(max(ranks)) if ranks else 0,
         "n_lowrank": n_lowrank, "n_dense": n_dense, "n_capped": capped,
         "n_retried": retried, "certified_error": certified,
         # storage="basis" only: the shared subspace's rank against the
         # summed per-basis ranks it replaces (aca.shared_subspace).
         "joint_over_summed": joint / summed if summed else 0.0}
    return b, r, fallbacks


def test_vectors(layout, rng) -> list:
    """BENCH_RANDOM_VECTORS seeded Gaussian vectors + the unit x-translation
    on the u-slots (the calibrated identity)."""
    n = layout.n_unknowns
    vecs = [rng.standard_normal(n) for _ in range(defaults.BENCH_RANDOM_VECTORS)]
    Z = translation_basis(layout)
    vecs.append(Z[:, 0] / np.max(np.abs(Z[:, 0])))
    return vecs


def operator_error(hasm, A_dense, rng) -> dict:
    """max-norm relative error of A_h v per test vector, against the
    dense operator when given, else against BENCH_EXACT_ROWS exact rows."""
    layout = hasm.layout
    vecs = test_vectors(layout, rng)
    errs = []
    if A_dense is not None:
        for v in vecs:
            ref = A_dense @ v
            errs.append(float(np.max(np.abs(hasm.matvec(v) - ref))
                              / np.max(np.abs(ref))))
        method, n_rows = "dense", layout.n_unknowns
    else:
        n_rows = min(defaults.BENCH_EXACT_ROWS, layout.n_unknowns)
        rows = np.sort(rng.choice(layout.n_unknowns, n_rows, replace=False))
        for v in vecs:
            ref = hasm.matvec_exact_rows(v, rows)
            errs.append(float(np.max(np.abs(hasm.matvec(v)[rows] - ref))
                              / np.max(np.abs(ref))))
        method = "exact_rows"
    return {"error": max(errs), "per_vector": errs, "method": method,
            "n_rows": int(n_rows)}


def solution_error(sol, ref) -> dict:
    per_slot = {k: float(np.max(np.abs(sol[k] - ref[k]))
                         / np.max(np.abs(ref[k])))
                for k in ref if np.max(np.abs(ref[k])) > 0}
    return {"error": max(per_slot.values()), "per_slot": per_slot,
            "reference": "dense_lu"}


def median_matvec_ms(matvec, n: int, rng) -> float:
    v = rng.standard_normal(n)
    times = []
    for _ in range(defaults.BENCH_MATVEC_REPEATS):
        t0 = time.perf_counter()
        matvec(v)
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


# ---------------------------------------------------------------------
# one rung
# ---------------------------------------------------------------------

def run_rung(model: str, scale: float, backend: str, opts: dict,
             cache_dir: pathlib.Path, mu_inc: float) -> dict:
    rec: dict = {"model": model, "scale": scale, "backend": backend,
                 "load_avg": os.getloadavg()[0], "git": git_hash(),
                 "host": platform.node(), "ram_gb": (total_ram_bytes() or 0) / 1e9,
                 "opts": dict(opts, jump=JUMP, eps=EPS, mu_inc=mu_inc)}
    phases: dict = {}
    rec["phases"] = phases
    rng = np.random.default_rng(0)

    region_model, meshes, mesh_s, cached = build_meshes(model, scale,
                                                        cache_dir, mu_inc)
    phases["mesh"] = mesh_s
    rec["mesh_cached"] = cached
    t0 = time.perf_counter()
    system = generate_system(region_model)
    phases["system"] = time.perf_counter() - t0
    patches = [p for r in region_model.regions for p in r.patches]
    seen: dict = {}
    for p in patches:
        seen[id(p)] = p
    rec["n_tris"] = int(sum(p.n_triangles for p in seen.values()))
    rec["n_fault_tris"] = int(sum(f.n_triangles for r in region_model.regions
                                  for f in r.faults))
    rec["tris_per_patch"] = {p.name: int(p.n_triangles) for p in seen.values()}
    n = system.layout.n_unknowns
    rec["n_unknowns"] = int(n)
    dense_ok = n <= defaults.BENCH_DENSE_MAX_UNKNOWNS
    print(f"  [{model} x{scale:g} {backend}] {rec['n_tris']} tris, "
          f"{n} unknowns, load {rec['load_avg']:.2f}", flush=True)

    if backend == "dense":
        if not dense_ok:
            rec["skipped"] = (f"{n} unknowns > BENCH_DENSE_MAX_UNKNOWNS "
                              f"{defaults.BENCH_DENSE_MAX_UNKNOWNS}")
            return rec
        t0 = time.perf_counter()
        dense = AssembledDense(system, EPS, "direct", jump=JUMP)
        phases["assemble"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sol = dense.solve()
        phases["lu"] = time.perf_counter() - t0
        phases["build"] = phases["assemble"] + phases["lu"]
        x = np.concatenate([sol[s.name].ravel() for s in system.layout.slots])
        rec["true_relres"] = float(np.linalg.norm(dense.A @ x - dense.b)
                                   / np.linalg.norm(dense.b))
        rec["cond_estimate"] = float(dense.report.cond_estimate)
        phases["matvec_ms"] = median_matvec_ms(lambda v: dense.A @ v, n, rng)
        rec["iterations"] = None
        rec["converged"] = rec["true_relres"] < defaults.GMRES_RTOL
        rec["bytes"] = {"total": int(dense.A.nbytes), "per_unknown": 8.0 * n}
        rec["peak_rss_gb"] = peak_rss_gb()
        rec["numba"] = numba_threads()
        rec["defaults"] = defaults_snapshot()
        return rec

    # ---- compressed backend ----
    hb_kwargs = dict(tol=opts["tol"], min_leaf=opts["leaf"], eta=opts["eta"])
    if opts.get("storage"):
        hb_kwargs["storage"] = opts["storage"]
    hb = HBackend(jump=JUMP, **hb_kwargs)
    rec["opts"]["storage"] = hb.storage
    t0 = time.perf_counter()
    with _HopPhases() as ph:
        hasm = hb.assemble(system, EPS)
    phases["build"] = time.perf_counter() - t0
    phases.update(ph.t)
    phases["other"] = phases["build"] - sum(ph.t.values())

    t0 = time.perf_counter()
    sol = hasm.solve()
    wall = time.perf_counter() - t0
    report = hasm.report
    phases["precond"] = report.precond_summary["build_s"]
    phases["solve"] = wall - phases["precond"]
    rec["iterations"] = int(report.iterations)
    rec["converged"] = bool(report.converged)
    rec["stagnated"] = bool(report.stagnated)
    rec["true_relres"] = float(report.true_relres)
    rec["precond"] = report.precond_summary
    rec["rungs"] = {"+".join(sb["slots"]): sb["rung"]
                    for sb in report.precond_summary["super_blocks"]}
    phases["matvec_ms"] = median_matvec_ms(hasm.matvec, n, rng)
    rec["bytes"], rec["ranks"], rec["fallbacks"] = operator_stats(hasm)
    rec["peak_rss_gb"] = peak_rss_gb()      # before any dense reference
    rec["numba"] = numba_threads()
    rec["defaults"] = defaults_snapshot()

    # ---- accuracy references ----
    t0 = time.perf_counter()
    if dense_ok:
        dense = AssembledDense(system, EPS, "direct", jump=JUMP)
        rec["operator"] = operator_error(hasm, dense.A, rng)
        rec["solution"] = solution_error(sol, dense.solve())
        del dense
    else:
        rec["operator"] = operator_error(hasm, None, rng)
        rec["solution"] = None
    rec["reference_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------------
# child-process protocol
# ---------------------------------------------------------------------

def child_argv(args, model, scale, backend, out_path) -> list:
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()),
            "--_child", "--_out", str(out_path),
            "--model", model, "--scale", f"{scale:g}", "--backend", backend,
            "--eta", f"{args.eta:g}", "--leaf", str(args.leaf),
            "--tol", f"{args.tol:g}", "--mesh-cache", str(args.mesh_cache),
            "--mu-inc", f"{args.mu_inc:g}"]
    if args.storage:
        argv += ["--storage", args.storage]
    return argv


def run_child(args, model, scale, backend) -> dict:
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="bench_rung_")
    os.close(fd)
    out_path = pathlib.Path(out_path)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(child_argv(args, model, scale, backend, out_path),
                              cwd=ROOT, timeout=args.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"model": model, "scale": scale, "backend": backend,
                "timed_out": True, "timeout_s": args.timeout,
                "wall_s": time.perf_counter() - t0}
    finally:
        rec = None
        if out_path.exists():
            text = out_path.read_text()
            rec = json.loads(text) if text.strip() else None
            out_path.unlink()
    if rec is None:
        rec = {"model": model, "scale": scale, "backend": backend,
               "error": f"child exited {rc} without a record"}
    rec["wall_s"] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------------
# reporting and the gate
# ---------------------------------------------------------------------

TABLE = ("{model:>14} {scale:>5} {backend:>5} {tris:>7} {unk:>7} {build:>8} "
         "{mv:>8} {it:>4} {bpu:>7} {rss:>6} {op:>8} {sol:>8}")


def table_header() -> str:
    return TABLE.format(model="model", scale="scale", backend="bk",
                        tris="tris", unk="unk", build="build(s)",
                        mv="mv(ms)", it="it", bpu="B/unk", rss="RSS", op="op_err",
                        sol="sol_err")


def table_row(rec: dict) -> str:
    if rec.get("timed_out"):
        return (f"{rec['model']:>14} {rec['scale']:>5g} {rec['backend']:>5} "
                f"TIMED OUT after {rec['timeout_s']:.0f} s")
    if rec.get("error"):
        return (f"{rec['model']:>14} {rec['scale']:>5g} {rec['backend']:>5} "
                f"ERROR: {rec['error'].strip().splitlines()[-1]}")
    if rec.get("skipped"):
        return (f"{rec['model']:>14} {rec['scale']:>5g} {rec['backend']:>5} "
                f"{rec['n_tris']:>7} {rec['n_unknowns']:>7} skipped: "
                f"{rec['skipped']}")
    ph = rec["phases"]
    it = rec.get("iterations")
    op = (rec.get("operator") or {}).get("error")
    sol = (rec.get("solution") or {}).get("error")
    return TABLE.format(
        model=rec["model"], scale=f"{rec['scale']:g}", backend=rec["backend"],
        tris=rec["n_tris"], unk=rec["n_unknowns"], build=f"{ph['build']:.1f}",
        mv=f"{ph['matvec_ms']:.1f}", it="-" if it is None else it,
        bpu=f"{rec['bytes']['per_unknown']:.0f}", rss=f"{rec['peak_rss_gb']:.2f}",
        op="-" if op is None else f"{op:.1e}",
        sol="-" if sol is None else f"{sol:.1e}")


def compact(rec: dict) -> dict:
    """The gate's view of a record: what ``bench-json:`` carries."""
    out = {k: rec.get(k) for k in ("model", "scale", "backend")}
    for k in ("timed_out", "error", "skipped"):
        if rec.get(k):
            out[k] = True
            return out
    out["n_unknowns"] = rec["n_unknowns"]
    out["phases"] = {k: round(v, 3) for k, v in rec["phases"].items()}
    out["iterations"] = rec.get("iterations")
    out["fallbacks"] = rec.get("fallbacks")
    out["peak_rss_gb"] = round(rec["peak_rss_gb"], 3)
    out["op_error"] = (rec.get("operator") or {}).get("error")
    out["sol_error"] = (rec.get("solution") or {}).get("error")
    out["converged"] = rec.get("converged")
    out["stagnated"] = rec.get("stagnated", False)
    out["git"] = rec.get("git")
    return out


def baseline_from_commit(rev: str) -> list:
    body = subprocess.run(["git", "log", "-1", "--format=%B", rev], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    for line in body.splitlines():
        if line.startswith("bench-json:"):
            return json.loads(line[len("bench-json:"):].strip())
    raise SystemExit(f"FAIL: no 'bench-json:' line in the commit message of {rev}")


def gate(records: list, baseline: list) -> list:
    """Reasons the current run fails against the baseline (empty = PASS)."""
    ram = total_ram_bytes()
    base = {(b["model"], b["scale"], b["backend"]): b for b in baseline}
    fails = []
    for rec in map(compact, records):
        tag = f"{rec['model']} x{rec['scale']:g} {rec['backend']}"
        if rec.get("timed_out") or rec.get("error"):
            fails.append(f"{tag}: did not finish")
            continue
        if rec.get("skipped"):
            continue
        if rec["backend"] == "hmat":
            if rec["op_error"] is None or \
                    rec["op_error"] > defaults.BENCH_OPERATOR_ERROR_MAX:
                fails.append(f"{tag}: operator error {rec['op_error']} > "
                             f"{defaults.BENCH_OPERATOR_ERROR_MAX:g}")
            if rec["sol_error"] is not None and \
                    rec["sol_error"] > defaults.BENCH_SOLUTION_ERROR_MAX:
                fails.append(f"{tag}: solution error {rec['sol_error']:.2e} > "
                             f"{defaults.BENCH_SOLUTION_ERROR_MAX:g}")
            if not rec["converged"] or rec["stagnated"]:
                fails.append(f"{tag}: not converged")
        if ram and rec["peak_rss_gb"] * 1e9 > defaults.BENCH_RSS_RAM_FRACTION_MAX * ram:
            fails.append(f"{tag}: RSS {rec['peak_rss_gb']:.1f} GB > "
                         f"{defaults.BENCH_RSS_RAM_FRACTION_MAX:g} x RAM")
        b = base.get((rec["model"], rec["scale"], rec["backend"]))
        if b is None or b.get("timed_out") or b.get("error") or b.get("skipped"):
            continue
        for phase, t in rec["phases"].items():
            t0 = b["phases"].get(phase)
            if t0 is None or t0 < defaults.BENCH_TIME_FLOOR_S:
                continue
            if t > defaults.BENCH_TIME_RATIO_MAX * t0:
                fails.append(f"{tag}: phase {phase} {t:.2f} s > "
                             f"{defaults.BENCH_TIME_RATIO_MAX:g} x {t0:.2f} s")
        if rec["peak_rss_gb"] > defaults.BENCH_RSS_RATIO_MAX * b["peak_rss_gb"]:
            fails.append(f"{tag}: RSS {rec['peak_rss_gb']:.2f} GB > "
                         f"{defaults.BENCH_RSS_RATIO_MAX:g} x {b['peak_rss_gb']:.2f}")
        if rec["iterations"] is not None and b["iterations"] is not None and \
                rec["iterations"] > b["iterations"] + defaults.BENCH_ITER_SLACK:
            fails.append(f"{tag}: iterations {rec['iterations']} > "
                         f"{b['iterations']} + {defaults.BENCH_ITER_SLACK}")
        if rec["fallbacks"] is not None and b["fallbacks"] is not None and \
                rec["fallbacks"] > b["fallbacks"]:
            fails.append(f"{tag}: fallbacks {rec['fallbacks']} > {b['fallbacks']}")
    return fails


# ---------------------------------------------------------------------
# the single-pair compression probe
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=MODELS, default="fault_box")
    ap.add_argument("--scale", type=float, nargs="+", default=None,
                    help="mesh scales (every edge / scale); default "
                         "1 1.6 2.6 for fault_box, 1 for topo_inclusion")
    ap.add_argument("--backend", choices=BACKENDS, nargs="+",
                    default=["hmat"])
    ap.add_argument("--eta", type=float, default=defaults.ADMISSIBILITY_ETA)
    ap.add_argument("--leaf", type=int, default=defaults.CLUSTER_MIN_LEAF)
    ap.add_argument("--tol", type=float, default=defaults.BLOCK_COMPRESSION_TOL)
    ap.add_argument("--storage", choices=("basis", "combined"), default=None,
                    help="HBackend storage (default: HBackend's own)")
    ap.add_argument("--mu-inc", type=float, default=3.0,
                    help="topo_inclusion: inclusion shear modulus, GPa (host 30)")
    ap.add_argument("--mesh-cache", type=pathlib.Path,
                    default=DEFAULT_MESH_CACHE,
                    help="npz mesh cache directory, outside the tree")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds per rung; a rung past it is recorded as timed out")
    ap.add_argument("--json", type=str, default=None,
                    help="write the full records here")
    ap.add_argument("--gate", type=str, default=None, metavar="REV",
                    help="compare with the bench-json: line of REV's commit "
                         "message; PASS:/FAIL:, exit 1 on FAIL")
    ap.add_argument("--panel", type=int, default=None,
                    help="side count for the large-panel compression "
                         "primitive (2*panel^2 triangles per mesh); "
                         "skips the ladders")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_out", type=pathlib.Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    scales = args.scale or DEFAULT_SCALES[args.model]
    opts = dict(eta=args.eta, leaf=args.leaf, tol=args.tol,
                storage=args.storage)

    if args._child:
        try:
            rec = run_rung(args.model, scales[0], args.backend[0], opts,
                           args.mesh_cache, args.mu_inc)
        except Exception:                        # the parent reports it
            rec = {"model": args.model, "scale": scales[0],
                   "backend": args.backend[0], "error": traceback.format_exc()}
        args._out.write_text(json.dumps(rec))
        return

    if args.panel:
        records = [bench_panel(args.panel, args.storage or "combined", args.tol)]
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(records, indent=1))
        return

    baseline = baseline_from_commit(args.gate) if args.gate else None
    records = []
    print(table_header(), flush=True)
    for scale in scales:
        for backend in args.backend:
            tag = f"{args.model} x{scale:g} {backend}"
            if args.gate:
                load = wait_for_load(tag)
                if load > defaults.BENCH_LOAD_MAX:
                    print(f"FAIL: bench gate refused: load average "
                          f"{load:.2f} > {defaults.BENCH_LOAD_MAX:g} before "
                          f"{tag}, after waiting "
                          f"{defaults.BENCH_LOAD_WAIT_S:.0f} s")
                    sys.exit(1)
            rec = run_child(args, args.model, scale, backend)
            records.append(rec)
            print(table_row(rec), flush=True)

    if args.json:
        out = pathlib.Path(args.json)
        out.write_text(json.dumps(records, indent=1))
        print(f"\nwrote {out}")
    print("\nbench-json: " + json.dumps([compact(r) for r in records],
                                        separators=(",", ":")))

    if baseline is not None:
        fails = gate(records, baseline)
        for f in fails:
            print(f"  {f}")
        if fails:
            print(f"FAIL: bench vs {args.gate}: {len(fails)} breach(es)")
            sys.exit(1)
        print(f"PASS: bench vs {args.gate}: {len(records)} rung(s) within "
              f"the BENCH_* bands")


if __name__ == "__main__":
    main()
