"""Execute a ``Run`` into a self-describing directory.

    runs/20261001T163007Z-fault_box-01ed703-7b1e/
        STATUS            one word: RUNNING | OK | FAIL
        config.py         verbatim copy of the config that was loaded
        resolved.json     what it took: spec, effective kwargs, defaults, env
        report.json       what it got: iterations, residuals, timings, memory
        fields_<state>.npz
        MANIFEST          sha256 and size per file, written LAST

WHY JSON AND NOT TOML, which is a deviation worth stating. ``None`` is the
load-bearing value in a resolved spec: it means "this knob was not set, so the
callee's own default applied", and the callee's default IS the number in
``defaults``. TOML has no null, so the only encoding is to omit the key -- which
makes "not set" indistinguishable from "not in the schema", exactly the
distinction the override design turns on. JSON has null natively. (3.13 reads
TOML in the stdlib but cannot write it, so TOML would also have meant
hand-rolling a writer for the most provenance-critical file in the tree, where a
quiet float or escaping bug is worse than no dump at all.) ``show --format
toml`` can render it on demand; one source of truth, two views.

STATUS is a separate one-word file so an interrupted run is identifiable
without parsing JSON, and MANIFEST is written last so its presence means the
artifact set is complete.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import pathlib
import re
import secrets
import shutil
import time

import numpy as np

from mbem import config as cfg
from mbem import provenance as prov

RUNS = prov.REPO / "runs"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-") or "x"


def new_run_dir(name: str, runs_dir: pathlib.Path | None = None) -> pathlib.Path:
    """``<UTC>-<name>-<git>-<4 hex>``, created exclusively.

    UTC with no colons: legal on every filesystem, pastes into a shell
    unquoted, and sorts lexicographically = chronologically. ``mkdir`` without
    ``exist_ok`` IS the collision check -- atomic, no lock, no retry logic.
    """
    base = runs_dir or RUNS
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(4):
        d = base / f"{stamp}-{_slug(name)}-{_slug(prov.git_hash())}-{secrets.token_hex(2)}"
        try:
            d.mkdir(parents=True)
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"could not mint a run directory under {base}")


def _effective_backend(hb) -> dict:
    """The kwargs an HBackend actually holds, read off the object.

    By introspection, never a hand-written list: a new HBackend keyword is
    recorded the day it is added. bench_scaling's equivalent is hand-listed and
    already omits precision, n_workers, sweep and fmm_order.
    """
    import inspect
    out = {}
    for name in inspect.signature(type(hb).__init__).parameters:
        if name == "self":
            continue
        if hasattr(hb, name):
            out[name] = prov.json_safe(getattr(hb, name))
        elif isinstance(getattr(hb, "opts", None), dict) and name in hb.opts:
            out[name] = prov.json_safe(hb.opts[name])
    return out


def _effective_eps(model, eps) -> dict:
    """Resolved eps per patch. ``"auto"`` is not a number, and recording only
    the word would make the dump a lie the moment EPS_OVER_H moved."""
    from mbem.kernels import basis as kb
    out = {}
    for region in model.regions:
        for p in list(region.patches) + list(region.faults):
            try:
                a = np.atleast_1d(np.asarray(kb.resolve_patch_eps(eps, p), float))
            except Exception as exc:                          # noqa: BLE001
                out[p.name] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            out[p.name] = {"n": int(a.size), "min": float(a.min()),
                           "max": float(a.max()), "mean": float(a.mean())}
    return out


def _model_fingerprint(model, system) -> dict:
    """What was actually built, read back off the model.

    The config names a builder rather than describing patches, so this is how a
    run records the model it got: names, BCs, orders, triangle counts, the
    INFERRED orientation sigma, and a sha256 per mesh. The mesh hash is the
    join key that proves two runs used the same geometry however their builder
    parameters were spelled.
    """
    regions = []
    for r in model.regions:
        patches = []
        for p in list(r.patches) + list(r.faults):
            v, t = p.mesh.vertices, p.mesh.triangles
            patches.append({
                "name": p.name, "bc": str(p.bc).split(".")[-1],
                "order": int(p.order), "n_triangles": int(p.n_triangles),
                "sigma": float(model.orientation(r, p))
                         if p in r.patches else None,
                "mesh_sha256": hashlib.sha256(
                    np.ascontiguousarray(v).tobytes()
                    + np.ascontiguousarray(t).tobytes()).hexdigest()[:16]})
        regions.append({"name": r.name,
                        "material": {"mu": r.material.mu, "lam": r.material.lam},
                        "patches": patches})
    return {"regions": regions,
            "n_unknowns": int(system.layout.n_unknowns),
            "n_tris": int(sum(p["n_triangles"] for r in regions
                              for p in r["patches"])),
            "slots": [s.name for s in system.layout.slots]}


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(prov.json_safe(payload), indent=1,
                               sort_keys=True, allow_nan=False) + "\n")


def _manifest(run_dir: pathlib.Path) -> None:
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST":
            b = p.read_bytes()
            lines.append(f"{hashlib.sha256(b).hexdigest()}  {len(b):>12d}  "
                         f"{p.relative_to(run_dir)}")
    (run_dir / "MANIFEST").write_text("\n".join(lines) + "\n")


def execute(run: cfg.Run, run_dir: pathlib.Path,
            config_path: pathlib.Path | None = None,
            save: bool = True) -> dict:
    """Build, solve every state, and write the run directory. Returns report."""
    from mbem.model import generate_system

    (run_dir / "STATUS").write_text("RUNNING\n")
    if config_path is not None:
        shutil.copy2(config_path, run_dir / "config.py")
    phases: dict = {}
    report: dict = {"name": run.name, "phases": phases, "states": []}
    t_run = time.perf_counter()

    t0 = time.perf_counter()
    mesh_fn = cfg.resolve(run.model.geometry.builder, cfg.MESH_BUILDERS)
    bundle = mesh_fn(scale=run.model.geometry.scale,
                     **dict(run.model.geometry.params))
    phases["mesh"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model_fn = cfg.resolve(run.model.builder, cfg.MODEL_BUILDERS)
    model = model_fn(bundle, **dict(run.model.params))
    system = generate_system(model)
    phases["system"] = time.perf_counter() - t0

    known = {r.name for r in model.regions}
    for st in run.states:
        unknown = set(st.materials) - known
        if unknown:
            raise KeyError(f"state {st.label!r} names regions not in the model: "
                           f"{sorted(unknown)} (model has {sorted(known)})")

    eps = run.model.eps
    t0 = time.perf_counter()
    if run.backend.kind == "dense":
        from mbem.backends.dense import AssembledDense
        asm = AssembledDense(system, eps, run.backend.dense_mode,
                             jump=run.backend.jump, deflate=run.backend.deflate)
        effective_backend = {"kind": "dense", "mode": run.backend.dense_mode,
                             "jump": run.backend.jump,
                             "deflate": run.backend.deflate}
    else:
        from mbem.backends import HBackend
        hb = HBackend(**run.backend.kwargs())
        asm = hb.assemble(system, eps)
        effective_backend = dict(_effective_backend(hb), kind=run.backend.kind)
    phases["build"] = time.perf_counter() - t0
    phases.update({k: v for k, v in getattr(asm, "far_phases", {}).items()})

    resolved = {
        "schema": 1,
        "run": {"id": run_dir.name, "name": run.name,
                "started_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(timespec="seconds"),
                "config_path": str(config_path) if config_path else None,
                "notes": run.notes, "tags": list(run.tags)},
        "spec": dataclasses.asdict(run),
        "builders": {"mesh": cfg.builder_name(run.model.geometry.builder),
                     "model": cfg.builder_name(run.model.builder)},
        "effective": {"backend": effective_backend,
                      "solve": run.solve.kwargs(),
                      "eps": _effective_eps(model, eps)},
        "model": _model_fingerprint(model, system),
        "env": prov.environment(),
        "defaults": prov.defaults_snapshot(),
    }
    if save:
        _write_json(run_dir / "resolved.json", resolved)

    cur, ok = asm, True
    for st in run.states:
        if st.materials:
            cur = cur.rebuild_for_materials(
                {k: m.build() for k, m in st.materials.items()})
        t0 = time.perf_counter()
        sol = cur.solve(**run.solve.kwargs()) if run.backend.kind != "dense" \
            else cur.solve()
        wall = time.perf_counter() - t0
        rep = cur.report
        row = {"label": st.label, "solve_s": wall,
               "rss_after_gb": prov.peak_rss_gb()}
        for attr in ("iterations", "converged", "stagnated", "true_relres",
                     "cond_estimate"):
            if getattr(rep, attr, None) is not None:
                row[attr] = prov.json_safe(getattr(rep, attr))
        if getattr(rep, "precond_summary", None):
            row["precond"] = prov.json_safe(rep.precond_summary)
        ok &= bool(getattr(rep, "converged", True))
        report["states"].append(row)

        if save and run.outputs.save_fields:
            want = run.outputs.slots or tuple(sol)
            missing = [s for s in want if s not in sol]
            if missing:
                raise KeyError(f"outputs.slots names slots the solution does "
                               f"not have: {missing} (have {sorted(sol)})")
            np.savez(run_dir / f"fields_{_slug(st.label)}.npz",
                     **{k: sol[k] for k in want})

    report["wall_s"] = time.perf_counter() - t_run
    report["memory"] = {"peak_rss_gb": prov.peak_rss_gb()}
    report["status"] = "OK" if ok else "FAIL"
    if save:
        _write_json(run_dir / "report.json", report)
        (run_dir / "STATUS").write_text(("OK" if ok else "FAIL") + "\n")
        _manifest(run_dir)
    return report
