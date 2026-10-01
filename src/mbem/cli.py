"""``python -m mbem <verb>``: run a config, list runs, show one, run the gates.

    mbem run configs/fault_box.py [--set scale=1.6] [--dry-run]
    mbem list [-n 20]
    mbem show <run-dir> [--section spec|effective|model|env|defaults|report]
    mbem verify [-k fmm] [--fast]
    mbem publish <run-dir> [--figure KEY ...]

``python -m mbem`` rather than a console script: an entry point would have to be
reinstalled whenever it moved, and this works the moment ``pip install -e .``
has run.

``verify`` DELEGATES to ``tests/run_all.py`` and creates no run directory. The
gates write nothing into the tree, which is a property worth keeping, so the
verb is one entry point and not a second implementation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

from mbem import config as cfg
from mbem import provenance as prov
from mbem import runner


def _coerce(text: str, kind: str):
    """``--set k=v`` typed by the run_spec annotation, else parsed as JSON."""
    if kind == "float":
        return float(text)
    if kind == "int":
        return int(text)
    if kind == "bool":
        if text.lower() in ("true", "1", "yes"):
            return True
        if text.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"{text!r} is not a bool")
    if kind == "str":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def cmd_run(a) -> int:
    path = pathlib.Path(a.config)
    accepted = cfg.parameters(path)
    params = {}
    for item in a.set or []:
        if "=" not in item:
            print(f"--set needs KEY=VALUE, got {item!r}")
            return 2
        k, _, v = item.partition("=")
        if accepted and k not in accepted:
            print(f"{path.name} does not accept {k!r}; it accepts "
                  f"{sorted(accepted)}")
            return 2
        params[k] = _coerce(v, accepted.get(k, ("any", None))[0])

    run = cfg.load(path, **params)
    if a.dry_run:
        # Validate and print, touching no mesh and no backend: the common
        # iteration is "did I state this right", and it should cost nothing.
        print(json.dumps(prov.json_safe({"name": run.name,
                                         "spec": run, "params": params}),
                         indent=1, sort_keys=True))
        return 0

    run_dir = pathlib.Path(a.out) if a.out else runner.new_run_dir(
        run.name, pathlib.Path(a.runs_dir) if a.runs_dir else None)
    if a.out:
        run_dir.mkdir(parents=True, exist_ok=True)
        if any(run_dir.iterdir()):
            print(f"--out {run_dir} is not empty; refusing to overwrite a run")
            return 2
    print(f"run {run_dir.name}")
    try:
        rep = runner.execute(run, run_dir, path)
    except Exception:
        (run_dir / "STATUS").write_text("FAIL\n")
        raise
    for st in rep["states"]:
        extra = (f"{st['iterations']} iters, relres {st['true_relres']:.2e}"
                 if "iterations" in st else
                 f"cond {st.get('cond_estimate', float('nan')):.2e}")
        print(f"  {st['label']:10s} {extra}  ({st['solve_s']:.1f} s)")
    print(f"  {rep['status']}  {rep['wall_s']:.1f} s total, "
          f"peak RSS {rep['memory']['peak_rss_gb']:.2f} GB")
    print(f"  -> {run_dir}")
    return 0 if rep["status"] == "OK" else 1


def _runs(runs_dir: pathlib.Path):
    if not runs_dir.exists():
        return []
    return sorted((d for d in runs_dir.iterdir() if d.is_dir()), reverse=True)


def cmd_list(a) -> int:
    rows = _runs(pathlib.Path(a.runs_dir) if a.runs_dir else runner.RUNS)
    if not rows:
        print("no runs yet")
        return 0
    for d in rows[:a.n]:
        status = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() \
            else "?"
        bits = ""
        rp = d / "report.json"
        if rp.exists():
            try:
                rep = json.loads(rp.read_text())
                st = rep.get("states", [{}])[0]
                bits = (f"{rep.get('wall_s', 0):7.1f} s  "
                        f"{st.get('iterations', '-'):>4} it")
            except json.JSONDecodeError:
                bits = "  (unreadable report)"
        print(f"{status:8s} {d.name:58s} {bits}")
    print(f"{len(rows)} run(s)")
    return 0


def cmd_show(a) -> int:
    d = pathlib.Path(a.run_dir)
    which = {"report": "report.json"}.get(a.section, "resolved.json")
    f = d / which
    if not f.exists():
        print(f"{f} not found")
        return 2
    payload = json.loads(f.read_text())
    if a.section and a.section not in ("report",):
        if a.section not in payload:
            print(f"no section {a.section!r}; have {sorted(payload)}")
            return 2
        payload = payload[a.section]
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_verify(a) -> int:
    argv = [sys.executable, str(prov.REPO / "tests" / "run_all.py")]
    if a.only:
        argv += ["-k", a.only]
    if a.fast:
        argv += ["--fast"]
    return subprocess.run(argv, cwd=prov.REPO).returncode


def cmd_publish(a) -> int:
    """Copy chosen figures into the tracked gallery, with provenance.

    Deliberate, because the alternative is what this repo did before: every
    demo overwrote one tracked PNG in place, so the committed image was whatever
    ran last and nothing recorded which code produced it.
    """
    d = pathlib.Path(a.run_dir)
    gallery = prov.REPO / "docs" / "figures"
    gallery.mkdir(parents=True, exist_ok=True)
    figs = sorted(p for p in d.glob("fig_*") if p.suffix in (".png", ".pdf"))
    if a.figure:
        keys = set(a.figure)
        figs = [p for p in figs if p.stem.removeprefix("fig_") in keys]
    if not figs:
        print(f"no figures to publish in {d}")
        return 2
    git = "unknown"
    rj = d / "resolved.json"
    if rj.exists():
        git = json.loads(rj.read_text()).get("env", {}).get("git", "unknown")
    import hashlib
    prov_file = gallery / "PROVENANCE.tsv"
    lines = []
    for p in figs:
        shutil.copy2(p, gallery / p.name)
        lines.append(f"{p.stem.removeprefix('fig_')}\t{d.name}\t{git}\t"
                     f"{hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
        print(f"  published {p.name}")
    with prov_file.open("a") as fh:
        if prov_file.stat().st_size == 0:
            fh.write("figure\trun_id\tgit\tsha256_16\n")
        fh.write("\n".join(lines) + "\n")
    print(f"  provenance appended to {prov_file.relative_to(prov.REPO)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mbem", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("run", help="run a config into a new run directory")
    p.add_argument("config")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--out", default=None, help="explicit run directory")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="validate and print the spec; build nothing")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("list", help="runs, newest first")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--runs-dir", default=None)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="one run's resolved spec or report")
    p.add_argument("run_dir")
    p.add_argument("--section", default=None)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("verify", help="run the gates (writes nothing)")
    p.add_argument("-k", "--only", default=None)
    p.add_argument("--fast", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("publish", help="copy a run's figures into docs/figures")
    p.add_argument("run_dir")
    p.add_argument("--figure", action="append")
    p.set_defaults(fn=cmd_publish)

    a = ap.parse_args(argv)
    return a.fn(a)
