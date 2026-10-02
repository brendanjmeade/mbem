"""Gate: the config layer cannot diverge from the code it drives.

A config layer's characteristic failure is not a crash, it is a SECOND
statement of something the codebase states once. This repo keeps the fault sign
in one place, the eps rule in one place and every tolerance in
``defaults.py``, because the second statement is the one that goes wrong -- and
a slip-pairing sign error is invisible at nu = 1/4. So:

  a  the model a CONFIG builds is the model the GATES build, entrywise. Not
     "similar": same patch names, BCs, orders, inferred sigma, unknown count,
     and the same fault Burgers vector array. This is why a config names a
     builder instead of describing patches.
  b  a run does not mutate ``defaults``. The backends bind their defaults from
     it at def time, so a config that appeared to override one by rebinding
     would be a silent lie; overriding means passing a keyword, and a field
     left None must leave no trace.
  c  the resolved dump ROUND-TRIPS back to the same Run, with None surviving.
     None means "unset, so defaults applied", which is the distinction the
     whole override design rests on -- and the reason the dump is JSON, since
     TOML has no null and could only omit the key.
  d  ``effective.backend`` covers every HBackend keyword, by introspection.
     A dump that silently stops recording a new knob is worse than no dump.
  e  gate-only acceptance criteria are NOT reachable from a config. A run that
     could move BENCH_* or *_PARITY* could declare its own success.
  f  a run directory is complete and self-describing: STATUS, MANIFEST whose
     hashes match, and the fields npz.

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import tempfile

import numpy as np

from mbem import ElasticMaterial, config as cfg, defaults, provenance, runner
from mbem.cases.fault_box import build_fault_box, build_model
from mbem.cases.registry import fault_box_meshes, fault_box_model
from mbem.model import generate_system

CHECKS: list = []
CONFIGS = provenance.REPO / "configs"


def check(name, ok, extra="") -> bool:
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:62s} {extra}")
    return ok


def a_same_model() -> None:
    """[a] config builder == the builder the gates call, entrywise."""
    print("\n[a] the config path builds the gates' model")
    direct = build_model(
        build_fault_box(half_x=100.0, z_bottom=-60.0, fault_half_len=30.0,
                        fault_depth=18.0, near_field_radius=50.0,
                        edge_fault=4.0, edge_near=25.0, edge_far=50.0,
                        edge_side=50.0),
        0.01, ElasticMaterial(mu=30.0, lam=30.0))
    via = fault_box_model(fault_box_meshes())

    sd, sv = generate_system(direct), generate_system(via)
    check("same unknown count", sd.layout.n_unknowns == sv.layout.n_unknowns,
          f"{sd.layout.n_unknowns}")
    check("same slot names",
          [s.name for s in sd.layout.slots] == [s.name for s in sv.layout.slots])

    def describe(m):
        return [(r.name, r.material.mu, r.material.lam,
                 tuple((p.name, str(p.bc), p.order, p.n_triangles,
                        float(m.orientation(r, p))) for p in r.patches),
                 tuple((f.name, f.order, f.n_triangles) for f in r.faults))
                for r in m.regions]
    check("same regions, patches, BCs, orders and inferred sigma",
          describe(direct) == describe(via))

    fd = [f for r in direct.regions for f in r.faults]
    fv = [f for r in via.regions for f in r.faults]
    check("same number of faults", len(fd) == len(fv), f"{len(fd)}")
    same = len(fd) == len(fv) and all(
        np.array_equal(x.value_array(), y.value_array())
        for x, y in zip(fd, fv))
    # Bitwise, not to a tolerance: this is the Burgers vector, and a tolerance
    # would accept a sign flip on a component that happens to be small.
    check("fault Burgers vector bitwise identical", same)


def b_defaults_untouched(run_dir: pathlib.Path) -> dict:
    """[b] a run leaves defaults exactly as it found them."""
    print("\n[b] a run does not mutate defaults")
    before = {k: v for k, v in vars(defaults).items() if k.isupper()}
    run = cfg.load(CONFIGS / "fault_box.py", backend="hmat")
    report = runner.execute(run, run_dir, CONFIGS / "fault_box.py")
    after = {k: v for k, v in vars(defaults).items() if k.isupper()}
    moved = sorted(k for k in before
                   if not _same(before[k], after.get(k, object())))
    check("no defaults.* changed by the run", not moved, f"{moved[:4]}")
    check("the run converged", report["status"] == "OK",
          f"{report['states'][0].get('iterations')} iters")
    return report


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def c_round_trip(run_dir: pathlib.Path) -> None:
    """[c] resolved.json -> Run -> the same spec, with None preserved."""
    print("\n[c] the resolved dump round-trips, and None survives")
    resolved = json.loads((run_dir / "resolved.json").read_text())
    spec = resolved["spec"]
    unset = [k for k, v in spec["backend"].items() if v is None]
    check("backend fields left unset are present AND null", len(unset) >= 4,
          f"{len(unset)} null: {unset[:4]}")
    check("solve records only what was set",
          set(resolved["effective"]["solve"]) == {"rtol"},
          f"{sorted(resolved['effective']['solve'])}")
    rebuilt = cfg.Run(
        name=spec["name"],
        model=cfg.Model(
            geometry=cfg.Geometry(**spec["model"]["geometry"]),
            builder=spec["model"]["builder"],
            params=spec["model"]["params"], eps=spec["model"]["eps"]),
        backend=cfg.Backend(**spec["backend"]),
        solve=cfg.Solve(**spec["solve"]),
        states=tuple(cfg.State(s["label"],
                               {k: cfg.Material(**m)
                                for k, m in s["materials"].items()})
                     for s in spec["states"]),
        outputs=cfg.Output(slots=tuple(spec["outputs"]["slots"]),
                           save_fields=spec["outputs"]["save_fields"],
                           figures=tuple(spec["outputs"]["figures"])),
        notes=spec["notes"], tags=tuple(spec["tags"]))
    original = cfg.load(CONFIGS / "fault_box.py", backend="hmat")
    check("Run rebuilt from the dump equals the Run that ran",
          provenance.json_safe(rebuilt) == provenance.json_safe(original))
    eps = resolved["effective"]["eps"]
    numeric = all(isinstance(v.get("mean"), float) for v in eps.values())
    check('eps recorded as NUMBERS per patch, not the word "auto"', numeric,
          f"{len(eps)} patch(es)")


def d_effective_covers_backend(run_dir: pathlib.Path) -> None:
    """[d] every HBackend keyword appears in the dump."""
    print("\n[d] effective.backend covers the backend's whole surface")
    import inspect

    from mbem.backends import HBackend
    want = {p for p in inspect.signature(HBackend.__init__).parameters
            if p != "self"}
    got = set(json.loads((run_dir / "resolved.json").read_text()
                         )["effective"]["backend"])
    missing = sorted(want - got)
    check("no HBackend keyword unrecorded", not missing, f"missing {missing}")


def e_gate_criteria_unreachable() -> None:
    """[e] a config cannot name a gate-only acceptance criterion."""
    print("\n[e] gate criteria are not run inputs")
    gate_like = re.compile(r"^(BENCH_|.*_PARITY|GMRES_ITER_|.*_CEILING)")
    criteria = {k for k in vars(defaults) if k.isupper() and gate_like.match(k)}
    check("defaults has gate-only criteria to protect", len(criteria) > 5,
          f"{len(criteria)} names")
    fields = set()
    for dc in (cfg.Backend, cfg.Solve, cfg.Model, cfg.Output, cfg.Run):
        fields |= {f.name.upper() for f in dc.__dataclass_fields__.values()}
    overlap = sorted(criteria & fields)
    check("no spec field names one of them", not overlap, f"{overlap}")
    # And the specs offer no generic escape hatch into defaults.
    hatch = [f for dc in (cfg.Run, cfg.Backend, cfg.Solve)
             for f in dc.__dataclass_fields__
             if f in ("overrides", "defaults")]
    check("no generic defaults-override field on the spec", not hatch, f"{hatch}")


def g_figures_resolve() -> None:
    """[g] every registered figure resolves, and is classified exactly once.

    The registry holds STRINGS resolved on demand, which is what keeps importing
    mbem free of matplotlib -- and also what lets an entry rot unnoticed, since
    nothing touches it until someone asks for that figure. So the gate resolves
    all of them. It also checks the classification is a partition: a figure is
    model-free, or spans a study, or takes one run, and the CLI dispatches on
    exactly that, so a figure in two sets or in neither would be dispatched
    wrongly or not at all.
    """
    print("\n[g] the figure registry")
    from mbem import figures as F
    bad = []
    for key in sorted(F.FIGURES):
        try:
            F.resolve(key)
        except Exception as exc:                     # noqa: BLE001
            bad.append(f"{key}: {type(exc).__name__}: {exc}")
    check(f"all {len(F.FIGURES)} registered figures resolve", not bad,
          f"{bad[:2]}")
    both = sorted(F.STUDY_FIGURES & F.MODEL_FREE)
    check("no figure is both model-free and a study figure", not both, f"{both}")
    unknown = sorted((F.STUDY_FIGURES | F.MODEL_FREE) - set(F.FIGURES))
    check("no classification names an unregistered figure", not unknown,
          f"{unknown}")
    run_level = sorted(set(F.FIGURES) - F.STUDY_FIGURES - F.MODEL_FREE)
    check("the remainder are run figures", True, f"{run_level}")
    # save_figure is the one statement of the png+pdf convention; the demos
    # each carried their own copy, which rule 10 calls a bug twelve times over.
    import inspect
    src = inspect.getsource(F.save_figure)
    check("save_figure writes both png and pdf",
          '"png"' in src and '"pdf"' in src)


def f_run_dir_complete(run_dir: pathlib.Path, report: dict) -> None:
    """[f] the directory is complete, and MANIFEST actually matches."""
    print("\n[f] the run directory is self-describing")
    for name in ("STATUS", "MANIFEST", "resolved.json", "report.json",
                 "config.py", "fields_base.npz"):
        check(f"{name} present", (run_dir / name).exists())
    check("STATUS is the report status",
          (run_dir / "STATUS").read_text().strip() == report["status"])
    bad = []
    for line in (run_dir / "MANIFEST").read_text().splitlines():
        sha, _size, rel = line.split(None, 2)
        p = run_dir / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            bad.append(rel)
    check("every MANIFEST hash matches its file", not bad, f"{bad}")
    with np.load(run_dir / "fields_base.npz") as z:
        check("fields npz holds the solution slots", len(z.files) > 0,
              f"{len(z.files)} slot(s)")


def main() -> bool:
    print("=" * 76)
    print("Config layer: one statement of the model, one of every number")
    print("=" * 76)
    a_same_model()
    with tempfile.TemporaryDirectory(prefix="mbem_cfg_gate_") as tmp:
        # Outside the tree: the gates write nothing into the repo, and this one
        # must not be the exception.
        run_dir = pathlib.Path(tmp) / "run"
        run_dir.mkdir()
        report = b_defaults_untouched(run_dir)
        c_round_trip(run_dir)
        d_effective_covers_backend(run_dir)
        e_gate_criteria_unreachable()
        f_run_dir_complete(run_dir, report)
    g_figures_resolve()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: config layer "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
