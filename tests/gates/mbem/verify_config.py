"""Gate: the config layer cannot diverge from the code it drives.

A config layer's characteristic failure is not a crash, it is a SECOND
statement of something the codebase states once. This repo keeps the fault sign
in one place, the eps rule in one place and every tolerance in
``defaults.py``, because the second statement is the one that goes wrong -- and
a slip-pairing sign error is invisible at nu = 1/4. So:

  a  the model a CONFIG builds is the model the GATES build, entrywise. Not
     "similar": same patch names, BCs, orders, layouts, dof counts, inferred
     sigma, unknown count, and the same fault value array (the Burgers vector,
     or a traction fault's target). For the fault box and for the penny crack.
     This is why a config names a builder instead of describing patches.
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
  h  a traction-BC fault survives the run path: configs/penny.py through
     runner.execute records bc FAULT_TRACTION, layout continuous, n_dofs and
     the locked count in the fingerprint, ONE numeric eps for the fault, the
     slot "b:penny", and writes it to the fields npz with (n_dofs, 3) rows.
  i  a STATE is the base model with its own materials replaced, on every
     backend. The two ``rebuild_for_materials`` disagree about a region the
     map does not name (dense keeps the operator's current material,
     compressed resets to the base), so the runner hands over the complete
     map and a run means one thing. Pinned on DENSE, the backend whose own
     rule is the other one: a state after an unrelated state equals the same
     state run alone, an empty state after a non-empty one is the base again,
     and both differ from what carrying materials over would give. Also: two
     labels that collide as file names are refused, and the dump's states
     resolve to the materials the run was solved at.
  j  a LAPLACE run (``Run.laplace``) derives its states from the model the
     builder returns, so the unrelaxed moduli are stated once. Through
     configs/plate_substrate.py and runner.execute: the elastic state plus one
     per s, each recorded with its s and its moduli; those moduli are the
     rheology's at that s, with the bulk modulus unmoved; the elastic state is
     bitwise the same model run with no rheology at all; ``laplace.run_stack``
     returns the stack in the order the fit takes it; a stored Laplace state is
     read back at its own moduli; and the run is REFUSED when the relaxing
     region is not on one scalar eps, or names a region the model lacks.
  k  GRAVITY reaches a run through the builder's parameters and nowhere else
     (``rho_plate`` / ``rho_substrate`` of the layered box): the dump records
     the ``Gravity`` the model carried, and null when it carried none; the
     model rebuilt from the dump for ``mbem sample`` carries the same one, so
     a volume is read with the buoyancy it was solved with; it changes the
     solution, the vertical motion most; and a Laplace run is REFUSED when
     the stratification is heavy side up.

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import dataclasses
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


def _same_model(label: str, direct, via) -> None:
    sd, sv = generate_system(direct), generate_system(via)
    check(f"{label}: same unknown count", sd.layout.n_unknowns == sv.layout.n_unknowns,
          f"{sd.layout.n_unknowns}")
    check(f"{label}: same slot names",
          [s.name for s in sd.layout.slots] == [s.name for s in sv.layout.slots])

    def describe(m):
        return [(r.name, r.material.mu, r.material.lam,
                 tuple((p.name, str(p.bc), p.order, p.layout, p.n_dofs, p.n_triangles,
                        float(m.orientation(r, p))) for p in r.patches),
                 tuple((f.name, str(f.bc), f.order, f.layout, f.n_dofs, f.allow_opening,
                        f.n_triangles) for f in r.faults))
                for r in m.regions]
    check(f"{label}: same regions, patches, BCs, orders, layouts, dofs and sigma",
          describe(direct) == describe(via))

    fd = [f for r in direct.regions for f in r.faults]
    fv = [f for r in via.regions for f in r.faults]
    check(f"{label}: same number of faults", len(fd) == len(fv), f"{len(fd)}")
    same = len(fd) == len(fv) and all(
        np.array_equal(x.value_array(), y.value_array())
        for x, y in zip(fd, fv))
    # Bitwise, not to a tolerance: this is the Burgers vector (or the traction
    # target), and a tolerance would accept a sign flip on a component that
    # happens to be small.
    check(f"{label}: fault value array bitwise identical", same)


def a_same_model() -> None:
    """[a] config builder == the builder the gates call, entrywise."""
    print("\n[a] the config path builds the gates' model")
    direct = build_model(
        build_fault_box(half_x=100.0, z_bottom=-60.0, fault_half_len=30.0,
                        fault_depth=18.0, near_field_radius=50.0,
                        edge_fault=4.0, edge_near=25.0, edge_far=50.0,
                        edge_side=50.0),
        0.01, ElasticMaterial(mu=30.0, lam=30.0))
    _same_model("fault_box", direct, fault_box_model(fault_box_meshes()))
    from mbem.cases import penny
    from mbem.cases.registry import penny_meshes, penny_model
    mat = ElasticMaterial(mu=30.0, lam=30.0)
    direct = penny.build_model(penny.penny_mesh(1.0, 1.0 / 8.0, "cdt"), mat,
                               drop=penny.stress_drop(30.0, 30.0, 1e-3, 1.0), order=2)
    _same_model("penny", direct, penny_model(penny_meshes()))
    from mbem.cases import layered_box
    from mbem.cases.registry import layered_box_meshes, layered_box_model
    kw = dict(half_x=150.0, z_interface=-30.0, z_bottom=-150.0,
              fault_half_len=40.0, fault_depth=15.0, near_field_radius=60.0,
              edge_fault=5.0, edge_near=20.0, edge_far=50.0, edge_side=50.0,
              edge_interface=15.0)
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="clamped")
    _same_model("layered_box", direct,
                layered_box_model(layered_box_meshes(**kw), sides="clamped"))
    kw["dip"] = 30.0
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="clamped", rake=90.0)
    _same_model("layered_box, a thrust", direct,
                layered_box_model(layered_box_meshes(**kw), sides="clamped",
                                  rake=90.0))


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


def h_penny_run(run_dir: pathlib.Path) -> None:
    """[h] a traction-BC fault through the run path."""
    print("\n[h] configs/penny.py through runner.execute")
    run = cfg.load(CONFIGS / "penny.py")
    report = runner.execute(run, run_dir, CONFIGS / "penny.py")
    check("the penny run finished", report["status"] == "OK", report["status"])
    resolved = json.loads((run_dir / "resolved.json").read_text())
    entries = [p for r in resolved["model"]["regions"] for p in r["patches"]]
    f = [p for p in entries if p["name"] == "penny"]
    check("the fingerprint has the fault, bc FAULT_TRACTION, layout continuous",
          len(f) == 1 and f[0]["bc"] == "FAULT_TRACTION" and f[0]["layout"] == "continuous",
          f"{f[0]['bc'] if f else '-'} / {f[0]['layout'] if f else '-'}")
    check("n_dofs and n_locked recorded (shared nodes, the locked rim)",
          bool(f) and f[0]["n_dofs"] > f[0]["n_triangles"] and f[0]["n_locked"] > 0,
          f"n_dofs {f[0]['n_dofs'] if f else '-'}, locked {f[0]['n_locked'] if f else '-'}")
    eps = resolved["effective"]["eps"]["penny"]
    check("one numeric eps on the fault (min == max)",
          isinstance(eps.get("mean"), float) and eps["min"] == eps["max"], f"{eps.get('mean')}")
    check('slot "b:penny" in the layout', "b:penny" in resolved["model"]["slots"],
          f"{resolved['model']['slots']}")
    with np.load(run_dir / "fields_base.npz") as z:
        ok = "b:penny" in z.files and z["b:penny"].shape == (f[0]["n_dofs"], 3) if f else False
        check("fields npz holds b:penny with (n_dofs, 3) rows", ok,
              f"{z['b:penny'].shape if 'b:penny' in z.files else z.files}")


def _fields(run_dir: pathlib.Path, label: str) -> np.ndarray:
    with np.load(run_dir / f"fields_{cfg.slug(label)}.npz") as z:
        return np.concatenate([z[k].ravel() for k in sorted(z.files)])


def _rel(a, b) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def i_states(tmp: pathlib.Path) -> None:
    """[i] what a state's materials are, stated once and honoured."""
    print("\n[i] a state is the base with its own materials, on any backend")
    path = CONFIGS / "topo_inclusion.py"
    base = cfg.load(path, surface="flat", backend="dense", scale=0.35)
    soft = cfg.Material(mu=6.0, lam=11.0)        # the inclusion, state A
    stiff = cfg.Material(mu=45.0, lam=20.0)      # the HOST, state B
    quiet = cfg.Output(slots=(), figures=())

    def run(tag, states):
        d = tmp / tag
        d.mkdir()
        r = dataclasses.replace(base, states=states, outputs=quiet)
        rep = runner.execute(r, d, path)
        check(f"run {tag!r} finished", rep["status"] == "OK", rep["status"])
        return d

    chain = run("chain", (cfg.State("base"),
                          cfg.State("A", {"inclusion": soft}),
                          cfg.State("B", {"host": stiff}),
                          cfg.State("again")))
    alone = run("alone", (cfg.State("B", {"host": stiff}),))
    carried = run("carried", (cfg.State("AB", {"inclusion": soft,
                                               "host": stiff}),))
    b = _fields(chain, "B")
    got = _rel(b, _fields(alone, "B"))
    check("B after A == B alone (nothing carries over)", got < 1e-10,
          f"{got:.1e}")
    got = _rel(_fields(chain, "again"), _fields(chain, "base"))
    check("an empty state after others == the base state", got < 1e-10,
          f"{got:.1e}")
    miss = _rel(b, _fields(carried, "AB"))
    check("... and B is NOT A's inclusion carried into B (tripwire)",
          miss > 1e-3, f"{miss:.1e}")

    spec = json.loads((chain / "resolved.json").read_text())["spec"]
    live = cfg.state_overrides(
        dataclasses.replace(base, states=(cfg.State("A", {"inclusion": soft}),
                                          cfg.State("B", {"host": stiff}))
                            ).states)
    dumped = cfg.state_overrides(spec["states"])
    check("the dump's states resolve to the materials that were solved",
          all(dumped[k] == live[k] for k in live)
          and dumped["base"] == {} and dumped["again"] == {},
          f"{sorted(dumped)}")

    try:
        dataclasses.replace(base, states=(cfg.State("s=1e+03"),
                                          cfg.State("s=1e-03")))
        refused = ""
    except ValueError as e:
        refused = str(e)
    check("labels that collide as file names are refused",
          "collide as file names" in refused,
          f"{cfg.slug('s=1e+03')} / {cfg.slug('s=1e-03')}")


def j_laplace_run(tmp: pathlib.Path) -> None:
    """[j] a Laplace run through the runner."""
    print("\n[j] a Laplace run: states derived from the model, by rheology")
    from mbem import cli, laplace, rheology
    path = CONFIGS / "plate_substrate.py"
    small = dict(half_x=150.0, fault_half_len=40.0, backend="dense",
                 t_min=0.3, t_max=3.0, per_decade=2)
    run = cfg.load(path, **small)
    d = tmp / "laplace"
    d.mkdir()
    rep = runner.execute(run, d, path)
    check("the Laplace run finished", rep["status"] == "OK", rep["status"])
    resolved = json.loads((d / "resolved.json").read_text())
    rows = resolved["states"]
    n = len(run.laplace.s)
    check("states: the elastic one, then one per s",
          len(rows) == n + 1 and rows[0]["label"] == "elastic"
          and rows[0]["s"] == "inf" and rows[0]["materials"] == {},
          f"{len(rows)} states for {n} s")
    s_rec = [float(r["s"]) for r in rows[1:]]
    check("... in decreasing s, each recorded",
          s_rec == sorted(run.laplace.s, reverse=True))

    base = cfg.load(path, **small)
    plain = dataclasses.replace(base, laplace=None)
    dp = tmp / "plain"
    dp.mkdir()
    runner.execute(plain, dp, path)
    check("the elastic state == the same model with no rheology (bitwise)",
          np.array_equal(_fields(d, "elastic"), _fields(dp, "base")))

    mx = run.laplace.rheology["substrate"]
    mu0 = lam0 = 30.0                         # the builder's substrate
    K = lam0 + 2.0 * mu0 / 3.0
    worst = worst_k = 0.0
    for r in rows[1:]:
        m = r["materials"]["substrate"]
        mu, lam = rheology.moduli(mx, mu0, lam0, float(r["s"]))
        worst = max(worst, abs(m["mu"] - mu) / mu0, abs(m["lam"] - lam) / lam0)
        worst_k = max(worst_k, abs(m["lam"] + 2.0 * m["mu"] / 3.0 - K) / K)
        if set(r["materials"]) != {"substrate"}:
            worst = np.inf
    check("each state: the substrate at the rheology's moduli of s, alone",
          worst < 1e-14, f"{worst:.1e}")
    check("... and its bulk modulus has not moved", worst_k < 1e-14,
          f"{worst_k:.1e}")

    s, stack = laplace.run_stack(d)
    ok = (np.array_equal(s, np.array(s_rec))
          and all(v.shape[0] == n + 1 for v in stack.values())
          and np.array_equal(stack["u:top"][0],
                             np.load(d / "fields_elastic.npz")["u:top"])
          and np.array_equal(stack["u:top"][-1], np.load(
              d / f"fields_{cfg.slug(rows[-1]['label'])}.npz")["u:top"]))
    check("run_stack: the stack in the fit's order, elastic first", ok,
          f"{sorted(stack)[:3]}...")
    try:
        laplace.run_stack(dp)
        refused = False
    except ValueError:
        refused = True
    check("run_stack refuses a run that is not a Laplace run", refused)

    _b, model, _e = cli._rebuild(resolved)
    last = rows[-1]
    got = next(r for r in cli._state_model(
        resolved, model, cfg.slug(last["label"])).regions
        if r.name == "substrate").material
    check("a stored Laplace state is read back at its own moduli",
          (got.mu, got.lam) == (last["materials"]["substrate"]["mu"],
                                last["materials"]["substrate"]["lam"])
          and got.mu < 0.5 * mu0, f"mu {got.mu:.3f} of {mu0}")

    def refused_with(run_bad, exc, fragment):
        db = tmp / f"bad{len(list(tmp.iterdir()))}"
        db.mkdir()
        try:
            runner.execute(run_bad, db, path)
        except exc as e:
            return fragment in str(e)
        return False

    auto = dataclasses.replace(run, model=dataclasses.replace(
        run.model, eps="auto"))
    check('REFUSED: the relaxing region on eps="auto" (per-element)',
          refused_with(auto, ValueError, "ONE scalar eps"))
    ghost = dataclasses.replace(run, laplace=dataclasses.replace(
        run.laplace, rheology={"mantle": mx}))
    check("REFUSED: a rheology for a region the model does not have",
          refused_with(ghost, KeyError, "not in the model"))


def k_gravity_run(tmp: pathlib.Path) -> None:
    """[k] gravity through the config path."""
    print("\n[k] gravity: a builder parameter, recorded and rebuilt")
    from mbem import cli, laplace
    from mbem.model import Gravity
    path = CONFIGS / "plate_substrate.py"
    small = dict(half_x=150.0, fault_half_len=40.0, backend="dense",
                 t_min=0.3, t_max=3.0, per_decade=2)
    rho = dict(rho_plate=2700.0, rho_substrate=3300.0)
    run = cfg.load(path, **small, **rho)
    d = tmp / "gravity"
    d.mkdir()
    rep = runner.execute(run, d, path)
    check("the run under gravity finished", rep["status"] == "OK", rep["status"])
    resolved = json.loads((d / "resolved.json").read_text())
    want = Gravity(density={"plate": 2700.0, "substrate": 3300.0},
                   surface=("top",))
    rec = resolved["model"]["gravity"]
    check("the dump records the gravity the model carried",
          rec == json.loads(json.dumps(dataclasses.asdict(want))), f"{rec}")
    bare = json.loads((tmp / "laplace" / "resolved.json").read_text())
    check("... and null for a model that carried none",
          "gravity" in bare["model"] and bare["model"]["gravity"] is None)
    _b, model, _e = cli._rebuild(resolved)
    check("the model rebuilt from the dump carries the same gravity",
          model.gravity == want and model.unstable_contrasts() == [])
    s, stack = laplace.run_stack(d)
    _s0, stack0 = laplace.run_stack(tmp / "laplace")
    uz, uz0 = stack["u:top"][-1, :, 2], stack0["u:top"][-1, :, 2]
    uy, uy0 = stack["u:top"][-1, :, 1], stack0["u:top"][-1, :, 1]
    dz = float(np.linalg.norm(uz - uz0) / np.linalg.norm(uz0))
    dy = float(np.linalg.norm(uy - uy0) / np.linalg.norm(uy0))
    check("it changes the solution, the vertical motion most",
          dz > 1.0e-3 and dz > 3.0 * dy,
          f"u_z by {dz:.2e}, fault-parallel by {dy:.2e}")

    # ``gravity`` scales g and leaves the densities, so a sweep over it is the
    # same body with and without: at 0 no patch carries a buoyancy term.
    from mbem.cases.registry import layered_box_meshes, layered_box_model
    off = cfg.load(path, **small, **rho, gravity=0.0)
    weightless = layered_box_model(
        layered_box_meshes(**off.model.geometry.params), **off.model.params)
    check("gravity=0 keeps the densities and applies no buoyancy anywhere",
          weightless.gravity is not None and off.model.params["g"] == 0.0
          and all(weightless.buoyancy(r, q) is None
                  for r in weightless.regions for q in r.patches))
    thrust = cfg.load(path, **small, dip=30.0, rake=90.0)
    check("a dip-slip plate names the in-plane figures, strike slip the other",
          thrust.outputs.figures == ("plate_thrust", "plate_cycle")
          and run.outputs.figures == ("plate_substrate",)
          and thrust.model.geometry.params["dip"] == 30.0
          and thrust.model.params["rake"] == 90.0)
    # ``refine`` is one ray: the plate's two faces and the interface's eps,
    # the three a plate that bends is first order in, and nothing else
    fine = cfg.load(path, **small, dip=30.0, rake=90.0, refine=2.0)
    gp, gf = thrust.model.geometry.params, fine.model.geometry.params
    check("refine halves the top, the interface and the substrate's eps "
          "together, and leaves the rest",
          all(gf[k] == gp[k] / 2.0 for k in ("edge_near", "edge_interface"))
          and all(fine.model.eps[k] == thrust.model.eps[k] / 2.0
                  for k in ("interface", "sides_lower", "base"))
          and all(gf[k] == gp[k] for k in gp
                  if k not in ("edge_near", "edge_interface"))
          and fine.model.eps["top"] == thrust.model.eps["top"] == "auto")

    heavy = cfg.load(path, **small, rho_plate=3300.0, rho_substrate=2700.0)
    db = tmp / "heavy"
    db.mkdir()
    try:
        runner.execute(heavy, db, path)
        refused = False
    except ValueError as e:
        refused = "heavy side up" in str(e) and "interface" in str(e)
    check("REFUSED: a Laplace run with the heavier region on top", refused)


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
        penny_dir = pathlib.Path(tmp) / "penny"
        penny_dir.mkdir()
        h_penny_run(penny_dir)
        i_states(pathlib.Path(tmp))
        j_laplace_run(pathlib.Path(tmp))
        k_gravity_run(pathlib.Path(tmp))
    g_figures_resolve()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: config layer "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
