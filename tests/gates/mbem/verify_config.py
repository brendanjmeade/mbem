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
     returns the stack in the order the fit takes it and ``laplace.run_fit``
     the fit, with the tail of an Andrade substrate read from the run's own
     record (and its window, short of ``tau_A``, warned of in the report),
     and the band of an extended Burgers one, the exponent and its two
     rates; a stored Laplace state is read back at its own moduli; and the
     run is REFUSED when the relaxing region is not on one scalar eps, or
     names a region the model lacks, or is given a negative
     ``andrade_alpha``, or two substrates at once.
  k  GRAVITY reaches a run through the builder's parameters and nowhere else
     (``rho_plate`` / ``rho_substrate`` of the layered box): the dump records
     the ``Gravity`` the model carried, and null when it carried none; the
     model rebuilt from the dump for ``mbem sample`` carries the same one, so
     a volume is read with the buoyancy it was solved with; it changes the
     solution, the vertical motion most; and a Laplace run is REFUSED when
     the stratification is heavy side up.
  l  ``--dry-run`` is dry on a SWEEP too: every child is validated and named,
     and no study directory is made.
  n  a WINDOW reaches a run through the builder (``sides="window"``) and
     nowhere else: the dump records the far field the model carried; the
     runner asks for every state's walls at once, and a state is then the
     dense solve of the model AT THAT STATE'S MATERIALS, walls included; and
     those are not the elastic state's walls.
  m  a fault that CREEPS reaches a run through ``Laplace.faults``
     (``rheology.ViscousFault``): each state records the fault's drag at its
     s, infinite in the elastic one, where the fault is locked; a state is
     the dense solve with that drag; the dump round-trips; and it is REFUSED
     on a prescribed slip, on a fault the model lacks and on the compressed
     backend, which solves no slip. The same run with ``backend.lu =
     "single"`` (``la.lu``: single-precision factors, each solve refined in
     double): the switch is in the dump, every state's row says how it was
     factored, and the state is the double run's (1e-12); a ``backend.lu``
     that is none of the three is refused, and so is ``"gpu"`` where no
     card answers: a card is asked for and never looked for, so the run
     above is LAPACK's double with one in the machine.
  o  a fault STEPPED IN TIME reaches a run through ``Run.stepper``: the run
     writes ``history.npz`` and no state's fields; the history is
     ``stepper.march`` on the reduction of the same model; the law, one that
     is not linear, round-trips through the dump; ``mbem sample`` refuses the
     run; it is REFUSED with a solved fault left without a law, and on the
     compressed kernel under a continuous fault; and at P0 in the whole space
     the compressed kernel (``backend=hmat``) is the dense run, which left
     to itself steps with the stabilised method (``mbem.rkc``) and says so.

Run from anywhere. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import pathlib
import re
import sys
import tempfile
import warnings

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
                        float(m.orientation(r, p)), p.slip) for p in r.patches),
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
    kw["fault_depth"] = 45.0                 # through the interface at 30 km
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="clamped", rake=90.0)
    _same_model("layered_box, a thrust through the interface", direct,
                layered_box_model(layered_box_meshes(**kw), sides="clamped",
                                  rake=90.0))
    lower = [f.name for r in direct.regions for f in r.faults]
    check("... its lower reach a second fault, in the substrate",
          lower == ["fault", "fault_lower"]
          and [len(r.faults) for r in direct.regions] == [1, 1])
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="window", rake=90.0,
                                     lower="creep")
    via = layered_box_model(layered_box_meshes(**kw), sides="window",
                            rake=90.0, lower="creep")
    _same_model("layered_box, a window, the lower reach creeping", direct, via)
    check("... every wall a window on the same far field, of the plate's slip",
          via.far_field.patches == layered_box.WALLS
          and via.far_field.spec == direct.far_field.spec
          and via.far_field.spec["depth"] == 30.0)
    kw.update(fault_depth=30.0, slab_depth=80.0, slab_half_len=50.0)
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="clamped", rake=90.0)
    _same_model("layered_box, a plate that subducts", direct,
                layered_box_model(layered_box_meshes(**kw), sides="clamped",
                                  rake=90.0))
    # ... the megathrust taken down the slab's face, where it creeps
    kw.update(fault_depth=45.0, slab_half_len=kw["fault_half_len"],
              edge_lower=7.5)
    direct = layered_box.build_model(layered_box.build_layered_box(**kw), 1e-3,
                                     mat, mat, sides="clamped", rake=90.0,
                                     lower="creep")
    _same_model("layered_box, the megathrust down the slab's face", direct,
                layered_box_model(layered_box_meshes(**kw), sides="clamped",
                                  rake=90.0, lower="creep"))
    reach = [p for p in direct.regions[0].patches if p.name == "fault_lower"]
    check("... its reach there an interface that slips, of both regions, "
          "and no fault",
          len(reach) == 1 and reach[0].slip == "solved"
          and any(p is reach[0] for p in direct.regions[1].patches)
          and [len(r.faults) for r in direct.regions] == [1, 0]
          and "b:fault_lower" in [s.name for s in
                                  generate_system(direct).layout.slots])


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
    fit = laplace.run_fit(d)
    check("run_fit: the stored run's fit, a Maxwell body's with no tail",
          fit.tails == () and np.array_equal(fit.s, s)
          and "warnings" not in rep)
    # An Andrade substrate by the same path: the tail its fit needs is read
    # back from the run's own record, and a window that stops above tau_A
    # is warned of, in the report too.
    soft = cfg.load(path, **small, andrade_alpha=0.25, tau_A=0.03)
    da = tmp / "andrade"
    da.mkdir()
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter("always")
        rep_a = runner.execute(soft, da, path)
    check("an Andrade substrate: the stored run's fit carries its tail",
          rep_a["status"] == "OK" and laplace.run_fit(da).tails == (0.25,),
          f"{laplace.run_fit(da).tails}")
    check("... and its window, short of tau_A, is warned of and reported",
          len(rep_a.get("warnings", [])) == 1
          and any("tau_A" in str(w.message) for w in heard),
          f"{len(rep_a.get('warnings', []))} in the report")
    try:
        cfg.load(path, **small, andrade_alpha=-0.25)
        refused = False
    except ValueError:
        refused = True
    check("REFUSED: a negative andrade_alpha, which would be a Maxwell "
          "body unsaid", refused)
    # An extended Burgers substrate: Kelvin elements, which the record holds
    # as lists, and the band they were cut from, which is the fit's tail
    band = cfg.load(path, **small, band_strength=0.5, tau_L=1.0e-3)
    db = tmp / "band"
    db.mkdir()
    rep_b = runner.execute(band, db, path)
    body = rheology.from_spec(json.loads((db / "resolved.json").read_text())
                              ["spec"]["laplace"]["rheology"]["substrate"])
    check("an extended Burgers substrate: the record gives the body back, "
          "and the stored run's fit carries its band",
          rep_b["status"] == "OK"
          and body == rheology.ExtendedBurgers.band(
              tau_M=1.0, strength=0.5, alpha=1.0 / 3.0, tau_L=1.0e-3,
              tau_H=1.0)
          and body.tail == (1.0 / 3.0, 1.0 / 1.0e-3, 1.0)
          and laplace.run_fit(db).tails == (body.tail,)
          and "warnings" not in rep_b, f"{laplace.run_fit(db).tails}")
    try:
        cfg.load(path, **small, band_strength=1.0, andrade_alpha=0.25)
        refused = False
    except ValueError:
        refused = True
    check("REFUSED: an Andrade substrate and an extended Burgers one at "
          "once", refused)

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
    mega = cfg.load(path, **small, dip=30.0, rake=90.0, depth=45.0)
    check("a fault deeper than the plate names the eps of its lower reach",
          mega.model.eps.get("fault_lower") == "auto"
          and "fault_lower" not in thrust.model.eps
          and mega.model.geometry.params["fault_depth"] == 45.0)
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


def m_creeping_fault(tmp: pathlib.Path) -> None:
    """[m] a fault's law through the config path."""
    print("\n[m] a creeping fault: a law on a solved fault, by state")
    from mbem import laplace, rheology
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system
    path = CONFIGS / "penny.py"
    law = rheology.ViscousFault(tau=1.0, width=1.0)
    run = dataclasses.replace(cfg.load(path), laplace=cfg.Laplace(
        rheology={}, s=laplace.s_grid(0.3, 3.0, per_decade=2),
        faults={"penny": law}))
    d = tmp / "creep"
    d.mkdir()
    rep = runner.execute(run, d, path)
    check("the run finished", rep["status"] == "OK", rep["status"])
    resolved = json.loads((d / "resolved.json").read_text())
    rows = resolved["states"]
    check("the elastic state is recorded with an infinite drag",
          rows[0]["drag"] == {"penny": "inf"}, f"{rows[0]['drag']}")
    worst = max(abs(r["drag"]["penny"] - law.drag(30.0, float(r["s"])))
                for r in rows[1:])
    check("each state records the law's drag at its s (mu tau s / width)",
          worst < 1e-12, f"{worst:.1e}")
    check("... where the fault is locked: no slip",
          not np.any(np.load(d / "fields_elastic.npz")["b:penny"]))
    bundle = cfg.resolve("penny", cfg.MESH_BUILDERS)(
        scale=1.0, **dict(run.model.geometry.params))
    model = cfg.resolve("penny", cfg.MODEL_BUILDERS)(
        bundle, **dict(run.model.params))
    last = rows[-1]
    direct = AssembledDense(
        generate_system(model), run.model.eps, "direct",
        fault_drag={"penny": last["drag"]["penny"]}).solve()["b:penny"]
    got = np.load(d / f"fields_{cfg.slug(last['label'])}.npz")["b:penny"]
    check("a state == the dense solve with that drag (bitwise)",
          np.array_equal(got, direct))
    check("... and is not the fault without it",
          _rel(got, np.load(d / "fields_elastic.npz")["b:penny"] + AssembledDense(
              generate_system(model), run.model.eps,
              "direct").solve()["b:penny"]) > 0.05)
    check("the law comes back from the dump as itself",
          rheology.from_spec(resolved["spec"]["laplace"]["faults"]["penny"])
          == law)

    # the same run with its matrix factored in single precision and every
    # solve refined in double (la.lu): the switch, recorded and heard
    single = dataclasses.replace(run, backend=dataclasses.replace(
        run.backend, lu="single"))
    ds = tmp / "creep_single"
    ds.mkdir()
    rep_s = runner.execute(single, ds, path)
    how = json.loads((ds / "resolved.json").read_text())["effective"]["backend"]
    check("backend.lu: recorded, and the run above was LAPACK's double",
          how["lu"] == "single"
          and resolved["effective"]["backend"]["lu"] == "double"
          and rep["states"][-1]["lu"]["how"] == "double",
          f"{how['lu']}, {resolved['effective']['backend']['lu']}")
    note = rep_s["states"][-1]["lu"]
    check("... every state says how it was factored and refined",
          all(r["lu"]["how"] == "single" and not r["lu"].get("fallback")
              for r in rep_s["states"]) and note["passes"] >= 1, f"{note}")
    worst = _rel(np.load(ds / f"fields_{cfg.slug(last['label'])}.npz")["b:penny"],
                 got)
    check("... and is the double run's state to 1e-9", worst < 1e-9,
          f"{worst:.1e}")
    try:
        cfg.Run(name="x", model=run.model, backend=cfg.Backend(lu="quad"))
        ok = False
    except ValueError as e:
        ok = "factored" in str(e)
    check("REFUSED: a backend.lu that is none of the three", ok)
    # a card is asked for, never looked for: a run that asks where none
    # answers (the probe made to fail) is refused before anything is built
    from mbem.la import lu as dense_lu
    there = dense_lu._cupy

    def no_card():
        raise RuntimeError("how='gpu' needs a CUDA device through cupy")
    dense_lu._cupy = no_card
    dg = tmp / "creep_no_card"
    dg.mkdir()
    try:
        runner.execute(dataclasses.replace(run, backend=dataclasses.replace(
            run.backend, lu="gpu")), dg, path, save=False)
        ok = False
    except RuntimeError as e:
        ok = "CUDA" in str(e) and not list(dg.glob("fields_*"))
    finally:
        dense_lu._cupy = there
    check("REFUSED: backend.lu = 'gpu' where no card answers (not run on "
          "the host instead)", ok)

    def refused_with(run_bad, exc, fragment):
        dd = tmp / f"creep_refused_{len(list(tmp.iterdir()))}"
        dd.mkdir()
        try:
            runner.execute(run_bad, dd, path, save=False)
        except exc as e:
            return fragment in str(e)
        return False

    ghost = dataclasses.replace(run, laplace=dataclasses.replace(
        run.laplace, faults={"elsewhere": law}))
    check("REFUSED: a law for a fault the model does not have",
          refused_with(ghost, KeyError, "not FAULT_TRACTION faults"))
    fixed = dataclasses.replace(run, model=dataclasses.replace(
        run.model, params=dict(run.model.params, prescribed=True)))
    check("REFUSED: a law on a PRESCRIBED slip",
          refused_with(fixed, KeyError, "not FAULT_TRACTION faults"))
    hmat = dataclasses.replace(run, backend=cfg.Backend(kind="hmat"))
    check("REFUSED: the compressed backend, which solves no slip",
          refused_with(hmat, ValueError, "dense backend only"))


def n_window_run(tmp: pathlib.Path) -> None:
    """[n] a window through the config path."""
    print("\n[n] a window: every state solved with its own walls")
    from mbem.backends.dense import AssembledDense
    from mbem.cases import layered_box
    from mbem.model import generate_system
    path = CONFIGS / "plate_substrate.py"
    run = cfg.load(path, half_x=150.0, fault_half_len=40.0, backend="dense",
                   t_min=0.3, t_max=3.0, per_decade=2, sides="window")
    d = tmp / "window"
    d.mkdir()
    rep = runner.execute(run, d, path)
    check("the run finished", rep["status"] == "OK", rep["status"])
    resolved = json.loads((d / "resolved.json").read_text())
    far = resolved["model"]["far_field"]
    check("the dump records the far field and its walls",
          far is not None and tuple(far["patches"]) == layered_box.WALLS
          and far["spec"]["half_len"] == 40.0, f"{far and far['patches']}")
    check("the walls of every state were asked for at once",
          "far_field" in rep["phases"])
    bundle = cfg.resolve(run.model.geometry.builder, cfg.MESH_BUILDERS)(
        scale=run.model.geometry.scale, **dict(run.model.geometry.params))
    model = cfg.resolve(run.model.builder, cfg.MODEL_BUILDERS)(
        bundle, **dict(run.model.params))
    last = resolved["states"][-1]
    state = model.with_materials({k: cfg.Material(**m).build()
                                  for k, m in last["materials"].items()})
    wall = next(p for p in model.far_field_patches() if p.name == "sides_upper")
    check("a state's walls are not the elastic state's",
          _rel(state.value_of(wall), model.value_of(wall)) > 0.05,
          f"{_rel(state.value_of(wall), model.value_of(wall)):.3f}")
    # the same state with its walls WRITTEN IN, at its materials: nothing of
    # the run's path -- no rebuild, no stack -- is in this solve
    from mbem.model import BCType, Patch, Region, RegionModel
    written = {p.name: Patch(p.name, p.mesh, p.bc, value=state.value_of(p))
               for p in model.far_field_patches()}
    shared: dict = {}
    regions = []
    for r in state.regions:
        patches = [written.get(p.name) or shared.setdefault(p.name, p)
                   for p in r.patches]
        regions.append(Region(r.name, r.material, patches,
                              probe_point=r.probe_point, faults=r.faults))
    direct = AssembledDense(generate_system(RegionModel(regions)),
                            run.model.eps, "direct").solve()
    got = np.load(d / f"fields_{cfg.slug(last['label'])}.npz")
    check("a state == the dense solve at its materials with its walls "
          "written in",
          max(_rel(got[k], direct[k]) for k in direct) < 1e-12,
          f"{max(_rel(got[k], direct[k]) for k in direct):.1e}")

    # ... and READ BACK from the folder alone, on a section: `mbem sample`
    # must give the state its moduli and its walls, as the solve had them,
    # and read it near a face from the recovered displacement
    # (``recover.NearField``) unless asked for the element values.
    from mbem import cli, vti, volume as V
    from mbem.recover import NearField
    slug = cfg.slug(last["label"])
    grid = V.section_grid(model, "y", 0.0, 25.0)
    where = V.classify(model, grid)

    def sampled(*flags):
        out = tmp / ("window_section" + "".join(flags))
        with contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(["sample", str(d), "--section", "y=0",
                             "--spacing", "25", "--state", slug,
                             "--no-eigenstress", "--out", str(out), *flags])
        return (vti.read(out / f"volume_{slug}.vti")["arrays"]["u"]
                .reshape(-1, 3) if code == 0 else np.zeros((1, 3)))

    def read(with_model, near=None):
        return V.sample(with_model, direct, run.model.eps, grid, where,
                        eigenstress=False, near=near)["u"]

    near = NearField(model)
    got, want = sampled(), read(state, near)
    check("mbem sample --section reads a state with its moduli and walls",
          got.shape == want.shape and _rel(got, want) < 1e-5,
          f"{grid.points.shape[0]} points, "
          f"{_rel(got, want) if got.shape == want.shape else float('nan'):.1e}")
    check("... near a face from the recovered displacement, by default",
          _rel(want, read(state)) > 1e-3, f"{_rel(want, read(state)):.3f}")
    plain = sampled("--plain")
    check("... and from the element values with --plain",
          plain.shape == want.shape and _rel(plain, read(state)) < 1e-5,
          f"{_rel(plain, read(state)) if plain.shape == want.shape else float('nan'):.1e}")
    check("the base model would read the same state wrongly",
          _rel(read(model, near), want) > 0.02,
          f"{_rel(read(model, near), want):.3f}")


def l_sweep_dry_run(tmp: pathlib.Path) -> None:
    from mbem import cli
    runs = tmp / "dry"
    runs.mkdir()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        code = cli.main(["run", str(CONFIGS / "plate_substrate.py"),
                         "--sweep", "half_x=400,800",
                         "--sweep", "sides=free,clamped", "--dry-run",
                         "--runs-dir", str(runs)])
    try:
        named = [r["name"] for r in json.loads(out.getvalue())["runs"]]
    except (ValueError, KeyError):
        named = []
    check("[l] a sweep with --dry-run names its children and makes no study",
          code == 0 and len(named) == 4 and not any(runs.iterdir()),
          f"exit {code}, {len(named)} named, "
          f"{len(list(runs.iterdir()))} directories made")


def o_stepped_run(tmp: pathlib.Path) -> None:
    """[o] a stepped fault through the config path."""
    print("\n[o] a stepped fault: a law that is not linear, in time")
    from mbem import cli, rheology, stepper
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system
    path = CONFIGS / "penny.py"
    base = cfg.load(path)
    drop = 1.0e-3 * 30.0                        # the scale of the crack's drop
    law = rheology.RateStrengtheningFault(v0=1.0e-3, a_sigma=0.3 * drop,
                                          rake=0.0)
    times = (0.3, 1.0, 3.0)
    run = dataclasses.replace(base, stepper=cfg.Stepper(
        faults={"penny": law}, t=times))
    d = tmp / "stepped"
    d.mkdir()
    rep = runner.execute(run, d, path)
    check("the run finished", rep["status"] == "OK", rep["status"])
    check("it wrote a history and no state's fields",
          (d / "history.npz").exists() and not list(d.glob("fields_*.npz")))
    resolved = json.loads((d / "resolved.json").read_text())
    check("its one state is the elastic body",
          [r["label"] for r in resolved["states"]] == ["elastic"])
    bundle = cfg.resolve("penny", cfg.MESH_BUILDERS)(
        scale=1.0, **dict(run.model.geometry.params))
    model = cfg.resolve("penny", cfg.MODEL_BUILDERS)(
        bundle, **dict(run.model.params))
    body = stepper.Body([AssembledDense(
        generate_system(model), run.model.eps, "direct").condense()])
    direct = stepper.march(body, stepper.law_rate(
        body, {"penny": law}, {"penny": model.regions[0].material.mu}), times)
    with np.load(d / "history.npz") as z:
        hist = {k: z[k] for k in z.files}
    check("the history's times are the ones asked for",
          np.array_equal(hist["t"], np.array(times)))
    check("the slip == stepper.march on the reduction of the same model",
          _rel(hist["b:penny"], direct.fault("penny")) < 1e-12,
          f"{_rel(hist['b:penny'], direct.fault('penny')):.1e}")
    check("... and the traction the law saw is beside it",
          _rel(hist["traction:penny"], direct.fault("penny", "traction")) < 1e-12)
    check("the fault really slips, and more as time goes",
          0.0 < np.abs(hist["b:penny"][0]).max() < np.abs(hist["b:penny"][-1]).max())
    check("the law comes back from the dump as itself",
          rheology.from_spec(resolved["spec"]["stepper"]["faults"]["penny"])
          == law)
    check("the report says what the stepper did",
          rep["stepper"]["slip_unknowns"] == body.n
          and rep["stepper"]["law_evaluations"] > 0, f"{rep['stepper']}")
    with contextlib.redirect_stdout(io.StringIO()) as said:
        code = cli.main(["sample", str(d), "--out", str(tmp / "stepped_vol")])
    check("mbem sample refuses a stepped run, and says why",
          code == 1 and "stepped run" in said.getvalue(), f"exit {code}")

    def refused_with(run_bad, exc, fragment):
        dd = tmp / f"stepped_refused_{len(list(tmp.iterdir()))}"
        dd.mkdir()
        try:
            runner.execute(run_bad, dd, path, save=False)
        except exc as e:
            return fragment in str(e)
        return False

    ghost = dataclasses.replace(run, stepper=cfg.Stepper(
        faults={"elsewhere": law}, t=times))
    check("REFUSED: a solved fault left without a law",
          refused_with(ghost, KeyError, "every one of them a law"))
    hmat = dataclasses.replace(run, backend=cfg.Backend(kind="hmat"))
    check("REFUSED: the compressed kernel under a continuous fault",
          refused_with(hmat, ValueError, "P0 faults"))

    # ... and at P0, in the whole space, the compressed kernel IS the route:
    # the config's own stepped crack, dense and compressed
    # (one integrator for the two, so that what is compared is the operator)
    got = {}
    for kind, method in (("dense", "RK45"), ("hmat", "RK45"), ("hmat", "")):
        dd = tmp / f"stepped_p0_{kind}_{method or 'own'}"
        dd.mkdir()
        rep = runner.execute(cfg.load(path, order=0, law="friction",
                                      backend=kind, method=method), dd, path)
        with np.load(dd / "history.npz") as z:
            got[kind, method] = (rep, z["b:penny"])
    dense, hmat, own = got["dense", "RK45"], got["hmat", "RK45"], got["hmat", ""]
    check("at P0 the compressed kernel is the dense run",
          hmat[0]["status"] == "OK" and _rel(hmat[1], dense[1]) < 1e-5,
          f"{_rel(hmat[1], dense[1]):.1e}")
    check("... and says what it stored",
          hmat[0]["kernel"]["elements"] == hmat[1].shape[1]
          and "kernel" not in dense[0], f"{hmat[0]['kernel']['elements']} elements")
    check("left to itself it steps with the stabilised method, to the "
          "same slip, and the report says which",
          _rel(own[1], hmat[1]) < 1e-4
          and (own[0]["stepper"]["method"], hmat[0]["stepper"]["method"],
               dense[0]["stepper"]["method"]) == ("RKC", "RK45", "RK45"),
          f"{_rel(own[1], hmat[1]):.1e}")


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
        l_sweep_dry_run(pathlib.Path(tmp))
        m_creeping_fault(pathlib.Path(tmp))
        n_window_run(pathlib.Path(tmp))
        o_stepped_run(pathlib.Path(tmp))
    g_figures_resolve()
    n_bad = sum(1 for c in CHECKS if not c)
    print("-" * 76)
    ok = not n_bad
    print(f"{'PASS' if ok else 'FAIL'}: config layer "
          f"({len(CHECKS) - n_bad} of {len(CHECKS)} checks)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
