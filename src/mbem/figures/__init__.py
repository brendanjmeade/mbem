"""Figures a config can ask for by name, and the context they are handed.

A figure maker is registered under a name, so ``Output.figures=("fault_only",)``
is all a config needs and the runner writes the result into the run's own
directory. The registry holds STRINGS resolved on demand, so importing ``mbem``
or validating a config never imports matplotlib.

TWO KINDS, because the figures this repo actually wants are of two shapes:

* a RUN figure reads one run -- its solved fields, its model, its meshes;
* a STUDY figure reads SEVERAL runs, because the quantity it draws is a
  difference between them. The topography showcase is the reason: the
  topography effect is ``u(topo) - u(flat)``, and the warped and flat surfaces
  are different meshes, hence different operators, hence different runs. A
  figure that needs two operators cannot be a property of one run, and
  pretending otherwise is how the old script ended up solving four states in one
  process and writing a single npz nobody could trace.

``save_figure`` is here because the ``dpi=300, bbox_inches="tight"`` +
``for ext in ("png", "pdf")`` block was written twelve times across the demos,
which rule 10 calls a bug twelve times over.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# name -> "module:qualname". Strings, resolved lazily.
FIGURES: dict[str, str] = {
    # run-level
    "fault_only": "mbem.figures.fault:fault_only",
    # study-level (several runs)
    "topo_inclusion_showcase": "mbem.figures.topo:showcase",
    "topo_inclusion_contour": "mbem.figures.topo:contour",
    "backend_agreement": "mbem.figures.compare:backend_agreement",
    "eps_convergence": "mbem.figures.convergence:eps_convergence",
    "onfault_stress": "mbem.figures.onfault:onfault_stress",
    # model-free: no model, no backend, no solve -- the kernels alone
    "point_kernel": "mbem.figures.point_kernel:figure",
    "triangle_field": "mbem.figures.triangle_field:figure",
    "triangle_eps_sweep": "mbem.figures.triangle_eps_sweep:figure",
    "anelastic_subtraction": "mbem.figures.anelastic_subtraction:figure",
    "onfault_convergence": "mbem.figures.onfault_convergence:figure",
    "eps_h_convergence": "mbem.figures.eps_h_convergence:figure",
}

# Which registry entries take a STUDY (several runs) rather than one run.
STUDY_FIGURES = {"topo_inclusion_showcase", "topo_inclusion_contour",
                 "backend_agreement", "eps_convergence",
                 "onfault_stress"}
# Which take no model at all: pure kernel figures, no solve.
MODEL_FREE = {"point_kernel", "triangle_field", "triangle_eps_sweep",
              "anelastic_subtraction", "onfault_convergence",
              "eps_h_convergence"}


def resolve(name: str) -> Callable:
    target = FIGURES.get(name)
    if target is None:
        raise KeyError(f"unknown figure {name!r}; known: {sorted(FIGURES)}")
    mod, _, qual = target.partition(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def save_figure(fig, out_dir: pathlib.Path, key: str) -> list:
    """Write ``fig_<key>.png`` and ``.pdf`` into ``out_dir``. 300 dpi, tight.

    One statement of the convention the demos each carried their own copy of.
    PNG for fast iteration and for the tracked gallery, PDF (vector) because
    that is what a manuscript build wants.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_{key}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        written.append(p)
    return written


# ---------------------------------------------------------------------
# what a maker is handed
# ---------------------------------------------------------------------

@dataclass
class RunData:
    """One run, read back off its directory.

    Read from the DIRECTORY rather than passed in memory, so a figure can be
    remade from an archived run months later with the same code path the runner
    uses. That is the whole point of the run folder being self-describing.
    """
    run_dir: pathlib.Path
    fields: dict = field(default_factory=dict)      # state label -> {slot: array}
    meshes: dict = field(default_factory=dict)      # name -> vertices/triangles
    resolved: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.resolved.get("run", {}).get("name", self.run_dir.name)

    def param(self, key: str, default=None):
        """A geometry parameter the run was built with, e.g. ``surface``."""
        return (self.resolved.get("spec", {}).get("model", {})
                .get("geometry", {}).get("params", {}).get(key, default))


def load_run(run_dir: str | pathlib.Path) -> RunData:
    d = pathlib.Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(d)
    out = RunData(d)
    for p in sorted(d.glob("fields_*.npz")):
        with np.load(p) as z:
            out.fields[p.stem.removeprefix("fields_")] = {k: z[k] for k in z.files}
    mp = d / "meshes.npz"
    if mp.exists():
        with np.load(mp) as z:
            out.meshes = {k: z[k] for k in z.files}
    for name, attr in (("resolved.json", "resolved"), ("report.json", "report")):
        p = d / name
        if p.exists():
            setattr(out, attr, json.loads(p.read_text()))
    return out


@dataclass
class RunContext:
    """A run figure's context: the run directory AND the live objects.

    Live, because a figure often has to EVALUATE -- free-surface stress on a
    grid, an interior profile -- and that needs the ``RegionModel`` and the same
    ``eps`` the operator used, not just the solved slot arrays. Handing over the
    model is also what keeps ``eps`` stated once: the figure cannot pick its own.

    ``run_dir`` is where the figure is written and where the provenance already
    is, so a maker never has to decide either.
    """
    run_dir: pathlib.Path
    model: Any
    bundle: Any
    eps: Any
    solutions: dict = field(default_factory=dict)   # state label -> {slot: array}
    resolved: dict = field(default_factory=dict)
    # The assembled operator, for a figure about the OPERATOR rather than the
    # solution: block structure, stored bytes, ranks. None for the dense
    # backend's figures, which have no block structure to show.
    asm: Any = None
    system: Any = None

    @property
    def base(self) -> dict:
        """The first state's solution -- what a single-state figure wants."""
        return next(iter(self.solutions.values()))


@dataclass
class Study:
    """Several runs under one directory, one per point of the swept product.

    ``runs`` is keyed by the swept VALUE when one parameter was swept, which is
    what the single-axis figures want. ``rows`` keeps each child's full parameter
    dict, so a two-axis sweep can be selected on either -- a figure that
    contrasts P0 against P1 tops across an eps ladder needs both.
    """
    study_dir: pathlib.Path
    runs: dict = field(default_factory=dict)        # key -> RunData
    rows: list = field(default_factory=list)        # [(params dict, RunData)]
    swept: tuple = ()

    def by(self, key) -> RunData:
        k = str(key)
        if k not in self.runs:
            raise KeyError(f"study has no run for {key!r}; "
                           f"has {sorted(self.runs)}")
        return self.runs[k]

    def where(self, **params) -> list:
        """Every run whose swept parameters match, in sweep order."""
        out = [r for pr, r in self.rows
               if all(str(pr.get(k)) == str(v) for k, v in params.items())]
        if not out:
            raise KeyError(f"no run matches {params}; sweep had "
                           f"{[pr for pr, _ in self.rows]}")
        return out

    def one(self, **params) -> RunData:
        got = self.where(**params)
        if len(got) != 1:
            raise KeyError(f"{params} matches {len(got)} runs, wanted 1")
        return got[0]

    def axis(self, key: str) -> list:
        """The distinct values of one swept parameter, in sweep order."""
        seen = []
        for pr, _r in self.rows:
            v = pr.get(key)
            if v not in seen:
                seen.append(v)
        return seen


def load_study(study_dir: str | pathlib.Path) -> Study:
    d = pathlib.Path(study_dir)
    meta = json.loads((d / "study.json").read_text())
    swept = meta.get("swept", ())
    s = Study(d, swept=tuple(swept) if isinstance(swept, list) else (swept,))
    for row in meta.get("rows", []):
        run = load_run(d / row["dir"])
        s.rows.append((row["params"], run))
        if len(s.swept) == 1:
            s.runs[str(row["params"][s.swept[0]])] = run
    return s
