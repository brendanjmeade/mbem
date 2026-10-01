"""What a run IS: the spec objects, the builder registry, and the loader.

A study is a Python module, because the thing being specified is partly code.
Geometry is mesh generation and boundary conditions are `Patch` objects, and
restating those as a data schema would state the fault sign convention and the
eps rule a SECOND time -- conventions this codebase keeps in exactly one place
each, precisely because the second statement is the one that goes wrong (and at
nu = 1/4 a sign error in the slip pairing is invisible). So a config NAMES A
BUILDER and supplies its parameters; the builder stays where it is, and the
gates and the studies provably construct the same model because they call the
same function.

``None`` MEANS "DO NOT PASS THIS", so the callee's own default applies. That
matters more than it looks: ``HBackend.__init__`` and ``AssembledH.solve`` bind
their defaults from ``mbem.defaults`` at def time, so those values are frozen
into the signature and CANNOT be moved by rebinding ``defaults``. A config that
offered to override them that way would be a silent lie. Overriding one means
passing the keyword, which is what these dataclasses do -- and a field left
``None`` is absent from the call, so ``defaults`` stays the single source of
the number.

What a config may NOT set: the gate-only acceptance criteria in ``defaults``
(``BENCH_*``, ``*_PARITY*``, ``GMRES_ITER_CEILING``). Those are how a gate
decides PASS, not inputs to a solve, and a run that could move them could
declare its own success.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------

EpsSpec = float | str | Mapping[str, Any]      # scalar | "auto" | {patch: spec}

BACKENDS = ("dense", "hmat", "fmm")


@dataclass(frozen=True)
class Material:
    """GPa. Coefficients always come from (mu, lam), never via 1/(1-2nu)."""
    mu: float
    lam: float

    def build(self):
        from mbem import ElasticMaterial
        return ElasticMaterial(mu=self.mu, lam=self.lam)


@dataclass(frozen=True)
class Geometry:
    """Which mesh builder, and its parameters. ``scale`` divides every edge."""
    builder: str | Callable
    params: Mapping[str, Any] = field(default_factory=dict)
    scale: float = 1.0


@dataclass(frozen=True)
class Model:
    """Geometry plus the model builder that puts boundary conditions on it.

    ``eps`` is stated ONCE here and is passed by the runner both to assembly
    and to every evaluation. The evaluate_* functions take eps again,
    independently of what the operator used, which is a standing way for the
    two to drift apart; one field closes it.
    """
    geometry: Geometry
    builder: str | Callable
    params: Mapping[str, Any] = field(default_factory=dict)
    eps: EpsSpec = "auto"


@dataclass(frozen=True)
class Backend:
    """``kind`` picks the operator; every other field is a kwarg or None.

    tol/min_leaf/eta/max_admissible/min_aca govern far="aca" only;
    fmm_order/domain/m2l govern far="fmm" only. Passing the other family's
    knobs is accepted and has no effect, which is why both are named here.
    """
    kind: str = "hmat"                  # "dense" | "hmat" | "fmm"
    jump: str = "calibrated"
    deflate: bool = False
    dense_mode: str = "direct"          # dense only
    tol: float | None = None
    min_leaf: int | None = None
    eta: float | None = None
    max_admissible: int | None = None
    min_aca: int | None = None
    storage: str | None = None
    sweep: bool = False
    fmm_order: Mapping[str, int] | None = None
    domain: str | None = None
    m2l: str | None = None
    verbose: bool = False

    def kwargs(self) -> dict:
        """The HBackend keywords, with every unset field omitted."""
        out = {"jump": self.jump, "deflate": self.deflate,
               "sweep": self.sweep, "verbose": self.verbose,
               "far": "aca" if self.kind == "hmat" else "fmm"}
        for name in ("tol", "min_leaf", "eta", "max_admissible", "min_aca",
                     "storage", "fmm_order", "domain", "m2l"):
            v = getattr(self, name)
            if v is not None:
                out[name] = dict(v) if isinstance(v, Mapping) else v
        return out


@dataclass(frozen=True)
class Solve:
    """1:1 with AssembledH.solve. None = the method's own default."""
    rtol: float | None = None
    restart: int | None = None
    maxiter: int | None = None
    precond_max_dense: int | None = None
    precond_above_dense: str | None = None
    precond_hodlr_max: int | None = None
    precond_bj_chunk: int | None = None
    recycle: bool | None = None

    def kwargs(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None}


@dataclass(frozen=True)
class State:
    """One material state of a run.

    Several states share ONE assembly through ``rebuild_for_materials``, which
    is the whole reason they are states of a run rather than separate runs: the
    compressed geometry is reused. Anything that changes the MESH is a
    different run, because it is a different operator.
    """
    label: str
    materials: Mapping[str, Material] = field(default_factory=dict)


@dataclass(frozen=True)
class Output:
    """``slots`` empty means every slot. At 4M unknowns that is tens of GB per
    state, so a large run should name the slots it actually wants."""
    slots: tuple[str, ...] = ()
    save_fields: bool = True
    figures: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    name: str
    model: Model
    backend: Backend = field(default_factory=Backend)
    solve: Solve = field(default_factory=Solve)
    states: tuple[State, ...] = (State("base"),)
    outputs: Output = field(default_factory=Output)
    notes: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if self.backend.kind not in BACKENDS:
            raise ValueError(f"backend.kind {self.backend.kind!r} "
                             f"not in {BACKENDS}")
        if self.backend.jump not in ("half", "calibrated"):
            raise ValueError(f"jump {self.backend.jump!r}")
        if self.model.geometry.scale <= 0:
            raise ValueError(f"scale {self.model.geometry.scale} must be > 0")
        if not self.states:
            raise ValueError("a run needs at least one state")
        labels = [s.label for s in self.states]
        if len(set(labels)) != len(labels):
            raise ValueError(f"duplicate state labels: {labels}")


# ---------------------------------------------------------------------
# the builder registry
# ---------------------------------------------------------------------

# Values are STRINGS, resolved on demand: importing this module must not pull in
# triangle, matplotlib or a mesh generator, so that `list` and a validation-only
# pass stay instant.
MESH_BUILDERS = {
    "fault_box": "mbem.cases.registry:fault_box_meshes",
    "topo_inclusion": "mbem.cases.registry:topo_inclusion_meshes",
}
MODEL_BUILDERS = {
    "fault_box": "mbem.cases.registry:fault_box_model",
    "topo_inclusion": "mbem.cases.registry:topo_inclusion_model",
}


def resolve(ref: str | Callable, table: Mapping[str, str]) -> Callable:
    """A callable from a callable, a ``module:qualname``, or a registry name."""
    if callable(ref):
        return ref
    target = table.get(ref, ref)
    if ":" not in target:
        raise KeyError(f"unknown builder {ref!r}; known: {sorted(table)}")
    mod, _, qual = target.partition(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def builder_name(ref: str | Callable) -> str:
    """How a builder is recorded in the run's resolved dump."""
    if callable(ref):
        return f"{ref.__module__}:{ref.__qualname__}"
    return str(ref)


# ---------------------------------------------------------------------
# loading a config module
# ---------------------------------------------------------------------

def load(path: str | pathlib.Path, **params) -> Run:
    """The ``Run`` a config module declares.

    The module must define exactly one of ``RUN`` (a fixed study) or
    ``run_spec(**kwargs) -> Run`` (a parameterized one). Exactly one, because
    "both" leaves which one wins implicit, and the parameters of ``run_spec``
    ARE the study's command-line surface -- ``--set k=v`` needs no registration
    and can be listed straight off the signature.
    """
    path = pathlib.Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"_cfg_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    has_run, has_fn = hasattr(mod, "RUN"), callable(getattr(mod, "run_spec", None))
    if has_run == has_fn:
        raise ValueError(
            f"{path.name} must define exactly one of RUN or run_spec(); "
            f"found {'both' if has_run else 'neither'}")
    if has_fn:
        run = mod.run_spec(**params)
    else:
        if params:
            raise ValueError(f"{path.name} defines RUN, so it takes no --set "
                             f"parameters (given {sorted(params)})")
        run = mod.RUN
    if not isinstance(run, Run):
        raise TypeError(f"{path.name} produced {type(run).__name__}, not Run")
    return run


def parameters(path: str | pathlib.Path) -> dict:
    """``{name: (annotation, default)}`` a config's ``run_spec`` accepts."""
    import inspect
    path = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"_cfg_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "run_spec", None)
    if not callable(fn):
        return {}
    out = {}
    for name, p in inspect.signature(fn).parameters.items():
        ann = p.annotation if p.annotation is not inspect._empty else None
        out[name] = (getattr(ann, "__name__", str(ann)) if ann else "any",
                     None if p.default is inspect._empty else p.default)
    return out
