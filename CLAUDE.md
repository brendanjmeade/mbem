# CLAUDE.md

Guidance for Claude Code in this repo: a research package for the **mollified
boundary element method (mbem)** and mollified elastic dislocation theory, built
on the Cortez regularization `r -> sqrt(r^2 + eps^2)` of the Kelvin/Somigliana
kernels. 3-D linear elasticity, full space, no half space, no viscoelasticity,
no LaTeX. `README.md` has the physics and references; `BACKLOG.md` is the one
status document and history lives in `git log`.

Installed, not path-hacked: `pip install -e .` once, then everything imports as
a package from anywhere. If you find yourself writing `sys.path.insert`, that
is the bug.

## Map

| path | what it is |
|---|---|
| `src/mbem/` | **The trunk.** Region-graph model, P0/P1/P2 nodal patches and faults, dense and block-compressed backends (`far="aca"` flat H + ACA, `far="fmm"` Chebyshev bbFMM), preconditioned FGMRES, elastic-stress readout with the exact eigenstress removed. New solver or kernel work goes here. |
| `src/mbem/cases/` | The reference models the gates and the studies share: `fault_box`, `inclusion`, `topo_inclusion`. Library code, because 13 gates and 6 studies build from them. |
| `src/mollified_kernel/`, `src/moss_kernel/` | **Two frozen, INDEPENDENT copies** of the analytic mollified kernels. The difference is the point: the gates compare them entrywise, so neither may be "fixed" to agree with the other. `moss_kernel` alone defines `analytical_eigenstress_kernel` and `eigenstress_batch`. |
| `src/clq/` | **Frozen oracle.** Closed-form mollified kernels on one triangle for P0/P1/P2 nodal density, numpy. Separate derivation, so it is the parity reference that is genuinely independent. No development. |
| `src/mollified_bem.py` and friends | Frozen legacy oracles, kept as top-level modules: `ElasticMaterial` and `TriMesh` are defined in `mollified_bem.py` and re-exported from `mbem`, plus `anelastic`, `tde_reference`, `local_box_mesh*`, `inclusion_mesh`. |
| `tests/` | `run_all.py` (the authoritative runner) and `test_gates.py` (pytest over the same set). 44 gates in `gates/{mbem,clq,moss_kernel}`. |
| `configs/` | A study is a Python module declaring `RUN` or `run_spec(**kwargs) -> Run`. It NAMES a builder rather than describing patches, so it cannot restate the fault sign or the eps rule; `verify_config` proves the config path and the gates build the same model, fault Burgers vector included. |
| `runs/` | One self-describing folder per run (gitignored): `resolved.json` (spec, effective kwargs, resolved eps per patch, all 110 defaults, environment), `report.json`, `fields_<state>.npz`, `STATUS`, `MANIFEST`. |
| `studies/` | Runnable demos and `bench_scaling.py`, the performance harness to run before and after touching assembly, compression or evaluation. It keeps its own provenance helpers deliberately, so its committed `bench-json:` baselines stay comparable. |
| `docs/` | `figures/` (the curated, tracked PNG gallery), `clq.md`, `clq-derivation.md`. |
| `ddbem/`, `fbem/` | Closed. `FINDINGS.md` only: the P0/P1/P2 convergence study the higher-order patches rest on, and why the force-element BEM was dropped. Do not rebuild either without reading it. |

The paper and its public reproducibility package are **not here**: `moss/`
(manuscript, `mhf/`) and `medt_paper/` were moved to
`~/Desktop/moss-org-paper-archive/` and are recoverable from history and from
tag `medt_paper-vendored-2026-09-17`. `medt_paper` is published separately
(Zenodo DOI); publishing is a manual export, **never a push**.

## Running

```bash
pip install -e .                      # once; no other installer

python -m mbem run configs/fault_box.py          # a study -> a new runs/ folder
python -m mbem run configs/topo_inclusion.py --set surface=flat --set backend=fmm
python -m mbem run configs/fault_box.py --dry-run   # validate only, build nothing
python -m mbem list                              # runs, newest first
python -m mbem show <run-dir> --section effective
python -m mbem publish <run-dir>                 # figures -> docs/figures, with provenance
python -m mbem verify [-k fmm] [--fast]          # the gates

python tests/run_all.py               # all 44 gates, exit 1 on any FAIL
pytest -m "not slow"                  # same set, pytest front end
python studies/mbem/demo_fault_only.py
```

`/Users/meade/micromamba/bin/python` on this machine. Numba compiles on first
call; demos run at paper resolution. `cutde` is needed by two gates, which FAIL
rather than skip without it.

## Rules

1. **Oracles are frozen.** `src/{mollified_kernel,moss_kernel,clq}` and the
   legacy flat modules are never "improved"; new code is gated against them by
   entrywise parity. A kernel, mesh or assembly change adds a gate.
   `verify_oracle_provenance` pins every oracle by resolved path AND sha256, so
   a deliberate edit must update `oracle_manifest.json` in the same commit.
2. **Every gate prints `PASS:`/`FAIL:` at column 0 as its last such line and
   exits 1 on FAIL.** Gates are spawned as subprocesses, by file path — never
   imported into a shared interpreter, because the two kernel copies would
   contend for one module identity and the `defaults` rebinds would leak.
3. **The fault sign is stated once** (`FAULT_ORIENTATION`, `model/core.py`) and
   read through `RegionModel.orientation`. `Patch.value` on a fault is the
   Burgers vector `b = u(+n) - u(-n)`, the same sign as `clq` and cutde. Never
   write a `+-1` for it anywhere; `selfcheck` pins the direction.
4. **Every mollified double layer carries an eigenstress** `C:eps*` of its
   smeared jump — fault slip and boundary `u_p` alike. Removed in the stress
   readout (`evaluate_stress`, default), never in the solve, per element with
   that element's eps. Stress presented as elastic must have it subtracted.
5. **eps is per source element.** Scalar, `(N_src,)` array, per-patch dict or
   `"auto"` (0.1 h on a boundary patch; on a FAULT one value, 0.07 min h,
   because per-element widths smear a uniform slip unequally). What governs the
   error is a collocation point's clearance from element edges in units of eps,
   not h: budget eps/h before expecting refinement or higher order to pay.
6. **`jump="calibrated"`** is the default on both backends. `"half"` with
   eps/h > 0.5 is non-convergent and warns. Calibrated on an all-Neumann model
   needs `deflate=True`; the backends refuse otherwise.
7. **The slip -> displacement pairing** is `U_ij = -[mu n_m dG_ij/dx_m + lam
   n_j dG_im/dx_m + mu n_m dG_im/dx_j]` (slip and normal on C's first index
   pair). A lam/mu swap is invisible at nu = 1/4, so kernel gates run at
   nu != 1/4.
8. **Material coefficients come from `(mu, lam)`**, never through a
   `1/(1-2nu)` intermediate.
9. **Numba `parallel=True` kernels are never called from Python threads**
   (macOS workqueue crash); the `*_serial` nogil variants exist for that.
10. **Numbers live in one place:** tolerances and thresholds in
    `mbem/defaults.py`, the collocation free term in `equations.py`, kernel
    identifiers in `kernels/__init__.py`. A convention written twice is a bug.
    A config overrides a number by PASSING THE KEYWORD, never by rebinding
    `defaults` — the backends bind their defaults at def time, so a rebind is a
    silent no-op there. A field left `None` is absent from the call, so
    `defaults` stays the source. Gate-only criteria (`BENCH_*`, `*_PARITY*`)
    are not reachable from a config at all: a run that could move them could
    declare its own success.
11. **Lean.** Docstrings state the rule and the reason; measurements and
    history go in commit messages; no probe scripts in the tree; extend a gate
    before adding one; no new top-level documents.
12. Units: km, GPa; slip 0.001 km = 1 m; `(N, 3)` arrays.
13. **Higher-order patches:** `Patch.order` in {0, 1, 2}, discontinuous nodal
    layout in clq's node order, collocation at the shrunk nodes, the free term
    the shape-function matrix `N_k(x_c)` times the per-point diagonal. The
    compressed backend is P0-only.
14. **Git.** One local repo, no remote; the user commits on request.
15. Figures follow the house matplotlib style (`studies/*/\_paper_style.py`, and
    the `matplotlib-figure-style` skill). A study writes its figure beside
    itself; `docs/figures/` is the curated tracked gallery and changes only
    deliberately.
