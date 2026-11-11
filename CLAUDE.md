# CLAUDE.md

Guidance for Claude Code in this repo: a research package for the **mollified
boundary element method (mbem)** and mollified elastic dislocation theory, built
on the Cortez blob regularization of the Kelvin/Somigliana
kernels. 3-D linear elasticity, full space, no half space, no viscoelasticity,
no LaTeX. `README.md` has the physics and references; `BACKLOG.md` is the one
status document and history lives in `git log`.

Installed, not path-hacked: `pip install -e .` once, then everything imports as
a package from anywhere. If you find yourself writing `sys.path.insert`, that
is the bug.

## Map

| path | what it is |
|---|---|
| `src/mbem/` | **The trunk.** Region-graph model, P0/P1/P2 nodal patches and faults (slip prescribed, or solved from a traction condition: `BCType.FAULT_TRACTION`), the continuous P2 shared-node fault layout (`model/shared_nodes.py`), dense and block-compressed backends (`far="aca"` flat H + ACA, `far="fmm"` Chebyshev bbFMM), preconditioned FGMRES, elastic-stress readout with the exact eigenstress removed. New solver or kernel work goes here. |
| `src/mbem/cases/` | The reference models the gates and the studies share: `fault_box`, `inclusion`, `topo_inclusion`, `penny` (the penny-shaped crack, Eshelby and Sneddon: the traction-BC fault's anchor). Library code, because 14 gates and 7 studies build from them. |
| `src/mollified_kernel/`, `src/moss_kernel/` | **Two frozen, INDEPENDENT copies** of the analytic mollified kernels. The difference is the point: the gates compare them entrywise, so neither may be "fixed" to agree with the other. `moss_kernel` alone defines `analytical_eigenstress_kernel` and `eigenstress_batch`. |
| `src/clq/` | **Frozen oracle.** Closed-form mollified kernels on one triangle for P0/P1/P2 nodal density, numpy. Separate derivation, so it is the parity reference that is genuinely independent. No development. |
| `src/mollified_bem.py` and friends | Frozen legacy oracles, kept as top-level modules: `ElasticMaterial` and `TriMesh` are defined in `mollified_bem.py` and re-exported from `mbem`, plus `anelastic`, `tde_reference`, `local_box_mesh*`, `inclusion_mesh`. |
| `tests/` | `run_all.py` (the authoritative runner) and `test_gates.py` (pytest over the same set). 47 gates in `gates/{mbem,clq,moss_kernel}`. |
| `configs/` | Eight studies. A study is a Python module declaring `RUN` or `run_spec(**kwargs) -> Run`. It NAMES a builder rather than describing patches, so it cannot restate the fault sign or the eps rule; `verify_config` proves the config path and the gates build the same model, fault Burgers vector included. |
| `runs/` | One self-describing folder per run (gitignored): `resolved.json` (spec, effective kwargs, resolved eps per patch, every uppercase `defaults` constant, environment), `report.json`, `fields_<state>.npz`, `STATUS`, `MANIFEST`. |
| `src/mbem/volume.py`, `src/mbem/vti.py` | **Volumetric readout.** `mbem sample <run>` re-evaluates a stored solution on a 3-D grid and writes `.vti` (VTK ImageData, hand-rolled: meshio cannot write it and `vtk` is a 100 MB wheel). Post-processing only -- it rebuilds the model from the run's spec and skips the assembly and the solve. Each point is classified by `RegionModel.point_in_region` (the Gauss closure identity, so topography is handled by the warped mesh itself) and carries `clearance_h` / `clearance_eps`: the mollification band is FLAGGED, never blanked, because on this model it overstates the topography effect 33x. |
| `src/mbem/figures/` | The figure makers, named in a lazy registry so importing `mbem` never imports matplotlib. Three kinds, and the CLI dispatches on which: **model-free** (the kernels alone, `mbem figure KEY`), **run** (one solve), **study** (several runs, because the quantity is a difference between operators). `save_figure` is the one statement of the png+pdf convention the demos each carried a copy of. |
| `studies/` | `bench_scaling.py`, the performance harness to run before and after touching assembly, compression or evaluation — it keeps its own provenance helpers deliberately, so its committed `bench-json:` baselines stay comparable. Plus `demo_triangle_quickstart.py`, which prints numbers and draws nothing. `studies/clq/` holds the clq oracle's own demos. |
| `website/` | Astro static site (`npm run dev` / `npm run build` inside `website/`). `.github/workflows/deploy-website.yml` deploys it to GitHub Pages on every push to `main` that touches `website/**`, so pushing there publishes it. Nothing is computed in the browser: `public/data/` is a quantised readout of one stored run, made by `mbem sample <study-dir> --difference` and then `website/scripts/make_payload.py`. The encoder is `mbem.webexport`, kept in the package so that `verify_volume` [g] gates it. Fields are log10 uint8 and never float textures; `website/README.md` explains why. |
| `docs/` | `figures/` (the curated, tracked PNG gallery), `clq.md`, `clq-derivation.md`, `clq-rules.md`. |
| `docs/viscoelastic.md` | The derivation the viscoelastic extension rests on (branch `ve1`; the Laplace route and the time-stepper it describes are NOT built yet). With elastic bulk and viscoelastic shear, every operator of a region is geometry times `{1, g} x {1, mu, 1/mu}`, `g = 1/(1-nu)`, which holds only for the eigenstress-removed stress (rule 4) and is pinned by `verify_viscoelastic`. Also why a region that relaxes needs ONE scalar eps on its closed boundary, and the open floating-plate problem. |
| `docs/ddbem-findings.md` | Two CLOSED lines of investigation, kept because the live code rests on them: the free-term row-sum identity the calibrated diagonal implements, where to collocate P1/P2, and the measured "eps, not h, sets the error". |
| `docs/fbem-findings.md` | Why the force-element BEM was dropped (it stalls on free-traction rows, and higher order widens the gap). Do not rebuild it without reading this. |

The paper and its public reproducibility package are **not here, and not in
history either**: `moss/` (manuscript, `mhf/`) and `medt_paper/` were moved to
`~/Desktop/moss-org-paper-archive/`, and when this repo was published the two
were removed from every commit, because a push publishes history and the
manuscript is unpublished. Both are recoverable from that archive directory and
from the complete pre-filter mirror beside it,
`~/Desktop/moss-org-paper-archive/moss-org-prefilter.git` (77 commits, tag
`medt_paper-vendored-2026-09-17`). `medt_paper` is published separately
(Zenodo DOI); publishing is a manual export, **never a push**.

## Running

```bash
pip install -e .                      # once; no other installer
pip install -e '.[tde,test,viz]'      # + cutde, vtk, pytest/sympy/mpmath: gates FAIL without them

python -m mbem run configs/fault_box.py          # a study -> a new runs/ folder
python -m mbem run configs/topo_inclusion.py --sweep surface=topo,flat
python -m mbem run configs/onfault_stress.py --sweep eps=4,2,1 --sweep order_top=0,1
python -m mbem run configs/penny.py --sweep a_over_h=8,16 --sweep order=0,2   # the traction-BC crack
python -m mbem figure point_kernel                # model-free: no solve
python -m mbem run configs/fault_box.py --dry-run   # validate only, build nothing
python -m mbem list                              # runs, newest first
python -m mbem show <run-dir> --section effective
python -m mbem sample <run-dir> [--spacing 4]    # a 3-D grid -> .vti volumes
python -m mbem sample <study-dir> --difference   # every child on ONE grid, plus differences
python -m mbem publish <run-dir>                 # figures -> docs/figures, with provenance
python -m mbem verify [-k fmm] [--fast]          # the gates

python tests/run_all.py               # all 47 gates, exit 1 on any FAIL
python tests/run_all.py --suite clq -k eigenstress   # one suite, substring filter (--list, --fast)
python tests/gates/mbem/verify_penny_crack.py        # one gate, by path, as run_all spawns it
pytest -k penny_crack                 # one gate in-process, with a traceback
pytest -m "not slow"                  # same set, pytest front end
python studies/mbem/demo_triangle_quickstart.py   # prints numbers, draws nothing
```

The interpreter differs by machine (on the Mac it is
`/Users/meade/micromamba/bin/python`), so check `python -c 'import mbem'` before
running anything; on the Linux machine it is `~/miniforge3/envs/main/bin/python`.
numba must resolve a THREAD-SAFE threading layer: `numba.threading_layer()`
must be `"omp"` or `"tbb"` (Linux resolves `"tbb"`), never `"workqueue"`.
`triangle`'s PyPI sdist ships Cython output that does not compile on current
Pythons: regenerate it (`python -m cython -3 triangle/core.pyx` in the unpacked
sdist, then `pip install --no-deps .`) before `pip install -e .`.
Numba compiles on first call; demos run at paper resolution. `cutde` (two gates)
and `vtk` (`verify_volume`'s independent reader) are test requirements: those
gates FAIL rather than skip without them.

## Rules

1. **Oracles are frozen.** `src/{mollified_kernel,moss_kernel,clq}` and the
   legacy flat modules are never "improved"; new code is gated against them by
   entrywise parity. A kernel, mesh or assembly change adds a gate.
   `verify_oracle_provenance` pins every oracle by resolved path AND sha256, so
   a deliberate edit must update `oracle_manifest.json` in the same commit.
2. **Every gate is `main() -> bool`, prints `PASS:`/`FAIL:` at column 0 as its
   last such line, and exits on it.** One convention, all 47: the return value
   is the authority and the printed line cross-checks it, so a gate whose two
   disagree is reported as MISMATCH rather than trusted. Do not write a
   POSIX-style `return 0` for success — under `assert main()` that is exactly
   inverted. `pytest` imports the gates and asserts on the return value;
   `tests/run_all.py` spawns them and asserts on the exit code. A gate that
   must own its interpreter (`verify_oracle_provenance`, which checks WHICH
   file each oracle import resolves to) is listed in `test_gates.SUBPROCESS_ONLY`
   with the reason.
3. **The fault sign is stated once** (`FAULT_ORIENTATION`, `model/core.py`) and
   read through `RegionModel.orientation`. `Patch.value` on a fault is the
   Burgers vector `b = u(+n) - u(-n)`, the same sign as `clq` and cutde. Never
   write a `+-1` for it anywhere; `selfcheck` pins the direction.
4. **Every mollified double layer carries an eigenstress** `C:eps*` of its
   smeared jump — fault slip and boundary `u_p` alike. Removed in the stress
   readout (`evaluate_stress`, default) and in the traction rows of a
   `FAULT_TRACTION` fault (`KERNEL_HT`), never in a displacement row, per
   element with that element's eps. Stress presented as elastic must have it
   subtracted; a traction row that kept it would load the fault with a rigid
   translation of the body (`verify_penny_crack` [f]).
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
    the shape-function matrix `N_k(x_c)` times the per-point diagonal. Faults
    may instead be `layout="continuous"` (P2 only, `model/shared_nodes.py`):
    one value per SHARED node, collocation AT the nodes, no free term -- the
    kink is what mollification bounds, so the on-fault readout converges
    (1.2-1.8 eps/a, mesh-independent) where P0 centroids are floored. A
    `BCType.FAULT_TRACTION` fault (P0, or continuous P2) solves its slip from
    the elastic traction at its nodes (`equations.py` docstring): three
    Cartesian unknowns per node, the tangential rows plus `b . n = 0` unless
    `allow_opening`, the crack front locked by default, dense `mode="direct"`
    only. Continuous P1 (too stiff) and discontinuous P2 traction faults are
    refused. The compressed backend is P0, discontinuous, prescribed-slip only.
14. **Git.** `main` is the only branch; the public remote is
    `github.com/brendanjmeade/mbem`. The user commits on request.
15. Figures follow the house matplotlib style (`src/mbem/figures/style.py`, and
    the `matplotlib-figure-style` skill). A study writes its figure beside
    itself; `docs/figures/` is the curated tracked gallery and changes only
    deliberately.
