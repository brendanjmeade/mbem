# CLAUDE.md

Guidance for Claude Code in this repo: a research package for the **mollified
boundary element method (mbem)** and mollified elastic dislocation theory, built
on the Cortez blob regularization of the Kelvin/Somigliana
kernels. 3-D linear elasticity, full space, no half-space kernel, no LaTeX;
linear viscoelasticity by the Laplace route (rule 16); a solved fault under
a law that is not linear, stepped in time (rule 20);
gravity as buoyancy on the boundary (rule 17); a box whose walls are a
window on a layered half-space (rule 18). `README.md` has the physics and references; `BACKLOG.md` is the one
status document and history lives in `git log`.

Installed, not path-hacked: `pip install -e .` once, then everything imports as
a package from anywhere. If you find yourself writing `sys.path.insert`, that
is the bug.

## Map

| path | what it is |
|---|---|
| `src/mbem/` | **The trunk.** Region-graph model, P0/P1/P2 nodal patches and faults (slip prescribed, or solved from a traction condition: `BCType.FAULT_TRACTION`), the continuous P2 shared-node fault layout (`model/shared_nodes.py`), dense and block-compressed backends (`far="aca"` flat H + ACA, `far="fmm"` Chebyshev bbFMM), preconditioned FGMRES, elastic-stress readout with the exact eigenstress removed. New solver or kernel work goes here. |
| `src/mbem/cases/` | The reference models the gates and the studies share: `fault_box`, `inclusion`, `topo_inclusion`, `penny` (the penny-shaped crack, Eshelby and Sneddon: the traction-BC fault's anchor), `layered_box` (an elastic plate over a substrate with a fault from the surface -- vertical strike-slip in the plate unless `dip` and `rake` say otherwise, and THROUGH the interface if `fault_depth` exceeds the plate, its lower reach a second fault, `fault_lower`, in the substrate -- and what it is compared with: the antiplane closed forms, and for dip slip `dip_slip_profile`, a plane-strain reference under gravity that shares nothing with the boundary elements; plus `two_mesh_limit`, what a plate that bends needs; `sides="window"` makes every wall of the box carry the far field of the prescribed slip, `lower="creep"` makes the megathrust's lower reach a solved fault that creeps, `slab_depth` lets the plate subduct as an elastic slab the interface wraps, and a fault deeper than the plate then runs down the slab's upper face: `fault_lower` is there a part of the interface that SLIPS, rule 21). `layered_space` is the same layer system in THREE dimensions, for a finite rectangular fault at the surface and at depth: the reference a finite rupture is compared with, and what a window's walls are made of. `fault_box` takes `walls="window"` too. Library code, because the gates and the studies both build from them. |
| `src/mbem/rheology.py`, `src/mbem/laplace.py` | **Linear viscoelasticity, the Laplace route.** `rheology`: `Maxwell`, `Burgers`, `Andrade` as operational shear moduli, stated in relaxation TIMES and modulus ratios (never viscosities), bulk modulus elastic; a rheology with no fastest relaxation says so (`tail`, `rheology.tails`); the states of a Laplace run; the one-eps rule. `laplace`: the collocation points for a window of times, and `Prony`, the fixed-pole fit that turns the stack of per-s elastic solutions into value, rate, periodic-cycle sums and the velocity of a fault locked between repeating earthquakes, each as one linear map (with `tails`, a body's power-law end in one term; `run_fit` is the fit of a stored run). No kernel work: an operator at any s is the elastic one at `(mu(s), lam(s))`. Also `rheology.ViscousFault`: a solved fault that resists its slip rate is, at each s, a fault with a stiffness (`Laplace.faults`, `State.drag`), so afterslip needs no time-stepper either. `PowerLawFault` and `RateStrengtheningFault` are not linear: they have a `rate` and no `drag`, and a fault with one is stepped (rule 20). |
| `src/mbem/stepper.py` | **A solved fault through time, under a law that need not be linear** (rule 20). The body is linear, so each state is REDUCED TO THE FAULT (`AssembledDense.condense`: traction = `S slip + L`), the body's relaxation is carried as the Laplace fit's own exponentials (`Prony.coefficients`), and the faults alone are integrated (`march`, `scipy`'s `RK45`). `Body` holds the states and makes the fit's relaxed limit passive; `law_rate` turns `rheology` laws into a rate. `Run.stepper` is how a run asks for it; the result is `history.npz`. The Laplace route is untouched: its states are this one's precompute. P0 faults INSIDE a region need no such reduction (`backend=hmat` in a stepped run): `KernelBody` steps on `la/faultkernel.py`'s `FaultKernel`, the faults' own traction kernel as two flat-H matrices that serve every material, plus, in a body with a boundary, what the boundary returns of a slip in low rank (`la/coupling.py`'s `Coupling`: `P M_k W'`, of the rank the BOUNDARY's mesh gives it), the body factored densely (on a GPU where the run says so, `la/lu.py`: single-precision factors, every solve refined in double) or, past what a dense matrix holds, in blocks (`la/blr.py`: the compressed operator in block low-rank LU, whose `schur` is also what a factorization stopped early leaves). `rkc.py` is the integrator a fine fault wants: stabilised explicit Runge-Kutta-Chebyshev, the square root of an embedded pair's steps, and what a `KernelBody`, or any body whose fit has a tail, is stepped with. |
| `src/mollified_kernel/`, `src/moss_kernel/` | **Two frozen, INDEPENDENT copies** of the analytic mollified kernels. The difference is the point: the gates compare them entrywise, so neither may be "fixed" to agree with the other. `moss_kernel` alone defines `analytical_eigenstress_kernel` and `eigenstress_batch`. |
| `src/clq/` | **Frozen oracle.** Closed-form mollified kernels on one triangle for P0/P1/P2 nodal density, numpy. Separate derivation, so it is the parity reference that is genuinely independent. No development. |
| `src/mollified_bem.py` and friends | Frozen legacy oracles, kept as top-level modules: `ElasticMaterial` and `TriMesh` are defined in `mollified_bem.py` and re-exported from `mbem`, plus `anelastic`, `tde_reference`, `local_box_mesh*`, `inclusion_mesh`. |
| `tests/` | `run_all.py` (the authoritative runner) and `test_gates.py` (pytest over the same set). 49 gates in `gates/{mbem,clq,moss_kernel}`. |
| `configs/` | Nine studies. A study is a Python module declaring `RUN` or `run_spec(**kwargs) -> Run`. It NAMES a builder rather than describing patches, so it cannot restate the fault sign or the eps rule; `verify_config` proves the config path and the gates build the same model, fault Burgers vector included. |
| `runs/` | One self-describing folder per run (gitignored): `resolved.json` (spec, effective kwargs, resolved eps per patch, every uppercase `defaults` constant, environment), `report.json`, `fields_<state>.npz`, `STATUS`, `MANIFEST`. |
| `src/mbem/volume.py`, `src/mbem/vti.py` | **Volumetric readout.** `mbem sample <run>` re-evaluates a stored solution on a 3-D grid, or with `--section y=0` on one plane of it, and writes `.vti` (VTK ImageData, hand-rolled: meshio cannot write it and `vtk` is a 100 MB wheel). Post-processing only -- it rebuilds the model from the run's spec and skips the assembly and the solve. Each point is classified by `RegionModel.point_in_region` (the Gauss closure identity, so topography is handled by the warped mesh itself) and carries `clearance_h` / `clearance_eps`: the band near a patch is FLAGGED, never blanked: there a value rests on the readout near a face (rule 19), not on the solve alone. |
| `src/mbem/recover.py` | **The readout near a face** (rule 19). `recovery_operator`: a value per element -> a continuous P2 density, by a quadratic fitted over two rings of elements, never across a fault (`shared_nodes.split_nodes`) or a crease. `NearField`: what `evaluate_displacement`, `evaluate_stress` and `volume.sample` take as `near=`, built once for a model's geometry and good for every state. Post-processing: no solve, no stored run and no backend changes. |
| `src/mbem/figures/` | The figure makers, named in a lazy registry so importing `mbem` never imports matplotlib. Three kinds, and the CLI dispatches on which: **model-free** (the kernels alone, `mbem figure KEY`), **run** (one solve), **study** (several runs, because the quantity is a difference between operators). `save_figure` is the one statement of the png+pdf convention the demos each carried a copy of. |
| `studies/` | `bench_scaling.py`, the performance harness to run before and after touching assembly, compression or evaluation — it keeps its own provenance helpers deliberately, so its committed `bench-json:` baselines stay comparable. Plus `demo_triangle_quickstart.py`, which prints numbers and draws nothing. `studies/clq/` holds the clq oracle's own demos. |
| `website/` | Astro static site (`npm run dev` / `npm run build` inside `website/`). `.github/workflows/deploy-website.yml` deploys it to GitHub Pages on every push to `main` that touches `website/**`, so pushing there publishes it. Nothing is computed in the browser: `public/data/` is a quantised readout of one stored run, made by `mbem sample <study-dir> --difference` and then `website/scripts/make_payload.py`. The encoder is `mbem.webexport`, kept in the package so that `verify_volume` [g] gates it. Fields are log10 uint8 and never float textures; `website/README.md` explains why. A SIBLING site, <https://brendanjmeade.github.io/ve/>, shows the Laplace runs through time; its source is its own repo (`brendanjmeade/ve`), and only its numerics are here: the second half of `mbem.webexport` (`encode_modes`, `time_weights`, `time_tables`, `node_sheet`), gated by `verify_volume` [i]. Time is a map whose weights sum to ~200, so a state is never quantised and combined in a page: a stack is shipped as orthonormal modes with the weights already applied. |
| `docs/` | `figures/` (the curated, tracked PNG gallery), `clq.md`, `clq-derivation.md`, `clq-rules.md`. |
| `docs/viscoelastic.md` | The derivation viscoelasticity rests on. With elastic bulk and viscoelastic shear, every operator of a region is geometry times `{1, g} x {1, mu, 1/mu}`, `g = 1/(1-nu)`, which holds only for the eigenstress-removed stress (rule 4). The Laplace route built on it and its closed-form anchors; a solved fault stepped in time under a law that is not linear (section 6: the reduction to the fault, the closed forms it is held to, what it costs, the fault's own kernel compressed and a boundary's part of it in low rank, the stabilised integrator, and the boundary-memory design it replaced); why a region that relaxes needs ONE scalar eps; gravity as a condition on the boundary (section 7), the sphere it is gated against, and under it the thrust in the plate, the two-mesh limit that cancels what P0 costs a plate that bends, the earthquake cycle of a locked fault, and a megathrust through the interface; the walls of a box as a window, and the reference in three dimensions (section 8); a fault that creeps (section 9); a plate that subducts (section 10); and what is open (section 11: what a window's walls do not know, the time-stepper, slip on an interface, gravity's coupling with compression). |
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
python -m mbem run configs/plate_substrate.py --sweep half_x=400,800 --sweep sides=free,clamped   # a Laplace run
python -m mbem run configs/plate_substrate.py --set rho_plate=2700 --set rho_substrate=3300       # ... under gravity
python -m mbem run configs/plate_substrate.py --set dip=30 --set rake=90 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=10 --set t_max=100 --sweep refine=1,2 --sweep gravity=0,1                     # ... a thrust, as a two-mesh pair
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=10 --set near_field_radius=200 --set half_x=800 --set t_max=250 --sweep refine=1,2   # ... a megathrust through the interface
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=10 --set near_field_radius=200 --set fault_half_len=150 --set half_x=400 --set sides=window --set t_max=100 --sweep refine=1,2   # ... in a WINDOW: a finite rupture, a small box
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=10 --set near_field_radius=200 --set fault_half_len=150 --set half_x=400 --set sides=window --set t_max=100 \
    --set t_min=0.02 --set refine=2 --sweep tau_K=0.1,0   # ... on a BURGERS substrate, against its Maxwell twin: the window reaches below the transient
python -m mbem run configs/plate_substrate.py --set t_max=100 --sweep andrade_alpha=0.3333,0   # an ANDRADE substrate against its Maxwell twin: no fastest relaxation, a fit with a tail
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=15 --set edge_interface=22.5 --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window --set half_x=300 --set depth_box=200 \
    --set fault_half_len=100 --set t_max=100 --set backend=dense --set lower=creep --sweep creep_tau=1,1e6 --sweep refine=1,1.5   # ... its lower reach creeping: afterslip
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=15 --set edge_interface=22.5 --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window --set half_x=300 --set depth_box=200 \
    --set fault_half_len=100 --set t_max=100 --set backend=dense --set lower=creep --set route=step --sweep creep_law=viscous,friction,power   # ... STEPPED IN TIME, under laws that are not linear
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=15 --set edge_interface=22.5 --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window --set half_x=300 --set depth_box=200 \
    --set fault_half_len=100 --set t_max=100 --set backend=hmat --set lower=creep --set route=step --set creep_law=friction \
    --set eps_lower=0.3225 --sweep edge_lower=1.25,0.625   # ... the reach REFINED under the same box, on its compressed kernel: 35,128 and 142,092 slip unknowns
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=15 --set edge_interface=22.5 --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window --set half_x=300 --set depth_box=200 \
    --set fault_half_len=100 --set t_max=100 --set backend=hmat --set lower=creep --set route=step --set creep_law=friction \
    --set eps_lower=0.3225 --set edge_lower=1.25 --set refine=2 --set factor=blocks   # ... the BOX refined instead, its body factored in blocks: 44,085 boundary unknowns
#   ... --set lu=gpu on any of the dense runs above: the matrix factored in single precision on a CUDA card (cupy), every solve refined in double
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=30 --set rho_plate=3300 --set rho_substrate=3300 \
    --set edge_near=10 --set near_field_radius=150 --set sides=window --set half_x=500 --set depth_box=300 --set fault_half_len=100 --set t_max=100 --sweep slab_depth=0,150   # ... the plate subducting: a slab
python -m mbem run configs/fault_box.py --set walls=window     # the elastic box on a half-space
python -m mbem figure point_kernel                # model-free: no solve
python -m mbem run configs/fault_box.py --dry-run   # validate only, build nothing
python -m mbem list                              # runs, newest first
python -m mbem show <run-dir> --section effective
python -m mbem sample <run-dir> [--spacing 4]    # a 3-D grid -> .vti volumes
python -m mbem sample <study-dir> --difference   # every child on ONE grid, plus differences
python -m mbem sample <run-dir> --section y=0    # one plane: a vertical section at mid-strike
python -m mbem sample <run-dir> --plain          # the element-constant readout everywhere (rule 19)
python -m mbem publish <run-dir>                 # figures -> docs/figures, with provenance
python -m mbem verify [-k fmm] [--fast]          # the gates

python tests/run_all.py               # all 49 gates, exit 1 on any FAIL
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
   last such line, and exits on it.** One convention, all 49: the return value
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
14. **Git.** `main` is the published branch (a push to it that touches
    `website/**` deploys the site), so work happens on a branch and reaches
    it by a pull request. The public remote is `github.com/brendanjmeade/mbem`. The user
    commits on request.
15. Figures follow the house matplotlib style (`src/mbem/figures/style.py`, and
    the `matplotlib-figure-style` skill). A study writes its figure beside
    itself; `docs/figures/` is the curated tracked gallery and changes only
    deliberately.
16. **Viscoelasticity is the Laplace route** (`docs/viscoelastic.md`): a run
    with `Run.laplace` solves the elastic state and one elastic problem per
    real s, its states DERIVED from the model the builder returns, and
    `mbem.laplace.Prony` maps the stack to time. Shear relaxes and the bulk
    modulus does not. A rheology is relaxation times and ratios; time has the
    unit its tau has. **A body with no fastest relaxation is fitted with its
    tail** (`rheology.Andrade`, whose spectrum runs on to short times as a
    power law): the plain fit is 3-21 % off THROUGH the window and more
    points a decade do not help, so `Prony(tails=rheology.tails(...))` puts
    every pole beyond the grid in one term, at no more solves (1e-4 to
    7e-3). **The tail is the spectrum's end only at times short of
    `tau_A`**: a window that stops above it is 2-14 % off in stress (a
    tenth of that in displacement), so `t_min <= tau_A`, and a run that
    breaks it is warned of, not refused (`rheology.warn_window`).
    **The fit of a stored run is `laplace.run_fit(run_dir)`**, never
    `Prony(s)`: a consumer that builds its own gives such a body the plain
    fit and still looks smooth (`webexport`'s tables take `tails` with no
    default). **A region that relaxes carries ONE scalar eps on its
    closed boundary** (`"auto"` loses its pressure response as mu(s)/K falls
    below ~1e-3); the runner refuses otherwise. A traction-driven problem in a
    fluid creeps and needs `Prony(secular=True)`. The box is part of the
    model unless its walls are a window (rule 18): there is no half-space
    kernel, a plate with free sides floats as the substrate relaxes, and a
    result is trusted only inside the fit's window of times.
    **A plate that BENDS is only as good as its two faces:** dip slip over a
    relaxed substrate is short in the vertical by an amount first order in
    the P0 elements of the top and the interface and in the interface's eps
    (12 % at 10 km / 15 km / 1.5 km on a 30 km plate, 6 % at half of each;
    `verify_viscoelastic` [n] pins it). Strike slip does not bend the plate
    and never showed it, so budget those three before trusting a thrust --
    or cancel it: the error is first order, and two meshes with the three in
    ONE ratio (`--sweep refine=1,2`, `layered_box.two_mesh_limit`) leave
    3 % where they carry 14 % and 8 %. It removes that error and no other.
    **A locked fault's cycle** is `Prony.locked_rate_weights`: velocity
    measured from steady slip, per unit slip rate. Its mean over the cycle
    is elastic back slip whatever relaxes, and it needs a window of a few
    periods, not a settled body.
    **A fault that cuts the plate hands its relaxation to the walls.** A
    megathrust through the interface has two halves that converge as rigid
    blocks once the substrate lets go: clamped walls hold them (0.81 of the
    reference's convergence from 800 km away, 0.61 from 400) and free ones
    let them go too far (1.11). `sides="window"` mends it (rule 18): 2 % in
    the horizontal, and with the two-mesh limit for the vertical the
    megathrust is within 1.6 % of the reference in time and 0.5 % through
    its cycle, in a box 400 km to its walls.
    **A fault that creeps** is `rheology.ViscousFault` on a SOLVED fault
    (`Laplace.faults`; `layered_box` `lower="creep"`): at each s a fault with
    a stiffness (`State.drag`, infinite in the elastic state, where it is
    locked), handed to `equations.apply_fault_rows` by the dense backend,
    which is the only one that solves a slip. Its pairs of meshes are too
    close for a two-mesh limit.
    **A slab** (`slab_depth`) is the plate's own region continued down dip.
    Give plate and mantle one density: a slab of a lighter plate hanging in
    a heavier mantle is unstable once anything relaxes, and the run is
    refused. Below the plate's base the megathrust runs down the slab's
    upper face, an interface that slips (rule 21).
    **A state is the base model with its own materials**, never cumulative,
    and is read back through `RegionModel.with_materials`.
17. **Gravity is a condition on the boundary** (`model.Gravity`, held as
    `RegionModel.gravity`; `docs/viscoelastic.md` section 7): every region a
    density, the free surface NAMED. Region R sees on patch p the traction
    `t_p + B u_p`, `B = -(rho_R - rho_ref) g n (x) up`, stated once in
    `RegionModel.buoyancy`: `rho_ref` is 0 on the free surface, the first
    region's density on an interface, and `rho_R` (no term) on every other
    outer face -- the sides and base of a box are cuts, not cliffs. No term on
    a fault. It is exact for an incompressible material and off by
    `rho g L / K` otherwise, and it is NOT a body force: nothing enters the
    volume or the kernels. It is a `BlockTerm` with `col_op = B`, a kernel
    block on a transformed column, so **every consumer of `system.terms` goes
    through `term.col_vector` / `term.col_block`**; one that reads `x[col]`
    itself drops gravity from that piece and still converges. `Patch.value` on
    the free surface is the applied load; `evaluate_stress` returns the stress
    at a fixed point of space (material stress minus `rho g (u . up) I`).
    `deflate` is refused under gravity, and a Laplace run refuses a contrast
    heavy side up (a relaxing body has a growing mode there). The exact
    reference is a sphere with RADIAL buoyancy (`Gravity(center=...)`).
18. **The walls of a box are a window, and a datum is read through the
    model** (`model.FarField`, held as `RegionModel.far_field`;
    `docs/viscoelastic.md` section 8). Named PRESCRIBED_DISPLACEMENT walls
    carry the displacement the model's own PRESCRIBED slip produces in an
    unbounded reference medium (`cases.layered_space`: a plate over a
    half-space, a finite rectangular fault, three dimensions). The data
    depend on the materials, so on the state: **every prescribed datum is
    read through `RegionModel.value_of(patch)`**, never `patch.value_array()`
    -- a consumer that reads the patch gives a window's walls zeros and still
    converges -- and a backend takes a state's right-hand side from
    `equations.rhs_terms_at`. `with_materials` carries the far field to the
    state it builds. A window is exact for what its reference solves and
    blind to the far field of anything else: a solved slip, a slab. **A
    fault is as long as it is:** a study with a window among its runs is
    compared with the reference in three dimensions, not with plane strain,
    which an 800 km megathrust misses by 5-10 % once the substrate relaxes.
19. **Near a face the field is read from a continuous displacement**
    (`recover.NearField`, the `near=` of the evaluators and of
    `volume.sample`; `mbem sample` uses it unless `--plain`). Within half an
    element of a boundary patch the readout resolves the element-constant
    displacement it integrates, each edge a small dislocation line: 19 % in
    stress at 0.1 h with the EXACT value on every element, so no solver that
    returns a value per element removes it and none is changed. Three parts,
    and each is needed: the DISPLACEMENT of every order-0 boundary patch is
    recovered as a continuous P2 density (not across a fault, where it
    jumps); **the traction is left as solved** -- where a fault cuts an
    interface each region's traction has a real spike, and smoothing it moved
    the megathrust's interior 7-9 %; and the mollification of the recovered
    patches is narrowed to a fiftieth (the blob's tail costs (eps/d)^2 well
    beyond the layer itself), which is **safe only with the recovered
    density** (element values read at a narrower width are worse than
    before). Only the pairs within two elements are read this way, so a
    section costs an eighth more, and the far field is the same sum, bitwise. A
    point within a tenth of an element of a face -- a grid row lying on one
    -- is read from that far inside its region (closer, the recovered
    density's kinks and the traction's steps are salt-and-pepper at the
    narrowed width), and WHICH region
    a point on a face is in is a rule, not the solid angle (0 or 4 pi there
    as rounding has it): the first region that owns the patch
    (`volume.classify`). Faults are untouched: a fault's width is the model's
    own.
20. **A fault whose law is not linear is stepped in time, and only the
    fault is** (`mbem.stepper`, `Run.stepper`; `docs/viscoelastic.md`
    section 6). The body is linear, so at each state it is reduced to its
    solved faults (`AssembledDense.condense`: `traction = S slip + L`, the
    traction being what a fault's rows hold to zero, eigenstress removed,
    less its prescribed change), and in time `S` is the Laplace fit's own
    exponentials, one decaying vector per pole: the Laplace route's states
    are the stepper's precompute, and a run costs what the linear one
    costs. That reduction is the dense backend's (the compressed kernel
    below is the other way to a body); every solved fault is given a law,
    and a law is in `stepper.faults` or `laplace.faults`, never both. Three things it rests
    on. **The fit's relaxed limit is made passive** (`Body`): as fitted it
    misses by 1e-3 with the wrong sign where a fault sits in a region that
    relaxes completely, and modes GROW by 2-5 % over the window. **The
    window is the body's, the step is the fault's**: the Laplace points
    must reach below the body's fastest relaxation and past the last time
    asked for (`march` refuses a later one), while the step is bounded by
    the fit's fastest pole, not by the law. A body with no fastest
    relaxation (rule 16) hands the stepper its fit's tail as memory too: a
    decade of it, three poles more, and the rest gone at once
    (`Prony.memory`; ten times the fastest rate, so `march` gives such a
    body RKC). **The last decade of a long window is the relaxed limit's:**
    where the limit pushed and was moved, a stepped result is off by two to
    five times the report's `removed_over_elastic` at the window's end
    (0.3 % on the megathrust's reach in an Andrade substrate at 100 tau,
    up to 3 % for a stronger transient) and at the fit's error to a tenth
    of the window. **A law that is not linear
    knows how large the earthquake was**: a stepped run is not per unit
    slip. A law is a `rate` for a traction in the fault's row frame, with no
    state variable yet; its result is `history.npz`, every slot at every
    time, and no state to `mbem sample`.
    **P0 faults inside a region step on their own compressed kernel**
    (`stepper.KernelBody`, `la.faultkernel.FaultKernel`; `backend=hmat`):
    the kernel is `mu` times a function affine in `g`, so two compressed
    matrices at the ends of the run's range of `g` are every state, and a
    stage is two products whatever the number of states. Its far blocks
    are built by SPAWNED processes, the traction kernel being a parallel
    numba function (rule 9): a script that starts a build keeps its work
    under `if __name__ == "__main__"`. **In a body with a boundary** the
    faults' stiffness has a second part, what the boundary returns of a
    slip, `-A_fb A_bb^-1 A_bf`. As a matrix on the fault it is not low
    rank; as a product THROUGH the boundary it is, at a rank set by the
    boundary's mesh and not the fault's (`la.coupling.Coupling`: 1,193 and
    1,764 at 1e-6 under 17,778 boundary unknowns, the same at 4,260 and at
    68,024 slip unknowns), so a state is the body's factorization and one
    solve per column of a basis, not per slip unknown; the fault stops
    being the limit (71,046 elements on a megathrust's reach: 22 minutes,
    35 GB). **The body is factored densely, or in blocks** (`la.blr`,
    `Stepper.factor`; by size, `defaults.STEPPER_DENSE_BOUNDARY`): the
    compressed backend's operator cut into clusters that hold every slot
    of their patch, block low-rank LU with pivoting inside a diagonal
    block, truncated PER BLOCK at the tolerance the operator is compressed
    at (2e-6 to 4e-6 against dense LU at 1e-6, on the megathrust's body at
    its softest). It costs N^2 where dense LU costs N^3 and is the slower
    one under some 60,000 unknowns. **A dense matrix may be factored on a
    card** (`backend.lu = "gpu"`, `la.lu`; any dense solve, a Laplace
    state's as much as a stepped body's): the factors in SINGLE precision,
    every solve refined against the double matrix on the host and stopped
    on the size of its correction, because a workstation card has no fast
    double (2 s for LAPACK's 42 at 44,085 unknowns, up to the 107,000 a
    48 GB card holds; a state's fields the same to 3e-10, a stepped
    history to 8e-8). A card is ASKED FOR and never looked for: unasked the
    factors are LAPACK's double on every machine, so a result does not
    depend on where it was made, and a run that asks where none answers is
    refused. A matrix whose corrections do not contract is factored in
    double after all and says so, and what a state then costs is its
    ASSEMBLY. **A stepped afterslip is only as good
    as the box's faces:** with the plate's two faces and the substrate's
    width halved the reach's mean afterslip is a third more early and an
    eighth more late (0.235 of the slip at 100 tau for 0.208), under
    either factorization; quote the box with the number. Not for a slip
    ON an interface (rule 21), whose stiffness is all the body's. **Two things
    about the fit's relaxed limit there.** The kernel's is two numbers,
    corrected in `H_lo` and `H_hi - H_lo` and NOT weight by weight (the
    two fitted limits all but cancel). And where the faults' region has
    relaxed to a FLUID within the fit's error the whole limit is moved to
    the slowest pole, kernel and boundary alike, because a fluid carries
    no shear; where it has not, the boundary's part is left as fitted and
    the run warns that nothing checks it.
    **A fine fault is stiff, and `mbem.rkc` is its integrator.** An
    embedded pair's step is bounded by the fault's shortest wavelength
    (4.3 times the evaluations for four times the elements a radius), a
    Chebyshev method's stages buy stability instead of order, and the
    count then grows as the square root or not at all (3,342 and 3,355
    evaluations at 2,164 and 142,092 slip unknowns of the megathrust's
    reach). `march` gives a `KernelBody` RKC and a dense `Body` the pair,
    which is as good or better on a small fault, unless its fit has a tail
    (30,044 evaluations under the pair for RKC's 4,823 on the megathrust's
    reach in an Andrade substrate, one answer); the report says which.
21. **Slip between two regions is slip ON AN INTERFACE, not a fault**
    (`Patch.slip` on an INTERFACE patch, `"prescribed"` or `"solved"`;
    `model/equations.py`; `verify_interface_slip`). The traction stays one
    unknown and the displacement jumps by `b = u(+n side) - u(-n side)`,
    the fault convention: the slot `u` is the -n side's, and the region the
    stored normal points INTO (`RegionModel.slip_side`) sees `u + b`, in its
    boundary rows and in **every readout, which therefore asks for a
    patch's displacement AS A REGION SEES IT** (`evaluate._density(...,
    region)`): one that read the slot alone would read the wrong side and
    still look smooth. A solved slip's rows are algebraic, `t - kappa b =
    dt` in the plane and `b . n = 0`, with no kernel and no eigenstress,
    and otherwise it is a FAULT_TRACTION fault (`RegionModel.solved_faults`,
    `BlockSystem.traction_faults`: a drag, the reduction, the stepper).
    Order 0, dense backend, and ONE density across it (a buoyant slipping
    interface is refused). `layered_box` uses it where the megathrust
    leaves the plate under a slab: `fault_lower` is then a part of the
    slab's upper face, with the megathrust's normal (its orientation is
    STATED, the plate with its slab not being convex) and the substrate's
    one eps. **What a freed reach slips against the rupture's edge is set
    by that eps** (0.23 / 0.30 / 0.34 of the slip at 3 / 1.5 / 0.75 km),
    not by its elements: quote the width with the number.
