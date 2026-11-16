# CLAUDE.md

Guidance for Claude Code in this repo: a research package for the **mollified
boundary element method (mbem)** and mollified elastic dislocation theory, built
on the Cortez blob regularization of the Kelvin/Somigliana
kernels. 3-D linear elasticity, full space, no half-space kernel, no LaTeX;
linear viscoelasticity by the Laplace route (rule 16), no time-stepper yet;
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
| `src/mbem/cases/` | The reference models the gates and the studies share: `fault_box`, `inclusion`, `topo_inclusion`, `penny` (the penny-shaped crack, Eshelby and Sneddon: the traction-BC fault's anchor), `layered_box` (an elastic plate over a substrate with a fault from the surface -- vertical strike-slip in the plate unless `dip` and `rake` say otherwise, and THROUGH the interface if `fault_depth` exceeds the plate, its lower reach a second fault, `fault_lower`, in the substrate -- and what it is compared with: the antiplane closed forms, and for dip slip `dip_slip_profile`, a plane-strain reference under gravity that shares nothing with the boundary elements; plus `two_mesh_limit`, what a plate that bends needs; `sides="window"` makes every wall of the box carry the far field of the prescribed slip, `lower="creep"` makes the megathrust's lower reach a solved fault that creeps, `slab_depth` lets the plate subduct as an elastic slab the interface wraps). `layered_space` is the same layer system in THREE dimensions, for a finite rectangular fault at the surface and at depth: the reference a finite rupture is compared with, and what a window's walls are made of. `fault_box` takes `walls="window"` too. Library code, because the gates and the studies both build from them. |
| `src/mbem/rheology.py`, `src/mbem/laplace.py` | **Linear viscoelasticity, the Laplace route.** `rheology`: `Maxwell`, `Burgers` as operational shear moduli, stated in relaxation TIMES and modulus ratios (never viscosities), bulk modulus elastic; the states of a Laplace run; the one-eps rule. `laplace`: the collocation points for a window of times, and `Prony`, the fixed-pole fit that turns the stack of per-s elastic solutions into value, rate, periodic-cycle sums and the velocity of a fault locked between repeating earthquakes, each as one linear map. No kernel work: an operator at any s is the elastic one at `(mu(s), lam(s))`. Also `rheology.ViscousFault`: a solved fault that resists its slip rate is, at each s, a fault with a stiffness (`Laplace.faults`, `State.drag`), so afterslip needs no time-stepper either. |
| `src/mollified_kernel/`, `src/moss_kernel/` | **Two frozen, INDEPENDENT copies** of the analytic mollified kernels. The difference is the point: the gates compare them entrywise, so neither may be "fixed" to agree with the other. `moss_kernel` alone defines `analytical_eigenstress_kernel` and `eigenstress_batch`. |
| `src/clq/` | **Frozen oracle.** Closed-form mollified kernels on one triangle for P0/P1/P2 nodal density, numpy. Separate derivation, so it is the parity reference that is genuinely independent. No development. |
| `src/mollified_bem.py` and friends | Frozen legacy oracles, kept as top-level modules: `ElasticMaterial` and `TriMesh` are defined in `mollified_bem.py` and re-exported from `mbem`, plus `anelastic`, `tde_reference`, `local_box_mesh*`, `inclusion_mesh`. |
| `tests/` | `run_all.py` (the authoritative runner) and `test_gates.py` (pytest over the same set). 47 gates in `gates/{mbem,clq,moss_kernel}`. |
| `configs/` | Nine studies. A study is a Python module declaring `RUN` or `run_spec(**kwargs) -> Run`. It NAMES a builder rather than describing patches, so it cannot restate the fault sign or the eps rule; `verify_config` proves the config path and the gates build the same model, fault Burgers vector included. |
| `runs/` | One self-describing folder per run (gitignored): `resolved.json` (spec, effective kwargs, resolved eps per patch, every uppercase `defaults` constant, environment), `report.json`, `fields_<state>.npz`, `STATUS`, `MANIFEST`. |
| `src/mbem/volume.py`, `src/mbem/vti.py` | **Volumetric readout.** `mbem sample <run>` re-evaluates a stored solution on a 3-D grid and writes `.vti` (VTK ImageData, hand-rolled: meshio cannot write it and `vtk` is a 100 MB wheel). Post-processing only -- it rebuilds the model from the run's spec and skips the assembly and the solve. Each point is classified by `RegionModel.point_in_region` (the Gauss closure identity, so topography is handled by the warped mesh itself) and carries `clearance_h` / `clearance_eps`: the mollification band is FLAGGED, never blanked, because on this model it overstates the topography effect 33x. |
| `src/mbem/figures/` | The figure makers, named in a lazy registry so importing `mbem` never imports matplotlib. Three kinds, and the CLI dispatches on which: **model-free** (the kernels alone, `mbem figure KEY`), **run** (one solve), **study** (several runs, because the quantity is a difference between operators). `save_figure` is the one statement of the png+pdf convention the demos each carried a copy of. |
| `studies/` | `bench_scaling.py`, the performance harness to run before and after touching assembly, compression or evaluation — it keeps its own provenance helpers deliberately, so its committed `bench-json:` baselines stay comparable. Plus `demo_triangle_quickstart.py`, which prints numbers and draws nothing. `studies/clq/` holds the clq oracle's own demos. |
| `website/` | Astro static site (`npm run dev` / `npm run build` inside `website/`). `.github/workflows/deploy-website.yml` deploys it to GitHub Pages on every push to `main` that touches `website/**`, so pushing there publishes it. Nothing is computed in the browser: `public/data/` is a quantised readout of one stored run, made by `mbem sample <study-dir> --difference` and then `website/scripts/make_payload.py`. The encoder is `mbem.webexport`, kept in the package so that `verify_volume` [g] gates it. Fields are log10 uint8 and never float textures; `website/README.md` explains why. |
| `docs/` | `figures/` (the curated, tracked PNG gallery), `clq.md`, `clq-derivation.md`, `clq-rules.md`. |
| `docs/viscoelastic.md` | The derivation viscoelasticity rests on. With elastic bulk and viscoelastic shear, every operator of a region is geometry times `{1, g} x {1, mu, 1/mu}`, `g = 1/(1-nu)`, which holds only for the eigenstress-removed stress (rule 4). The Laplace route built on it and its closed-form anchors; the time-stepper with boundary memory, which is a design and NOT built; why a region that relaxes needs ONE scalar eps; gravity as a condition on the boundary (section 7), the sphere it is gated against, and under it the thrust in the plate, the two-mesh limit that cancels what P0 costs a plate that bends, the earthquake cycle of a locked fault, and a megathrust through the interface; the walls of a box as a window, and the reference in three dimensions (section 8); a fault that creeps (section 9); a plate that subducts (section 10); and what is open (section 11: what a window's walls do not know, the time-stepper, slip on an interface, gravity's coupling with compression). |
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
    --set edge_near=15 --set edge_interface=22.5 --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window --set half_x=300 --set depth_box=200 \
    --set fault_half_len=100 --set t_max=100 --set backend=dense --set lower=creep --sweep creep_tau=1,1e6 --sweep refine=1,1.5   # ... its lower reach creeping: afterslip
python -m mbem run configs/plate_substrate.py --set dip=20 --set rake=90 --set depth=30 --set rho_plate=3300 --set rho_substrate=3300 \
    --set edge_near=10 --set near_field_radius=150 --set sides=window --set half_x=500 --set depth_box=300 --set fault_half_len=100 --set t_max=100 --sweep slab_depth=0,150   # ... the plate subducting: a slab
python -m mbem run configs/fault_box.py --set walls=window     # the elastic box on a half-space
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
    unit its tau has. **A region that relaxes carries ONE scalar eps on its
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
    **A slab** (`slab_depth`) is the plate's own region continued down dip,
    welded below the megathrust. Give plate and mantle one density: a slab
    of a lighter plate hanging in a heavier mantle is unstable once anything
    relaxes, and the run is refused.
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
