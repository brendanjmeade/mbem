# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A self-contained **mollified boundary element method (BEM)** for 3-D linear
elasticity in the **full space** (no half-space/Mindlin, no viscoelasticity in
the kernels, no LaTeX). Singular Kelvin/Somigliana kernels are regularized
Cortez-style with `r -> r_eps = sqrt(r^2 + eps^2)`, and each triangle's
contribution is integrated **analytically**, so the mollification width `eps`
is decoupled from mesh size `h` (Ferranti & Cortez). See `README.md` for the
physics and reference papers.

## Running things

There is **no build, no installer, no pytest, no CI**. Everything is plain
scripts run with the system Python from the repo root.

```bash
pip install numpy scipy matplotlib numba triangle   # sympy only for one verify script

# Correctness checks — each prints "PASS"/"FAIL" (no test runner, no exit codes to assert on)
python verify/verify_analytical_vs_quadrature.py    # analytic == high-order quadrature
python verify/verify_arbitrary_triangle.py          # arbitrary-triangle / rigid / scaling
python verify/verify_batch_vs_scalar.py             # vectorized == scalar
python verify/verify_pde_residual.py                # regularized Kelvin satisfies the PDE (needs sympy)
python verify/verify_dd_pairing.py                  # lambda/mu pairing of the DD displacement kernel (nu sweep)

# Demos — run from repo root; each writes fig_*.png/.pdf into the repo root
python examples/demo_fault_only.py                  # fault-only BEM (displacement + elastic stress)
python examples/demo_hmatrix.py                     # H-matrix FGMRES vs dense LU reference
python examples/demo_anelastic_subtraction.py       # anelastic term -> finite on-fault stress
# full table of demos is in README.md
```

Scripts insert the repo root onto `sys.path` themselves (`examples/*` do
`sys.path.insert(0, ROOT)` then `import mollified_bem`, `from mbem...`), so they
must be launched from the repo root, not from inside `examples/`. BEM demos run
at paper resolution — tens of seconds to a few minutes; numba JIT adds a
first-call warmup.

## Two-layer architecture: frozen legacy oracles + the `mbem` rebuild

This is the single most important thing to understand before editing.

**Legacy flat modules are FROZEN validation oracles** — do not "improve" them;
new code is tested for entrywise parity against them:

- `mollified_bem.py` — defines `ElasticMaterial` (`mu`, `lam`; `.nu`, `.E`),
  `TriMesh`, the full-space mollified Kelvin U/T kernels, and the original
  hand-written `assemble_BEM_matrices`. `ElasticMaterial` and `TriMesh` are
  still the live types used everywhere (including `mbem`).
- `mollified_kernel/` — point-source kernels (`mollified_elastic_kernels.py`)
  and the analytic per-triangle integration, scalar (`analytical_kernels.py`)
  and vectorized (`analytical_batch.py`).
- `local_box_mesh.py`, `local_box_mesh_eq.py`, `inclusion_mesh.py` — mesh
  builders (layered boxes, fault-aligned tops, cylindrical inclusions). Fault
  traces are embedded as **exact mesh edges** so no triangle straddles the slip
  discontinuity.
- `anelastic.py` — `eigenstress_at_points`: a mollified slip source returns
  *total* stress; subtract the anelastic (eigenstrain) term to get the genuine
  **elastic** stress. Stress demos must apply this or on-fault stress diverges
  like `1/eps`. The corrected interior on-fault stress *converges to a constant*
  (`eps^2`) as `eps -> 0` — validated against the classical-TDE finite part
  (`tde_reference.py`, a `cutde` full/half-space reference).
- `tde_reference.py` — `classical_tde_stress(...)`: independent classical
  triangular-dislocation stress (`cutde`, full- or half-space) for the SAME
  triangulated fault, the validation oracle for on-fault stress. cutde uses a
  `[strike,dip,tensile]` slip in the TDE frame and the OPPOSITE slip-sign
  convention to the mbem fault, so reconcile with `compute_efcs_to_tdcs_rotations`
  and a global `g = +-1` (see the demos).

**Where the eigenstress subtraction belongs.** NOT in the BEM solve: `mbem` is
formulated in *displacement*, so a fault enters only as a smooth `-H@slip`
influence on the RHS and the `1/eps` eigenstress (a *stress* quantity) never
touches assembly/collocation. It belongs in the **stress readout**
(`evaluate_stress`, `subtract_anelastic=True`). Sign subtlety: the fault stress
term there is `-Sdd@slip` (mirroring the `-H@slip` displacement), so its divergent
part is `-C:eps_star` and removing it *adds* the eigenstress — verified by the
fact that on-fault elastic stress stays finite as `eps -> 0` (the
`verify/verify_evaluate_stress.py` finiteness gate, which a self-consistency check
alone would miss).

**`mbem/` is the rebuilt solver stack** — geometry-general, faster, and the
place to do new work.

### How `mbem` assembles a solve

The flow is `RegionModel -> generate_system -> Backend.assemble -> solve`:

1. **`mbem/model/core.py` — `RegionModel`.** A graph of `Region`s, each bounded
   by oriented `Patch`es with a `BCType` (`FREE_TRACTION`, `PRESCRIBED_DISPLACEMENT`,
   `INTERFACE`, `FAULT`). The hard problem the legacy solvers hand-coded
   case-by-case is the **boundary orientation sign** `sigma(R,p) = +1 iff patch
   p's stored normals point out of region R`. Here it is **inferred
   geometrically** from signed solid angles and validated by the Gauss closure
   identity (`sum_p sigma*Omega = 4*pi`) and interface antisymmetry. Interface
   patches shared by two regions **must be the same `Patch` object** (incidence
   is by `id()`); `orientation_overrides` is the escape hatch for non-star-shaped
   regions.
2. **`mbem/model/layout.py` — `UnknownLayout`.** Deterministic slot ordering
   (`u`/`t` per patch) reproducing the legacy unknown orderings exactly.
3. **`mbem/model/equations.py` — `generate_system`.** Emits one `BlockSystem`
   from THE sign rule (the docstring is the spec): `A[row,u_p] += sigma*H + 1/2 I`,
   `A[row,t_p] += -sigma*G`; prescribed values and fault slip move to the RHS.
   `H` is the T-kernel (slip/displacement -> displacement), `G` the U-kernel.
4. **Backends** (`mbem/backends/`): `DenseBackend` (LU, oracle-parity) and
   `HBackend` (block-compressed + preconditioned FGMRES). Both consume the same
   `BlockSystem`, and both support `jump="half" | "calibrated"` (the H-path
   calibration computes `C_q = -sum_p sigma*rowsum(H_qp)` through the
   compressed pairs via constant-field matvecs, so the applied operator
   annihilates constants exactly; gate: `verify/verify_hbackend.py`). For
   all-Neumann + calibrated models (exact rigid-translation null space) pass
   `deflate=True`: dense solves the translation-bordered system, HBackend
   projects the null space out of FGMRES (`verify_deflation_estimate.py`).
   `HBackend(storage="basis")` keeps geometry-only per-basis factors (B-fold
   memory, free material recombination — best for Laplace sweeps);
   `storage="combined"` keeps only material-combined payloads (1x memory, the
   mode for very large models; unseen materials trigger transient
   re-compression). Block compression is parallel across LARGE blocks
   (serial nogil kernels in threads — do NOT call the `parallel=True` kernels
   from concurrent Python threads, the macOS workqueue layer crashes) with
   per-block-seeded rng (bitwise deterministic). `mbem/estimate.py` predicts
   memory per mode (`estimate_memory`, `choose_dense_mode`) before assembling.
5. **`mbem/evaluate.py`** — interior-field representation formulas, same `sigma`
   and prescribed-value handling as the boundary equations. BOTH evaluators are
   **matrix-free numba contraction drivers** (never materialize the
   `(3N_obs, 3N_src)` influence: O(N_obs) memory at any source count):
   `evaluate_displacement` via `t_disp_contract`/`u_disp_contract`
   (`verify_disp_contract.py`: machine parity vs the dense matrices AND the
   legacy oracle; 250k obs x 1.4k src = 9 s / 0.4 GB where dense needed 49 GB)
   and `evaluate_stress` via `dd_stress_contract`/`kelvin_stress_contract`
   (order-7 / rank-4 moment recursion — `I7`, `T2[5]`, `T2[7]`, `T4[7]`;
   `verify_stress_assembler.py`, ~5000x over the scalar loop). For REPEATED
   evaluation on a fixed grid (sweeps, time series) `DisplacementEvaluator`
   compresses the obs-grid influence once (`PointCloud` adapter + ACA) and
   applies it per solution at matvec cost. Both evaluators warn when obs
   points sit within ~0.5*local-h of a **boundary patch** (the volume
   representation is MESH-limited there — the c=1/2 boundary-jump transition
   smears over h, NOT eps; shrinking eps makes tractions blow up instead;
   faults are exempt — the eigenstress subtraction handles their near field).
   The stress drivers are material-applied; a stress geometry-basis split is
   the remaining optimization.

### Cross-cutting design ideas

- **Material-basis decomposition (`mbem/kernels/basis.py`).** Each U/T influence
  matrix is `M(material) = sum_k c_k(mu,lam) * B_k` with geometry-only `B_k`
  (3 for U, 6 for T: three `N[P] = n_j tr P` blocks with `lam*C1` and three
  `R[P] = n_m P_ijm + n_k P_ikj` blocks with `mu*C1`; `eps^2` baked in). Assemble the basis ONCE, recombine per
  material — this is what makes `rebuild_for_materials` cheap (the viscoelastic
  Laplace-sweep primitive) and works for complex `mu_tilde(s)`. **Binding rule:**
  compute coefficients from `(mu, lam)` directly, never via a `1/(1-2nu)`
  intermediate (it blows up at the fluid limit `nu -> 1/2`).
- **`DenseBackend` modes** (`mode=`): `"legacy"` routes every block through
  `mollified_bem.assemble_BEM_matrices` for entrywise parity (scalar eps only);
  `"basis"` caches geometry bases for cheap material rebuilds (~9x memory;
  `mbem.estimate.choose_dense_mode` picks by RAM); `"direct"` is memory-light
  numba in-loop assembly (calibration shares the per-build block cache — do not
  reintroduce the double assembly; gate: `verify_dense_backend.py`
  bit-identity). `jump="calibrated"` repairs the Gauss identity for thin panels
  but makes rigid translations an exact null space on all-Neumann models — use
  `deflate=True` there (bordered solve; matches the half-jump physics to <1%,
  `verify_deflation_estimate.py`).
- **`eps` is a per-source-element `(N_src,)` array everywhere**; a scalar is
  promoted to a constant array (= legacy global-eps behavior). Per-patch eps via
  a `{patch_name: eps}` dict. `eps="auto"` (opt-in) resolves per element to
  `EPS_OVER_H * mean-edge-length`, keeping eps/h fixed under grading and
  h-refinement (order-2 convergence gate: `verify_eps_auto.py`). The
  eigenstress subtraction needs a near-uniform FAULT eps (a scalar is safest;
  `evaluate_stress` raises on a graded fault eps).
- **`mbem/la/`** — the linear-algebra layer for `HBackend`: `cluster` (cluster
  trees / admissibility; `MAX_ADMISSIBLE_BLOCK=2048` caps the block side — the
  dense-fallback bomb of a failed 4096 block is ~7 GB and 4x slower, measured),
  `aca` (per-basis low-rank block compression, true-residual stopping,
  verification + dense fallback), `hop.PairCompressed` (compressed
  material-basis pair operator; LRU-bounded per-material views —
  `HOP_VIEW_CACHE_MAX`), `hodlr` (direct HODLR solver),
  `solver.fgmres` (right-preconditioned, true-residual verified),
  `preconditioner.BlockGaussSeidel` (3-rung diagonal ladder: dense LU <=
  `MAX_DENSE_PRECOND_DOF`, HODLR <= `PRECOND_HODLR_MAX_DOF`, then cluster
  block-Jacobi — the rung that scales to 1e5-1e6-element patches), `scaling`
  (Ruiz/physics equilibration — NOTE: measured to be a NO-OP for FGMRES here:
  with right preconditioning and BGS's exact diagonal solves, column scaling
  with a consistently transformed preconditioner yields the identical
  iteration; it is not wired into the solve). `mbem/la/cluster.py` is a clean
  reimplementation of the frozen legacy `hmatrix.py`.
- **`mbem/defaults.py`** is the single source of truth for every tolerance and
  threshold. Put new numeric constants here, not inline.
- **`mbem/wrappers.py`** — `solve_*_v2` functions match legacy signatures but
  route through the region-graph core; useful as worked examples of building a
  `RegionModel` (`build_three_region_box_model`, `build_vertical_fault_zone_model`).

The `mbem` docstrings reference an "approved plan" / design doc that is **not in
this repo** — treat those mentions as historical; the docstrings themselves are
the authoritative spec.

## Kernel index pairing (fixed 2026-09-04)

The slip -> displacement (T) contraction is the traction operator applied to
the Kelvin solution -- slip and normal share C's FIRST index pair:
`U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]`.
Until 2026-09-04 `analytical_dd_displacement`, `dd_displacement_batch`,
`integrate_dd_displacement_numerical`, the numba T basis (`_contract_LM`, now
`_contract_NR`) and `examples/demo_point_kernel.py` had lam and mu swapped on
the first two terms.  This is invisible at nu = 1/4 (lam == mu), which every
demo and every parity gate used, and ~20-60 % off at nu = 0.3.  The stress
kernels (`analytical_stress_kernel`, `dd_stress_contract`) always had the
correct pairing.  The six-block T basis is unchanged in count and
coefficients; only the block *meaning* changed.  Gate:
`verify/verify_dd_pairing.py` (closed-cube Gauss closure `sum U = -I` at
nu in {0.1, 0.25, 0.3, 0.45}, quadrature of `kelvin_T_mollified`, FD-Hooke
consistency, moss parity); `verify_batch_vs_scalar.py` and
`verify_arbitrary_triangle.py` now also run at nu = 0.3; the stress-side
gates `verify_stress_assembler.py` (numba vs scalar) also run at nu = 0.3,
while `verify_evaluate_stress.py` (cutde reference) still runs at nu = 1/4
only.  The same pre-fix form still exists OUTSIDE msd: `moss2/` (scalar,
batch, numerical reference), `moss/mollified_kernel/analytical_batch.py` and
`moss/mollified_kernel/analytical_kernels.py::integrate_dd_displacement_numerical`
(moss fixed only the scalar `analytical_dd_displacement`), the `moss/mh_deploy/`
copies, `moss/manuscript/scripts/_quad_assembly.py`, and eq. `U-integrated` of
`moss/docs/mollified_kernels.tex`.

## Conventions

- Numerics are **numba**-accelerated in `mbem/kernels/tri_kernels.py`; expect
  JIT warmup on first call. The `parallel=True` kernels must NEVER be called
  from concurrent Python threads (macOS workqueue crash) — the `*_serial`
  nogil variants exist for that (parallel ACA uses them).
- Units in examples mix displacement (km) and traction (GPa); the
  BlockGaussSeidel preconditioner's exact diagonal solves absorb the unit
  mixing (measured — see the scaling note above).
- When adding a kernel, mesh, or assembly path, add a `verify/` script that
  checks it against the legacy oracle or analytics and prints `PASS`/`FAIL`.
- `examples/bench_scaling.py` is the performance regression harness (model
  ladder + `--panel N` large-compression primitive; `--json` for tracking).
  Run it before/after touching assembly, compression, or evaluation.
