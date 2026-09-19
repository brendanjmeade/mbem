# CLAUDE.md

Guidance for Claude Code when working in `msd/`, the trunk of the mollified
BEM (`../BACKLOG.md` is the one status document; history is in `git log`).

## What this is

A self-contained mollified boundary element method for 3-D linear elasticity
in the full space. Singular Kelvin/Somigliana kernels are regularized
Cortez-style, `r -> sqrt(r^2 + eps^2)`, and every triangle's contribution is
integrated analytically, so the mollification width `eps` is a free
per-element parameter, not a quadrature artefact. No half space, no
viscoelasticity, no LaTeX. `README.md` has the physics and references.

## Running

```bash
pip install numpy scipy matplotlib numba triangle    # cutde for the classical-TDE gates/demos
python verify/run_all.py          # every gate, sequentially; exit 1 on any FAIL
python verify/verify_solved_bvp.py   # one gate; each prints PASS:/FAIL: and exits 1 on FAIL
python examples/demo_fault_only.py   # demos write fig_*.png/.pdf into this directory
```

Run everything from `msd/` (scripts put the repo root on `sys.path`
themselves). Numba compiles on first call; demos run at paper resolution.

## Map

**Frozen oracles** (flat modules; never "improve" them — new code is gated by
entrywise parity against them):
`mollified_bem.py` (`ElasticMaterial`, `TriMesh` — still the live types — and
the hand-written assembler), `mollified_kernel/` (point kernels; scalar and
vectorized analytic triangle integration), `anelastic.py` (the infinite-plane
eigenstress approximation; right deep inside an element, up to 2x off at its
edges), `local_box_mesh*.py`, `inclusion_mesh.py` (meshes; fault traces are
exact mesh edges), `tde_reference.py` (`cutde` classical-TDE stress, full or
half space; its slip sign is minus msd's).

**`mbem/`** — the live solver. Flow: `RegionModel -> generate_system ->
Backend.assemble -> solve -> evaluate_*`.

| module | role |
|---|---|
| `model/core.py` | `RegionModel`, `Region`, `Patch`, `BCType`; orientation `sigma(R,p)` inferred from solid angles and validated (closure identity, interface antisymmetry, fault containment); `FAULT_ORIENTATION` |
| `model/layout.py` | deterministic unknown slots (`u`/`t` per patch) |
| `model/equations.py` | `generate_system`: the one sign rule `A[row,u_p] += sigma H + diag`, `A[row,t_p] -= sigma G`, prescribed values and fault slip to the RHS; `COLLOCATION_JUMP`; the calibrated diagonal |
| `backends/dense.py`, `backends/hmat.py` | `DenseBackend` (LU) and `HBackend` (block-compressed + preconditioned FGMRES); same `BlockSystem`, same `jump`/`deflate` API, `solve()` returns the slot dict, `asm.report` |
| `kernels/tri_kernels.py` | numba analytic triangle kernels: U/T basis stacks, matrix-free contraction drivers, stress and eigenstress drivers (`*_serial` variants for threads) |
| `kernels/basis.py` | material-basis recombination, `resolve_eps`/`resolve_patch_eps`, mesh arrays |
| `evaluate.py` | interior displacement/stress from a solution; `_double_layer_stress` pairs each `Sdd` term with its eigenstress; `DisplacementEvaluator` for repeated grids |
| `geometry.py` | exact point-to-triangle distance (near-boundary warning, fault containment) |
| `la/` | `cluster` (trees, admissibility), `aca`, `hop.PairCompressed`, `hodlr`, `solver.fgmres`, `preconditioner.BlockGaussSeidel` (dense LU / HODLR / block-Jacobi ladder) |
| `selfcheck.py` | runtime guard: refuses to run if the fault sign convention is wrong (three cached stages, two Poisson ratios) |
| `defaults.py` | every tolerance and threshold |
| `estimate.py`, `topography.py`, `wrappers.py` | memory prediction; vertical surface warp; the `build_vertical_fault_zone_model` example |

`verify/` holds the gates (`_sphere.py` is the shared exact-Kelvin harness);
`examples/` the demos and `bench_scaling.py`, the performance harness to run
before and after touching assembly, compression or evaluation.

## Rules

1. **Oracles are frozen.** A kernel, mesh or assembly change adds a `verify/`
   script gated against the oracle or an analytic solution, `PASS:`/`FAIL:`,
   exit 1 on FAIL.
2. **The fault sign is stated once** (`FAULT_ORIENTATION`, `model/core.py`)
   and read through `RegionModel.orientation`; every site uses the same
   `-sigma`. `Patch.value` on a fault is `u(-n) - u(+n) = -b`, minus the
   conventional Burgers vector; `ddbem`/`clq` use `+b`. Never write a `+-1`
   for it anywhere; `selfcheck` pins the direction against hardcoded physics.
3. **Every mollified double layer carries an eigenstress** `C:eps*` of its
   smeared jump — fault slip and boundary `u_p` alike. It is removed in the
   stress readout (`evaluate_stress`, default), never in the solve, with the
   exact finite-triangle form (`eigenstress_contract`), per element with the
   element's own eps. Stress presented as elastic must have it subtracted.
4. **eps is per source element.** Scalar, `(N_src,)` array, per-patch dict or
   `"auto"` (= 0.1 h, basis in `defaults.py`). Displacement floors at
   eps/h ~ 0.125 and conditioning improves as eps/h drops; on-fault stress
   wants eps <= ~0.07 h on the fault and <= 0.125 h on a top patch near a
   trace, and the first element row below a free surface is trustworthy only
   for depth >~ max(h_top, 5 eps) (`../BACKLOG.md`). What governs the error
   is a collocation point's clearance from element edges in units of eps,
   not h: budget eps/h before expecting refinement or higher order to pay.
5. **`jump="calibrated"`** is the default on both backends (constant fields
   annihilated exactly). `"half"` with eps/h > 0.5 is non-convergent and
   warns. Calibrated on an all-Neumann model needs `deflate=True`; the
   backends refuse otherwise.
6. **The slip -> displacement pairing** is `U_ij = -[mu n_m dG_ij/dx_m + lam
   n_j dG_im/dx_m + mu n_m dG_im/dx_j]` (slip and normal on C's first index
   pair). A lam/mu swap is invisible at nu = 1/4, so kernel gates run at
   nu != 1/4 (`verify_dd_pairing.py`).
7. **Material coefficients come from `(mu, lam)`**, never through a
   `1/(1-2nu)` intermediate (`kernels/basis.py`).
8. **Numba `parallel=True` kernels are never called from Python threads**
   (macOS workqueue crash); the `*_serial` nogil variants exist for that.
9. **Numbers live in one place:** tolerances and thresholds in `defaults.py`,
   the collocation free term in `equations.py`, kernel identifiers in
   `kernels/__init__.py`. A convention written twice is a bug.
10. **Lean.** Docstrings state the rule and the reason; history and
    measurements go in commit messages; no probe scripts in the tree; extend
    a gate before adding one.
11. Units in the examples: km, GPa; slip 0.001 km = 1 m; `(N, 3)` arrays.
