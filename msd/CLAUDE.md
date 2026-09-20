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
half space; its slip is the same Burgers vector as a fault `Patch.value`).

**`mbem/`** — the live solver. Flow: `RegionModel -> generate_system ->
Backend.assemble -> solve -> evaluate_*`.

| module | role |
|---|---|
| `model/core.py` | `RegionModel`, `Region`, `Patch` (`order` 0/1/2: nodal layout, `nodes`, `collocation_points`, `collocation_shape`, nodal `value_array`), `BCType`; orientation `sigma(R,p)` inferred from solid angles and validated (closure identity, interface antisymmetry, fault containment); `FAULT_ORIENTATION` |
| `model/layout.py` | deterministic unknown slots (`u`/`t` per patch, `3 * n_nodes` each) |
| `model/equations.py` | `generate_system`: the one sign rule `A[row,u_p] += sigma H + diag`, `A[row,t_p] -= sigma G`, prescribed values and fault slip to the RHS; `COLLOCATION_JUMP`; the calibrated diagonal per collocation point, spread over the element's nodes by `N_k(x_c)` |
| `backends/dense.py`, `backends/hmat.py` | `DenseBackend` (LU; any mix of P0/P1/P2 patches, rows at collocation points) and `HBackend` (block-compressed + preconditioned FGMRES; P0 only, refuses higher order); same `BlockSystem`, same `jump`/`deflate` API, `solve()` returns the slot dict, `asm.report`; the half-jump and collocation-near-fault guards |
| `kernels/tri_kernels.py` | numba analytic triangle kernels: U/T basis stacks, matrix-free contraction drivers, stress and eigenstress drivers (`*_serial` variants for threads) |
| `kernels/tri_nodal.py` | the same drivers plus `order` for Lagrange P0/P1/P2 nodal density (clq's moment machinery in numba; columns `3*(K*s + k) + j`, densities `(K*N_s, 3)`); order 0 routes to `tri_kernels` |
| `kernels/basis.py` | material-basis recombination, `resolve_eps`/`resolve_patch_eps`, mesh arrays, Lagrange node lattice and shape functions in the kernels' node order |
| `evaluate.py` | interior displacement/stress from a solution at each patch's order; `_double_layer_stress` pairs each `Sdd` term with its eigenstress; `DisplacementEvaluator` for repeated grids (P0 only) |
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
   `-sigma`. `Patch.value` on a fault is the Burgers vector
   `b = u(+n) - u(-n)` with `n` the stored normal, the same sign as
   `ddbem`/`clq` and cutde. Never write a `+-1` for it anywhere; `selfcheck`
   pins the direction against hardcoded physics.
3. **Every mollified double layer carries an eigenstress** `C:eps*` of its
   smeared jump — fault slip and boundary `u_p` alike. It is removed in the
   stress readout (`evaluate_stress`, default), never in the solve, with the
   exact finite-triangle form (`eigenstress_contract`), per element with the
   element's own eps. Stress presented as elastic must have it subtracted.
4. **eps is per source element.** Scalar, `(N_src,)` array, per-patch dict or
   `"auto"` (= 0.1 h on a boundary patch; on a FAULT one value, 0.07 min h,
   because per-element widths smear a uniform slip unequally and put tens
   of percent on the on-fault stress; basis in `defaults.py`). Displacement floors at
   eps/h ~ 0.125 and conditioning improves as eps/h drops; on-fault stress
   wants eps <= ~0.07 h on the fault and <= 0.125 h on a top patch near a
   trace, and the first element row below a free surface is trustworthy only
   for depth >~ max(h_top, 5 eps) under a P0 top; a P1 top drops the h_top
   term, leaving ~3-5 eps_top (`verify_solved_bvp` A4). What governs the error
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
12. **Higher-order patches:** `Patch.order` in {0, 1, 2}, discontinuous
    nodal layout in clq's node order, collocation at the shrunk nodes
    (`COLLOCATION_SHRINK_BY_ORDER`), the free term is the shape-function
    matrix `N_k(x_c)` times the per-point diagonal, and the compressed
    backend is P0-only. Gate: `verify_nodal_solve.py`.
