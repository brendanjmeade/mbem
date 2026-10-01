# msd — mollified full-space boundary element method

A clean, self-contained implementation of a **mollified boundary element method
(BEM)** for 3-D linear elasticity, in the full space.  Singular Kelvin/Somigliana
kernels are regularized Cortez-style,

```
r  ->  r_eps = sqrt(r^2 + eps^2),
```

which removes the singular surface integrals and lets the mollification width
`eps` and the mesh size `h` be chosen independently.  Each triangle's kernel
contribution is integrated **analytically**, so (following Ferranti & Cortez)
`eps` is decoupled from `h` — there is no quadrature error tying them together.

This package contains only the elastic full-space machinery: the point-source
kernel, the triangular displacement-discontinuity solution, the `mbem` BEM
solver (region model + calibrated dense backend), and worked examples.  There is
**no half-space / Mindlin code, no viscoelasticity, and no LaTeX** here.

Boundary patches and faults carry a piecewise-constant (P0) density by default
or a Lagrange P1/P2 nodal density (`Patch(..., order=1|2)`): one code path,
collocation at the shrunk element nodes, the free term a shape-function matrix,
kernels integrated in closed form for every order (`mbem/kernels/tri_nodal.py`,
gated against the frozen `../clq` oracle).  Higher order pays on traction
(Neumann) rows and is the cure for the first element row of on-fault stress
below a free surface (beyond ~3-5 eps_top; `verify/verify_solved_bvp.py` A4);
the compressed backend is P0-only.

## The anelastic (eigenstrain) term

A fault's `Patch.value` is the Burgers vector `b = u(+n face) - u(-n face)` for
stored normal `n`, the same sign clq and cutde use (`mbem/selfcheck.py` pins it
at runtime).

A fault slip is an **anelastic** (inelastic / eigen-) strain.  Stress read off a
mollified slip source is therefore the *total* stress `C:eps_total =
C:eps_elastic + C:eps_star` inside the ~`eps` fault zone; on the fault it is
dominated by the eigenstress, peaking at `(3/4) mu s / eps` and diverging as
`eps -> 0`.  To recover the genuine **elastic** stress you must subtract the
anelastic term:

```
sigma_elastic = sigma_total - C:eps_star,
```

The live implementation is the **exact finite-triangle** form in
`mbem/kernels/tri_kernels.py::eigenstress_contract`, applied by
`mbem.evaluate_stress(subtract_anelastic=True)` (the default); `anelastic.py`
(`eigenstress_at_points`) is the frozen infinite-plane / nearest-triangle
*approximation*, right deep inside a large element and up to 2x too large at
element edges; the demos use the exact form through
`mbem.evaluate._stress_from_source(..., "eigen", ...)`.
`examples/demo_anelastic_subtraction.py` shows that the
corrected on-fault stress stays finite (bounded, `eps`-independent) while the
raw value blows up like `1/eps`.

The same eigenstress lives in **every** mollified double layer, not only faults:
a boundary patch's `u_p` is a jump between the field inside the region and zero
outside, and its smeared eigenstress is non-physical inside the body.  Since
2026-09-19 `evaluate_stress` subtracts it for boundary patches too — that term,
not a mesh limit, was the interior stress error near boundaries that did not
improve under refinement (`verify/verify_boundary_eigenstress.py`).

The subtracted on-fault elastic stress not only stays finite — at the fault
interior it **converges to a constant** as `eps -> 0` (observed order `eps^2`),
and to the *physically correct* value: `examples/demo_onfault_convergence.py`
matches it to the finite part of the classical triangular-dislocation stress
(an independent `cutde` full-space reference) to ~1e-7.  (The interior converges
to a constant; the genuine elastic field still concentrates at the fault tips,
so the interior, not the peak, is the convergence metric.)

Because the BEM is solved in **displacement** (a fault enters only as a smooth
slip -> displacement source on the RHS), the eigenstress never enters the solve
— subtracting it there would be wrong.  It belongs in the **stress readout**:
`mbem.evaluate_stress` sums the boundary and fault stress kernels and removes the
fault eigenstress, producing the genuine on-fault elastic (Coulomb) stress the
BEM previously could not.  `examples/demo_bem_onfault_stress.py` reads it off a
free-surface box solve and recovers the half-space dislocation stress.

## Layout

```
mollified_bem.py          TriMesh, ElasticMaterial, full-space Kelvin kernels
mollified_kernel/         point-source kernel + analytic triangle integration
  mollified_elastic_kernels.py   regularized Kelvin point kernels
  analytical_kernels.py          analytic per-triangle integration (scalar)
  analytical_batch.py            analytic per-triangle integration (vectorized)
mbem/                     region-model BEM solver (dense backend, numba kernels,
                          topography); the calibrated successor to the legacy code
local_box_mesh_eq.py      equilateral Delaunay box / fault mesh builders
inclusion_mesh.py         host + cylindrical-inclusion mesh builder
anelastic.py              anelastic (eigenstrain) eigenstress -> elastic stress
examples/                 runnable demos (figures land in the repo root)
verify/                   kernel correctness checks (print PASS/FAIL)
```

## Install

```
pip install numpy scipy matplotlib numba triangle threadpoolctl
# plus an OpenMP runtime numba can load (conda-forge llvm-openmp, or brew libomp)
# cutde is only needed for the on-fault-stress demos/verify (classical TDE reference)
```

numba must resolve its OpenMP threading layer (`numba.threading_layer() ==
"omp"`) and threadpoolctl must be importable: the compressed backend runs
nogil kernels on a Python thread pool while parallel kernels run beside them,
which the OpenMP layer tolerates and the default workqueue layer does not,
and threadpoolctl is what pins BLAS to one thread inside that pool.

## Verify the kernels

Every gate prints one final `PASS: <title>` / `FAIL: <title>` line and exits 1
on FAIL; `python verify/run_all.py` runs them all sequentially and tabulates
verdict, wall time and exit code.

```
python verify/verify_analytical_vs_quadrature.py   # analytic == high-order quadrature
python verify/verify_arbitrary_triangle.py         # arbitrary-triangle / rigid / scaling
python verify/verify_batch_vs_scalar.py            # vectorized == scalar
python verify/verify_dd_pairing.py                 # lambda/mu pairing of the DD displacement kernel (closure, nu sweep); point kernel satisfies the regularized Navier equation
python verify/verify_evaluate_stress.py            # mbem stress == classical TDE (cutde); on-fault elastic stays finite
python verify/verify_stress_assembler.py           # numba batched stress assemblers == scalar oracle (machine precision)
python verify/verify_eigenstress_exact.py           # EXACT finite-triangle eigenstress == moss/clq oracles; sign; rim disagreement with anelastic.py
python verify/verify_boundary_eigenstress.py       # boundary double layers subtracted too: near-boundary interior stress converges (icosphere vs exact Kelvin)
python verify/verify_solved_bvp.py                 # assembly -> BCs -> solve -> displacement vs cutde half-space and manufactured solutions (end to end)
python verify/verify_disp_contract.py              # matrix-free displacement evaluation == dense matrices == legacy oracle
python verify/verify_dense_backend.py              # dense assembly invariants (calibration cache bit-identity, rebuilds)
python verify/verify_hbackend.py                   # H backend == dense (ACA, calibrated jump, combined storage, rung ladder)
python verify/verify_eps_auto.py                   # eps="auto" (0.1 h): kernel order-2 convergence, solved-BVP accuracy vs Kelvin, half-jump guard
python verify/verify_deflation_estimate.py         # all-Neumann rigid-body deflation; memory estimator
python verify/verify_nodal_kernels.py              # P0/P1/P2 numba kernels == clq oracle (1e-12); edge primitives; order 0 == the P0 path
python verify/verify_nodal_solve.py                # P1/P2 solves: P0 bitwise, patch test, rigid covariance, h-convergence, P1 faults, refusals
```

## Scaling

The solver stack scales far beyond the dense backend's ~30k-element cap:
`HBackend(jump="calibrated", storage="combined")` assembles the operator
block-compressed (matrix-free ACA, parallel across blocks, 1x material-combined
storage), FGMRES is preconditioned by a block-Gauss-Seidel ladder whose rung
past the dense-LU cap (cluster block-Jacobi) has O(N) build cost at any patch
size (a HODLR rung is reachable explicitly, for memory over speed), and field
evaluation is matrix-free (O(N_obs) memory; a 250k-point map over a model that
would need a 49 GB dense operator runs in ~9 s and 0.4 GB).
`mbem.estimate.estimate_memory` predicts the footprint of each backend/mode
before assembling; `examples/bench_scaling.py` is the regression harness:
`--model fault_box|topo_inclusion --scale s` mesh ladders with per-rung JSON
records (phase times, iterations, ranks, bytes, RSS, operator error against
the dense operator or exact matrix-free rows), `--gate REV` against the
`bench-json:` line of a commit message, and `--panel N` for the large-block
compression primitive alone.

## Examples

Run from the package root; each writes `fig_*.png/.pdf` to the repo root.

| Feature | Script |
|---|---|
| Mollified **point-source** kernel (singular vs mollified) | `python examples/demo_point_kernel.py` |
| **Triangle full-space solution** — minimal easy-calling | `python examples/demo_triangle_quickstart.py` |
| Triangle stress field / integration methods | `python examples/demo_triangle_field.py` |
| Mollified field over an `eps` ladder | `python examples/demo_triangle_eps_sweep.py` |
| **eps–h convergence** (analytic decoupling) | `python examples/demo_eps_h_convergence.py` |
| **BEM eps-convergence** at fixed mesh | `python examples/demo_bem_eps_convergence.py` |
| **Fault-only** BEM (displacement + elastic stress) | `python examples/demo_fault_only.py` |
| **On-fault stress convergence** (elastic, full-space; ->const, vs classical TDE) | `python examples/demo_onfault_convergence.py` |
| **On-fault stress from a BEM solve** (elastic, vs half-space TDE) | `python examples/demo_bem_onfault_stress.py` |
| **Fault + inclusion + topography** BEM | `python examples/make_topo_inclusion.py` then `python examples/render_topo_inclusion_contour.py --smooth` |
| **H-matrix** (block-compressed FGMRES vs dense) | `python examples/demo_hmatrix.py` |
| **Anelastic term subtraction** (finiteness) | `python examples/demo_anelastic_subtraction.py` |

The BEM demos run at paper resolution (tens of seconds to a few minutes; the
four-state topography+inclusion solve is the longest).  The free surfaces are
fault-ALIGNED (the surface-breaking fault trace is embedded as exact mesh edges
via `make_top_patch_with_fault`), so no triangle straddles the slip
discontinuity.  The `--mu-inc 3.0` inclusion is the soft `mu/10` body in the
showcase; `render_topo_inclusion_contour.py` grids the surface displacements
(500x500) and draws filled contours for the four-state decomposition (full /
inclusion-only / topography-only / topography+inclusion).  A simpler
flat-shaded renderer (`render_topo_inclusion.py`) is also provided.

The H-matrix demo solves the `mu/10` inclusion with the block-compressed backend
(`mbem.backends.HBackend(eta=0.8)`, ACA + preconditioned FGMRES) and reproduces
the dense LU reference to ~1e-9.  (Use `eta=0.8`: the looser default
admissibility can accept inaccurate low-rank factors for a near-field block.)

## References

The mollification follows the method of regularized Stokeslets and its surface
(triangulated-BIE) extension:

- Cortez, R. (2001). *The Method of Regularized Stokeslets.* SIAM Journal on
  Scientific Computing, 23(4), 1204–1225. doi:10.1137/S106482750038146X
- Cortez, R., Fauci, L., & Medovikov, A. (2005). *The Method of Regularized
  Stokeslets in Three Dimensions: Analysis, Validation, and Application to
  Helical Swimming.* Physics of Fluids, 17(3), 031504. doi:10.1063/1.1830486
- Ferranti, D., & Cortez, R. (2024). *Regularized Stokeslet Surfaces.*
  arXiv:2310.14470 [math.NA]. — establishes the analytic per-triangle
  integration that decouples `eps` from `h`, the property exploited here.
