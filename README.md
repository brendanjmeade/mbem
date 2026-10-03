# mbem — mollified full-space boundary element method

A clean, self-contained implementation of a **mollified boundary element method
(BEM)** for 3-D linear elasticity, in the full space.  Following Cortez, the
regularization is applied to the SOURCE, not to the kernel: the point force is
replaced by a smooth blob `phi_eps` of unit integral and the elastostatic
equations are then solved exactly, giving

```
G_ij = C1 [ (3-4nu) delta_ij / R_eps  +  d_i d_j / R_eps^3
                                      +  2(1-nu) eps^2 delta_ij / R_eps^3 ]

  R_eps = sqrt(r^2 + eps^2),  C1 = 1/(16 pi mu (1-nu)),
  L_ik G_kj = -delta_ij phi_eps,  phi_eps = 15 eps^4 / (8 pi R_eps^7).
```

The first two terms are the classical Kelvin solution carrying `R_eps` in place
of `r`.  **The third has no singular counterpart** and is what the blob
convolution contributes: without it the regularized Cauchy-Navier residual is
~1e-2, with it ~1e-16 at every Poisson ratio
(`tests/gates/moss_kernel/verify_pde_residual.py`, symbolically).  Simply
substituting `r -> R_eps` gives a smooth function that solves nothing, and it is
the PDE, not the smoothness, that licenses integrating over the element the
observation point lies on.

This removes the singular surface integrals and lets the mollification width
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
gated against the frozen `src/clq` oracle).  Higher order pays on traction
(Neumann) rows and is the cure for the first element row of on-fault stress
below a free surface (beyond ~3-5 eps_top; `tests/gates/mbem/verify_solved_bvp.py` A4);
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
`python -m mbem figure anelastic_subtraction` shows that the
corrected on-fault stress stays finite (bounded, `eps`-independent) while the
raw value blows up like `1/eps`.

The same eigenstress lives in **every** mollified double layer, not only faults:
a boundary patch's `u_p` is a jump between the field inside the region and zero
outside, and its smeared eigenstress is non-physical inside the body.  Since
2026-09-19 `evaluate_stress` subtracts it for boundary patches too — that term,
not a mesh limit, was the interior stress error near boundaries that did not
improve under refinement (`tests/gates/mbem/verify_boundary_eigenstress.py`).

The subtracted on-fault elastic stress not only stays finite — at the fault
interior it **converges to a constant** as `eps -> 0` (observed order `eps^2`),
and to the *physically correct* value: `python -m mbem figure onfault_convergence`
matches it to the finite part of the classical triangular-dislocation stress
(an independent `cutde` full-space reference) to ~1e-7.  (The interior converges
to a constant; the genuine elastic field still concentrates at the fault tips,
so the interior, not the peak, is the convergence metric.)

Because the BEM is solved in **displacement** (a fault enters only as a smooth
slip -> displacement source on the RHS), the eigenstress never enters the solve
— subtracting it there would be wrong.  It belongs in the **stress readout**:
`mbem.evaluate_stress` sums the boundary and fault stress kernels and removes the
fault eigenstress, producing the genuine on-fault elastic (Coulomb) stress the
BEM previously could not.  `configs/onfault_stress.py` reads it off a
free-surface box solve and recovers the half-space dislocation stress.

## Layout

```
src/mbem/                 region-model BEM solver: dense and block-compressed
                          backends (far="aca" flat H + ACA, far="fmm" bbFMM),
                          numba kernels, topography, preconditioned FGMRES
  backends/               dense and block-compressed assembly
  model/                  region graph, patches, faults, the free term
  kernels/                numba kernels, P0/P1/P2 closed forms
  la/                     ACA, flat H-matrix, bbFMM, FGMRES, preconditioners
  cases/                  reference models the gates and studies share
  figures/                the figure makers, in a lazy registry
src/mollified_kernel/     point-source kernel + analytic triangle integration
  mollified_elastic_kernels.py   regularized Kelvin point kernels
  analytical_kernels.py          analytic per-triangle integration (scalar)
  analytical_batch.py            analytic per-triangle integration (vectorized)
src/moss_kernel/          a second, INDEPENDENT copy of the analytic kernels;
                          the gates compare the two entrywise
src/clq/                  frozen oracle: separate closed-form derivation
src/mollified_bem.py      TriMesh, ElasticMaterial, full-space Kelvin kernels
src/local_box_mesh.py     box / fault mesh builders
src/local_box_mesh_eq.py  equilateral Delaunay box / fault mesh builders
src/inclusion_mesh.py     host + cylindrical-inclusion mesh builder
src/anelastic.py          anelastic (eigenstrain) eigenstress -> elastic stress
src/tde_reference.py      classical triangular-dislocation reference
configs/                  seven studies, each a Python module declaring a run
tests/                    run_all.py + test_gates.py; 45 gates (print PASS/FAIL)
studies/                  bench_scaling.py and the printing quickstarts
docs/                     clq derivation, the two closed-line findings
```

`docs/figures/` is the tracked figure gallery and ships **empty by design** —
`python -m mbem publish <run-dir>` is what puts a figure there, with the
provenance of the run that made it. Nothing writes to it as a side effect.

## Install

Python >= 3.11. MIT licensed.

```
pip install -e .                 # numpy scipy matplotlib numba triangle threadpoolctl
pip install -e '.[tde,test,viz]'  # + cutde (two gates FAIL without it), pytest,
                                  #   and vtk (verify_volume's independent reader)
# plus an OpenMP runtime numba can load (conda-forge llvm-openmp, or brew libomp)
```

One editable install and everything imports as a package from any working
directory. Nothing in the solver's import path manipulates `sys.path`, and
adding some there would be a bug. Three places legitimately do: the `__main__`
self-test blocks of the two frozen `analytical_kernels.py` copies, and the gate
harness (`tests/conftest.py`, plus two `moss_kernel` gates), which has to place
a specific oracle directory ahead of an identically-named sibling.

numba must resolve its OpenMP threading layer (`numba.threading_layer() ==
"omp"`) and threadpoolctl must be importable: the compressed backend runs
nogil kernels on a Python thread pool while parallel kernels run beside them,
which the OpenMP layer tolerates and the default workqueue layer does not,
and threadpoolctl is what pins BLAS to one thread inside that pool.

## Run a study

A study is a Python config module; each run writes a new self-describing folder
under `runs/` (gitignored).

```
python -m mbem run configs/fault_box.py
python -m mbem run configs/topo_inclusion.py --set surface=flat --set backend=fmm
python -m mbem run configs/fault_box.py --dry-run   # validate only, build nothing
python -m mbem list
python -m mbem show <run-dir> --section effective
python -m mbem verify [-k fmm] [--fast]             # the gates, through the CLI
```

A run folder holds `resolved.json` (the spec, the keyword arguments actually
passed, eps resolved to numbers per patch, all 111 `defaults` constants, and the
environment), `report.json` (iterations, residuals, phase timings, peak RSS),
the solution fields as `.npz`, and a `MANIFEST` of sha256 hashes.

A config *names* a mesh and model builder rather than describing patches and
boundary conditions, so it cannot restate the fault sign convention or the eps
rule. `tests/gates/mbem/verify_config.py` proves the config path builds the same
model the gates do, fault Burgers vector bitwise included.

## Verify the kernels

Every gate prints one final `PASS: <title>` / `FAIL: <title>` line and exits 1
on FAIL; `python tests/run_all.py` runs all 45 sequentially and tabulates
verdict and wall time; `pytest` is the same set behind a second front end.
`--fast` / `-m "not slow"` skips the ~900 s FMM gate.

```
python tests/gates/mbem/verify_analytical_vs_quadrature.py   # analytic == high-order quadrature
python tests/gates/mbem/verify_arbitrary_triangle.py         # arbitrary-triangle / rigid / scaling
python tests/gates/mbem/verify_batch_vs_scalar.py            # vectorized == scalar
python tests/gates/mbem/verify_dd_pairing.py                 # lambda/mu pairing of the DD displacement kernel (closure, nu sweep); point kernel satisfies the regularized Navier equation
python tests/gates/mbem/verify_evaluate_stress.py            # mbem stress == classical TDE (cutde); on-fault elastic stays finite
python tests/gates/mbem/verify_stress_assembler.py           # numba batched stress assemblers == scalar oracle (machine precision)
python tests/gates/mbem/verify_eigenstress_exact.py           # EXACT finite-triangle eigenstress == moss/clq oracles; sign; rim disagreement with anelastic.py
python tests/gates/mbem/verify_boundary_eigenstress.py       # boundary double layers subtracted too: near-boundary interior stress converges (icosphere vs exact Kelvin)
python tests/gates/mbem/verify_solved_bvp.py                 # assembly -> BCs -> solve -> displacement vs cutde half-space and manufactured solutions (end to end)
python tests/gates/mbem/verify_disp_contract.py              # matrix-free displacement evaluation == dense matrices == legacy oracle
python tests/gates/mbem/verify_dense_backend.py              # dense assembly invariants (calibration cache bit-identity, rebuilds)
python tests/gates/mbem/verify_hbackend.py                   # H backend == dense (ACA, calibrated jump, combined storage, rung ladder)
python tests/gates/mbem/verify_eps_auto.py                   # eps="auto" (0.1 h): kernel order-2 convergence, solved-BVP accuracy vs Kelvin, half-jump guard
python tests/gates/mbem/verify_deflation_estimate.py         # all-Neumann rigid-body deflation; memory estimator
python tests/gates/mbem/verify_nodal_kernels.py              # P0/P1/P2 numba kernels == clq oracle (1e-12); edge primitives; order 0 == the P0 path
python tests/gates/mbem/verify_nodal_solve.py                # P1/P2 solves: P0 bitwise, patch test, rigid covariance, h-convergence, P1 faults, refusals
```

## The field in the volume

The solve reports its answer on the boundary. `mbem sample` re-evaluates a
stored run's solution on a 3-D grid and writes VTK ImageData, which opens by
dragging into ParaView Glance or with desktop ParaView:

```
python -m mbem sample <run-dir>                       # 4 km default
python -m mbem sample <study-dir> --spacing 8 --difference
python -m mbem sample <run-dir> --state het --no-eigenstress
```

Post-processing, not a re-solve: the model is rebuilt from the run's own spec
(the mesher is re-run, ~3 ms, because `meshes.npz` does not carry
`bundle.scalars` and so cannot reproduce a fault), the per-patch `mesh_sha256`
is checked against the stored fingerprint, and the assembly and solve are
skipped. The run must have saved every slot — `outputs=Output(slots=())` — since
an interior point needs the density on all of them.

Each `.vti` carries displacement, elastic stress and strain as six named
components each, the subtracted eigenstress `C:eps*`, and `u_mag`,
`von_mises`, `mean_stress`, `max_shear`, `dilatation`. A study samples every
child on **one shared grid**, which is what makes a cross-run difference well
defined when the two runs have different meshes; a difference is written only
where both bodies contain the point, and its scalars are recomputed from the
differenced tensors rather than differenced (the von Mises of a perturbation is
not a difference of von Mises values).

Two arrays decide whether a value is trustworthy. `region` is 0 outside the
body — points above the topography and outside the box are NaN, because 0 is a
value. `clearance_h` and `clearance_eps` give each point's distance to the
nearest element in units of that element's size and of its own eps. **Both are
flagged, never blanked**, and it matters: on the showcase model the topography
effect reads 1287 mm at a point lying on the surface and 96 mm once
`clearance_h > 0.5`.

### What the eigenstress subtraction does and does not buy

Subtracting `C:eps*` is what makes a stress *finite and convergent* where the
singular formulation diverges — on a fault, which is why `_warn_near_boundary`
exempts faults. It does **not** make the stress exact, and the two limits are
separable only if eps is given absolutely (`eps="auto"` ties it to 0.1 h).
Measured on the manufactured `u = A x` box, whose interior stress is exactly
`C:A`, over h in (20, 10, 5) km x eps in (0.6, 1.5, 3.6) km:

| | eps = 0.6 | eps = 1.5 | eps = 3.6 |
|---|---|---|---|
| deep interior (`clearance_h > 2`), h = 10 km | 7.3e-3 | 1.66e-2 | 4.21e-2 |
| deep interior, h = 5 km | 6.8e-3 | 1.71e-2 | 4.13e-2 |
| degradation near a boundary (`clearance_h` 0.15-0.3) | 12.9x | 4.7x | 2.9x |

So there is a floor everywhere, and it is **eps/L for the DOMAIN size L** —
not eps/h. So 1 % needs eps <~ 0.01 L, and refining the mesh does not help.
Measured, each factor isolated:

| held fixed | varied | result |
|---|---|---|
| L, eps | h: 10 -> 5 km | moves under 8 % |
| L, h | eps x8 | `eps^1.01` at P1 and P2 |
| eps, eps/h = 0.145 | L: 40 -> 120 km | `err*L/eps` = 1.060, 1.091, 1.101 (`err*h/eps` moves 3x) |
| L, h, eps | order P0 -> P1 -> P2 | P1 and P2 agree to 0.5 % |
| L, h, eps | jump calibrated -> half | agree to 9 % |
| — | a CONSTANT field | 3e-15 at every eps and every order |

It is not the discretization (flat in h), not the density order (P1 == P2), and
not the calibration (half == calibrated). What is left is the smeared boundary
itself: in the interior a symmetric unit-integral blob has zero first moment and
reproduces a linear field exactly, but AT A BOUNDARY the body is on one side
only, the convolution is truncated, and the effective surface sits O(eps) off —
which costs a field with a gradient a relative `eps/L`. That last step is an
inference from the scalings rather than a derivation, but it is what survives
after order and free term were ruled out, and it predicts the asymmetry below:
a fault is interior, its blob is two-sided, and this error does not arise there.

Proximity to a *solved* boundary degrades it a further 3-13x on top. Neither
variable governs alone: binning on `clearance_h` leaves a 4.2x spread across
(h, eps) and on `clearance_eps` a 3.8x spread, which is why both arrays ship.

**A fault is exempt**, and for a stronger reason than its slip being prescribed
data. Because the regularization is applied to the SOURCE, a fault's smearing
over eps IS the finite-width fault zone the method exists to represent, so a
point 2 eps from the fault is reading the model. A boundary patch's smearing
has no such warrant — the free surface is not physically smeared — so within
~eps of it you are inside a layer the numerics invented. The same quantity,
distance in units of eps, is physics at one and an artefact at the other, which
is why `clearance_h`/`clearance_eps` are measured to BOUNDARIES only and
`fault_eps` is reported separately and is not a defect measure.

One practical consequence, against what the warning text advises: at fixed eps,
refining h by 4x improved the near-boundary residual only 1.27x, while standing
off from `clearance_h` 0.2 to 2 improved it 13x. Evaluate deeper; refining the
patch alone barely helps.

Cost is linear in the interior point count and dominated by the stress kernel:
measured 2.3 ms per point on the 11k-triangle showcase model (0.22 for
displacement, 2.09 for stress with the eigenstress in the same pass). So 8 km
spacing is ~1.6 min a state and 4 km is ~12 min.

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
before assembling; `studies/mbem/bench_scaling.py` is the regression harness:
`--model fault_box|topo_inclusion --scale s` mesh ladders with per-rung JSON
records (phase times, iterations, ranks, bytes, RSS, operator error against
the dense operator or exact matrix-free rows), `--gate REV` against the
`bench-json:` line of a commit message, and `--panel N` for the large-block
compression primitive alone.

## Examples

Every figure is produced by a named maker, and the output lands in a new
directory under `runs/` together with the provenance of whatever produced it.
Three kinds, because the figures are genuinely of three shapes:

**Model-free** — the kernels alone, no solve:

```
python -m mbem figure point_kernel            # singular vs mollified point source
python -m mbem figure triangle_field          # triangle stress field
python -m mbem figure triangle_eps_sweep      # the field over an eps ladder
python -m mbem figure eps_h_convergence       # eps-h decoupling (analytic)
python -m mbem figure anelastic_subtraction   # the anelastic term is finite
python -m mbem figure onfault_convergence     # on-fault stress -> const, vs TDE
python studies/mbem/demo_triangle_quickstart.py     # prints numbers, no figure
```

**One solve:**

```
python -m mbem run configs/fault_only.py      # displacement + elastic surface stress
```

**Several runs**, because the quantity drawn is a difference between operators —
a different mesh, eps or backend is a different operator, so it is a different
run:

```
# fault + inclusion + topography: the four-state decomposition
python -m mbem run configs/topo_inclusion.py --sweep surface=topo,flat

# the fast operator against the reference, same model and eps
python -m mbem run configs/backend_agreement.py --sweep backend=hmat,dense

# surface field convergence in eps, at a fixed mesh
python -m mbem run configs/eps_convergence.py --sweep eps=12,8,6,4,3,2

# on-fault shear vs the classical TDE, and the first-row P0/P1 contrast
python -m mbem run configs/onfault_stress.py --sweep eps=4,2,1 --sweep order_top=0,1

# what the coarse box sides cost the top surface, top mesh held fixed
python -m mbem run configs/side_grading.py --sweep edge_side=80,40,20,10
```

`python -m mbem publish <run-dir>` copies chosen figures into the tracked
gallery at `docs/figures/` and appends a provenance line naming the run, the
commit and the file hash. Nothing writes there as a side effect: before this,
every demo overwrote one tracked image in place, so the committed figure was
whatever ran last and nothing recorded which code produced it.

The BEM demos run at paper resolution (tens of seconds to a few minutes; the
four-state topography+inclusion solve is the longest).  The free surfaces are
fault-ALIGNED (the surface-breaking fault trace is embedded as exact mesh edges
via `make_top_patch_with_fault`), so no triangle straddles the slip
discontinuity.  The inclusion is the soft `mu/10` body in the showcase; any
model keyword is reachable from the command line as `--set mu_inc=3.0`.
The `topo_inclusion_showcase` figure draws the three rows of the
decomposition (raw field / inclusion effect / topography effect) and
`topo_inclusion_contour` the topography effect alone.

The topography effect differences two fields that agree to 43-72x their
difference, so it inherits ~2e-4 relative error from fields accurate to ~2e-6.
That is a property of the quantity, not of the backend: the ACA and FMM far
fields land in the same place.

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
