# ddbem — displacement-discontinuity collocation BEM on mollified triangles

The foundation the program builds on: a **direct / DD** formulation whose
unknown is a displacement discontinuity, with **one code path for constant,
linear and quadratic** nodal slip (Lagrange P0 / P1 / P2) on flat triangles,
Cortez-regularised (`R = sqrt(r^2 + eps^2)`). Slip is the conventional
`Delta u = u(+nhat) - u(-nhat)`; msd's fault `value` is the opposite, so
ddbem/clq slip = -(msd slip).

Why DD and not the indirect single layer: the force-element (equivalent-body-
force) BEM was built, measured and dropped — `../fbem/FINDINGS.md`. Its unknown
is a **density** with a genuine `rho^(-1/3)` edge singularity on a polyhedron,
so free-traction rows stall at `O(h^0.31)` against the direct BIE's `O(h^0.90)`,
and neither p-refinement nor mesh grading rescues it. The DD unknown is a
**displacement**, which is bounded.

**Stage 1** is the kernel / assembly layer (`ddbem/assemble.py`, `layout.py`,
`mesh.py`). **Stage 2** is the solver stack (`ddbem/model.py`, `shapes.py`):
patches with boundary conditions, collocation rows, the free term, and the
solve. **Stage 3** is independent verification (seeded defects run against the
gates, "Adversarial verification" below) and the convergence study
(`convergence.py`), which is what the sub-project exists to produce.

**The one-line answer of the convergence study.** On the interior field DD
converges cleanly at every order on both a sphere and a cube. On the BOUNDARY
trace over a cube it does not: the error is governed by how many mollification
lengths the collocation point is clear of its own element's edge. Over 60
cached cube solves per order, `log(err)` regressed on `log h` alone has
**R² = 0.00 / 0.02 / 0.02** at P0 / P1 / P2, and regressed on
`log(clearance/eps)` alone **R² = 0.70 / 0.96 / 0.96** — the mesh explains
nothing and the mollification budget explains almost everything. At the tree's
default `eps/h = 0.3` P1 and P2 are *worse* than P0 at matched unknowns. This
is `../fbem/FINDINGS.md` sec. 2 reproduced for the DD formulation, on the
quantity that shows it.

## Running

No build, no installer, no pytest — the tree has none anywhere. Run from the
`ddbem` root with a Python that has numpy and scipy; the msd parity gate also
needs numba.

```bash
PY=/Users/meade/micromamba/bin/python     # this machine
$PY verify/run_all.py                     # every gate prints one PASS/FAIL line
$PY verify/verify_kernels.py              # kernels: parity, PoU, eigenstress, quadrature
$PY verify/verify_layout.py               # conventions: columns, ordering, nodes, scatter
$PY verify/verify_solver.py               # rows, BCs, free term, exact solutions
$PY verify/verify_msd_parity.py           # P0 solver == msd's solver, entrywise
$PY verify/verify_convergence.py          # the convergence study's HARNESS
$PY verify/verify_coverage.py             # what the seeded-defect sweep found unpinned
$PY examples/fault_in_a_box.py            # worked example
$PY bench/sweep_collocation.py            # where to collocate a P1/P2 element
$PY bench/bench_assembly.py               # assembly timings
$PY bench/seeded_defects.py               # break it on purpose; see which gates notice
$PY bench/seeded_defects.py --list        # the 17 defects, + the CTRL_/HARD_ controls

# the convergence study (stage 3).  Results are cached in convergence_cache/,
# so `run` resumes and `report` is instant.
$PY convergence.py plans                          # what there is to run
$PY convergence.py run --plan headline            # both geometries, both rows
$PY convergence.py run --plan eps_ratio eps_abs   # the eps sweep
$PY convergence.py run --plan msd msd_abs         # msd's frozen P0 anchor
$PY convergence.py run --plan ptest jump shrink   # p-refinement at a common point
$PY convergence.py report                         # tables + rates
$PY convergence.py collapse --metric u_surf       # is it h, or is it eps?
```

`convergence.py run` takes `--geoms`, `--bcs`, `--orders`, `--levels`,
`--tag` and `--max-dof`, so one plan can be split across several processes
(one cache file per tag; `report` merges and de-duplicates). The whole study
above is ~300 solves and about an hour on 8 cores. `bench/study_convergence.py`
is stage 2's earlier, single-geometry version and is superseded by this.

Scripts put the ddbem root on `sys.path` themselves, so launch them from the
root, not from inside `verify/`. `clq` is found at `../clq` by file location
(`DDBEM_CLQ_ROOT` overrides).

## API

### Matrices (stage 1)

```python
import ddbem

A = ddbem.displacement_matrix(x_field, tri_verts, eps, mu, nu, order)         # (3 N_f, n_dof)
T = ddbem.traction_matrix(x_field, n_field, tri_verts, eps, mu, nu, order)    # (3 N_f, n_dof)
S = ddbem.stress_matrix(x_field, tri_verts, eps, mu, nu, order)               # (6 N_f, n_dof) Voigt
E = ddbem.eigenstress_matrix(x_field, tri_verts, eps, mu, nu, order)          # (6 N_f, n_dof) Voigt
```

`tri_verts` is `(N_tri, 3, 3)`; the unit normal is `(v2-v1) x (v3-v1)`
normalised, from the vertex order alone (clq's rule — there is deliberately no
separate normal argument). Slip sign `Delta u = u(+nhat) - u(-nhat)`. `eps` is a
scalar or a per-source-element `(N_tri,)` array. Units km / GPa / years
(slip 0.001 km = 1 m).

### Solver (stage 2)

```python
v, t = ddbem.icosphere(2, radius=5.0)          # or ddbem.box(...), ddbem.rectangle(...)
box   = ddbem.tri_verts(v, t)
fault = ddbem.tri_verts(*ddbem.rectangle([-2, -1.5, 0], [4, 0, 0], [0, 3, 0], 2, 2))

model = ddbem.Model(
    [ddbem.Patch("surface", box,   ddbem.BCType.FREE_TRACTION,          eps="auto"),
     ddbem.Patch("fault",   fault, ddbem.BCType.FAULT, value=[1e-3, 0, 0], eps="auto")],
    mu=30.0, nu=0.25, order=1, jump="calibrated")
print(model.report())                          # sizes, eps/h budget, row types
sol = model.solve()                            # -> Solution
u   = sol.displacement(points)                 # (N, 3)
s   = sol.stress(points)                       # (N, 6) Voigt, ELASTIC
us  = sol.trace_displacement("surface")        # boundary displacement
```

`BCType`: `FREE_TRACTION` (traction prescribed, default zero),
`PRESCRIBED_DISPLACEMENT`, `FAULT` (the density itself is prescribed;
an interior source, no rows), `INTERFACE` (**declared, raises** — see below).
`Patch` takes per-patch `eps`, `order` and `orientation`; `value` may be a
constant, a per-element or per-node array, or a callable `f(points)` /
`f(points, normals)`. `Model` takes `order`, `eps`, `shrink`,
`jump="calibrated"|"half"`, `neumann_row="traction"|"exterior"`;
`Model.solve(constrain="rigid"|"translations"|"none")`.

Geometry and budget helpers: `element_nodes`, `collocation_points`,
`element_normals`, `element_areas`, `element_h`, `node_clearance`, `eps_report`,
`eps_auto`, **`shrink_for_clearance`**, `shape_at`,
`collocation_shape_matrix`, `voigt_to_tensor`, `tensor_to_voigt`,
`discontinuous`, `continuous`; shapes: `icosphere`, `box`, `rectangle`,
`tri_verts`, `shapes.is_closed`, `shapes.enclosed_volume`, `shapes.weld`.

## The formulation

One unknown field: a DD density `q` on every boundary element, in ONE uniform
medium. The whole model is a single representation,

```
u(x) = sum_boundary D(x, y) q(y) + sum_faults D(x, y) s(y)
```

with `D` the slip→displacement kernel and `T` its hypersingular traction
counterpart. A mollified kernel evaluated ON its own element returns the blob
AVERAGE across the discontinuity, so with `sigma = +1` for outward normals the
one-sided traces are

```
u_interior(x_c) = [D q](x_c) - (sigma/2) q(x_c)
u_exterior(x_c) = [D q](x_c) + (sigma/2) q(x_c)
```

and the traction has **no** jump (the traction of a double layer is continuous),
so a traction row carries no analytic free term. For a nodal density
`q(x_c) = sum_k N_k(x_c) q_k`: the free term is the **matrix of shape-function
values at the collocation point**, not a scalar 1/2.

Three row types:

| row | equation | used for | analytic row sum |
|---|---|---|---|
| `TRACTION` | `t(x_c) = t_bar` | free / prescribed traction (the default) | 0 |
| `INTERIOR_DISPLACEMENT` | `u_int(x_c) = u_bar` | prescribed displacement | I/2 |
| `EXTERIOR_NULL` | `u_ext(x_c) = 0` | `neumann_row="exterior"` | I/2 |

`EXTERIOR_NULL` is the second-kind alternative for a body whose **entire**
boundary is traction-free: zero boundary traction makes the whole exterior
field vanish, so "the representation is null outside" says the same thing. It is
exactly msd's equation rewritten in `q`, and it is refused on any model with a
non-zero prescribed traction or a prescribed displacement.

The two are not equally well behaved, and the gate measures the difference on
the same fault-in-a-sphere problem. On a traction-free body the interior trace
must be `u_int = -sigma q`, because the exterior field vanishes. Under the
exterior row that *is* the equation, so it holds to 1e-15. Under the
hypersingular row it holds only where the collocation points are, and the gap —
the discrete exterior field the traction row does not remove — is **0.51 on the
20-triangle sphere and 0.194 on the 80-triangle one** (falling at `O(h^1.6)`).
The interior *stress* fields of the two nevertheless agree to 9.8e-2 and
3.1e-2 respectively: the part of the density they disagree about is the part
that produces no interior field. Use the traction row because it takes any
mixed BC; know that the second-kind row is the cleaner operator where it
applies.

## The free term is a row-sum identity, and it generalises

Write `R` for a row's 3x3 action on the rigid-translation mode. A rigid
translation `c` of the body is the DD density `q = -sigma c` (the interior moves
by `c`, the exterior does not move at all), and each row's answer on it is known
exactly — `u_int = c`, `u_ext = 0`, `t = 0`. So

```
F = sigma_c * (R - target),   target = I for an interior-displacement row, else 0
R[i,j] = - sum_{p,s,k} sigma_p * A_raw[row_i, col(p,s,k,j)]
```

applied to the collocation element's own nodes, weighted by `N_k(x_c)`.
`jump="half"` substitutes the analytic `R` (I/2 or 0); `jump="calibrated"` uses
the measured one. **It generalises to P1/P2 for free**: it is a row-sum
identity, and the partition of unity `sum_k N_k = 1` carries it across
unchanged. At P0 it reduces to msd's `C_q = -sum_p sigma rowsum H_qp`, which
the parity gate confirms entrywise.

Three measured facts came out of implementing it.

**1. `R` for a displacement row is exactly `phi(x_c) * I`** — a scalar times the
identity, isotropic to 1e-15 at every order, on a sphere and on a box, at
nu = 0.25/0.30/0.45. `phi` is the mollified smoothed indicator function at the
collocation point, and the classical free term is the assertion `phi = 1/2`.
Measured: **0.288 … 0.466**, i.e. the analytic jump is wrong by up to **42 %**
(the low end is a box corner, where the surface is not flat; on the sphere it is
0.44–0.49). This is what the calibration repairs, and it is why it matters.

**2. The hypersingular row needs no calibration at all.** Its raw row sum is
already **machine zero** (worst 7.1e-14 relative at nu = 0.45, P2). That is not
luck. The mollified closure identity `sum_S D(x, y) = -phi(x) I` is exact
*pointwise* (it is fact 1), so a uniform DD `q` on a closed surface produces
exactly `u = -phi q` — a purely anelastic field whose total stress IS its own
eigenstress `C:eps*`. Subtracting the exact finite-triangle eigenstress
(`../BACKLOG.md`) therefore leaves **zero elastic stress everywhere**,
and the hypersingular row annihilates the rigid mode before any calibration.
Tripwire in the gate: with `subtract_eigenstress=False` the same row sum is
**4.4e15 times larger** (7.4e0 against 1.7e-15).
So the answer to "does the calibration generalise to the hypersingular row" is
*it is already exact there* — the work the calibration does on a displacement
row has been done in advance by the eigenstress policy.

**3. The free term must be the shape-function MATRIX, and a rigid-body test
cannot see that.** With `sum_k N_k = 1`, a constant density gives the same
answer whether the free term is `N_k(x_c)` or the identity — so the patch test
passes either way, and a code that only ever runs a patch test will never learn
that its higher-order free term is wrong. Measured on a solve whose density
varies inside an element (the Kelvin problem, 80-triangle sphere,
`shrink = 0.5`): the identity is **1.12x worse at P1 and 0.91x at P2**. So the
difference is real but is *not* a one-sided accuracy claim; `N_k(x_c)` is used
because it is the derived form (the free term multiplies the density *at the
collocation point*, which is `sum_k N_k(x_c) q_k`), not because it always wins.
It is a no-op at `shrink = 0` and for traction rows, whose calibrated free term
is identically zero by fact 2.

### The rigid-motion constraint

A body whose whole boundary is traction-free is determined only up to a rigid
motion, so the continuum operator has a **6**-dimensional null space
(translations *and* rotations: both give zero stress, so both are annihilated by
a traction row, and both make the exterior field vanish, so both are annihilated
by an exterior row). `Model.solve(constrain=...)` borders the system,
`[[A, Z], [Z^T, 0]]`:

* `"translations"` (3) — msd's `deflate=True`. With calibration these are an
  **exact** discrete null space, so the constraint force is zero and the rows
  are satisfied to machine precision (measured residual 8e-16).
* `"rigid"` (6, the default) — also borders the rotations, which are only
  *near*-null discretely. That improves conditioning (97 vs 247 on the parity
  problem) and returns a canonical representative, at the price of a non-zero
  rotation constraint force: the rows are then satisfied to **3.7e-4 on the
  20-triangle sphere and 6.4e-6 on the 80-triangle one** — it converges away
  under refinement and sits ~2 decades below the solution error, but it is not
  zero and the gate says so.
* `"none"` — the default when any patch prescribes a displacement, which anchors
  the body already.

### Patch test

Prescribe `u = c` on a closed sphere: the exact density is `q = -sigma c`.

| order | `jump="calibrated"` | `jump="half"` |
|---|---|---|
| P0 | 3.7e-15 | 1.9e-2 |
| P1 | 1.2e-14 | ~7e-2 |
| P2 | 8.9e-14 | 1.8e-1 |

(80-triangle icosphere, nu = 0.25; the nu = 0.30 / 0.45 rows are the same to a
factor of 2, and the half-jump range over all nine cases is 1.9e-2 … 1.8e-1.
The half-jump column is the tripwire: the gate is measuring the calibration,
not the linear solver.)

### Exact solution: a Kelvin point force outside the body

80-triangle icosphere, `eps/h = 0.3`, `jump="calibrated"`, interior
displacement error, with the coarse (20-triangle) value and the rate in h:

| rows | P0 | P1 | P2 |
|---|---|---|---|
| Dirichlet | 4.97e-2 → 4.24e-3 (+4.16) | 4.36e-2 → 2.80e-3 (+4.64) | 4.20e-2 → 2.82e-3 (+4.57) |
| Neumann | 5.70e-3 → 3.22e-3 (+0.97) | 3.74e-3 → 1.30e-3 (+1.79) | 3.22e-3 → 8.62e-4 (+2.23) |

A mixed model (36 Dirichlet + 44 traction-free triangles in one system) reaches
1.5e-2 at nu = 0.30 and 1.8e-2 at nu = 0.45.

## Where to collocate a P1/P2 element

A P1 node is a vertex and a P2 node is a vertex or an edge midpoint, so at
`shrink = 0` every higher-order collocation point sits **on** the element
boundary. `ddbem.mesh.barycentric_nodes` pulls it in,
`lam_c = (1-t) lam_node + t/3` (fbem's parametrisation; it swept the same `t`
for a single-layer density and chose 0.5). The basis does not move — only the
collocation point — which is what makes the free term a matrix.

`bench/sweep_collocation.py`, 80-triangle icosphere, Kelvin point force outside,
**interior displacement error / condition number**:

| bc | P | eps/h | t = 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| Dirichlet | 1 | 0.30 | 5.95e-3 / 6.2 | 4.60e-3 / 7.0 | 3.49e-3 / 8.2 | 2.72e-3 / 9.4 | 2.80e-3 / 12 | 2.84e-3 / 19 |
| Dirichlet | 2 | 0.30 | 2.818e-3 / 17 | 2.817e-3 / 23 | 2.817e-3 / 25 | 2.818e-3 / 25 | 2.819e-3 / 66 | 2.818e-3 / 204 |
| Dirichlet | 1 | 0.15 | 3.71e-3 / 8.2 | 3.13e-3 / 9.3 | 2.75e-3 / 11 | 2.38e-3 / 12 | 1.71e-3 / 14 | 1.85e-3 / 24 |
| Dirichlet | 2 | 0.15 | 1.745e-3 / 23 | 1.724e-3 / 25 | 1.704e-3 / 31 | 1.685e-3 / 40 | 1.650e-3 / 78 | 1.627e-3 / 262 |
| Neumann | 1 | 0.30 | **2.46e-1 / 5.1e19** | 7.42e-3 / 2.2e3 | 3.86e-3 / 343 | 2.57e-3 / 343 | 1.30e-3 / 343 | 5.55e-4 / 343 |
| Neumann | 2 | 0.30 | **2.09e-2 / 3.5e19** | 2.11e-3 / 5.7e3 | 1.50e-3 / 702 | 1.17e-3 / 702 | 8.62e-4 / 702 | 7.62e-4 / 1.9e3 |

Read off:

* **`shrink = 0` is numerically singular for the hypersingular rows** —
  condition number 3.5e19 … 5.1e19, and the error is 25–190x worse than at any
  positive `t`. The collocation points are then the shared vertices and
  midpoints of the mesh, so adjacent elements collocate at the *same* point with
  nearly parallel normals. It is merely mediocre for displacement rows
  (cond 6–23), which is how a Dirichlet-only code can ship with `shrink = 0` and
  never notice. This alone justifies the parameter.
* The traction-row error then falls **monotonically** over the whole range
  measured, while the condition number is flat from t = 0.2 to 0.6 and starts
  to inflate at 0.7 for P2 (702 → 1.9e3 → 4.6e3 at 0.8). The displacement rows
  are much less sensitive, and the P2 Dirichlet rows are *completely flat*
  (2.817e-3 … 2.819e-3 across the whole range at eps/h = 0.3, equal to P1's
  best) — that is the eps floor, not the collocation scheme, so at eps/h = 0.3
  the choice is made entirely by the traction rows.
* **The right `t` is the one that buys one eps of clearance, and the sweep says
  so sharply.** At eps/h = 0.15 the P1 hypersingular error has a clear minimum
  at `t = 0.6` — 9.86e-5, **6.7x** better than `t = 0.5` (6.60e-4) and 4.5x /
  8.1x better than 0.7 / 0.8 — and `t = 0.6` is exactly where the node clearance
  first reaches **1.03 eps**. At eps/h = 0.3 no `t <= 0.8` reaches 1 eps
  (the best is 0.69) and the error is still falling at the end of the sweep.
  Both facts say the same thing, and it is `../fbem/FINDINGS.md` sec. 2 turned
  into a rule you can act on: **put the collocation point about one
  mollification length clear of the element boundary, and no further.**
  `ddbem.shrink_for_clearance(tri_verts, order, eps)` returns that `t`
  (0.58 at eps/h = 0.15, 0.31 at 0.08, capped at `SHRINK_MAX = 0.8` when the
  target is out of reach), and `verify_solver.py` gates both the geometry and
  the physics: at eps/h = 0.15 the rule's `t = 0.58` gives **1.59e-4 against the
  fixed default's 6.60e-4, 4.2x better**, on the P1 traction rows.
* **Chosen default: `COLLOCATION_SHRINK_BY_ORDER = {0: 0.0, 1: 0.5, 2: 0.5}`.**
  fbem's independent sweep of the same parameter chose 0.5 too, and 0.5 sits
  inside the flat-conditioning plateau at both orders. But 0.5 is **a
  conditioning-guarded fixed number, not an error minimum**; the clearance rule
  above is the better answer and is deliberately NOT the default (it is one
  measured rule on one geometry, and making it automatic would hide the
  eps/h budget it depends on). Pass it as `Model(shrink=...)`, and sweep around
  it on the real geometry rather than inheriting a sphere's answer.

## The convergence study (stage 3)

`convergence.py`. Exact solution: a Kelvin point force **outside** the body
(`verify/_exact.py` for why that is the only exact solution a boundary-only
representation can be asked to reproduce), solved as a Dirichlet problem
(displacement rows) and as a Neumann problem (the hypersingular traction row).
Two domains, because `../fbem/FINDINGS.md` found the stall to be **edge-driven**
and the sphere is what separated the two:

* **sphere** — icosphere, radius 1, 20 / 80 / 320 / 1280 triangles. Smooth; its
  only edges are the O(h) dihedral creases of the polyhedron, which flatten
  under refinement.
* **cube** — side 2, 48 / 108 / 192 / 432 triangles. Twelve permanent 90°
  edges and eight corners — the 270° exterior wedge that gave the single-layer
  density its `rho^(-1/3)` singularity.

Three error measures, all relative to `max|u_exact|`, ~320 cached solves:

| measure | what it is |
|---|---|
| `u_int` | interior displacement on a fixed 7-point cloud at \|x\| ≤ 0.3 (never within ~0.6 h of the surface, because the representation is mesh-limited there) |
| `u_surf` | the BOUNDARY trace (`Solution.trace_displacement`) at the collocation points, Neumann rows only — on a Dirichlet row the trace *is* the data |
| `u_surf_cen` | the same trace read at the element **centroid** for every order, so a p-comparison is not confounded by the collocation point moving with the order |

### 1. The interior field converges; the boundary trace over a cube does not

Rate in h of `u_int`, `eps/h = 0.3`, `jump="calibrated"`, full ladders:

| geom | rows | msd P0 | ddbem P0 | ddbem P1 | ddbem P2 |
|---|---|---|---|---|---|
| sphere | Dirichlet | +0.37 † | **+2.15** | +2.57 | +2.54 |
| sphere | Neumann | **+1.45** | +0.86 | +1.20 | +1.23 |
| cube | Dirichlet | +0.51 | **+1.78** | +1.50 | +1.46 |
| cube | Neumann | **+1.16** | +0.33 | +0.73 | +0.75 |

† msd's sphere/Dirichlet ladder is non-monotone (8.6e-3 → 1.5e-2 → 9.0e-3 →
4.7e-3), so its fitted rate is a description, not a convergence order.

Rate in h of the boundary trace `u_surf_cen` (common point, 3-level ladders):

| geom | eps/h | P0 | P1 | P2 |
|---|---|---|---|---|
| sphere | 0.30 | +0.96 | +1.03 | +0.78 |
| cube | 0.30 | +0.74 | **+0.19** | **+0.08** |
| cube | 0.15 | +0.19 | **−0.31** | **−0.12** |

**The cube stalls the surface and the sphere does not, and higher order makes
the stall worse, not better.** The edge-concentration statistic says the same
thing directly: the share of the squared surface error carried by collocation
points within 0.3 of a cube edge, divided by their share of the points, *grows*
under refinement — 1.39 → 1.59 at P0, 1.65 → 1.98 at P1, 1.55 → 1.83 at P2
(48 → 192 triangles, eps/h = 0.3). That is the signature of an edge feature the
mesh is not resolving. The interior field hides it, because the representation
integrates the surface density and smooths it away.

### 2. What sets the error is `eps`, not `h` — measured, not asserted

`convergence.py collapse` pools every cached case of one (geometry, row type)
and regresses `log err` on `log h`, on `log(clearance/eps)`, and on both.
`clearance/eps` is the in-plane distance from a collocation point to its own
element's boundary, in mollification lengths (`ddbem.node_clearance`).

Boundary trace, Neumann rows, 60 cube cases and 50 sphere cases per order
(R², and the fitted exponent):

| geom | order | `err ~ h` | `err ~ clearance/eps` | both |
|---|---|---|---|---|
| cube | P0 | 0.00 (−0.03) | **0.70** (−1.28) | 0.77 |
| cube | P1 | 0.02 (−0.36) | **0.96** (−2.10) | 0.99 |
| cube | P2 | 0.02 (−0.34) | **0.96** (−2.12) | 1.00 |
| sphere | P0 | 0.00 (+0.09) | 0.63 (−1.36) | 0.87 (h: +0.89) |
| sphere | P1 | 0.05 (+0.43) | 0.57 (−1.25) | 0.95 (h: +1.27) |
| sphere | P2 | 0.00 (+0.15) | 0.62 (−1.62) | 0.83 (h: +1.15) |

Read this carefully, because it is the study's central result:

* **On the cube, `h` explains 0–2 % of the variance of the surface error and
  the mollification budget explains 70–96 %.** Refining the mesh at fixed
  `eps/h` buys almost nothing on the boundary; the exponent in
  `clearance/eps` is −2.1 at P1/P2. Adding `h` to the clearance model moves the
  h-exponent only to +0.53.
* **On the sphere the two-variable fit keeps a real `h` term** (+0.89 … +1.27),
  i.e. once the eps budget is fixed, refining the mesh *does* help. That is
  exactly the sphere/polyhedron separation `fbem` found, now for DD.
* Within one order and shrink, `clearance/eps ∝ h/eps`, so "the clearance
  controls it" and "`eps/h` controls it" are the same statement. The regression
  separates them from `h` because the absolute-eps ladders vary `eps/h` at fixed
  order.

The same numbers seen directly, cube Neumann, boundary trace at the centroid:

| clearance/eps | case | 192-triangle `u_surf_cen` |
|---|---|---|
| 0.21 | P1, P2 at eps/h = 0.5 | 4.66e-1, 1.47e-1 |
| 0.35 | P1, P2 at eps/h = 0.3 | 1.47e-1, 9.74e-2 |
| 0.41 | P0 at eps/h = 0.5 | 2.39e-1 |
| 0.69 | P0 at eps/h = 0.3 | 5.19e-2 |
| 0.69 | P1, P2 at eps/h = 0.15 | 3.74e-2, 3.52e-2 |
| 1.38 | P0 at eps/h = 0.15 | 4.44e-2 |

P1 and P2 at `eps/h = 0.15` have exactly the clearance P0 has at
`eps/h = 0.3` — and land at the same error. **The order of the basis is not
what is being measured; the clearance is.**

### 3. Does higher-order DD pay? Only on one row type, and only with eps room

At **matched unknown count** (error interpolated onto the P0 ladder's dof
range; > 1 means higher order is worse), interior field, calibrated:

| geom | rows | eps | P1/P0 | P2/P0 |
|---|---|---|---|---|
| sphere | Dirichlet | eps/h = 0.3 | 3.38 | 7.93 |
| sphere | Dirichlet | eps = 0.12 | 1.27 | 1.35 |
| cube | Dirichlet | eps/h = 0.3 | 1.74 | 3.20 |
| cube | Dirichlet | eps = 0.12 | 1.07 | 1.00 |
| sphere | Neumann | eps/h = 0.5 | 1.30 | 2.09 |
| sphere | Neumann | eps/h = 0.3 | **0.51** | **0.53** |
| sphere | Neumann | eps/h = 0.15 | **0.27** | **0.22** |
| cube | Neumann | eps/h = 0.5 | 2.21 | 3.87 |
| cube | Neumann | eps/h = 0.3 | 1.42 | 1.48 |
| cube | Neumann | eps/h = 0.15 | **0.96** | **0.88** |
| cube | Neumann | eps = 0.12 | **0.83** | **0.48** |

* **On the displacement rows higher order essentially never pays.** Across
  every cached Dirichlet case the best it ever does is a 4 % improvement (cube,
  `eps = 0.25` absolute: P1/P0 = 0.96, P2/P0 = 0.97), and at `eps/h = 0.3` it
  costs 1.7–7.9x; at `eps/h = 0.5` up to 11.9x.
* **On the hypersingular rows it pays on the sphere (up to 4.5x) and only pays
  on the cube once `eps/h ≤ 0.15` or `eps` is held fixed in absolute terms.**
  At the tree's default `eps/h = 0.3` on a cube, P1 and P2 *cost* 1.4–1.5x.
* This is `fbem`'s inversion, with the sign it predicted: the extra nodes are
  worth their cost only when there is a mollification length between them.
* The unknown-count axis here is as unfavourable as it can be, because the
  solver only supports the **discontinuous** layout (18 unknowns per triangle at
  P2 against P0's 3). Continuous P1 would be ~1.5 — a 6x shift of this whole
  table that the study cannot measure, and the reason the continuous layout is
  the most valuable open item in the sub-project.

### 4. The free-term calibration changes the RATE, not just the constant

Dirichlet rows, `eps/h = 0.3`, interior field, rate in h:

| geom | P0 | P1 | P2 |
|---|---|---|---|
| sphere, `jump="calibrated"` | **+2.15** | **+2.57** | **+2.54** |
| sphere, `jump="half"` | +0.72 | +0.70 | +0.57 |
| cube, `jump="calibrated"` | **+1.78** | **+1.50** | **+1.46** |
| cube, `jump="half"` | +0.96 | +1.07 | +1.01 |

On the finest available mesh the calibrated answer is 6.1x more accurate at P0
(1280 triangles: 8.02e-4 against 4.86e-3) and 11.5x at P1 (320 triangles:
1.92e-3 against 2.21e-2). On the coarsest mesh the analytic half is sometimes *better* (sphere P0,
2.09e-2 against 5.66e-2), so this is a rate statement, not a uniform one. It is
the strongest argument for the calibration in the sub-project: the stage-2 gates
show it is exact on the rigid mode, and this shows it buys a rate.

### 5. The msd anchor, and one place ddbem is beaten

msd's frozen P0 direct BIE (`mbem`, numba) on the same meshes, same eps, same
data — a *different formulation* (u and t as boundary unknowns), so this is not
a parity check (`verify_msd_parity.py` is, on the one problem where the two
coincide) but a reference rate.

* **ddbem wins the Dirichlet problem outright**: sphere, 1280 triangles,
  8.0e-4 against msd's 4.7e-3 (rate +2.15 against +0.37); cube, 432 triangles,
  2.7e-3 against 5.4e-3. msd's Dirichlet unknown is the traction, through a
  first-kind single-layer operator, with the classical 1/2 jump.
* **msd wins the Neumann problem**: sphere, 1280 triangles, 1.1e-3 against
  ddbem P0's 2.7e-3 (rate +1.45 against +0.86); cube, 2.2e-3 against 2.8e-3
  (+1.16 against +0.33). ddbem P2 closes the gap in h but not per unknown.

  **The reason is structural and it is worth stating plainly.** msd applies
  Neumann data through the *single layer* `G` (a smoothing operator) and keeps
  the second-kind `1/2 I + sigma H` on the left. ddbem has one density, so it
  must state `t(x_c) = t_bar` through the *hypersingular* operator, which
  amplifies discretisation error. When `t_bar = 0` the two are the same equation
  (`neumann_row="exterior"`, gated entrywise) and the gap disappears — so this
  costs nothing on a traction-free body, and it is a real cost whenever a
  non-zero traction is prescribed.

### 6. What stage 2 got wrong, and it is a warning about this kind of study

`bench/study_convergence.py` (stage 2) reported that on Dirichlet rows at
`eps/h = 0.3` "P0 → P2 buys 1.50x". That number came from a **3-point**
observation cloud at one mesh level. Re-measured on the same solves:

| cloud | 80 triangles | 320 triangles |
|---|---|---|
| stage 2's 3 points | P0/P2 = **1.50x** | 0.66x |
| this study's 7 points | 0.88x | 0.69x |
| an independent 26-point shell at \|x\| = 0.3 | 0.90x | 0.65x |

**The claim reverses when you add observation points or refine once.** At 320
triangles all three clouds agree that P0 is ~1.5x *better* than P2 at matched h,
let alone at matched cost. The Neumann-row claim survives the same test
(P0/P2 = 3.73 / 4.85 / 2.95 across the three clouds at 80 triangles, 2.75 /
3.13 / 2.35 at 320) — the direction is robust, the factor is good to ~2x only.
Treat any single-cloud, two-level ratio in this sub-project as a direction, not
a number.

### 7. Honest limits of the study

* **Pure Python.** The largest case is 7776 unknowns (cube P2, 432 triangles,
  ~2 min). Nothing here approaches production scale; the numba port of the outer
  assembly loop is the prerequisite for a ladder that could settle the P2 rates.
* **The interior cloud sits 0.64 h from the surface on the coarsest cube**, so
  the coarsest rung of that ladder is at the edge of the mesh-limited zone
  msd warns about (`mbem/evaluate.py::_warn_near_boundary`). The rungs that
  matter for the rates are not.
* **Interior stress is recorded but never used for a rate.** It is measurably
  non-monotone in h (stage 2 documented the same).
* **No fault patch appears anywhere in the study** — it is a pure boundary-value
  convergence study. The fault path is gated in `verify_solver.py` at P0 only.
* The three-point `eps/h` sweep and three- or four-rung ladders mean every rate
  quoted here is a 3–4 point least-squares slope. The regressions in §2 pool
  20–60 cases and are the more trustworthy statements.


## Gates

`verify/verify_kernels.py` — 79 checks (stage 1): P0 parity with four
independent frozen msd entry points, block-relative, tolerance 1e-11, worst
measured 2.9e-12 at nu = 0.45 on a sliver (see the script docstring for why that
margin is 3.4x); a swapped-pairing tripwire; partition of unity and node order
against an independent barycentric basis; eigenstress exactness and the
exact-vs-point discrimination; quadrature against a Duffy-collapsed Gauss rule.

`verify/verify_layout.py` — 42 checks (stage 1): column layout, F-contiguity,
node order, clearance, scalar/array/graded eps equivalence, Euler unknown
counts, `traction_matrix == stress_matrix . n`.

`verify/verify_solver.py` — 137 checks (stage 2):

1. `ddbem.shape_at` against `clq.shape_functions` (a different code path), with
   a node-swap tripwire; the collocation shape matrix is `I` at `shrink = 0`,
   is a partition of unity and is *not* `I` when shrunk; closure, orientation
   and enclosed volume of the shapes with a flipped-triangle tripwire; and the
   four refusals the design rests on (INTERFACE, calibration on an open
   boundary, an inward-oriented boundary, exterior rows on a non-traction-free
   model).
2. The row-sum identity, on a sphere **and a box**, at nu = 0.25/0.30/0.45 and
   P0/P1/P2: isotropy of `R`, exactness of the calibrated rows on the rigid
   mode, machine-zero hypersingular row sums, the eigenstress tripwire, and the
   shape-matrix measurement.
3. The patch test and its half-jump tripwire.
4. The exterior Kelvin point force, Dirichlet rows and hypersingular rows, at
   three nu, gated on the error RATIO between two mesh levels (a ratio cannot be
   passed by loosening a tolerance). The reference's own stress formula is
   checked against a central difference of its own displacement first.
5. The one-eps clearance rule (`shrink_for_clearance`): that it hits its target
   where the target is reachable, caps out where it is not, and that its `t`
   beats the fixed default on the P1 hypersingular rows. A mixed
   Dirichlet / traction model in one system.
6. The two free-traction row types on the same fault-in-a-sphere problem: their
   interior fields agree, and agree better on the finer mesh; and
   `trace_displacement == -sigma q` to machine precision under the exterior row.

`verify/verify_msd_parity.py` — 24 checks (needs numba): at nu = 0.25/0.30/0.45
and both jump conventions, `A_ddbem == A_msd` entrywise (worst 2.2e-15),
`b_ddbem == -b_msd` (3.0e-14), the solved density `q == -u_msd` (3.9e-14) and
the interior displacement field (4.0e-15). This is machine precision, not
discretisation accuracy, because on a traction-free body the two formulations
are the same equation written twice.

`verify/verify_convergence.py` — 39 checks (stage 3): the convergence study's
HARNESS, not its result. That the planted exact solution really is a
homogeneous solution (`div sigma = 0` by central differences) and really is
exterior to both domains; that `fit_rate` recovers a planted exponent exactly
and in any units and returns NaN rather than a number on one point; that the
error measures are zero on the exact field, that the Neumann measure is blind
to a rigid motion and *not* blind to a non-rigid one; the cube-edge classifier
against hand-computed distances, and the edge-concentration statistic against
its two exact values (1 for a uniform error, 1/fraction for an edge-only one);
that `solve_ddbem` equals an inline hand-built solve to 0.0; that
`trace_at_bary` equals `Solution.trace_displacement` to 0.0 at the collocation
barycentrics, at three orders and both jump conventions; that the msd anchor
converges and lands within 3x of ddbem P0; and that the cache round-trips and
that a re-run of a cached case is skipped rather than recomputed or duplicated.

`verify/verify_coverage.py` — 25 checks (stage 3): the checks the seeded-defect
sweep proved were missing, written in their own file so that the scripts the
sweep was run against stay exactly as they were audited. The Voigt row order of
`stress_matrix` against clq's raw tensor flattened by hand, and of
`Solution.stress` against `Solution.traction` on `e_x`/`e_y` facets (the
traction path never touches `VOIGT_PAIRS`, so it is not circular); the far field
at 1e3 and 1e4 element lengths against a Gauss quadrature of msd's frozen point
kernel, with `far_field="analytic"` as a tripwire (9e4x and 1e7x worse there);
`orientation = -1` at P0 and P1, that it reproduces the outward model's interior
field to 1e-15 and that the two free terms differ, so `sigma` is demonstrably
load-bearing; and the FAULT path above P0 — a callable slip sampled at
`clq.nodes` exactly, and a P1/P2 fault carrying a linear slip field against the
same field on a 16x refined P0 fault. Every check carries a tripwire.

### Adversarial verification of the gates (stage 3)

A gate that has never failed is an untested gate. `bench/seeded_defects.py`
seeds seventeen defects into **copies** of `ddbem` and `clq` (a sandbox per
defect, `DDBEM_CLQ_ROOT` pointed at the copy; nothing in the real tree is
touched) and runs the real gates against each. Fourteen were caught. **Three
were not, and those are the finding** — they are now gated by
`verify/verify_coverage.py`, which is checked to catch all three
(`python bench/seeded_defects.py X2_voigt_yz_xz_swapped X3_far_field_analytic
X5_orientation_ignored --gates verify_coverage.py` → 3/3).

| defect | caught by | first check that fires |
|---|---|---|
| `lam`/`mu` swapped in clq's slip → displacement pairing | kernels, solver, msd_parity | `nu=0.3: U vs msd assemble_T_matrix_batch` 1.0e0 |
| `lam`/`mu` swapped in clq's slip → stress pairing | kernels, solver | `nu=0.3: H vs msd analytical_stress_kernel` 8.2e-1 |
| `lam`/`mu` swapped in the eigenstress column tensor | kernels, layout, solver | `hypersingular row sum is 0` 4.2e-1 |
| eigenstress not subtracted at all | kernels, layout, solver | `elastic == total - C:eps* exactly` |
| eigenstress **2 % low** | kernels, layout, solver | `hypersingular row sum is 0` 8.4e-2 |
| P2 node order permuted in `mesh.barycentric_nodes` | layout, solver | `P2: element_nodes == clq.nodes` 6.2e-1 |
| P2 shape functions permuted in `mesh.shape_at` | solver only | `shape_at == clq.shape_functions` 1.0e0 |
| free term = `I` instead of `N_k(x_c)` | solver only | `P1 … is NOT the identity (free term is a matrix)` |
| free term forced to the analytic 1/2 | solver, msd_parity | `calibrated u_int row exact on rigid` 3.4e-2 |
| sign flip on the slip → displacement block | all four | `U vs msd assemble_T_matrix_batch` 2.0e0 |
| sign flip on clq's U, H and E together | kernels, solver, msd_parity | same |
| per-element `eps` replaced by `eps[0]` | layout only | `P1: graded eps array == per-element assembly (bitwise)` |
| rigid row-sum computed with the wrong sign | solver, msd_parity | `calibrated u_int row exact on rigid` 9.3e-1 |
| `COLLOCATION_SHRINK_BY_ORDER` 0.5 → 0.3 | solver, **marginally** | `P2: substituting I for N_k(x_c) changes the answer` (x0.955 against a 0.95 threshold) |
| **Voigt row order: `yz` and `xz` swapped** | **nothing** (now: coverage) | — |
| **`FAR_FIELD` default `hybrid` → `analytic`** | **nothing** (now: coverage) | — |
| **patch `orientation` ignored in the free term** | **nothing** (now: coverage) | — |

**The `lam`/`mu` swap is exactly invisible at `nu = 1/4`, and the data says so
without being asked.** Of the 59 checks that fail across the three gates with
the pairing swapped in, **zero** are labelled `nu=0.25`. Two independent
mechanisms catch it: the `NU_SWEEP` in the parity checks, and
`part_node_order`'s independent re-derivation of the pairing inside
`quad_reference` (which happens to run at `nu = 0.30`). A control run with
`NU_SWEEP = (0.25,)` still fails on the second; a control with *every* `nu`
literal in `verify_kernels.py` set to 0.25 makes that gate **pass with the bug
in**, leaving only `verify_solver.py` (whose exact-solution gates hard-code
`nu = 0.30`) to catch it. That is the documented claim, demonstrated.

#### The three gaps, and how much each one matters

1. **The Voigt row order of the public stress API is pinned by nothing.**
   `defaults.VOIGT_PAIRS` is documented as `(xx, yy, zz, yz, xz, xy)`, but
   `voigt_to_tensor` and `tensor_to_voigt` are built from the same tuple and so
   stay mutually inverse, `verify_kernels.py` converts both sides with ddbem's
   own converter, and `verify_solver.py` computes an interior stress error but
   **never gates it** (`_voigt()` there is a hand-written `(xx,yy,zz,yz,xz,xy)`
   that would have caught it). Swapping two rows changes what every caller of
   `Solution.stress` and `stress_matrix` reads out of columns 3 and 4, and every
   gate passes. Fix: gate `Solution.stress` against `_voigt(kelvin_stress(...))`
   in `verify_solver.py` — the numbers are already computed there.

2. **No gate exercises the far field.** Every observation set in `verify/` sits
   within ~4 element lengths, where clq's `hybrid` and `analytic` producers agree
   to 0.0 (measured). They do not agree further out: 8.7e-12 at 20 L, 6.5e-10 at
   100 L, 2.0e-4 at 1e4 L and **0.86 at 1e5 L**. So the `FAR_FIELD` default is
   load-bearing and unpinned. Low severity for this tree's geometry (a 1 km
   element read 50 km away is 50 L, where the difference is ~1e-10), real for
   anything that evaluates a small fault at continental range. **Now gated** in
   `verify_coverage.py` against a Gauss quadrature of msd's frozen point kernel
   at 1e3 and 1e4 L (independent of clq's closed form *and* of its far-field
   switch), with the `analytic` producer as an explicit tripwire.

3. **No gate ever builds a model with `orientation = -1`.** The only
   inward-normal test in `verify_solver.py` is that a *flipped mesh with the
   default `+1`* raises. Deleting `orientation` from the free term therefore
   changes nothing any gate sees. Checked by hand here: the same Kelvin
   Dirichlet problem posed on a flipped sphere with `orientation=-1` reproduces
   the outward-oriented answer to **3.2e-16**, so the code is right — it was the
   *gate coverage* that was missing. **Now gated** in `verify_coverage.py` at P0
   and P1, including that the density flips sign with the stored normal (modulo
   the `v1, v3, v2` node permutation `shapes.flip` induces) and that the two
   models' free terms genuinely differ, so `sigma` is demonstrably load-bearing.

Two further coverage gaps found by reading, verified by hand, not gated:

* **A fault patch at P1/P2 was never solved by any gate** (`gate_row_types_agree`
  uses `order=0` throughout), and neither was a callable `FAULT` value. **Now
  gated** in `verify_coverage.py`: `Patch.nodal_density` at P0/P1/P2 with a
  spatially varying callable matches sampling the callable at `clq.nodes`
  element by element to 0.0 exactly, and a P1 and a P2 fault carrying a linear
  slip field agree with the same field on a 16x refined P0 fault to 7.4e-4
  (and with each other to 1.3e-15).
* **`X6` above is the honest reading of the collocation-shrink default**: the
  single check that fires when 0.5 becomes 0.3 clears its threshold by 10 %
  (x0.955 against "> 0.05 change"), which is a coincidence, not a pin.
  `gate_clearance_rule` compares the measured 1-eps rule against a *hard-coded*
  0.5, not against `defaults.COLLOCATION_SHRINK_BY_ORDER`, so the shipped
  default is effectively unpinned.

## What is deliberately not here

* **Material interfaces.** `BCType.INTERFACE` is declared and **raises**. One DD
  density in one uniform medium cannot carry a material contrast: matching `u`
  and `t` across it needs a second density per region — the Somigliana single
  layer, whose density is the *physical* boundary traction (the same clq kernel
  as the dropped force element, used for a completely different unknown) — plus
  a two-region orientation graph like msd's `RegionModel`. The design leaves
  room (independent oriented patches, per-patch `eps` and `order`, per-block row
  emission, a per-row 3x3 free term) and implements none of it.
* **A continuous nodal layout in the solver.** `ddbem.continuous` exists and is
  gated algebraically (stage 1), but nothing in `model.py` uses it: condensing
  columns leaves `3 K N_tri` collocation rows against `3 n_global` unknowns, so
  a continuous DD collocation system is over-determined and needs either one
  collocation point per *global* node (whose normal is then ambiguous at a
  geometric edge) or a least-squares / Galerkin solve. That is the decision
  where higher-order DD actually pays (continuous P1 is 1.52 unknowns per
  triangle against P0's 3) and it is still open.
* **A free term for a fault that reaches the boundary.** A boundary collocation
  point within ~1 eps of a fault element sees the fault's blob average, not the
  one-sided value the boundary condition means. The model **warns**; it does not
  correct. This is the outcrop case in `medt_paper/topo_inclusion`.
* **A smoothing operator for non-zero prescribed traction.** With one density
  the only way to state `t(x_c) = t_bar` is through the hypersingular operator,
  and the study (§5) measures the cost: msd's two-field direct BIE, which
  applies the same data through the single layer `G`, converges faster on the
  Neumann problem (+1.45 against +0.86 on the sphere, +1.16 against +0.33 on the
  cube). It costs nothing when `t_bar = 0` — then `neumann_row="exterior"` is
  msd's own equation, gated entrywise — so this only bites on models that
  prescribe a non-zero traction.
* **A pin on the collocation-shrink default.** The seeded-defect sweep shows
  that changing `COLLOCATION_SHRINK_BY_ORDER` from 0.5 to 0.3 trips exactly one
  check, by 10 % of its threshold — a coincidence, not a pin. `gate_clearance_rule`
  compares the measured 1-eps rule against a hard-coded 0.5, not against the
  default. Nothing else here would notice the default moving, and no gate
  distinguishes 0.5 from 0.3 on physics; the sweep in `bench/sweep_collocation.py`
  is the only evidence for the value.
* **Speed.** See below.

## Measured assembly cost

`bench/bench_assembly.py`, tilted square fault, centroid collocation
(`N_f = N_tri`), `eps/h = 0.3`, single-threaded numpy (2026-09-18).

| N_tri | order | displacement | traction (H + E) | matrix | k pairs/s (disp / trac) |
|---|---|---|---|---|---|
| 512 | P0 | 11.2 s | 18.2 s | 1536 x 1536, 0.02 GiB | 23.5 / 14.4 |
| 512 | P1 | 16.9 s | 29.3 s | 1536 x 4608, 0.05 GiB | 15.5 / 8.9 |
| 512 | P2 | 26.0 s | 45.6 s | 1536 x 9216, 0.11 GiB | 10.1 / 5.8 |
| 2048 | P0 | 64.6 s | 121.9 s | 6144 x 6144, 0.28 GiB | 64.9 / 34.4 |
| 2048 | P1 | 99.8 s | 243.2 s | 6144 x 18432, 0.84 GiB | 42.0 / 17.2 |
| 2048 | P2 | 156.9 s | 405.7 s | 6144 x 36864, 1.69 GiB | 26.7 / 10.3 |

**This is pure Python + numpy and it is slow.** The same P0 slip → displacement
matrix takes `mbem.kernels.tri_kernels.t_matrix_direct` (numba) **0.012 s at
N_tri = 512 and 0.137 s at 2048** — 930x and 470x faster. ddbem is a readable,
gated reference: everything is vectorised over observation points with one
triangle per call, so the outer loop is a mechanical numba port. Nothing here
should be used at production scale until that port exists. The solver makes it
worse in one specific way: a P1/P2 model collocates at `K` points per element,
so the field-point count is `K N_tri`, and the matrix is `3 K N_tri` square.

## Rules inherited from the tree

* **`clq/`, `msd/`, `moss/`, `medt_paper/` and `fbem/` are read-only from here.**
  `clq/verify/verify_baseline_bitwise.py` pins `U`/`H`/`E` against 648 byte
  hashes; if it ever fails, something reached into clq — report it, do **not**
  regenerate the manifest. (It passes as of this writing.)
* Numeric constants live in `ddbem/defaults.py`, not inline.
* Everything is numpy-vectorised over observation points with one triangle per
  call, so a numba port of the outer loop stays mechanical.
* Every new code path gets a `verify/` script ending in one
  `PASS: <title> (<n> checks)` line; `verify/run_all.py` aggregates them.
