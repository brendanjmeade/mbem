# ddbem: findings

Closed on 2026-09-19 after its P1/P2 kernels and solver were ported into
`msd/mbem` (the trunk). The code is in git history at commit `aa80569`. The
convergence study's cached results were deleted with the code; every number
below is in this file. Slip sign throughout: ddbem/clq slip = u(+n) - u(-n),
the same as msd/mbem.

At closure the trunk's P1/P2 solver was compared with ddbem entrywise on the
80-triangle icosphere with an interior square fault (ddbem's exterior row
against msd's direct row, both jump conventions, nu = 0.25 and 0.30, P0/P1/P2
faults including a linear-slip callable, both orientations, and two patches of
different order in one system): the row and column layouts coincide (identity
permutation), A to 5.4e-15, b to 1.2e-13, the solved density to 3.3e-13
(condition numbers up to 2e3), the interior field to 1.3e-14. The trunk's
`verify_nodal_solve.py` and `verify_nodal_kernels.py` carry the gates from
here on.

## What it was

A direct / displacement-discontinuity (DD) collocation BEM: one DD density `q`
per boundary element in ONE uniform medium, one code path for P0 / P1 / P2
Lagrange nodal slip on flat triangles, Cortez-regularised
(`R = sqrt(r^2 + eps^2)`), on `clq`'s closed-form kernels. Why DD and not the
indirect single layer: the force-element BEM (`fbem/FINDINGS.md`) has a density
unknown with a genuine `rho^(-1/3)` edge singularity on a polyhedron, so its
free-traction rows stall at O(h^0.31) against the direct BIE's O(h^0.90), and
neither p-refinement nor grading rescues it; the DD unknown is a displacement,
which is bounded.

**The one-line answer of the convergence study.** The interior field converges
cleanly at every order on a sphere and a cube; the BOUNDARY trace over a cube
does not, and what governs it is how many mollification lengths the collocation
point is clear of its own element's edge. Over 60 cube solves per order,
`log(err)` on `log h` alone has R² = 0.00 / 0.02 / 0.02 at P0 / P1 / P2; on
`log(clearance/eps)` alone, 0.70 / 0.96 / 0.96. At `eps/h = 0.3` P1 and P2 are
*worse* than P0 at matched unknowns — `fbem/FINDINGS.md` §2, reproduced for DD.

## The formulation

A mollified kernel evaluated ON its own element returns the blob AVERAGE across
the discontinuity: with `sigma = +1` for outward normals,
`u_int/ext(x_c) = [D q](x_c) ∓ (sigma/2) q(x_c)`, and the traction has no jump.
For a nodal density `q(x_c) = sum_k N_k(x_c) q_k` the free term is therefore
the matrix of shape-function values at the collocation point, not a scalar 1/2.

| row | equation | used for | analytic row sum |
|---|---|---|---|
| `TRACTION` | `t(x_c) = t_bar` | free / prescribed traction (default) | 0 |
| `INTERIOR_DISPLACEMENT` | `u_int(x_c) = u_bar` | prescribed displacement | I/2 |
| `EXTERIOR_NULL` | `u_ext(x_c) = 0` | second kind; whole boundary traction-free | I/2 |

`EXTERIOR_NULL` is msd's equation rewritten in `q`; on a traction-free body the
two rows' interior stress fields agree to 9.8e-2 (20 triangles) and 3.1e-2 (80).
At P0 the solver matched msd's entrywise (`A` 2.2e-15, interior field 4.0e-15).

## The free term is a row-sum identity, and it generalises

`R` is a row's 3x3 action on the rigid translation `q = -sigma c`, whose answer
on every row is exact (`u_int = c`, `u_ext = 0`, `t = 0`): the free term is
`F = sigma_c (R - target)`, `target = I` for an interior-displacement row and 0
otherwise, `R[i,j] = -sum_{p,s,k} sigma_p A_raw[row_i, col(p,s,k,j)]` over the
collocation element's own nodes weighted by `N_k(x_c)`; `jump="half"` uses the
analytic `R`, `jump="calibrated"` the measured one. The partition of unity
carries it to P1/P2 unchanged; at P0 it is msd's `C_q = -sum_p sigma rowsum H`.

1. **`R` for a displacement row is exactly `phi(x_c) I`**, isotropic to 1e-15 at
   every order, sphere and box, nu = 0.25 / 0.30 / 0.45; `phi` is the mollified
   indicator at the collocation point, and the classical free term asserts
   `phi = 1/2`. Measured **0.288 … 0.466** — up to 42 % off (the low end is a
   box corner; the sphere gives 0.44–0.49). This is what the calibration fixes.
2. **The hypersingular row needs no calibration**: its raw row sum is machine
   zero (worst 7.1e-14, nu = 0.45, P2), because `sum_S D = -phi I` is exact
   pointwise, so a uniform `q` gives `u = -phi q`, an anelastic field whose total
   stress IS its eigenstress `C:eps*`, and the standing policy subtracts it
   exactly. Tripwire: without the subtraction the row sum is **4.4e15x larger**
   (7.4e0 against 1.7e-15).
3. **The free term must be the shape-function MATRIX, and a patch test cannot
   see it**: a constant density answers the same under `N_k(x_c)` or `I`. Where
   the density varies inside an element (Kelvin problem, 80-triangle sphere,
   `shrink = 0.5`) `I` is 1.12x worse at P1 and 0.91x at P2 — real, not
   one-sided; `N_k(x_c)` is the derived form.

**Rigid-motion constraint** (a traction-free body has a 6-dimensional null
space). Bordering with the 3 translations (msd's `deflate=True`): with
calibration an **exact** discrete null space, rows satisfied to 8e-16. With all
6 (the default): conditioning 97 vs 247, but the rotations are only near-null,
so the rows hold to **3.7e-4 (20-triangle sphere) and 6.4e-6 (80)** — not zero.

**Patch test**, `u = c` on an 80-triangle icosphere, nu = 0.25 (0.30 / 0.45
the same to 2x; half-jump range over all nine cases 1.9e-2 … 1.8e-1):

| order | `jump="calibrated"` | `jump="half"` |
|---|---|---|
| P0 | 3.7e-15 | 1.9e-2 |
| P1 | 1.2e-14 | ~7e-2 |
| P2 | 8.9e-14 | 1.8e-1 |

**Exact solution, a Kelvin point force outside the body.** 80-triangle
icosphere, `eps/h = 0.3`, calibrated, interior error, 20-triangle value → 80,
rate in h; a mixed model (36 Dirichlet + 44 traction-free triangles) reaches
1.5e-2 at nu = 0.30 and 1.8e-2 at nu = 0.45:

| rows | P0 | P1 | P2 |
|---|---|---|---|
| Dirichlet | 4.97e-2 → 4.24e-3 (+4.16) | 4.36e-2 → 2.80e-3 (+4.64) | 4.20e-2 → 2.82e-3 (+4.57) |
| Neumann | 5.70e-3 → 3.22e-3 (+0.97) | 3.74e-3 → 1.30e-3 (+1.79) | 3.22e-3 → 8.62e-4 (+2.23) |

## Where to collocate a P1/P2 element

At `shrink = 0` every P1/P2 collocation point (vertex or midpoint) sits ON the
element boundary; it was pulled in by `lam_c = (1-t) lam_node + t/3` with the
basis unmoved. 80-triangle icosphere, exterior Kelvin force, interior error /
condition number:

| bc | P | eps/h | t = 0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|---|---|---|
| Dirichlet | 1 | 0.30 | 5.95e-3 / 6.2 | 4.60e-3 / 7.0 | 3.49e-3 / 8.2 | 2.72e-3 / 9.4 | 2.80e-3 / 12 | 2.84e-3 / 19 |
| Dirichlet | 2 | 0.30 | 2.818e-3 / 17 | 2.817e-3 / 23 | 2.817e-3 / 25 | 2.818e-3 / 25 | 2.819e-3 / 66 | 2.818e-3 / 204 |
| Dirichlet | 1 | 0.15 | 3.71e-3 / 8.2 | 3.13e-3 / 9.3 | 2.75e-3 / 11 | 2.38e-3 / 12 | 1.71e-3 / 14 | 1.85e-3 / 24 |
| Dirichlet | 2 | 0.15 | 1.745e-3 / 23 | 1.724e-3 / 25 | 1.704e-3 / 31 | 1.685e-3 / 40 | 1.650e-3 / 78 | 1.627e-3 / 262 |
| Neumann | 1 | 0.30 | **2.46e-1 / 5.1e19** | 7.42e-3 / 2.2e3 | 3.86e-3 / 343 | 2.57e-3 / 343 | 1.30e-3 / 343 | 5.55e-4 / 343 |
| Neumann | 2 | 0.30 | **2.09e-2 / 3.5e19** | 2.11e-3 / 5.7e3 | 1.50e-3 / 702 | 1.17e-3 / 702 | 8.62e-4 / 702 | 7.62e-4 / 1.9e3 |

* **`shrink = 0` is numerically singular for the hypersingular rows** (cond
  3.5e19 … 5.1e19, error 25–190x worse than any positive `t`: adjacent elements
  collocate at one shared point) and only mediocre for displacement rows
  (cond 6–23), so a Dirichlet-only code never notices.
* Traction-row error falls monotonically; cond is flat from t = 0.2 to 0.6 and
  inflates at 0.7 for P2 (702 → 1.9e3 → 4.6e3 at 0.8). P2 Dirichlet rows are
  flat (2.817e-3 … 2.819e-3, P1's best): the eps floor.
* **The right `t` buys one eps of clearance, and no more.** At eps/h = 0.15 the
  P1 hypersingular error has a sharp minimum at `t = 0.6` — 9.86e-5, 6.7x
  better than `t = 0.5` (6.60e-4), 4.5x / 8.1x better than 0.7 / 0.8 — exactly
  where the node clearance first reaches 1.03 eps; at eps/h = 0.3 no `t <= 0.8`
  reaches 1 eps (best 0.69). The rule gives `t` = 0.58 at eps/h = 0.15 and 0.31
  at 0.08 (capped at 0.8); its 0.58 gave 1.59e-4 against 6.60e-4, 4.2x better.
* **Chosen default: shrink 0 at P0, 0.5 at P1 and P2** — fbem's independent
  sweep chose 0.5 too, and it sits in the flat-conditioning plateau at both
  orders. It is a conditioning-guarded number, not an error minimum; the
  clearance rule was deliberately not automatic, since that would hide the
  eps/h budget it depends on.

## The convergence study

Exact solution: a Kelvin point force OUTSIDE the body (the only exact solution
a boundary-only representation can reproduce), as a Dirichlet and as a Neumann
(hypersingular-row) problem, on a **sphere** (icosphere, radius 1, 20 / 80 /
320 / 1280 triangles; only O(h) creases) and a **cube** (side 2, 48 / 108 /
192 / 432 triangles; twelve permanent 90° edges). ~320 solves; errors relative
to `max|u_exact|`: `u_int` on a fixed 7-point cloud at |x| <= 0.3; `u_surf`,
the boundary trace at the collocation points (Neumann rows only); `u_surf_cen`,
the trace at the element centroid, the same point at every order.

### 1. The interior field converges; the boundary trace over a cube does not

Rate in h of `u_int`, `eps/h = 0.3`, calibrated, full ladders († non-monotone:
8.6e-3 → 1.5e-2 → 9.0e-3 → 4.7e-3, so a description, not an order):

| geom | rows | msd P0 | ddbem P0 | ddbem P1 | ddbem P2 |
|---|---|---|---|---|---|
| sphere | Dirichlet | +0.37 † | **+2.15** | +2.57 | +2.54 |
| sphere | Neumann | **+1.45** | +0.86 | +1.20 | +1.23 |
| cube | Dirichlet | +0.51 | **+1.78** | +1.50 | +1.46 |
| cube | Neumann | **+1.16** | +0.33 | +0.73 | +0.75 |

Rate in h of the boundary trace `u_surf_cen` (3-level ladders):

| geom | eps/h | P0 | P1 | P2 |
|---|---|---|---|---|
| sphere | 0.30 | +0.96 | +1.03 | +0.78 |
| cube | 0.30 | +0.74 | **+0.19** | **+0.08** |
| cube | 0.15 | +0.19 | **−0.31** | **−0.12** |

The cube stalls the surface, the sphere does not, and higher order makes it
worse. The squared-error share of collocation points within 0.3 of a cube edge,
over their share of points, *grows* from 48 to 192 triangles — 1.39 → 1.59 at
P0, 1.65 → 1.98 at P1, 1.55 → 1.83 at P2: an unresolved edge feature, hidden in
the interior because the representation smooths it.

### 2. What sets the error is `eps`, not `h` — measured, not asserted

Every cached case of one (geometry, row type) pooled; `log err` on `log h`, on
`log(clearance/eps)` (in-plane distance from the collocation point to its own
element's boundary, in eps) and on both. Boundary trace, Neumann rows, 60 cube
/ 50 sphere cases per order (R², exponent):

| geom | order | `err ~ h` | `err ~ clearance/eps` | both |
|---|---|---|---|---|
| cube | P0 | 0.00 (−0.03) | **0.70** (−1.28) | 0.77 |
| cube | P1 | 0.02 (−0.36) | **0.96** (−2.10) | 0.99 |
| cube | P2 | 0.02 (−0.34) | **0.96** (−2.12) | 1.00 |
| sphere | P0 | 0.00 (+0.09) | 0.63 (−1.36) | 0.87 (h: +0.89) |
| sphere | P1 | 0.05 (+0.43) | 0.57 (−1.25) | 0.95 (h: +1.27) |
| sphere | P2 | 0.00 (+0.15) | 0.62 (−1.62) | 0.83 (h: +1.15) |

On the cube `h` explains 0–2 % of the variance and the budget 70–96 %, with
exponent −2.1 in `clearance/eps` at P1/P2; adding `h` moves its exponent only
to +0.53. On the sphere the joint fit keeps a real `h` term (+0.89 … +1.27):
with the eps budget fixed, refining helps — fbem's sphere/polyhedron
separation, now for DD. Directly: P1 and P2 at eps/h = 0.15 and P0 at 0.3 share
a clearance of 0.69 eps and land together (3.74e-2 / 3.52e-2 / 5.19e-2, 192
triangles). The order of the basis is not what is being measured.

### 3. Does higher-order DD pay? Only on one row type, and only with eps room

At matched unknown count (error interpolated onto the P0 ladder's dof range;
> 1 means higher order is worse), interior field, calibrated:

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

On displacement rows higher order essentially never pays (best 4 %, cube at
eps = 0.25 absolute; 1.7–7.9x worse at eps/h = 0.3, up to 11.9x at 0.5). On
hypersingular rows it pays on the sphere (up to 4.5x) and on the cube only once
eps/h <= 0.15 or eps is fixed in absolute terms — fbem's inversion, with the
sign it predicted. The unknown axis is as unfavourable as it can be: only the
discontinuous layout was solved (18 unknowns per P2 triangle against P0's 3);
continuous P1 would be ~1.5, a 6x shift of this table.

### 4. The free-term calibration changes the RATE, not just the constant

Dirichlet rows, eps/h = 0.3, interior field, rate in h:

| geom | P0 | P1 | P2 |
|---|---|---|---|
| sphere, calibrated | **+2.15** | **+2.57** | **+2.54** |
| sphere, half | +0.72 | +0.70 | +0.57 |
| cube, calibrated | **+1.78** | **+1.50** | **+1.46** |
| cube, half | +0.96 | +1.07 | +1.01 |

On the finest mesh the calibrated answer is 6.1x better at P0 (1280 triangles:
8.02e-4 against 4.86e-3) and 11.5x at P1 (320: 1.92e-3 against 2.21e-2); on
the coarsest the analytic half is sometimes better (sphere P0, 2.09e-2 against
5.66e-2). A rate statement, not a uniform one.

### 5. The msd anchor, and one place ddbem is beaten

msd's frozen P0 direct BIE on the same meshes, eps and data is a different
formulation (`u` and `t` as unknowns): a reference rate, not parity. ddbem wins
Dirichlet outright — sphere, 1280 triangles, 8.0e-4 against 4.7e-3 (+2.15
against +0.37); cube, 432, 2.7e-3 against 5.4e-3 — where msd's unknown is the
traction through a first-kind single layer. msd wins Neumann — sphere 1.1e-3
against P0's 2.7e-3 (+1.45 against +0.86); cube 2.2e-3 against 2.8e-3 (+1.16
against +0.33); P2 closes the gap in h, not per unknown. The reason is
structural: msd applies Neumann data through the single layer `G` (smoothing)
and keeps the second-kind `1/2 I + sigma H` on the left; one density must state
`t(x_c) = t_bar` through the hypersingular operator, which amplifies error. At
`t_bar = 0` the two are the same equation (the exterior row, gated entrywise).

### 6. A warning about single-cloud ratios

An earlier single-geometry pass reported "P0 → P2 buys 1.50x" on Dirichlet rows
at eps/h = 0.3, from a 3-point cloud at one mesh level. Re-measured (P0/P2):

| cloud | 80 triangles | 320 triangles |
|---|---|---|
| the original 3 points | **1.50x** | 0.66x |
| this study's 7 points | 0.88x | 0.69x |
| an independent 26-point shell at \|x\| = 0.3 | 0.90x | 0.65x |

The claim reverses on adding points or refining once; at 320 triangles all
three clouds agree P0 is ~1.5x *better* than P2 at matched h. The Neumann-row
claim survives (P0/P2 = 3.73 / 4.85 / 2.95 at 80 triangles, 2.75 / 3.13 / 2.35
at 320): direction robust, factor good to ~2x. A single-cloud, two-level ratio
is a direction, not a number.

### 7. Honest limits of the study

* Pure Python: the largest case was 7776 unknowns (cube P2, 432 triangles,
  ~2 min); ladders this short do not settle the P2 rates.
* The interior cloud sits 0.64 h from the surface on the coarsest cube, at the
  edge of the mesh-limited zone; the rungs that matter for the rates are not.
* Interior stress was recorded, never used for a rate: non-monotone in h.
* No fault patch anywhere: a pure boundary-value study. The fault path was
  gated at P0, and at P1/P2 only against a 16x refined P0 fault (7.4e-4).
* Every rate is a 3–4 point least-squares slope; the pooled regressions of §2
  (20–60 cases) are the more trustworthy statements.

## Seeded defects: what the gates could and could not see

Seventeen defects seeded into copies of `ddbem` and `clq`, the real gates
(kernels, layout, solver, msd parity) run against each: fourteen caught, three
not — those three are the finding.

| defect | caught by | first check that fires |
|---|---|---|
| `lam`/`mu` swapped in clq's slip → displacement pairing | kernels, solver, msd parity | `U` vs msd at nu = 0.3, 1.0e0 |
| `lam`/`mu` swapped in clq's slip → stress pairing | kernels, solver | `H` vs msd at nu = 0.3, 8.2e-1 |
| `lam`/`mu` swapped in the eigenstress column tensor | kernels, layout, solver | hypersingular row sum 4.2e-1 |
| eigenstress not subtracted at all | kernels, layout, solver | `elastic == total - C:eps*` |
| eigenstress 2 % low | kernels, layout, solver | hypersingular row sum 8.4e-2 |
| P2 node order permuted | layout, solver | element nodes vs clq, 6.2e-1 |
| P2 shape functions permuted | solver only | shape functions vs clq, 1.0e0 |
| free term = `I` instead of `N_k(x_c)` | solver only | P1 free term is not the identity |
| free term forced to the analytic 1/2 | solver, msd parity | calibrated row exact on rigid, 3.4e-2 |
| sign flip on the slip → displacement block | all four | `U` vs msd, 2.0e0 |
| sign flip on clq's U, H and E together | kernels, solver, msd parity | same |
| per-element `eps` replaced by `eps[0]` | layout only | graded eps == per-element (bitwise) |
| rigid row-sum computed with the wrong sign | solver, msd parity | calibrated row exact on rigid, 9.3e-1 |
| collocation shrink 0.5 → 0.3 | solver, **marginally** | `I` vs `N_k(x_c)` changes the answer (x0.955 against 0.95) |
| **Voigt rows `yz` and `xz` swapped** | **nothing** | — |
| **far-field default `hybrid` → `analytic`** | **nothing** | — |
| **patch `orientation` ignored in the free term** | **nothing** | — |

The `lam`/`mu` swap is exactly invisible at nu = 1/4: of the 59 checks that
fail with it in, zero are labelled nu = 0.25. The three gaps, each gated after:

1. **The Voigt row order of the public stress API was pinned by nothing**: both
   converters came from one tuple and stayed mutually inverse, the kernel gate
   used them on both sides, and the solver gate computed an interior stress
   error but never gated it. Cure: stress against a hand-written Voigt of the
   exact solution, and traction against stress on `e_x`/`e_y` facets.
2. **No gate exercised the far field.** Every observation sat within ~4 element
   lengths, where clq's `hybrid` and `analytic` producers agree to 0.0; further
   out they differ by 8.7e-12 at 20 L, 6.5e-10 at 100 L, 2.0e-4 at 1e4 L and
   0.86 at 1e5 L. Cure: gate at 1e3 and 1e4 L against a Gauss quadrature of
   msd's frozen point kernel, the analytic producer as tripwire (9e4x, 1e7x).
3. **No gate ever built a model with `orientation = -1`.** The code was right
   (a flipped sphere reproduces the outward answer to 3.2e-16); the coverage
   was missing. Cure: gate at P0 and P1, with the two free terms shown to
   differ, so `sigma` is load-bearing.

The marginal row is the honest reading of the shrink default: 0.5 → 0.3 trips
one check by 10 % of its threshold — a coincidence, not a pin.

## What is deliberately not here

* **Material interfaces.** `BCType.INTERFACE` was declared and raised. One DD
  density in one medium cannot carry a contrast: matching `u` and `t` needs a
  second density per region — the Somigliana single layer, whose density is
  the physical traction (the force element's kernel, a different unknown) —
  plus a two-region orientation graph like msd's `RegionModel`.
* **A continuous nodal layout in the solver.** Condensing columns leaves
  `3 K N_tri` collocation rows against `3 n_global` unknowns: over-determined,
  needing one collocation point per *global* node (ambiguous normal at an edge)
  or a least-squares / Galerkin solve. That is where higher-order DD actually
  pays (continuous P1 is 1.52 unknowns per triangle against P0's 3); open.
* **A free term for a fault that reaches the boundary.** A boundary collocation
  point within ~1 eps of a fault element sees the fault's blob average, not the
  one-sided value the BC means; the model warned, it did not correct. This is
  the outcrop case in `medt_paper/topo_inclusion`.
* **A smoothing operator for non-zero prescribed traction** — §5 measures the
  cost; free when `t_bar = 0`. **A pin on the shrink default** (above).

## Measured assembly cost

Tilted square fault, centroid collocation (`N_f = N_tri`), eps/h = 0.3,
single-threaded numpy:

| N_tri | order | displacement | traction (H + E) | matrix | k pairs/s (disp / trac) |
|---|---|---|---|---|---|
| 512 | P0 | 11.2 s | 18.2 s | 1536 x 1536, 0.02 GiB | 23.5 / 14.4 |
| 512 | P1 | 16.9 s | 29.3 s | 1536 x 4608, 0.05 GiB | 15.5 / 8.9 |
| 512 | P2 | 26.0 s | 45.6 s | 1536 x 9216, 0.11 GiB | 10.1 / 5.8 |
| 2048 | P0 | 64.6 s | 121.9 s | 6144 x 6144, 0.28 GiB | 64.9 / 34.4 |
| 2048 | P1 | 99.8 s | 243.2 s | 6144 x 18432, 0.84 GiB | 42.0 / 17.2 |
| 2048 | P2 | 156.9 s | 405.7 s | 6144 x 36864, 1.69 GiB | 26.7 / 10.3 |

Pure Python + numpy: the same P0 slip → displacement matrix takes
`mbem.kernels.tri_kernels.t_matrix_direct` (numba) 0.012 s at N_tri = 512 and
0.137 s at 2048 — 930x and 470x faster. A P1/P2 model also collocates at `K`
points per element, so its matrix is `3 K N_tri` square. That is why the port
into `msd/mbem` is the trunk and this file is the record.
