# `fbem` — force-element (equivalent body force) BEM

A boundary element method for piecewise-uniform elastic solids in which
material contrasts are carried by **equivalent body forces in a single uniform
reference medium** rather than by matching displacement and traction across
sub-domain boundaries.

It is the assembly layer on top of `../clq`'s mollified Kelvin single-layer
triangle element, and its target is to re-solve `../medt_paper/topo_inclusion`
— soft cylindrical inclusion + topography + surface-breaking strike-slip fault
— and compare against that model's cached matching-BC solution.

`clq/`, `msd/`, `moss/` and `medt_paper/` are read-only inputs to this
sub-project. The only file outside `fbem/` this work touches is the umbrella
`../CLAUDE.md`, which gained a map row for it.

---

## The formulation

### Representation

One unknown field: a force density `q` (force per unit **area**) on every
boundary triangle, collocated at centroids, living in a **uniform** reference
medium — the host, `mu0`, `lam0`. The displacement anywhere is

```
u(x) = u_F(x) + ∫_S G^eps(x, y) q(y) dS(y)
```

with `G^eps` the mollified Kelvin tensor and `u_F` the fault's displacement
field **in that same uniform host**. There is no `(u, t)` pair and no
sub-domain bookkeeping: 3 unknowns per triangle, not 6.

### Why a uniform medium is legitimate here

The representation's stress, `sigma^rep = C0 : grad u`, is the **host** stress
everywhere, including inside the inclusion. The *true* stress inside the
inclusion is `C1 : grad u`. The two are proportional,

```
C1 = a C0    with   a = mu1/mu0,
```

**iff the two materials share a Poisson ratio** (`lam0 mu1 - lam1 mu0 = 0`).
In `topo_inclusion` they do: host `(mu, lam) = (30, 30)`, inclusion
`(3, 3)`, so `nu = 1/4` on both sides and `a = 0.1` exactly.

That equality is what makes the interior equilibrium automatic:
`div(C1 : grad u) = a div(C0 : grad u) = 0`. It is also what kills the
*volume* part of the polarization `tau = chi_V (C1 - C0) : (eps - eps*)`,
whose divergence is the general equivalent body force; with equal Poisson
ratios the volume term vanishes identically and the body force collapses onto
the interface, where `grad mu` is a delta. **With unequal Poisson ratios a
genuine volume source appears and this code is not applicable as written.**
The driver has no guard against that today — it is stated here and asserted
only indirectly, via `nu` being computed from the host alone.

### The three row types

`n` is the field normal, `t_on = B q + t_F` the **on-surface** traction of the
representation (the single layer's average value there), `B` the
adjoint-double-layer block, and the layer's jump is

```
t(+n) - t(-n) = -q            (clq's convention: div sigma + f phi_eps = 0)
```

so `t(-n) = t_on + q/2` and `t(+n) = t_on - q/2`. The free term is exactly
`1/2` on a flat element — L0 measures 0.4999079, with the one-sided values
0.4996883 and 0.5001276, so the *orientation* is pinned too, not just the
magnitude.

| row | surface | equation |
|---|---|---|
| **R1** | free traction, solid on the `-n` side | `(1/2) q + B q = -t_F` |
| **R2** | interface, `n` out of the inclusion | `((1+a)/2) q - (1-a) B q = (1-a) t_F` |
| **R3** | prescribed displacement `u = 0` | `[G - (eps/4) M] q = -u_F` |

**R2 is just continuity of the true traction** across the interface. The host
is on the `+n` side and sees `t(+n)`; the inclusion is on the `-n` side and
sees `a t(-n)`. Setting them equal,

```
t_on - q/2 = a (t_on + q/2)   ⟹   ((1+a)/2) q - (1-a) t_on = 0.
```

It degenerates correctly at both ends: `a = 1` gives `q ≡ 0` (no interface at
all, and the assembled rows are *bitwise* the identity — gate L1); `a = 0`
gives `t(+n) = 0`, a traction-free cavity.

**Displacement needs no equation.** The single layer is continuous, so
continuity of `u` across the interface holds identically by construction.
That is the entire economy of the method, and the reason the unknown count
drops from 31 194 to 22 545 on this model (−27.7 %).

**The inclusion's outcrop is not a special case.** Where the soft inclusion
reaches the free surface (`inclusion_top`), the condition is
`a sigma^rep . n = 0`, and since `a ≠ 0` the contrast cancels: it is exactly
R1. No extra row type, no `a` anywhere.

### The `(eps/4)` surface bias

A mollified single layer evaluated **on** its own element does not return the
sharp-limit value. The Cortez kernel is `G * phi_eps`, the sharp field has a
`(g/2)|z|` kink through the layer, and the blob's 1-D marginal has
`<|z|> = eps/2`, giving

```
u_on,eps = u_sharp + (eps/4) g,    g = -q_t/mu - q_n n/(lam + 2mu)
```

(L0 measures the coefficient at 0.250048 for normal `q`, 0.250008 for
tangential and 0.250012 for oblique — pinning both denominators
independently). `assembly.eps_bias_operator` returns
the per-element `3x3` `M` with `g = M q`. It is subtracted from the diagonal
block of the Dirichlet rows, and the driver reports the surface field **both
with and without** the correction rather than picking one — the cached
matching-BC solution is itself a mollified field, so which one is the right
comparand is an empirical question, not a definitional one.

### Sign conventions (the live hazard)

1. `clq`, `moss` and `msd` use `div sigma + f = 0`, so a closed surface around
   an element carries `∮ sigma.n dS = -∫ f dS`. The `../body_forces_bem`
   drafts write `div sigma = f` — **the opposite**. Every density here is
   minus theirs.
2. `mbem` assembles a `FAULT` patch with scale `-1` on the right-hand side, so
   the `value` stored on that patch is `-Delta_u` in `clq`'s convention
   (`Delta u = u(+nhat) - u(-nhat)`). `topo_inclusion.py` negates it and
   **gates** the choice: the full-space fault field at the free surface is
   regressed against the cached `(flat, hom)` state and must come back with
   slope near `+2` (free-surface doubling), not `-2`.
   Measured: slope `+2.081`, correlation `+0.983`.
3. Mesh winding is **never trusted**. `model.Patch` takes an `orient`
   specification (`("up",)`, `("down",)`, `("radial", (x, y))`) and flips each
   normal to satisfy it geometrically. Outer surfaces point out of the solid;
   interfaces point out of the **inclusion**. The choice is then verified by
   the divergence theorem on both closed surfaces — `sum(area * n) = 0` and
   `sum(area * n.x)/3 = ` the enclosed volume, with sign.

---

## Layout

| file | what |
|---|---|
| `assembly.py` | numba dense assemblers (`traction_matrix` = `B`, `displacement_matrix` = `G`, `stress_matrix`, `apply_displacement`, `eps_bias_operator`) built on `msd`'s frozen pair kernels. Fills `[source, field]` C-contiguous and returns `.T`, so the matrix is **Fortran-ordered** and `lu_factor(overwrite_a=True)` factors in place. |
| `assembly_ho.py` | the same influence matrices for a **discontinuous P0/P1/P2 nodal** force density, built on `clq.influence` rather than `msd`'s numba constant-density kernels. Element `e` owns unknowns `3*K*e + 3*k + c` with `K = 1, 3, 6`, so there is no connectivity; `order=0` reproduces `assembly.py` entrywise. Pure numpy and ~50× slower — and because P1/P2 triple/sextuple the unknowns on the same mesh, error must be read against UNKNOWN COUNT, not against `h`. |
| `ho_convergence.py` | the P0/P1/P2 **rows** (R1 + R3 for a nodal density, collocated at shrunk nodes) and the convergence **study** built on them: L2's box, exact solution, ε convention, metric and edge exclusion, swept over order and mesh. A study, not a gate. Answer: higher order does **not** repair the rate, and at matched unknown count it is 1.3–1.5× *worse* than P0. |
| `model.py` | `Patch` (orientation), `ForceElementModel` (rows, assembly, solve, evaluation), `fault_source`. |
| `geometry.py` | rebuilds and caches the `topo_inclusion` meshes from the original builder. |
| `topo_inclusion.py` | the trial: two solves (`a = 0.1` and `a = 1`) sharing one traction block, compared against the cached matching-BC fields. |
| `eps_ladder.py` | the trial at ε = 1.5, 3, 6. |
| `regen_reference.py` | re-solves the direct BIE (read-only import of `medt_paper`) at chosen ε, so the comparison is matched rather than against a cache pinned at one ε. |
| `compare_ladder.py` | the matched-ε table; reports `amp = ‖FE‖/‖BIE‖` next to the regression slope, because the slope's direction is easy to misread. |
| `eps_split.py` | ε on the interface and on the outer boundary varied independently (`eps_arr` is per element). |
| `verify/` | PASS/FAIL gates; `run_all.py` runs them all, or a subset by substring. |

Run everything with `/Users/meade/micromamba/bin/python` from this directory.
There is no installer.

```bash
PY=/Users/meade/micromamba/bin/python
$PY verify/run_all.py            # every gate, one PASS/FAIL line each
$PY verify/run_all.py l0 l1      # a subset
```

The gates, in order of what they pin:

| gate | pins |
|---|---|
| `verify_l0_operators.py` | the three conventions the rows rest on — parity with `clq`, the traction jump and its free term, displacement continuity, the `(ε/4)` bias, self blocks, the closed-surface column identity |
| `verify_l1_zero_contrast.py` | α = 1 is exactly inert: R2 rows bitwise identity, densities bitwise zero, and the solve reduces to the homogeneous box |
| `verify_l2_head_to_head.py` | accuracy against `msd`'s direct BIE on the same box and the same exact solution |
| `verify_l3_eshelby.py` | R2's `(1+α)/2` and `(1−α)` structure against Eshelby's closed form, both load cases, soft **and** stiff |
| `verify_ho_assembly.py` | `assembly_ho.py`: the order-0 reduction to `assembly.py`, partition of unity for P1/P2, an independent Gauss rule and kernel transcription entrywise per node, `clq`'s reciprocity identity read off the assembled matrix, and the assembly timings (`--quick` drops the N_tri = 2048 row, which is most of its ~6 min) |

## Status

`verify/run_all.py`: **4 of 5 gates pass.**

L0 (operator conventions and self terms), L1 (zero contrast is exactly inert),
L3 (Eshelby, soft and stiff) and `ho_assembly` (the P0/P1/P2 assembler) pass.
**L2 — the head-to-head against `msd`'s
direct BIE — fails**, and that failure is the main result: the force element
converges at O(h^0.32) on free-traction rows over a polyhedron against the
direct BIE's O(h^0.90), so its error ratio grows under refinement rather than
staying bounded. No defect was found in this code; the deficit is a property
of the indirect single-layer formulation on Neumann boundaries. With Dirichlet
rows alone the force element is 4–8× *better* than the direct BIE.

Read `RESULTS.md` before building on this — in particular §"L2 — head-to-head
against the direct BIE" for what the method costs, and §"A model I proposed
and the data rejected" for a mechanism that looked right and was not.
