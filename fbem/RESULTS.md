# Results — force-element BEM on `topo_inclusion`

All numbers measured on this machine, 2026-09-17, with
`/Users/meade/micromamba/bin/python`. `clq/`, `msd/`, `moss/` and
`medt_paper/` were read but not modified; the umbrella `../CLAUDE.md` gained a
map row for this sub-project.

## Summary

The equivalent-body-force formulation **works and is cheaper, but it is not a
drop-in accuracy replacement.** It solves the full `topo_inclusion` model,
reproducing the matching-BC solution's spatial pattern at correlation
**+0.998** with **27.7 % fewer unknowns** and **20× better conditioning** —
and it under-predicts the inclusion amplitude by **12.4 %** at the production
ε = 3 km (→ 7.7 % with ε chosen per surface, which is free).

No defect was found anywhere in the rows: every convention they rest on is
measured correct to 1e-4 or better (L0), the α = 1 limit is bitwise inert
(L1), 19 seeded row defects and 8 structural mutations of R2 are all caught
(adversarial L1, L3), and the interface row reproduces Eshelby's closed form
at O(h) for soft *and* stiff inclusions (L3).

**The limitation is structural, and L2 found it.** Head-to-head against the
direct BIE on the same box and the same exact solution, the force element's
error ratio is not a bounded constant but a *trend*: it converges at O(h^0.32)
against the direct BIE's O(h^0.90), so the ratio grows like h^−0.58 and
crosses 1.5× between 2048 and 3200 triangles. The whole deficit lives in the
**free-traction rows R1** — with Dirichlet rows alone the force element is
**4–8× better** than the direct BIE. For a free-surface-dominated earthquake
problem (4558 of 7515 elements here are free-traction) that is the governing
constraint.

| | matching-BC (direct BIE) | force element |
|---|---|---|
| unknowns | 31 194 (`u`,`t` pairs) | **22 545** (−27.7 %) |
| condition number | 4.4–5.8 × 10⁵ | **2.8 × 10⁴** |
| assemble + 2 solves | ~140 s | **~25 s** (one shared traction block) |
| interface equations | displacement **and** traction matching | traction only (displacement automatic) |
| inclusion outcrop | special case | falls out of R1 (α cancels) |
| accuracy, Dirichlet rows | baseline | **4–8× better**, O(h^1.7) |
| accuracy, Neumann rows on a smooth surface | O(h^1.31), superconvergent | O(h^0.90), textbook O(h) |
| accuracy, Neumann rows on a polyhedron | O(h^0.90) | **O(h^0.32)** — the problem |

**Verdict:** worth pursuing, but the R1 rows need work before this competes on
accuracy. The direct BIE's edge over them is bought by its rigid-body-calibrated
jump; finding the indirect analogue of that calibration is the pivotal open
problem, and the naive candidate is ruled out below.

## What was verified

| check | result | gate |
|---|---|---|
| assembler vs `clq` (traction / displacement / stress), block-relative | 6.1e-13 / 6.1e-14 / 4.4e-13 | L0 |
| traction jump `t(+n) − t(−n) = −q` | `\|J₀ + q\|/\|q\|` = 2.0e-4; amplitude `c` = 0.99999 | L0 |
| **free term on a flat element** | **0.4999079** (`\|c − ½\|` = 9.2e-5) | L0 |
| on-element value is the average (no extra self term) | 7.4e-10 | L0 |
| displacement continuity across the layer | 4.4e-6 of `\|u_on\|` | L0 |
| `(ε/4)` bias coefficient, normal / tangential / oblique `q` | 0.250048 / 0.250008 / 0.250012 | L0 |
| self block, flat | equilateral 1.4e-17 (machine zero), scalene 5.7e-3, sliver 1.6e-2 | L0 |
| self block, curved (icosphere / cylinder) | diag/rowsum 2.8e-3 / 1.1e-2 | L0 |
| column identity, source strictly inside | 2.9e-5 → 4.6e-7 → 6.0e-9 at 80/320/1280 facets | L0 |
| normal orientation, both closed surfaces | closure 5.3e-16 / 9.6e-16; volumes exact / 1.1e-3 (polygonal circle); **0 winding flips** on all six patches | — |
| fault sign (`slip = -0.01 ŝ`) vs cached `(flat, hom)` | slope **+2.081**, corr **+0.983** (free-surface doubling) | — |
| zero contrast α = 1: interface rows | `max\|A[R2] − I\| = 0.0` **bitwise** | L1 |
| zero contrast α = 1: interface densities | `max\|q\| = 0.0` **bitwise**, at production scale (2883 interface triangles) | L1 |
| zero contrast is not a decoupling bug | at α = 0.1 the same interface moves the top surface by 21 % | L1 |
| α = 1 reproduces the homogeneous box (not the fault-only field) | bitwise identical outer block and RHS | L1 |
| **Eshelby sphere**: interior stress vs closed form, worst over hyd + shear and α = 0.1/0.5/2.0 | **2.49e-2** (320 tri) → **1.33e-2** (1280), clean O(h) | L3 |
| Eshelby: direction of the concentration | soft α<1 → 0.203/0.179 (below remote); stiff α=2 → 1.292/1.342 (above) | L3 |
| Eshelby: solved density vs exact polarization `(1−α) C₀:ε_in · n` | O(h), ratios 1.8 / 1.7 | L3 |
| Eshelby: interface-only conditioning | 2.03–2.25, **mesh-independent** | L3 |
| R2 rows as assembled vs the documented formula | 0.0 **exact** | L3 |
| **adversarial**: 19 seeded row/bookkeeping defects (incl. α→1/α, (1+a)/2→(1−a)/2, q sign flip) | **19/19 caught, 0 false passes**, on 5 unseen meshes | refute L1 |
| **adversarial**: 8 structural mutations of R2 | **8/8 caught** (2.1×–105× over the gate) | L3 |
| the cached reference is current | re-solved with today's kernels: **5.1e-15** relative change | — |

That last row matters: the cache predates today's λ/μ traction-pairing port,
so it was suspect. The port is documented as invisible at ν = ¼ (λ = μ on both
materials here); re-solving confirms that numerically rather than on the
docstring's word.

## L2 — head-to-head against the direct BIE: **FAIL**

This was the gate designated as able to stop the project, and it fails. Same
box, same exact solution (a slip triangle placed outside it, self-checking to
`max|div σ| = 4.7e-9`), same ε, both formulations solved and scored on the
same points.

Relative L2 error of surface displacement, force element ÷ direct BIE, over
six mesh densities:

| triangles | 512 | 1152 | 2048 | 3200 | 4608 | 6272 |
|---|---|---|---|---|---|---|
| FE ÷ direct | 0.877 | 1.158 | 1.371 | **1.545** | 1.693 | **1.825** |

**The ratio is not a constant — it is a trend.** The force element converges
at O(h^0.32), the direct BIE at O(h^0.90); the ratio grows like h^−0.58 and
crosses the 1.5× limit between 2048 and 3200 triangles. Refining the mesh
makes the *relative* standing worse, which no tolerance choice can fix.

### Where it comes from, and where it does not

The agent localised it rather than just reporting it:

| experiment | result |
|---|---|
| **all-Dirichlet box** (R3/G rows only, no free-traction row) | force element is **4–8× BETTER**, at O(h^1.7) |
| **sphere, pure Neumann, no edges** (R1 rows without a polyhedron) | FE O(h^0.90) — the textbook O(h) for constant elements; direct BIE **superconvergent** at O(h^1.31) |
| error energy within 0.6h of a box edge (9.3 % of the area) | FE **31.9 %** (3.4× concentration), direct 20.5 % (2.2×) |
| fixed band away from every edge | FE still only O(h^0.58) |
| rigid-body fraction of the error at the finest mesh | FE 0.351, direct 0.154 |
| conditioning, 512 → 6272 triangles | FE 2.0e2 → 9.9e2; direct 7.6e2 → 3.3e3 (**FE 3× better**) |
| sign controls (t_F flipped, prescribed traction flipped) | both blow up 20× — the gate has power |

So: **the entire deficit lives in the free-traction rows R1.** With Dirichlet
rows alone the force element is markedly *better* than the direct BIE. On a
smooth Neumann surface it achieves the O(h) that constant elements are
entitled to — it is the direct BIE that does better than entitled, at
O(h^1.31), which is what its rigid-body-calibrated jump buys. Add polyhedron
edges and the force element degrades further, to O(h^0.32).

### What it does and does not mean

**No defect was found in `fbem/`.** Every convention checkable independently
is correct: R1/R3 signs (flip controls blow up 20×), the single-layer jump
against `clq.force_stress`, `Patch`'s geometric orientation (zero flips needed
on outward-wound box and sphere meshes), and `eps_bias_operator` (validated
against `clq`, converging to the predicted `g` at clean O(ε)).

Two caveats matter for reading this across to the production result, and both
are the gate's own:

1. **L2 scores absolute `u`; the production quantity is `Δu = u(het) − u(hom)`,**
   two solves sharing the same outer rows and anchor, in which correlated error
   cancels. 35 % of the force element's error is rigid-body at the finest mesh
   (vs 15 % for the direct BIE), and the common-mode fraction is larger still.
   A 1.8× ratio on absolute `u` does **not** imply 1.8× on `Δu` — and in fact
   the measured `Δu` deficit on `topo_inclusion` is 12.4 %, not 80 %.
2. **The production mesh straddles the crossing.** `host_top` has h/L =
   0.0375–0.075, against L2's 1.5× crossing at h/L ≈ 0.05. The production
   sides and base are 3–7× coarser than anything L2 tested, and L2 used
   uniform meshes only — grading toward edges, the remedy its own diagnosis
   implies, was not tried.

### The bottom line

The force element is cheaper (−27.7 % unknowns), better conditioned (3–20×),
structurally simpler (no interface displacement equation, no special case at
the outcrop), and **more accurate on Dirichlet boundaries**. It is **less
accurate on Neumann boundaries**, and that gap *widens* under refinement
instead of staying bounded. For a free-surface-dominated earthquake problem —
which `topo_inclusion` is, with 4558 of 7515 elements on free-traction
surfaces — that is the governing limitation, and it is the honest reason the
5.3 % floor below does not go away.

## The amplitude deficit

Scored on the "inclusion only" panel, `u(flat, het) − u(flat, hom)`, on 3951
of 4406 surface centroids (excluding 6 km around the outcrop rim, where the
free term is not ½ and does not converge in h). Each formulation is
differenced against **its own** homogeneous state, so errors common to both
states cancel.

`amp = ||FE|| / ||BIE||`; below 1 means the force element is too small.

| ε (km) | amp FE/BIE | rel_rms | corr |
|---|---|---|---|
| 1.5 | 0.9191 | 0.099 | +0.99825 |
| 3.0 | 0.8723 | 0.153 | +0.99600 |

The direct BIE's own ε-sensitivity over the same range is only 2.0 %, so the
force element is **~3.7× more ε-sensitive**. Both move the same way (amplitude
grows as ε shrinks), consistent with a common sharp limit. Two points are not
enough to extrapolate honestly — the 32× ε_int sweep below is what actually
resolves the ε-dependence, and it shows the ε-dependent part vanishing
superlinearly onto a floor near 5 %, not onto zero.

The `(ε/4)` on-surface bias correction helps consistently
(0.9191 vs 0.9126 at ε = 1.5; 0.8723 vs 0.8596 at ε = 3.0).

### Where the deficit lives

`eps` is per source element, so the interface and the outer boundary can be
mollified independently. Slope vs the direct BIE at the matching ε_out
(> 1 means the force element is too small):

| ε_int \ ε_out | 1.5 | 3.0 |
|---|---|---|
| **1.5** | 1.0862 | 1.0995 |
| **3.0** | 1.1291 | 1.1419 |

The two effects are **separable and additive to three digits**: raising ε_int
by 1.5 km costs +0.0426 (at either ε_out), raising ε_out by 1.5 km costs
+0.0131 (at either ε_int). The interface carries **3.2×** the error of the
whole outer boundary — free surface, sides, base and the fault's traction
field combined.

### How the deficit decomposes

Sweeping ε_int alone over a factor of 32, at ε_out = 3 (`excess` = slope − 1;
h ≈ 6 km on the interface):

| ε_int (km) | ε/h | slope | excess |
|---|---|---|---|
| 0.1875 | 0.031 | 1.08185 | 0.0819 |
| 0.375 | 0.063 | 1.08304 | 0.0830 |
| 0.75 | 0.125 | 1.08679 | 0.0868 |
| 1.50 | 0.25 | 1.09953 | 0.0995 |
| 3.00 | 0.50 | 1.14187 | 0.1419 |
| 6.00 | 1.00 | 1.23993 | 0.2399 |

Two things to read off.

**It is monotone — there is no minimum.** Shrinking ε_int never turns around,
even at ε/h = 0.031. That is expected here and worth stating: `clq`/`msd`
integrate the mollified kernel in **closed form** over each triangle, so a
small ε/h does not degrade the element integral the way a quadrature rule
would. Only the constant-density representation limits accuracy.

**The ε_int-dependent part is superlinear.** Successive halvings shrink the
excess by factors of 2.3, 3.3, 3.4, 3.2 — i.e. roughly ε^1.7 to ε², not ε.
It saturates at **≈ 0.0815**.

So the total deficit at ε = 3 km (12.42 %, as `1 − 1/slope`) decomposes as

| source | size | removable by |
|---|---|---|
| interface mollification, ~O(ε²) | 4.9 % | smaller ε on the interface — **free**, already supported |
| outer-boundary mollification | 2.3 % | smaller ε elsewhere — costly (the fault source is the expensive part) |
| floor | 5.3 % | mesh refinement only |
| **total** | **12.4 %** | |

(The 5.3 % floor extrapolates ε_out → 0 linearly from the two available
points, so it is the softest number in the table.)

### A model I proposed and the data rejected

I first explained the deficit as *dilution*: in the matching-BC formulation the
material jump is geometric and sharp and ε only smooths the kernel, whereas
here **the contrast *is* the interface layer**, so mollifying it grades the
contrast over a shell of thickness ε and weakens it. That predicts a deficit of
area·ε/volume = 41 233 × ε / 883 573, i.e. 7.0 % at ε = 1.5 and 14.0 % at
ε = 3 — against 8.1 % and 12.4 % measured.

The agreement is coincidental and the model is **wrong**. It is linear in ε,
and the sweep above shows the ε-dependent part is superlinear; the apparent fit
came from comparing the model against the *total* deficit, most of which is the
ε-independent floor. Mollification by a symmetric blob preserves the layer's
monopole and cancels its dipole by symmetry, so the leading surviving error
should be O(ε²) — which is what the sweep actually measures. The qualitative
statement (ε grades the contrast, so the inclusion looks weaker; hence the
sign) survives; the linear magnitude does not.

**Practical consequence, unchanged:** ε should be chosen *per surface* — small
on material interfaces, usual elsewhere. That is already supported (`eps_arr`
is per element) and costs nothing: at ε_int = 0.375, ε_out = 3 the deficit
falls from 12.4 % to 7.7 % with the same mesh, same runtime, same conditioning
(2.81e4 throughout).

## Reproducing

```bash
PY=/Users/meade/micromamba/bin/python
$PY geometry.py                 # rebuild + cache the meshes (7515 boundary tri)
$PY topo_inclusion.py --check   # conventions + fault sign only (~150 s first run)
$PY topo_inclusion.py           # the two-solve trial (~25 s after the fault cache)
$PY eps_ladder.py               # eps = 1.5, 3, 6
$PY regen_reference.py 3.0 1.5  # re-solve the direct BIE at matching eps
$PY compare_ladder.py           # the matched-eps table above
$PY eps_split.py                # the interface/outer factorial
$PY verify/run_all.py           # every gate, one PASS/FAIL line each
```

`geometry.py` and the fault source cache to the scratchpad, so only the first
run of each pays for them; everything after `topo_inclusion.py --check` is
seconds to a couple of minutes.

## Open

- **The 5.3 % floor — L2 identifies it: the free-traction rows R1.**
  My earlier guess, interface discretization, is refuted by L3: it measures the
  force element's **absolute** error against Eshelby on a smooth spherical
  interface as 2.49e-2 at h/R = 0.196 and 1.33e-2 at h/R = 0.098, and
  extrapolating that O(h) fit to `topo_inclusion`'s h/R = 0.08 gives only
  **~1.1 %** — at most a fifth of the floor. L2 then shows where the rest is:
  the indirect single layer stalls on free-traction rows over a polyhedron,
  and `topo_inclusion` has 4558 of 7515 elements there, with an outcrop rim
  and a bottom corner on top of the box's own 12 edges.
- **The remedy L2 points at, untested: graded refinement toward edges.** Its
  own summary is blunt — "uniform refinement buys the force element almost
  nothing". The error energy concentrates 3.4× within 0.6h of an edge, but no
  exclusion radius makes the ratio flat, so this is worth trying and is not
  guaranteed to work.
- **The pivotal open problem: an indirect analogue of the calibrated jump.**
  L2 measured the direct BIE at O(h^1.31) on a smooth Neumann surface, i.e.
  *better* than constant elements are entitled to, against the force element's
  textbook O(h^0.90). That superconvergence is what `jump="calibrated"` buys.
  Finding the equivalent for the indirect operator would close most of the gap;
  the naive candidate is ruled out immediately below.
- **The free term is not the problem.** The direct BIE uses
  `jump="calibrated"`, a free term set so a rigid-body field is annihilated
  exactly, replacing the analytic ½; the force element uses the analytic ½,
  which invited the guess that this was the gap. It is not: L0 measures the
  flat-element free term at **0.4999079**, and the on-element value as the
  exact average to 7.4e-10. The obvious indirect analogue — calibrating on the
  closed-surface equilibrium identity `Σ_q A_q B_qp = −(A_p/2) I` — is not
  usable as a *row* calibration anyway, because that identity needs exact
  integration over the field surface, which one-point collocation does not
  provide (L0 measures it converging as 2.9e-5 → 4.6e-7 → 6.0e-9 over
  80/320/1280 facets, so it holds in the limit but not on a working mesh).
  The invalid attempt is kept, marked, in the scratchpad.
- L0 does record one genuinely suggestive number: with the source **on** the
  surface, the discrete net transmitted force is 0.4738 of −qA at ε/h = 0.5
  and 0.4867 at ε/h = 0.25 (extrapolating to 0.49967). That is a ~5 % shortfall
  at production settings with the same sign as the deficit — but it scales
  like ε/h, whereas the measured ε_int dependence is ~ε², so it is probably
  not the mechanism either. Worth one more look.
- **L3's own caveat, carried forward.** A sphere is a weak discriminator of
  the *sign* of the B block: flipping it is caught (1.01e-1 at 320, and
  notably non-converging) but only by ~4×, and at α = 0.5 alone it would be
  MISSED (2.53e-2 against a 3.0e-2 gate). The `(1−α)` scaling on B is the
  other weak control — dropping it costs only 1.4× at α = 0.1. Both want an
  **ellipsoidal** inclusion to pin them properly, as the spec already says.
  Nothing currently in `verify/` pins the B sign unambiguously.
- Unequal Poisson ratios need the volume polarization term; not implemented,
  and not guarded against.
- Only the flat-surface pair was run. The topography states are built
  (`geometry.load()["host_top_topo_*"]`) but untried.
