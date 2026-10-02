# Force-element BEM — findings from a closed line of investigation

**Status: closed, 2026-09-18.** The equivalent-body-force ("force element")
BEM was built, verified and measured against the existing displacement-
discontinuity (DD) / matching-BC direct BIE, and then dropped. This file is the
record. The code is gone from the working tree; it is recoverable in git
history at **`47e00ff`** on branch `fbem-force-elements` (see also `b9e06f2`,
`2725372`, `d11b85d`).

Nothing here is a reason to rebuild it. It is written so the same ground is not
re-explored by accident, and so the parts that *are* reusable are visible.

---

## What it was

One unknown field: a force density `q` (force per unit **area**) on every
boundary triangle, in a **single uniform** reference medium. A material
contrast entered only through its interface, with no matching of displacement
and traction across sub-domains:

```
u(x) = u_F(x) + ∫_S G^eps(x, y) q(y) dS(y)
```

Rows, with `n` the field normal, `t_on = B q + t_F`, `a = mu_inc/mu_host`, and
the single-layer jump `t(+n) − t(−n) = −q`:

| row | surface | equation |
|---|---|---|
| R1 | free traction | `(1/2) q + B q = −t_F` |
| R2 | interface | `((1+a)/2) q − (1−a) B q = (1−a) t_F` |
| R3 | prescribed `u = 0` | `[G − (eps/4) M] q = −u_F` |

R2 is traction continuity of the *true* traction, `t(+n) = a t(−n)`.
Displacement continuity across the interface holds identically, because a
single layer is continuous — that was the whole economy of the method.

## What worked, and is worth remembering

* **It solved the production model.** The topo_inclusion model
  (`src/mbem/cases/topo_inclusion.py`):
  22 545 unknowns vs 31 194 (−27.7 %), condition number 2.8e4 vs 4.4–5.8e5,
  both solves in ~25 s, correlation **+0.998** with the cached matching-BC
  solution, amplitude 12.4 % low at ε = 3 km (7.7 % with ε chosen per surface).
* **Dirichlet rows were 4–8× better than the direct BIE**, at O(h^1.64). This
  was never explained, and unexplained superiority is a clue worth keeping.
* **Interfaces were clean.** Eshelby: 2.49e-2 at 320 triangles → 1.33e-2 at
  1280, clean O(h), soft *and* stiff, with interface-only conditioning
  2.03–2.25 that was **mesh-independent**.
* **The inclusion outcrop needed no special case** — `a` cancels out of R1.
* **It tolerated graded meshes far better than the direct BIE**: conditioning
  9.9e2 → 1.5e5 across β = 1→3, against the direct BIE's 2.1e3 → **5.7e19**.

## Why it was dropped

### 1. It stalls on free-traction rows over a polyhedron

Head-to-head against the direct BIE, same box, same exact solution, same ε:

| | rate in h | rate vs unknowns |
|---|---|---|
| force element | **+0.31** | −0.15 |
| direct BIE | **+0.90** | −0.45 |

The error ratio grows like h^−0.58 and crosses 1.5× between 2048 and 3200
triangles — refinement makes the *relative* standing worse. The deficit is
entirely in the free-traction rows; on a **sphere** (same rows, no edges) it
recovers O(h^0.90).

**Cause, measured.** The density has a genuine edge singularity. Regressing
`log|q|` on log(distance to edge) gives exponent **−0.328, −0.328, −0.329**
across two meshes and a 3× change in ε — i.e. **ρ^(−1/3)**, exactly the
exterior 270° wedge prediction (λ = π/(3π/2) = 2/3, traction ~ ρ^(λ−1)),
stable in both h and ε. The direct BIE's Neumann unknown is `u`, which is
bounded; the indirect unknown is a density that is not.

### 2. Higher order does not fix it

P0 +0.31, P1 +0.32, P2 +0.28 in h. At matched unknown count higher order lost
**0 of 10** comparisons, with the gap widening (P1 1.35→1.38× worse than P0,
P2 1.44→1.57×). Error-energy concentration near edges was identical to within
4 % across all three orders.

**But mollification is a co-equal blocker, and this is the transferable
lesson.** At ε = 0.3h no collocation point is even one mollification length
clear of the element boundary (0.79ε for P0, 0.39ε for P1/P2), and the analytic
free term ½ needs that clearance. Hold ε *fixed in absolute terms* and it
inverts: P0 goes flat below ε/h = 0.15 while P1/P2 keep improving, and at
ε = 0.25 absolute **P1 beats P0 by 1.37×** at matched unknowns.

> **ε is not only a regularisation. It is a floor on the resolvable structure
> of the unknown.** No basis can represent detail below ε. Any future
> higher-order work — including higher-order DD — must budget ε/h before
> expecting p-refinement to pay.

### 3. Graded meshes do not rescue it

Grading *is* the textbook cure for ρ^(−1/3), and it demonstrably works in the
sharp limit: with ε pinned at 0.05 ≪ h_min, β = 2 grading lifts the rate from
O(h^0.45) to **O(h^1.01)** and beats uniform by 22 % at 13 824 dof.

At any usable ε it loses. At ε = 0.3h, matched unknowns: uniform 0.0368,
β = 1.5 0.0402, β = 2 0.0402, β = 3 0.0462, with rate-per-unknown flat across
β. The apparent rate gain *in h* is an artifact — grading inflates `h_max` by
2.3× at fixed dof.

Three further walls, any one of which is fatal:

* Equidistributing the error for ρ^s needs β ≥ 2/(1+2s); for s = −1/3 that is
  **β ≥ 6**, a 3-million-fold element-size range on one face at n = 20.
* At the bottom rim, where free sides meet the Dirichlet base, the density goes
  as ρ^(−0.95…−0.98), for which that formula **diverges** — no algebraic
  grading recovers O(h) at a Neumann/Dirichlet junction.
* Production has no ε headroom where it matters: `host_top` is already at
  ε/h = 0.367 and `inclusion_top` at 0.65. Only the sides and base (0.047) have
  room, and grading the top rim pushes ε/h the wrong way on a curve that is
  already flat.

### 4. The interface economy inverts under unequal Poisson ratios

The saving is **exactly** the interface: 31 194 − 22 545 = 8 649 = 3 × 2 883,
the interface triangle count. Every saved unknown is an interface unknown — and
that saving is conditional on `lam0*mu1 − lam1*mu0 = 0`, i.e. equal Poisson
ratios.

With unequal Poisson a **volume** polarization source appears, requiring a
tetrahedral mesh of the inclusion: ~25 000–35 000 tets → **75 000–105 000
unknowns, against 8 649 saved**. A soft sedimentary basin with a different
Poisson ratio from host rock does not degrade the method; it inverts its
premise. Meanwhile the one decisive advantage (Dirichlet rows) applies to
`host_base`: **74 of 7 515 triangles, 0.98 % of the model**.

### 5. The two repairs that looked available are not

* **Hybrid** (direct BIE on the free surface, force elements on interfaces).
  The claim that it needs no hypersingular operator is **false**: the interface
  traction row needs `n·[SG t − SH u]`, and `SH` is the hypersingular kernel as
  an assembled block. "Different surfaces" buys freedom from the *free term*,
  not from the kernel. At the outcrop rim, where the two surfaces meet, the
  discrete Somigliana traction at the real standoff (2.43 km, ε = 3 km) is
  **76–80 % wrong**, flat in h, and *worse* under refinement. `msd` already
  knows: `mbem/evaluate.py::_warn_near_boundary`, "shrinking eps does NOT help".
* **Symmetric Galerkin.** `W` is *not* a new derivation — it exists twice
  already (`clq.influence(want=("H","E"))`, `msd._dd_stress_pair`) — and
  Galerkin assembly is N²·Q_outer, not N²·Q², because the inner integral is
  closed form. The blocker is different: **the mollified `W` self-block is
  46–73 % wrong at production ε/h** (26.7 % of the sharp value at ε/h = 0.5,
  54.1 % at 0.3, scale-invariant). The single layer has an exact O(ε)
  correction; `W` has nothing equivalent. And coercivity bounds error by best
  approximation, which the edge singularity still limits.

## What survives and is still in the tree

* **`clq`'s force kernel** — `clq.force_displacement`, `clq.force_stress`,
  `want` keys `G`/`S`, P0/P1/P2, including `eps = 0` evaluated **on** the
  element, which the DD kernels cannot do. Correct, gated, and independent of
  this decision. Kept deliberately.
* **`tests/gates/clq/verify_baseline_bitwise.py`** — pins `U`/`H`/`E` against 648
  byte hashes from the pre-force-element tree. Written because the force-element
  work silently moved `U` by ~1e-16 and no existing gate could see it.
* The measured fact that ε/h governs whether p-refinement pays (§2 above).

## One unclaimed win, found while refuting grading

Grading the box **sides** toward the top rim — not the top itself — costs
**+2.4 % unknowns, cuts top-surface L2 by 22 %, and *improves* conditioning
2.2×**, because it removes a mesh-size discontinuity rather than creating
anisotropy. Refining the sides as well: +21 % unknowns, −43 % error. This is a
property of the *mesh*, not of the force element, and it should transfer to any
formulation on the `topo_inclusion` geometry. It has not been tried with the
direct BIE — whose conditioning is far more fragile under grading (5.7e19 at
β = 3), so it would need care.
