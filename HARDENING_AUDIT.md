# Hardening audit of the constant-slip mollified (ε > 0) path — 2026-09-18

A four-area audit of the P0 mollified DD path: copy drift, gate coverage,
physics gaps, and whether the published result reproduces. Everything here was
**measured**, not inferred. Three findings were fixed the same day; the rest is
the backlog.

The scripts behind the measurements are in `audit_2026-09-18/scripts/`
(salvaged from a session scratchpad, so they survive). They need only numpy and
run from the repo root.

---

## Retired worries — checked, and fine

These cost real effort to establish. **Do not re-check them.**

* **The λ/μ traction-pairing fix is present and correct in all 13 copies.**
  Verified *numerically* at ν = 0.30 and 0.45, not by grepping for a comment
  (`scripts/audit_pairing.py`, 8 s). The swapped form differs by 73 % / 104 %
  there and coincides to 6.3e-14 at ν = ¼.
* **The production path is end-to-end correct against an independent code.**
  msd's free-surface BEM vs `cutde`'s half-space TDE: 4.0 % median on surface
  displacement, cosine +0.9993. That 4 % is box truncation, i.e. physics.
* msd's numba kernels agree with clq's independent closed form to machine
  precision; the two `mbem` forks are bitwise identical on the kernels.
* `ddbem`'s three known gate holes (Voigt row order, far-field default,
  orientation sign) do **not** exist in msd or clq.
* The 6-mode rigid-body border does not touch stress and does not apply to the
  production model. The msd-vs-`ddbem` Neumann gap is real in code and
  irrelevant to production — nothing to adopt.
* On-fault elastic stress is **not** wrecked by free-surface proximity, despite
  the fault occupying the top 20 km — **partly retracted 2026-09-19.** That was
  measured on the strike shear σ_xy only, on which a horizontal free surface's
  eigenstress is identically zero (it is ∝ sym(u_p ⊗ ẑ)). The other traction
  components in the top element row carried the un-subtracted boundary
  eigenstress of item 4: σ_xz 0.05–0.36 MPa at production settings,
  wrong-signed at the corner elements, and 3–9× the 0.6 MPa strike shear on
  finer meshes. Removed by the 2026-09-19 fix (median residual vs the
  half-space classical reference 0.0025 MPa, 0.4 % of the shear). What
  survives for σ_xy is now quantified and is item 9 below.
* 9 of 10 `medt_paper` figures regenerate locally, and all four tracked caches
  are current with today's kernels (fig04, fig06, fig08 each re-derived).

## Fixed on 2026-09-18

| finding | commit |
|---|---|
| msd subtracted the **approximate** eigenstress on-fault — 2.0× too large at the rim, absolute error *growing* under refinement (5.6 → 35.3 MPa) | `bd19ef7` |
| **Nothing pinned the fault's sign**: a flipped RHS passed 28/28 gates | `70f5761` |
| The sign convention was written **five** times independently | `2876697` |

## Fixed on 2026-09-19

| finding | where |
|---|---|
| `evaluate_stress` never subtracted the eigenstress of the **boundary** double layers (only faults') — the whole "h-independent near-boundary stress" of item 4, and a spurious 0.05–0.36 MPa σ_xz on the fault's top row at production | `msd/mbem/evaluate.py::_double_layer_stress`; gate `verify/verify_boundary_eigenstress.py` (12 checks, 2 s) |
| `EPS_OVER_H = 1.25` (item 2): now 0.1; `jump="calibrated"` default; half + ε/h > 0.5 warns; calibrated + all-Neumann without `deflate` raises | `msd/mbem/defaults.py`, `backends/dense.py`, `backends/hmat.py`; gate `verify/verify_eps_auto.py` checks 4–5 |

---

# The backlog

Ranked. Effort is against this codebase.

## 1. A ~3 % fault amplitude error is invisible to the entire suite

**Severity: major. The silent one.**

Seeding `scale = -1.0 → -1.03` in `msd/mbem/model/equations.py` gives **0 of 144
failures in `verify_solved_bvp.py` and 13/13 scripts passing.** Direction is
unchanged, so the cosine anchor cannot see it; the magnitude anchor has a 12 %
tolerance because msd's box is a truncated half space while cutde's is infinite.
A few-percent amplitude error hides in that allowance.

**Fix:** an absolute amplitude anchor. The manufactured uniform-strain solution
(Anchor B in `verify_solved_bvp.py`) is exactly P0-representable and has no
truncation floor — extend it to carry a fault, or add a fault case whose total
moment is known in closed form. *Effort: hours.*

## 2. ~~`EPS_OVER_H = 1.25` is a shipped footgun~~ — FIXED 2026-09-19

`defaults.EPS_OVER_H` is now **0.1** (per-element ε = 0.1 h), with the
measured basis in the `defaults.py` comment and the full sweep in
`review_2026-09-19/scripts/item2/`. The deferral rationale here ("changes
results for anything using `eps="auto"`") was empty: nothing in-tree used
`"auto"` except the gates. Also done: `jump="calibrated"` is the default of
both backends; `jump="half"` with ε/h > 0.5 on any source patch warns
(`HALF_JUMP_MAX_EPS_OVER_H`); calibrated on an all-Neumann model without
`deflate` now **raises** in `AssembledDense`, `DenseBackend.assemble` and
`AssembledH.solve` (review #70/#27) instead of returning 1e9 km under a
warning; `verify_eps_auto.py` check 4 solves the Kelvin sphere ladder with
`"auto"` at the default jump and asserts rate and ceiling; the [d2] window
of `verify_eigenstress_exact.py` is pinned to the literal 1.25 h it was
measured at (review #35). One correction to this file: **conditioning
improves as ε/h drops** (fault box cond 37 / 44 / 108 / 1.7e4 at 0.05 /
0.1 / 0.3 / 1.25), so the "745× worse conditioned" was right about 1.25
and there is no conditioning penalty for going small. The public
`medt_paper/topo_inclusion/mbem/defaults.py` fork still carries 1.25 (item 7).

## 3. ~~No documented ε/h operating envelope~~ — DONE 2026-09-19, and revised

The envelope is in `msd/CLAUDE.md` ("The eps/h operating envelope") with
the 2026-09-19 sweep (1280-triangle icosphere vs exact Kelvin, ε/h = 0.05 →
1.25): Dirichlet interior u floors at ε/h ≤ 0.125 (8.6e-4; 6.5e-3 at 0.3),
Neumann surface u keeps improving to 0.05 (6.9e-3 vs 2.2e-2 at 0.3), and
cond(A) falls with ε/h. The 2026-09-18 "[0.10, 0.30], below 0.10 nothing
improves" was displacement-only and optimistic at its upper end; for
on-fault stress the rules are tighter (review finding 4, item 9). Production
still sits at ε/h = 0.367 on `host_top` and 0.65 on the inclusion surfaces —
**above** every rule; that decision (item 9) is open.

## 4. ~~Interior stress near a boundary is h-independent~~ — FIXED 2026-09-19; it was the un-subtracted boundary eigenstress

**Misdiagnosed on 2026-09-18.** The flat 2.32e-1 → 2.15e-1 → 2.06e-1 over a
16× refinement (Neumann, ε/h = 0.3, d/h ≈ 0.66) was not a mesh limit. Every
`-σ SH @ u_p` boundary term is a mollified double layer whose smeared
(fictitious) jump carries an eigenstress `μ u_p ⊗ n Φ_ε(d)` inside the body,
and `evaluate_stress` removed that term for faults only (found by
`CODE_REVIEW_2026-09-19.md` finding 1; `ddbem` had always removed it). With
it removed for every patch (`_double_layer_stress`), the same ladder
re-measured (`scripts/standoff.py` unchanged, output
`standoff_with_boundary_eigen.jsonl`; the old run is `standoff.jsonl`):

| ε/h = 0.3, 1280 tri | d/h 0.25 | 0.5 | 0.66 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| Dirichlet, old → new | 1.20e1 → 1.81e-1 | 1.99 → 6.1e-2 | 7.9e-1 → 3.4e-2 | 1.5e-1 → 3.1e-2 | 2.6e-2 → 2.7e-2 |
| Neumann, old → new | 3.07 → 2.11e-1 | 5.2e-1 → 1.0e-1 | 2.1e-1 → 7.4e-2 | 7.1e-2 → 6.1e-2 | 4.8e-2 → 4.7e-2 |

* **It converges now:** at every standoff d/h ≥ 0.5 the error falls at
  O(h^0.6–0.8) for both BC types (80 → 320 → 1280), the far-field P0 rate.
* **Usable standoff at 1280 tri, ε/h = 0.3:** < 10 % beyond d/h ≈ 0.39
  (Dirichlet) / 0.50 (Neumann), i.e. ~1.3–1.7 ε — versus ~1 h before.
* **Residual mesh-limited zone:** Dirichlet only, d/h < ~0.35, error 14–30 %
  improving < 15 % per halving of h — the piecewise-constant u_p density,
  now visible at its true size. Neumann converges everywhere measured.
* **"Smaller ε does not help" was wrong** — it described the un-subtracted
  term (∝ μ|u_p|/ε). Now the near-boundary error is nearly ε-independent:
  Dirichlet d/h = 0.66: 3.8e-2 / 3.4e-2 / 3.2e-2 at ε/h = 0.15 / 0.3 / 0.5
  (was 1.05e-1 / 7.8e-1 / 2.9).
* Displacements are bit-identical; stress beyond ~3 ε of every surface is
  unchanged to < 1e-3.

**Also done (review finding 6):** `_warn_near_boundary` now uses an exact
point-to-triangle distance (the centroid metric flagged 0 of 20 on-surface
vertex points); the 0.5 h threshold is about right (~10 % error at d/h
0.4–0.5), so the former "raise it to 2 h" recommendation is withdrawn.

## 5. Nothing checks that the 13 duplicated copies agree

**Severity: major.** `CLAUDE.md` warns that a kernel fix must be applied to
every copy by hand. Five copies have **zero** gate coverage:
`moss/mhf/analytical_kernels.py`,
`moss/mh_deploy/mollified_kernel/analytical_kernels.py`,
`medt_paper/mollified_kernel/analytical_kernels.py`,
`medt_paper/mhf/analytical_kernels.py`,
`medt_paper/topo_inclusion/mbem/kernels/tri_kernels.py`, plus all five
`anelastic.py` copies. `medt_paper` contains **no verify script at all**, and
every figure script hardcodes ν = 0.25 — the one value where the historical bug
is invisible.

**Fix:** `scripts/audit_pairing.py` already does this and runs in 8 s. Promote it
to `msd/verify/verify_copy_parity.py`, make it **fail loudly on a missing copy
rather than skipping**, and extend it to the `anelastic.py` copies
(`scripts/audit_eigen.py` is the start). *Effort: hours.*

Note `msd/mollified_kernel/` is the only copy of `analytical_kernels.py` /
`analytical_batch.py` in the tree **lacking** the exact eigenstress kernel — it
is a frozen oracle, so this is expected, but the parity gate must know it.

## 6. The on-fault stress gate is magnitude-only

**Severity: major.** `msd/verify/verify_evaluate_stress.py:174-178` computes
`growth = abs(t_lo)/max(abs(t_hi),1e-30)` and accepts `0.8 < growth < 1.25`.
Seeding `0.75 → 0.70` in `msd/anelastic.py:140` (a 7 % error in the Cortez
marginal) **flips the Coulomb sign and still passes.**

**Fix:** replace the absolute-value ratio with a signed one, and add a third ε
so it is a ladder rather than a pair. *Effort: hours.*

## 7. The public `medt_paper` artifact cannot build

**Severity: blocker for the public artifact; zero effect locally.**
**Outward-facing — do not act without asking.**

From a fresh clone of the DOI'd state (`babc085`, Zenodo
10.5281/zenodo.22712173), `make figs` dies on figure 1: every script writes into
`manuscript/figures/`, a directory deleted from the repo and created by only 2
of 10 scripts. `pip install -r requirements.txt` also leaves figure 9
unimportable (no sympy).

**And there is no push path.** The umbrella repo has no remote; `medt_paper` is
a plain subdirectory. The public repo's history and origin URL survive **only**
inside `~/Desktop/moss-org.zip`.

**Fix:** `os.makedirs(out_dir, exist_ok=True)` in the eight scripts that lack
it (or a `$(FIGDIR):` prerequisite in the Makefile), add sympy to
`requirements.txt`, re-establish a clone with the remote, and cut a new Zenodo
version. *Effort: hours, plus a decision that is not mine.*

## 8. Smaller, real

* **No reproducibility test** in `medt_paper`. A `make check` rendering the
  cheap figures (~26 s measured) and byte-comparing would be enough.
* **Figure 7 plots TOTAL stress**, contradicting `medt_paper/README.md` line 81
  and `EIGENSTRESS_AUDIT.md`. Either subtract the eigenstress and regenerate, or
  relabel it explicitly as a kernel demo — which the policy does permit.
  *Needs a decision.*
* **The manuscript's "ample box" claim does not survive measurement.** Beyond
  ~40 km the fig-10 surface field departs from a classical half space; the u_z
  panel is a box artifact over about a third of its area
  (`scripts/prod_vs_classical` route, cached solution vs `cutde` on the
  identical 633-triangle fault). Soften `main.tex:130` to a measured statement.
  *Needs a decision.*
* Three msd demos (`demo_anelastic_subtraction`, `demo_onfault_convergence`,
  `demo_fault_only`) still call `anelastic.eigenstress_at_points` directly and
  carry the rim error fixed in `bd19ef7`; `fig_bem_onfault_stress.png/.pdf` are
  stale.
* `medt_paper/mhf/anelastic.py` and moss's vendored copies still hold the
  approximate eigenstress. Benign today, and exactly what item 5 is for.

## 9. On-fault stress in the first element row below a free surface is +30–46 % high — DIAGNOSED 2026-09-19: numerical, two mechanisms, curable

**Severity: major** for near-surface on-fault Coulomb / stress-drop
statements — and the same measurements expose a **−13 % near-trace surface
displacement** at production settings.

Found while re-measuring the retired worry above, with an anchor the tree
did not have: the production `_fault_box` solved by msd (calibrated jump)
against `tde_reference.classical_tde_stress`'s **half-space** finite part
at the fault centroids. The reference is triangulation-independent at the
first row to 1e-8 (4× and 16× finer cutde meshes, and a 2-triangle
rectangle). Laterally deep elements, σ_xy, msd/ref − 1 by depth bin:

| depth bin | h 8 km, ε 3 (production) | h 2, ε 0.25 | h 2, ε 0.125 | h 1, ε 0.25 |
|---|---|---|---|---|
| first element row | — | +0.315 | +0.309 | +0.382 |
| 1–2 km | +0.052 | +0.059 | +0.042 | +0.050 |
| 2–3 km | +0.152 | +0.021 | +0.019 | +0.027 |
| 3–4 km | +0.130 | +0.016 | +0.016 | +0.020 |
| 4–6 km | +0.101 | +0.014 | +0.014 | +0.015 |
| 8–12 km | +0.014 | +0.008 | +0.008 | +0.008 |
| bottom rim | −0.349 | +0.001 | +0.001 | +0.001 |

Four experiments (`review_2026-09-19/scripts/item9/`: decomposition,
top-patch ladder, mirror source, ddbem P1/P2) established the following.

**Why it is amplified.** At the first-row centroid (depth z ≈ h/3) the
classical σ_xy is the small residual of two large cancelling terms — the
fault's own full-space top-edge field (+) and its half-space image (−) —
with |image|/|total| ≈ D/(2z): 3.7 at production, 14 at h = 2 km (antiplane
model: 4.7 / 18). A relative error δ in either piece is (D/2z)·δ in the
total. A full-space BEM builds the image numerically from the top patch, so
this amplification is intrinsic to it; a half-space kernel (`moss/mhf`)
carries the image analytically.

**What the error is.** Three sources feed the residual: (a) the mollified
fault's own top-edge smearing, ∝ ε_f/z (−21 % of the fault term at
ε_f = z, −0.3 % at ε_f = 0.24 z); (b) the solved surface density is the
ε_f-smeared trace step, which the top patch renders faithfully (−21 / −8.5 /
−3 % of the image at ε_f/z_row1 = 0.94 / 0.47 / 0.24); (c) the P0
**staircase rendering** of the top double layer at a point h/3 below it —
ε-independent, −2.4 % of the image at h_top = h_f on msd's Triangle-meshed
top (→ +33 % of the total), about −1 % on a structured strip (ddbem P0 shows
+12 %; the factor is mesh shape, not the basis). With uniform ε and
h_top = h_f, (a) and (b) are the same smeared line mirrored and cancel to
leading order, leaving (c): the ε-independent, h-scaled signature in the
table. **The solve is right:** the solved `u:top` matches the half-space
surface displacement to 0.5 % at ε/h_top = 0.12; substituting the exact
surface displacement on the same mesh still gives +29 %, and on a 4× finer
top mesh +1.4 %.

**Cures, measured:**

| approach | first row | cost | note |
|---|---|---|---|
| **P1 top density** on the near-trace band at ε/h ≤ 0.125 (`ddbem`) | +11 % → **−0.7 %** (at the floor); same at D/h = 10 | 3× unknowns on the strip | P2 adds nothing; at ε/h = 0.25 P1/P2 are **worse** than P0 (collocation nodes 0.6 ε from the fault plane) |
| **msd today:** h_top ≤ h_f/4 at the trace **and** ε_f ≤ 0.07 h_f **and** ε_top ≤ 0.125 h_top | +39 % → **+4.5 %** (h_f = 8: h_top 2, ε_f 0.5, ε_top 0.125; row 2 +1.9 %, row 3 +1.1 %) | n_top +25–30 %, ~1 s solve | the fault's ε must drop from 3 to 0.5 km |
| refine h_top alone at fixed ε | **no** — flips sign and saturates at −25 % (ε_f = 0.25 h_f), or climbs to +70 % if ε/h_top rises | — | (a) and (b) stop cancelling |
| mirror-fault (method of images) source | **no** — the solution doubles | — | a source outside the body is invisible to a DIRECT BIE (Betti: c·u_m + H·u_m − G·t_m = 0), and its RHS on z = 0 equals the real fault's by symmetry; only an indirect formulation or a half-space kernel can carry an image |
| exclude the first row from readouts | leaves row 2 at +5–9 %; at production the y = 0 profile is > 5 % down to 6.8 km | — | policy only |

**Rule (measured on y = 0 profiles):** on-fault σ_xy is within 5 % only
for z ≳ max(~1 h_top at the trace, ~5 ε). **New footgun:** a per-patch ε
with ε_top ≠ ε_f at a surface-breaking trace moves the first row by ±100 %
unless ε_f ≤ 0.07 h_f already.

**The free-surface observable.** Surface displacement (`u:top`) against the
half-space classical: at ε/h_top = 0.12 the trace-adjacent elements are
within 0.5 %; at **production ε/h_top = 0.35** (ε = 3, h_top = 8.5) they
are **−13 %** at 2–4 km from the trace (trace jump 0.85 b vs 0.97 b). Beyond
32 km every configuration is box truncation: −4 % (32–64 km), −19 %
(64–128), −39 % (128–200) — item 8's "ample box" claim, quantified. (Points
evaluated 1 m below the surface return u/2 — inside the mollified layer —
so surface fields must be read from `u:top`, never evaluated there.)

**Decision needed:** production ε/h_top = 0.35 and ε/h_f = 0.42 sit above
the envelope on exactly the surfaces the observations come from. Order:
items 2/3 first (ε down to ≤ 0.125 h on the top and ≤ 0.07 h on the
fault), then P1 on the near-trace band (`ddbem`, needs the numba port for
production scale) or h_top ≤ h_f/4 in msd.

---

## What the gates still cannot see

Recorded so nobody assumes more coverage than exists:

* **Stress through a solved BVP is unanchored.** `verify_solved_bvp.py` closes
  assembly → BCs → solve → *displacement* against two independent anchors.
  Nothing does the same for *stress* — measured: reverting the `bd19ef7`
  eigenstress fix gives **0 of 144 failures** in that gate. It is caught only by
  the two gates from that commit. This is the same structural hole, one level
  down.
* **The near-trace band is untested** — for *displacement*. Anchor A excludes
  points within 1.5 ε of the trace, because mollification legitimately smears
  the surface step there. For on-fault *stress* the band is now measured (item
  9): +30–46 % in the first element row, at every h and ε.
* `verify_solved_bvp.py` covers `DenseBackend("direct")` only — not `HBackend`,
  `mode='legacy'/'basis'`, `deflate=True`, or an all-Neumann model.
  `HBackend` and deflation are covered by `verify_hbackend` /
  `verify_deflation_estimate` (parity and null-space checks, not physics
  anchors); `eps='auto'` is anchored since 2026-09-19 by `verify_eps_auto`
  check 4 (Kelvin sphere ladder at the default jump); **`mode='legacy'` and
  `mode='basis'` are compared to nothing** (review #31 — correct today at
  1e-16 / 5e-15, ungated).
* One fault geometry only: a single vertical plane, n̂ = +x̂, pure strike slip.
  No dipping fault, no tensile component, no non-axis-aligned case.
* Every gate is a **threshold** gate, not a convergence gate. Per-patch ε
  misrouting degrades accuracy by 2.5× and passes, because the headroom is
  larger than the damage.
