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
  the fault occupying the top 20 km.
* 9 of 10 `medt_paper` figures regenerate locally, and all four tracked caches
  are current with today's kernels (fig04, fig06, fig08 each re-derived).

## Fixed on 2026-09-18

| finding | commit |
|---|---|
| msd subtracted the **approximate** eigenstress on-fault — 2.0× too large at the rim, absolute error *growing* under refinement (5.6 → 35.3 MPa) | `bd19ef7` |
| **Nothing pinned the fault's sign**: a flipped RHS passed 28/28 gates | `70f5761` |
| The sign convention was written **five** times independently | `2876697` |

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

## 2. `EPS_OVER_H = 1.25` is a shipped footgun

**Severity: major. The live one.**

`msd/mbem/defaults.py:91` sets `eps="auto"` to ε/h = 1.25 — **4.2× `ddbem`'s
default and ~8× the measured optimum**, 745× worse conditioned, with no accuracy
gate anywhere. Worse, **`1.25` together with `jump="half"` is a non-convergent
combination**, measured on a manufactured uniform-strain solution on an
icosphere ladder (`scripts/probe_uniform.py`).

**Fix:** set `EPS_OVER_H = 0.3` with the sweep recorded in the comment; make
`jump="calibrated"` the default, or warn when `jump=="half"` and
`max(eps)/min(h) > 0.5`; give `verify_eps_auto.py` a real accuracy assertion
against the exact Kelvin solution. **Changing this alters results for anything
using `eps="auto"` — it needs a deliberate decision, not a silent bump.**
*Effort: hours for the defaults, a day with the gate.*

## 3. No documented ε/h operating envelope

**Severity: major.** The tree has none, and ε/h governs accuracy *and*
conditioning — so this is load-bearing for the preconditioning work.

Measured (`scripts/floor.py`, msd P0, 1280-triangle icosphere, exact Kelvin
reference, ε/h swept 0.02 → 1.25):

* **Floor.** Dirichlet interior u: 7.47e-4 / 7.07e-4 / 6.66e-4 / 8.29e-4 at
  ε/h = 0.02 / 0.075 / 0.10 / 0.15 — flat. Neumann surface u: 6.58e-3 / 6.78e-3
  / 7.87e-3 at 0.02 / 0.04 / 0.075. **Below ε/h ≈ 0.10 nothing improves.**
* **Envelope: use ε/h ∈ [0.10, 0.30].** Below 0.10 you pay and gain nothing;
  above ~0.5 convergence degrades.
* Production sits at ε/h = 0.367 on `host_top` and 0.65 on the inclusion
  surfaces — *above* the envelope. Only `host_sides` and `host_base` (0.047)
  have room.

**Fix:** write it into `msd/CLAUDE.md` and `msd/README.md` as a table. *Effort:
a day.*

## 4. Interior stress near a boundary is h-**independent**

**Severity: major**, and worse than "non-monotone" as previously recorded.

At ε/h = 0.3, Neumann, relative interior stress error at d/h ≈ 0.66:
**2.32e-1 (80 tri) → 2.15e-1 (320) → 2.06e-1 (1280)** — essentially flat across
a 16× refinement, while the deep interior converges normally
(`scripts/standoff.py`, `standoff.jsonl`). **Refinement only narrows the bad
zone; it does not improve it.** msd's `_warn_near_boundary` fires at 0.5 local
h — about 4× too late.

**Fix:** raise the threshold to 2.0 local h, add an ε arm (`d < 2*eps` on the
nearest patch), and make the message quantitative. Document the usable standoff
for plotting volumetric stress. *Effort: hours.*

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

---

## What the gates still cannot see

Recorded so nobody assumes more coverage than exists:

* **Stress through a solved BVP is unanchored.** `verify_solved_bvp.py` closes
  assembly → BCs → solve → *displacement* against two independent anchors.
  Nothing does the same for *stress* — measured: reverting the `bd19ef7`
  eigenstress fix gives **0 of 144 failures** in that gate. It is caught only by
  the two gates from that commit. This is the same structural hole, one level
  down.
* **The near-trace band is untested.** Anchor A excludes points within 1.5 ε of
  the trace, because mollification legitimately smears the surface step there —
  so the near field, the regime the mollified method exists to handle, has no
  independent anchor.
* `verify_solved_bvp.py` covers `DenseBackend("direct")` only — not `HBackend`,
  `mode='legacy'/'basis'`, `deflate=True`, `eps='auto'`, or an all-Neumann
  model. Those are covered by other gates, which is why several seeded defects
  were caught by `verify_hbackend` and `verify_deflation_estimate` instead.
* One fault geometry only: a single vertical plane, n̂ = +x̂, pure strike slip.
  No dipping fault, no tensile component, no non-axis-aligned case.
* Every gate is a **threshold** gate, not a convergence gate. Per-patch ε
  misrouting degrades accuracy by 2.5× and passes, because the headroom is
  larger than the damage.
