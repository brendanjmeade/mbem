# Backlog

The one status document for `moss-org`. History and measurements live in
`git log` (each commit message carries its numbers); rules live in each
package's `CLAUDE.md`. Trunk: `msd/mbem`. `ddbem` is to be closed and `clq`
frozen as an oracle once P1/P2 are ported.

## Standing rules

* Every stress presented as elastic subtracts the eigenstress `C:eps*` of
  every mollified double layer — fault slip and boundary `u_p` alike
  (`evaluate_stress`, default). Raw totals are kernel demos only.
* Fault slip sense: `Patch.value = u(-n) - u(+n) = -b`; `ddbem`/`clq` use `+b`.
  `mbem.selfcheck` pins it; never restate it.
* eps/h: `eps="auto"` = 0.1 h; on-fault stress wants <= 0.07 h on the fault and
  <= 0.125 h on a top patch near a trace (`msd/CLAUDE.md`, envelope section).
* Lean: docstrings state the rule; no dates or review numbers in code; no probe
  scripts in the tree; extend a gate before adding one.

## Open — correctness

1. **Production eps** (fault box: eps/h = 0.42 on the fault, 0.35 on the top)
   sits above every rule: trace-adjacent surface displacement −13 % vs the
   half-space classical (0.5 % at eps/h = 0.12); on-fault sigma_xy +5–15 % in
   the top 6 km. Decision needed before any paper number is re-quoted.
2. **First element row below a free surface**: on-fault sigma_xy +30–46 % at
   every h and eps (P0 staircase of the top layer, amplified by the
   image/total ratio ~D/2z). Measured cures: P1 top density at eps/h <= 0.125
   (to the floor) or h_top <= h_f/4 with eps_f <= 0.07 h_f and eps_top <=
   0.125 h_top (+4.5 %). Refining h_top alone, or a mirror source, does not work.
3. Box truncation: surface displacement −4 % (32–64 km), −19 % (64–128),
   −39 % (128–200) on the 200-km box — the manuscript's "ample box" claim.
4. Copy parity: five kernel copies in `moss`/`medt_paper` have no gate;
   `mode="basis"`/`"legacy"` of the dense backend are compared to nothing.
5. Near-trace displacement band (|x| < 1.5 eps) has no independent anchor.

## Open — clean (B sweep, mostly done)

* Done: the 1/2 free term, the calibrated diagonal, the eps-spec resolution,
  density selection and kernel tags are each stated once; `la/scaling.py`,
  the three-region wrappers and the unreachable complex path are gone; both
  backends share `Backend(jump, deflate).assemble(...).solve()` -> dict with
  `asm.report`; every gate exits 1 on FAIL and `msd/verify/run_all.py` runs
  them; the Navier-residual check replaced the sympy script; the demos use
  the exact eigenstress.
* Left: `medt_paper/topo_inclusion/mbem` is a frozen fork on the old API
  (item 7 of the consolidation).

## Open — before the first benchmark (C)

* An H-matrix configuration with real low-rank blocks and single-digit
  fallback, gated (`ADMISSIBILITY_ETA = 2.0` is used by no caller; at 10k
  unknowns the "compressed" operator is 1.4–2.1x dense).
* Per-patch / "auto" eps through the H path; a convergence-rate gate.

## Consolidation

Port `clq`'s P1/P2 moment recursions into `mbem/kernels/tri_kernels.py`
(numba, gated 1e-12 against clq; shrunk collocation 0.5, nodes >= 1 eps from
any fault plane), add `Patch.order`, adopt the conventional slip sign at that
point; close `ddbem` (findings kept), freeze `clq`; `medt_paper` pins a tagged
version instead of vendored copies (its fork still has EPS_OVER_H = 1.25 and
its `make figs` cannot build from a fresh clone); trim `moss` to the paper
and `mhf`. Outward-facing steps need a go-ahead.
