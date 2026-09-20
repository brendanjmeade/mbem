# Backlog

The one status document for `moss-org`. History and measurements live in
`git log` (each commit message carries its numbers); rules live in each
package's `CLAUDE.md`. Trunk: `msd/mbem`. `clq` is the frozen oracle for its
P0/P1/P2 kernels; `ddbem` and `fbem` are closed (`FINDINGS.md` each).

## Standing rules

* Every stress presented as elastic subtracts the eigenstress `C:eps*` of
  every mollified double layer — fault slip and boundary `u_p` alike
  (`evaluate_stress`, default). Raw totals are kernel demos only.
* Fault slip: `Patch.value = b = u(+n) - u(-n)`, the same as `clq` and cutde;
  `mbem.selfcheck` pins it. Never restate it.
* eps/h: `eps="auto"` = 0.1 h on boundary patches and ONE value 0.07 min h on
  a fault (per-element widths on a uniform-slip fault are wrong); a top patch
  near a trace wants <= 0.125 h; on-fault stress in the first element row
  needs a P1 top and is eps_top-limited within ~3 eps_top of the surface
  (`msd/CLAUDE.md` rule 4).
* Higher order (`Patch.order` 1 or 2) pays on traction rows and on the first
  row, not on displacement rows (first-kind traction unknown; cond grows
  ~15x per order), and only with eps room (`ddbem/FINDINGS.md`).
* Lean: docstrings state the rule; no dates or review numbers in code; no probe
  scripts in the tree; extend a gate before adding one; no new documents.

## Open — correctness

1. Production eps (decided: `"auto"` everywhere; the 200-km box unchanged).
   Fault box vs the half space: trace-adjacent surface u −0.1 %, on-fault
   sigma_xy at 8–18 km within 1 %.
2. First element row (closed: P1 top, `build_model(order_top=1)`, gated in
   `verify_solved_bvp` A4 vs cutde's half space; centroids deeper than
   3 eps_top go from +5–13 % at P0 to <= 3.5 %). What is left is the
   mollification band itself: within ~3 eps_top the row is eps-limited at
   any order and reaches 1 % only at eps_top = 0.025 h_top (gated; conditioning
   unchanged at 1.1e4). Decide whether the `"auto"` rule for a top patch
   near a trace should drop to 0.025–0.05 h with the C-item conditioning study.
3. Box truncation (decided: the 200-km box stays): surface displacement is
   −4 % (32–64 km), −19 % (64–128), −39 % (128–200) vs the half space; the
   manuscript's "ample box" sentence should be softened to a measured
   statement (outward-facing, needs a go-ahead).
4. Copy parity: `moss/mollified_kernel` and `medt_paper/mollified_kernel` are
   byte-identical today but ungated; `medt_paper/mhf` is a pre-correction fork
   of `moss/mhf`; `mode="basis"`/`"legacy"` of the dense backend are compared
   to nothing.
5. Near-trace displacement band (|x| < 1.5 eps) has no independent anchor.
6. Thin triangles: P1/P2 lose (L/height)^2 digits; refused below
   height/L = 1e-3 (`NODAL_MIN_HEIGHT_OVER_L`); the far-field switch keys on
   the longest edge, so a needle observed along its axis stays closed-form.

## Open — before the first benchmark (C)

* An H-matrix configuration with real low-rank blocks and single-digit
  fallback, gated (`ADMISSIBILITY_ETA = 2.0` is used by no caller; at 10k
  unknowns the "compressed" operator is 1.4–2.1x dense). Per-patch / "auto"
  eps through the H path; a convergence-rate gate.
* The compressed backend is P0-only (`la.hop.require_order0`);
  `MAX_ADMISSIBLE_BLOCK` and `ACA_FALLBACK_CONCURRENCY` are sized for K = 1 and
  must shrink by K when it gets `order`.
* The general-order pair code (`tri_nodal.py`) at P0 is 5–7x slower per pair
  than the unrolled P0 code (`tri_kernels.py`), so order 0 still routes to the
  old pair kernels; retire them once the general code is measured equal.
* `medt_paper/topo_inclusion/mbem` is a frozen fork on the old API; figure 10
  is rendered from its cache.

## Outward-facing (need a go-ahead)

* Publish `medt_paper`'s 2026-09-17 changes (the public copy is at `babc085`)
  and the Provenance section; manual export, never a push from here.
* Manuscript: soften the "ample box" sentence (item 3); state the on-fault
  near-trace caveat (item 2).
