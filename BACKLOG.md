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

## Scale program (approved 2026-09-20; the plan file has the details)

Decisions: target 1e5–1e6 elements on the interface topology (4 unknowns per
triangle, so 0.4–4 M unknowns), designed first around ~1 M unknowns with an FMM
far field as the follow-on; this workstation only (16 cores, 128 GB, CPU);
fast operator 1e-4 relative, FGMRES rtol 1e-8; one stored operator (~2x) for
the 2–22 material solves per geometry; P1/P2 into the fast path after the
flat-H fixes; region graphs = host + a few inclusions; numba on the OpenMP
layer and `threadpoolctl` are requirements.

Measured facts the program rests on: the reported cond 5e5 of the interface
models is a 1-norm estimate (2-norm 1e3–8e3; sigma_min is a smooth global mode
set by the coarse box sides, not the inclusion corner); the interface-aware
block-Gauss-Seidel is already size-independent (22 -> 24 iterations from 11k
to 29k unknowns; 12–13 on the fault box at every size), the count being 21
geometric outlier eigenvalues; region-level Gauss-Seidel gives 10 (7 with two
sweeps); the compressed build is 96–98 % Python overhead (ACA loop,
recompression under BLAS oversubscription) with kernel work at 2–4 %; the
flat H at eta 0.8 with per-basis storage is 1.4–2x dense at 10k unknowns and
would be ~5 / 17 / 65 GB at 100k / 300k / 1M unknowns once fixed (combined
storage, eta 2 with a real certificate, tol 1e-4, leaf 96, numba ACA).

Work packages, in order (each a commit with the harness numbers):
WP0 harness + `precond_summary` + convergence-rate gate; WP1 flat-H policy
(leaf 96, tol 1e-4, eta 2 in every caller, dense-on-rank-cap, full-row/column
certificate, single-use RHS matrix-free, combined storage); WP2 BLAS thread
control; WP3 batched dense leaves and numba matvec; WP4 ACA in numba; WP5
shared-subspace per-basis storage and preconditioner reuse across materials;
WP6 preconditioning ladder (E1 ladder to 107k, E2 GCRO-DR recycling, E3
region super-blocks, E4/E5 conditional); WP7 P1/P2 in the fast path; WP8
far-field engine; WP9 evaluation at scale.

Done: WP0-WP5a, E2 and WP9's eigenstress near list (`ec58a3f`, `cec5606`,
`645d676`). At 31k unknowns the build is 32 s (was: did not finish in 28 min)
and the matvec 28 ms; preconditioner reuse across a material sweep costs +1 to
+3 iterations and saves 2.5-5x wall per solve.

**WP8 Trial A (fmm3dpy) is closed: rejected on speed.** The eps = 0 Kelvin
layers are exact combinations of its Laplace and Stokes kernels (single layer =
Stokeslet + Laplace charge; double layer = stresslet + one Laplace call with
nd = 4 carrying the dipoles and the charge gradient; both verified to 1e-15),
and the accuracy passes with room: an exact mollified near field to 8 h with
the eps = 0 far field beyond it lands at 4.8e-5 relative, against a 1e-4 budget,
already at c = 8. But the macOS wheel is a SERIAL build (no OpenMP in any of
its shared objects; identical times at 1 and 16 threads), and one far-field
application costs 16 s per kernel at 120k unknowns against a 0.3 s flat-H
matvec: 100x the budget, and 55-60 s per kernel extrapolated at 422k against a
10 s ceiling. It also exposes no tree object, so WP9's "one upward pass serving
the solve and any target set" is unreachable through it. If an FMM is scheduled
for the >= 1M-element rung it is Trial B, the in-house Chebyshev bbFMM; the
kernel decomposition derived for Trial A carries over to it unchanged.

Still true until the packages land: the compressed backend is P0-only
(`la.hop.require_order0`); `tri_nodal.py` at P0 is 5–7x slower per pair than
`tri_kernels.py`, so order 0 routes to the old pair kernels;
`medt_paper/topo_inclusion/mbem` is a frozen fork rendered from its cache.

## Outward-facing (need a go-ahead)

* Publish `medt_paper`'s 2026-09-17 changes (the public copy is at `babc085`)
  and the Provenance section; manual export, never a push from here.
* Manuscript: soften the "ample box" sentence (item 3); state the on-fault
  near-trace caveat (item 2).
