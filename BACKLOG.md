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

Done: WP0-WP5a, E2, E3, E4 and WP9's eigenstress near list (`ec58a3f` ..
`2665d2d`). The ladder now reaches **260,598 unknowns in 329 s** (operator
264 s, preconditioner 26 s, solve 39 s, 42 iterations, 46 GB), where before
this work 31k did not finish a build in 28 minutes. At 117k: 154 s total,
30 GB. Preconditioner reuse across a material sweep costs +1 to +3 iterations
and saves 2.5-5x wall per solve; a 14-material sweep at 117k runs in 815 s.

Two committed decisions were reversed by later measurement, which is the
pattern to expect here: the preconditioner rung above the dense cap was set to
HODLR at 117k and then to block-Jacobi once 269k was affordable (its HODLR
build grew 24x for 2.3x the unknowns and peaked at 92 GB); and the flexible
form of Krylov recycling was implemented before being measured to deflate the
wrong spectrum. Measure at the next rung up before trusting a policy.

**Closed: that "compression quality degrades with size" was a MESH failure.**
The 269k rung's 1.39e-4 operator error, all of it on the unit-translation
vector, came from 1,808 zero-area triangles -- 8 % of `host_top` -- piled on
one point of the fault trace. At that mesh scale the topography refinement
ring's vertex count is a multiple of four, so a ring vertex lands exactly on
the axis-aligned trace; Triangle splits the segment there and collapses. The
route from a collapsed element to a compression error is short: its cluster's
bounding box shrinks to a point, `min(diam) < eta dist` then admits a block
whose elements TOUCH, and such a block's row sums are O(1), so a RELATIVE
1e-4 block tolerance leaves an ABSOLUTE 1e-4 in the calibrated diagonal the
row sums build. Fixed at both ends -- `inclusion_mesh` rotates a refinement
ring off the other constrained segments (`RING_SEGMENT_CLEARANCE`), and
`Patch` refuses a mesh with a collapsed element
(`MIN_TRIANGLE_HEIGHT_OVER_L`) so the class cannot return silently. After the
fix the compressed operator does not degrade with size at all: operator error
4.2e-5 / 2.8e-5 / 1.6e-5 at 31k / 117k / 261k unknowns, ACA fallbacks 194 ->
32, rank-capped 927 -> 294, retried 1068 -> 684, and the calibration's own
error |C_h - C_exact| 4.9e-5 -> 1.6e-5 (3.6e-5 at 31k, 2.9e-5 at 117k). The
rung is also 10 % faster and 10 GB lighter. `BENCH_OPERATOR_ERROR_MAX` stays
1e-4, and `verify_hbackend` now gates the calibration row sums against the
exact kernels (`H_PARITY_CALIBRATION`) -- the one part of the operator an
entrywise parity does not bound.

**Next: the shared-subspace fold, not the kernel.** Measured at 261k unknowns,
the ACA phase (208 s of a 245 s assembly) is **68.9 % the serial fold** that WP5
added, 20.3 % the rest of the compression pool, and only **10.8 % kernel
evaluation**. So a free kernel would buy 1.12x. The fold's own cost is
corroborated from the other direction: WP5 reported it adding ~54 s at 117k,
and this decomposition finds 50.8 s there; three independent estimates of the
kernel's share (at 1, 6 and 12 quadrature points) agree to 1 %.

The fold is `shared_subspace` (`aca.py:487`): two `np.linalg.qr` of the
concatenated per-basis factors, B core GEMMs, then two `np.linalg.eigh` of
(K,K) Gram matrices. There is no SVD in it -- the SVD is one level out, in
`SharedLR.combine`, the per-material view. WP5 named the fix and skipped it for
budget: replace the two big `(3n, sum K)` QRs with block Gram-Schmidt. What the
factors actually arrive with, which decides whether that works: `V_b` has
exactly orthonormal columns, and `U_b = Qu (u s)` (`aca_numba.py:194-210`) is
orthogonal but NOT orthonormal -- the singular values ride on its columns,
spanning ~1e4 -- so its orthonormal basis is a column rescale away and its own
R factor is `diag(s_b)`. Block Gram-Schmidt therefore needs no per-basis QR,
and more to the point **needs no LAPACK factorization at all** (GEMMs, norms
and projections only), so unlike the present fold it can run in a numba nogil
kernel on the existing pool. Worth ~2.8x on the ACA phase, against quadrature's
1.09x.

Two constraints on any rewrite, and one correction to this file's own record:
scipy's economic QR is 3-25x faster but silently corrupts 2-6 blocks per model,
and numba's `np.linalg.qr` returned freed memory (fixed in `c480c75`). Pooling
the fold was recorded here as 9x slower; that is wrong. The 9x belongs to
`_recombine`, the per-material recombination loop (`hop.py:283-284`). The
fold's own measurement is **0.6x of one thread** (`hop.py:236-238`,
`aca.py:84-86`), i.e. ~1.7x slower -- bad enough to explain why WP5 left it on
the main thread, but not the wall the 9x implied.

**The memory measurement round (M0-M7): what actually holds the bytes.**
`operator_stats` had computed the near / low-rank / bases split at every rung
since WP0 and `compact()` dropped it before the `bench-json:` line, so no run
had ever recorded it. With it recorded (topo_inclusion, default policy):

  unknowns   near GB  lowrank GB  total GB  near %  ndense   nlr  mean rank
    31,098      1.79        0.36      2.15   83.1     4141   2123      19.6
   117,120      6.22        3.33      9.54   65.1     6533  14569      17.7
   260,598     14.34       10.86     25.20   56.9    29561  39185      16.5

The near field is **exactly O(N)** (exponent 0.98 over the full 8.4x range,
55 KB per unknown); the low-rank part grows at **N^1.60** and the total at
N^1.16. The far-field growth is NOT compression failing -- the mean rank FALLS
with size -- it is the block count, growing at N^1.37.

Four things were then measured and three of them are negative results:

* **Partition tuning is not a lever: 1.18x, measured.** The block-count growth
  is real (`MAX_ADMISSIBLE_BLOCK = 1024` leaves 0.5 % of blocks at the cap at
  31k, 18.9 % at 117k, 32.8 % at 261k with 4,059 interactions split), and a
  structural model over the partition alone predicts 2.07x from raising the cap,
  dropping `ACA_MIN_BLOCK` and halving the leaf. The real harness delivers
  25.20 -> 21.42 GB (1.18x) for +22 % build time, at equal iterations. The model
  is wrong because it assumes every admissible block compresses to rank 16: rank
  grows with block size, and small far blocks do not compress enough to be worth
  factoring, so they fall back to exact storage and land BACK in the near field.
  That is also why halving the leaf (near bytes scale exactly with leaf size in
  the partition) moves the measured near field by only 1.14x.
* **Algebraic H^2 is REJECTED on measurement: 3.2x WORSE, at two sizes.**
  Predicted from the flat-H factors themselves (the fold's Qu/Qv are the block
  bases; an H^2 cluster basis is their union over the blocks of one cluster --
  `shared_subspace`'s computation one level up). Far field at 31k / 117k:
  flat 1.05 / 8.16 GB against H^2 3.41 / 26.48 GB. The coupling matrices are
  88-89 % of it. Mechanism: sharing a basis across all of a cluster's
  interaction directions inflates its rank from ~17 to ~175, and with B = 6
  material bases each block then pays B k_t k_s instead of B k_u k_v -- 107x per
  block -- which swamps the 3x saved on basis storage. Cluster-to-summed rank is
  0.618 at 31k and 0.493 at 117k, so sharing does work; it cannot pay for
  quadratic coupling. (The nestedness check fails too, median residual 2-3e-1,
  but that only says ACA bases are not nested -- expected, since ACA picks
  pivots per block.) **The consequence for the far field is the useful part:**
  an O(N) scheme pays here only if its coupling operators are SHARED across
  blocks rather than stored per block -- translation-invariant M2L fixed by
  interpolation order, which is exactly Trial B's Chebyshev bbFMM and is not
  something an algebraic conversion can produce.
* **float32 storage is a free 2x.** Rounding the stored factors through float32
  in place leaves the operator error unchanged to four digits at every rung
  (2.791e-5 / 2.891e-5 / 1.397e-5 float64 against 2.791e-5 / 2.891e-5 /
  1.398e-5 with near AND low-rank single), because float32's 6e-8 sits 500x
  below the 1e-4 block tolerance the compression already spends. Saves 12.6 GB
  of 25.2 GB at 261k, including the cancellation-sensitive unit-translation
  vector. This is larger than every partition constant combined and costs no
  build time. What it still needs before adoption: a mixed-precision matvec
  (float32 storage, float64 accumulation) and a re-based determinism gate.
* **Out-of-core is nearly free once warm.** Spilling every flat buffer to disk
  and mapping it back: in-RAM 82.2 ms, mmap warm 82.9 ms (1.0x), mmap cold
  3819 ms (46.5x, 2.50 GB/s effective; the page cache could not be purged
  without sudo, so that is an optimistic bound). The matvec agrees bitwise --
  mapping is arithmetically transparent, and `U_flat`/`V_flat`/`D_flat` are
  already contiguous buffers, so this needs no new code path. The dense blocks
  are a linear scan and are the right thing to spill; the low-rank part is a
  gather and should stay resident.

Two robustness items the round exposed: `ADMISSIBILITY_ETA = 3` crashes the
build outright (`aca.py:141`, `SharedLR.combine`'s k x k SVD raises
`LinAlgError: SVD did not converge` with no fallback -- weaker admissibility
gives higher-rank, worse-conditioned cores); and every `(field patch, source
patch)` pair builds its OWN cluster tree and partition, so admissibility is
never tested across the union of the geometry (~48 independent partitions on
this model). One global tree is the prerequisite for any FMM anyway.

**Point sources for the far field: measured, works, does not pay yet (1.06x).**
A 6-point symmetric rule is 5.8x cheaper than the exact triangle integration on
the T kernel at the ACA's own row/column granularity (inside numba; a
Python-level comparison misleadingly shows 1.2-1.9x), and it is accurate
enough: at 261k, 99.1 % of far-field element pairs sit where 6 points meet
1e-4, 0.5 % need 12, 0.3 % must stay exact. Work-weighted, the median
far-field pair sits at r/h = 37 and only 3.2 % of the work is closer than 8.
End to end it buys 1.06x on total wall, because of the fold above. Revisit
after the fold: it is then worth 1.35x of what remains, and it is the P2M stage
an FMM needs anyway. The gated prototype (~460 lines, verified against the
closed form at two Poisson ratios) is in the session scratchpad, not the tree.
One trap recorded: an admissible block can have a source element LARGER than
its own separation on a graded mesh (min gap/h_src 0.73 at 261k), because
admissibility tests `min(diam) < eta dist` and never sees h_src -- so a rule
order must be chosen per block from its own gap/max(h_src), not globally.

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
