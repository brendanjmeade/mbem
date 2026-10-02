# Backlog

The one status document for this repo. History and measurements live in
`git log` (each commit message carries its numbers); rules live in the root
`CLAUDE.md`. Trunk: `src/mbem`, installed with `pip install -e .`. `src/clq`
and the two kernel copies are frozen oracles; the two closed lines are recorded in
`docs/{ddbem,fbem}-findings.md`. Gates: `python tests/run_all.py`, 43 of them.

**The tree was repackaged on 2026-10-01** (`bf2077c`..`f38d2dc`). Paths in
entries below this line predate it: `msd/mbem` is now `src/mbem`, `msd/verify`
is `tests/gates/mbem`, `msd/examples` is `studies/mbem`, and
`moss/mollified_kernel` is `src/moss_kernel`. The paper (`moss/`) and its
public package (`medt_paper/`) left the repo for
`~/Desktop/moss-org-paper-archive/`; they are in history and under tag
`medt_paper-vendored-2026-09-17`. Nothing numeric changed in the move -- the
17 frozen oracles are pinned by sha256 and all 17 were byte-identical
afterwards, and the gate stdout was diffed line by line against a baseline
taken before the first commit.

Three things the migration established that are worth knowing before changing
the tree again:

* **Two independent kernel copies, two package names.** `mollified_kernel` and
  `moss_kernel` are compared entrywise by the gates, and only `moss_kernel`
  defines `analytical_eigenstress_kernel` / `eigenstress_batch`. A swap between
  them is the one failure in this repo that stays GREEN: the parity residual
  just slides from ~1e-12 to ~1e-16, which no tolerance rejects.
  `verify_oracle_provenance` exists for that and pins each oracle by resolved
  path AND sha256. Three files had hardcoded the prefix `mollified_kernel.` and
  would have silently resolved to the wrong copy once the package was
  installed.
* **Two parity clauses are currently vacuous**, and predate the migration:
  `verify_dd_pairing` [d] "msd scalar vs moss scalar" is exactly `0.00e+00`
  and `verify_eigenstress_exact` [a]'s moss residuals are ~1e-16, because for
  those functions the two copies are byte-identical CODE. They still trip if
  the copies diverge, so they stay -- but the parity that is genuinely
  independent is `src/clq`'s, which derives its kernels separately. Do not cite
  the first two as independent evidence.
* **Gates are spawned as subprocesses, by file path, deliberately.** Four gate
  filenames exist in two suites each; the two kernel copies must not contend
  for one module identity; and several gates rebind `defaults.X` around a
  clause. `tests/run_all.py` also pins a gate COUNT per suite, because
  discovery is a glob and a glob pointed at the wrong directory yields "0 / 0
  PASS" and exit 0 -- a green empty suite, the one result a runner must not be
  able to report.

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
  ~15x per order), and only with eps room (`docs/ddbem-findings.md`).
* Lean: docstrings state the rule; no dates or review numbers in code; no probe
  scripts in the tree; extend a gate before adding one; no new documents.

## Open — measured, not yet acted on

**THE BOX SIDES CARRY 1.7 % TOP-SURFACE ERROR, AND REMOVING IT COSTS 4.5 %.**
`docs/fbem-findings.md` recorded an unclaimed result from the closed
force-element line -- that removing the mesh-size discontinuity at the top rim
by refining the box SIDES (not the top) was nearly free and worth a lot, was a
property of the mesh rather than of the formulation, and had never been tried
with the direct BIE. It has now been tried, and it transfers: measured on
topo_inclusion with the TOP MESH HELD FIXED at 9 km, so the only thing varying
is the sides, against the edge_side = 10 km run as reference.

    side   far  unknowns   +dof   host_top L2 (vs 10 km sides)
      80    80     31,098   +0.0%   1.657e-02      <- the shipping default
      80    40     31,812   +2.3%   1.696e-02      base only: NO effect
      40    80     32,490   +4.5%   5.867e-03      sides only: -65 %
      40    40     33,204   +6.8%   6.137e-03      both: no better than sides
      20    20     41,592  +33.7%   1.954e-03
      10    10     74,934 +141.0%   reference

It is ENTIRELY THE SIDES, exactly as the closed line said: the base at 80 -> 40
moves nothing (-2.3 %, i.e. noise) for its 2.3 % of extra unknowns, while the
sides at 80 -> 40 remove 65 % of the error for 4.5 %. Conditioning improves too,
measured with the dense backend: 5.468e5 -> 3.275e5, **1.67x better**. The
original claim was 22 % and 2.2x; the L2 effect here is larger and the
conditioning effect smaller.

**Why this deserves attention out of proportion to its size.** 1.7e-02 is
**two orders of magnitude larger than the far-field operator tolerance** this
program spends most of its effort on (1e-4), and ~80x the 2e-4 that the
topography decomposition inherits from cancellation. The default mesh has been
the dominant error term on the quantity the showcase figure draws, and nothing
in the gate suite was looking at it -- every parity clause compares an operator
against another operator on the SAME mesh, so a mesh-induced error is invisible
to all of them by construction.

**NOT changed: the default.** `edge_side = 80` is what every committed
`bench-json:` baseline and every number in this file was measured at, so moving
it would silently invalidate the ladder. The knob is now exposed
(`mbem.cases.topo_inclusion.build(edge_side=, edge_far=)`, defaults unchanged)
and `configs/side_grading.py` reproduces the table above. Changing the default
is a decision to take deliberately, with a ladder rerun attached.

Untested, and the reason the closed line stopped short: true GRADING of the
sides toward the rim rather than uniform refinement. It would cost less than
4.5 % for the same gain, but `make_vertical_panel_eq` takes a scalar
`target_edge` and `inclusion_mesh.py` is a frozen oracle, so it needs a new
conforming panel mesher -- and `docs/fbem-findings.md` warns that direct-BIE
conditioning is far more fragile under grading (5.7e19 at beta = 3).

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

**Why the fold does not pool: measured, and it is not the buffer lock.** This
file recorded that pooling the fold is slow because OpenBLAS serializes on its
buffer lock, so a fix "has to avoid LAPACK contention rather than parallelize
around it". Both halves of that are wrong, and the second sent the named fix at
the one part that was never the problem.

A LAPACK-free thin QR (modified Gram-Schmidt, one reorthogonalization, rank
detection by residual norm, stored row-major so every dot and axpy is
contiguous) matches `shared_subspace` to 2.3e-15 with identical ranks, needs no
BLAS call at all -- and as a bare nogil kernel on the existing pool it scales
**11.3x at 16 threads on large blocks and 10.3x on small ones**. The pool is
fine. What does not scale is the GIL-held numpy WRAPPER around the
factorization, and wrapping that same kernel in it reproduces the collapse
exactly (large blocks 3.15x at 4 workers then 0.22x at 16; small blocks 0.28x
at 4 and 0.02x at 16), as does the LAPACK version (0.13x at 16).

Per-fold profile, which explains the shape:

    stage                    m = 3072        m = 384
    QR (already nogil)    37.9 ms  86.8 %   4.6 ms  45.0 %
    _unit_gram x2          2.3 ms   5.3 %   2.3 ms  21.9 %
    _principal (eigh) x2   2.1 ms   4.8 %   2.0 ms  19.7 %
    cores, proj, final     1.4 ms   3.1 %   1.4 ms  13.3 %

The wrapper costs ~5.7 ms per fold at ANY block size, because it works on
(K, K) matrices with K = sum of the per-basis ranks (~102), independent of the
block's own size. So the GIL-held fraction is 13 % on a 1024-element block
(Amdahl ceiling 7.6x) and 55 % on a 128-element one (ceiling 1.8x), and the
production partition is dominated by the small end. Measured scaling is worse
than Amdahl in both cases, so it is convoy, not just serialization.

The fix that follows is therefore NOT the one WP5 named: replacing the QR
attacks the part that already scales. It is to put the WHOLE fold in one nogil
kernel -- the Gram matrices, the truncation, the core products and the final
`Qu @ Wu` with it. The only hard piece is `_principal`'s symmetric
eigendecomposition of a (K, K) Gram; numba's LAPACK is not trustworthy here
(its `np.linalg.qr` returned freed memory, `c480c75`), so that wants a
hand-written cyclic Jacobi, ~3x slower serially than `eigh` and fully nogil.
Projected: fold 143 s -> ~14 s at 261k, ACA phase 208 -> 79 s, assembly
245 -> 116 s, i.e. **~2.1x on assembly**.

NOT SCHEDULED. Assembly is not the binding constraint -- memory is, and A1
halved it -- and an interpolation far field has no ACA factors to fold, so this
code is the first thing Path C deletes. Recorded so the next person does not
re-derive the wrong diagnosis.

Two constraints that do still hold for any rewrite: scipy's economic QR is
3-25x faster but silently corrupts 2-6 blocks per model, and numba's
`np.linalg.qr` returned freed memory (fixed in `c480c75`). And one more
correction: pooling the fold was recorded here as 9x slower. That 9x belongs to
`_recombine`, the per-material recombination loop (`hop.py:283-284`); the
fold's own figure is 0.6x of one thread (`hop.py:236-238`, `aca.py:84-86`).
`_recombine` is numpy under the GIL too, so its 9x is likely the same convoy
and its stated cause wants re-measuring before it is trusted.

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

**The far-field decision is made: adaptive octree + Chebyshev bbFMM (B, C).**
Three gates were measured before committing the weeks, and two of them moved
the plan.

*B's stated justification was wrong.* "One global tree" was costed at 1.76x on
the near field; measured over four tree configurations at fixed leaf and
admissibility, GLOBALITY buys 1.04-1.22x (and is worse two-sided at scale 1)
while the OCTREE BOX SHAPE buys 2.2-2.7x. The 1.76x also held `ACA_MIN_BLOCK`
fixed across configurations -- a rule about ACA's certificate, which a bbFMM
does not have, and which fired on 0 % of the baseline's bytes and 76 % of the
octree's. The only defensible reason to make the tree global is that a SHARED
translation-invariant M2L table requires one tree. That is sufficient; the
near-field argument is not.

*C's gate passes.* `K(eps) = K0 + eps_j^2 K1` holds 1e-4 on 100 % of admissible
blocks and 100 % of far-field work (13x headroom); one pass fails on 62.5 %.
K0 is the classical Kelvin kernel (2.6e-12 against the eps -> 0 analytic
stack), K0 and K1 are translation-invariant to 1.7e-8, and K1 is homogeneous of
exactly K0's degree minus 2 -- so BOTH M2L tables are shared across a level and
eps enters only as a per-source weight `w_j eps_j^2`. Per-element eps is a
non-issue; the constraint is eps/h. A per-leaf scalar eps is ruled out (15.8x
spread inside one 96-element leaf). Order p = 6 (U) and p = 8 (T), in the
Frobenius norm -- at p = 8 the max-entry error is 1.3-3.0e-4, so 1e-4 is NOT
met entrywise. The 18 -> 6 component lever holds: `D_ijm = C_qmkl dU_ik/dy_l`
to 2.8e-16 at two Poisson ratios, so T reuses U's table by differentiating the
Chebyshev basis, at ~2x the error and still inside 1e-4 at p = 8.

*The near field is not the constraint, and the 58-vs-95 GB question was a
category error.* Measured on the real graded geometry with real U/V/W/X lists,
two independent implementations agreeing to the integer pair count at ncrit
64/128 and within 1-3 % at 16/32 (self-checks: U+V+2W = N^2 exactly,
4,467,852,964 = 66,842^2; U symmetric; brute-force adjacency):

  scale 3, KiB/unknown f64        4M unknowns, f32
  ncrit    U     W     X   U+W+X    U only   U+W+X
     16  2.01  1.39  1.00   4.40    3.8 GiB   8.4 GiB
     32  4.69  1.92  1.57   8.18    8.9 GiB  15.6 GiB
     64  7.45  4.88  3.49  15.83   14.2 GiB  30.2 GiB
    128 19.25  8.49  6.65  34.38   36.7 GiB  65.6 GiB

With the ~3 GB far field that is ~13 GB of 128 at 4M unknowns. The earlier
figures compared different sets: the H-matrix near field IS the FMM's U+W+X,
confirmed on the live operator to 5 % (16.4 M pair-keys against 15.6 M octree
U+W+X, U alone 6.9 M), because an H-matrix has no M2P/P2L and must store every
mixed-level pair densely.

**ncrit = 32.** 16 saves 5 GiB and doubles M2L (335,766 V box pairs against
165,698); 64 nearly doubles near-field kernel pairs (12.7 M -> 23.2 M) for a
38 % M2L saving; 128 is worse again. V-list length is mean 28-34 and max 81-84
against 189 for a uniform tree, and the U list is 12-14 boxes not 27 -- these
are 2-D surfaces in a 3-D tree.

**W and X need M2P/P2L, and not because of the grading.** W+X is 0.75-1.2x the
U list; punting both to direct evaluation costs 1.8-2.2x on the near field and
more on flops, because a W record is fat (147 element pairs against 55.6 for a
U record: a W source is an internal box with a subtree under it). Removing the
element-size constraint entirely still leaves W+X at 0.83x U, so the mixed-level
adjacency comes from refinement contrast between surfaces (fault and inclusion
fine, host coarse), not from the big host triangles. W and X have identical
distinct-pair counts -- exact transposes -- which is a free correctness check.

**The interpolation domain is a PLACEMENT question, and the fix prescribed here
was wrong.** Placing an element at the finest level whose box edge is at least
its own size leaves a max protrusion of 0.53 box edges, and Chebyshev
interpolation is invalid for any source outside its box. This file prescribed
"enlarge each box's interpolation domain to the bounding box of its own
contents, keep the M2L table on the nominal boxes" -- that is `fmm.py`'s
`domain="canonical"`, and it is IDENTICALLY the plain cube (measured 3e-15 at
every p through 12: the extent lattice reproduces every polynomial of degree
< p per axis and a cube Chebyshev weight is one, so extent-P2M composed with
the transform IS cube-P2M). It fails the operator clause. The paragraph also
ruled out the placement rule on the strength of STRICT containment (52 %
pinned, 284 KiB/unknown) and never priced a modest safety factor, which is the
thing that works.

Measured at the operator, fault zone at refine 1, p = 6 (U) / 8 (T),
far-isolated max over 9 test vectors against `FMM_OPERATOR_PARITY` = 2e-4:

  extent, safety 1.0 (today)            1.082e-04   0/9 over
  cube,   safety 1.5                    7.166e-05   0/9      0.66x the extent
  extent, safety 1.5                    9.490e-05   0/5
  cube,   safety 1.0                    6.654e-04   4/5      FAILS by 3.3x
  cube + inflate (1.25, 1.0), safety 1.5  2.393e-04 1/5      FAILS

So the bare cube fails, and the cure is not the domain but
`OCTREE_PLACEMENT_SAFETY` 1.0 -> 1.5, which cuts protrusion 0.333 -> 0.200 and
takes the cube from 6.15x worse than the extent to 0.66x, i.e. BETTER. The
control that proves the mechanism: the extent barely moves under the same
change (1.082e-04 -> 9.490e-05), having no extrapolation to cure. Raising p
instead is worse and dearer -- p = 8/10 clears the bar at only 1.17x margin for
2.6x the M2L work and 3.8x the table memory.

**Inflating the domain is actively harmful, which was not obvious.**
`inflate=(1.25, 1.0)` nearly contains (max|xhat| 1.114) and is 3.3x WORSE than
not inflating at 1.393. Two non-adjacent boxes are 2 edges apart centre to
centre while their lattices span +-f/2 edges, so M2L separation is
(2 - f_s/2 - f_t/2) cos(pi/2p) and vanishes near f = 2: at topo's containment
factor 2.055 the two node lattices interpenetrate (separation 0.027 edges,
(eps/r)^2 = 13, error 9.81 at p = 6). Containment is NOT what buys accuracy --
extrapolating mildly with the full V separation beats interpolating with a
degraded one.

**Why the cube is the only O(1) far field.** Each far pass is homogeneous, so a
level-l table is a level-l' table times a power of two: 258 of the cube's 316
distinct transfer keys serve more than one level, up to 5 each, and 316 is the
classical bbFMM transfer-vector count -- O(1) in N. The extent shares nothing
(0 of 21,740 keys serve more than one level; 686 of 702 V box pairs distinct on
the fault zone), so its table count is proportional to N: ~3.3 M V box pairs
and ~13.8 TiB at p = 6 at 4M unknowns, which means it cannot precompute M2L at
all and every V pair re-evaluates the kernel -- the per-block coupling that made
algebraic H^2 3.2x worse than flat H. Shared tables at 240 transfer vectors are
1.00 GiB at p = 6 and 5.63 GiB at p = 8, independent of N.

The price is the near field: safety 1.5 costs 2.28x element pairs on
topo_inclusion (1.64x on U alone) and buys 38 % fewer V box pairs and 24 %
fewer tables; ncrit does not pay it back (ncrit 16 + safety 1.5 measures
6.455 M against ncrit 32's 6.579 M). At 4M that is ~35.6 GiB near + 1-5.6 GiB
far, ~37-41 GiB of 128.

**Settled at the target geometry, and it is the X LIST, not the domain.** The
"dense reference" that capped every operator number at the 2.6k fault zone was
never needed: every term of the far-isolated metric is a matvec, and
`matvec_exact_rows` over ALL rows is matrix-free -- 2.2 s and 2.0 GiB at 31,098
unknowns where a dense matrix is 7.2 GiB. Validated against the dense operator
where dense is affordable (`A_far + A_U + D = A` to 0.0e+00 on four different
near/far partitions) and against the gate's own [d] row.

With that, topo_inclusion itself, p = 6/8, all rows, against 2e-4:

  config                          31,098      117,120     260,598
  cube,   safety 2.75           5.13e-05 ok  2.37e-04 OVER  3.06e-03 OVER
  extent, safety 2.75           8.47e-05 ok  4.25e-05 ok    1.04e-03 OVER
  extent, safety 1.0 (default)  7.20e-04 OVER

So NOTHING passes at scale, the incumbent default included, and a safety factor
tuned on one mesh does not transfer. The cause is the X (P2L) list, whose
target is a FIELD POINT, so its separation is not the V list's. Replacing X by
its exact value: scale 1 cube 1.5, 4.78e-03 -> 9.63e-05 (50x); scale 2 cube
2.75, 2.37e-04 -> 9.08e-05 (2.6x); scale 3 cube 2.75, 3.06e-03 -> 1.12e-04, a
27x cure that turns 15x-over into passing. Negative control: where the X margin
is already 2.178 the same substitution moves nothing (1.02x). Over the 15 topo
configurations with operator numbers, X margin >= 2.026 passes 5/5 and <= 1.487
fails 10/10; protrusion leaves nothing residual once the margin is accounted
for. Safety only ever worked by moving that margin -- at 2.75 it is 2.178 at
scale 1, 1.400 at scale 2 and 1.000 at scale 3.

**The X rule, LANDED** (`la/octree.XMargin`, `defaults.FMM_X_MARGIN = 2.0`,
gated by `verify_octree` [g]). The test travels as one object because its three
parts -- source geometry, target interpolation domain, threshold -- have to;
`InteractionLists(tree)` with no margin still gives the domain-blind lists, so
no existing caller changed, and `FmmTree` now builds its domains BEFORE its
lists and hands over its own `tgt_dom`. One difference from the table below,
which substituted the WHOLE entry: the traversal descends, so a refused entry
comes back as deeper X entries wherever the smaller target domain clears the
threshold and only what cannot descend goes direct -- at scale 3 extent safety
2.75, 5 refusals release 8,685 element pairs of which 6,660 reach U and the
rest return as 6 deeper entries. The operator error is the same to five digits
either way (8.6754e-05 at x1 extent 1.5, against the sweep's 8.6754e-05), so
the descent is the cheaper form of the same fix. Endpoints gated: threshold 0
reproduces the domain-blind lists entry for entry and the operator BITWISE
(max|difference| 0.000e+00 on the fault zone at cube x1.7, where the rule
otherwise moves 89 of 144 entries); threshold infinity empties X into U over
exactly the element pairs X held. The conservation identity `U + V + 2W = N^2`
is exact with the rule on, off, and at both endpoints, on three refinements and
three domains. The rule is one-sided, so U/W/X stop being exact transposes as
soon as it bites -- W is M2P, whose target is a field point with no domain to
be inside.

The specification as measured:

**B IS CLOSED** (`octree.py`, `fmm.py`, `verify_octree.py`, `verify_fmm.py`).
Settled parameters, each measured not chosen: `OCTREE_NCRIT = 32`,
`OCTREE_PLACEMENT_SAFETY = 2.0`, `FMM_X_MARGIN = 2.0`. Both interpolation
domains pass at 260,598 unknowns with the X rule on, at ~1e-4 iso and ~7e-6
naive, so the domain is C's to choose on the shared-M2L-table count alone. At
the shipping safety 2.0 that is cube 514 / 1,048 / 1,536 (level, offset) keys
across the three scales against the extent's 10,992 / 50,064 / 132,020, and
after the level fold 218 / 316 / 316 cube TABLES against the extent's nothing
shared. The 486 / 1,010 / 1,450 and 8,316 / 37,303 / 58,518 this paragraph
carried are reproducible but belong to safety 2.75, which 2.0 superseded.

Placement was settled on 64 full FMM matvecs over ~9 h: safety {1.0, 1.5, 2.0,
2.5} x {cube, extent} x all three vectors at scales 1 and 2, and 2.0 x both
domains x all three at scale 3. Worst margin over both limits and both domains:

  safety   scale 1      scale 2      scale 3
  1.0      FAIL 0.22x   FAIL 0.36x   --
  1.5      pass 2.6x    FAIL 0.55x   FAIL 0.62x (cube; extent passes 1.08x)
  2.0      pass 2.7x    pass 1.3x    pass 1.1x
  2.5      pass 2.8x    pass 1.8x    pass 1.1x

**1.5 is exactly the trap this program keeps falling into**: it passes scale 1
by 2.6x and fails scale 2, and the failure is NOT monotone in N -- at scale 3
the cube fails while the extent passes. No single mesh, and no extrapolation
from two, would have caught it. Scale 2 also needs BOTH limits: safety 1.5 cube
reads iso 5.5362e-05 (passing) against naive 6.9074e-05 (failing), which is the
dual limit earning its place. And the binding vector is the TRANSLATION, not a
Gaussian -- its far field is 1.25x its own ||A v||, so it is the only one that
stresses the naive limit.

2.0 is a bound rather than another fit because protrusion bounds it: the cube's
failure is ordered by protrusion alone (passes 0.329, fails 0.372 and 0.405,
so its extrapolation limit is in (0.33, 0.37)), and `prot <= c/safety` with
c <= 0.67 is a THEOREM, not a measurement -- placement is by centroid and size
is the longest edge, so |v - g| = |(v - b) + (v - c)|/3 <= 2L/3 with equality
only for a degenerate triangle. Worst c measured over safety 1.0-3.0 at three
scales is 0.608. Safety 1.5 guarantees only 0.447, outside the bracket; 2.0
guarantees 0.335, inside it. `OCTREE_PROTRUSION_C` is gated over a sweep that
includes the shipping default, so a default chosen on one mesh cannot quietly
stop controlling protrusion on another.

**The scale-3 margin is 1.1x and placement cannot widen it.** Safety 2.0 and
2.5 have the same absolute error to four digits there (3.4743e-05 against
3.4740e-05). What is left at 260,598 unknowns is the interpolation ORDER:
`FMM_OPERATOR_PARITY = 2e-4` is nearly saturated by it. **C inherits 1.1x of
room, not 2x.**

Near field at safety 2.0: 127.0 / 113.6 / 106.0 element pairs per unknown at
the three scales, projecting to 424.0 M pairs and 28.4 GiB at 4M unknowns
against safety 1.0's 266.6 M / 17.9 GiB (+59 %) and 2.5's 780.6 M / 52.3 GiB
(+84 % for no accuracy). The rate FALLS with N at 2.0 (127 -> 114 -> 106) while
1.0's rises (52.8 -> 54.0 -> 66.6), so the penalty shrinks with scale.

Verification, stronger than the sum identity the gate runs: the full ordered
element-pair COVERAGE-COUNT matrix, every entry exactly 1, over 100 list builds
(5 meshes x 3 domains x 5 thresholds x 2 safeties), sums exact to 904,265,041
and 13,853,760,804 at scale 4. Plus two properties the gate does not check --
V and W entry sets are bit-identical to the domain-blind traversal, so the rule
cannot leak into M2L/M2P, and the diagonal and every AABB-overlapping pair stay
in U, so a singular pair can never reach a P2L entry. The no-op endpoint is
bitwise against a pristine `git archive` of the pre-rule commit.

**The X rule, located.** When forming an X entry, take
`rho = min over the vertices of the source box's resident elements of
max_d |v_d - c_d| / h_d` against the TARGET box's interpolation domain
`(c, h)`; if `rho < 2.0` do not emit it -- keep descending the target and
re-test, and where it cannot descend send that `res(a) x res(b)` block to
direct. X only; V and W untouched. Five lines at `octree.py:422`
(`_descend_sub_res`), where the descent and the U fallback already exist; it
closes the open item that module's docstring already names. One structural
consequence: the lists are built on nominal cubes and are domain-blind, but the
margin is not -- the same box pair reports 1.000 on the cube and 1.450 on the
extent -- so `InteractionLists` has to be told which domain will be
interpolated in.

Measured, clause-[d] iso over three test vectors against 2e-4, real
re-partitioned matvecs:

  config                no rule     with rule            all X exact
  x3 cube saf 2.75    5.985e-03   2.768e-04 FAIL 1.4x    2.747e-04 FAIL
  x3 cube saf 2.00    5.953e-03   1.866e-04 PASS         2.420e-04 FAIL
  x3 extent saf 2.75  1.223e-03   3.315e-04 FAIL 1.7x    3.290e-04 FAIL
  x1 cube saf 1.5     4.777e-03   1.038e-04 PASS         9.625e-05
  x2 extent saf 2.50  3.215e-04   8.631e-05 PASS         1.466e-04
  x2 cube saf 2.75    2.785e-04   8.360e-05 PASS         1.354e-04

Price at scale 3: 3 entries, 540 element pairs, 0.0009 % of the near field
(22 entries / 18,446 pairs / 0.067 % on the cube at safety 2.0). Moving the
WHOLE X list instead costs +54.3 % and still fails.

2.0 and not the fitted minimum: the located value moved 1.45 -> 1.90 -> 1.65,
once per newly measured configuration, and 1.65 clears the entry it must catch
(rho 1.6437) by 0.4 %. 2.0 is a superset everywhere, sits on the plateau the
sweep found (error unchanged from 1.65 to 4.0 in all six instrumented
configurations), and `rho > 1` is the well-posedness condition -- the source
strictly outside the interpolant's own domain -- so 2.0 is that plus margin.
**`eta >= 0.30` is NOT an equivalent form and must not be used**: at x2 extent
saf 2.50, where the failure is X-caused and `rho < 1.65` cures it
(3.215e-04 -> 8.631e-05), `eta >= 0.30` selects ZERO entries.

The rule is NECESSARY, NOT SUFFICIENT. At x1 cube safety 1.0 (protrusion
0.527) no threshold passes -- even all-X-exact leaves 1.243e-03 -- so placement
still has to control protrusion independently. Only ONE configuration passes at
scale 3: cube at safety 2.0. With the rule in, `OCTREE_PLACEMENT_SAFETY = 1.0`
is what is left of B: what is wrong at safety 1.0 on the cube is 0.527 box
edges of protrusion at P2M, which no X threshold touches.

**The gate's own metric was penalising the fix, and is corrected.** The
far-isolated error divides by `||A_far v||` of the CURRENT partition -- which
is exactly what a change under test moves. Demoting 3 X entries at 260,598
unknowns, 540 element pairs of 57 million, drops that denominator by 36 %, so a
fix leaving the absolute error untouched reads as a 1.6x REGRESSION; and the
x3 cube safety-2.0 "pass" and safety-2.75 "fail" have the SAME absolute error
(2.9048e-05 against 2.8999e-05) on denominators differing by 49 %. Tuning
against a denominator that moves when you touch the operator is how the
threshold wandered. `_far_operator` now states that the denominator must come
from one fixed partition, and the operator clause gates a SECOND limit,
`FMM_OPERATOR_PARITY_NAIVE = 5e-5` over `||A v||` -- weaker, but a pure ratio
of the operator to itself that no change to the near/far split can move.
Neither alone is safe: the naive one is too loose, the isolated one lets a
variant pass by shrinking its own far field.

The most useful number the study produced: once X is handled, the absolute
error over `||A v||` is **uniformly ~1e-5 at every scale and both domains**
(8.4e-6 to 1.24e-5), against up to 5.8e-4 before. The operator is in better
shape than the isolated metric has been reporting.

With X exact so nothing else contaminates it, the cube's own extrapolation
limit is protrusion 0.33-0.35 box edges (f_src 1.66-1.71), and the safety
factor that guarantees it on topo's grading is 2.0, not 2.75 -- measured
protrusion 0.273 / 0.286 / 0.285 at the three scales, and the sawtooth bound
prot <= c/safety with c <= 0.67 makes it a guarantee rather than a coincidence.
Safety 3.0 is the ceiling: it leaves the fault-zone model at 100 % U list, i.e.
no far field at all.

**The far-field byte projection in this file is 36x low.** "316 keys ~ 1.0 GiB"
counted ONE U table at p = 6 with no eps pass. At the shipping orders with
`FMM_EPS_TERMS = 2` a transfer key holds U(p=6) 2 (3.216)^2 = 6.4 MiB PLUS
T(p=8) 2 (3.512)x(9.512) = 108.0 MiB, so 114.4 MiB per key. At 4M unknowns the
the extent is ~98 TiB. **The ~295-318 GiB this paragraph carried for the cube
was itself wrong, by 8.7x, and by ignoring the level fold established 200 lines
above.** 114.4 MiB is per stored TABLE, and the stored tables are the distinct
transfer OFFSETS, not the (level, offset) keys: the level folds out exactly, so
514/1048/1536 is a count of keys and not of tables. The offset count saturates
at 316 = 7^3 - 3^3, measured 218/316/316 at safety 2.0, 240/316/316 at 1.5 and
316/316/316 at 1.0 over the three scales, and 316 is a combinatorial CEILING
rather than a fit -- V entries are same-level by construction and their parents
adjacent, so the offset lies in [-3,3]^3 minus [-1,1]^3. It is the one number in
this program that needs no extrapolation. The cube's whole table set is
therefore **35.3 GiB at float64, 17.7 GiB at float32, independent of N**, or
17.6 / 8.8 GiB stored as the 14 (U) + 26 (T) distinct scalar p^3 x p^3
geometry-only tables, which also makes the table material-INDEPENDENT instead of
one copy per region material. The extent is untouched and still O(N): it cannot
precompute M2L at all.

So the 18 -> 6 differentiated-basis lever is an optimisation of ~3.2x, NOT a
precondition for fitting in 128 GiB, and the claim that it "is now load-bearing"
was an artefact of the same arithmetic. Its identity is exact -- verified
symbolically in all 27 components over both eps passes at symbolic mu/lam, a
stronger statement than the 2.8e-16 float check this file cites, which is
reproducible from nothing in the tree -- but differentiating an interpolant
costs 1.4-2.9x of accuracy at equal p, and the 1.1x operator margin at scale 3
cannot absorb that, so taking it means shared p = 9 and it is then worth ~1.57x.
Price the order change before writing the differentiated P2M: the order is the
expensive half.

**The fold's trap is the homogeneity DEGREE, and it is silent.** The U far
passes are homogeneous of degree -1 and -3, the T passes of -2 and -4: every T
term carries one extra factor of d in the numerator, so T's degree is the radial
power minus one, not the radial power. Using the power for T rescales a reused
table by exactly 2x too much per level of reuse (4x, 8x, 16x further out),
measured rel 5.00e-01. The fold is a power of two and therefore BITWISE, so gate
it with `np.array_equal` and not a tolerance -- a factor of 2 is invisible to
every tolerance in this file.

**The largest single term at 4M is not the far field, it is the preconditioner,
and it appears in no budget here.** `PRECOND_BJ_CHUNK_DOF = 9000` and the
cluster block-Jacobi rung stores one dense `lu_factor` per chunk
(`preconditioner.py:428-431`), whose own comment states build memory is
O(N x chunk): at 4M unknowns that is 4e6 x 9000 x 8 = 268 GiB, 2.1x the
machine, against 30 GiB at chunk 1000 and 15 GiB at chunk 500. This rung is
also the only one whose iteration count grows with N, so the chunk is a direct
trade between the two, and it is unmeasured. Price it before C's far field:
C's own terms (35.3 GiB of tables, ~28 GiB of near field, 14.0 GiB of Chebyshev
stencil, 12.0 GiB of FGMRES workspace) fit in 128 GiB at float64 and this one
does not.

Near-field price, measured at 72 B per near element pair (`_near_blocks`
materialises a float64 block): safety 1.0 is 267M pairs and 18 GiB at 4M,
safety 2.0 is 424M and 28 GiB, safety 2.75 is 876M and 59 GiB. An X-exact
policy adds up to +54 % on top, though a real criterion moves only the pairs
below its threshold.

**C is startable, and its order of operations is not the one recorded above.**
Six things measured against the code rather than against this file:

1. **The table's win is the BLAS rate, not the algebra.** Per V pair at the
   shipping orders a dense table apply is 28.3 Mflop against 34.1 Mflop
   matrix-free -- 1.2x, i.e. nothing. What the table buys is getting the same
   arithmetic into GEMM: `_far_apply` issues nine un-optimised `np.einsum` calls
   (two of them 88 % of the cost, `fmm.py:339-340`) running at ~0.9 Gflop/s on
   one core, against 290-350 Gflop/s f64 and 650-685 f32 on the (1536, 4608) T
   shape once the batch exceeds ~64 columns. A matrix-free numba transcription
   reproduces `_far_apply` to 4.69e-15 and is already 27-31x faster for ZERO
   table bytes. **So numba first, table second** -- the reverse of the recorded
   plan, and the cheap half is also the one that cannot go wrong on memory.
2. **The table only pays key-major.** Per-pair batching gives 1.60 V entries per
   (pair key, offset): m = 1, which measures 14.6 Gflop/s bandwidth-bound
   against 336 at m = 114. Reaching the many-column regime needs one traversal
   covering many pairs, because `AssembledH` applies pair by pair and one
   operator matvec therefore does 1.6-8.1x redundant M2L applications and
   14-26x redundant upward-pass box visits against a single global traversal.
   Make the M2L structure a per-offset list of (target box, source box) from the
   first commit; adding the batch dimension afterwards is a rewrite.
3. **There is no matvec budget, so "close the gap" has no target.** Measured
   ~70-82 s per operator matvec at 2,592 unknowns (99.2 % M2L, 96-98 % of that
   the T kernel). Projected at 4M: ~30 h today, ~240 s with an f64 table and
   batched GEMM, ~71 s with f32 tables AND batching AND merged traversals.
   At the measured 23/37/42 iterations a 2 h solve needs ~144 s per matvec and a
   10 min solve needs ~12 s. State the budget as matvec seconds x iterations,
   beside the flat-H comparator (166.6 ms at 260,598 unknowns, `4ddee8c`).
4. **The drop-in claim is false on the default path.** `HBackend` and
   `AssembledH` default to `storage="combined"`, whose `_refresh_material_state`
   calls `pair.warm_views` (`hmat.py:219-222`), and `AssembledH.nbytes()` calls
   `pair.nbytes()`; `PairFMM` has neither, so the default raises `AttributeError`
   and the memory-reporting harness that produced every scale number in this file
   cannot run on an FMM operator at all. The working drop-in is the private
   `_shared=` kwarg at `storage="basis"`, which is what `verify_fmm` uses.
   Separately, no FMM operator has ever been built at `jump="calibrated"`, the
   production default -- every FMM number here is `jump="half"`.
5. **The shipping default domain shares nothing.** `domain="extent"` is the one
   domain whose table count is O(N). `domain="canonical"` is the interesting
   third option and is UNMEASURED with the X rule on: it keeps the extents for
   P2M/L2P/W/X and puts M2L alone on the shared cube lattice, so it would get the
   cube's 316 tables with the extent's W and X margins. Its recorded numbers
   (`fmm.py:87-90`) predate the X rule, which is the thing that was actually
   failing. Measuring it is a configuration change and no new code, and it
   should happen before any table is written.
6. **No gate reaches the far field above 2,592 unknowns**, so the 1.1x margin C
   inherits is unreproducible in-tree and every scale number here came from a
   scratchpad. The pair clauses also sit 2,000-23,000x under `FMM_PAIR_PARITY`
   on geometry orders easier than the offset (2,0,0) at which p = 6/8 was
   chosen. Extend clause [b] with a single-key sub-clause at that offset (6.3 ms
   at p = 6, 149 ms at p = 8), add `--backend fmm` to `bench_scaling.py` for the
   scale rung, and gate the table against the reference evaluator at ~1e-12
   behind an `m2l="evaluated"|"table"` switch -- never against the exact kernel
   at 1e-4. Table-vs-reference cannot be bitwise: the reference reduces over all
   source boxes of a V entry in one call, a table reduces per pair.

**C's first three steps, measured. The domain is CANONICAL.** With the X rule
on, canonical is the most accurate of the three domains at every scale AND
shares the cube's table exactly. topo_inclusion, safety 2.0, X margin 2.0,
p = 6/8, worst over all three harness vectors, far-isolated denominator from
ONE fixed domain-blind partition per scale so it cannot move with the domain
under test:

    scale  unknowns   domain      naive     far-iso  worst x  M2L tables  offsets
      1      31,098   extent    2.919e-05  1.018e-04   1.71x     10,992   10,992
      1      31,098   cube      1.838e-05  5.999e-05   2.72x        514      218
      1      31,098   canonical 1.460e-05  4.767e-05   3.42x        514      218
      2     117,120   extent    2.247e-05  1.685e-04   1.19x     50,064   50,064
      2     117,120   cube      2.264e-05  1.698e-04   1.18x      1,048      316
      2     117,120   canonical 2.043e-05  1.208e-04   1.66x      1,048      316
      3     260,598   extent    1.248e-05  1.078e-04   1.86x    132,020  132,019
      3     260,598   cube      1.513e-05  1.279e-04   1.56x      1,536      316
      3     260,598   canonical 1.513e-05  8.995e-05   2.22x      1,536      316

All nine pass both limits. The recorded "canonical is 2.6x better than the cube,
4.8x worse than the extent and still over FMM_OPERATOR_PARITY" (`fmm.py:87-90`)
was measured before the X rule and no longer holds: canonical is 1.3-1.4x better
than the cube and 1.2-2.1x better than the EXTENT. The mechanism holds under a
role-by-role look -- canonical contains exactly wherever the geometry enters
(P2M, L2P, C2L all max|xhat| = 1.0000) and keeps the extent's W and X margins and
X-rule refusals, extrapolating only in the M2L change of basis (m2c 1.50-1.55,
the same size as the cube's P2M 1.52-1.56). The offset count saturates at the
316 ceiling from scale 2 on, against the extent's 132,019 at scale 3.

Two cautions. Canonical creates NO headroom: at the binding scale (2) its margin
is 1.66x, and on the naive limit at scale 3 it is indistinguishable from the cube
(1.5125e-05 against 1.5128e-05), so what is left is the interpolation order. And
with safety pinned at 2.0 and only the domain varying, a GAUSSIAN binds the
far-isolated limit at every scale -- the translation binds the naive limit at
scale 3. The translation is the binding vector of the SAFETY sweep, not of every
sweep; quote both limits and all three vectors or none.

**The M2L evaluator is 85x faster and the table is now a second-order lever**
(`la/fmm_numba.py`, `PairFMM(m2l="numba")`). The reference's nine unoptimised
einsums ran at 0.8 Gflop/s on one core; the fused nogil kernel measures 67.0
Gflop/s at the real M2L shape (p = 8 T, nt = 512, ns = 3584, 209 Mflop): 266.8 ms
-> 3.12 ms, and 23.6 ms serial, i.e. 11.3x on one core and 7.6x more from the
pool. Parity against the reference evaluator: 7.8e-15 worst over three Poisson
ratios x both kernels x k = 1 and 3, 1.3e-14 through a whole pair traversal,
7.8e-15 on the W/M2P pair and 1.8e-15 on the X pair; end to end through the
operator 9.3e-16, 13 s against 817 s. Two entry points per kernel per the tree's
rule against a `parallel=True` kernel on a Python thread.

**So the shared table's price has to be re-asked, and the answer is that it only
pays key-major.** Measured per V pair at p = 8 T on an IDLE machine (a first
pass under load understated the table's peak rate by 2.3x and must not be
quoted), against the matrix-free numba kernel at 0.399 ms per V pair when seven
source boxes ride one call, 0.551 ms alone:

    batch m      1       8      32     114     512   ms per V pair
    table f64  0.499   0.182   0.092   0.064   0.061   (56.7 -> 461.3 Gflop/s)
    table f32  0.263   0.105   0.042   0.035   0.029  (107.7 -> 968.0 Gflop/s)

So the table crosses over between m = 1 and m = 8 and saturates by m ~ 114,
where it is worth 6.2x in f64 and 11.4x in f32 over the numba kernel. A
memoizing cache dropped into the existing per-pair loop would buy almost
nothing, because the measured batch width there is 1.60 V entries per (pair key,
offset) -- the loss-making end of that table. Key-major
batching is not an optimisation on top of the table, it is the precondition for
the table being worth any bytes at all -- and it needs ONE traversal over the
shared tree rather than one per pair. At the operator level scale 3 has 148,548
V box pairs over 316 offsets, 470 per offset on average, which is the regime the
table wants. Lifting the traversal above the pair is C's real next increment.

Table bytes re-derived from the shapes that run: T is (3p^3, 9p^3) = (1536, 4608)
per eps pass = 54.0 MiB, 108.0 MiB per transfer key, 33.3 GiB f64 / 16.7 GiB f32
over the 316 offsets, plus U's 2.0 GiB f64.

**THE TRAVERSAL IS LIFTED ABOVE THE PAIR** (`la/fmm.FarGroups`, `PairFMM` now
taking a sequence in either role, `AssembledH(..., _groups=)`). One traversal per
(region, kernel) replaces one per pair key: 6 traversals against 91 on the
fault-zone model, measured 66.6 s -> 16.6 s, **4.00x**, with the grouped operator
agreeing with the per-term sum to **8.7e-16**.

Why the grouping is exactly (region, kernel) and why it is exact:
`generate_system` couples a region's patches COMPLETELY -- `for q in
region.patches: for p in region.patches` -- and the term's scale is
`sigma(R, p)`, a function of the SOURCE patch alone, while the material is the
region's and one kernel takes one slot type (T reads u_p, U reads t_p). So sigma
folds into the source charge and one traversal computes every pair of the group.
That is a property of the equation generator, not of the FMM, so `FarGroups`
REFUSES a group whose coupling is incomplete or whose source patch carries two
scales rather than assuming it -- and the gate prints `complete True one_scale
True` for all six groups rather than trusting the refusal to be unreachable.

The win is two independent redundancies, not one: a per-pair traversal repeats
the upward pass once for every field mesh sharing a source mesh, and it leaves
the M2L with a batch of 1.6 V entries per transfer offset. On the fault zone the
batch moves 1.61-1.79 -> 2.25-3.11, which is still under the m ~ 8 crossover
where a shared table starts to beat the matrix-free kernel, so the table is not worth building on the
GATE model. On the target model it is, and the distribution -- not the mean --
says so. topo_inclusion, canonical, grouped traversal, V entries per transfer
offset:

    scale  unknowns  V entries  offsets  mean  V-weighted  median  p10  max
      1      31,098     27,479      218   126       404      58     15   823
      2     117,120    129,293      316   409     1,717     145     37 3,425
      3     260,598    293,243      316   928     2,292     471    147 5,484

    share of V pairs in keys with m >= 114:  72.7 % / 94.0 % / 99.6 %

The V-weighted mean is the cost-relevant one (sum m^2 / sum m) and it is 2-6x
the plain mean, so the pairs concentrate in the WIDE keys -- the opposite of the
heavy tail that would have made a table pay on a handful of offsets only. At
scale 3 the tenth percentile is 147, an order of magnitude above the m ~ 8
crossover, so a table is worth building on **316 of 316 keys** and a best-of
policy degenerates to the table everywhere. Projected M2L time against the numba
kernel: 5.91x / 6.41x / 6.52x. The win grows with N and has not saturated.

Gated in `verify_fmm` [d] against the PER-TERM operator at
`FMM_M2L_VARIANT_PARITY`, never against the dense operator: a grouped traversal
that silently dropped a whole patch pair would still pass `FMM_OPERATOR_PARITY`.

**The block-Jacobi price: 193 GiB at the shipping chunk, not 268, and it still
busts the machine.** The storage model is an identity -- stored bytes
= 8 x N x wmean with wmean = sum c^2 / sum c, reproduced at ratio 1.0000 at all
15 (scale, chunk) points -- but its constant was wrong: `build_cluster_tree`
halves a super-block until the leaf is under the cap, so the realised chunk lands
in (cap/2, cap] and never at cap, measured 0.617-0.738 of nominal on this mesh
family. 268 GiB was the f = 1 ceiling. Measured and projected:

    chunk  iters 31k/117k/261k  LU GiB @261k  precond GiB @4M  iters @4M  fits?
     9000        27 / 37 /  42        12.90            193.4   65/74/125  BUSTS
     4000        30 / 43 /  51         5.30             78.1  91/101/152  BUSTS
     2000        34 / 48 /  61         2.65             39.1 129/129/182  BUSTS
     1000        37 / 58 /  69         1.33             19.6 125/154/206  FITS
      500        42 / 63 /  89         0.66              9.8 234/234/265  FITS

Chunk 9000 can never fit, and that is provable rather than sampled: splitting any
n > cap yields children > cap/2, so every chunk exceeds 4500 DOF and the floor is
8 x 4e6 x 4500 = 134.1 GiB, geometry-independent. Chunk 2000 misses by 0.4 GiB
once C's own terms are counted, so **chunk 1000 is the first setting that fits**,
with 19.1 GiB spare, and it is also the flattest point on wall time -- the trade
is already non-monotone at 260,598, where chunk 500 is SLOWER than chunk 1000
(89 iterations and 29.2 s against 69 and 20.6 s). Do not treat 1000 as tuned:
nothing was measured above 260,598, the 4M iteration counts are three-point fits
extrapolated 15.3x with a 1.6x spread, and varying the slot mix at fixed N moves
the chunk-9000 figure over 140-260 GiB.

Two consequences that are in no budget. The preconditioner APPLY becomes a
first-class per-iteration cost at chunk 1000 -- 2.14 s projected at 4M, a serial
Python loop over 6,336 independent `lu_solve` calls, embarrassingly parallel --
so below ~2 s per matvec the preconditioner and not the FMM sets the iteration
cost. And the iteration count and the FGMRES workspace are coupled: 12.0 GiB
assumes `GMRES_RESTART = 200` while the chunk-1000 projection is 125-206
iterations, so the pessimistic end restarts and raising the restart to 300 costs
~6 GiB of the 19.1 GiB of headroom.

**A FOURTH RUNG, AND FLAT H's REAL SHAPE: THE BUILD IS THE WALL.** The 459,516
unknown rung (107,790 elements) was planned as M2 and never run. Run now,
topo_inclusion, hmat, float32 storage, `37c2ff8`:

  elements  unknowns   build  ACA%  precond  solve  it   matvec  operator   RSS   op err
     7,483    31,098   13.5 s   84    3.9 s   3.0 s  23  24.8 ms   1.00 GB   6.0  4.2e-05
    27,659   117,120   90.8 s   87   35.9 s  25.5 s  37  71.3 ms   4.44 GB  25.3  2.8e-05
    61,286   260,598  265.6 s   86   26.3 s  15.0 s  42 167.1 ms  11.73 GB  34.1  1.6e-05
   107,790   459,516  581.8 s   84   39.7 s  33.5 s  49 349.1 ms  23.90 GB  59.5  5.9e-05

Fitted over all four: **build N^1.39**, ACA N^1.40, **matvec N^0.97**, bytes
N^1.18, iterations N^0.28. (The build and ACA columns are PRE-FOLD; the fold
commit below takes them to 7.6 / 48.2 / 144.0 / 314.1 s at the same exponent.)

**The matvec is O(N); ASSEMBLY is what scales badly.** 84-87 % of the build is
the ACA phase at every rung and 68.9 % of that is the fold, so the fold is the
single highest-value flat-H item -- larger than every partition constant and the
kernel put together. Extrapolated to 1M elements the build is ~3.8 h.

**THE FOLD IS ONE NOGIL KERNEL AND RUNS ON THE POOL: 1.84x ON THE BUILD**
(`la/fold_numba.py`, `PendingLR.fold`, `hop._compress_all`). Measured on the
same ladder, everything else held:

    unknowns    build before   build now   speedup   aca before   aca now
      31,098        13.5 s        7.6 s      1.79x      11.4 s     5.5 s
     117,120        90.8 s       48.2 s      1.88x      78.7 s    36.1 s
     260,598       265.6 s      144.0 s      1.84x     228.0 s   106.2 s
     459,516       581.8 s      314.1 s      1.85x     488.6 s   222.1 s

**The speed-up is UNIFORM across the ladder** (spread 1.06x), so the scaling is
untouched: build stays N^1.38 against N^1.39, ACA N^1.37 against N^1.40, matvec
N^0.97 and bytes N^1.18 unchanged to two digits. That was an assumption worth
measuring rather than stating -- a fold speed-up concentrated at one end would
have moved the exponent, which is the number the whole flat-H-vs-bbFMM
comparison turns on.

And the operator is the same to 7 digits, not bitwise: iteration counts are
IDENTICAL at all four rungs, stored bytes are byte-identical at three of them,
and scale 4 differs by 6,384 bytes in 25.66 GB (2.5e-7) -- one block of ~100,000
keeping one extra column, which is the +1 on kv the prototype measured on 1 of
45 blocks. Operator error moves in the fifth digit (5.8830e-05 -> 5.8864e-05).
Against the projected 2.1x.

Two measurements forced the algorithm, and the OBVIOUS version of this kernel is
1.05x, i.e. nothing. A column-at-a-time Gram-Schmidt is 3.7x numpy's QR because
it is BLAS-2, so the QR is blocked into panels whose two orthogonalization
passes are four GEMMs. And a full (K, K) cyclic Jacobi is **23x `eigh`, not the
~3x this file projected**, and it is O(n^3) -- so the eigenproblem is DEFLATED by
a pivoted Cholesky first, which cuts n by ~1.3x per side. The deflation's error
bookkeeping is exact rather than heuristic: with `G = A A^T` the Schur-complement
trace bounds the discarded energy, and carrying it as already-spent budget makes
the truncation decision the reference's -- measured EQUAL keep counts on all 90
real Gram matrices of 45 blocks x 2 sides.

No LAPACK factorization anywhere, which is forced rather than stylistic: numba's
`np.linalg.qr` returned memory read after free (`c480c75`) and scipy's economic
QR silently corrupts 2-6 stored blocks per model. Only `np.dot` is borrowed.

**And the fix is the WRAPPER, not the QR** -- which this session's own summary
got wrong before re-reading the profile above. `np.linalg.qr` already releases
the GIL and is 86.8 % of a large fold, so replacing it attacks the part that
already scales. What is GIL-held is `_unit_gram` x2, `_principal` x2 and the core
products: ~5.7 ms per fold at ANY block size, because they work on (K, K)
matrices with K ~ 102 independent of the block. That is 13 % of a 1024-element
block but 55 % of a 128-element one, and the production partition is dominated by
the small end -- which is why the measured scaling is worse than Amdahl and the
shape is convoy rather than plain serialization. The fix is the WHOLE fold in one
nogil kernel, whose one hard piece is a hand-rolled cyclic Jacobi for the (K, K)
symmetric eigenproblem, numba's LAPACK having returned freed memory here
(`c480c75`). Projected 2.1x on assembly.

**The in-core ceiling is ~225-250k elements, not 1M.** Peak RSS is still 2.5x
the operator bytes at the largest rung (6.0x, 5.7x, 2.9x, 2.5x -- the transient
shrinks relatively but does not vanish), so 128 GiB is reached near 1M unknowns.
Extrapolated: 50k elements ~200 s build and ~13 s solve, 100k ~525 s and ~31 s,
250k ~33 min and ~1.6 min at the edge of core, 500k and 1M out of core.

**The operator error trend REVERSED at the fourth rung, and it was this file's
own claim.** Recorded: "the operator error IMPROVES with size, 4.2e-5 / 2.8e-5 /
1.6e-5". The fourth rung is **5.9e-05** -- a 3.7x jump back up, cutting the
margin against `BENCH_OPERATOR_ERROR_MAX = 1e-4` from 6.4x to 1.7x. Three rungs
of a monotone trend did not survive the fourth, which is the same shape as every
other collapsed claim here. Do not carry "improves with size" any further.

**And the argument for bbFMM is not the memory, it is the BUILD.** bbFMM has no
ACA and no compression -- a tree, a stencil and the near-field blocks, all O(N),
measured ~30 s at 260,598 unknowns against flat H's 265.6 s, and scaling N^1.0
against N^1.39. End to end at 1M elements: flat H ~3.8 h build + ~3.4 h
out-of-core solve; bbFMM ~8 min build + a solve set by its matvec constant.
5x memory was never the case; 28x on build is.

**THE PRECONDITIONER APPLY, AND A CORRECTION THIS FILE SHOULD KEEP.** Measured
at the shipping ladder, one apply and one matvec:

    N          threads 1                       threads 8
    117,120    apply 653.1 ms  ratio 9.15:1    606.1 ms  8.55:1   (1.08x)
    260,598    apply 659.1 ms  ratio 3.96:1    171.8 ms  1.02:1   (3.84x)

So the solve at 260,598 IS preconditioner-dominated ~4:1 unthreaded, and the
threaded apply takes it to parity: the iteration phase goes 34.8 s -> 13.9 s,
**2.5x**. An earlier note in this session said the 4:1 was wrong; that note was
itself wrong, because it compared against a ladder run that ALREADY had
threading on -- measuring the fix and concluding the problem never existed. Never
take a reference from a run the change under test has already moved.

**THE APPLY WAS NOT BANDWIDTH-BOUND, IT WAS THE WRONG BLAS CALL.** `getrs` with
ONE right-hand side is `laswp` plus two `trsm` on a single column, and `trsm` on
one column IS `trsv` -- but the `trsm` path reads the factor at 17-23 GB/s where
the same factor read by `trsv` runs at 79-83, which is this machine's single-core
streaming ceiling (74.5 GB/s measured). The apply is 0.25 flop/byte, so that
ratio is the whole story: it was **4.4x off the bound it was assumed to be at.**

Two changes, both EXACT -- nothing refactorized, nothing approximated. The dense
rung precomputes LAPACK's swap sequence into one gather at build, fuses it with
the layout permutation, and calls two TRSV (agrees with `lu_solve` entrywise to
2e-14 at n up to 27,921, identical iterations and true residual). The chunked
rung cannot use it, because scipy's TRSV wrapper HOLDS the GIL (70.4 ms at one
worker, 70.9 at eight, cpu/wall 1.01) and would serialize the pool it depends
on, so it gets a hand-written nogil triangular solve instead -- 2.5x slower
serially, which is why the rung keeps scipy on its serial path. And
`PRECOND_APPLY_MIN_CHUNKS` drops 8 -> 2: at chunk 9000 the six super-blocks at
117,120 unknowns hold 4, 1, 4, 8, 1, 4 chunks, so five of six never reached the
pool, which is the whole reason threading had bought 1.08x there.

    N          apply before   apply now    ratio to matvec
    117,120       653.1 ms     156.0 ms    8.50:1 -> 2.20:1   (4.19x)
    260,598       659.1 ms     116.6 ms    3.96:1 -> 0.69:1   (5.65x)

**At 260,598 the preconditioner apply is now CHEAPER than the matvec**, which
was the point: it had been the per-iteration cost.

float32 factors were measured and DEMOTED: 1.47x on top of this, and inside the
noise once the apply is already at one matvec. Worth taking later for the memory
(it halves the preconditioner's bytes at the target), not for the speed.

**Two things measured here and deliberately NOT done.** Lowering the dense cap
30,000 -> ~10,000 sends the big patch blocks to the threaded rung and measures
free at 117,120 (37 iterations either way, relres 7.02e-09 against 7.63e-09),
taking the apply to 68.0 ms and the preconditioner build 35.4 -> 10.4 s. But it
is a policy reversal measured at ONE scale, it costs +4 iterations at 31k where
the dense rung covers 100 % of the DOF, and the chunked rung is the one whose
iteration count grows with N. Confirm at 260,598 first. And a blocked triangular
solve with threaded off-diagonal updates was measured BOTH ways and is not
worth it -- with one right-hand side the update is a GEMV, not a GEMM, so it
moves the same bytes and only gains cores: 134.8 ms at 8 threads in one numba
parallel kernel against 89.4 ms for the serial two-TRSV.

**Superseded, kept for the shape of the error: the dense-LU rung's apply is NOT
threaded and is the binding cost below the dense cap.** At 117,120 unknowns threading buys 1.08x, because those
super-blocks sit under `dense_rung_max_dof` (~30,000) and take the dense rung,
whose apply is ONE `lu_solve` per super-block (`_permuted_lu_solve`) rather than
the chunked loop. That apply is ~600 ms, single-threaded (nrhs = 1 is level 2),
and it is 8.5x the matvec there -- which is also what makes 117k the anomalous
row above, at 36 s of preconditioner build and 617 ms per iteration.

**THE FMM IS IN `bench_scaling` (`--backend fmm`), AND THE FIRST RUNG SHOWS THE
INTEGRATION, NOT THE PHYSICS, IS WHAT COSTS.** topo_inclusion at 31,098
unknowns, canonical, compressed M2L, against `hmat` on the same rung:

                 hmat        fmm
    build        7.6 s     338.1 s
    matvec      25.2 ms   1175.3 ms
    solve        1.5 s     254.2 s
    iterations     23         25
    op error   4.2e-05    1.6e-05
    sol error  3.0e-04    8.7e-05
    bytes       1.08 GB    4.00 GB

Two things point opposite ways. The FMM is **more accurate than flat H on the
same rung** (2.6x on the operator, 3.4x on the solution), and its 4.00 GB is
mostly the 3.34 GB M2L table, which is N-INDEPENDENT and amortizes by scale 3.

But the build and the solve are dominated by per-pair traversals that
`FarGroups` was built to remove, and the phase split says so: `tree 0.08 s,
groups 2.16 s, pairs 0.07 s, other 335.8 s`. Two call sites:

* `AssembledH._calibration` takes the row sums as three constant-field
  `pair.matvec` calls per T pair -- ~135 separate FMM traversals -- which is
  the 335.8 s.
* `BlockGaussSeidel.__call__` calls `pair.matvec` for every strictly-lower
  Gauss-Seidel coupling, so each preconditioner APPLY runs more traversals:
  254 s of solve for 25 iterations against a 1.175 s matvec is ~9 s per
  iteration of preconditioner.

**BOTH FIXED, AND THE DIAGNOSIS WAS HALF WRONG.** The expensive thing was not
the traversals: each of the 91 `PairFMM` objects was building its OWN M2L
table. A block is the kernel on the lattice at one offset -- (kernel, order,
material) and nothing else -- so the table belongs to the TREE, and moving it
there is most of the win. The calibration was also grouped (its row sum is
`sum_p sigma(R,p) H_qp e_k`, which IS a group's matvec on a constant, three
traversals per region instead of three per pair) and that bought ~1 % on top.

    scale 1        original   shared table   + grouped calibration
    build           335.9 s      131.6 s          129.9 s
    solve           250.6 s          --           100.3 s
    peak RSS        17.4 GB       8.93 GB          8.60 GB
    iterations           25           23               23
    op / sol err   1.6e-05 / 8.7e-05  ..  unchanged at both

The redundant tables were half the resident set, which is also why `fmm_stats`
undercounted by 6.2x: it walked the groups, and the duplicates were on the
pairs. What is LEFT in the build (`other` 127.6 s of 129.9) is the one-time
table factorization, which is N-INDEPENDENT -- a fixed cost, not a scaling
problem.

**THE FMM LADDER, RE-RUN WITH EVERY FIX IN: THREE CROSSOVERS, MEASURED.**
topo_inclusion, canonical, compressed M2L, against the flat-H ladder on the
same rungs:

      N          build  precond   solve   TOTAL    matvec    B/unk   RSS   op err
   31,098  hmat    7.6      3.8     1.5    13.0     24.8   34,586   6.0  4.2e-05
  117,120  hmat   48.2     35.8     8.8    92.8     71.1   40,744  25.5  2.8e-05
           fmm   201.8     36.8   354.4   593.1   4758.7   57,530  24.6  9.4e-06
  260,598  hmat  144.0     26.4    12.8   183.3    168.6   48,345  34.3  1.6e-05
           fmm   238.9     27.4   732.6   998.9  10398.2   34,254  30.1  8.6e-06
  459,516  hmat  314.1     40.7    29.0   383.9    351.1   55,848  59.3  5.9e-05
           fmm   304.2     40.5  1337.3  1682.1  18869.2   27,941  45.3  1.8e-05

At 260,598 unknowns the FMM is **smaller (34,254 B/unknown against 48,345, and
30.1 GB of RSS against 34.3), more accurate (8.6e-06 against 1.6e-05) and takes
the same 42 iterations**. Those are crossovers, not projections: flat H's bytes
per unknown RISE with N and the FMM's FALL, because the M2L table is
N-independent.

**SUPERSEDED by the rerun below** -- the build had NOT crossed (404.3 against
315.3 once the M2L table is counted in the build that pays it), the solve ratio
is ~55x not ~46x, and the total at scale 4 is 3.69x not 4.4x. Kept for the shape
of the error: ~100 s of one-time table sat in the solve column, so the build
looked like it had crossed and the solve looked worse than it was.

**THE BUILD CROSSED AT SCALE 4, MEASURED: 304.2 s against 314.1 s.** The FMM's
build is N^0.43 over the last step -- mostly the one-time table factorization --
against flat H's N^1.38. At 459,516 unknowns the FMM is also 2.0x smaller per
unknown (27,941 against 55,848), 1.31x smaller in RSS and 3.2x more accurate
(1.8e-05 against 5.9e-05), at the same 49 iterations.

**BUT THE TOTAL IS STILL 4.4x WORSE, AND THAT IS THE NUMBER THAT MATTERS.**
1682.1 s against 383.9 s. The ratio improves -- 6.4x, 5.5x, 4.4x over the three
rungs -- only because the build crossed; the SOLVE ratio is flat at ~46x
(40x / 57x / 46x). flat H's total is 82 % build; the FMM's is 79 % solve. Any
claim that the FMM "wins" has to name which column it means.

What has NOT crossed is the matvec: 10,398 ms against 168.6 ms, **62x**, at a
clean N^0.98. That is the structural gap, and after compression it is no longer
mostly M2L.

The fixes moved the build and the solve a long way -- build 750.3 -> 201.8 s
(3.7x) at scale 2 and 1050.6 -> 238.9 s (4.4x) at scale 3, RSS 43.4 -> 24.6 and
55.5 -> 30.1 GB -- with every accuracy and iteration figure unchanged.

**THE RSS/BYTES GAP IS EXPLAINED, AND IT IS NOT THE OPERATOR.** The accounting
was extended to everything the operator holds -- the pairs' own near caches, the
per-box multipole and local expansions, and the widest M2L gather -- and it
REFUTED all four candidates. At 117,120 unknowns: near 1.67 GiB, pairs' near
caches **0.00 GiB**, tables 4.14, stencil 0.47, expansions 0.31, gather 0.05,
total 6.64 GiB against a 24.44 GB peak. The pairs' caches are empty because
grouping the calibration and the preconditioner means `pair_for` is barely
called any more, and the two transients are 0.36 GiB of an 18 GB gap.

Tracing RSS through a bare traversal instead found it: imports 0.11 GB, tree
0.24, groups 0.77, **first matvec 8.00** (building 4.14 GiB of table), second
matvec 8.18. So the FMM OPERATOR is 8.0 GB against 6.64 GiB accounted --
**1.12x, not 3.4x** -- and the accounting was right all along. The harness's
24.44 GB is mostly not the operator: the PRECONDITIONER's LU factors, measured
independently at 8.90 GB at this size (4 of 6 super-blocks on the dense rung),
plus the 91 PairFMM objects and the exact-rows reference. `fmm_stats` was never
meant to count the preconditioner, and it is a cost BOTH backends pay -- hmat's
peak here is 25.5 GB.

So the 4M memory estimate is the accounted fit times ~1.12 plus a preconditioner
we size independently: ~88 GB of operator x 1.12 = ~98 GB, plus 19.6 GiB at
`PRECOND_BJ_CHUNK_DOF = 1000`, **~119 GB -- it fits 128 GB, tightly**. The
earlier ~335 GB came from extrapolating a peak that conflated the operator with
a preconditioner whose size is a knob.

**Superseded: the open question for 4M is the RSS/bytes gap.**
Peak RSS is still 3.4x the accounted bytes (down from 6.2x once the duplicate
tables went). Four rungs now, and it is NOT shrinking: 3.7x, 3.4x, 3.5x. Fitting the
ACCOUNTED bytes as fixed + B N over the last two rungs gives ~3.8 GB +
19,700 B/unknown = **~88 GB at 4.26M, which fits**; the same fit on PEAK RSS
gives ~10 GB + 76,400 B/unknown = **~335 GB, which does not**, and a working set
is not spillable the way a stored operator is. **This is now the single thing
standing between the FMM and 1M elements.**

Extrapolated end to end at 4.26M, both paths land in the same place: flat H
~2.0 h of build (N^1.38) plus ~2.4 h of out-of-core solve, the FMM ~13 min of
build (N^0.43) plus ~4.2 h of solve. **~4.4 h against ~4.6 h -- a tie**, with the
FMM's build advantage exactly cancelled by its solve.

**THE PRECONDITIONER'S LOWER COUPLINGS ARE GROUPED NOW, AND IT IS WORTH 1.03x.**
`AssembledH.lower_applier` hands the preconditioner one traversal per (region,
kernel) for a super-block's strictly-lower couplings instead of one per term,
and `FarGroups` takes a `terms=` subset for it. Correct -- 6.385e-16 against the
per-term loop on 12 of 13 super-blocks -- and measured at scale 2: solve
354.4 -> 343.2 s, iterations back to 37. The cost analysis was right (the
preconditioner IS ~45 restricted traversals per apply) and the fix was not: each
super-block holds only a handful of lower terms, so merging them into a union
traversal is a wash against the restricted ones.

**It cost a SILENT bug on the way, and the gate now pins what would have caught
it.** The first version scattered by PATCH name where `sb.local_offset` is keyed
by SLOT name, so every lookup missed, nothing was subtracted, and the sweep
quietly degraded to block-Jacobi -- 37 iterations became 52 and the solve still
converged. No residual or error limit would have shown that. The clause now
compares the grouped applier against the per-term loop in situ AND against the
same preconditioner with every lower coupling dropped, requiring the second to
differ (measured 1.5e-01): a gate that only checked agreement would have passed
the broken version, both sides being zero.

**NEAR-ONLY LOWER COUPLINGS: 1.17x, FOR +3 ITERATIONS**
(`PRECOND_LOWER_NEAR_ONLY`, `PairFMM.matvec(far=False)`). A preconditioner is an
APPROXIMATION, so its off-diagonal couplings can drop the far field and keep the
exact near one: that changes the preconditioner and not the operator, so only the
iteration count can move. Measured at scale 2: solve 343.2 -> 293.3 s,
iterations 37 -> 40. The whole chain on that rung is 354.4 (per-term) -> 343.2
(grouped) -> 293.3 (near-only), 1.21x.

Gated as a genuine INTERMEDIATE, which is the only way to catch it doing nothing
or everything: the near-only apply sits 3.296e-02 from the full coupling and
1.594e-01 from dropping it, so it is neither. The equivalence clause pins the
flag OFF, because with it on the grouped applier is deliberately not the
per-term loop.

**BOTH FAR FIELDS SHIP, AND THE CHOICE IS A FLAG** (`HBackend(far="aca"|"fmm")`,
`defaults.FAR_FIELD`). Neither dominates, and the reason is that they fail on
different axes. ACA adapts its rank per block (~16) where the FMM's p^3 adapts
to nothing: the FMM does ~19.9 Mflop/unknown against ACA's ~9.5 Kflop, 2,100x,
and its matvec measures ~55x slower (48-64x over four rungs, no trend). The
remaining FMM matvec levers are the common basis (1.37x, Amdahl-capped) and
float32 (~1.3x), so ~30x survives any of them -- that gap does not close. What
the FMM wins is the BUILD exponent, N^0.283 against N^1.381 (84-87 % of the ACA
build being the ACA loop), and the bytes: 31,326 B/unknown against 55,848 at
459,516 unknowns, FALLING as N^-0.549 against ACA's N^+0.175 because the M2L
table is flat in N. Extrapolated to 4.26M that is a ~66 GB operator against
~419 GB, and it is the only reason 1M elements is reachable at all.

So the map is two-dimensional -- size x matvec count -- and the fourth cell is
covered by NEITHER:

| | few solves | many matvecs |
|---|---|---|
| fits RAM | aca (3.7x faster end to end at 460k) | aca (~55x per matvec) |
| huge | fmm (the only one that builds) | **nothing** |

1M elements under rate-and-state is exactly that cell: ACA needs ~419 GB of
operator (out-of-core ~106 s/matvec), the FMM fits at ~66 GB plus ~19.6 GiB of
preconditioner but pays ~55x per step.
That cell is what an H^2 would own -- adaptive rank AND shared bases -- and
H^2 is dead on this kernel: shared bases inflate rank 17 -> 175 algebraically,
reconfirmed at 100 -> 526/620 in the M2L common basis. Naming the gap here so
it is not rediscovered as a surprise.

**The open lever for that cell is TOLERANCE, not architecture.** Both operators
over-deliver against `BENCH_OPERATOR_ERROR_MAX = 1e-4`: 5.9e-05 (ACA) and
1.8e-05 (FMM) at 459,516. If a rate-and-state run tolerates 1e-3, ACA rank
falls and the bytes with it, moving the fits-in-RAM boundary out. That is a
sweep of `BLOCK_COMPRESSION_TOL` under the existing gate, not an engine.
UNMEASURED -- do not assume the saving.

**The flag cost one real bug fix, and it was the silent kind.**
`rebuild_for_materials` never forwarded `_groups`, so an FMM assembly lost its
grouping on every material step and fell back to the per-pair route (correct,
because that route reads the assembly's materials, but ~91 traversals where 6
would do, plus the per-PAIR calibration). Forwarding it naively would have been
WRONG rather than slow: `FarGroups` holds its own materials dict by reference
and it is not the assembly's, so the far field would have stayed at the old
modulus while the calibration diagonal and the RHS moved to the new one --
self-consistent, convergent, and the wrong problem. `FarGroups.for_materials`
re-points it in O(1) (only `materials` is material-dependent; the groups hold
geometry, eps, sigma and the tree) as a NEW object, because writing through
would move the far field of every assembly holding those groups. Measured on
the example below: the material step costs 52 s against the first state's
229.5 s, i.e. the traversal and the M2L table are genuinely reused.

`verify_fmm` clause [d] now gates both halves -- the rebuilt grouped operator
against the rebuilt per-term one at 5.551e-16 AND that the far field MOVED
(5.559e-02), because groups that silently kept the old material still agree
with the per-term route everywhere the far field does not reach -- plus the flag
itself at the SHIPPING settings (`domain="canonical"`, `m2l="table"`, 3.426e-06
against the exact dense operator), which no clause ran end to end before: they
all build on the zone's own tree at placement safety 1.0 and the
"extent"/"evaluated" defaults, to isolate the pieces. `storage="combined"` with
`far="fmm"` is refused rather than forced, because `PairFMM` has no
`warm_views` and the failure would otherwise be an AttributeError halfway
through an assembly.

**BOX + TOPO + INCLUSION, BOTH FAR FIELDS, END TO END** (`--backend` on
`examples/make_topo_inclusion.py`). The four-state decomposition at 31,098
unknowns (7,483 triangles), every state calibrated with `eps="auto"`, against
the committed dense reference:

| | build | 4 states | iterations | u vs dense | Du_topo vs dense |
|---|---|---|---|---|---|
| aca | 7.6 s | 46 s | 23 23 23 23 | 1.5-9.9e-06 | 1.0-1.7e-04 |
| fmm | 129.7 s | 560 s | 32 30 32 30 | 1.5-4.4e-06 | 1.6-2.9e-04 |

Both converge to `true_relres` ~6-8e-09 on all four states. 12x apart end to
end, which is the expected sign at 31k: this is far below the build crossing
(scale 4, 459,516 unknowns), so the FMM is paying its p^3 with none of its
build advantage yet.

**The two agree with EACH OTHER as independently as with dense**, which is the
check that rules out a shared bug: 2.1e-06 to 1.1e-05 on the four solution
states, and h-f ~ (h-d) + (f-d) throughout -- the two far fields scatter around
the dense LU independently rather than clustering together away from it. A
common wrong kernel, sign or free term would show as h-f much SMALLER than
either one's distance from dense. Two independent approximations landing
independently on the same answer is the stronger statement.

**And the decomposition error is a property of the QUANTITY, not the backend.**
The figure shows Du_topo = u(topo,het) - u(flat,het), a difference of two
fields that agree to 43-72x their difference. So displacement fields accurate
to ~2e-06 give a decomposition accurate to ~1.4e-04, and both backends land
there (1.0e-04 ACA, 1.6e-04 FMM on `host_top`; 1.7e-04 and 2.9e-04 on
`inclusion_top`) -- the amplification is the cancellation, exactly. Anyone
making this figure from a compressed backend is working at ~2e-04 on the
decomposition, not the ~1e-06 the fields suggest. But it is the SAME field at
the SAME amplitude: against dense, every decomposition has correlation
0.99999996 to 1.0 and an amplitude ratio within 8e-05 of unity, so the residual
is noise on a difference and not a disagreement. In physical units the worst
case is a 2.26 cm topography effect on `inclusion_top` (displacements
themselves 1.9-5.1 m) on which the two methods differ by ~8 um. The dense
reference stays the reference, and a non-dense run writes a
`_<backend>`-suffixed npz so it cannot overwrite it.

Known hole, deliberately not filled: `AssembledH.nbytes()` sums `p.nbytes()`
over the pairs and `PairFMM` has none, so asking an FMM assembly for its size
raises. The honest FMM accounting is `bench_scaling.fmm_stats` (which counts
the shared M2L tables ONCE, where a per-pair sum would multiply-count them);
inventing a second number that disagrees with the committed ladder would be
worse than the AttributeError. `estimate.estimate_memory` likewise has no
"fmm" mode.

**AND THE FMM "BUILD" IS 98 % M2L TABLE, NOT FAR FIELD -- WHICH MOVES THE
BUILD/SOLVE SPLIT OF EVERY FMM RUNG IN THIS FILE.** Measured on
topo_inclusion scale 1 (31,098 unknowns) by timing the pieces of a 129.7 s
build:

    tree 0.08   groups (the traversal) 2.19   pairs 0.07
    _calibration_grouped 126.85        _build_rhs 0.13
    the SAME calibration, called again on the warm assembly:  2.6

So the calibration is not expensive -- 2.6 s warm against the ACA path's 0.11 s
-- it is merely the FIRST consumer of the shared M2L table, and pays the whole
construction. Worse, it only touches the T groups, so after the build the table
cache holds T and not U:

    build: table keys ['H', 'H']            (T, two eps passes)
    matvec #1: 54.11 s -> ['G', 'G', 'H', 'H']    (U built here)
    matvec #2: 1.06 s   #3: 1.08 s   #4: 1.09 s   (steady state)

**The one-time table cost therefore straddles the two columns**: ~124 s inside
"build" and ~53 s inside the first matvec, which the harness counts as SOLVE.
Every FMM rung in this file understates its build and overstates its solve by
the U table; the totals are right, and so is the steady-state matvec ratio
(1.07 s here, ~55x), but the split is not. ~177 s of table against a 2.19 s traversal
and a 1.07 s matvec.

It also explains the build exponent rather than contradicting it: the table is
316 transfer offsets and is N-INDEPENDENT by construction -- that is what the
canonical domain buys -- so an FMM "build" dominated by it is nearly flat in N,
which is exactly the N^0.21-0.43 that was fit and attributed to the traversal.

**THE TABLE IS SHARED ACROSS TREES NOW, AND THE SPLIT IS HONEST**
(`fmm_table.shared_table`, `FMM_M2L_CACHE_MAX_TABLES = 8`,
`phases["tables"]`). A block is the kernel on the unit lattice at one transfer
offset, so it depends on the model through NOTHING -- the lattice is the
canonical Chebyshev grid and the half-width folds out as a power of two -- yet
it was held per `FmmTree` and so rebuilt per tree. Measured on the four-state
topography example at 31,098 unknowns, the second tree's build:

    129.7 s -> 7.0 s      (18.5x; the tables phase 179.11 -> 2.02 s)
    four states 558.8 s -> 382.3 s   (1.46x end to end)

and `HBackend._assemble_fmm` now builds the tables itself, so "build" means
ready to solve: 184.2 s of which `tables` 179.11, and the matvecs are
1.09 / 1.11 / 1.11 s -- steady from the FIRST one, where before the first cost
54.11 s inside the solve.

**It is numerically a no-op, which is the claim that matters.** Iterations and
`true_relres` are identical on all four states (32 / 30 / 32 / 30,
7.77e-09 / 5.74e-09 / 8.10e-09 / 5.97e-09) and every digit of the
agreement table above is unchanged.

**And it fixed a leak that was not an optimisation.** The key carries the
MATERIAL, so a sweep minted a table per novel material and nothing evicted the
superseded ones: measured 4 -> 5 -> 6 -> 7 tables and 3.34 -> 3.99 GB over
three rebuilds at 31k, where at 4M a T table reaches the
`FMM_M2L_TABLE_MAX_BYTES` cap and a 22-material sweep would add tens of GB of
dead blocks. That constant bounds ONE table and never bounded their number;
the LRU does, and the same measurement now reads 4 -> 8 -> 8 -> 8 with the
bytes plateauing. `FmmTree._tables` had to become a WEAK view for the bound to
mean anything -- one tree outlives every rebuild of a sweep, so a strong
reference there would pin every superseded material's table and defeat the
eviction entirely.

**The lattice is IN THE KEY, not checked afterwards**, which is what makes
cross-tree sharing safe rather than probably-safe: a tree whose lattice
differed would otherwise be handed blocks stated on a lattice it does not use.
Measured bitwise equal between two trees over different surfaces of one
geometry (max abs difference 0.000e+00), and `verify_fmm` clause [d] pins all
five things that could go wrong -- distinct trees, lattices bitwise equal
(`array_equal`, because a tolerance would hide exactly this), 4 keys reused BY
OBJECT, a novel material building its own, and the count bounded.

Not shared across PROCESSES: `bench_scaling` spawns a child per rung, so the
ladder gets the honest split and no reuse. Persisting the table to disk is the
next step on this thread and is unmeasured.

**THE LADDER, RERUN WITH EVERYTHING IN -- AND THE BUILD HAS NOT CROSSED.**
topo_inclusion scales 1-4, both far fields, near-only preconditioning, the
honest build/solve split, the shared table (per process, so each child rung
gets the split and no reuse). `build` includes `tables`:

    n        bk   build tables precond solve  TOTAL  mv(ms) it  B/unk   RSS  op_err
    31,098  hmat    7.6    0.0     4.0   1.5   13.1    24.9  23  34586   6.0 4.2e-05
    31,098   fmm  184.1  179.0     3.9  41.1  229.2  1189.6  32 132068   8.9 1.6e-05
    117,120 hmat   48.8    0.0    36.2   8.9   93.9    72.9  37  40744  25.4 2.8e-05
    117,120  fmm  283.4  261.4    37.1 207.9  528.5  4687.2  40  60889  25.9 9.4e-06
    260,598 hmat  144.0    0.0    26.5  12.9  183.4   174.8  42  48345  34.1 1.6e-05
    260,598  fmm  327.4  271.4    27.6 491.2  846.2  9910.3  45  37467  29.9 8.6e-06
    459,516 hmat  315.3    0.0    39.7  29.1  384.1   348.9  49  55848  59.3 5.9e-05
    459,516  fmm  404.3  287.6    40.4 971.0 1415.7 18651.9  49  31326  45.0 1.8e-05

**CORRECTION, and it was this file's own headline.** "THE BUILD CROSSED AT
SCALE 4, 304.2 s against 314.1" is WRONG: with the tables counted where they
are paid, the FMM build is **404.3 against 315.3, still 1.28x worse**. The
crossing was ~100 s of M2L table sitting in the solve column. It is
approaching fast and monotonically -- 24.16x, 5.80x, 2.27x, 1.28x -- and will
cross just past this rung, but it has not crossed. Exponents N^1.381 (ACA,
84-87 % the ACA loop) against **N^0.283** (FMM), and the low one is now
explained rather than just fit: the table saturates.

**THE MATVEC RATIO IS ~55x, NOT 46x**: 47.9 / 64.3 / 56.7 / 53.5 over the four
rungs, non-monotone, so a constant factor and not a trend -- both matvecs are
O(N) (ACA N^0.973, FMM N^1.014). Anywhere this file or a summary says 46x,
read ~55x with a 48-64x spread.

**TOTAL is 3.69x at scale 4, improved from 4.4x** by near-only preconditioning,
and improving with size (17.44 / 5.63 / 4.61 / 3.69). The FMM's SOLVE exponent
is the worse of the two, N^1.168 against N^1.047, because its iteration count
climbs faster off a lower base (32/40/45/49 against 23/37/42/49 -- they MEET at
scale 4).

**THE TABLE TERM IS MEASURED FLAT, which is the whole memory argument.** The
byte split at the four rungs:

    near       0.53  1.79  3.35  6.42   (N^1.0, the exact near blocks)
    lowrank    3.34  4.44  4.44  4.44   (the M2L tables: FLAT, 1,264 blocks)
    bases      0.14  0.51  1.13  1.97   (the per-element stencil)
    expansions 0.09  0.34  0.72  1.32   (per-box multipole/local)
    total      4.11  7.13  9.76 14.39   GB

4.44 GB over a 3.9x range in N, saturated after scale 1 once every transfer
offset that occurs has occurred. That is what makes B/unknown FALL --
**N^-0.549 against ACA's N^+0.175** -- and the bytes cross between 117k and
261k (1.49x, 0.77x), reaching 0.56x at 459,516.

**The 27,941 -> 31,326 B/unknown change is the ACCOUNTING completing, not the
operator growing.** 14.39 GB minus `expansions` 1.32 and `gather_peak` 0.23 is
12.84 GB = 27,941 B/unknown exactly; the committed ladder predates those two
terms being counted at all.

**EXTRAPOLATED TO 4.26M (last two rungs), and the trade is cleaner than a
tie.** Earlier this file said the two paths tie at ~4.4 h against ~4.6 h. They
do not:

    ACA  TOTAL N^1.303 -> 1.94 h   operator 419 GB   peak RSS N^0.974 -> 518 GB
    FMM  TOTAL N^0.907 -> 2.97 h   operator  66 GB   peak RSS N^0.719 -> 223 GB

**ACA is FASTER at the target and does not fit; the FMM fits and is 1.5x
slower.** That is the whole two-path case in one line, and it is a better
statement than the tie because it does not depend on the two curves crossing
at just the right place. The FMM operator estimate IMPROVED (66 GB against the
earlier ~98 GB) because B/unknown falls once the table is flat. The 223 GB peak
is the already-diagnosed gap -- preconditioner LU factors, the 91 pairs, the
exact-rows reference -- not the operator, and the preconditioner's share is a
knob (`PRECOND_BJ_CHUNK_DOF`); ~66 GB operator plus ~19.6 GiB preconditioner is
~87 GB and fits. A peak-RSS fit over two rungs is the weakest number here and
should not be the one a decision rests on.

**THE p LEVER IS DEAD, AND SO IS MOST OF WHAT WAS LEFT: AFTER COMPRESSION THE
FAR FIELD IS NO LONGER M2L-DOMINATED.** Measured on topo_inclusion scale 1
against exact rows, worst over a gaussian and the translation:

    p_U/p_T   matvec    worst error   margin on FMM_OPERATOR_PARITY
     6/8      1.16 s     1.686e-05          11.9x
     6/7      0.98 s     3.793e-05           5.3x
     6/6      0.89 s     2.090e-04          FAILS
     5/6      0.82 s     2.090e-04          FAILS

Half the earlier guess was right and half wrong. The OPERATOR's p-sensitivity is
far below a pair's -- 2.25x per order against the pair's 14x -- so p = 7 is
affordable on accuracy, which the pair rate said it would not be. But the SPEED
gain is 1.18x where p^6 predicts 2.3x, so it costs half the accuracy margin for
18 %. Not taken.

The reason is the rule this file keeps having to relearn: **re-profile after
every change.** Profiled after compression, the matvec is `_m2l_table` 57 %,
`far_apply` (W) 10 %, `tensordot` + `_separable` (L2P, M2M, L2L) ~25 %, near and
the rest ~8 %. M2L is ~56 % of the matvec, not the ~95 % it was, so EVERY
remaining M2L lever is Amdahl-capped: the common basis's 1.9x becomes ~1.37x and
float32's ~2x becomes ~1.3x.

**And 57 % of the matvec was PYTHON, not GEMM.** The same profile showed
`octree.cube` called 54,958 times -- twice per V pair, for a half-width that is a
function of the LEVEL alone -- and 176,221 reshapes. Taking the half-width from
`tree.level` and writing the gather into its output slice instead of through a
temporary is 1.14x, free: 1.16 -> 1.021 s at scale 1 and 4.76 -> 4.161 s at
scale 2.

**M2L COMPRESSION: 6.9-7.6x ON THE WHOLE FAR-FIELD MATVEC, AND 18x ON THE
TABLE.** The largest single result in the far field, and it comes from the one
standard bbFMM technique this implementation never had (`fmm.py`'s own SCOPE
docstring listed it as absent).

**Why it had to be this and not more engineering.** Measured at 117,120
unknowns, M2L is **2.33 Tflop per matvec = 19.9 Mflop per unknown**, against
flat H + ACA's ~9.5 **K**flop per unknown -- **2,100x the arithmetic**. Even at
this machine's measured f64 GEMM peak (460 Gflop/s) the M2L floor was 5.1 s
against flat H's 71 ms ACTUAL. bbFMM was never short of implementation; it was
carrying three orders of magnitude more work, and nobody had looked at the rank.

**The blocks are 2-5 % rank.** Measured over ALL 316 offsets, weighted by the
V pairs that use them, pass 0:

    kernel      tol      per-offset rank (min/med/max/weighted)   speed-up
    T p=8      1e-4          26 /  32 /  76 /  48                  24.2x
    T p=8      1e-6          49 /  66 / 160 / 100                  11.5x
    T p=8      1e-8          79 / 112 / 273 / 170                   6.8x
    U p=6      1e-6          34 /  46 / 102 /  66                   4.9x

1e-6 and not 1e-4: the truncation adds to the interpolation error and
`FMM_OPERATOR_PARITY` is 2e-4 with the scale-3 margin at 1.1x, so 1e-4 would
spend the budget twice. **These ranks do not depend on the mesh** -- a block is
the kernel on the lattice at one offset -- which makes them the one set of
numbers in this file that carries without an extrapolation argument.

**The common basis (Fong-Darve) was measured and is NOT taken yet.** It inflates
rank 100 -> 526/620 on T, the same failure mode that killed algebraic H^2
(17 -> 175), but still wins on flops because a pair costs ru x rv rather than
r (nr + nc): 21.7x against the per-offset 11.5x at 1e-6. On U at 1e-6 it LOSES
(4.5x against 4.9x). Worth ~1.9x more on T and left for later.

End to end, per-offset, canonical, X demoted:

    scale  N          numba      compressed    speed-up   vs numba
      1    31,098     7.81 s       1.14 s        6.85x    7.07e-07
      2   117,120    36.07 s       4.76 s        7.58x    8.42e-07

The table is **1840 MiB for all 316 T offsets** against 33 GiB dense (18x) and
is N-INDEPENDENT, so the per-material problem recorded above largely dissolves:
six groups cost ~6.4 GiB, not ~200.

**The factorization is randomized, and it had to be.** An exact `eigh` of the
(3p^3, 3p^3) Gram spends ~9 n^3 to find ~100 directions: measured, the table
took LONGER TO FACTOR than the operator took to run, 25 min and still going at
scale 1. A sketched range finder with one subspace iteration is all GEMM and
brought it to 180 s, which is also N-independent. `FMM_M2L_SKETCH = 256` caps
the rank it can find and clears the measured worst (160 for T, 102 for U); a
block that reaches the sketch is kept DENSE rather than silently truncated.

**The gate distinguishes two things it would have been easy to conflate.** The
numba kernel and the DENSE table are rearrangements of the reference and are
held at `FMM_M2L_VARIANT_PARITY` (measured 1e-15). The COMPRESSED table is an
approximation, so it is gated against the dense table -- isolating truncation
from rearrangement -- at its own tolerance with slack for accumulation through a
traversal. Holding it at 1e-12 would have asserted that the compression does
nothing.

**THE X LIST IS THE ONLY W/X ITEM WORTH TAKING, AND IT IS WORTH 1.2-1.3x --
NOT "the largest single item".** Node-pair counts by list, with the per-group
p^3 (U runs p = 6, T p = 8; a first pass that used one p^3 for both, and the
tree's total subtree rather than the pair's own, inflated the case):

    U 0.1 %    V 97.7 %    W 1.1 %    X 1.1 %

So the recorded "M2P/P2L are 9.4-16.9x more work than direct" is a ratio WITHIN
those lists, not their share of the operator. Re-measured with the two traps
fixed it is 8.8-16.9x for W and 12.0-24.4x for X.

**W is noise and X is not, and the reason is the kernel.** X carries 1.1 % of
the node-pairs at ~22 % of the time, because `_exact` integrates each source
triangle ANALYTICALLY to p^3 target nodes, ~25x the cost of a point evaluation;
W goes through the point kernel and is ~1 %. Demoting W would add 13.3 M near
pairs at scale 3 for under 1 %. Only X is demoted.

**The quadrature alternative was measured and LOSES.** Replacing the analytic
P2L with an n x n Duffy rule through the point kernel, on real X entries:

    kernel  rule        worst rel   median     speed-up
    U       n=2 (4 pt)   8.1e-04    2.4e-04      1.7x
    U       n=3 (9 pt)   1.1e-05    1.9e-06      1.4x
    T       n=2          3.5e-02    2.4e-03      4.0x
    T       n=3          2.8e-03    1.0e-04      2.5x

T at nine points is still 2.8e-03 worst case, **14x over FMM_OPERATOR_PARITY**,
for 2.5x. The "6-point rule, 5.8x cheaper" recorded from the H path is the right
order per evaluation and does not survive T's accuracy requirement here.

**Demotion is better than "cheaper": it makes X FREE.** An X entry re-evaluates
the analytic kernel EVERY matvec with no cache, while the U list it lands in is
cached per material. Caching X as it stands is the trap -- its blocks are
(3p^3, 3 res), 16.9 GB at scale 3 against demotion's 0.8 GB, which is the
"~1.6 TB at 4M" recorded elsewhere. Demotion is cheap precisely BECAUSE the
subtree is small, the same fact that made the expansion wasteful.

Measured (`FMM_X_MIN_SUBTREE = 216`, the smaller of the two lattices, so no
entry is demoted that the U kernel would still have won on):

    scale  N          matvec              X entries      near pairs        delta
      1    31,098     10.05 -> 7.79 s     6,041 -> 82    5.67 -> 7.40 M    4.9e-06
      2   117,120     42.94 -> 35.83 s   18,535 -> 602  20.0 -> 24.9 M    3.4e-06

1.29x and 1.20x, for +31 % / +24 % of near field. The delta is against a 2e-4
limit and is in the direction of EXACTNESS, which is worth something on its own
where the scale-3 margin is 1.1x.

The rule lives on `FmmTree`, not `InteractionLists`, and the default there is
OFF: whether an expansion beats direct depends on p^3, which the tree does not
know. `verify_octree` [g] pins the MARGIN rule with the size rule off -- on the
gate's small model the size rule refuses everything, which would have made the
selectivity checks vacuous -- and [h] pins the size rule's own endpoints: no-op
at 0, drains X at infinity, `U + V + W + X = N^2` EXACT at both and at the
shipping limit.

**THE SHARED M2L TABLE IS BUILT, CORRECT, AND WORTH 1.5-1.7x -- NOT THE 6.5x
THIS FILE PROJECTED** (`la/fmm_table.py`, `PairFMM(m2l="table")`). Tenth entry
for the table of collapsed claims, and the same shape as the others: a phase
measured in isolation, projected end to end.

What is right. The block is `K(H (u_t - u_s - 2d))` at unit half-width, one per
transfer OFFSET, and it reproduces `_far_apply` to **4.1e-15** over two kernels x
three Poisson ratios x p in {4,5} x four offsets x three half-widths. The level
folds out **BITWISE** -- halving every length multiplies a pass by exactly
`2**degree`, measured rel 0.000e+00 -- and the degree is the radial power for U
(1, 3) but **power - 1 for T** (2, 4), every T term carrying one more factor of d
upstairs. Using the power for T is exactly 2x per level of reuse; `verify_fmm`
[b] gates it with `np.array_equal`, which is the only limit that catches it.
At the operator the table agrees with the matrix-free kernel to 1.1e-15/2.2e-15.

What is wrong is the projected speed-up. Measured end to end on topo_inclusion,
canonical, whole far-field matvec, the ENTIRE table resident:

    scale 1 (31,098)  p=6/6  numba  3.24 s -> table 2.22 s   1.46x
    scale 2 (117,120) p=6/6  numba 12.58 s -> table  7.49 s   1.68x

Two reasons the 6.5x was never available. **Amdahl**: profiled after the numba
kernel landed, M2L is 73 % of the far field (`far_apply` 7.46 s of 10.26 s) and
the X list's exact triangle kernels are 22 % (`_exact` 2.21 s), so even a free
M2L caps the far field at ~3.7x. And the **batch spread**: the 6.2x per-pair
figure came from a uniform m = 114-512, while the real offsets have a median of
58 at scale 1 against a V-weighted mean of 404 -- the mean is carried by a few
wide keys, and the many narrow ones run at the skinny-GEMM end. The projection
used the V-weighted mean as if every key had it.

So the table is a real 1.5-1.7x that grows slowly with N, bought for ~8 GiB per
(material, kernel) at p = 6 and 33 GiB at p = 8. **It is per material**, because
`_far_params` folds the coefficient vector into the kernel, and a model with a
host and a few inclusions therefore builds one table per region: at the gate
model six tables compete for the cap and only 75 of 361 T offsets stay resident.
Before this is worth its memory at 4M it wants the material-free geometry basis
(14 U + 26 T scalar p^3 x p^3 tables, 57 MiB per key and ONE copy for every
region), which is measured but unwritten. Until then `FMM_M2L_TABLE_MAX_BYTES`
caps it and anything past the cap falls back to the matrix-free kernel, which is
exact and merely slower -- the table trades speed for memory, never accuracy.

**The next far-field target is the X list, not M2L.** 22 % of the far field is
`_exact` re-running the analytic triangle kernels every matvec, with no cached
form; that is now the largest single item after M2L and it is untouched.

**The block-Jacobi APPLY is threaded, and the knob that sets its chunk now
works.** Two separate defects, both silent.

The knob: `bj_chunk` was bound from `defaults` in the SIGNATURE, so rebinding
`defaults.PRECOND_BJ_CHUNK_DOF` at runtime was inert -- demonstrated directly, a
def-time binding still returns 9000 after the module attribute is set to 750 --
and `AssembledH.solve` had no chunk kwarg at all. Both fixed by the idiom the
same constructor already used for `max_dense`: `None` in the signature, resolved
in the body. `check_bj_rung` now pins that the knob is LIVE through both routes
(1500 -> 4 chunks, 750 -> 8 by kwarg and 8 by rebinding), which fails if it ever
silently stops working. The default stays 9000: chunk 1000 is the 4M setting and
is strictly worse at today's sizes (69 iterations and 29.2 s against 42 and
20.6 s at 260,598).

The apply: a serial Python loop over thousands of independent `lu_solve` calls
into DISJOINT index sets. `scipy`'s `lu_solve` releases the GIL -- measured
against a GIL-bound control through the same harness, which stayed at 1.02-1.10x
where the real call reached 7.2x -- so the loop threads on a process-wide pool
with a static contiguous partition, bitwise identical to serial and gated with
`np.array_equal`.

Three things the adversarial pass corrected, each of which would have gone into
the record wrong:

* **scipy's LAPACK here is Apple ACCELERATE, not the OpenBLAS numpy links.**
  `OPENBLAS_NUM_THREADS` and `threadpoolctl` do not govern the timed call at all,
  so the "pinned vs unpinned identical" experiment compared two identical
  configurations. The conclusion survives by the correct route -- with one
  right-hand side `lu_solve` is a level-2 solve measuring cpu/wall = 1.00 pinned
  or free, so there is nothing to oversubscribe -- but the reason is a property
  of the operation, not of thread policy. Note `lu_factor`, the BUILD, *is*
  threaded by Accelerate (cpu/wall 2.44), which is why only the apply was serial.
* **The thread count was chosen at a chunk size the cluster tree never
  produces.** An early sweep used m ~ 700 and found a knee at 10 with 16 threads
  unstable; the tree's median bisection actually yields ~976 DOF leaves at the 4M
  target, where 12 threads give 7.3-7.4x and 16 give 7.4-7.5x with no regression.
  `PRECOND_APPLY_THREADS = 12`, the performance-core count.
* **The microbenchmark timed bare `lu_solve`, not the apply's shape.** The real
  `solve_fn` does `z_inter[dof] = lu_solve(lu, r_inter[dof])`, two fancy-index
  gathers per chunk that HOLD the GIL: 5.7x rather than 7.1x at 12 threads.

And the honest headline, which is smaller than the loop's own speed-up: the
`lu_solve` loop is only 65 % of the apply. The other 35 % is the Gauss-Seidel
off-diagonal matvec, already numba `parallel=True`. Measured on the REAL apply,
topo_inclusion at chunk 1000, whole `M(r)`, bitwise identical at every count:

    scale  unknowns  chunks   serial    x4     x8    x12    x16
      2     117,120     189   53.4 ms  1.74  2.01   1.96   1.96
      3     260,598     396  114.7 ms  1.89  2.27   2.22   2.09

**2.0-2.3x on the apply, not the 7x the loop gets**, knee at 8 threads on both
scales and a regression by 16 on both. That is Amdahl on the 65/35 split -- and
the agreement between the predicted 2.2x and the measured 2.01/2.27x is an
independent confirmation of the split itself. `PRECOND_APPLY_THREADS = 8`: the
synthetic sweep over bare `lu_solve` said 10-12, and it was measuring the wrong
thing at the wrong size. The preconditioner goes from ~16 % of a 12 s far-field
matvec to ~7 %; against a 1 s matvec it is still most of the cost, and the next
target there is that off-diagonal matvec, not the loop.

**Two more silent knobs of the same shape, both fixed.** `AssembledH` cached
`_precond` and rebuilt it only when it was `None`, so a chunk sweep over ONE
assembled operator measured its first arm at every later point -- and agreed
with itself wherever the caps happened to coincide, which is how it survived.
The ladder caps are now part of the cache key. And the first version of the
gate exercised `BlockGaussSeidel` directly, so it would have stayed green if the
`solve()` plumbing were deleted; it now sweeps the chunk through `solve()` on one
operator, which fails if either the kwarg or the rebuild goes away.

Batching was measured and REJECTED: a hand-stacked batched `getrs` is 3.4x
slower than the loop and not bitwise (2.3e-15), and `np.linalg.solve` over a
stacked array is 20x slower because it refactorizes. Bucketing is not the
obstacle -- the tree produces only 9 distinct chunk sizes over 396 chunks -- a
single-RHS triangular solve is simply bandwidth-bound and the loop already
streams the factors once, contiguously.

**A landmine in any chunk sweep**: `bj_chunk` is not a kwarg of
`AssembledH.solve`, and `BlockGaussSeidel.__init__` binds
`defaults.PRECOND_BJ_CHUNK_DOF` at def time, so rebinding the default after
import is a silent no-op -- a sweep done that way reports "chunk 500" while
measuring 9000, and is invisible at the 9000 point because 9000 is the default.
Related: the shipping ladder at 31,098 is 23 iterations, not 27, because all six
super-blocks sit under the 30,000 dense cap and none reaches block-Jacobi; 27 is
the FORCED arm. `defaults.py`'s block-Jacobi comparison column labels that forced
31k row as "dense cap then block-Jacobi", which it is not.

**Re-measured with the rule as implemented** (real re-partitioned matvecs, all
rows, p = 6/8, worst over the test vectors; iso on a FIXED denominator, the
rule-off partition's `||A_far v||`):

  config                    iso: no rule   with rule   naive: no rule  with rule
  x1 extent saf 1.0 (3v)      7.2011e-04  1.7488e-04 ok    3.0749e-04  1.2537e-04 OVER
  x1 extent saf 1.5 (3v)      1.1479e-03  8.6754e-05 ok    4.4787e-04  1.6352e-05 ok
  x1 cube   saf 1.0 (3v)      4.2074e-03  8.9874e-04 OVER  1.8211e-03  1.6884e-04 OVER
  x2 extent saf 1.0 (1v)      8.8662e-04  1.3254e-04 ok    1.3058e-04  1.9519e-05 ok
  x3 extent saf 2.75 (1v)     1.0380e-03  1.2074e-04 ok    7.0441e-05  8.1939e-06 ok
  x3 cube   saf 2.75 (1v)     3.0614e-03  9.4206e-05 ok    2.0775e-04  6.3930e-06 ok

So **extent + the rule passes at scale 3** and the cube-vs-extent question does
collapse into the X rule: both domains land at 1e-04 iso and ~7e-06 naive at
260,598 unknowns, 8.6x and 32.5x better than without it, and every value
reproduces the sweep's whole-entry substitution to five digits (1.3327e-04,
1.0398e-04, 8.6754e-05). The price is 0.01-0.3 % of the U list against all-X-
direct's +37 to +68 % (x1 extent 1.5: 4.1292 -> 4.1346 % with the rule,
6.9428 % all-direct, at the same absolute error).

**The placement constant is settled: `OCTREE_PLACEMENT_SAFETY = 2.0`, the same
for both domains.** Swept 1.0 / 1.5 / 2.0 / 2.5 with the X rule active, on
topo_inclusion at all three scales and both domains, iso against ONE fixed
reference partition (safety 1.0, rule off) because safety is what moves the
partition. Worst margin over both limits and both domains, 1.0x being the
limit itself:

  safety   scale 1      scale 2      scale 3
  1.0      FAIL 0.22x   FAIL 0.36x   --
  1.5      pass 2.6x    FAIL 0.55x   FAIL 0.62x (cube; the extent passes)
  2.0      pass 2.7x    pass 1.3x    pass 1.1x
  2.5      pass 2.8x    pass 1.8x    pass 1.1x

**1.5 is the trap, and it is exactly the shape the record warned about.** It
passes at scale 1 by 2.6x, fails at scale 2, and the failure is NOT monotone
in N -- at scale 3 the cube fails (naive 7.9e-05, iso 3.2e-04) while the
extent passes (3.6e-05 / 1.9e-04) -- so no single mesh, and no extrapolation
from two, would have found it. What makes 2.0 a bound rather than another fit
is the CUBE, whose failure IS ordered by protrusion: it passes at 0.329 and
fails at 0.372 and 0.405, bracketing its extrapolation limit in (0.33, 0.37),
and the sawtooth guarantee prot <= 0.67/safety puts safety 1.5 at 0.447 --
outside the bracket -- against 0.335 at 2.0, inside it. Measured protrusion at
2.0 is 0.273 / 0.286 / 0.285, i.e. c = 0.545 / 0.571 / 0.569, and
`verify_octree` [a] now gates the bound over a sweep including the shipping
default so it cannot quietly stop holding on another mesh. The EXTENT has no
such predictor: it fails at protrusion 0.372 (scale 2) and passes at 0.405
(scale 3), and its worst V-list node gap (0.674 against 0.628) does not order
the two either -- on that domain 2.0 rests on the three-scale measurement
alone, which is a reason to prefer the cube now that both pass.

Both metrics were needed and so were all three vectors. At scale 2 safety 1.5
reads iso 5.5e-05, PASSING, against naive 6.9e-05, FAILING, and the vector
that fails is the unit TRANSLATION (far field 1.25x its own `||A v||`), which
the earlier scale-2 and scale-3 runs never used.

Price, at 72 B per near element pair: safety 2.0 is 424 M pairs / 28 GiB at 4M
unknowns against 1.0's 267 M / 18 GiB (+59 %), 1.5's 304 M / 20 GiB and 2.5's
781 M / 52 GiB (+84 % over 2.0 at the same error). Measured near element pairs
1.64 M / 6.33 M / 17.37 M at safety 1.0 and 3.97 M / 13.30 M / 27.62 M at 2.0,
i.e. 52.8 / 54.0 / 66.6 against 127.0 / 113.6 / 106.0 pairs per unknown -- the
2.0 rate FALLS with N while 1.0's rises, so the ratio is 2.4x at scale 1 and
1.6x at scale 3, and the projection uses scale 3's.

**The scale-3 margin is 1.1x and placement cannot widen it.** Safety 2.0 and
2.5 have the same absolute error to four digits there (3.4743e-05 against
3.4740e-05, gaussian 1, extent), so what remains at 260,598 unknowns is the
interpolation ORDER and `FMM_OPERATOR_PARITY` is nearly saturated by it. C
inherits 1.1x of room, not 2x.

**The default no longer suits the gates' own models, and they now say so.**
On the 2,592-unknown fault zone, safety 1.5 empties the W and X lists
outright, 2.0 cuts V from 2588 to 384 box pairs, and 2.5 and 3.0 leave U at
100 % -- no far field at all, the operator clause dividing roundoff by
roundoff. `verify_fmm` and `verify_octree` therefore PIN `SAFETY = 1.0` with
that measurement beside it; every gate number is unchanged (19/19, clause [d]
iso 4.283e-05 / 8.007e-05 / 6.278e-06 and [g]'s margins bit for bit).

The denominator caution is now demonstrated rather than argued: at x1 extent
1.5, all-X-direct has 0.3 % LOWER absolute error than the rule and reads
1.1721e-04 against 8.6754e-05 because its own `||A_far v||` is 26 % smaller;
on the fault zone at cube x1.7, demoting 89 entries leaves the absolute error
bit-identical and moves iso from 7.6266e-04 to 1.3094e-03.

Also still open: demote W/V box pairs whose effective gap after protrusion is
<= 0 to direct (4-7 % of W box pairs, +5 % on U). Cost of running these: the
reference bbFMM matvec is single-core numpy -- 924-1836 s at 31,098 unknowns,
5518-6069 s at 117,120, 4545-4760 s at 260,598 (the largest is CHEAPER than the
middle because its U fraction is 1.21 % against 0.66 %) -- and the exact
matrix-free reference is 20 s / 191 s / 393 s beside it. Six configurations run
concurrently is the way to afford a table like the one above.

Two facts about this model that the earlier record got wrong: the size grading
is driven by the FINE end (host_top at 0.177 km; the max edge at scale 3 is
36.4 km, not 88.9, and host_base is 0.89 % of the near field because it sits
200 km from everything), and a box can be occupied AND internal -- a pinned
element interacts directly with its own box's whole subtree, which costs
0.21-0.28 KiB/unknown and is what makes W/X nonempty here. Subdivide on the
TOTAL element count exceeding ncrit, not on the descendable count.

Still unmeasured: nobody has assembled an FMM operator and compared it end to
end against `AssembledH`. Every error above is per-block.

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
