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
