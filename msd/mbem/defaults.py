"""Single source of truth for mbem tolerances and thresholds.

Every numeric default a solver, compressor, or pipeline uses lives here,
and every name here is read by some caller, so the whole stack can be
audited in one place. Solution accuracy target ~1e-6; linear-algebra
layers run with ~100x margin under it.
"""

# --- Accuracy targets -------------------------------------------------
SOLUTION_RTOL = 1e-6          # end-to-end solver accuracy target
GMRES_RTOL = 1e-8             # true-residual stop (100x margin)
GMRES_RESTART = 200
GMRES_MAXITER = 600
# Stop FGMRES if the residual fails to improve by STAGNATION_FACTOR
# over STAGNATION_WINDOW iterations.
GMRES_STAGNATION_WINDOW = 100
GMRES_STAGNATION_FACTOR = 10.0
# How fast the preconditioned iteration count may grow with the problem
# size, as the exponent alpha of iterations ~ N^alpha between two meshes
# (verify_hbackend.py, convergence-rate check). An exponent, not a ratio,
# because the ratio only means something at one fixed size step and the
# ladders this is measured on step by 1.7x to 2.8x.
#
# On the EXACT rungs the count is the number of outlier eigenvalues of
# A M, not a function of N: alpha = 0.05 on the gate's fault-zone pair
# (18 -> 19 over 2.84x). The rung that carries every super-block past the
# dense cap, cluster block-Jacobi, does grow, because its chunk is fixed
# while the block is not: 27 / 34 / 37 / 42 iterations at 31k / 68k /
# 117k / 269k unknowns on topo_inclusion, i.e. alpha 0.29 / 0.15 / 0.15
# per step and 0.20 end to end over 8.7x, and 0.315 (18 -> 25) on the
# gate's own pair with the chunk cut to 1000 DOFs so that the rung
# actually subdivides there. 0.4 is that worst measurement plus the two
# iterations of drift BENCH_ITER_SLACK allows elsewhere (18 -> 27 is
# alpha 0.389); it is not slack for a preconditioner that does not
# scale -- the same check seeds one, a 96-DOF chunking, and requires
# this criterion to REJECT it (21 -> 41, alpha 0.641).
GMRES_ITER_GROWTH_ALPHA = 0.4
# ... and never past this absolute count on the gate's models: beyond it a
# ladder rung, not the operator, is what changed.
GMRES_ITER_CEILING = 40
# What shares a diagonal super-block of that ladder
# (la/preconditioner.group_slots): "patch" (one per patch, an interface's
# (u,t) pair together) or "region" (one per region, each interface owned
# by its inclusion, so the block is a closed single-region surface).
# "region" buys ITERATIONS and spends LADDER BUILD, measured on the
# inclusion ladder (10.9k / 29.7k / 83k unknowns, rtol 1e-8): 22 / 24 / 25
# iterations over a 0.5 / 9 / 408 s build with "patch" against 9 / 10 / 11
# over 1.3 / 273 / 4925 s with "region". The count is material-insensitive
# (9-10 at both mu_host/10 and mu_host/100), but the region block holds
# 72-76 % of the unknowns, so past MAX_DENSE_PRECOND_DOF only the HODLR
# rung can solve it and that build is superlinear (a 63k-DOF block is
# 4473 s and 3.8 GB). Iterations are not the currency there: the 14 the
# grouping saves at 83k are 3 s of FGMRES against 4500 s of extra build.
# So the default is "patch"; "region" is for a model whose region blocks
# still fit the dense-LU rung (~1e4 unknowns, where it costs ~1 s and
# halves the count) or for a fixed iteration budget (an expensive
# matvec). Its rung must be direct-type either way: with the block-Jacobi
# rung on the region block the count goes to 29 / 37, worse than "patch",
# so past the dense cap it also needs PRECOND_RUNG_ABOVE_DENSE = "hodlr"
# and the build that costs. The merged block's HODLR rank is NOT what
# stops it -- 690 at 29.7k and 1310 at 83k against 606 and 840 for the
# largest interface block, inside the 2x the work plan allowed.
PRECOND_GROUPING = "patch"

# --- Material sweeps: preconditioner reuse and Krylov recycling --------
# A sweep solves one geometry at many materials. The ladder's diagonal
# factorizations are the expensive part of a rebuild and they are STALE,
# not wrong, at a nearby material, so ``rebuild_for_materials`` keeps the
# preconditioner while every changed region's moduli stay within this
# log-step of the ones it was BUILT from (drift is measured against the
# build, not against the previous step, so a slow sweep still rebuilds).
# What a stale ladder costs at a 2x step, measured with the warm start
# on: +1 iteration on the 10.9k inclusion model, +2 on the 31k topo
# model, +3 on the 2.6k fault-zone model (+5 there without the warm
# start). What it saves is the whole ladder build, which is most of a
# solve's wall time at scale: 2.5x per solve at 10.9k (3.4 -> 1.4 s),
# 5x at 31k (18-26 -> 3.2-4.2 s).
PRECOND_REUSE_MAX_STEP = 0.6931471805599453   # = log 2
# What a kept ladder plus a warm start (x0 = the previous solution) may
# cost in iterations against a fresh build at the same material
# (measured +1 / +2 / +3 on the three models above; the warm start
# itself saves 2-4).
PRECOND_REUSE_ITER_SLACK = 4
# GCRO-DR recycled subspace dimension k (la/solver.RecycleSpace). The
# preconditioned operator has ~21 outlier eigenvalues, geometric and
# material-insensitive, and they are what harmonic-Ritz recycling
# harvests; 30 covers them with margin at 3 k n floats while a solve
# runs (30 x 900k x 8 B = 220 MB per array at 1M unknowns), k n stored
# between solves. Note what a recycled solve spends before its first
# iteration: k preconditioner applications and k matvecs re-deriving
# U = M Y and C = A U. At 20-24 baseline iterations that is more than
# the iterations it saves, so recycling buys iteration COUNT, not wall
# time, on these models.
GCRO_RECYCLE_DIM = 30
# Recycling is therefore opt-in per solve (``solve(recycle=True)``), not
# the default: a single solve gains nothing from it (measured: the same
# iterations and the same solution to 6e-15, plus the harvest) and pays
# the setup on every solve after it.
GCRO_RECYCLE_DEFAULT = False
# Ceiling on solves 2..N of the material sweep verify_hbackend runs
# (the 2.6k fault-zone model), where recycling measures 15 against 20
# fresh and 19-23 reused: 18 is that with margin for an operator change.
# It is NOT a universal count -- recycled sweeps measure 13-17 on the
# 10.9k inclusion model (22 fresh) and 15-22 on the 31k topo model (23-24
# fresh), the spread being the ladder's staleness at the solve, and the
# 12 of the work plan is reached only with a ladder rebuilt every solve
# (12-15 at 10.9k). The model-independent statement is the relative one
# the same check gates: recycling must never cost iterations against the
# same sweep without it.
GCRO_SWEEP_ITER_MAX = 18

# --- Conditioning ------------------------------------------------------
# Warn when a dense solve's 1-norm condition estimate exceeds this.
# cond * machine-eps ~ residual amplification: 1e10 * 2e-16 = 2e-6 is
# right at SOLUTION_RTOL, so anything above it can silently miss the
# accuracy target (thin panels / near-fluid materials / the spurious
# discretization resonance all push cond up long before LU "fails").
COND_WARN_THRESHOLD = 1e10

# --- Compression ------------------------------------------------------
# Relative Frobenius tolerance of every admissible block of the FAST
# OPERATOR (the compressed pairs FGMRES applies); the solve's own stop,
# GMRES_RTOL, is separate and stays 1e-8. 1e-4 is the operator target:
# the per-basis rank of a far-field T block drops 42 -> 16 against 1e-8,
# the preconditioned iteration count is the same at 1e-6 and 1e-4, and a
# 1e-4 operator error at 2-norm condition ~1e3-8e3 leaves ~2e-5 in the
# solution.
BLOCK_COMPRESSION_TOL = 1e-4
# Elements per leaf cluster. The admissible list is the same at 32 and
# 96 (eta 2) while the dense near-field leaf count drops 19k -> 1.9k at
# 10k elements: fewer, larger leaves, fewer per-block launches.
CLUSTER_MIN_LEAF = 96
ADMISSIBILITY_ETA = 2.0
# Admissible blocks smaller than this (elements per side) are stored
# dense: at small sizes the epsilon-rank is a large fraction of the
# block and cross approximation cannot be certified by sampling.
ACA_MIN_BLOCK = 64
# Cap on admissible block side (elements). Bounds what a block costs
# when it is applied EXACTLY (rank-capped, or still uncertified after the
# retry): one dense material block is (3 n)^2 x 8 bytes, 75 MB at 1024
# and 1.2 GB at 4096, and its kernel re-evaluation is per material. The
# cap is not free: splitting an admissible pair of side s into (s/1024)^2
# blocks multiplies its ACA's row/column evaluations by s/1024, which is
# 17 % of the whole ACA line budget at 117k unknowns and grows with size.
# Raise it only for smooth, well-separated geometry.
MAX_ADMISSIBLE_BLOCK = 1024
# ACA must converge within this fraction of full element rank, else the
# block is not low rank at this tolerance and is stored as its exact
# stack, like a near-field leaf (at eta 2 about a third of the
# 64-127-element blocks; factors of that rank are no smaller than the
# stack, and the SVD that used to follow bought nothing).
ACA_MAX_RANK_FRACTION = 1.0 / 3.0
# Certificate of every ACA block: this many full random rows and as many
# full random columns, evaluated exactly -- 2 x 3 (nr + nc) kernel pairs,
# negligible against the ACA's k (nr + nc) -- give an unbiased estimate
# of the block's relative Frobenius error (a fixed 10 x 10 sample did
# not): per basis at build, and per material when its view is combined
# (a per-basis bound does not cover a mixed-sign combination). A block
# is certified below CERTIFY_FACTOR x the tolerance: the ACA stops at
# half the tolerance and the recompression may add the other half.
ACA_CERTIFY_LINES = 3
ACA_CERTIFY_FACTOR = 3.0
# A failed certificate says the sampled stop fired early, not that the
# block has no low-rank form, so the ACA is RE-RUN once at the tolerance
# divided by this before the block is applied exactly. Measured on the
# 21 failures among the 324 admissible blocks of side >= 200 of the 31k
# topo model: the errors are marginal (3e-4 to 1.1e-3 against the 3e-4
# limit), a retry at 10x certifies 19 of the 21 at 1.4x the rank and
# 0.02-0.4 s, and a retry at 100x certifies fewer (3 hit the rank cap)
# at 1.9x the rank. The retry fires on ~2 % of blocks, so its cost is in
# the noise; what it replaces is not (see ACA_SVD_FALLBACK_MIN_SIDE).
ACA_RETRY_TOL_FACTOR = 10.0
# The B factor pairs of a block are folded into ONE shared subspace
# (aca.shared_subspace) whose truncation reproduces every basis to this
# fraction of the block tolerance. It is spent out of the block's error
# budget, so it must be small against it, and it buys both the material
# recombination (a k x k sum and SVD instead of a QR of the (3n, sum_b
# k_b) factors: 134x on 1024-element T blocks) and the stored far field
# (the joint rank is a fraction of the summed per-basis rank, the bases
# of one kernel spanning nearly the same row/column spaces). Measured on
# 1024-element far-field T blocks, joint/summed rank at 1 / 0.3 / 0.1 /
# 0.03 x the tolerance: 0.29 / 0.37 / 0.41 / 0.46, with the combined
# per-material rank (15.8) and the certified block error (8.4e-5)
# unchanged at every setting. 0.1 costs 1.4x the rank of the loosest
# setting and leaves the truncation an order below the tolerance: the
# combination it must survive can cancel (mixed-sign T coefficients),
# and sum_b |c_b| ||A_b|| / ||sum_b c_b A_b|| is what the error is
# multiplied by -- measured 1.5 over those blocks, so 0.1 holds to a
# cancellation factor of 10.
ACA_JOINT_TOL_FACTOR = 0.1
# Panel width of the fold's block Gram-Schmidt (la/fold_numba). The point of
# blocking is BLAS-3: a panel's two orthogonalization passes are four GEMMs,
# where the same algorithm one column at a time is BLAS-2 and measured 3.7x
# numpy's QR on production shapes.
ACA_FOLD_PANEL = 48
# Residual Schur-complement trace the fold's pivoted Cholesky deflation stops
# at, as a fraction of the truncation budget delta^2. The deflation exists
# because a hand-rolled cyclic Jacobi is 23x eigh and O(n^3), so the kernel
# stands or falls on the eigenproblem's SIZE; it cuts n by ~1.3x per side.
# The residual trace is carried as already-spent budget, so the truncation
# decision is the reference's -- measured EQUAL keep counts on all 90 real
# Gram matrices of 45 blocks x 2 sides. 1e-2 is small enough for that and
# large enough to deflate.
ACA_FOLD_CHOL_SLACK = 1e-2
# rcond of the ACA pivot block's pseudo-inverse (aca._pinv3, mirrored in
# aca_numba): a singular value below this fraction of the leading one is
# treated as zero, and a 3x3 pivot block with |det| <= rcond x ||P||_F^3
# falls back from the closed form to that SVD.
ACA_PINV_RCOND = 1e-12
# H-vs-dense parity gates (verify_hbackend, verify_deflation_estimate,
# demo_hmatrix), as multiples of BLOCK_COMPRESSION_TOL: operator and RHS
# ENTRIES within H_PARITY_OPERATOR x tol (each block is certified at
# ACA_CERTIFY_FACTOR x tol of its own norm, the view adds up to tol, and
# an entrywise max over a matrix of many blocks lands within an order of
# that: 5 x measured on the pair check); SOLUTIONS per slot within
# H_PARITY_SOLUTION x tol (2-norm condition ~1e3 x the operator error;
# the same ratio as BENCH_SOLUTION_ERROR_MAX / BENCH_OPERATOR_ERROR_MAX).
# Rebuild-vs-fresh, combined-vs-basis and determinism stay bitwise.
H_PARITY_OPERATOR = 50
H_PARITY_SOLUTION = 10
# The calibrated diagonal is built from the COMPRESSED row sums, so the
# block tolerance enters the operator's free term directly: |C_h -
# C_exact| is bounded by this multiple of tol x COLLOCATION_JUMP
# (verify_hbackend, calibration check). Nothing else bounds it -- an
# entrywise parity is relative and per block, while a row sum adds every
# block of a row and is compared with a free term of 0.5. Measured 0.059
# of tol x the free term on the gate's own refined fault-zone model, so
# 0.25 is that with a 4x margin, and the seeded arm it must reject -- the
# same model at a 100x looser tolerance -- lands 64x the limit. It is a
# gate on that model, not a universal bound: the topo_inclusion ladder
# measures 0.3-0.7 of tol x the free term at 31k-261k unknowns, growing
# with the number of admissible blocks a row crosses.
H_PARITY_CALIBRATION = 0.25
# A block that fails its certificate TWICE and whose caller needs a
# payload (the HODLR rung) is re-done from its exact stack: kept dense
# below this many elements on its shorter side, SVD-truncated above it.
# The OPERATOR path never comes here -- it applies such a block from the
# kernels instead. The reason is cost: the SVD is B x O(m n min(m, n))
# at one BLAS thread, 13 s on a 404-element T block, and at 31k unknowns
# three such blocks were 39 of the 76 CPU-seconds of all block
# compression; at 117k, where admissible blocks reach 756 elements, the
# same 3 % failure rate over 3,173 large blocks made assembly unfinishable.
ACA_SVD_FALLBACK_MIN_SIDE = 256
# Two implementations of the same seeded ACA -- aca._aca_python and the
# nogil kernel of aca_numba -- pivot identically (same lines from the
# same seed, same arithmetic order where it decides a pivot), but their
# recompressions call different LAPACK builds (numpy's and scipy's), so
# where the truncated singular values are nearly tied they discard a
# different tail: one tolerance's worth, and a rank that differs by at
# most one. Their factors are therefore compared -- with each against
# the exact block -- at this multiple of BLOCK_COMPRESSION_TOL
# (verify_hbackend; measured 1.1 x over the gate's blocks, and 1.3e-14
# where the truncation is not degenerate), never bitwise. Each
# implementation IS bitwise repeatable on its own, which is what the
# determinism check covers.
ACA_IMPL_PARITY = 4.0
# Per-material recombined views cached per PairCompressed (LRU), bounded
# by count and by bytes. The count must exceed the number of DISTINCT
# coefficient vectors one solve applies to a pair (preconditioner
# included) or every matvec re-runs the recompression; 16 covers models
# with up to ~8 regions. The bytes bound is what stops a material sweep
# from holding one operator per material: a combined T view is ~0.5 GB
# per pair at 100k unknowns, so a sweep keeps ~4 materials per pair
# (~20 GB over a 10-pair model) instead of 16. The MIN_KEEP most recent
# views -- the two region materials of an interface pair, one solve's
# working set on that pair -- are never evicted by the bytes bound.
HOP_VIEW_CACHE_MAX = 16
HOP_VIEW_CACHE_MAX_BYTES = 2_000_000_000
HOP_VIEW_CACHE_MIN_KEEP = 2
# The flat view's matvec (la/flatview.py) cuts the output DOFs into this
# many contiguous equal-work row chunks. Each chunk OWNS its rows, so the
# result is bitwise identical at every thread count, and the cut is fixed
# by the view rather than by the thread count. It is an upper bound --
# 16 chunks per core here, enough that the ragged per-row work of a block
# partition balances without the chunk CSR growing; the work floor below
# sets the count actually used.
FLATVIEW_ROW_CHUNKS = 256
# ... but never more chunks than this much work each: a chunk re-gathers
# the x entries of every dense block it touches, so an over-cut view pays
# the gather many times over (a 0.85 M-multiply-add pair lost 2.5x to 256
# chunks). 50 k multiply-adds is ~15 us of work, well above a thread's
# share of one fork/join.
FLATVIEW_MIN_CHUNK_WORK = 50_000
# Below this many multiply-adds a view's matvec runs in ONE thread: an
# OpenMP fork/join costs 0.1-0.3 ms here, and a small pair pays it on
# every term of every iteration (the fault box at 1.1k unknowns: 2.7 ms
# of launches, 0.3 ms once its nine terms went serial). 1e6 is a few
# tenths of a millisecond of work, an order above the launch.
FLATVIEW_PARALLEL_MIN_WORK = 1_000_000
# Flat view vs the block loop it replaced (verify_hbackend): the two sum
# the same terms in a different order -- numba loops against BLAS dot,
# the leaf kernel's in-loop coefficient combination against tensordot --
# so the gate is round-off, not bitwise (measured 6e-16 over the
# fault-zone views and 2e-15 over the 10.9k inclusion model's); what
# stays bitwise is the flat matvec across thread counts.
FLATVIEW_PARITY = 1e-12
# STORAGE precision of a flat view's factors and dense leaves, chosen from
# the block tolerance the view was compressed at -- single when that
# tolerance is this coarse or coarser, double otherwise. Arithmetic stays
# float64 everywhere: the scratch W, the output y and the gathered x slice
# are double, so every accumulation is double and only the stored operand
# is single (la/flatview.py). float32 carries ~6e-8 relative, so at the
# default 1e-4 the storage error is 500x under what the compression
# already spends -- measured on topo_inclusion, the operator error is
# unchanged to four digits at 31k / 117k / 261k unknowns (2.791e-5 /
# 2.891e-5 / 1.397e-5 double against 2.791e-5 / 2.891e-5 / 1.398e-5 with
# BOTH the factors and the near field single) for half the bytes. The rule
# is tied to the tolerance rather than exposed as a free switch because a
# convergence study at 1e-8 would otherwise be storage-limited without
# saying so: 1e-5 keeps at least a 100x margin over float32's own floor.
STORAGE_SINGLE_MIN_TOL = 1e-5
# The same flat view stored single against stored double (verify_hbackend):
# what the storage precision itself costs, measured separately from the
# code-equivalence check above so neither hides the other. FLATVIEW_PARITY
# stays 1e-12 on a float64-pinned view -- that clause is about the numba
# kernel reproducing the block loop, and it must not be loosened to make
# room for rounding. Measured 6.9e-8 on the fault-zone views and 8.9e-8 on
# the 10.9k inclusion model, i.e. float32's own floor; 1e-6 is that with
# an order of margin, and still 100x under BLOCK_COMPRESSION_TOL.
FLATVIEW_STORAGE_PARITY = 1e-6

# --- Adaptive octree, the FMM far field's tree (la/octree.py) ----------
# Elements per box before it subdivides. Measured on topo_inclusion at scale 3
# (66,842 elements), near field in KiB/unknown as U / U+W+X: 16 -> 2.01 / 4.40,
# 32 -> 4.69 / 8.18, 64 -> 7.45 / 15.83, 128 -> 19.25 / 34.38. 16 saves 5 GiB
# at 4M unknowns and DOUBLES the M2L count (335,766 box pairs against 165,698),
# which is not worth it out of 128 GB; 64 nearly doubles the near-field kernel
# pairs (12.7 M -> 23.2 M) for a 38 % M2L saving; 128 is worse again, its leaf
# occupancy reaching 52. At 32 the tree is depth 9 with mean occupancy 14.4.
OCTREE_NCRIT = 32
# An element may go no deeper than the level whose cube edge is this multiple
# of its own size. It is the FAR FIELD's accuracy constant, not a tree-shape
# preference, because what it controls is PROTRUSION -- how far an element
# hangs outside its own box, in box edges -- and protrusion breaks both
# interpolation domains at once. On the nominal cube a protruding source makes
# P2M an EXTRAPOLATION; on the box's contents extent, the only domain that
# contains it, the source domain grows to f_source ~ 2 box edges, and since two
# non-adjacent boxes are 2 edges apart centre to centre their M2L node sets
# then nearly touch (worst measured V-list gap 0.55 box edges at safety 1.0
# against 0.83 at 2.75, while the cube's stays 1.02 at every safety).
# Protrusion obeys a sawtooth bound prot <= c / safety, c being an element's
# own reach past its centroid over its longest edge: measured c = 0.49-0.61
# over topo_inclusion's three scales, under the 0.67 a sliver can reach.
#
# 2.0 is the smallest value meeting BOTH FMM_OPERATOR_PARITY and
# FMM_OPERATOR_PARITY_NAIVE at EVERY scale and on BOTH domains with the X rule
# (FMM_X_MARGIN) active. Measured on topo_inclusion at 31,098 / 117,120 /
# 260,598 unknowns, p = 6/8, all rows against the matrix-free exact operator,
# worst over the three test vectors, iso divided by ONE fixed reference
# partition (safety 1.0, rule off) because safety is what moves the partition.
# Worst margin over both limits and both domains, so 1.0x IS the limit:
#
#   safety   scale 1      scale 2      scale 3
#   1.0      FAIL 0.22x   FAIL 0.36x   --
#   1.5      pass 2.6x    FAIL 0.55x   FAIL 0.62x (cube; the extent passes)
#   2.0      pass 2.7x    pass 1.3x    pass 1.1x
#   2.5      pass 2.8x    pass 1.8x    pass 1.1x
#
# 1.5 is the trap and the reason the value is stated per scale: it passes at
# scale 1 by 2.6x, fails at scale 2, and the failure is NOT monotone in N --
# at scale 3 the cube fails (naive 7.9e-05, iso 3.2e-04) while the extent
# passes -- so no single mesh would have found it. The binding vector is the
# unit TRANSLATION, whose far field is 1.25x its own ||A v||, and at scale 2
# the binding limit is the NAIVE one: safety 1.5 there reads iso 5.5e-05,
# passing, against naive 6.9e-05, failing. What makes 2.0 a bound and not
# another fit is the CUBE, whose failure is ordered by protrusion alone: it
# passes at 0.329 and fails at 0.372 and 0.405, so its extrapolation limit is
# bracketed in (0.33, 0.37), and the sawtooth GUARANTEE 0.67 / safety is
# 0.447 box edges at safety 1.5 -- outside the bracket -- against 0.335 at
# 2.0, inside it. Measured protrusion at 2.0 is 0.273 / 0.286 / 0.285 at the
# three scales, i.e. c = 0.545 / 0.571 / 0.569. The EXTENT has no such single
# predictor -- it fails at protrusion 0.372 (scale 2) and passes at 0.405
# (scale 3), and its V-list node gap does not order the scales either -- so on
# that domain 2.0 rests on the measurement at three scales, not on a bound.
#
# The scale-3 margin is 1.1x and it is NOT the placement rule's to widen:
# 2.0 and 2.5 there have the same absolute error to four digits (3.4743e-05
# against 3.4740e-05 on gaussian 1, extent), so what is left at 260,598
# unknowns is the interpolation ORDER, and FMM_OPERATOR_PARITY is nearly
# saturated by it. Anything built on this tree has that 1.1x, not 2x, of room.
#
# The price is the near field at 72 B per near element pair: 424 M pairs and
# 28 GiB at 4M unknowns, against safety 1.0's 267 M / 18 GiB (+59 %) and
# 1.5's 304 M / 20 GiB. Going past 2.0 buys nothing and costs a lot -- 2.5 is
# 781 M / 52 GiB (+84 %) at the same error. Strict containment, the other way
# to kill protrusion, pins 52 % of elements at 284 KiB/unknown (~250 GiB).
# 2.5 is also a ceiling on small models: it leaves the fault-zone model at
# 100 % U list, no far field at all (3.0 likewise), which is why the gates
# pin their own placement safety instead of reading this one.
OCTREE_PLACEMENT_SAFETY = 2.0
# The sawtooth constant of the bound above, prot <= OCTREE_PROTRUSION_C /
# safety, so the guarantee is stated once and policed (verify_octree [a])
# rather than only asserted in a comment. It is a triangle's own reach past
# its centroid over its longest edge, which is 0.47 for a right isoceles
# triangle and tends to 0.67 for a sliver -- geometry, not a dial; placement
# can only divide it by the safety factor. Measured 0.49-0.61 on
# topo_inclusion over safety 1.0-3.0 at three scales, and 0.33-0.39 on the
# fault zone.
OCTREE_PROTRUSION_C = 0.67
# Hard depth limit, so a degenerate cloud cannot recurse without end. The
# 1e6-element target reaches ~10 levels; 21 is what three packed integer box
# coordinates fit in a 64-bit key.
OCTREE_LEVEL_CAP = 21

# --- Chebyshev black-box FMM (la/fmm.py) ------------------------------
# Chebyshev nodes per dimension (p^3 per box). Measured per admissible block
# at the canonical worst offset (2, 0, 0): the single-layer U kernel reaches
# 1e-4 relative Frobenius at p = 6, the double-layer T kernel, one derivative
# higher and so one order less smooth, only at p = 8. At p = 8 the MAX-ENTRY
# error of the T kernel is still 1.3-3.0e-4, i.e. 1e-4 is met in the
# Frobenius norm alone -- a gate on this path reports both norms or it
# reports the easier one.
FMM_ORDER_U = 6
FMM_ORDER_T = 8
# Terms of K(eps) = K0 + eps^2 K1 + O(eps^4) the far field carries. Both are
# translation-invariant and K1 is homogeneous of K0's degree minus 2, so the
# second term is the same machinery at a different radial power with the
# source weight w_j eps_j^2. Two terms hold 1e-4 on 100 % of admissible
# blocks; one term fails it on 62.5 %. The expansion parameter is
# (eps/r)^2 with r the separation the expansion is actually EVALUATED at --
# M2L node to node, M2P field point to node -- and that is NOT the two box
# edges the placement rule bounds cube centres by: interpolating on the
# enlarged extents pushes the two node sets toward each other, so the worst
# V-list node separation measures 0.68 box edges against the cube domain's
# 1.02. Measured on the fault-zone model under eps="auto" (eps_j = 0.1 h_j),
# worst over every far-field evaluation: (eps/r)^2 = 1.2e-2 and a truncation
# (eps/r)^4 = 1.5e-4 on the extent domain, 5.5e-3 and 3.1e-5 on the cube.
# Both sit under the interpolation error at p <= 8, which is why two terms
# hold, but the bound is the measurement, not the placement rule. A SCALAR
# eps carries no bound at all -- at eps = 3 on the same model (eps/r)^4 =
# 2.6e-2, 260x the target -- so a number measured there prices the
# expansion, not the FMM.
FMM_EPS_TERMS = 2
# Target-source point pairs one M2L evaluation may hold at once. The
# reference forms d and the radial weights over the whole (p^3, p^3) block,
# so this caps the working set (at p = 8 one block is 512 x 512 and ~25
# arrays of it, ~50 MB); larger V lists are chunked by source box.
FMM_MAX_POINT_PAIRS = 2_000_000
# Columns per traversal in PairFMM.to_dense, which the gates use and no
# solve does: the whole FMM runs once per chunk, so this trades memory for
# traversals.
FMM_DENSE_COLUMN_CHUNK = 48
# Near-field (U list) blocks cached per coefficient vector. One entry is one
# material's exact near field of the pair; the gate's operator sweep touches
# at most two materials per pair.
FMM_NEAR_CACHE_MAX = 4
# PairFMM against the exact dense pair (verify_fmm), relative, in BOTH the
# 2-norm/Frobenius and max-entry senses: the per-block design target itself.
# At the default orders the two-panel pair measures 5e-10 (U) and 7e-9 (T),
# four to six orders under it, because that geometry is far better separated
# than the canonical worst offset the orders were chosen at.
FMM_PAIR_PARITY = 1e-4
# One M2L evaluator against another (verify_fmm), relative. A variant such as
# PairFMM(m2l="numba") is the SAME arithmetic in a different summation order,
# so it is gated against the reference evaluator and never against the exact
# kernel: at FMM_PAIR_PARITY a variant could regress by four orders and still
# pass. Measured on the point kernels at three Poisson ratios, both kernels,
# k = 1 and 3: 7.8e-15 worst; through a whole pair traversal it is the pair's
# own accumulation on top of that.
FMM_M2L_VARIANT_PARITY = 1e-12
# The end-to-end operator error, as the FAR-ISOLATED metric (the error over
# the exact far field alone, not over the whole operator -- the near field is
# exact, so a naive relative error understates by the fraction of A v the far
# field carries: measured 10.1x and 16.4x on the two Gaussians and 1.6x on
# the unit translation, this operator's far field being 4.5 % of its max).
# Measured at the default orders on the fault-zone model at refine 1, eps
# "auto", jump "half": 4.3e-5 / 8.0e-5 / 6.3e-6 over gaussian 0 / gaussian 1
# / translation, against a naive 4.2e-6 / 4.9e-6 / 4.0e-6. The limit is the
# worst of those with 2.5x margin.
FMM_OPERATOR_PARITY = 2e-4
# The same error over ||A v|| instead -- weaker, but a pure ratio of the
# operator to itself, so unlike the isolated metric NO change to the near/far
# split can move it. Both are gated, because neither alone is safe: the
# isolated one states the accuracy that matters and the naive one stops a
# variant passing by shrinking its own far field. Measured need: demoting 3 X
# entries at 260,598 unknowns (540 element pairs of 57 M) drops ||A_far v|| by
# 36 %, so a fix that leaves the absolute error untouched reads as a 1.6x
# regression, and two configurations with the same absolute error (2.9048e-05
# / 2.8999e-05) land either side of FMM_OPERATOR_PARITY on denominators that
# differ by 49 %. Measured value: 4.2e-6 / 4.9e-6 / 4.0e-6 over the three test
# vectors at the default orders on the gate's model, and ~1.0e-5 on
# topo_inclusion at every scale and both domains once the X list is handled.
# 5e-5 is that worst measurement with 5x margin.
FMM_OPERATOR_PARITY_NAIVE = 5e-5
# p = 4 -> p = 8 must gain at least this on a pair. Measured 5e4 (U) and 4e4
# (T) on the two-panel pair; the floor is three orders under that, so it
# fails only if p has stopped controlling the error at all.
FMM_P_CONVERGENCE_GAIN = 1e2
# Two eps passes over one on a PAIR, same measurement. There the
# interpolation error is 1e-9 and the eps^2 term is the whole of the rest,
# so the gain is the expansion's own: measured x155 (U) and x1.2e4 (T) on
# the two-panel pair at the default orders under eps="auto".
FMM_EPS_TERM_GAIN = 1e1
# The same END TO END, where the gain is bounded by where the interpolation
# error sits: the eps^2 term is ~(eps/r)^2 = 2.5e-3 of the far field and the
# far field is 4.5 % of the fault-zone operator, so ~1e-4 of ||A v|| --
# above the p = 6 interpolation error (gain x2.3, measured) and BELOW the
# p = 4 one, where two passes are no better than one (x0.92, measured) and
# the comparison says nothing about the expansion.
FMM_EPS_TERM_GAIN_OPERATOR = 1.5
# X-list (P2L) admissibility, applied by la/octree.InteractionLists whenever
# the caller states which interpolation domain it will use. An X entry
# evaluates exact source integrals AT the target box's Chebyshev nodes and then
# interpolates over that box's domain, so it converges only if the source lies
# OUTSIDE the domain: rho -- the nearest source vertex's distance from the
# domain centre, per axis in units of that axis' half-width, combined with max
# -- must exceed 1 for the interpolant to be well posed at all. Below this
# multiple the entry is not emitted; the traversal descends the target box and
# re-tests, and the residents that cannot descend go direct.
#
# WHY IT EXISTS: without it the operator is 15-30x over FMM_OPERATOR_PARITY at
# 260,598 unknowns and this far field is unusable at any size. With it, 3
# entries and 540 element pairs -- 0.0009 % of the near field on
# topo_inclusion at scale 3 -- take the far-isolated error from 1.223e-03 to
# 3.315e-04 (extent) and 5.953e-03 to 1.866e-04 (cube, placement safety 2.0).
# Moving the WHOLE X list direct instead costs +54.3 % near field and STILL
# fails, so this is a selection, not a retreat from P2L.
#
# WHY 2.0: the located value moved 1.45 -> 1.90 -> 1.65 once per newly
# measured configuration, and 1.65 clears the entry it must catch (rho 1.6437)
# by 0.4 %. 2.0 is a superset everywhere and sits on the plateau the sweep
# found -- the error is unchanged from 1.65 to 4.0 in all six instrumented
# configurations -- while rho > 1 is the well-posedness condition it has to
# respect. The admissibility ratio eta >= 0.30 is NOT an equivalent form: at
# scale 2, extent, safety 2.50, where the failure IS X-caused and rho < 1.65
# cures it (3.215e-04 -> 8.631e-05), eta >= 0.30 selects ZERO entries, because
# rho is dominated by the domain's thinnest axis and a Euclidean gap over the
# half-diagonal is not.
FMM_X_MARGIN = 2.0
# Smallest subtree worth a P2L expansion, in elements. An X entry costs
# p^3 x res(b) whatever its subtree holds, against subtree x res(b) done
# directly, so anything under the interpolation lattice is pure loss -- and
# the median X subtree holds 9-11 elements against p^3 = 216 (U) / 512 (T).
# Measured on topo_inclusion, the list does 12.0-24.4x more work than direct.
# 216 is the SMALLER of the two lattices, so no entry is demoted that the U
# kernel would still have won on. Demotion is the only route that makes X
# free rather than cheaper: an X entry re-evaluates the analytic triangle
# kernel every matvec while the U list it lands in is cached per material,
# and it is EXACT where a quadrature P2L is not (a 9-point rule measured
# 2.8e-03 worst case on T, 14x over FMM_OPERATOR_PARITY, for 2.5x).
# It is not free in memory: the demoted pairs are stored near field.
FMM_X_MIN_SUBTREE = 216
# Resident bytes of shared M2L blocks. A T key at p = 8 is 108 MiB (two eps
# passes of (3p^3, 9p^3) float64), so the full 316-offset table is 33.3 GiB --
# affordable at the 4M target, not while a gate runs. Offsets past the cap
# fall back to the matrix-free kernel, which is exact and only slower, so this
# trades speed for memory and never accuracy. 2 GiB holds every offset at
# p <= 5 and the hottest few at the shipping orders.
FMM_M2L_TABLE_MAX_BYTES = 2 * 1024**3
# Floor on an interpolation domain's half-width, in units of the box's own
# cube edge. A flat patch leaves its boxes zero extent across the plane, and
# a zero width divides the rounding of a quadrature point by itself; the
# floor only enlarges a domain, so containment survives it. Relative to the
# cube rather than absolute because M2M and L2L evaluate a child's nodes in
# its PARENT's domain, and an absolute floor puts them outside it.
FMM_MIN_HALF_OVER_EDGE = 1e-6

# --- Preconditioner rung ladder ---------------------------------------
# Rung 1, the exact dense LU of a super-block, while it is under this
# many DOFs. The cap is a MEMORY decision: there is no build-time
# crossover below it. At the same iteration count the dense rung builds
# faster than the HODLR rung on every super-block measured from 4.9k to
# 36k DOF but one (the 117k rung's blocks: 21.9 s against 83.4 on the
# 27.9k topography patch, 7.0 against 130.2 on the 18.1k inclusion top,
# 43.1 against 69.4 on the 36.3k interface, and 31.1 against 27.2 on the
# 32.0k interface, the one block HODLR wins). The reason is geometric:
# weak admissibility splits a patch into halves that TOUCH, so the
# off-diagonal blocks are near field, exceed the ACA rank cap and enter
# EXACTLY -- and an exact block at the second level makes the HODLR
# build a dense one with worse constants (a 23.8k-DOF topography patch:
# 262 s and 8.9 GB against the dense LU's 14.7 s and 5.2 GB).
# What the dense rung spends is n^2 x 8 B of stored factor (~2.1x it at
# the build's own peak) and one triangular solve per FGMRES iteration,
# memory bound at ~19 GB/s and 2-10x dearer than a HODLR apply. That
# apply is what makes an UNCAPPED dense rung no faster: at 117k
# unknowns, capped at 30k it is a 127 s ladder and a 590 ms apply, and
# uncapped a 103 s ladder and a 1.32 s apply -- 144 s against 139 s of
# total solve, for 14.2 GB of stored factors against 27.3 GB (30.4 GB
# peak RSS against 51.0). 4 % of wall is what the cap costs and half the
# memory is what it buys.
MAX_DENSE_PRECOND_DOF = 30_000
# ... and never a block whose stored factorization would take more than
# this share of physical RAM. The ladder holds every block's factor at
# once, so what a machine can afford per block is a share of it: 6 % is
# 8.2 GB of the 128 GB here (32.1k DOF, so the DOF cap above binds
# first) and 1 GB on a 16 GB laptop (11.0k DOF).
# la/preconditioner.dense_rung_max_dof resolves the two into the cap the
# ladder uses.
PRECOND_DENSE_RAM_FRACTION = 0.06
# Which rung takes a super-block that is PAST the dense cap. The value
# is the rung's own name, as la/preconditioner reports it.
#
# "block_jacobi" (the default) because HODLR does not scale on these
# geometries. Measured on topo_inclusion, one operator per size and each
# policy solved off it (iterations, preconditioner build):
#   unknowns   dense cap then HODLR     dense cap then block-Jacobi
#    31,098    23 iters,    4.0 s       27 iters,   2.9 s
#    67,962    24 iters,  292.7 s       34 iters,   7.8 s
#   117,120    25 iters,  127.0 s       37 iters,  10.1 s
#   269,346    27 iters, 3131.9 s       42 iters,  28.4 s
# -- a 24x build for 2.3x the unknowns on the last step (and 91.9 GB of
# peak RSS, at the benchmark gate's own 0.7 x RAM limit) against 2.8x for
# block-Jacobi. The 10-20 iterations HODLR saves are seconds of FGMRES;
# its build is tens of minutes. The reason is geometric and is the same
# one HODLR_LEAF_ELEMS below records: weak admissibility splits a cluster
# into two halves that TOUCH, so the off-diagonal blocks are near field,
# exceed the ACA rank cap and enter EXACTLY, and the factorization
# degenerates into a dense one with worse constants -- which is why the
# build grows superlinearly while its rank stays as predicted.
#
# "hodlr" stays reachable as an explicit choice (this constant, or the
# ``above_dense`` argument of BlockGaussSeidel / the ``precond_above_dense``
# argument of AssembledH.solve) because where a block DOES compress it is
# an order less memory than the dense rung -- 0.12 GB against 2.23 GB on
# a 15.8k-DOF block -- and a cheaper apply with it. It is the rung for a
# machine that is memory-bound rather than time-bound, and for a fixed
# iteration budget (an expensive matvec).
#
# What decides between them on WALL time is how many solves one ladder
# build has to serve, since block-Jacobi buys build and spends solve:
# the crossover is k = (build_hodlr - build_bj) / (solve_bj - solve_hodlr),
# 9-13 solves per build at 117k unknowns (build 127-170 s against 35;
# solve 17.8 s at 25 iterations against 28.0 at 37) and ~120 at 269k,
# where the HODLR build is 3132 s against 29. A material sweep under the
# PRECOND_REUSE_MAX_STEP policy delivers 3.5 (14 solves, 4 builds), so
# block-Jacobi wins it at both sizes measured: at 117k the 14-step sweep
# is 815 s against 1219 reusing the ladder and 1196 against 2954
# rebuilding it every solve. Note also that the HODLR build is not even
# stable across a sweep's materials on one geometry -- 118 to 277 s for
# the same blocks -- while the block-Jacobi build holds 35 s.
PRECOND_RUNG_ABOVE_DENSE = "block_jacobi"
# Cap on the HODLR rung WHERE IT IS CHOSEN: above this many DOFs in one
# super-block its build is the wall outright (its root off-diagonal rank
# grows ~sqrt(N) and a block that is not low rank materializes a dense
# half-matrix), so even an explicit "hodlr" policy hands such a block to
# the block-Jacobi rung.
PRECOND_HODLR_MAX_DOF = 150_000
# The rung past the dense cap (see PRECOND_RUNG_ABOVE_DENSE): dense LU on
# cluster-tree chunks of at most this many DOFs, so its build is
# O(N x chunk) and its memory chunk x 8 B per DOF (72 KB/DOF at 9000) at
# ANY super-block size. It is the only rung whose iteration count grows
# with N -- on the topo ladder +4 iterations at 31k unknowns, +10 at 68k
# and +12 at 117k against the exact rungs' 23 / 24 / 25 -- and that growth
# is bounded, not forbidden: it is what GMRES_ITER_GROWTH_ALPHA measures.
# It is a separate number from MAX_DENSE_PRECOND_DOF (which it used to
# follow) because the two bound different things -- memory per DOF against
# memory per block -- and the measurement moved them an order apart.
PRECOND_BJ_CHUNK_DOF = 9000
# Threads the block-Jacobi APPLY runs its lu_solve calls on. The chunks own
# disjoint index sets, so this is a scatter with no reduction and the result
# is bitwise identical to the serial loop (`verify_hbackend.check_bj_rung`).
# It matters because the apply is a per-ITERATION cost: at the 4M target the
# only chunk that fits in memory is 1000, and the serial apply projects to
# ~1.9 s against a far-field matvec of a few seconds.
# scipy's LAPACK here is Apple ACCELERATE, not the OpenBLAS numpy links, so
# OPENBLAS_NUM_THREADS does not govern it; with one right-hand side lu_solve
# is a level-2 solve that measures cpu/wall = 1.00 either way, so there is
# nothing to oversubscribe and nothing to pin. Several right-hand sides would
# make it level 3 and that stops being true.
# 8, from the REAL apply and not from a synthetic sweep over bare lu_solve --
# which said 10-12, at a chunk size the cluster tree does not produce.
# Measured on topo_inclusion at chunk 1000, whole M(r), bitwise identical to
# serial at every count: scale 2 (117,120 unknowns, 189 chunks) 53.4 ms ->
# 1.74 / 2.01 / 1.96 / 1.96x at 4 / 8 / 12 / 16 threads; scale 3 (260,598,
# 396 chunks) 114.7 ms -> 1.89 / 2.27 / 2.22 / 2.09x. Both knee at 8 and both
# REGRESS by 16.
# The apply gains ~2.2x where the loop alone gains ~7x because the loop is
# only ~65 % of it: the rest is the Gauss-Seidel off-diagonal matvec, already
# numba parallel=True. That is Amdahl, not a threading defect, and it is why
# more threads buy nothing here.
PRECOND_APPLY_THREADS = 8
# Below this many chunks a super-block applies serially, on scipy's lu_solve
# rather than the nogil kernel. 8 was measured on the dispatch cost alone and
# was too high to be useful: at chunk 9000 the six super-blocks of
# topo_inclusion at 117,120 unknowns hold 4, 1, 4, 8, 1, 4 chunks, so five of
# six never reached the pool and threading bought 1.08x there against 3.84x at
# 260,598. 2 is the smallest count that can be split at all.
PRECOND_APPLY_MIN_CHUNKS = 2
# The nogil chunk solve against scipy's lu_solve on the same factor. Two
# kernels for the same triangular solve, summing in different orders, so this
# is roundoff and not an identity -- unlike the worker count, which cannot
# move the answer at all and is gated with array_equal. Measured 5e-15.
PRECOND_APPLY_PARITY = 1e-12

# --- HODLR ladder rung ------------------------------------------------
# Loose tolerance of the rung's approximate inverse. It cannot move the
# ANSWER (a preconditioner changes the path and FGMRES stops on the true
# residual), only the trade between its build and the iteration count:
# 1e-1 builds the 117k rung's HODLR blocks in 80 s instead of 127 and
# costs one iteration (26 against 25; 23 either way at 31k unknowns).
# 1e-2 keeps the count, which is the invariant this ladder is built
# around, and pays for it in a rung that now carries 2 of 6 blocks.
HODLR_PRECOND_TOL = 1e-2
# Dense leaf size (elements). Weak admissibility makes a HODLR node's two
# off-diagonal blocks the interaction of two cluster halves that TOUCH,
# and the deeper the tree the larger that seam is as a FRACTION of the
# block, so deep blocks are near field, exceed the ACA rank cap and
# enter EXACTLY -- which turns a level of the factorization dense, with
# worse constants. Fewer, larger leaves cut the depth and with it those
# blocks: 96 -> 384 takes the exact off-diagonal count from 30 of 62 to
# 0 of 14 on an 8.3k-DOF topography patch, from 48 of 62 to 6 of 14 on
# the inclusion top and from 156 of 254 to 14 of 62 at 23.8k DOF, at a
# third of the rank, and builds the whole HODLR ladder of the
# 31k-unknown model in 21.3 s instead of 25.8 at the same 23 iterations
# (the 117k rung's HODLR blocks: 310 s instead of 344). What it costs is
# the leaf LUs -- leaf x d x 8 B per DOF, 9-18 KB/DOF here -- and a
# ~1.5x dearer apply, tens of milliseconds against a build of tens of
# seconds on the blocks this rung now serves.
# Measured and NOT taken: splitting a cluster at its bounding-box
# MIDPOINT instead of at the principal-axis median is worse at either
# leaf size (70 of 86 off-diagonal blocks exact and rank 1359 on that
# same 8.3k-DOF patch, against 30 of 62 and 1032).
# What NO leaf size or split fixes is a wide flat patch whose halves
# share a full-width seam: at 23.8k DOF one quarter-size off-diagonal
# block is still exact (rank 5955) and the build is still 255 s against
# the dense LU's 14.7 s. Such a block belongs on rung 1.
HODLR_LEAF_ELEMS = 384

# --- Mollification ----------------------------------------------------
# eps="auto": per-element eps_j = EPS_OVER_H * h_j (h_j = mean edge).
# Basis (icosphere vs exact Kelvin, 1280 tri; eps/h = 0.05 / 0.1 / 0.3 / 1.25):
# Dirichlet interior u 9.6e-4 / 8.9e-4 / 6.5e-3 / 5.2e-2 (floor <= 0.125),
# Neumann surface u 6.9e-3 / 9.5e-3 / 2.2e-2 / 7.5e-2, cond(A) on the fault
# box 37 / 44 / 108 / 1.7e4 -- smaller eps/h is BETTER conditioned. On-fault
# stress wants eps <= ~0.07 h (rim) and eps_top <= 0.125 h near a trace.
EPS_OVER_H = 0.1
# On a FAULT, "auto" is ONE value for the whole surface, FAULT_EPS_OVER_H *
# min(h): per-element widths smear a uniform slip unequally across shared
# edges and the on-fault stress is off by tens of percent (58 % measured at
# 0.07 h per element, where any single scalar is exact); 0.07 keeps the rim
# rule on every element.
FAULT_EPS_OVER_H = 0.07
# jump="half" with eps/h above this on any source patch is NON-convergent
# under h-refinement (measured on a manufactured uniform-strain solution);
# the backends warn. jump="calibrated" (the default) has no such limit.
HALF_JUMP_MAX_EPS_OVER_H = 0.5
# Volume evaluation warns when an observation point lies within this many
# local h (mean edge) of a BOUNDARY patch, by exact point-to-triangle
# distance: the piecewise-constant density limits the representation there
# (~2e-1 relative stress error at d/h = 0.25 with eps/h = 0.3; ~10 % at 0.5).
NEAR_BOUNDARY_H_RATIO = 0.5
# The eigenstress readout sums each element only over the observation points
# within this many (eps_j + h_j) of its centroid (evaluate._eigen_near_list):
# the element's weight is (15 eps^4 / 8 pi) I7, ~(eps/R)^4 per element and
# ~(eps/R)^5 for the whole surface beyond R. Basis (fault box, eps="auto",
# so eps + h = 15-20 eps; remainder beyond c (eps + h) at c = 10 / 20 / 30 /
# 50): on the fault centroids and at 1-50 eps off the fault 4e-13 / 0 / 0 / 0
# of the on-fault eigenstress; on a 40x40 surface grid, pointwise against
# the local elastic stress, 3e-9 / 1e-10 / 3e-11 / 2e-12 (every double
# layer summed); on the 1280-triangle Kelvin sphere at eps/h = 0.3 (eps + h
# = 4.3 eps), a shell at d/h = 0.5 against the exact stress, 1.6e-8 / 0 /
# 0 / 0. 20 is the smallest rung under 1e-8 everywhere (~100x on the grid;
# 10 fails on the sphere) at ~pi (20 (eps + h))^2 / A_tri pairs per point.
EIGEN_NEAR_RADIUS_EPS = 20.0

# --- Nodal (P1/P2) triangle kernels, kernels/tri_nodal.py --------------
# The divergence-theorem closed form loses digits roughly like (R/L)^4-5 for
# quadratic-weighted moments (R = sqrt(|x - centroid|^2 + eps^2), L = longest
# edge; ~2e-9 relative at R = 10 L, 2e-4 at 100 L, O(1) by 500 L) and like
# (R/L)^8 on the n = 7 row that carries the eigenstress weight (1e-2 relative
# at 8 L, negligible in absolute terms because that weight is ~(eps/R)^4), so
# beyond D_STAR * L the per-node weighted tables come from a collapsed product
# Gauss rule of the smooth integrand instead (exact to ~1e-14).
NODAL_D_STAR = 10.0
NODAL_FAR_GAUSS_N = 12         # points per direction (144) up to D_STAR_DISTANT * L
NODAL_FAR_GAUSS_N_DISTANT = 8  # points per direction (64) beyond it
NODAL_D_STAR_DISTANT = 40.0
# Edge primitives int u^k / R^m du: same-sign spans with min|u| >= SERIES_U_OVER_RHO
# * rho use the large-|u| binomial series (terms decay like (rho/u)^2j, so 32
# terms at ratio 2 truncate below 2^-64); spans with max|u| <= SMALL_U_OVER_RHO
# * rho use the small-|u| series (the closed form's u^k reduction loses
# (rho/u)^2 per level there); everything else the conjugate closed forms.
NODAL_SERIES_U_OVER_RHO = 2.0
NODAL_SERIES_TERMS = 32
NODAL_SMALL_U_OVER_RHO = 0.5
# A nodal (P1/P2) density on a thin triangle loses (L / height)^2 digits in
# the closed form (height / L = 1e-4: 5e-9 relative; 1e-6: 1e-4; 1e-8: O(1)),
# so the nodal drivers refuse source triangles with height / L below this
# rather than return inaccurate columns (height = 2 area / L).
NODAL_MIN_HEIGHT_OVER_L = 1e-3

# --- Mesh validity, model/core.py ----------------------------------------
# A patch element is refused below this height over its longest edge
# (height = 2 area / L): it has no frame for the kernels (they return a
# zero block below 2 area = 1e-30 km^2), no h for eps="auto", and a
# collocation point on top of its neighbour's, which collapses its
# cluster's bounding box and makes the compressed backend admit a block
# whose elements touch. The floor only has to separate "collapsed" from
# "thin": every mesh in this tree measures 0.29-0.50 (quality-30 graded
# surfaces and structured panels alike), while a PSLG failure -- a
# refinement ring landing a vertex on a fault trace -- produced elements
# at 4e-13. 1e-6 is six orders below anything meshed here and seven above
# what it must reject, so a genuinely thin structured panel still passes.
MIN_TRIANGLE_HEIGHT_OVER_L = 1e-6

# --- Higher-order (P1/P2) collocation, model/core.py ---------------------
# Collocation pull-in per element order, lam_c = (1 - t) lam_node + t / 3 (the
# basis does not move). A P1/P2 node sits ON the element boundary, shared with
# the neighbour: at t = 0 adjacent elements collocate at one point and the
# Neumann rows are numerically singular (cond ~ 1e19), the Dirichlet rows only
# mediocre; 0.5 sits in the flat-conditioning plateau (0.2-0.6) of both orders
# and is a conditioning-guarded choice, not an error minimum (the traction-row
# error is still falling at 0.8; one eps of edge clearance is what pays). P0
# collocates at the centroid, where t is the identity.
COLLOCATION_SHRINK_BY_ORDER = {0: 0.0, 1: 0.5, 2: 0.5}
# Both backends warn when a boundary collocation point lies within this many
# FAULT eps of a fault element (exact distance; eps of the nearest fault
# element): the fault's mollified field there is the blob average across the
# slip surface, not the one-sided value the boundary condition means (a fault
# outcrop). Warned, not corrected.
COLLOCATION_FAULT_CLEARANCE_EPS = 1.0

# --- Benchmark gate, examples/bench_scaling.py --gate ---------------------
# Fast-operator accuracy target (relative max-norm of A_h v against the
# dense operator, or against exact matrix-free rows beyond
# BENCH_DENSE_MAX_UNKNOWNS); the solve keeps GMRES_RTOL under it.
BENCH_OPERATOR_ERROR_MAX = 1e-4
# Compressed solution vs the dense LU, max-norm relative, per slot.
BENCH_SOLUTION_ERROR_MAX = 1e-3
# Regression bands against the baseline rung in the reference commit.
BENCH_TIME_RATIO_MAX = 1.15        # any phase wall time
BENCH_TIME_FLOOR_S = 0.5           # phases shorter than this are timer noise, not gated
BENCH_RSS_RATIO_MAX = 1.10         # peak RSS
# Stored operator bytes per unknown (near + low-rank + shared bases).
# Tighter than the RSS band because it is the one quantity free of
# transients: RSS also carries the fallback stacks, the preconditioner
# and the Krylov basis, so a compression regression can hide inside it.
BENCH_BYTES_RATIO_MAX = 1.05
BENCH_RSS_RAM_FRACTION_MAX = 0.7   # peak RSS against physical RAM
BENCH_ITER_SLACK = 2               # FGMRES iterations may exceed the baseline by this
# A gate run is refused above this 1-minute load average: another process
# on the machine invalidates every timing.
BENCH_LOAD_MAX = 2.0
# ... but a LADDER's own previous rung is not another process: 16 busy
# threads leave the 1-minute average at 7-11, and that average is an
# exponential moving average with a 60 s time constant, so it needs about
# 60 s x ln(11 / 2) ~ 100 s of idle to come back under the limit. The gate
# therefore WAITS for the decay before each rung instead of refusing, up
# to this many seconds (3x that estimate); past it the load is somebody
# else's and the run is refused.
BENCH_LOAD_WAIT_S = 300.0
# Dense reference (operator + LU solution) up to this many unknowns
# (A + LU = 58 GB at 60k); beyond it the operator error is measured on
# BENCH_EXACT_ROWS matrix-free rows (AssembledH.matvec_exact_rows).
BENCH_DENSE_MAX_UNKNOWNS = 60_000
BENCH_EXACT_ROWS = 1024
BENCH_RANDOM_VECTORS = 3           # seeded test vectors, plus the unit translation
BENCH_MATVEC_REPEATS = 5           # matvec wall = median of this many

# --- Not here ---------------------------------------------------------
# The fault SIGN CONVENTION is not a tolerance and does not live here:
# it is ``FAULT_ORIENTATION`` in ``mbem/model/core.py``, next to the
# ``RegionModel.orientation`` accessor every site reads it through, and
# it is pinned at runtime by ``mbem/selfcheck.py``. Do not restate it.
