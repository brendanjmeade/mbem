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
BENCH_RSS_RAM_FRACTION_MAX = 0.7   # peak RSS against physical RAM
BENCH_ITER_SLACK = 2               # FGMRES iterations may exceed the baseline by this
# A gate run is refused above this 1-minute load average: another process
# on the machine invalidates every timing.
BENCH_LOAD_MAX = 2.0
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
