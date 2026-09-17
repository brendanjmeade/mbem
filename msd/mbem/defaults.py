"""Single source of truth for mbem tolerances and thresholds.

Every numeric default a solver, compressor, or pipeline uses must live
here, so the whole stack can be audited (and a test can assert nothing
drifts). Values follow the approved plan: solution accuracy target ~1e-6,
linear-algebra layers run with ~100x margin under it.
"""

# --- Accuracy targets -------------------------------------------------
SOLUTION_RTOL = 1e-6          # end-to-end solver accuracy target
GMRES_RTOL = 1e-8             # true-residual stop (100x margin)
GMRES_RESTART = 200
GMRES_MAXITER = 600
# Stop FGMRES if the residual fails to improve by STAGNATION_FACTOR
# over STAGNATION_WINDOW iterations (previously hardcoded in solver.py).
GMRES_STAGNATION_WINDOW = 100
GMRES_STAGNATION_FACTOR = 10.0

# --- Conditioning ------------------------------------------------------
# Warn when a dense solve's 1-norm condition estimate exceeds this.
# cond * machine-eps ~ residual amplification: 1e10 * 2e-16 = 2e-6 is
# right at SOLUTION_RTOL, so anything above it can silently miss the
# accuracy target (thin panels / near-fluid materials / the spurious
# discretization resonance all push cond up long before LU "fails").
COND_WARN_THRESHOLD = 1e10

# --- Compression ------------------------------------------------------
BLOCK_COMPRESSION_TOL = 1e-8  # rel-Frobenius per admissible block
# Relaxed preset for very large models: 100x looser than the default but
# still 100x under SOLUTION_RTOL; cuts low-rank ranks (and memory) 30-50%.
# Pass explicitly: HBackend(tol=defaults.BLOCK_COMPRESSION_TOL_RELAXED).
BLOCK_COMPRESSION_TOL_RELAXED = 1e-6
CLUSTER_MIN_LEAF = 32         # elements per leaf cluster
ADMISSIBILITY_ETA = 2.0
# Admissible blocks smaller than this (elements per side) are stored
# dense: at small sizes the epsilon-rank is a large fraction of the
# block and cross approximation cannot be certified by sampling.
ACA_MIN_BLOCK = 64
# Cap on admissible block side (elements). Controls the size of the
# dense FALLBACK bomb when a borderline block fails verification: at
# 4096 a failed T block transiently materializes a ~7 GB basis stack
# (measured: a 51k-tri panel pair took 471 s / 91 GB peak at 4096 vs
# 112 s / 33 GB at 2048). Larger values compress the far field slightly
# better; raise only for smooth, well-separated geometry.
MAX_ADMISSIBLE_BLOCK = 2048
# ACA must converge within this fraction of full element rank, else the
# block is declared not-low-rank and falls back to dense evaluation.
ACA_MAX_RANK_FRACTION = 1.0 / 3.0
# Max CONCURRENT dense fallbacks during parallel block compression --
# each fallback transiently materializes a full (B, 3nr, 3nc) basis
# stack (~7 GB at max_admissible=4096 for the T kernel).
ACA_FALLBACK_CONCURRENCY = 2
# Only admissible blocks at least this many elements per (shorter) side
# are compressed on the thread pool. Below it, per-eval kernel work is
# so small that block ACA is Python-overhead(GIL)-bound and threads only
# add contention (measured: 232 blocks of 72x72 ran 0.6x under threads);
# small blocks are compressed inline instead. Thread parallelism is for
# the LARGE blocks that dominate at 1e5-1e6 elements.
ACA_PARALLEL_MIN_SIDE = 192
# Per-material recombined views cached per PairCompressed (LRU). Must
# comfortably exceed the number of DISTINCT (material x kernel)
# coefficient vectors live in one solve (preconditioner included), or
# every matvec re-runs the per-block recompression; 16 covers models
# with up to ~8 regions. Bounds memory across Laplace sweeps.
HOP_VIEW_CACHE_MAX = 16

# --- Dense fallbacks --------------------------------------------------
MAX_DENSE_PRECOND_DOF = 9000  # exact dense LU below this, per block
# HODLR preconditioner rung cap: above this many DOFs in one super-block
# the HODLR build itself becomes the memory/time wall (its root
# off-diagonal rank grows ~sqrt(N) and its fallback materializes a dense
# half-matrix), so the ladder switches to the cluster BLOCK-JACOBI rung
# (dense LU on cluster-tree chunks of <= MAX_DENSE_PRECOND_DOF each; a
# few more FGMRES iterations, O(N x chunk) build memory at any N).
PRECOND_HODLR_MAX_DOF = 150_000

# --- HODLR ladder rung ------------------------------------------------
HODLR_PRECOND_TOL = 1e-2      # loose tol when used as a preconditioner
HODLR_LEAF_ELEMS = 96         # dense leaf size (elements)

# --- Basis-recombination parity gates (Phase 1) -----------------------
# Direct vs basis assembly differ only in floating-point evaluation
# order. At moderate nu the agreement is machine precision; near the
# fluid limit (nu -> 1/2) the lam*C1 coefficient amplifies cancellation
# between the N[P1]/N[P2] (trace) basis terms by ~|lam*C1|/|c_R2|, so the
# gate loosens accordingly (still far below any physical tolerance).
BASIS_PARITY_RTOL = 1e-13
BASIS_PARITY_RTOL_NEAR_FLUID = 1e-10

# --- Mollification ----------------------------------------------------
EPS_OVER_H = 1.25             # per-element eps_j = EPS_OVER_H * h_j (opt-in)
