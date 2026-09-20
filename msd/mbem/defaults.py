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

# --- Conditioning ------------------------------------------------------
# Warn when a dense solve's 1-norm condition estimate exceeds this.
# cond * machine-eps ~ residual amplification: 1e10 * 2e-16 = 2e-6 is
# right at SOLUTION_RTOL, so anything above it can silently miss the
# accuracy target (thin panels / near-fluid materials / the spurious
# discretization resonance all push cond up long before LU "fails").
COND_WARN_THRESHOLD = 1e10

# --- Compression ------------------------------------------------------
BLOCK_COMPRESSION_TOL = 1e-8  # rel-Frobenius per admissible block
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

# --- Not here ---------------------------------------------------------
# The fault SIGN CONVENTION is not a tolerance and does not live here:
# it is ``FAULT_ORIENTATION`` in ``mbem/model/core.py``, next to the
# ``RegionModel.orientation`` accessor every site reads it through, and
# it is pinned at runtime by ``mbem/selfcheck.py``. Do not restate it.
