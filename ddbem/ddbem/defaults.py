"""Single source of truth for numeric constants in ``ddbem``.

Put new numeric constants here, not inline (the msd/clq convention).
"""
from __future__ import annotations

# --- kernel evaluation -------------------------------------------------------
# Passed straight through to ``clq.influence``.  "hybrid" is clq's safe choice:
# the closed form within clq.defaults.D_STAR * L of the centroid, Gauss
# quadrature beyond (the divergence-theorem closed form loses digits with
# distance).  "analytic" is what the msd oracles use, so the P0 parity gate
# runs with FAR_FIELD_PARITY to compare like with like at large separations.
FAR_FIELD = "hybrid"
FAR_FIELD_PARITY = "analytic"

# --- Voigt convention --------------------------------------------------------
# Row order of ``stress_matrix``: (xx, yy, zz, yz, xz, xy).  This is the
# "engineering" ordering used by msd's figures; the shear rows carry NO factor
# of 2 (these are stresses, not strains).
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

# --- mollification budget ----------------------------------------------------
# fbem/FINDINGS.md sec.2: eps is a FLOOR on the resolvable structure of the
# unknown, not only a regularisation.  At eps = 0.3 h no collocation point of a
# P1/P2 element is even one mollification length clear of the element boundary
# (0.79 eps for P0, 0.39 eps for P1/P2 measured there), and p-refinement bought
# nothing.  EPS_OVER_H is therefore a DEFAULT TO BE OVERRIDDEN AND SWEPT, never
# a setting to rely on; ``ddbem.eps_report`` prints the budget that actually
# applies to a given mesh.
EPS_OVER_H = 0.3

# Barycentric pull-in for node collocation points: a P1/P2 node sits on the
# element boundary (vertex or edge midpoint), where it is shared with the
# neighbouring element and sits 0 clear of the edge.  ``shrink`` moves a node
# toward the centroid by this fraction of the way:
#     lam_collocation = (1 - shrink) * lam_node + shrink / 3.
# 0.0 = the geometric node.  This is the RAW default used by
# ``ddbem.mesh`` (Stage 1 chose no collocation scheme); the SOLVER default is
# COLLOCATION_SHRINK_BY_ORDER below, which is the one that was measured.
COLLOCATION_SHRINK = 0.0

# Solver collocation pull-in, per element order.  P0 collocates at the centroid,
# where shrink is the identity, so only P1/P2 have a choice.  0.5 is the outcome
# of ``bench/sweep_collocation.py`` (a sweep, not a guess; the full table is in
# the README).  What that sweep found:
#   * shrink = 0 is CATASTROPHIC for the hypersingular rows -- the collocation
#     points are then the shared vertices/midpoints of the mesh, adjacent
#     elements collocate at the SAME point with nearly parallel normals, and the
#     condition number is 3.5e19..5.1e19, i.e. numerically singular.  It is
#     merely mediocre for displacement rows (cond 6..23), which is why a code
#     that only ever solves Dirichlet problems can ship with shrink = 0 and not
#     notice.
#   * beyond ~0.1 the traction-row error falls MONOTONICALLY over the whole range
#     swept while the conditioning is flat from 0.2 to 0.6 and starts to inflate
#     at 0.7 (P2: 702 -> 1.9e3 -> 4.6e3 at 0.8).  0.5 sits inside that plateau.
#   * fbem's independent sweep of the same parameter for a single-layer density
#     also chose 0.5.
# BE CLEAR ABOUT WHAT 0.5 IS: a conditioning-guarded choice, NOT an error
# minimum.  The Neumann error is still falling at the end of the sweep (P1 at
# eps/h = 0.3 is 4.1x better at 0.8 than at 0.5, at the same condition number),
# and the sweep stops at 0.8 because t -> 1 collapses all K collocation points
# onto the centroid and the system degenerates.  Whether 0.7-0.8 is better in
# production is OPEN.
# The physical reason is ../fbem/FINDINGS.md sec.2: the collocation point has to
# be a finite number of mollification lengths clear of the element boundary
# before the operator means anything.  Sweep it again on a new geometry rather
# than trusting this number.
COLLOCATION_SHRINK_BY_ORDER = {0: 0.0, 1: 0.5, 2: 0.5}

# ``ddbem.mesh.shrink_for_clearance`` targets this many mollification lengths of
# in-plane clearance between a collocation point and its element's boundary.
# 1.0 is not a round number picked for tidiness: at eps/h = 0.15 the P1
# hypersingular error has a sharp minimum at shrink = 0.6, which is exactly
# where the clearance first reaches 1.03 eps, and it is 4.5x and 8x worse at
# 0.7 and 0.8.  Same statement as ../fbem/FINDINGS.md sec.2, now as a rule.
TARGET_CLEARANCE_EPS = 1.0
SHRINK_MAX = 0.8      # t -> 1 collapses every collocation point onto the centroid

# --- collocation rows --------------------------------------------------------
# Free term / jump convention.  "half" uses the analytic rigid-body row sum
# (1/2 I on a displacement row, 0 on a traction row); "calibrated" measures the
# actual discrete row sum and sets the free term so a rigid translation of the
# body is annihilated EXACTLY.  Calibration is a CLOSED-surface row-sum
# identity, so it is only defined when the boundary is a closed oriented
# manifold; ddbem.Model refuses it otherwise.
JUMP = "calibrated"

# Which equation a traction-free boundary row states.  "traction": the
# hypersingular row t(x_c) = t_bar, which is the general one (any mixed BC).
# "exterior": the second-kind row u_ext(x_c) = 0, which is msd's formulation
# and is only valid when EVERY boundary patch is traction-free with t_bar = 0.
NEUMANN_ROW = "traction"

# Rigid-body constraint for a model whose rows cannot see a rigid motion.
# "rigid" = 3 translations + 3 rotations (the full continuum null space of the
# interior traction-free problem), "translations" = msd's 3, "none" = solve as
# is.  Applied by bordering, [[A, Z], [Z^T, 0]].
CONSTRAIN = "rigid"

# --- geometry guards ---------------------------------------------------------
DEGENERATE_AREA_REL = 1e-14   # area / L^2 below this -> degenerate triangle
WELD_REL_TOL = 1e-9           # vertex merge tolerance, relative to model span

# Warn when a boundary collocation point is closer than this many eps to a
# fault element: the fault's mollified field is the blob average there, not the
# one-sided value the boundary condition means (a fault outcrop).
FAULT_CLEARANCE_EPS = 1.0
