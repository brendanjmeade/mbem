"""Block-equation generation from a RegionModel.

THE sign rule (single formula replacing every hand-written assembler):

For region R with material m(R), collocation patch q in dR, and
sigma(R,p) = +1 iff patch p's stored normals point out of R:

    A[row(R,q), u_p] += sigma(R,p) * H^{m(R)}_{qp} + D_q * delta_{qp}
    A[row(R,q), t_p] += -sigma(R,p) * G^{m(R)}_{qp}
    b[row(R,q)]      -= sum_{f in faults(R)} sigma(R,f) * H^{m(R)}_{qf} @ slip_f
                        (+ prescribed-value columns moved to the RHS
                         with their LHS coefficients)

H is the T-kernel (slip/displacement -> displacement) influence matrix
assembled with the patch's stored normals; G is the U-kernel. D_q is the
collocation free term on the single-valued u: ``COLLOCATION_JUMP * I``
(jump="half") or the calibrated ``C_q`` of ``calibrated_diagonal``
(jump="calibrated"); it never flips with sigma. The shared interface
traction unknown is t_p = sigma_stored * n_stored, so region R sees
sigma(R,p) * t_p — hence the -sigma on G.

Rows are the collocation points of q (K per element, ``Patch.order``),
columns the nodal densities. The free term of collocation point c (node kc
of element s) acts on u(x_c) = sum_k N_k(x_c) u[s, k], so on q's own
element columns it is ``N[kc, k] * D_c`` with ``N = q.collocation_shape()``
and D_c the (3,3) diagonal of that point; at P0 N = [[1]] and this is the
plain block diagonal. The row-sum identity behind the calibrated diagonal
carries to P1/P2 through the partition of unity ``sum_k N_k = 1``.

A FAULT is not a special case: ``sigma(R,f)`` is defined for it too
(``FAULT_ORIENTATION``, see ``core.py``), so a fault's slip enters the
RHS through the very same ``-sigma`` coefficient a prescribed boundary
displacement does -- and a FAULT_TRACTION fault, whose slip b_f is an
unknown, enters the boundary rows as the COLUMN block ``+sigma(R,f) H_qf``:
the prescribed-u coefficient on the LEFT-hand side (its -sigma is the
right-hand-side form).

THE TRACTION ROWS of a FAULT_TRACTION fault f, at its own collocation nodes c
(normal n_c), are the elastic traction of every source of its region, with
the signs of ``evaluate_stress`` term by term, equal to the prescribed
traction change dt (``Patch.value``):

    n_c . sigma(x_c) = dt(x_c),
    sigma(x) = sum_p [ -sigma(R,p) (S_H - E)_{xp} u_p + sigma(R,p) S_G,xp t_p ]
             - sum_g sigma(R,g) (S_H - E)_{xg} b_g

(S_H the double-layer stress, E its eigenstress, S_G the Kelvin stress), so
with Ht = n_c . (S_H - E) and Gt = n_c . S_G the blocks are

    A[b_f, u_p] += -sigma_p Ht     A[b_f, t_p] += +sigma_p Gt
    A[b_f, b_g] += -sigma_g Ht     (g includes f itself; no free term, no diag)
    b[b_f]      += +sigma_p Ht u_bar_p - sigma_p Gt t_bar_p + sigma_g Ht slip_g + dt

A fault row has NO free term: the traction of a dislocation is continuous
across its own surface, so there is no jump to halve, and mollified the
hypersingular integral is an ordinary one. It is NOT calibrated either. The
eigenstress IS removed in these rows (``KERNEL_HT`` is total minus
eigenstress): the elastic stress of a constant closed double layer vanishes
only once it is removed (docs/ddbem-findings.md, finding 2), so without it a
rigid translation of a free-traction body would load its faults and the
calibrated null space would no longer be one. ``apply_fault_rows`` then
turns the Cartesian rows into the row frame: with ``allow_opening=False``
the tangential pair and the constraint n_c . b_c = 0, with ``allow_opening``
the three traction components; locked nodes get identity rows (b = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import defaults
from ..kernels import KERNEL_GT, KERNEL_HT, KERNEL_T, KERNEL_U
from ..selfcheck import ensure_fault_convention
from .core import BCType, Patch, Region, RegionModel
from .layout import Slot, UnknownLayout

# The free term of the collocation BIE: the single-valued boundary
# displacement carries half of itself on its own row. Stated here only;
# backends read it through ``BlockTerm.diag`` / ``RhsTerm.scale_half``.
COLLOCATION_JUMP = 0.5


@dataclass(frozen=True)
class BlockTerm:
    row: Slot
    col: Slot
    kernel: str                  # KERNEL_T (H) | KERNEL_U (G) | KERNEL_HT | KERNEL_GT
    field_patch: Patch
    source_patch: Patch
    region: Region               # material provider
    scale: float                 # +-1 from the sigma rule
    diag: float = 0.0            # free term on this block: COLLOCATION_JUMP
                                 # on the q == p u-block, else 0; applied per
                                 # collocation point through term_diagonal /
                                 # add_block_diagonal


@dataclass(frozen=True)
class RhsTerm:
    """b[row] += scale * K^{m(region)}_{q,source} @ vector."""
    row: Slot
    kernel: str
    field_patch: Patch
    source_patch: Patch
    region: Region
    scale: float
    vector: np.ndarray           # flattened known nodal value (3*K*N_source,)
    add_half_of_vector: bool = False   # also b[row] += scale_half * vector
                                       # (interpolated to the collocation points)
    scale_half: float = 0.0


@dataclass(frozen=True)
class RhsValue:
    """b[row] += vector: a datum that lands on the right-hand side as it
    is, with no kernel -- a traction fault's prescribed traction change, one
    (3,) per dof node, flattened."""
    row: Slot
    vector: np.ndarray


@dataclass
class BlockSystem:
    model: RegionModel
    layout: UnknownLayout
    terms: list[BlockTerm] = field(default_factory=list)
    rhs_terms: list[RhsTerm] = field(default_factory=list)
    rhs_values: list[RhsValue] = field(default_factory=list)

    def mesh_pairs(self) -> set:
        return {(id(t.field_patch), id(t.source_patch), t.kernel)
                for t in self.terms}

    def traction_faults(self) -> list:
        """[(region, fault)] for every FAULT_TRACTION fault, in slot order."""
        return [(r, f) for r in self.model.regions for f in r.faults
                if f.bc is BCType.FAULT_TRACTION]


def generate_system(model: RegionModel) -> BlockSystem:
    if any(r.faults for r in model.regions):
        ensure_fault_convention()
    if any(f.bc is BCType.FAULT_TRACTION for r in model.regions for f in r.faults):
        ensure_fault_convention("traction")
    layout = UnknownLayout(model)
    system = BlockSystem(model=model, layout=layout)

    for region in model.regions:
        for q in region.patches:
            row = layout.row_slot(region, q)

            for p in region.patches:
                sigma = float(model.orientation(region, p))

                # ---- u_p term: sigma * H + D_q delta_qp ----
                if layout.has_slot(p, "u"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "u"), kernel=KERNEL_T,
                        field_patch=q, source_patch=p, region=region,
                        scale=sigma,
                        diag=COLLOCATION_JUMP if p is q else 0.0))
                else:
                    # u_p prescribed: move (sigma H + D_q delta_qp) @ u_bar
                    u_bar = p.value_array().ravel()
                    if np.any(u_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel=KERNEL_T, field_patch=q,
                            source_patch=p, region=region, scale=-sigma,
                            vector=u_bar,
                            add_half_of_vector=(p is q),
                            scale_half=-COLLOCATION_JUMP))

                # ---- t_p term: -sigma * G ----
                if layout.has_slot(p, "t"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "t"), kernel=KERNEL_U,
                        field_patch=q, source_patch=p, region=region,
                        scale=-sigma))
                else:
                    # t_p prescribed: move -sigma G @ t_bar to RHS (+sigma)
                    t_bar = p.value_array().ravel()
                    if np.any(t_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel=KERNEL_U, field_patch=q,
                            source_patch=p, region=region, scale=sigma,
                            vector=t_bar))

            # ---- fault sources of this region ----
            # Same -sigma rule as the prescribed-u branch above; for a
            # fault ``orientation`` returns FAULT_ORIENTATION. A solved slip
            # is a column block with the LEFT-hand-side coefficient +sigma
            # (the u_p term above), where the prescribed slip's RHS has -sigma.
            for f in region.faults:
                sigma = float(model.orientation(region, f))
                if f.bc is BCType.FAULT_TRACTION:
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(f, "b"), kernel=KERNEL_T,
                        field_patch=q, source_patch=f, region=region,
                        scale=sigma))
                    continue
                slip = f.value_array().ravel()
                if np.any(slip):
                    system.rhs_terms.append(RhsTerm(
                        row=row, kernel=KERNEL_T, field_patch=q,
                        source_patch=f, region=region, scale=-sigma,
                        vector=slip))

        # ---- traction rows of this region's solved faults ----
        # The elastic traction at the fault's nodes, every source with the
        # sign evaluate_stress gives it (module docstring); the prescribed
        # ones move to the RHS with the opposite sign.
        for f in region.faults:
            if f.bc is not BCType.FAULT_TRACTION:
                continue
            row = layout.slot(f, "b")
            for p in region.patches:
                sigma = float(model.orientation(region, p))
                if layout.has_slot(p, "u"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "u"), kernel=KERNEL_HT,
                        field_patch=f, source_patch=p, region=region,
                        scale=-sigma))
                else:
                    u_bar = p.value_array().ravel()
                    if np.any(u_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel=KERNEL_HT, field_patch=f,
                            source_patch=p, region=region, scale=sigma,
                            vector=u_bar))
                if layout.has_slot(p, "t"):
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(p, "t"), kernel=KERNEL_GT,
                        field_patch=f, source_patch=p, region=region,
                        scale=sigma))
                else:
                    t_bar = p.value_array().ravel()
                    if np.any(t_bar):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel=KERNEL_GT, field_patch=f,
                            source_patch=p, region=region, scale=-sigma,
                            vector=t_bar))
            for g in region.faults:
                sigma = float(model.orientation(region, g))
                if g.bc is BCType.FAULT_TRACTION:
                    system.terms.append(BlockTerm(
                        row=row, col=layout.slot(g, "b"), kernel=KERNEL_HT,
                        field_patch=f, source_patch=g, region=region,
                        scale=-sigma))
                else:
                    slip = g.value_array().ravel()
                    if np.any(slip):
                        system.rhs_terms.append(RhsTerm(
                            row=row, kernel=KERNEL_HT, field_patch=f,
                            source_patch=g, region=region, scale=sigma,
                            vector=slip))
            system.rhs_values.append(RhsValue(row=row,
                                              vector=f.value_array().ravel()))

    return system


# ---- the traction rows' frame, constraint and locked nodes ----------------

def traction_row_frame(fault: Patch) -> np.ndarray:
    """(n_dofs, 3, 3) row frame of a FAULT_TRACTION fault: per node the rows
    [strike; dip; normal] (strike = z x n, dip = n x strike, x and y on a
    horizontal node, as ``mhs.tdcs`` builds it) when ``allow_opening`` is
    False, the identity when the three Cartesian traction components are
    enforced. Any orthonormal tangent pair gives the same solution; this one
    is deterministic."""
    n = np.asarray(fault.collocation_normals(), float)
    if fault.allow_opening:
        return np.ascontiguousarray(np.broadcast_to(np.eye(3), (n.shape[0], 3, 3)))
    s = np.cross(np.array([0.0, 0.0, 1.0]), n)
    length = np.linalg.norm(s, axis=1)
    flat = length <= defaults.FRAME_FLAT_TOL
    s[flat] = [1.0, 0.0, 0.0]
    s[~flat] /= length[~flat, None]
    d = np.cross(n, s)
    d /= np.linalg.norm(d, axis=1)[:, None]
    return np.ascontiguousarray(np.stack([s, d, n], axis=1))


def apply_fault_rows(A: np.ndarray, b: np.ndarray, system: BlockSystem) -> None:
    """In place, for every FAULT_TRACTION fault: rotate its Cartesian traction
    rows and RHS into ``traction_row_frame``; with ``allow_opening=False``
    overwrite each free node's third row by the constraint n_c . b_c = 0;
    overwrite every locked node's three rows by b_c = 0. The constraint and
    identity rows are scaled by the mean |diagonal| of the fault's own
    tangential self-block, so no new constant enters the system."""
    for region, f in system.traction_faults():
        slot = system.layout.slot(f, "b")
        r0, r1 = slot.offset, slot.stop
        n = f.n_dofs
        F = traction_row_frame(f)
        step = max(1, int(defaults.ASSEMBLY_SOURCE_CHUNK))
        for c0 in range(0, n, step):
            c1 = min(n, c0 + step)
            rows = slice(r0 + 3 * c0, r0 + 3 * c1)
            blk = A[rows].reshape(c1 - c0, 3, -1)
            A[rows] = np.einsum("nij,njc->nic", F[c0:c1], blk).reshape(3 * (c1 - c0), -1)
            b[rows] = np.einsum("nij,nj->ni", F[c0:c1], b[rows].reshape(-1, 3)).ravel()
        lock = f.locked_mask(region.patches)
        free = np.flatnonzero(~lock)
        diag = np.array([A[r0 + 3 * c + i, r0 + 3 * c + i]
                         for c in free for i in (0, 1)])
        scale = float(np.mean(np.abs(diag))) if diag.size else 1.0
        if scale == 0.0:
            scale = 1.0
        if not f.allow_opening and free.size:
            nrm = np.asarray(f.collocation_normals(), float)
            rows = r0 + 3 * free + 2
            A[rows, :] = 0.0
            for j in range(3):
                A[rows, r0 + 3 * free + j] = scale * nrm[free, j]
            b[rows] = 0.0
        locked = np.flatnonzero(lock)
        if locked.size:
            for i in range(3):
                rows = r0 + 3 * locked + i
                A[rows, :] = 0.0
                A[rows, rows] = scale
                b[rows] = 0.0


# ---- the collocation diagonal, shared by every backend --------------------

def calibrated_diagonal(system: BlockSystem, rowsum) -> dict:
    """{(id(region), id(q)): C (q.n_nodes,3,3)} -- the rigid-body-calibrated
    collocation diagonal of each BIE row (region R, collocation point c of
    patch q):

        C_c = -sum_{p in dR} sigma(R,p) * rowsum_over_all_columns H^{m(R)}_{qp}[c]

    so that a constant displacement over ALL of dR (prescribed patches
    included) with zero traction is annihilated exactly, where the
    mollified Gauss identity sum_j H_qj = -1/2 I holds only approximately.
    ``rowsum(region, q, p) -> (q.n_nodes,3,3)`` is the backend's own row-sum
    of the H block it applies (over every node column of p), so the
    calibrated operator is exact for the operator actually used (dense
    block, or compressed pair). A constant nodal density is constant at
    every collocation point (sum_k N_k = 1), which is why the identity is
    order-independent.
    """
    model = system.model
    calib: dict = {}
    for region in model.regions:
        for q in region.patches:
            C = np.zeros((q.n_nodes, 3, 3))
            for p in region.patches:
                C -= float(model.orientation(region, p)) * rowsum(region, q, p)
            calib[(id(region), id(q))] = C
    return calib


def term_diagonal(term: BlockTerm, calib: dict | None):
    """The (q.n_nodes,3,3) per-collocation-point diagonal D_c this term adds
    to its block, or None: ``calib[(region, q)]`` under the calibrated jump
    (``calib`` is a dict), else ``term.diag * I``."""
    if not term.diag:
        return None
    if calib is not None:
        return calib[(id(term.region), id(term.field_patch))]
    return np.broadcast_to(term.diag * np.eye(3),
                           (term.field_patch.n_nodes, 3, 3))


def diagonal_matvec(D: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(Nc,3,3) block-diagonal times a flattened (3*Nc,) vector of values
    AT the collocation points."""
    return np.einsum("nij,nj->ni", D, x.reshape(-1, 3)).ravel()


def add_block_diagonal(A: np.ndarray, r0: int, c0: int, D: np.ndarray,
                       shape: np.ndarray):
    """A[r0 + 3c + a, c0 + 3(K s + k) + b] += shape[kc, k] * D[c, a, b] in
    place, for collocation point c = K s + kc of element s: the free term of
    each row spread over its own element's K node columns by the shape
    functions at the collocation point (``Patch.collocation_shape``, (K, K)).
    At P0 ``shape = [[1]]`` and this is the plain block diagonal."""
    K = shape.shape[0]
    c = np.arange(D.shape[0])
    s, kc = c // K, c % K
    rows = r0 + 3 * c
    for k in range(K):
        w = shape[kc, k]
        cols = c0 + 3 * (K * s + k)
        for a in range(3):
            for b in range(3):
                A[rows + a, cols + b] += w * D[:, a, b]


def add_jump_rhs(system: BlockSystem, calib: dict | None, b: np.ndarray):
    """Prescribed-displacement rows: the free term multiplies the KNOWN
    u_bar of the row's own patch at the collocation points (the nodal value
    interpolated by ``Patch.collocation_values``), so it moves to the RHS
    with its LHS sign -- ``b[row] -= C_c @ u_bar(x_c)`` (calibrated) or
    ``b[row] += scale_half * u_bar(x_c)`` (scale_half = -COLLOCATION_JUMP).
    The emitting RhsTerm is the one with ``add_half_of_vector``."""
    for rt in system.rhs_terms:
        if not rt.add_half_of_vector:
            continue
        r0, r1 = rt.row.offset, rt.row.stop
        u_c = rt.field_patch.collocation_values(rt.vector).ravel()
        if calib is None:
            b[r0:r1] += rt.scale_half * u_c
        else:
            b[r0:r1] -= diagonal_matvec(
                calib[(id(rt.region), id(rt.field_patch))], u_c)
