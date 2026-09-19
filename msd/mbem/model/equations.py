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

A FAULT is not a special case: ``sigma(R,f)`` is defined for it too
(``FAULT_ORIENTATION``, see ``core.py``), so a fault's slip enters the
RHS through the very same ``-sigma`` coefficient a prescribed boundary
displacement does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kernels import KERNEL_T, KERNEL_U
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
    kernel: str                  # KERNEL_T (H) | KERNEL_U (G)
    field_patch: Patch
    source_patch: Patch
    region: Region               # material provider
    scale: float                 # +-1 from the sigma rule
    diag: float = 0.0            # free term on this block: COLLOCATION_JUMP
                                 # on the q == p u-block, else 0


@dataclass(frozen=True)
class RhsTerm:
    """b[row] += scale * K^{m(region)}_{q,source} @ vector."""
    row: Slot
    kernel: str
    field_patch: Patch
    source_patch: Patch
    region: Region
    scale: float
    vector: np.ndarray           # flattened known value (3*N_source,)
    add_half_of_vector: bool = False   # also b[row] += scale_half * vector
    scale_half: float = 0.0


@dataclass
class BlockSystem:
    model: RegionModel
    layout: UnknownLayout
    terms: list[BlockTerm] = field(default_factory=list)
    rhs_terms: list[RhsTerm] = field(default_factory=list)

    def mesh_pairs(self) -> set:
        return {(id(t.field_patch), id(t.source_patch), t.kernel)
                for t in self.terms}


def generate_system(model: RegionModel) -> BlockSystem:
    if any(r.faults for r in model.regions):
        ensure_fault_convention()
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
            # fault ``orientation`` returns FAULT_ORIENTATION.
            for f in region.faults:
                sigma = float(model.orientation(region, f))
                slip = f.value_array().ravel()
                if np.any(slip):
                    system.rhs_terms.append(RhsTerm(
                        row=row, kernel=KERNEL_T, field_patch=q,
                        source_patch=f, region=region, scale=-sigma,
                        vector=slip))

    return system


# ---- the collocation diagonal, shared by every backend --------------------

def calibrated_diagonal(system: BlockSystem, rowsum) -> dict:
    """{(id(region), id(q)): C (Nq,3,3)} -- the rigid-body-calibrated
    collocation diagonal of each BIE row (region R, collocation patch q):

        C_q = -sum_{p in dR} sigma(R,p) * rowsum_j H^{m(R)}_{qp}

    so that a constant displacement over ALL of dR (prescribed patches
    included) with zero traction is annihilated exactly, where the
    mollified Gauss identity sum_j H_qj = -1/2 I holds only approximately.
    ``rowsum(region, q, p) -> (Nq,3,3)`` is the backend's own row-sum of
    the H block it applies, so the calibrated operator is exact for the
    operator actually used (dense block, or compressed pair).
    """
    model = system.model
    calib: dict = {}
    for region in model.regions:
        for q in region.patches:
            C = np.zeros((q.n_triangles, 3, 3))
            for p in region.patches:
                C -= float(model.orientation(region, p)) * rowsum(region, q, p)
            calib[(id(region), id(q))] = C
    return calib


def term_diagonal(term: BlockTerm, calib: dict | None):
    """The (Nq,3,3) diagonal D_q this term adds to its block, or None:
    ``calib[(region, q)]`` under the calibrated jump (``calib`` is a dict),
    else ``term.diag * I``."""
    if not term.diag:
        return None
    if calib is not None:
        return calib[(id(term.region), id(term.field_patch))]
    return np.broadcast_to(term.diag * np.eye(3),
                           (term.field_patch.n_triangles, 3, 3))


def diagonal_matvec(D: np.ndarray, x: np.ndarray) -> np.ndarray:
    """(Nq,3,3) block-diagonal times a flattened (3*Nq,) vector."""
    return np.einsum("nij,nj->ni", D, x.reshape(-1, 3)).ravel()


def add_block_diagonal(A: np.ndarray, r0: int, c0: int, D: np.ndarray):
    """A[r0 + 3i + a, c0 + 3i + b] += D[i, a, b] (in place)."""
    idx = 3 * np.arange(D.shape[0])
    for a in range(3):
        for b in range(3):
            A[r0 + idx + a, c0 + idx + b] += D[:, a, b]


def add_jump_rhs(system: BlockSystem, calib: dict | None, b: np.ndarray):
    """Prescribed-displacement rows: the free term multiplies the KNOWN
    u_bar of the row's own patch, so it moves to the RHS with its LHS
    sign -- ``b[row] -= C_q @ u_bar`` (calibrated) or
    ``b[row] += scale_half * u_bar`` (scale_half = -COLLOCATION_JUMP).
    The emitting RhsTerm is the one with ``add_half_of_vector``."""
    for rt in system.rhs_terms:
        if not rt.add_half_of_vector:
            continue
        r0, r1 = rt.row.offset, rt.row.stop
        if calib is None:
            b[r0:r1] += rt.scale_half * rt.vector
        else:
            b[r0:r1] -= diagonal_matvec(
                calib[(id(rt.region), id(rt.field_patch))], rt.vector)
