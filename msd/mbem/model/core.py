"""Region-graph data model with automatic boundary orientation.

A ``RegionModel`` is a set of ``Region``s, each bounded by oriented
``Patch``es. The orientation sign

    sigma(R, p) = +1  iff patch p's STORED normals point out of region R

is what the legacy solvers hand-derive case by case (the sign flips in
``mollified_bem.py:853-889`` and ``local_box_bem.py:212-332``). Here it
is inferred geometrically from signed solid angles:

  * If p (with its stored winding) ENCLOSES the region's probe point
    (|Omega| ~ 4*pi), the patch is the region's outer closed boundary:
    sigma = sign(Omega)   (+4*pi <=> wound outward around the probe).
  * If p is an open patch (0 < |Omega| < 2*pi), the van-Oosterom solid
    angle is NEGATIVE when viewed from the side the normal points to,
    so sigma = sign(Omega) again (normal toward probe => into R => -1).
  * If Omega ~ 0, p is a closed surface NOT enclosing the probe — an
    interior cavity boundary of R. Its winding is probed from inside
    the cavity (vertex centroid): outward-wound cavity walls point INTO
    R, so sigma = -sign(Omega_inside).

Validation: for every region the Gauss closure identity
    sum_p sigma(R,p) * Omega_p(x_R) = 4*pi
must hold (cavity patches contribute 0), and every INTERFACE patch must
get opposite signs from its two regions. ``orientation_overrides`` is
the escape hatch for pathological (strongly non-star-shaped) regions.

FAULTS get an orientation too — see ``FAULT_ORIENTATION`` below. It is
the ONE statement of the fault sign convention in msd; every site that
needs it calls ``RegionModel.orientation`` and uses the same ``-sigma``
expression a boundary patch uses, so there is no second place to get it
wrong. Runtime guard: ``mbem.selfcheck``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class BCType(enum.Enum):
    FREE_TRACTION = "free_traction"              # t prescribed (default 0); u unknown
    PRESCRIBED_DISPLACEMENT = "prescribed_displacement"  # u prescribed; t unknown
    INTERFACE = "interface"                      # u, t unknown; shared by 2 regions
    FAULT = "fault"                              # prescribed slip; interior source


# =====================================================================
# THE FAULT SIGN CONVENTION — stated here, ONCE, and nowhere else.
# =====================================================================
# sigma(R, p) = +1 iff patch p's stored normals point OUT of region R.
# For a boundary patch that is a genuine choice: R lies on one side of p
# and the stored winding either agrees with "out of R" or it does not,
# and RegionModel infers which from the signed solid angle.
#
# A FAULT has no such choice to make. It is INTERIOR to its region: the
# same region, with the same material, lies on BOTH of its faces, so
# there is no "out of R" to compare its normal against. What fixes the
# sign instead is the fault's OWN normal, which is already the reference
# direction of the slip it carries,
#
#     Du = u(x + 0+ n) - u(x - 0+ n),      n = the fault's stored normal,
#
# so "the region is on the +n side" is as true as "the region is on the
# -n side" and the only self-consistent answer is the one that makes the
# fault's normal its own outward direction:
FAULT_ORIENTATION = +1
# A fault is therefore NOT a special case of the sign rule. It is the
# sign rule at sigma = +1, and every formula treats it exactly like a
# PRESCRIBED boundary displacement — one coefficient, -sigma:
#
#   solve      (equations.py)  b[row] += -sigma * H_qf @ slip_f
#   readout u  (evaluate.py)   u      -= sigma * H_xf @ slip_f
#   readout s  (evaluate.py)   sigma_ij -= sigma * SH_xf @ slip_f
#                              (+ sigma * eigenstress, the divergent part
#                               of that same term, when subtracting it)
#   readout u  (evaluate.py)   DisplacementEvaluator term sign, -sigma
#                              (the compressed form of the same readout)
#
# which together are the single representation formula
#     u(x) = sum_p sigma G t_p - sum_p sigma H u_p - sum_f sigma_f H slip_f
# taken in the interior (readout) and in the boundary limit (solve).
#
# Flipping this constant does not make msd wrong in a visible way: it
# makes every fault answer consistently backwards, which no internal
# consistency check can see. ``mbem.selfcheck`` therefore pins it to a
# HARDCODED physical direction at runtime and refuses to let the library
# produce output if it has moved.


@dataclass(eq=False)
class Patch:
    """A named triangulated boundary piece with a boundary condition.

    ``value`` is the prescribed quantity, broadcastable to (N_tri, 3):
    traction for FREE_TRACTION, displacement for PRESCRIBED_DISPLACEMENT,
    slip for FAULT. ``None`` means zeros. Patch identity (``is``) is what
    links an INTERFACE patch shared by two regions — share the object.
    """
    name: str
    mesh: object              # TriMesh-compatible (vertices, triangles, ...)
    bc: BCType
    value: np.ndarray | None = None

    @property
    def n_triangles(self) -> int:
        return self.mesh.n_triangles

    def value_array(self) -> np.ndarray:
        v = np.zeros((self.n_triangles, 3)) if self.value is None \
            else np.broadcast_to(np.asarray(self.value, dtype=float),
                                 (self.n_triangles, 3))
        return np.ascontiguousarray(v)


@dataclass(eq=False)
class Region:
    name: str
    material: object                    # ElasticMaterial-compatible (mu, lam)
    patches: list                       # boundary Patches, declaration order
    probe_point: np.ndarray             # strictly interior point
    faults: list = field(default_factory=list)
    orientation_overrides: dict = field(default_factory=dict)  # name -> +-1


def _solid_angle_sum(mesh, x: np.ndarray) -> float:
    """Sum of van-Oosterom signed solid angles of all triangles from x."""
    tv = mesh.vertices[mesh.triangles]          # (M, 3, 3)
    r1 = tv[:, 0] - x
    r2 = tv[:, 1] - x
    r3 = tv[:, 2] - x
    R1 = np.linalg.norm(r1, axis=1)
    R2 = np.linalg.norm(r2, axis=1)
    R3 = np.linalg.norm(r3, axis=1)
    numer = np.einsum("ni,ni->n", r1, np.cross(r2, r3))
    denom = (R1 * R2 * R3
             + R3 * np.einsum("ni,ni->n", r1, r2)
             + R1 * np.einsum("ni,ni->n", r2, r3)
             + R2 * np.einsum("ni,ni->n", r1, r3))
    return float(np.sum(2.0 * np.arctan2(numer, denom)))


def patch_solid_angle(patch: Patch, x: np.ndarray) -> float:
    return _solid_angle_sum(patch.mesh, np.asarray(x, dtype=float))


_CLOSED_THRESHOLD = 2.0 * np.pi     # |Omega| above this => probe enclosed
_OPEN_FLOOR = 1e-3                  # |Omega| below this => treated as closed-cavity
_CLOSURE_TOL = 1e-6                 # closure identity tolerance (steradian, relative)


class RegionModel:
    def __init__(self, regions: list[Region]):
        self.regions = regions
        self._sigma: dict[tuple[str, str], int] = {}
        self.validate()

    # -- orientation ------------------------------------------------

    def _infer_sigma(self, region: Region, patch: Patch) -> int:
        if patch.name in region.orientation_overrides:
            return int(region.orientation_overrides[patch.name])
        x = np.asarray(region.probe_point, dtype=float)
        omega = patch_solid_angle(patch, x)
        if abs(omega) > _CLOSED_THRESHOLD or abs(omega) > _OPEN_FLOOR:
            return 1 if omega > 0 else -1
        # Closed patch not enclosing the probe: a cavity boundary.
        cavity_probe = patch.mesh.vertices.mean(axis=0)
        omega_in = patch_solid_angle(patch, cavity_probe)
        if abs(omega_in) < _CLOSED_THRESHOLD:
            raise ValueError(
                f"cannot infer orientation of patch '{patch.name}' for "
                f"region '{region.name}': solid angle from probe is "
                f"{omega:.3e} sr and the patch does not enclose its own "
                f"vertex centroid either; supply orientation_overrides")
        return -1 if omega_in > 0 else 1

    def orientation(self, region: Region, patch: Patch) -> int:
        """sigma(region, patch) — the ONE accessor for the sign rule.

        Answers for a boundary patch of ``region`` (inferred and validated
        in :meth:`validate`) and for a FAULT of ``region`` alike. A fault
        is interior to its region; its own normal defines the convention,
        so there is no side to choose and the answer is
        ``FAULT_ORIENTATION`` by construction (see the module header).
        Because faults answer here, every call site uses the same
        ``-sigma`` expression and no site carries a sign of its own.
        """
        if patch.bc is BCType.FAULT:
            if not any(f is patch for f in region.faults):
                raise ValueError(
                    f"patch '{patch.name}' is a FAULT but is not a fault of "
                    f"region '{region.name}'")
            return FAULT_ORIENTATION
        return self._sigma[(region.name, patch.name)]

    # -- validation --------------------------------------------------

    def validate(self) -> None:
        names = [r.name for r in self.regions]
        if len(set(names)) != len(names):
            raise ValueError("region names must be unique")

        # Patch incidence by identity
        incidence: dict[int, list[Region]] = {}
        by_id: dict[int, Patch] = {}
        pnames: dict[str, Patch] = {}
        for r in self.regions:
            for p in r.patches:
                incidence.setdefault(id(p), []).append(r)
                by_id[id(p)] = p
                existing = pnames.get(p.name)
                if existing is not None and existing is not p:
                    raise ValueError(
                        f"two distinct Patch objects share the name "
                        f"'{p.name}' — interface patches must be SHARED "
                        f"object instances")
                pnames[p.name] = p

        for pid, regs in incidence.items():
            p = by_id[pid]
            if p.bc is BCType.INTERFACE and len(regs) != 2:
                raise ValueError(f"interface patch '{p.name}' must belong to "
                                 f"exactly 2 regions, found {len(regs)}")
            if p.bc in (BCType.FREE_TRACTION, BCType.PRESCRIBED_DISPLACEMENT) \
                    and len(regs) != 1:
                raise ValueError(f"patch '{p.name}' ({p.bc.value}) must belong "
                                 f"to exactly 1 region, found {len(regs)}")
            if p.bc is BCType.FAULT:
                raise ValueError(f"fault patch '{p.name}' belongs in "
                                 f"Region.faults, not Region.patches")

        # Orientation inference + closure check
        for r in self.regions:
            total = 0.0
            for p in r.patches:
                sigma = self._infer_sigma(r, p)
                self._sigma[(r.name, p.name)] = sigma
                total += sigma * patch_solid_angle(p, r.probe_point)
            if abs(total - 4.0 * np.pi) > _CLOSURE_TOL * 4.0 * np.pi:
                raise ValueError(
                    f"region '{r.name}' fails the closure identity: "
                    f"sum sigma*Omega = {total:.6f} sr, expected 4*pi = "
                    f"{4*np.pi:.6f}. Check probe_point is interior and "
                    f"patch windings/orientation_overrides.")

        # Interface antisymmetry
        for pid, regs in incidence.items():
            p = by_id[pid]
            if p.bc is BCType.INTERFACE:
                s0 = self._sigma[(regs[0].name, p.name)]
                s1 = self._sigma[(regs[1].name, p.name)]
                if s0 != -s1:
                    raise ValueError(
                        f"interface '{p.name}' has non-antisymmetric "
                        f"orientations: {regs[0].name}:{s0}, "
                        f"{regs[1].name}:{s1}")

    # -- convenience -------------------------------------------------

    def interface_regions(self, patch: Patch) -> list[Region]:
        """Regions incident to a patch, in model region order."""
        return [r for r in self.regions if any(q is patch for q in r.patches)]

    def sigma_table(self) -> dict[tuple[str, str], int]:
        return dict(self._sigma)
