"""Deterministic unknown layout for a RegionModel.

Slot emission rule (reproduces the legacy orderings exactly):
regions in model order; within a region, patches in declaration order,
skipping patches already emitted; FREE_TRACTION emits its u-slot,
PRESCRIBED_DISPLACEMENT emits its t-slot, INTERFACE emits u then t.

A slot holds the patch's dof values, 3 * n_dofs entries: the element-major
nodal density (K nodes per element) of a discontinuous patch, the shared-node
values of a continuous one; a patch's rows are its n_dofs collocation
points, so the system is square at every order and layout. After a region's
patches come the "b" slots of its FAULT_TRACTION faults (declaration order):
the solved Burgers vector, three Cartesian components per node, whose rows
are the traction conditions at the fault's own nodes.

Row assignment: the BIE of region R collocated on patch q lands on
  * q's u-slot if q is FREE_TRACTION,
  * q's t-slot if q is PRESCRIBED_DISPLACEMENT,
  * for INTERFACE: the first incident region (model order) takes the
    u-slot row, the second takes the t-slot row — keeping the
    second-kind (1/2 I + sigma H) blocks on the u-row diagonals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import BCType, Patch, Region, RegionModel


@dataclass(frozen=True)
class Slot:
    patch: Patch
    kind: str          # "u" | "t" | "b"
    offset: int
    size: int          # 3 * patch.n_dofs

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.patch.name}"

    @property
    def stop(self) -> int:
        return self.offset + self.size


class UnknownLayout:
    def __init__(self, model: RegionModel):
        self.model = model
        self.slots: list[Slot] = []
        self._by_key: dict[tuple[int, str], Slot] = {}

        offset = 0
        emitted: set[int] = set()
        for region in model.regions:
            for patch in region.patches:
                if id(patch) in emitted:
                    continue
                emitted.add(id(patch))
                kinds = {
                    BCType.FREE_TRACTION: ("u",),
                    BCType.PRESCRIBED_DISPLACEMENT: ("t",),
                    BCType.INTERFACE: ("u", "t"),
                }[patch.bc]
                for kind in kinds:
                    slot = Slot(patch, kind, offset, 3 * patch.n_dofs)
                    self.slots.append(slot)
                    self._by_key[(id(patch), kind)] = slot
                    offset += slot.size
            for f in region.faults:
                if f.bc is not BCType.FAULT_TRACTION or id(f) in emitted:
                    continue
                emitted.add(id(f))
                slot = Slot(f, "b", offset, 3 * f.n_dofs)
                self.slots.append(slot)
                self._by_key[(id(f), "b")] = slot
                offset += slot.size
        self.n_unknowns = offset

    def has_kind(self, kind: str) -> bool:
        return any(s.kind == kind for s in self.slots)

    def slot(self, patch: Patch, kind: str) -> Slot:
        return self._by_key[(id(patch), kind)]

    def has_slot(self, patch: Patch, kind: str) -> bool:
        return (id(patch), kind) in self._by_key

    def row_slot(self, region: Region, patch: Patch) -> Slot:
        if patch.bc is BCType.FAULT_TRACTION:
            return self.slot(patch, "b")
        if patch.bc is BCType.FREE_TRACTION:
            return self.slot(patch, "u")
        if patch.bc is BCType.PRESCRIBED_DISPLACEMENT:
            return self.slot(patch, "t")
        if patch.bc is BCType.INTERFACE:
            first, second = self.model.interface_regions(patch)
            if region is first:
                return self.slot(patch, "u")
            if region is second:
                return self.slot(patch, "t")
            raise ValueError(f"region '{region.name}' is not incident to "
                             f"interface '{patch.name}'")
        raise ValueError(f"no collocation row for bc {patch.bc}")
