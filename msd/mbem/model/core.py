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
import inspect
from dataclasses import dataclass, field

import numpy as np

from .. import defaults
from ..kernels.basis import lagrange_nodes, lagrange_shape, n_nodes


def _call_value(fn, points, normals):
    """``fn(points)``, or ``fn(points, normals)`` when ``fn`` takes two
    required positional arguments (ddbem's convention): a traction datum
    needs the element normal at every node, and the caller should not have
    to rebuild the repeat-by-K itself. C callables count as one-argument."""
    try:
        n_pos = sum(1 for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty)
    except (TypeError, ValueError):
        n_pos = 1
    return fn(points, normals) if n_pos >= 2 else fn(points)


class BCType(enum.Enum):
    FREE_TRACTION = "free_traction"              # t prescribed (default 0); u unknown
    PRESCRIBED_DISPLACEMENT = "prescribed_displacement"  # u prescribed; t unknown
    INTERFACE = "interface"                      # u, t unknown; shared by 2 regions
    FAULT = "fault"                              # prescribed slip; interior source


# THE FAULT SIGN CONVENTION, stated once. A fault is interior to its region
# (same material on both faces), so its sigma is a convention: every site
# treats it as a prescribed boundary displacement with the one coefficient
# -sigma. With -1 that coefficient is +1 on the fault source and, n being
# the stored normal, Patch.value = u(x + 0+ n) - u(x - 0+ n) = b, the
# Burgers vector -- the same sign clq, ddbem and cutde use. +1 would be
# equally self-consistent (value = -b); mbem.selfcheck pins -1 against
# hardcoded physics.
FAULT_ORIENTATION = -1


@dataclass(eq=False)
class Patch:
    """A named triangulated boundary piece with a boundary condition.

    ``value`` is the prescribed quantity: traction for FREE_TRACTION,
    displacement for PRESCRIBED_DISPLACEMENT, slip for FAULT (the Burgers
    vector ``value = b = u(+n face) - u(-n face)`` with ``n`` the stored
    normal -- the sign clq and cutde use); ``None`` means zeros. It is a NODAL
    density of the patch's Lagrange order: a (3,) constant, an (N_tri, 3)
    per-element array (every node of the element), a (K N_tri, 3) nodal
    array, or a callable sampled at the nodes -- ``f(points (M, 3)) -> (M, 3)``,
    or ``f(points, normals)`` with the element normal at every node (a
    traction datum). Patch identity (``is``) is what links an INTERFACE
    patch shared by two regions -- share the object.

    ``order`` in {0, 1, 2} (K = 1, 3, 6 nodes per element) sets the
    DISCONTINUOUS nodal layout: row K s + k is node k of element s in the
    kernels' node order (P0 centroid; P1 v1, v2, v3; P2 v1, v2, v3, m12, m23,
    m31), matching the kernel column layout 3 (K s + k) + j. One collocation
    point per node, pulled toward the centroid by
    ``defaults.COLLOCATION_SHRINK_BY_ORDER`` (the basis does not move), so
    the system stays square. At P0 the node IS the centroid.
    """
    name: str
    mesh: object              # TriMesh-compatible (vertices, triangles, ...)
    bc: BCType
    value: object = None
    order: int = 0

    def __post_init__(self):
        if self.order != int(self.order):       # 1.5 must not truncate to P1
            raise ValueError(f"patch '{self.name}': order must be an integer "
                             f"0, 1 or 2 (got {self.order!r})")
        self.order = int(self.order)
        n_nodes(self.order)                     # raises unless 0, 1, 2
        self._refuse_degenerate()

    def _refuse_degenerate(self) -> None:
        """A patch's mesh may not carry a collapsed element.

        A triangle whose height over its longest edge is below
        ``defaults.MIN_TRIANGLE_HEIGHT_OVER_L`` has no frame for the
        kernels to integrate over (they return a zero block), no mesh
        scale for ``eps="auto"``, and a collocation point on top of its
        neighbour's -- which shrinks its cluster's bounding box to a
        point, so the compressed backend's admissibility test accepts a
        block whose elements TOUCH and compresses a near field. It is a
        mesher failure and is refused here, at the one place every
        backend goes through, rather than diagnosed later as a
        compression or conditioning problem.
        """
        tv = np.asarray(self.mesh.vertices, float)[np.asarray(self.mesh.triangles)]
        edges = tv[:, [1, 2, 0], :] - tv
        L = np.linalg.norm(edges, axis=2).max(axis=1)
        two_area = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0],
                                           tv[:, 2] - tv[:, 0]), axis=1)
        h_over_L = two_area / np.where(L > 0.0, L, 1.0) ** 2
        bad = np.nonzero(h_over_L <= defaults.MIN_TRIANGLE_HEIGHT_OVER_L)[0]
        if bad.size:
            worst = int(bad[np.argmin(h_over_L[bad])])
            raise ValueError(
                f"patch '{self.name}': {bad.size} of {tv.shape[0]} elements "
                f"are degenerate (height / longest edge at or below "
                f"{defaults.MIN_TRIANGLE_HEIGHT_OVER_L:g}; worst "
                f"{h_over_L[worst]:.1e} at element {worst})")

    @property
    def n_triangles(self) -> int:
        return self.mesh.n_triangles

    @property
    def n_nodes(self) -> int:
        """K * N_tri: density rows / collocation points of the patch."""
        return n_nodes(self.order) * self.mesh.n_triangles

    def _points(self, lam) -> np.ndarray:
        """Points (K N_tri, 3) at barycentric ``lam`` (K, 3) of every element,
        element-major."""
        tv = np.asarray(self.mesh.vertices, float)[np.asarray(self.mesh.triangles)]
        return np.einsum("kv,svc->skc", lam, tv).reshape(-1, 3)

    def nodes(self) -> np.ndarray:
        """Node coordinates (K N_tri, 3); at P0 the mesh's own centroids."""
        if self.order == 0:
            return np.asarray(self.mesh.centroids(), float)
        return self._points(lagrange_nodes(self.order))

    def collocation_points(self) -> np.ndarray:
        """Collocation points (K N_tri, 3): the nodes pulled toward the
        centroid by ``COLLOCATION_SHRINK_BY_ORDER[order]``."""
        t = defaults.COLLOCATION_SHRINK_BY_ORDER[self.order]
        if t == 0.0:
            return self.nodes()
        return self._points(lagrange_nodes(self.order, t))

    def collocation_shape(self) -> np.ndarray:
        """N[kc, k] = N_k(x_c) (K, K): the shape functions at the collocation
        points, which spread a row's free term over its element's nodes.
        The identity at P0 and at shrink 0."""
        t = defaults.COLLOCATION_SHRINK_BY_ORDER[self.order]
        return lagrange_shape(self.order, lagrange_nodes(self.order, t))

    def value_array(self) -> np.ndarray:
        """The prescribed value as a nodal density (K N_tri, 3) (docstring)."""
        K = n_nodes(self.order)
        if self.value is None:
            return np.zeros((self.n_nodes, 3))
        if callable(self.value):
            normals, _ = self.mesh.normals_and_areas()
            v = np.asarray(_call_value(self.value, self.nodes(),
                                       np.repeat(normals, K, axis=0)), dtype=float)
            if v.shape != (self.n_nodes, 3):
                raise ValueError(
                    f"patch '{self.name}': callable value returned {v.shape}, "
                    f"expected {(self.n_nodes, 3)}")
            return np.ascontiguousarray(v)
        v = np.asarray(self.value, dtype=float)
        if K > 1 and v.shape == (self.n_nodes, 3):
            return np.ascontiguousarray(v)
        v = np.broadcast_to(v, (self.n_triangles, 3))
        if K > 1:
            v = np.repeat(v, K, axis=0)
        return np.ascontiguousarray(v)

    def collocation_values(self, nodal) -> np.ndarray:
        """A nodal density (K N_tri, 3) interpolated to the collocation
        points, ``sum_k N[kc, k] v[s, k]`` -- what the free term multiplies."""
        K = n_nodes(self.order)
        v = np.asarray(nodal, float).reshape(self.n_triangles, K, 3)
        return np.einsum("ck,skj->scj", self.collocation_shape(),
                         v).reshape(-1, 3)


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
        is interior to its region, so there is no side to infer and the
        answer is the convention ``FAULT_ORIENTATION`` (module header).
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

        # Faults: FAULT bc, one owning region, model-wide unique names
        # (eps is looked up by name, so a collision takes another patch's
        # eps), and containment in the owner via the same closure sum.
        fowner: dict[int, tuple[Patch, list[str]]] = {}
        fnames: dict[str, Patch] = {}
        for r in self.regions:
            for f in r.faults:
                if f.bc is not BCType.FAULT:
                    raise ValueError(
                        f"fault '{f.name}' of region '{r.name}' has bc "
                        f"{f.bc.value}; Region.faults holds FAULT patches only")
                fowner.setdefault(id(f), (f, []))[1].append(r.name)
                if f.name in pnames or fnames.get(f.name, f) is not f:
                    raise ValueError(
                        f"fault '{f.name}' of region '{r.name}' shares its "
                        f"name with another patch or fault; names must be "
                        f"unique model-wide")
                fnames[f.name] = f
        for f, owners in fowner.values():
            if len(owners) != 1:
                raise ValueError(
                    f"fault '{f.name}' must belong to exactly 1 region, "
                    f"found {len(owners)}: {owners}")
        # Containment is tested at EVERY fault-triangle centroid (a fault
        # whose mean vertex is inside can still poke through an interface):
        # the closure sum says which region a point is in, and an exact
        # distance test catches a fault lying ON a patch, where the solid
        # angle is ambiguous.
        from ..geometry import distance_to_mesh
        for r in self.regions:
            for f in r.faults:
                cs = f.mesh.centroids()
                for p in r.patches:
                    d, idx = distance_to_mesh(cs, p.mesh)
                    h = np.linalg.norm(p.mesh.vertices[p.mesh.triangles[idx, 1]]
                                       - p.mesh.vertices[p.mesh.triangles[idx, 0]],
                                       axis=1)
                    k = int(np.argmin(d / h))
                    if d[k] < 1e-9 * h[k]:
                        raise ValueError(
                            f"fault '{f.name}' triangle {k} lies ON boundary "
                            f"patch '{p.name}' of region '{r.name}'; a fault "
                            f"on an interface must be split: split it at the "
                            f"interface")
                for k, c in enumerate(cs):
                    total = sum(self._sigma[(r.name, p.name)]
                                * patch_solid_angle(p, c) for p in r.patches)
                    if abs(total - 4.0 * np.pi) > _CLOSURE_TOL * 4.0 * np.pi:
                        raise ValueError(
                            f"fault '{f.name}' triangle {k} is not inside "
                            f"region '{r.name}': closure sum at its centroid "
                            f"= {total:.6f} sr, expected 4*pi")

    # -- convenience -------------------------------------------------

    def is_anchored(self) -> bool:
        """True if some patch prescribes displacement. On an un-anchored
        model ``jump="calibrated"`` makes rigid translations an exact null
        space, so the backends require ``deflate=True`` there."""
        return any(p.bc is BCType.PRESCRIBED_DISPLACEMENT
                   for r in self.regions for p in r.patches)

    def interface_regions(self, patch: Patch) -> list[Region]:
        """Regions incident to a patch, in model region order."""
        return [r for r in self.regions if any(q is patch for q in r.patches)]

    def sigma_table(self) -> dict[tuple[str, str], int]:
        return dict(self._sigma)
