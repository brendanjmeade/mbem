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
wrong. Runtime guard: ``mbem.selfcheck``. Both fault kinds (prescribed
slip, FAULT; solved slip, FAULT_TRACTION) answer alike.

A region with NO boundary patch is the whole space: nothing to close, every
point inside, and its faults the only sources or unknowns (the penny-crack
gate lives there). It must have at least one fault.

GRAVITY is a property of the model, not of a patch or a material: ``Gravity``
below, held as ``RegionModel.gravity``, and turned into the one condition it
imposes by ``RegionModel.buoyancy``.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass, field, replace
from typing import Mapping

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
    FAULT_TRACTION = "fault_traction"            # traction change prescribed; slip unknown


#: The two fault kinds. Both live in ``Region.faults``, both carry
#: FAULT_ORIENTATION, both take one eps per fault; they differ in what is
#: known on them (the slip, or the traction it must produce).
FAULT_KINDS = (BCType.FAULT, BCType.FAULT_TRACTION)

#: A point this close to a boundary patch, in units of the nearest element's
#: edge, lies ON it: a fault element centroid there is refused (the solid
#: angle is ambiguous), a traction fault's boundary NODE there is a trace on
#: that patch and stays free (``Patch.locked_mask``).
_ON_PATCH_TOL = 1e-9


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
    normal -- the sign clq and cutde use), and for FAULT_TRACTION the
    prescribed traction CHANGE at the collocation nodes, the datum its solved
    slip must produce (a drop ``dtau`` along ``s`` is ``value = -dtau s``);
    ``None`` means zeros. It is a NODAL datum of the patch's layout: a (3,)
    constant, an (N_tri, 3) per-element array (every node of the element;
    discontinuous layout only), an (n_dofs, 3) nodal array, or a callable
    sampled at the dof nodes -- ``f(points (M, 3)) -> (M, 3)``, or
    ``f(points, normals)`` with the normal at every node (a traction datum).
    ``value_array()`` is that datum at the dof nodes; ``density()`` is the
    element-major (K N_tri, 3) form the kernels take. Patch identity
    (``is``) is what links an INTERFACE patch shared by two regions -- share
    the object.

    ``order`` in {0, 1, 2} (K = 1, 3, 6 nodes per element) and ``layout``:

    * ``"discontinuous"`` (default): row K s + k is node k of element s in
      the kernels' node order (P0 centroid; P1 v1, v2, v3; P2 v1, v2, v3,
      m12, m23, m31), matching the kernel column layout 3 (K s + k) + j. One
      collocation point per node, pulled toward the centroid by
      ``defaults.COLLOCATION_SHRINK_BY_ORDER`` (the basis does not move), so
      the system stays square. At P0 the node IS the centroid.
    * ``"continuous"`` (order 2, faults only): one value per SHARED node
      (welded vertices, then edge midpoints; ``model/shared_nodes.py``),
      ``n_dofs`` of them; element s reads its six through ``column_ids()``,
      so the slip is continuous across every edge by construction. The
      collocation points are the shared nodes themselves, unshrunk: the
      point of the layout, legal because the mollified kernel bounds the
      gradient kink there. Faults only, because a boundary row's free term
      is spread over its own element's columns (``collocation_shape``) and a
      shared node has no own element; a fault row has no free term.

    FAULT_TRACTION: the slip is an unknown of the solve (slot ``"b"``, three
    Cartesian components per node) and the rows are the elastic traction at
    the nodes, equal to ``value``. ``allow_opening=False`` (default) enforces
    the two tangential components and constrains ``b . n = 0``; the value may
    then carry no normal component (project it, or set ``allow_opening=True``
    to enforce all three). ``locked``: nodes whose slip is fixed at zero -- a
    bool mask over the dof nodes, a callable of their coordinates, or
    ``None`` for the default: every boundary node of a continuous fault that
    does not lie on a boundary patch of its region (a crack front inside the
    body locks; a trace on the free surface stays free), and nothing at P0.
    Allowed: order 0 discontinuous (centroid collocation, the P0 control) and
    order 2 continuous.
    """
    name: str
    mesh: object              # TriMesh-compatible (vertices, triangles, ...)
    bc: BCType
    value: object = None
    order: int = 0
    layout: str = "discontinuous"
    allow_opening: bool = False
    locked: object = None

    def __post_init__(self):
        if self.order != int(self.order):       # 1.5 must not truncate to P1
            raise ValueError(f"patch '{self.name}': order must be an integer "
                             f"0, 1 or 2 (got {self.order!r})")
        self.order = int(self.order)
        n_nodes(self.order)                     # raises unless 0, 1, 2
        if self.layout not in ("discontinuous", "continuous"):
            raise ValueError(f"patch '{self.name}': layout must be "
                             f"'discontinuous' or 'continuous' (got {self.layout!r})")
        if self.layout == "continuous":
            if self.bc not in FAULT_KINDS:
                raise ValueError(
                    f"patch '{self.name}': a continuous layout is for faults "
                    f"only (bc {self.bc.value}): a boundary row's free term is "
                    f"spread over its own element's columns, which a shared "
                    f"node does not have, and unshrunk hypersingular boundary "
                    f"rows are singular (docs/ddbem-findings.md)")
            if self.order != 2:
                raise ValueError(
                    f"patch '{self.name}': a continuous layout is P2 only "
                    f"(got order {self.order}): continuous P1 measured 15-22 % "
                    f"too stiff and first order on the penny crack, and P0 "
                    f"has no shared node")
        if self.bc is BCType.FAULT_TRACTION:
            ok = ((self.order == 0 and self.layout == "discontinuous")
                  or (self.order == 2 and self.layout == "continuous"))
            if not ok:
                raise ValueError(
                    f"patch '{self.name}': a FAULT_TRACTION fault is order 0 "
                    f"discontinuous (centroid collocation) or order 2 "
                    f"continuous (shared nodes); got order {self.order} "
                    f"{self.layout} -- discontinuous P1/P2 traction rows "
                    f"collocate at shrunk nodes and are not gated")
        elif self.allow_opening or self.locked is not None:
            raise ValueError(f"patch '{self.name}': allow_opening and locked "
                             f"apply to a FAULT_TRACTION fault only")
        self._shared = None
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
        """K * N_tri: density rows of the patch (the kernels' element-major
        layout), at either layout."""
        return n_nodes(self.order) * self.mesh.n_triangles

    @property
    def continuous(self) -> bool:
        return self.layout == "continuous"

    @property
    def shared(self):
        """The shared-node table of a continuous patch, built once per
        patch (``model/shared_nodes.py``); None for the discontinuous layout."""
        if not self.continuous:
            return None
        if self._shared is None:
            from .shared_nodes import build_shared_nodes
            self._shared = build_shared_nodes(self.mesh, self.order)
        return self._shared

    @property
    def n_dofs(self) -> int:
        """Dof nodes: the shared nodes of a continuous patch, else the
        K N_tri density rows. Slots are 3 n_dofs; collocation points
        n_dofs."""
        return self.shared.n if self.continuous else self.n_nodes

    def column_ids(self):
        """(N_tri, K) shared-node id of every element node (continuous), or
        None: the kernels' own element-major columns."""
        return self.shared.ids if self.continuous else None

    def expand(self, values) -> np.ndarray:
        """Dof-node values (n_dofs, 3) -> the element-major density (K N_tri, 3)."""
        if self.continuous:
            return self.shared.expand(values)
        v = np.asarray(values, float)
        if v.shape != (self.n_nodes, 3):
            raise ValueError(f"patch '{self.name}': values must be "
                             f"({self.n_nodes}, 3), got {v.shape}")
        return np.ascontiguousarray(v)

    def _points(self, lam) -> np.ndarray:
        """Points (K N_tri, 3) at barycentric ``lam`` (K, 3) of every element,
        element-major."""
        tv = np.asarray(self.mesh.vertices, float)[np.asarray(self.mesh.triangles)]
        return np.einsum("kv,svc->skc", lam, tv).reshape(-1, 3)

    def nodes(self) -> np.ndarray:
        """Node coordinates (K N_tri, 3) of the element-major layout; at P0
        the mesh's own centroids. (A continuous patch's dof nodes are
        ``shared.coords``; this is still its density's node table.)"""
        if self.order == 0:
            return np.asarray(self.mesh.centroids(), float)
        return self._points(lagrange_nodes(self.order))

    def collocation_points(self) -> np.ndarray:
        """Collocation points (n_dofs, 3): the nodes pulled toward the
        centroid by ``COLLOCATION_SHRINK_BY_ORDER[order]``, or, continuous,
        the shared nodes themselves."""
        if self.continuous:
            return self.shared.coords
        t = defaults.COLLOCATION_SHRINK_BY_ORDER[self.order]
        if t == 0.0:
            return self.nodes()
        return self._points(lagrange_nodes(self.order, t))

    def collocation_normals(self) -> np.ndarray:
        """Unit normals (n_dofs, 3) at the collocation points: the element
        normal at every node, or the node normals of a continuous patch."""
        if self.continuous:
            return self.shared.normals
        normals, _ = self.mesh.normals_and_areas()
        return np.ascontiguousarray(np.repeat(np.asarray(normals, float),
                                              n_nodes(self.order), axis=0))

    def collocation_shape(self) -> np.ndarray:
        """N[kc, k] = N_k(x_c) (K, K): the shape functions at the collocation
        points, which spread a row's free term over its element's nodes.
        The identity at P0 and at shrink 0. A continuous patch has no free
        term to spread and refuses."""
        if self.continuous:
            raise ValueError(f"patch '{self.name}': a continuous patch has no "
                             f"free term (fault rows carry none) and no "
                             f"collocation shape")
        t = defaults.COLLOCATION_SHRINK_BY_ORDER[self.order]
        return lagrange_shape(self.order, lagrange_nodes(self.order, t))

    def value_array(self) -> np.ndarray:
        """The prescribed datum at the dof nodes, (n_dofs, 3) (docstring)."""
        if self.continuous:
            sh = self.shared
            n = sh.n
            if self.value is None:
                v = np.zeros((n, 3))
            elif callable(self.value):
                v = np.asarray(_call_value(self.value, sh.coords, sh.normals),
                               dtype=float)
                if v.shape != (n, 3):
                    raise ValueError(
                        f"patch '{self.name}': callable value returned "
                        f"{v.shape}, expected {(n, 3)} (one row per shared node)")
            else:
                v = np.asarray(self.value, dtype=float)
                if v.shape == (3,):
                    v = np.broadcast_to(v, (n, 3))
                elif v.shape != (n, 3):
                    raise ValueError(
                        f"patch '{self.name}': a continuous patch takes a (3,) "
                        f"constant, an ({n}, 3) shared-node array or a "
                        f"callable; got {v.shape} (a per-element or "
                        f"element-major array has no shared-node meaning)")
            v = np.ascontiguousarray(v)
        else:
            K = n_nodes(self.order)
            if self.value is None:
                v = np.zeros((self.n_nodes, 3))
            elif callable(self.value):
                normals, _ = self.mesh.normals_and_areas()
                v = np.asarray(_call_value(self.value, self.nodes(),
                                           np.repeat(normals, K, axis=0)), dtype=float)
                if v.shape != (self.n_nodes, 3):
                    raise ValueError(
                        f"patch '{self.name}': callable value returned {v.shape}, "
                        f"expected {(self.n_nodes, 3)}")
                v = np.ascontiguousarray(v)
            else:
                v = np.asarray(self.value, dtype=float)
                if not (K > 1 and v.shape == (self.n_nodes, 3)):
                    v = np.broadcast_to(v, (self.n_triangles, 3))
                    if K > 1:
                        v = np.repeat(v, K, axis=0)
                v = np.ascontiguousarray(v)
        if self.bc is BCType.FAULT_TRACTION and not self.allow_opening:
            self._refuse_normal_component(v)
        return v

    def _refuse_normal_component(self, v: np.ndarray) -> None:
        """With ``allow_opening=False`` only the tangential traction is
        enforced: a value with a normal component would be projected
        silently, so it is refused instead."""
        scale = float(np.max(np.abs(v))) if v.size else 0.0
        if scale == 0.0:
            return
        nn = np.einsum("ij,ij->i", v, self.collocation_normals())
        worst = float(np.max(np.abs(nn)))
        if worst > defaults.TRACTION_VALUE_NORMAL_TOL * scale:
            raise ValueError(
                f"fault '{self.name}': the traction value has a normal "
                f"component ({worst:.2e} against |value| {scale:.2e}) but "
                f"allow_opening=False enforces the tangential components "
                f"only: project the value onto the tangent plane, or set "
                f"allow_opening=True to enforce all three")

    def density(self) -> np.ndarray:
        """The prescribed value as the kernels' element-major density
        (K N_tri, 3): ``expand(value_array())``. A FAULT_TRACTION fault has
        no prescribed density (its slip is solved) and refuses."""
        if self.bc is BCType.FAULT_TRACTION:
            raise ValueError(
                f"fault '{self.name}' is a FAULT_TRACTION fault: its slip is "
                f"solved, not prescribed -- read solution['b:{self.name}'] "
                f"(dof nodes) or expand() it")
        return self.expand(self.value_array())

    def collocation_values(self, nodal) -> np.ndarray:
        """A nodal density (K N_tri, 3) interpolated to the collocation
        points, ``sum_k N[kc, k] v[s, k]`` -- what the free term multiplies."""
        K = n_nodes(self.order)
        v = np.asarray(nodal, float).reshape(self.n_triangles, K, 3)
        return np.einsum("ck,skj->scj", self.collocation_shape(),
                         v).reshape(-1, 3)

    def integrate(self, values) -> np.ndarray:
        """Integral over the patch of a dof-node field (n_dofs, 3) -> (3,):
        at P0 the centroid value times the area, at P1 the vertex rule, at
        P2 the midpoint rule (A/3 times the three edge-node values, exact
        for quadratics) -- the moment of a slip field, at either layout."""
        if self.continuous:
            return self.shared.integrate(values)
        K = n_nodes(self.order)
        v = np.asarray(values, float)
        if v.shape != (self.n_nodes, 3):
            raise ValueError(f"patch '{self.name}': values must be "
                             f"({self.n_nodes}, 3), got {v.shape}")
        v = v.reshape(self.n_triangles, K, 3)
        _, areas = self.mesh.normals_and_areas()
        areas = np.asarray(areas, float)
        if self.order == 2:
            per = v[:, 3:6].sum(axis=1) / 3.0
        elif self.order == 1:
            per = v[:, 0:3].sum(axis=1) / 3.0
        else:
            per = v[:, 0]
        return (areas[:, None] * per).sum(axis=0)

    def locked_mask(self, patches=()) -> np.ndarray:
        """(n_dofs,) bool, FAULT_TRACTION only: the nodes whose slip is fixed
        at zero (docstring). ``patches`` are the boundary patches of the
        owning region, which the default rule needs to tell a crack front
        from a trace."""
        if self.bc is not BCType.FAULT_TRACTION:
            raise ValueError(f"patch '{self.name}': locked nodes are a "
                             f"FAULT_TRACTION notion")
        n = self.n_dofs
        if self.locked is None:
            if not self.continuous:
                return np.zeros(n, bool)
            sh = self.shared
            mask = sh.boundary.copy()
            idx = np.flatnonzero(mask)
            if idx.size and patches:
                from ..geometry import distance_to_mesh
                pts = sh.coords[idx]
                on = np.zeros(idx.size, bool)
                for p in patches:
                    d, near = distance_to_mesh(pts, p.mesh)
                    tv = np.asarray(p.mesh.vertices, float)[
                        np.asarray(p.mesh.triangles)[near]]
                    h = np.linalg.norm(tv[:, 1] - tv[:, 0], axis=1)
                    on |= d < _ON_PATCH_TOL * h
                mask[idx[on]] = False
            return mask
        if callable(self.locked):
            m = np.asarray(self.locked(self.collocation_points()), bool).reshape(-1)
        else:
            m = np.asarray(self.locked, bool).reshape(-1)
        if m.shape != (n,):
            raise ValueError(f"patch '{self.name}': locked mask must have "
                             f"{n} entries (one per dof node), got {m.shape}")
        return m


@dataclass(eq=False)
class Region:
    name: str
    material: object                    # ElasticMaterial-compatible (mu, lam)
    patches: list                       # boundary Patches, declaration order; [] = the full space
    probe_point: np.ndarray             # strictly interior point (unused for the full space)
    faults: list = field(default_factory=list)
    orientation_overrides: dict = field(default_factory=dict)  # name -> +-1

    @property
    def is_full_space(self) -> bool:
        """No boundary patch: the whole space, its faults the only sources
        or unknowns."""
        return len(self.patches) == 0


@dataclass(frozen=True)
class Gravity:
    """Gravity, as buoyancy on the boundary (docs/viscoelastic.md, section 7).

    The body is at rest under gravity before it is deformed, its prestress
    hydrostatic. A displacement then carries prestressed material with it, and
    where a boundary separates two densities that leaves a restoring load: a
    surface lifted by ``w`` carries ``rho g w`` more rock than before. With the
    material treated as incompressible IN THE GRAVITY TERMS ONLY, that is the
    whole effect -- the equations inside a region stay homogeneous, the
    kernels are untouched, and gravity is a condition on its boundary:

        region R sees on patch p the traction   t_p + B u_p,
        B = -(rho_R - rho_ref) g  n (x) up      (n the STORED normal)

    with ``rho_ref`` what lies on the other side of ``p``:

    * 0 on a patch named in ``surface`` (air): the free surface, where the
      load is the weight of the relief the displacement builds;
    * the first incident region's density on an INTERFACE, so the first region
      sees ``t_p`` itself, the second ``t_p`` plus the buoyancy of the
      contrast, and equal densities give exactly the model without gravity;
    * ``rho_R`` on every other outer face (no term): the sides and base of a
      box are cuts through material that continues, not cliffs, and a
      hydrostatic surrounding of the same density loads them with nothing.
      It is also a reservoir, which is what lets a relaxed substrate keep its
      pressure as a half-space would.

    A fault carries no term: the stress at a POINT is continuous across it.

    ``density`` (kg/m^3) names EVERY region; ``g`` is m/s^2. ``up`` is the
    direction buoyancy acts along, one vector for a flat earth, or, with
    ``center``, the radial direction from that point (a sphere).

    The traction of a slot or a datum is the MATERIAL one where the two
    differ: on the free surface ``Patch.value`` is the applied load, and the
    stress read back by ``evaluate_stress`` is the stress at a fixed point in
    space, which is the material stress minus ``rho g (u . up) I`` -- the shear
    components are the same.

    The error of the incompressible treatment is of relative order
    ``rho g L / K`` (L the scale of the deformation, K the bulk modulus): 6 %
    per 100 km at K = 50 GPa, and nothing in an incompressible body.
    """
    density: Mapping
    surface: tuple = ()
    g: float = defaults.GRAVITY
    up: tuple = (0.0, 0.0, 1.0)
    center: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "density",
                           {str(k): float(v) for k, v in self.density.items()})
        surface = (self.surface,) if isinstance(self.surface, str) \
            else tuple(self.surface)
        object.__setattr__(self, "surface", surface)
        if any(not (v >= 0.0 and np.isfinite(v)) for v in self.density.values()):
            raise ValueError(f"densities must be finite and >= 0: {self.density}")
        if not (self.g >= 0.0 and np.isfinite(self.g)):
            raise ValueError(f"g {self.g} must be finite and >= 0")
        up = np.asarray(self.up, float)
        if up.shape != (3,) or not np.linalg.norm(up) > 0.0:
            raise ValueError(f"up {self.up!r} must be a nonzero 3-vector")
        object.__setattr__(self, "up", tuple(float(v) for v in
                                             up / np.linalg.norm(up)))
        if self.center is not None:
            c = np.asarray(self.center, float)
            if c.shape != (3,):
                raise ValueError(f"center {self.center!r} must be a point")
            object.__setattr__(self, "center", tuple(float(v) for v in c))

    def weight(self, region_name: str) -> float:
        """``rho g`` of a region in the trunk's units, GPa/km."""
        return (self.density[region_name] * self.g
                * defaults.WEIGHT_GPA_PER_KM)

    def up_at(self, points) -> np.ndarray:
        """Unit up vectors (N, 3) at ``points``."""
        pts = np.asarray(points, float).reshape(-1, 3)
        if self.center is None:
            return np.broadcast_to(np.asarray(self.up), pts.shape).copy()
        d = pts - np.asarray(self.center)
        r = np.linalg.norm(d, axis=1)
        if np.any(r == 0.0):
            raise ValueError("a node lies at the centre of gravity: no up there")
        return d / r[:, None]


def _solid_angle_sum(mesh, x: np.ndarray) -> float:
    """Sum of van-Oosterom signed solid angles of all triangles from x.

    Delegates: the formula lives in ``geometry._solid_angle_rows`` because the
    volume sampler needs it over a grid, and a second copy here is the one that
    would drift.
    """
    from ..geometry import solid_angle_batch
    return float(solid_angle_batch(mesh, np.asarray(x, dtype=float))[0])


def patch_solid_angle(patch: Patch, x: np.ndarray) -> float:
    return _solid_angle_sum(patch.mesh, np.asarray(x, dtype=float))


_CLOSED_THRESHOLD = 2.0 * np.pi     # |Omega| above this => probe enclosed
_OPEN_FLOOR = 1e-3                  # |Omega| below this => treated as closed-cavity
_CLOSURE_TOL = 1e-6                 # closure identity tolerance (steradian, relative)


class RegionModel:
    def __init__(self, regions: list[Region], gravity: Gravity | None = None):
        self.regions = regions
        self.gravity = gravity
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
        if patch.bc in FAULT_KINDS:
            if not any(f is patch for f in region.faults):
                raise ValueError(
                    f"patch '{patch.name}' is a {patch.bc.name} but is not a "
                    f"fault of region '{region.name}'")
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
            if p.bc in FAULT_KINDS:
                raise ValueError(f"fault patch '{p.name}' belongs in "
                                 f"Region.faults, not Region.patches")

        # Orientation inference + closure check (a region with no boundary
        # patch is the full space: nothing to close, but it must carry a
        # fault, or there is nothing to solve or evaluate)
        for r in self.regions:
            if r.is_full_space:
                if not r.faults:
                    raise ValueError(
                        f"region '{r.name}' has no boundary patch and no "
                        f"fault: nothing to solve or evaluate (an empty "
                        f"patch list is the full space)")
                continue
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
                if f.bc not in FAULT_KINDS:
                    raise ValueError(
                        f"fault '{f.name}' of region '{r.name}' has bc "
                        f"{f.bc.value}; Region.faults holds FAULT patches only "
                        f"(FAULT or FAULT_TRACTION)")
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
                    if d[k] < _ON_PATCH_TOL * h[k]:
                        raise ValueError(
                            f"fault '{f.name}' triangle {k} lies ON boundary "
                            f"patch '{p.name}' of region '{r.name}'; a fault "
                            f"on an interface must be split: split it at the "
                            f"interface")
                inside, total = self.point_in_region(r, cs, closure=True)
                if not inside.all():
                    k = int(np.argmax(~inside))
                    raise ValueError(
                        f"fault '{f.name}' triangle {k} is not inside "
                        f"region '{r.name}': closure sum at its centroid "
                        f"= {total[k]:.6f} sr, expected 4*pi")

        self._validate_gravity(pnames)

    def _validate_gravity(self, pnames: dict) -> None:
        """Every region has a density, and each ``surface`` name is an outer
        FREE_TRACTION patch: the free surface is where the air is, and a
        clamped face has no relief to weigh."""
        g = self.gravity
        if g is None:
            return
        names = {r.name for r in self.regions}
        if set(g.density) != names:
            raise ValueError(
                f"gravity.density must name every region and no other: got "
                f"{sorted(g.density)}, the model has {sorted(names)}")
        for name in g.surface:
            p = pnames.get(name)
            if p is None or p.bc is not BCType.FREE_TRACTION:
                raise ValueError(
                    f"gravity.surface names {name!r}, which is not a "
                    f"FREE_TRACTION boundary patch of the model "
                    f"(patches: {sorted(pnames)})")

    # -- gravity -----------------------------------------------------

    def _buoyancy_weight(self, region: Region, patch: Patch) -> float:
        """``(rho_R - rho_ref) g`` in GPa/km (``Gravity``): zero without
        gravity, on a fault, and wherever the far side is as dense."""
        g = self.gravity
        if g is None or patch.bc in FAULT_KINDS:
            return 0.0
        if patch.bc is BCType.INTERFACE:
            ref = g.weight(self.interface_regions(patch)[0].name)
        elif patch.name in g.surface:
            ref = 0.0
        else:
            return 0.0
        return g.weight(region.name) - ref

    def buoyancy(self, region: Region, patch: Patch):
        """``B`` (patch.n_nodes, 3, 3) with ``region`` seeing the traction
        ``t_p + B u_p`` on ``patch``, or None where there is no term.

        The ONE statement of the buoyancy condition (``Gravity``): the
        equations add ``-sigma G B`` to the u columns, the evaluators add
        ``B u_p`` to the single-layer density, and neither restates it.
        ``B = -w n (x) up`` per density node, ``n`` the stored element normal
        and ``up`` taken at the node.
        """
        w = self._buoyancy_weight(region, patch)
        if w == 0.0:
            return None
        normals, _ = patch.mesh.normals_and_areas()
        n = np.repeat(np.asarray(normals, float), n_nodes(patch.order), axis=0)
        return -w * np.einsum("ni,nj->nij", n,
                              self.gravity.up_at(patch.nodes()))

    def unstable_contrasts(self) -> list:
        """Names of the patches whose density contrast is somewhere heavy
        side UP. An elastic body holds that; a body that relaxes does not --
        it has a growing mode, which the Laplace route's decaying exponentials
        cannot represent, so a Laplace run refuses a model that lists any."""
        out = []
        if self.gravity is None:
            return out
        for r in self.regions:
            for p in r.patches:
                w = self._buoyancy_weight(r, p)
                if w == 0.0:
                    continue
                normals, _ = p.mesh.normals_and_areas()
                n_out = self._sigma[(r.name, p.name)] * np.asarray(normals, float)
                rise = np.einsum("ni,ni->n", n_out,
                                 self.gravity.up_at(p.mesh.centroids()))
                if np.any(w * rise < -defaults.BUOYANCY_UNSTABLE_TOL * abs(w)) \
                        and p.name not in out:
                    out.append(p.name)
        return out

    # -- convenience -------------------------------------------------

    def point_in_region(self, region, points, closure: bool = False):
        """Which of ``points`` lie strictly inside ``region``: ``(N,) bool``.

        The Gauss closure identity this module is built on, read as a
        classifier rather than as a check::

            sum_p sigma(R,p) * Omega_p(x) = 4*pi   iff x is inside R

        It is exact for any region graph, which is why the volume sampler uses
        it instead of an analytic test per geometry: a cylinder test would be a
        second statement of containment, right for one model and silent on the
        next. ``validate`` reads fault containment off this same call.

        A point ON a patch has an ambiguous solid angle and is classified
        either way; ``validate`` guards that case separately with an exact
        distance test, and the volume sampler has the near-field clearance
        arrays for it. ``closure=True`` also returns the raw sum, for an error
        message or a diagnostic.
        """
        from ..geometry import solid_angle_batch
        region = region if not isinstance(region, str) else \
            next(r for r in self.regions if r.name == region)
        pts = np.asarray(points, float).reshape(-1, 3)
        if region.is_full_space:
            total = np.full(pts.shape[0], 4.0 * np.pi)
            inside = np.ones(pts.shape[0], bool)
            return (inside, total) if closure else inside
        total = np.zeros(pts.shape[0])
        for p in region.patches:
            total += self._sigma[(region.name, p.name)] * \
                solid_angle_batch(p.mesh, pts)
        inside = np.abs(total - 4.0 * np.pi) <= _CLOSURE_TOL * 4.0 * np.pi
        return (inside, total) if closure else inside

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

    def with_materials(self, material_map: dict) -> "RegionModel":
        """This model with some regions' materials replaced, nothing else.

        ``material_map`` maps region name -> ElasticMaterial-compatible. A
        solution solved after ``rebuild_for_materials`` must be READ BACK
        through the result, because the evaluators take the moduli from the
        ``Region`` and the rebuild changes them only inside the operator:
        read through the base model, another state's densities are combined
        with the base coefficients, and the answer is wrong and plausible.

        Geometry is shared, not copied: the same Patch objects (an interface
        is identified by ``is``) and the same orientation table, which depend
        on the mesh alone. So nothing is re-validated, and this model is left
        untouched.
        """
        unknown = set(material_map) - {r.name for r in self.regions}
        if unknown:
            raise KeyError(f"unknown region(s) {sorted(unknown)} "
                           f"(regions: {[r.name for r in self.regions]})")
        new = RegionModel.__new__(RegionModel)
        new.regions = [replace(r, material=material_map[r.name])
                       if r.name in material_map else r for r in self.regions]
        new.gravity = self.gravity
        new._sigma = self._sigma
        return new
