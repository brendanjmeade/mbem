"""Patches, boundary conditions, collocation rows, and the solve.

THE FORMULATION
===============
One unknown field: a displacement discontinuity (DD) density ``q`` on every
boundary element, in ONE uniform medium, with the mollified Cortez kernels of
``clq``.  The whole model is a single representation,

    u(x) = sum_{boundary elements} D(x, y) q(y)  +  sum_{faults} D(x, y) s(y),

evaluated by :mod:`ddbem.assemble`.  ``D`` is the slip -> displacement kernel;
its traction counterpart is the hypersingular ``T``.  Nothing else is layered
on: no single layer, no fictitious force density.

Why this and not the indirect single layer: ``../fbem/FINDINGS.md``.  The force
element's unknown is a DENSITY carrying a genuine ``rho^(-1/3)`` edge
singularity on a polyhedron, and its free-traction rows stall at ``O(h^0.31)``
against the direct formulation's ``O(h^0.90)``; p-refinement and mesh grading
were both measured and neither rescues it.  A DD density is a DISPLACEMENT
JUMP, which is bounded.

SIDES, SIGNS, AND THE FREE TERM
-------------------------------
``q = Delta u = u(+nhat) - u(-nhat)`` (clq's convention; ``nhat`` comes from the
vertex order alone).  ``sigma_p = +1`` when patch ``p``'s stored normals point
OUT of the body, ``-1`` when they point in.  A mollified kernel evaluated ON its
own element returns the blob AVERAGE across the discontinuity, so the two
one-sided traces are

    u_interior(x_c) = [D q](x_c) - (sigma/2) q(x_c)
    u_exterior(x_c) = [D q](x_c) + (sigma/2) q(x_c)

and the traction has NO jump at all (the traction of a double layer is
continuous), so a traction row carries no analytic free term.  For a nodal
density ``q(x_c) = sum_k N_k(x_c) q_k``: the free term is the MATRIX of shape
function values at the collocation point, not a scalar 1/2.  That is where an
un-shrunk P1/P2 element and P0 differ, and where the equivalent fbem work was
most likely to be wrong.

THREE ROW TYPES
---------------
==============================  ==========================  =================
row                             equation                    analytic row sum
==============================  ==========================  =================
``TRACTION``                    ``t(x_c) = t_bar``          0
``INTERIOR_DISPLACEMENT``       ``u_int(x_c) = u_bar``      I/2
``EXTERIOR_NULL``               ``u_ext(x_c) = 0``          I/2
==============================  ==========================  =================

``TRACTION`` is the general free-/prescribed-traction row and is the default.
``EXTERIOR_NULL`` is the alternative row for a body whose ENTIRE boundary is
traction-free: zero boundary traction forces the whole exterior field to vanish,
so "the representation is null outside" is an equivalent, second-kind
statement of the same condition.  It is exactly msd's equation
(``mbem/model/equations.py``: ``1/2 I + sigma H``) rewritten in ``q``, with
``u_p = -sigma_p q_p`` and ``slip_msd = -q_fault`` -- which is what
``verify/verify_msd_parity.py`` uses for an entrywise P0 parity gate against
msd's dense backend.  It is REFUSED on any model with a non-zero prescribed
traction or a prescribed displacement, because then the exterior field is not
zero.

THE JUMP / FREE TERM CALIBRATION  (the highest-value item in this stage)
-----------------------------------------------------------------------
Write ``R`` for the 3x3 action of a row on the rigid-translation mode.  A rigid
translation ``c`` of the body is the DD density ``q = -sigma c`` on the closed
boundary (interior moves by ``c``, exterior does not move at all), and each row
has an exactly known answer on it: ``u_int = c``, ``u_ext = 0``, ``t = 0``.  So
the free term is fixed by a ROW-SUM IDENTITY,

    F = sigma_c * (R - target),     target = I (interior row) or 0 (others),
    R[i, j] = - sum_{p, s, k} sigma_p * A_raw[row_i, col(p, s, k, j)],

applied to the collocation element's own nodes weighted by ``N_k(x_c)``.

* ``jump="half"`` substitutes the ANALYTIC ``R`` (I/2 or 0) -- the classical
  collocation jump.
* ``jump="calibrated"`` uses the MEASURED ``R`` of the assembled matrix, so the
  discrete operator annihilates a rigid translation to machine precision and
  every quadrature/mollification error in the row is absorbed into the self
  block.  This is msd's ``jump="calibrated"`` (``C_q = -sum_p sigma rowsum
  H_qp``): identical at P0, and the derivation above is why it generalises --
  it is a row-sum identity and the partition of unity ``sum_k N_k = 1`` carries
  it to P1/P2 unchanged.  It also extends to the HYPERSINGULAR row, which msd
  has no equivalent of.

Calibration is a CLOSED-surface identity.  On an open or inconsistently
oriented boundary the rigid mode is not ``-sigma c`` and the identity is simply
false, so :class:`Model` checks closure (:func:`ddbem.shapes.is_closed`) and
refuses to calibrate without it.

WHAT IS NOT HERE
----------------
* **Material interfaces.**  ``BCType.INTERFACE`` is declared and RAISES.  A
  single DD density in ONE uniform medium cannot carry an interface: matching
  ``u`` and ``t`` across a material contrast needs a second density per region
  (the Somigliana single layer, whose density is the PHYSICAL boundary
  traction -- not the fictitious force density of the dropped force element,
  though it is the same clq kernel, ``clq.force_displacement``).  That is a
  two-region direct BIE, i.e. a second unknown field, a second representation
  and an orientation graph like msd's ``RegionModel``.  The design here leaves
  room for it -- patches are independent, oriented, per-patch ``eps`` and
  ``order``, rows are emitted per (field patch, source patch) block, and the
  free term is already a per-row 3x3 matrix -- but none of it is implemented,
  and pretending otherwise would be worse than the exception.
* **A continuous nodal layout.**  :func:`ddbem.layout.continuous` exists and is
  gated algebraically, but nothing here uses it.  Condensing columns leaves
  ``3 K N_tri`` collocation ROWS against ``3 n_global`` unknowns, so a
  continuous DD collocation system is over-determined: it needs either one
  collocation point per GLOBAL node -- whose element normal is ambiguous at a
  geometric edge, which is exactly where sharing a node is already a modelling
  error -- or a least-squares / Galerkin solve.  That is the decision where
  higher-order DD actually pays (continuous P1 is 1.52 slip unknowns per
  triangle against P0's 3, measured in ``verify/verify_layout.py``), and it is
  still open.
* **A jump term for a fault that reaches the boundary.**  A boundary
  collocation point within ~1 eps of a fault element sees the fault's blob
  average, not the one-sided value the boundary condition means.  The model
  WARNS (``defaults.FAULT_CLEARANCE_EPS``); it does not correct.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import defaults, shapes
from .assemble import displacement_matrix, stress_matrix, traction_matrix
from .layout import n_nodes
from .mesh import (collocation_shape_matrix, element_h, element_nodes,
                   element_normals, eps_auto, eps_report)


def _call_value(fn, points, normals):
    """Call a boundary-value callable with ``(points)`` or ``(points, normals)``."""
    import inspect
    try:
        n_pos = sum(1 for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    and p.default is p.empty)
    except (TypeError, ValueError):                 # builtins, C callables
        n_pos = 1
    return fn(points, normals) if n_pos >= 2 else fn(points)


class BCType(enum.Enum):
    """What is prescribed on a patch."""

    FREE_TRACTION = "free_traction"
    """Traction prescribed (default zero); the DD density is unknown."""

    PRESCRIBED_DISPLACEMENT = "prescribed_displacement"
    """Interior displacement prescribed; the DD density is unknown."""

    FAULT = "fault"
    """The DD density itself is prescribed: an interior source, no rows."""

    INTERFACE = "interface"
    """Declared, NOT implemented -- see the module docstring."""


class RowType(enum.Enum):
    TRACTION = "traction"
    INTERIOR_DISPLACEMENT = "interior_displacement"
    EXTERIOR_NULL = "exterior_null"


# analytic rigid-translation row sum R, and the row's exact answer on the mode
_ANALYTIC_R = {RowType.TRACTION: 0.0,
               RowType.INTERIOR_DISPLACEMENT: 0.5,
               RowType.EXTERIOR_NULL: 0.5}
_TARGET = {RowType.TRACTION: 0.0,
           RowType.INTERIOR_DISPLACEMENT: 1.0,
           RowType.EXTERIOR_NULL: 0.0}


# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Patch:
    """A named, oriented triangulated piece of the model.

    Parameters
    ----------
    name : str
    tri_verts : (N_tri, 3, 3)
        Vertex coordinates.  The unit normal is ``(v2-v1) x (v3-v1)``
        normalised -- clq's rule, taken from the vertex order alone.
    bc : BCType
    value : None | (3,) | (N_tri, 3) | (N_tri, K, 3) | (N_tri*K, 3) | callable
        The prescribed quantity.  For ``FREE_TRACTION`` it is the traction at
        the COLLOCATION points (default zero); for
        ``PRESCRIBED_DISPLACEMENT`` the interior displacement there; for
        ``FAULT`` the NODAL slip density.  A callable is evaluated at the
        relevant points and must return ``(M, 3)``.
    eps : None | float | (N_tri,) | "auto"
        Per-patch mollification width; ``None`` takes the model's.
    order : None | 0 | 1 | 2
        Per-patch element order; ``None`` takes the model's.  Mixing orders is
        allowed -- the unknowns are per element, so a P0 box with a P2 fault is
        a legal model.
    orientation : +1 | -1
        ``+1`` when the stored normals point OUT of the body.  Ignored for
        faults (a fault has no inside).
    """

    name: str
    tri_verts: np.ndarray
    bc: BCType
    value: object = None
    eps: object = None
    order: object = None
    orientation: int = 1

    def __post_init__(self):
        t = np.ascontiguousarray(np.asarray(self.tri_verts, float))
        if t.ndim == 2 and t.shape == (3, 3):
            t = t[None]
        if t.ndim != 3 or t.shape[1:] != (3, 3):
            raise ValueError(f"patch {self.name!r}: tri_verts must be "
                             f"(N_tri, 3, 3), got {t.shape}")
        self.tri_verts = t
        if int(self.orientation) not in (1, -1):
            raise ValueError(f"patch {self.name!r}: orientation must be +-1")
        self.orientation = int(self.orientation)

    # -- geometry -------------------------------------------------------

    @property
    def n_tri(self) -> int:
        return self.tri_verts.shape[0]

    @property
    def is_fault(self) -> bool:
        return self.bc is BCType.FAULT

    def normals(self) -> np.ndarray:
        return element_normals(self.tri_verts)

    def n_local(self, order: int) -> int:
        return n_nodes(order)

    def n_dof(self, order: int) -> int:
        return 3 * n_nodes(order) * self.n_tri

    def collocation_points(self, order: int, shrink: float) -> np.ndarray:
        """``(N_tri*K, 3)``; row ``K s + k`` matches columns ``3(K s + k) + j``."""
        return element_nodes(self.tri_verts, order, shrink).reshape(-1, 3)

    def collocation_normals(self, order: int) -> np.ndarray:
        """Element normal of each collocation point's own element."""
        return np.repeat(self.normals(), n_nodes(order), axis=0)

    # -- prescribed values ---------------------------------------------

    def values_at(self, order: int, points: np.ndarray,
                  normals: np.ndarray | None = None) -> np.ndarray:
        """Prescribed value as ``(N_tri*K, 3)`` at ``points`` (the row points).

        A callable taking ONE argument is called ``value(points)``; one taking
        two is called ``value(points, normals)`` with the collocation element's
        own normals -- which is what a traction boundary condition needs, and
        the reason a user should not have to rebuild the repeat-by-K themselves.
        """
        K = n_nodes(order)
        m = self.n_tri * K
        if self.value is None:
            return np.zeros((m, 3))
        if callable(self.value):
            if normals is None:
                normals = self.collocation_normals(order)
            v = np.asarray(_call_value(self.value, points, normals), float)
            if v.shape != (m, 3):
                raise ValueError(f"patch {self.name!r}: callable value returned "
                                 f"{v.shape}, expected {(m, 3)}")
            return np.ascontiguousarray(v)
        v = np.asarray(self.value, float)
        if v.shape == (3,):
            return np.broadcast_to(v, (m, 3)).copy()
        if v.shape == (self.n_tri, 3):
            return np.repeat(v, K, axis=0)
        if v.shape == (self.n_tri, K, 3):
            return v.reshape(m, 3).copy()
        if v.shape == (m, 3):
            return np.ascontiguousarray(v)
        raise ValueError(f"patch {self.name!r}: value has shape {v.shape}; "
                         f"expected (3,), {(self.n_tri, 3)}, "
                         f"{(self.n_tri, K, 3)} or {(m, 3)}")

    def nodal_density(self, order: int) -> np.ndarray:
        """Prescribed NODAL density, flattened ``(3 K N_tri,)`` in column order.

        The nodes here are the GEOMETRIC nodes (``shrink`` moves collocation
        points, never the basis), so a callable slip distribution is sampled at
        ``clq.nodes``.
        """
        pts = element_nodes(self.tri_verts, order, 0.0).reshape(-1, 3)
        return self.values_at(order, pts).reshape(-1)


# ---------------------------------------------------------------------------
# assembled system
# ---------------------------------------------------------------------------

@dataclass
class System:
    """The assembled collocation system and everything needed to read it."""

    model: "Model"
    A: np.ndarray
    b: np.ndarray
    rowsum: np.ndarray            # (n_coll, 3, 3) measured R, before free terms
    free_term: np.ndarray         # (n_coll, 3, 3) F actually applied
    col_slice: dict               # patch name -> slice into the unknowns
    row_slice: dict               # patch name -> slice into the rows
    row_type: dict                # patch name -> RowType
    n_dof: int

    def rigid_defect(self) -> float:
        """``max |R - R_analytic|``: how far the RAW rows are from the identity.

        This is the number the calibration removes.  It is a pure diagnostic of
        the discretisation (mollification leak, quadrature, element size) and is
        printed by :meth:`Model.report`.
        """
        ra = np.array([_ANALYTIC_R[self.row_type[n]]
                       for n, sl in self.row_slice.items()
                       for _ in range((sl.stop - sl.start) // 3)])
        eye = np.eye(3)[None] * ra[:, None, None]
        return float(np.max(np.abs(self.rowsum - eye)))


@dataclass
class Solution:
    """A solved model: the nodal DD density plus field evaluation."""

    model: "Model"
    system: System
    q: np.ndarray                 # (n_dof,) boundary density, column order
    residual: float
    cond: float
    n_constraints: int
    constraint_forces: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def patch_density(self, name: str) -> np.ndarray:
        """``(N_tri, K, 3)`` nodal density of one boundary patch."""
        p = self.model.patch(name)
        order = self.model.order_of(p)
        sl = self.system.col_slice[name]
        return self.q[sl].reshape(p.n_tri, n_nodes(order), 3)

    def displacement(self, points) -> np.ndarray:
        """Displacement ``(N, 3)`` from the full representation."""
        return self.model._evaluate(points, self.q, kind="displacement")

    def stress(self, points, subtract_eigenstress: bool = True) -> np.ndarray:
        """Elastic stress ``(N, 6)`` in Voigt order (xx, yy, zz, yz, xz, xy)."""
        return self.model._evaluate(points, self.q, kind="stress",
                                    subtract_eigenstress=subtract_eigenstress)

    def traction(self, points, normals) -> np.ndarray:
        """Traction ``(N, 3)`` on facets with the given normals."""
        return self.model._evaluate(points, self.q, kind="traction",
                                    normals=normals)

    def trace_displacement(self, name: str) -> np.ndarray:
        """Interior-side displacement ``(N_tri*K, 3)`` at a patch's collocation points.

        The boundary displacement -- which for a fault model IS the answer
        (surface displacement), and which the representation formula cannot give
        directly because a point on the surface returns the blob average of the
        two sides.  The one-sided value needs the free term, and this uses the
        SAME free term the rows use (calibrated or half), so it is consistent
        with the equation that was solved.

        Cost: one more displacement assembly over the whole model.  For a body
        whose entire boundary is traction-free the exterior field vanishes and
        the answer is just ``-sigma q`` with no assembly at all -- gated in
        ``verify/verify_solver.py``, and worth special-casing in any caller that
        knows it is in that case.
        """
        return self.model._trace_displacement(self.q, self.model.patch(name))


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class Model:
    """A DD collocation BEM model: patches + material + row conventions.

    ``patches`` are assembled in declaration order; the unknown vector is the
    concatenation of each non-fault patch's discontinuous nodal density, and the
    rows are the concatenation of each non-fault patch's collocation rows, in
    the same order, so ``A`` is square by construction.
    """

    def __init__(self, patches, mu, nu, *, order: int = 0, eps=None,
                 shrink=None, jump: str = defaults.JUMP,
                 neumann_row: str = defaults.NEUMANN_ROW,
                 far_field: str = defaults.FAR_FIELD,
                 validate: bool = True):
        self.patches = list(patches)
        self.mu = float(mu)
        self.nu = float(nu)
        self.order = int(order)
        self.eps = eps
        self._shrink = shrink
        if jump not in ("half", "calibrated"):
            raise ValueError(f"jump must be 'half' or 'calibrated', got {jump!r}")
        if neumann_row not in ("traction", "exterior"):
            raise ValueError(f"neumann_row must be 'traction' or 'exterior', "
                             f"got {neumann_row!r}")
        self.jump = jump
        self.neumann_row = neumann_row
        self.far_field = far_field
        self.boundary = [p for p in self.patches if not p.is_fault]
        self.faults = [p for p in self.patches if p.is_fault]
        self._closed = None
        self._volume = None
        if validate:
            self.validate()

    # -- lookups --------------------------------------------------------

    def patch(self, name: str) -> Patch:
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(f"no patch named {name!r}")

    def order_of(self, patch: Patch) -> int:
        return self.order if patch.order is None else int(patch.order)

    def shrink_of(self, patch: Patch) -> float:
        if self._shrink is not None:
            return float(self._shrink)
        return float(defaults.COLLOCATION_SHRINK_BY_ORDER[self.order_of(patch)])

    def eps_of(self, patch: Patch) -> np.ndarray:
        e = patch.eps if patch.eps is not None else self.eps
        if e is None:
            raise ValueError("no eps: set Model(eps=...) or Patch(eps=...)")
        if isinstance(e, dict):
            e = e[patch.name]
        if isinstance(e, str):
            if e != "auto":
                raise ValueError(f"eps must be a number, array or 'auto', got {e!r}")
            return eps_auto(patch.tri_verts)
        a = np.asarray(e, float)
        if a.ndim == 0:
            return np.full(patch.n_tri, float(a))
        if a.shape != (patch.n_tri,):
            raise ValueError(f"patch {patch.name!r}: eps has shape {a.shape}, "
                             f"expected () or {(patch.n_tri,)}")
        return a

    def row_type(self, patch: Patch) -> RowType:
        if patch.bc is BCType.PRESCRIBED_DISPLACEMENT:
            return RowType.INTERIOR_DISPLACEMENT
        if patch.bc is BCType.FREE_TRACTION:
            return (RowType.EXTERIOR_NULL if self.neumann_row == "exterior"
                    else RowType.TRACTION)
        raise ValueError(f"patch {patch.name!r}: no row type for {patch.bc}")

    # -- validation -----------------------------------------------------

    def validate(self) -> None:
        names = [p.name for p in self.patches]
        if len(set(names)) != len(names):
            raise ValueError(f"patch names must be unique, got {names}")
        if not self.boundary:
            raise ValueError("a model needs at least one non-fault patch")
        for p in self.patches:
            if p.bc is BCType.INTERFACE:
                raise NotImplementedError(
                    f"patch {p.name!r}: BCType.INTERFACE is declared but not "
                    "implemented.  A single DD density in ONE uniform medium "
                    "cannot carry a material contrast: matching u and t across "
                    "an interface needs a second density per region (the "
                    "Somigliana single layer) and a two-region direct BIE.  See "
                    "ddbem/model.py, 'WHAT IS NOT HERE'.")
            if self.order_of(p) not in (0, 1, 2):
                raise ValueError(f"patch {p.name!r}: order must be 0, 1 or 2")
            self.eps_of(p)
            a = 0.5 * np.linalg.norm(np.cross(p.tri_verts[:, 1] - p.tri_verts[:, 0],
                                              p.tri_verts[:, 2] - p.tri_verts[:, 0]),
                                     axis=1)
            L2 = element_h(p.tri_verts, "max_edge") ** 2
            if np.any(a <= defaults.DEGENERATE_AREA_REL * L2):
                bad = int(np.argmin(a / L2))
                raise ValueError(f"patch {p.name!r}: triangle {bad} is degenerate "
                                 f"(area/L^2 = {float(a[bad] / L2[bad]):.3e})")

        oriented = np.concatenate([p.tri_verts if p.orientation > 0
                                   else shapes.flip(p.tri_verts)
                                   for p in self.boundary])
        self._closed = shapes.is_closed(oriented)
        self._volume = shapes.enclosed_volume(oriented)
        if self._closed and self._volume <= 0.0:
            raise ValueError(
                "the boundary is a closed oriented surface but its enclosed "
                f"volume is {self._volume:.3e} <= 0: the normals point INTO the "
                "body.  Flip the triangles or set orientation=-1 on the "
                "patches -- ddbem needs sigma = +1 to mean 'outward'.")
        if self.jump == "calibrated" and not self._closed:
            raise ValueError(
                "jump='calibrated' needs a CLOSED, consistently oriented "
                "boundary: the calibration is the row-sum identity of a rigid "
                "translation, which is only a mode of a closed surface.  Pass "
                "jump='half', or close/re-orient the boundary.")
        if self.neumann_row == "exterior":
            for p in self.boundary:
                zero = p.value is None or (
                    not callable(p.value) and not np.any(np.asarray(p.value, float)))
                if p.bc is not BCType.FREE_TRACTION or not zero:
                    raise ValueError(
                        "neumann_row='exterior' states u_exterior = 0, which is "
                        "only true when EVERY boundary patch is traction-free "
                        f"with t_bar = 0; patch {p.name!r} is not.  Use "
                        "neumann_row='traction'.")
            if not self._closed:
                raise ValueError("neumann_row='exterior' needs a closed boundary")
        self._warn_fault_clearance()

    def _warn_fault_clearance(self) -> None:
        if not self.faults:
            return
        worst = np.inf
        n_close = 0
        for q in self.boundary:
            x = q.collocation_points(self.order_of(q), self.shrink_of(q))
            for f in self.faults:
                e = float(np.max(self.eps_of(f)))
                c = element_nodes(f.tri_verts, 0, 0.0).reshape(-1, 3)
                d = np.linalg.norm(x[:, None, :] - c[None, :, :], axis=2).min(axis=1)
                worst = min(worst, float(d.min() / e))
                n_close += int(np.sum(d < defaults.FAULT_CLEARANCE_EPS * e))
        if n_close:
            warnings.warn(
                f"{n_close} boundary collocation point(s) lie within "
                f"{defaults.FAULT_CLEARANCE_EPS:g} eps of a fault element "
                f"(min d/eps = {worst:.2f}): the fault's mollified field there "
                f"is the blob AVERAGE across the slip surface, not the "
                f"one-sided value the boundary condition means.  This is the "
                f"fault-outcrop case; ddbem warns, it does not correct it.")

    @property
    def closed(self) -> bool:
        if self._closed is None:
            self.validate()
        return bool(self._closed)

    # -- reporting ------------------------------------------------------

    def report(self) -> str:
        """Human-readable model summary: sizes, eps budget, row types."""
        lines = [f"Model: mu = {self.mu:g}, nu = {self.nu:g}, "
                 f"jump = {self.jump}, neumann_row = {self.neumann_row}",
                 f"  boundary closed = {self.closed}, "
                 f"enclosed volume = {self._volume:.6g}"]
        n = 0
        for p in self.patches:
            o = self.order_of(p)
            sh = self.shrink_of(p)
            e = self.eps_of(p)
            budget = eps_report(p.tri_verts, e, o, sh)
            kind = "fault " if p.is_fault else "bdy   "
            dof = p.n_dof(o)
            if not p.is_fault:
                n += dof
            lines.append(f"  {kind}{p.name:16s} N_tri={p.n_tri:6d}  P{o}  "
                         f"sigma={p.orientation:+d}  dof={dof:6d}  "
                         f"row={'-' if p.is_fault else self.row_type(p).value}")
            lines.append("      " + str(budget).replace("\n", "\n  "))
        lines.append(f"  unknowns = {n}")
        return "\n".join(lines)

    # -- assembly -------------------------------------------------------

    def _blocks(self, q: Patch, sources, kernel: RowType):
        """Influence of every source patch on patch ``q``'s collocation rows."""
        oq = self.order_of(q)
        x = q.collocation_points(oq, self.shrink_of(q))
        out = []
        for p in sources:
            op = self.order_of(p)
            if kernel is RowType.TRACTION:
                B = traction_matrix(x, q.collocation_normals(oq), p.tri_verts,
                                    self.eps_of(p), self.mu, self.nu, op,
                                    far_field=self.far_field)
            else:
                B = displacement_matrix(x, p.tri_verts, self.eps_of(p),
                                        self.mu, self.nu, op,
                                        far_field=self.far_field)
            out.append(B)
        return x, out

    def assemble(self) -> System:
        """Build ``A`` and ``b``: influence blocks, then the free term."""
        col_slice, row_slice, row_type = {}, {}, {}
        n = 0
        for p in self.boundary:
            d = p.n_dof(self.order_of(p))
            col_slice[p.name] = slice(n, n + d)
            row_slice[p.name] = slice(n, n + d)
            row_type[p.name] = self.row_type(p)
            n += d
        A = np.zeros((n, n))
        b = np.zeros(n)

        fault_q = [p.nodal_density(self.order_of(p)) for p in self.faults]

        for q in self.boundary:
            rt = row_type[q.name]
            r = row_slice[q.name]
            x, blocks = self._blocks(q, self.boundary, rt)
            for p, B in zip(self.boundary, blocks):
                A[r, col_slice[p.name]] = B
            # prescribed value on the right-hand side
            if rt is not RowType.EXTERIOR_NULL:
                b[r] = q.values_at(self.order_of(q), x,
                                   q.collocation_normals(self.order_of(q))
                                   ).reshape(-1)
            # interior sources
            if self.faults:
                _, fb = self._blocks(q, self.faults, rt)
                for B, s in zip(fb, fault_q):
                    b[r] -= B @ s

        rowsum = self._rowsum(A, col_slice)
        free = self._apply_free_term(A, rowsum, col_slice, row_slice, row_type)
        return System(model=self, A=A, b=b, rowsum=rowsum, free_term=free,
                      col_slice=col_slice, row_slice=row_slice,
                      row_type=row_type, n_dof=n)

    def _rowsum(self, A, col_slice) -> np.ndarray:
        """``R[c, i, j]``: each row's action on the rigid-translation mode.

        The mode is ``q(p, s, k, j) = -sigma_p c_j`` -- a translation of the
        interior by ``c`` with the exterior at rest -- so ``R`` is a
        sigma-signed column sum, taken component by component.
        """
        n = A.shape[1]
        sgn = np.empty(n)
        for p in self.boundary:
            sgn[col_slice[p.name]] = -float(p.orientation)
        R = np.empty((A.shape[0] // 3, 3, 3))
        for j in range(3):
            col = A[:, j::3] @ sgn[j::3]
            R[:, :, j] = col.reshape(-1, 3)
        return R

    def _apply_free_term(self, A, rowsum, col_slice, row_slice, row_type):
        """``F = sigma_c (R - target)`` spread over the collocation element's nodes."""
        F = np.empty_like(rowsum)
        for q in self.boundary:
            rt = row_type[q.name]
            o = self.order_of(q)
            K = n_nodes(o)
            r0 = row_slice[q.name].start
            c0 = col_slice[q.name].start
            n_coll = q.n_tri * K
            c_lo = r0 // 3
            R = (rowsum[c_lo:c_lo + n_coll] if self.jump == "calibrated"
                 else np.broadcast_to(np.eye(3) * _ANALYTIC_R[rt],
                                      (n_coll, 3, 3)))
            Fq = float(q.orientation) * (R - np.eye(3) * _TARGET[rt])
            F[c_lo:c_lo + n_coll] = Fq
            N = collocation_shape_matrix(o, self.shrink_of(q))     # (K, K)
            for c in range(n_coll):
                s, kc = divmod(c, K)
                rr = r0 + 3 * c
                for k in range(K):
                    w = N[kc, k]
                    if w:
                        cc = c0 + 3 * (K * s + k)
                        A[rr:rr + 3, cc:cc + 3] += w * Fq[c]
        return F

    # -- solve ----------------------------------------------------------

    def rigid_modes(self, system: System, which: str) -> np.ndarray:
        """Orthonormal ``(n_dof, m)`` basis of the rigid-motion DD modes.

        A rigid motion ``a + b x x`` of the interior, with the exterior at rest,
        is the density ``q = -sigma (a + b x x)`` at each node.  Translations are
        an EXACT null space of every calibrated row; rotations are a null space
        of the CONTINUUM operator (the interior traction-free problem is unique
        only up to a rigid motion) and are near-null discretely, which is why
        ``"rigid"`` (6 modes) is the default and msd's 3-translation deflation
        leaves three tiny singular values behind.
        """
        if which == "none":
            return np.zeros((system.n_dof, 0))
        cols = []
        for p in self.boundary:
            o = self.order_of(p)
            nodes = element_nodes(p.tri_verts, o, 0.0).reshape(-1, 3)
            cols.append((nodes, -float(p.orientation), system.col_slice[p.name]))
        m = 3 if which == "translations" else 6
        if which not in ("translations", "rigid"):
            raise ValueError(f"constrain must be 'none', 'translations' or "
                             f"'rigid', got {which!r}")
        Z = np.zeros((system.n_dof, m))
        for nodes, sg, sl in cols:
            blk = np.zeros((nodes.shape[0], 3, m))
            for k in range(3):
                blk[:, k, k] = sg
            if m == 6:
                for k in range(3):
                    e = np.zeros(3)
                    e[k] = 1.0
                    blk[:, :, 3 + k] = sg * np.cross(e[None, :], nodes)
            Z[sl] = blk.reshape(-1, m)
        Zc = np.linalg.qr(Z)[0] if Z.shape[1] else Z
        return Zc

    def solve(self, system: System | None = None, *,
              constrain: str | None = None) -> Solution:
        """Assemble (if needed) and solve.

        ``constrain`` borders the system with the rigid-motion modes,
        ``[[A, Z], [Z^T, 0]]``, and returns the Z-orthogonal representative.
        Defaults to :data:`ddbem.defaults.CONSTRAIN` when no row can see a rigid
        motion (every boundary patch traction-free), and to ``"none"``
        otherwise -- a prescribed displacement anchors the body already.
        """
        from scipy.linalg import lu_factor, lu_solve, get_lapack_funcs

        if system is None:
            system = self.assemble()
        if constrain is None:
            anchored = any(p.bc is BCType.PRESCRIBED_DISPLACEMENT
                           for p in self.boundary)
            constrain = "none" if anchored else defaults.CONSTRAIN
        Z = self.rigid_modes(system, constrain)
        n, m = system.n_dof, Z.shape[1]
        if m:
            K = np.zeros((n + m, n + m))
            K[:n, :n] = system.A
            K[:n, n:] = Z
            K[n:, :n] = Z.T
            rhs = np.concatenate([system.b, np.zeros(m)])
        else:
            K, rhs = system.A, system.b
        anorm = float(np.linalg.norm(K, 1))
        lu = lu_factor(K)
        gecon = get_lapack_funcs(("gecon",), (lu[0],))[0]
        rcond, info = gecon(lu[0], anorm, norm="1")
        cond = (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
        x = lu_solve(lu, rhs)
        q = x[:n]
        res = float(np.linalg.norm(system.A @ q + (Z @ x[n:] if m else 0.0)
                                   - system.b))
        scale = float(np.linalg.norm(system.b)) or 1.0
        return Solution(model=self, system=system, q=q, residual=res / scale,
                        cond=cond, n_constraints=m,
                        constraint_forces=x[n:] if m else np.zeros(0))

    def _trace_displacement(self, q, patch: Patch) -> np.ndarray:
        """Interior trace at ``patch``'s collocation points (see Solution)."""
        o = self.order_of(patch)
        K = n_nodes(o)
        x = patch.collocation_points(o, self.shrink_of(patch))
        cols, n = {}, 0
        for p in self.boundary:
            d = p.n_dof(self.order_of(p))
            cols[p.name] = slice(n, n + d)
            n += d
        u = np.zeros(3 * x.shape[0])
        R = np.zeros((x.shape[0], 3, 3))
        for p in self.boundary:
            B = displacement_matrix(x, p.tri_verts, self.eps_of(p), self.mu,
                                    self.nu, self.order_of(p),
                                    far_field=self.far_field)
            u += B @ q[cols[p.name]]
            for j in range(3):
                R[:, :, j] -= float(p.orientation) * (
                    B[:, j::3].sum(axis=1).reshape(-1, 3))
        for f in self.faults:
            B = displacement_matrix(x, f.tri_verts, self.eps_of(f), self.mu,
                                    self.nu, self.order_of(f),
                                    far_field=self.far_field)
            u += B @ f.nodal_density(self.order_of(f))
        if self.jump == "calibrated":
            F = float(patch.orientation) * (R - np.eye(3))
        else:
            F = np.broadcast_to(-0.5 * float(patch.orientation) * np.eye(3),
                                (x.shape[0], 3, 3))
        N = collocation_shape_matrix(o, self.shrink_of(patch))
        qp = q[cols[patch.name]].reshape(patch.n_tri, K, 3)
        at_c = np.einsum("ck,tkj->tcj", N, qp).reshape(-1, 3)
        return u.reshape(-1, 3) + np.einsum("cij,cj->ci", F, at_c)

    # -- field evaluation -----------------------------------------------

    def _evaluate(self, points, q, kind: str, normals=None,
                  subtract_eigenstress: bool = True):
        """Representation formula at arbitrary points (boundary + faults)."""
        x = np.atleast_2d(np.asarray(points, float))
        rows = {"displacement": 3, "traction": 3, "stress": 6}[kind]
        out = np.zeros((x.shape[0] * rows,))
        system_cols = {}
        n = 0
        for p in self.boundary:
            d = p.n_dof(self.order_of(p))
            system_cols[p.name] = slice(n, n + d)
            n += d
        if q.shape[0] != n:
            raise ValueError(f"density has {q.shape[0]} entries, model has {n}")
        if kind == "traction":
            nf = np.asarray(normals, float)
            if nf.ndim == 1:
                nf = np.broadcast_to(nf, x.shape)
        for p in self.patches:
            o = self.order_of(p)
            dens = (p.nodal_density(o) if p.is_fault else q[system_cols[p.name]])
            if not np.any(dens):
                continue
            if kind == "displacement":
                B = displacement_matrix(x, p.tri_verts, self.eps_of(p), self.mu,
                                        self.nu, o, far_field=self.far_field)
            elif kind == "traction":
                B = traction_matrix(x, nf, p.tri_verts, self.eps_of(p), self.mu,
                                    self.nu, o, far_field=self.far_field,
                                    subtract_eigenstress=subtract_eigenstress)
            else:
                B = stress_matrix(x, p.tri_verts, self.eps_of(p), self.mu,
                                  self.nu, o, far_field=self.far_field,
                                  subtract_eigenstress=subtract_eigenstress)
            out += B @ dens
        return out.reshape(x.shape[0], rows)
