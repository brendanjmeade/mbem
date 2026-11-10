"""The penny-shaped crack: the reference case of the traction-BC fault.

A circular crack of radius ``a`` in the plane z = 0 of an infinite body (a
full-space region: no boundary patch), stored normal +z, shear along +x. Two
exact solutions anchor it (mu, nu from (mu, lam), rule 8):

* Eshelby's shear crack: a uniform shear traction drop ``dtau`` gives the
  elliptical slip delta(r) = delta0 sqrt(1 - r^2 / a^2) with
  dtau = pi (2 - nu) mu delta0 / (8 (1 - nu) a); moment 2 pi delta0 a^2 / 3.
* Sneddon's pressurised crack: a uniform pressure ``p`` gives the opening
  4 (1 - nu) p / (pi mu) sqrt(a^2 - r^2).

SIGN. ``Patch.value`` on a fault is b = u(+n) - u(-n) (rule 3). With n = +z
the Eshelby slip b = +delta(r) x_hat means the +z face moves toward +x: that
relaxes a pre-stress sigma_xz = +tau0, whose traction on the +z face
(+tau0 x_hat) drags the top toward +x. The elastic traction CHANGE the slip
produces on the crack is therefore -dtau x_hat, and that is what a
FAULT_TRACTION fault's ``value`` is: a drop of ``dtau`` along s is
``value = -dtau s``, written once, in ``build_model``. The gate
(``tests/gates/mbem/verify_penny_crack.py``) pins both directions against
literals, never against this file: solving with value = -dtau x_hat must give
b_x > 0, and imposing b = +delta x_hat must read t_x < 0.

THE MESH. N_rim = round(2 pi a / h) vertices EXACTLY on the circle, joined by
segments and filled by Shewchuk's Triangle with a 30-degree quality bound and
a maximum area; the rim stays unsplit (YY), so every boundary node of the
shared-node table is a rim vertex or a chord midpoint (h^2 / 8a inside the
circle, below eps at every a/h used). ``PENNY_AREA_FACTOR`` calibrates
Triangle's maximum area so the mean edge comes out at h: Triangle's ``a`` is
a ceiling, and the equilateral area 0.433 h^2 gives a mean edge of 0.83 h
and far more elements than a/h suggests.
"""
from __future__ import annotations

import numpy as np

from mbem.model import BCType, Patch, Region, RegionModel
from mollified_bem import TriMesh

#: Triangle's maximum-area constraint as a multiple of h^2. Measured (cdt,
#: flags pq30a..YYQ): 0.65 gives a mean edge of 1.00-1.01 h with 487 / 1942
#: elements at a/h 8 / 16 and a minimum angle of 30 degrees; the equilateral
#: 0.433 gives 0.83 h. A mesher calibration, so it lives here.
PENNY_AREA_FACTOR = 0.65
MIN_ANGLE_DEG = 30
#: The hex lattice keeps its interior points this far inside the rim (in h):
#: closer, Triangle's fill between the lattice and the ring collapses.
LATTICE_RIM_CLEARANCE_OVER_H = 0.5


def _ring(a: float, h: float, phase: float) -> np.ndarray:
    n = max(int(round(2.0 * np.pi * a / h)), 12)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(phase)
    return np.column_stack([a * np.cos(theta), a * np.sin(theta)])


def _to_mesh(verts2: np.ndarray, tris: np.ndarray) -> TriMesh:
    """Lift to z = 0 and wind every triangle so its normal is +z."""
    verts = np.column_stack([verts2, np.zeros(verts2.shape[0])])
    tris = np.asarray(tris, np.int64)
    p = verts2[tris]
    orient = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
              - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    flip = orient < 0.0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return TriMesh(vertices=np.ascontiguousarray(verts),
                   triangles=np.ascontiguousarray(tris))


def penny_mesh(a: float = 1.0, h: float = 0.125, kind: str = "cdt", *,
               jitter: float = 0.25, seed: int = 0, phase: float = 0.0) -> TriMesh:
    """A disc of radius ``a`` in z = 0, normal +z, rim vertices on the circle
    at spacing ``h``.

    ``kind``: ``"cdt"`` the quality fill (the mesher a study would use);
    ``"lattice"`` a hexagonal lattice of spacing h inside the rim ring
    (``LATTICE_RIM_CLEARANCE_OVER_H``), Delaunay-joined, no Steiner points;
    ``"jitter"`` the cdt with every interior vertex displaced by
    uniform(-jitter h, +jitter h) in the plane from a seeded generator, the
    rim untouched -- the mesh-independence control.
    """
    import triangle as tr

    ring = _ring(a, h, phase)
    n = ring.shape[0]
    seg = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])
    if kind in ("cdt", "jitter"):
        area = PENNY_AREA_FACTOR * h * h
        out = tr.triangulate({"vertices": ring, "segments": seg},
                             f"pq{MIN_ANGLE_DEG}a{area:.12f}YYQ")
    elif kind == "lattice":
        dy = h * np.sqrt(3.0) / 2.0
        rows = int(np.ceil(a / dy)) + 1
        cols = int(np.ceil(a / h)) + 1
        pts = []
        for j in range(-rows, rows + 1):
            for i in range(-cols, cols + 1):
                x = i * h + (0.5 * h if j % 2 else 0.0)
                y = j * dy
                if np.hypot(x, y) < a - LATTICE_RIM_CLEARANCE_OVER_H * h:
                    pts.append((x, y))
        verts = np.vstack([ring, np.array(pts)])
        out = tr.triangulate({"vertices": verts, "segments": seg}, "pYYQ")
    else:
        raise ValueError(f"kind must be 'cdt', 'lattice' or 'jitter', got {kind!r}")
    verts2 = np.asarray(out["vertices"], float)
    tris = np.asarray(out["triangles"], np.int64)
    if kind == "jitter":
        rng = np.random.default_rng(seed)
        interior = np.arange(verts2.shape[0]) >= n          # the ring comes first
        verts2 = verts2.copy()
        verts2[interior] += rng.uniform(-jitter * h, jitter * h, (int(interior.sum()), 2))
    return _to_mesh(verts2, tris)


def eshelby_slip(a: float, delta0: float):
    """The Eshelby slip as a callable of points: +x, delta0 sqrt(1 - r^2/a^2),
    exactly zero at and beyond the rim."""
    def slip(points):
        x = np.asarray(points, float)
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        out = np.zeros((x.shape[0], 3))
        out[:, 0] = delta0 * np.sqrt(np.clip(1.0 - r2 / (a * a), 0.0, None))
        return out
    return slip


def _nu(mu: float, lam: float) -> float:
    return lam / (2.0 * (lam + mu))


def stress_drop(mu: float, lam: float, delta0: float, a: float) -> float:
    """Eshelby: the uniform shear drop that produces the peak slip delta0."""
    nu = _nu(mu, lam)
    return np.pi * (2.0 - nu) * mu * delta0 / (8.0 * (1.0 - nu) * a)


def eshelby_moment(delta0: float, a: float) -> float:
    """Integral of the Eshelby slip over the crack, 2 pi delta0 a^2 / 3."""
    return 2.0 * np.pi * delta0 * a * a / 3.0


def sneddon_opening(a: float, p: float, mu: float, lam: float):
    """Sneddon: the opening of a penny crack under uniform pressure p, as a
    callable of points (+z component), 4 (1 - nu) p / (pi mu) sqrt(a^2 - r^2)."""
    nu = _nu(mu, lam)
    amp = 4.0 * (1.0 - nu) * p / (np.pi * mu)

    def opening(points):
        x = np.asarray(points, float)
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        out = np.zeros((x.shape[0], 3))
        out[:, 2] = amp * np.sqrt(np.clip(a * a - r2, 0.0, None))
        return out
    return opening


def _layout(order: int) -> str:
    return "continuous" if order else "discontinuous"


def _space(material, fault: Patch) -> RegionModel:
    return RegionModel([Region("space", material, [], probe_point=np.zeros(3),
                               faults=[fault])])


def build_model(mesh: TriMesh, material, *, drop: float = 0.0,
                pressure: float = 0.0, order: int = 2,
                allow_opening: bool = False) -> RegionModel:
    """The crack as a FAULT_TRACTION fault in the full space: a shear traction
    drop ``drop`` (> 0) along +x and/or a pressure ``pressure`` (> 0, needs
    ``allow_opening``). The value is the traction CHANGE the slip must
    produce, ``-drop x_hat - pressure z_hat`` (module docstring)."""
    if pressure and not allow_opening:
        raise ValueError("a pressure drives the opening: pass allow_opening=True")
    value = np.array([-float(drop), 0.0, -float(pressure)])
    fault = Patch("penny", mesh, BCType.FAULT_TRACTION, value=value, order=order,
                  layout=_layout(order), allow_opening=allow_opening)
    return _space(material, fault)


def build_slip_model(mesh: TriMesh, material, *, delta0: float = 1e-3,
                     order: int = 2, a: float = 1.0) -> RegionModel:
    """The crack with the Eshelby slip PRESCRIBED (bc FAULT): the readout
    fixture, whose on-fault elastic traction must come back as -dtau x_hat."""
    fault = Patch("penny", mesh, BCType.FAULT, value=eshelby_slip(a, delta0),
                  order=order, layout=_layout(order))
    return _space(material, fault)
