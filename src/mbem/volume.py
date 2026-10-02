"""Sample a solved model on a 3-D grid: the field everywhere, not just on it.

The solver reports its answer on the boundary. This evaluates the SAME
representation formula at grid points in the interior, so the volumetric
displacement and elastic stress can be looked at. Nothing here is new physics:
``evaluate_displacement`` / ``evaluate_stress`` already take arbitrary points,
and this module is the grid, the region bookkeeping, the near-field metric and
the chunk loop around them.

Three things it is careful about.

**Region, not geometry.** Every point is classified by ``RegionModel.
point_in_region`` -- the Gauss closure identity the model already validates
faults with -- and evaluated with ITS OWN region's material and patch set.
Asking for the host field at a point inside the inclusion returns a plausible
wrong answer and nothing in the evaluator would object. Because the classifier
reads the actual boundary meshes, TOPOGRAPHY IS FREE: a point between the flat
z = 0 plane and a hill above it is inside the warped top patch and is found to
be inside. An explicit ``z > h(x, y)`` test would be a second statement of the
surface, right until someone changes the bump.

**``warn_near=False``, deliberately, and the clearance computed here instead.**
The evaluator's near-boundary warning runs a k=32 exact point-to-triangle
distance whose temporaries are (N, 32, 3) -- ~8-12 KB per point against ~400 B
for the evaluation itself, so on a 500k-point grid the warning costs 4-6 GB and
dwarfs the thing it is warning about. The same information is written as data
instead: ``clearance_h`` and ``clearance_eps``, below.

**Serial chunks.** All five contraction kernels are ``parallel=True`` and
already ``prange`` over the observation points, which is the axis a driver
would want to split. Rule 9 forbids calling them from Python threads and no
``*_contract_serial`` variants exist, so the chunk loop is serial on the main
thread and the parallelism comes from inside.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from mbem import defaults
from mbem.evaluate import evaluate_displacement, evaluate_stress
from mbem.geometry import distance_to_mesh
from mbem.kernels import basis as kb

# Tensor components written out, in the order a viewer labels them.
SYM6 = (("xx", 0, 0), ("yy", 1, 1), ("zz", 2, 2),
        ("xy", 0, 1), ("yz", 1, 2), ("xz", 0, 2))


@dataclasses.dataclass(frozen=True)
class Grid:
    """A regular grid, ordered x-fastest so it is ImageData without a copy."""
    origin: tuple
    spacing: tuple
    dims: tuple                      # (nx, ny, nz) POINT counts
    points: np.ndarray               # (N, 3), C order over (nz, ny, nx)

    @property
    def shape(self) -> tuple:
        nx, ny, nz = self.dims
        return (nz, ny, nx)

    def fold(self, a: np.ndarray) -> np.ndarray:
        """``(N, ...)`` -> ``(nz, ny, nx, ...)``."""
        return a.reshape(self.shape + a.shape[1:])


def model_bounds(model) -> tuple:
    """Bounding box of every boundary patch, as ``(lo, hi)`` 3-vectors.

    Read off the meshes rather than from a case's constants, so the grid
    follows the model it is sampling -- including a warped top, whose vertices
    reach above z = 0.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for r in model.regions:
        for p in r.patches:
            v = np.asarray(p.mesh.vertices, float)
            lo = np.minimum(lo, v.min(axis=0))
            hi = np.maximum(hi, v.max(axis=0))
    return lo, hi


def make_grid(model, spacing: float, pad: float = 0.0) -> Grid:
    """A grid covering the model, at ``spacing`` (km), snapped to whole cells."""
    lo, hi = model_bounds(model)
    lo, hi = lo - pad, hi + pad
    axes, origin, dims = [], [], []
    for d in range(3):
        n = int(np.floor((hi[d] - lo[d]) / spacing)) + 1
        axes.append(lo[d] + spacing * np.arange(n))
        origin.append(float(lo[d]))
        dims.append(n)
    Z, Y, X = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return Grid(origin=tuple(origin), spacing=(spacing,) * 3,
                dims=tuple(dims), points=points)


def classify(model, grid: Grid) -> np.ndarray:
    """``(N,) int8``: 0 outside every region, else 1-based region index.

    A point inside two regions cannot happen for a valid model (the regions
    partition the body), but a point ON an interface is ambiguous by solid
    angle; first match wins and ``clearance_h`` is what flags it.
    """
    code = np.zeros(grid.points.shape[0], np.int8)
    for i, r in enumerate(model.regions, start=1):
        hit = model.point_in_region(r, grid.points) & (code == 0)
        code[hit] = i
    return code


def clearance(model, grid: Grid, eps) -> tuple:
    """``(clearance_h, clearance_eps)``, each ``(N,)``, min over all patches.

    Two metrics because the repo states the near-field condition two ways and
    each answers a different question. ``clearance_h`` is distance over element
    size, exactly what ``evaluate._warn_near_boundary`` computes and compares
    against ``NEAR_BOUNDARY_H_RATIO``; ``clearance_eps`` is distance over that
    element's own eps, which is what rule 5 names as the thing that governs the
    error. They are not interchangeable: ``eps="auto"`` is 0.1 h per element on
    a boundary patch but ONE value of 0.07 min(h) on a fault, so the two
    disagree by more than a constant factor wherever a fault is involved.

    Faults are included here even though the evaluator's warning skips them --
    on-fault evaluation is legitimate, but a viewer should still be able to see
    where the fault's own mollification reaches.
    """
    n = grid.points.shape[0]
    c_h = np.full(n, np.inf)
    c_eps = np.full(n, np.inf)
    for r in model.regions:
        for p in list(r.patches) + list(r.faults):
            d, idx = distance_to_mesh(grid.points, p.mesh)
            h = kb.element_sizes(p.mesh)[idx]
            e = np.broadcast_to(np.atleast_1d(kb.resolve_patch_eps(eps, p)),
                                (p.mesh.n_triangles,))[idx]
            c_h = np.minimum(c_h, d / h)
            c_eps = np.minimum(c_eps, d / e)
    return c_h, c_eps


def _strain(sig: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Hooke inverted from ``(mu, lam)`` only -- never via ``1/(1-2nu)``.

    ``sig = 2 mu e + lam tr(e) I``, so ``tr(sig) = (2 mu + 3 lam) tr(e)``.
    """
    tr_e = np.trace(sig, axis1=1, axis2=2) / (2.0 * mu + 3.0 * lam)
    e = sig.copy()
    e[:, 0, 0] -= lam * tr_e
    e[:, 1, 1] -= lam * tr_e
    e[:, 2, 2] -= lam * tr_e
    return e / (2.0 * mu)


def _derived(u: np.ndarray, sig: np.ndarray, e: np.ndarray) -> dict:
    """Scalars a volume render can actually be coloured by."""
    tr = np.trace(sig, axis1=1, axis2=2)
    dev = sig - (tr / 3.0)[:, None, None] * np.eye(3)
    w = np.linalg.eigvalsh(sig)                  # ascending
    return {
        "u_mag": np.linalg.norm(u, axis=1),
        "von_mises": np.sqrt(1.5 * np.einsum("nij,nij->n", dev, dev)),
        "mean_stress": tr / 3.0,
        "max_shear": 0.5 * (w[:, 2] - w[:, 0]),
        "dilatation": np.trace(e, axis1=1, axis2=2),
    }


def sample(model, solution: dict, eps, grid: Grid, code: np.ndarray,
           chunk: int = None, eigenstress: bool = True,
           progress=None) -> dict:
    """Evaluate one solved state on ``grid``. Returns ``(N, ...)`` arrays.

    ``code`` comes from :func:`classify` and decides which region's material
    and patch set each point is evaluated with. Points outside every region are
    NaN: they are not in the body, and 0 would be a value.

    With ``eigenstress`` the anelastic part ``C:eps*`` is written too, so the
    mollification is visible rather than silently removed. It is nearly free
    because ``evaluate_stress(parts=True)`` returns it from the same pass:
    measured on this model, the elastic stress is 2.13 ms/point and the
    eigenstress term 0.38 ms of that, so recovering it as total - elastic
    would have cost a second 1.75 ms/point ``dd`` evaluation to learn nothing
    new.
    """
    chunk = chunk or defaults.VOLUME_CHUNK
    n = grid.points.shape[0]
    out = {
        "u": np.full((n, 3), np.nan),
        "sigma": np.full((n, 3, 3), np.nan),
        "strain": np.full((n, 3, 3), np.nan),
    }
    if eigenstress:
        out["eigenstress"] = np.full((n, 3, 3), np.nan)

    for i, r in enumerate(model.regions, start=1):
        where = np.flatnonzero(code == i)
        if where.size == 0:
            continue
        mu, lam = r.material.mu, r.material.lam
        for s in range(0, where.size, chunk):
            sel = where[s:s + chunk]
            pts = grid.points[sel]
            out["u"][sel] = evaluate_displacement(
                model, r, solution, pts, eps, warn_near=False)
            got = evaluate_stress(model, r, solution, pts, eps,
                                  subtract_anelastic=True, warn_near=False,
                                  parts=eigenstress)
            sig = got[0] if eigenstress else got
            out["sigma"][sel] = sig
            out["strain"][sel] = _strain(sig, mu, lam)
            if eigenstress:
                out["eigenstress"][sel] = got[1]
            if progress is not None:
                progress(r.name, min(s + chunk, where.size), where.size)

    scalars = {k: np.full(n, np.nan) for k in
               ("u_mag", "von_mises", "mean_stress", "max_shear", "dilatation")}
    inside = np.flatnonzero(code > 0)
    if inside.size:
        d = _derived(out["u"][inside], out["sigma"][inside],
                     out["strain"][inside])
        for k, v in d.items():
            scalars[k][inside] = v
    out.update(scalars)
    return out


PRIMARY = ("u", "sigma", "strain", "eigenstress")


def difference(fa: dict, fb: dict, inside: np.ndarray) -> dict:
    """``fa - fb`` where both are defined, with the SCALARS recomputed.

    Differencing the derived scalars instead would be wrong, not merely
    imprecise: the von Mises of a stress perturbation is not the difference of
    two von Mises values, and a viewer handed the latter shows negative "von
    Mises", which cannot exist. So only the tensors and the vector are
    subtracted, and the scalars are rebuilt from the result.
    """
    out = {}
    for k in PRIMARY:
        if k in fa and k in fb:
            d = fa[k] - fb[k]
            d[~inside] = np.nan
            out[k] = d
    n = inside.size
    scalars = {k: np.full(n, np.nan) for k in
               ("u_mag", "von_mises", "mean_stress", "max_shear", "dilatation")}
    sel = np.flatnonzero(inside)
    if sel.size:
        d = _derived(out["u"][sel], out["sigma"][sel], out["strain"][sel])
        for k, v in d.items():
            scalars[k][sel] = v
    out.update(scalars)
    return out


def as_vti_arrays(grid: Grid, fields: dict, code: np.ndarray,
                  c_h: np.ndarray, c_eps: np.ndarray) -> dict:
    """Flatten to the name -> ``(nz, ny, nx, ...)`` dict the writer wants.

    Tensors are split into the six independent components under their usual
    names rather than written as 9-vectors, because that is what a viewer can
    put on a colour bar.
    """
    arrays = {}
    for name, a in fields.items():
        if a.ndim == 3:                                  # (N,3,3) tensor
            for comp, i, j in SYM6:
                arrays[f"{name}_{comp}"] = grid.fold(a[:, i, j])
        elif a.ndim == 2:                                # (N,3) vector
            arrays[name] = grid.fold(a)
        else:
            arrays[name] = grid.fold(a)
    arrays["region"] = grid.fold(code.astype(float))
    arrays["clearance_h"] = grid.fold(c_h)
    arrays["clearance_eps"] = grid.fold(c_eps)
    return arrays


def timed(label: str, fn, log=None):
    """Run ``fn``, return ``(value, seconds)``, optionally printing."""
    t0 = time.perf_counter()
    v = fn()
    dt = time.perf_counter() - t0
    if log is not None:
        log(f"  {label:22s} {dt:8.2f} s")
    return v, dt
