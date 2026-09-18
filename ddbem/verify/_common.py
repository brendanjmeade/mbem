"""Shared helpers for the ddbem verification scripts.

Every ``verify_*.py`` is run from the ddbem root as ``python verify/<name>.py``
(any interpreter with numpy; the parity gates also need numba, because they
call msd's ``mbem`` kernels) and ends with exactly one line starting with
``PASS`` or ``FAIL``.  ``run_all.py`` launches each gate with ``sys.executable``
and reads ``line.split(":")[0]`` -- the colon matters.

msd oracle modules are loaded BY FILE PATH (clq's convention) so nothing here
depends on msd being importable as a package, except ``mbem``, which is a real
package and is imported by putting ``../msd`` on ``sys.path``.  Nothing in this
directory writes to msd, clq, moss, medt_paper or fbem.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOSS_ORG = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MU = 1.0
NU_SWEEP = (0.25, 0.30, 0.45)      # the lam/mu pairing bug is invisible at 1/4


# ---------------------------------------------------------------------------
# oracle loading
# ---------------------------------------------------------------------------

def load_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def msd_analytical():
    """Frozen scalar oracles: analytical_dd_displacement / analytical_stress_kernel."""
    return load_module("_msd_analytical_kernels",
                       MOSS_ORG / "msd" / "mollified_kernel" / "analytical_kernels.py")


def msd_batch():
    """Frozen P0 assembly: assemble_T_matrix_batch / dd_displacement_batch."""
    return load_module("_msd_analytical_batch",
                       MOSS_ORG / "msd" / "mollified_kernel" / "analytical_batch.py")


def msd_points():
    """Frozen point kernels: kelvin_d2G / dd_stress_kernel."""
    return load_module("_msd_point_kernels",
                       MOSS_ORG / "msd" / "mollified_kernel"
                       / "mollified_elastic_kernels.py")


def msd_anelastic():
    """The POINT/marginal eigenstress approximation (NOT what ddbem uses)."""
    return load_module("_msd_anelastic", MOSS_ORG / "msd" / "anelastic.py")


def msd_mbem():
    """The mbem numba kernel module and its material coefficients (numba import)."""
    p = str(MOSS_ORG / "msd")
    if p not in sys.path:
        sys.path.insert(0, p)
    from mbem.kernels import tri_kernels as tk       # noqa: E402
    from mbem.kernels.basis import t_coeffs          # noqa: E402
    return tk, t_coeffs


# ---------------------------------------------------------------------------
# error measures
# ---------------------------------------------------------------------------

def relmax(a, b) -> float:
    """max|a - b| / max|b| over the whole array (absolute if b is ~0)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ref = float(np.max(np.abs(b)))
    diff = float(np.max(np.abs(a - b)))
    return diff / ref if ref > 1e-300 else diff


def block_relmax(A, B, n_field, n_src, rows_per_field, cols_per_src) -> float:
    """Worst BLOCK-relative difference between two influence matrices.

    An influence matrix is a grid of (field element, source element) blocks
    whose entries pass through zero individually -- a per-ENTRY relative
    measure is meaningless there -- so each block is normalised by its own
    largest magnitude.  A block whose own scale is below 1e-300 is compared
    absolutely.  Returns the worst ratio over all blocks.
    """
    A = np.asarray(A, float).reshape(n_field, rows_per_field, n_src, cols_per_src)
    B = np.asarray(B, float).reshape(n_field, rows_per_field, n_src, cols_per_src)
    num = np.abs(A - B).max(axis=(1, 3))
    den = np.abs(B).max(axis=(1, 3))
    safe = np.where(den > 1e-300, den, 1.0)
    return float(np.max(num / safe))


# ---------------------------------------------------------------------------
# independent geometry / shape functions (NOT clq's)
# ---------------------------------------------------------------------------

def barycentric(tri, pts):
    """Barycentric coordinates (M, 3) of in-plane points, by signed areas.

    Independent of ``clq.shape``: this is the textbook signed-area formula,
    not the monomial/affine machinery the kernels use.
    """
    tri = np.asarray(tri, float)
    p = np.asarray(pts, float).reshape(-1, 3)
    v1, v2, v3 = tri
    nrm = np.cross(v2 - v1, v3 - v1)
    A2 = np.linalg.norm(nrm)
    nh = nrm / A2
    l1 = np.cross(v2 - p, v3 - p) @ nh / A2
    l2 = np.cross(v3 - p, v1 - p) @ nh / A2
    l3 = np.cross(v1 - p, v2 - p) @ nh / A2
    return np.stack([l1, l2, l3], axis=1)


def shape_independent(tri, order, pts):
    """Lagrange shape functions (M, K), textbook form, node order
    [v1, v2, v3, m12, m23, m31].  Independent of ``clq.shape``."""
    lam = barycentric(tri, pts)
    if order == 0:
        return np.ones((lam.shape[0], 1))
    if order == 1:
        return lam
    if order == 2:
        l1, l2, l3 = lam[:, 0], lam[:, 1], lam[:, 2]
        return np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                         4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=1)
    raise ValueError(f"order {order}")


def gauss_triangle(tri, n_quad):
    """Collapsed Legendre product rule on a triangle: points (Q,3), weights (Q,).

    Same construction as msd's ``integrate_dd_displacement_numerical`` (Duffy
    collapse of a tensor Gauss-Legendre rule), independent of
    ``clq.moments.gauss_triangle``.
    """
    tri = np.asarray(tri, float)
    v1, v2, v3 = tri
    x, w = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    a = x[:, None] * np.ones(n_quad)[None, :]                  # xi1
    b = x[None, :] * (1.0 - x[:, None])                        # xi2
    ww = w[:, None] * w[None, :] * (1.0 - x[:, None])
    area2 = np.linalg.norm(np.cross(v2 - v1, v3 - v1))
    y = ((1 - a - b)[..., None] * v1 + a[..., None] * v2 + b[..., None] * v3)
    return y.reshape(-1, 3), (ww * area2).reshape(-1)


# ---------------------------------------------------------------------------
# a small, deliberately irregular test mesh
# ---------------------------------------------------------------------------

def test_triangles():
    """Six non-degenerate, differently shaped and oriented triangles.

    Includes a near-equilateral, a thin sliver (aspect ~8), a strongly tilted
    one and a reversed-orientation copy (so the nhat sign enters), all with
    edge lengths of order 1 so ``far_field="analytic"`` is trustworthy over the
    observation set (clq loses digits beyond ~20 L).
    """
    t = np.array([
        [[0.00, 0.00, 0.00], [1.00, 0.00, 0.00], [0.30, 0.90, 0.05]],
        [[1.00, 0.00, 0.00], [1.40, 1.00, 0.10], [0.30, 0.90, 0.05]],
        [[0.00, 0.00, 0.00], [0.30, 0.90, 0.05], [-0.60, 0.50, 0.30]],
        [[0.37, -0.81, 0.44], [1.92, 0.11, -0.63], [-0.25, 1.57, 1.22]],   # msd's
        [[-0.50, -0.40, 0.20], [0.60, -0.55, 0.15], [0.05, -0.33, 0.18]],  # sliver
        [[0.30, 0.90, 0.05], [1.00, 0.00, 0.00], [0.55, 0.45, 0.90]],      # flipped-ish
    ])
    return np.ascontiguousarray(t)


def test_observers(include_on_plane=True):
    """Observation points: near, far, behind, and (optionally) on an element plane."""
    obs = [[0.50, 0.40, 0.70], [2.00, 1.00, 1.00], [-0.20, 0.30, -0.40],
           [0.80, -0.60, 0.25], [3.50, -2.00, 1.50], [0.10, 0.10, 0.60],
           [-1.20, 0.80, -0.90], [1.10, 1.30, -0.35]]
    if include_on_plane:
        # inside element 0's plane, and outside its footprint but in-plane
        obs += [[0.433333333, 0.30, 0.0166666667], [1.60, -0.30, 0.0]]
    return np.ascontiguousarray(np.array(obs, float))


# ---------------------------------------------------------------------------
# reporting (clq's Report, so run_all's verdict parsing matches)
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, title: str):
        self.title = title
        self.rows = []
        print("=" * 78)
        print(title)
        print("=" * 78)

    def check(self, name: str, value: float, tol: float, extra: str = "") -> bool:
        ok = bool(value < tol)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:56s} {value:10.3e}  (tol {tol:.0e}) {extra}")
        return ok

    def check_bool(self, name: str, ok: bool, extra: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:56s} {extra}")
        return ok

    def note(self, text: str):
        print(f"       .  {text}")

    def finish(self) -> bool:
        ok = all(r[1] for r in self.rows)
        n_fail = sum(1 for r in self.rows if not r[1])
        print("-" * 78)
        if ok:
            print(f"PASS: {self.title} ({len(self.rows)} checks)")
        else:
            print(f"FAIL: {self.title} ({n_fail} of {len(self.rows)} checks failed)")
        return ok
