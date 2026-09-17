"""Shared helpers for the clq verification scripts.

Every ``verify_*.py`` is run from the clq root as ``python verify/<name>.py``
(any interpreter with numpy; ``verify_primitives``, ``verify_pointwise`` and
``verify_regressions`` also need sympy / mpmath) and ends with exactly one
line starting with ``PASS`` or ``FAIL``.  ``run_all.py`` launches each gate
with ``sys.executable``.
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

# A generic tilted, non-right, non-unit triangle (msd's verify_arbitrary_triangle).
TRI = np.array([[0.37, -0.81, 0.44],
                [1.92, 0.11, -0.63],
                [-0.25, 1.57, 1.22]])
MU, NU_DEFAULT = 1.0, 0.3


def load_module(name: str, path):
    """Import a legacy oracle file under a private module name."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def msd_analytical():
    return load_module("_msd_analytical_kernels",
                       MOSS_ORG / "msd" / "mollified_kernel" / "analytical_kernels.py")


def msd_batch():
    return load_module("_msd_analytical_batch",
                       MOSS_ORG / "msd" / "mollified_kernel" / "analytical_batch.py")


def moss_analytical():
    return load_module("_moss_analytical_kernels",
                       MOSS_ORG / "moss" / "mollified_kernel" / "analytical_kernels.py")


def msd_anelastic():
    return load_module("_msd_anelastic", MOSS_ORG / "msd" / "anelastic.py")


def relmax(a, b) -> float:
    """max|a - b| / max|b| (absolute if b is ~0)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ref = np.max(np.abs(b))
    diff = np.max(np.abs(a - b))
    return diff / ref if ref > 1e-300 else diff


def random_rotation(seed=0):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class Report:
    """Collect named checks and print the final verdict."""

    def __init__(self, title: str):
        self.title = title
        self.rows = []
        print("=" * 72)
        print(title)
        print("=" * 72)

    def check(self, name: str, value: float, tol: float, extra: str = "") -> bool:
        ok = bool(value < tol)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {value:10.3e}  (tol {tol:.0e}) {extra}")
        return ok

    def check_bool(self, name: str, ok: bool, extra: str = "") -> bool:
        ok = bool(ok)
        self.rows.append((name, ok))
        print(f"  [{'ok' if ok else 'XX'}] {name:52s} {extra}")
        return ok

    def finish(self) -> bool:
        ok = all(r[1] for r in self.rows)
        n_fail = sum(1 for r in self.rows if not r[1])
        print("-" * 72)
        if ok:
            print(f"PASS: {self.title} ({len(self.rows)} checks)")
        else:
            print(f"FAIL: {self.title} ({n_fail} of {len(self.rows)} checks failed)")
        return ok
