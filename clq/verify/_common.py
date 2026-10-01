"""Shared helpers for the clq verification scripts.

Every ``verify_*.py`` is run from the clq root as ``python verify/<name>.py``
(any interpreter with numpy; ``verify_primitives``, ``verify_pointwise`` and
``verify_regressions`` also need sympy / mpmath) and ends with exactly one
line starting with ``PASS`` or ``FAIL``.  ``run_all.py`` launches each gate
with ``sys.executable``.
"""
from __future__ import annotations

import numpy as np

# A generic tilted, non-right, non-unit triangle (msd's verify_arbitrary_triangle).
TRI = np.array([[0.37, -0.81, 0.44],
                [1.92, 0.11, -0.63],
                [-0.25, 1.57, 1.22]])
MU, NU_DEFAULT = 1.0, 0.3


# The oracles are ordinary imports of installed packages. They were loaded by
# FILE PATH under private module names, which the two kernel copies required
# while they shared the name ``mollified_kernel``; they are now installed under
# distinct names, so one interpreter can hold both and the path arithmetic --
# which also forced clq, msd and moss to stay siblings -- is gone.
#
# Each accessor asserts WHICH copy it got. The two are independent
# implementations compared entrywise here, so a name resolving to the wrong one
# would make the comparison a copy against itself: still PASS, residual merely
# sliding from ~1e-12 to ~1e-16. verify_oracle_provenance (in msd) pins this
# repo-wide; these asserts make each clq gate fail on its own too.

def msd_analytical():
    import mollified_kernel.analytical_kernels as m
    assert "msd" in m.__file__, m.__file__
    return m


def msd_batch():
    import mollified_kernel.analytical_batch as m
    assert "msd" in m.__file__, m.__file__
    return m


def moss_analytical():
    import moss_kernel.analytical_kernels as m
    assert "moss" in m.__file__, m.__file__
    return m


def msd_anelastic():
    import anelastic
    return anelastic


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
