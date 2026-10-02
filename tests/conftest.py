"""pytest configuration: gate discovery, import isolation, defaults isolation.

THE GATES ARE NOW IMPORTED, not spawned, so three things that a subprocess made
free have to be arranged explicitly.

1. FILENAME COLLISIONS. Four gate names exist in two suites each
   (verify_dd_pairing, verify_arbitrary_triangle, verify_batch_vs_scalar,
   verify_analytical_vs_quadrature). ``importmode=importlib`` plus a per-suite
   package name keeps them distinct, so ``mbem``'s and ``moss_kernel``'s
   same-named gates are different modules and neither shadows the other.

2. SIBLING IMPORTS. Three gates import ``_sphere`` and one imports
   ``verify_hbackend``, which worked only because a script's own directory is
   ``sys.path[0]``. Each gate directory is put on the path here instead, once.

3. DEFAULTS ISOLATION, the one that would otherwise bite silently. Gates rebind
   ``mbem.defaults`` values around a clause -- 13 sites across verify_fmm and
   verify_hbackend -- and in a shared interpreter a test that failed mid-clause
   would leak its value into every test after it, changing results without
   changing any code. The autouse fixture snapshots every uppercase default and
   restores it, so the leak is impossible rather than merely unlikely.

What a subprocess also gave for free and is now GAINED rather than lost: numba
compiles once for the whole session instead of once per gate.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
GATE_DIRS = {
    "mbem": HERE / "gates" / "mbem",
    "clq": HERE / "gates" / "clq",
    "moss_kernel": HERE / "gates" / "moss_kernel",
}

# tests/ for `import run_all`, then each gate directory for the sibling imports
# (_sphere, verify_hbackend, _common, _quad_assembly).
for _p in [HERE, *GATE_DIRS.values()]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(autouse=True)
def defaults_isolated():
    """Every test sees ``mbem.defaults`` as it shipped, and leaves it that way.

    Restores rather than merely checking, so one gate's deliberate rebind --
    or an exception thrown between a set and its restore -- cannot change what a
    later gate measures. Yields the module so a test may rebind freely.
    """
    from mbem import defaults
    held = {k: v for k, v in vars(defaults).items() if k.isupper()}
    try:
        yield defaults
    finally:
        for k, v in held.items():
            setattr(defaults, k, v)
        for k in [k for k in vars(defaults) if k.isupper() and k not in held]:
            delattr(defaults, k)       # a default a test invented
