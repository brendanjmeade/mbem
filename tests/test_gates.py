"""pytest adapter over the gate suites: one test per gate.

    pytest                     # all 43
    pytest -k fmm              # select
    pytest -m "not slow"       # skip the 900 s one
    pytest -x                  # stop at the first failure

THE TEST SPAWNS THE GATE AS A SUBPROCESS and does not import it. pytest
normally collects by import, and that is exactly what must not happen here:

  * Four gate filenames exist in two suites each (verify_dd_pairing,
    verify_arbitrary_triangle, verify_batch_vs_scalar,
    verify_analytical_vs_quadrature). Collected by import they would contend
    for one module name.
  * ``mollified_kernel`` and ``moss_kernel`` are two independent copies of the
    same kernels, compared entrywise by the gates. One module identity for both
    would make those comparisons a copy against itself -- which still PASSES,
    with the residual sliding from ~1e-12 to ~1e-16.
  * Several gates rebind ``defaults.X`` around a clause. In a shared
    interpreter a gate that failed mid-clause would leak that value into every
    gate after it.

Invoking by path also keeps ``sys.path[0]`` the gate's own directory, which is
how ``verify_fmm`` imports ``verify_hbackend`` and how three gates import
``_sphere``. ``norecursedirs = ["tests/gates"]`` in pyproject.toml is the belt
to this braces: it stops pytest collecting the gate files directly.

``run_all.py`` remains the authoritative runner and the two must agree on the
gate set -- ``test_gate_count`` is what keeps them agreeing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import run_all as R


def _gates():
    out = []
    for suite in sorted(R.SUITES):
        for path in R.discover(suite):
            gid = f"{suite}/{path.stem}"
            marks = [pytest.mark.slow] if gid in R.SLOW else []
            out.append(pytest.param(path, id=gid, marks=marks))
    return out


GATES = _gates()


def test_gate_count():
    """Every suite holds the number of gates it is pinned to.

    Discovery is a glob, so a wrong directory finds nothing and every other
    test in this file silently passes by vacuity. This is the test that makes a
    green empty suite impossible, so it is a test and not an assumption.
    """
    for suite, (directory, expected) in sorted(R.SUITES.items()):
        found = R.discover(suite)
        assert len(found) == expected, (
            f"{suite}: {len(found)} gates in {directory}, expected {expected}")
    assert len(GATES) == sum(n for _d, n in R.SUITES.values())


@pytest.mark.parametrize("gate", GATES)
def test_gate(gate):
    """One gate, in its own interpreter. Its verdict line is the assertion."""
    proc = subprocess.run([sys.executable, str(gate)], cwd=str(R.ROOT),
                          capture_output=True, text=True)
    verdict = R.verdict(proc.stdout)
    if proc.returncode != 0 or verdict != "PASS":
        tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
        err = "\n".join(proc.stderr.strip().splitlines()[-5:])
        pytest.fail(f"{gate.name}: verdict={verdict} rc={proc.returncode}\n"
                    f"--- last stdout ---\n{tail}\n"
                    f"--- stderr tail ---\n{err}")
