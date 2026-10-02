"""Native pytest tests: each gate is IMPORTED and its verdict is the assertion.

    pytest                      # all 43
    pytest -k fmm               # select
    pytest -m "not slow"        # skip the ~900 s one
    pytest -x --lf              # stop at the first failure, then rerun just it

Before this, each gate ran as a subprocess and its printed verdict was scraped
from stdout. That was necessary while the two copies of the mollified kernels
shared the name ``mollified_kernel`` and could shadow each other in one
interpreter; packaging them as ``mollified_kernel`` and ``moss_kernel`` removed
the reason, so the gates can be called directly and a failure can carry a
traceback and a captured log instead of an exit code.

IT ALSO REQUIRED GIVING 24 GATES A VERDICT THEY DID NOT HAVE. They printed
``FAIL`` and exited 0 -- the clq suite discarded ``Report.finish()``'s bool
entirely and its ``__main__`` was a bare ``main()`` -- so nothing but a stdout
scrape could see a failure. ``main()`` now returns its verdict in all 43 and
``__main__`` exits on it, which is what makes ``assert gate.main()`` meaningful.

ONE GATE STAYS A SUBPROCESS ON PURPOSE: ``mbem/verify_oracle_provenance``
resolves each frozen oracle through ``importlib`` and checks WHICH FILE it got.
Run in this process it would observe a module table every other gate has already
populated, so it would be testing the session rather than a clean resolution.
A subprocess is the isolation it is actually asserting.
"""

from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

import run_all as R

# Gates that must keep their own interpreter, with the reason.
SUBPROCESS_ONLY = {
    "mbem/verify_oracle_provenance":
        "asserts which file each oracle import resolves to, so it needs a "
        "module table no other gate has touched",
}


def _gates():
    out = []
    for suite in sorted(R.SUITES):
        for path in R.discover(suite):
            gid = f"{suite}/{path.stem}"
            marks = [pytest.mark.slow] if gid in R.SLOW else []
            out.append(pytest.param(suite, path, id=gid, marks=marks))
    return out


GATES = _gates()


def test_gate_count():
    """Every suite holds the number of gates it is pinned to.

    Discovery is a glob, so a wrong directory finds nothing and every other
    test here passes by vacuity. This is what makes a green EMPTY suite
    impossible, so it is a test rather than an assumption.
    """
    for suite, (directory, expected) in sorted(R.SUITES.items()):
        found = R.discover(suite)
        assert len(found) == expected, (
            f"{suite}: {len(found)} gates in {directory}, expected {expected}")
    assert len(GATES) == sum(n for _d, n in R.SUITES.values())


def _import_gate(suite: str, path: pathlib.Path):
    """Import one gate under a suite-qualified name.

    Qualified because four gate filenames exist in two suites each; under the
    bare stem the second import would return the first suite's module and the
    test would silently check the wrong file.
    """
    name = f"_gate_{suite}_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


@pytest.mark.parametrize("suite,path", GATES)
def test_gate(suite, path, capsys):
    """Run one gate in this process; its own verdict is the assertion."""
    gid = f"{suite}/{path.stem}"
    if gid in SUBPROCESS_ONLY:
        proc = subprocess.run([sys.executable, str(path)], cwd=str(R.ROOT),
                              capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{gid} ({SUBPROCESS_ONLY[gid]})\n"
            + "\n".join(proc.stdout.strip().splitlines()[-25:]))
        return

    mod = _import_gate(suite, path)
    assert callable(getattr(mod, "main", None)), f"{gid} has no main()"
    # One gate takes argv (it accepts a --rebaseline flag); every other takes
    # nothing. Read it off the signature rather than naming the gate here, so a
    # second one needing arguments does not fail mysteriously.
    import inspect
    needs = [p for p in inspect.signature(mod.main).parameters.values()
             if p.default is inspect._empty
             and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    verdict = mod.main([]) if needs else mod.main()
    out = capsys.readouterr().out

    # A gate must still SAY what it decided, at column 0, as its last such
    # line. The printed verdict and the return value are two statements of one
    # fact, and a gate whose return said PASS while its output said FAIL would
    # be the worst of both.
    printed = next((ln.split(":")[0] for ln in reversed(out.splitlines())
                    if ln.startswith(("PASS", "FAIL"))), None)
    tail = "\n".join(out.strip().splitlines()[-25:])
    assert printed is not None, f"{gid} printed no column-0 verdict\n{tail}"
    assert bool(verdict) == (printed == "PASS"), (
        f"{gid}: main() returned {verdict!r} but printed {printed}\n{tail}")
    assert verdict, f"{gid} FAILED\n{tail}"
