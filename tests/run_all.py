"""Run every gate in every suite. PASS/FAIL per gate, exit 1 on any failure.

    python tests/run_all.py                 # all suites
    python tests/run_all.py -k fmm          # substring over <suite>/<name>
    python tests/run_all.py --suite clq
    python tests/run_all.py --fast          # skip the SLOW ones
    python tests/run_all.py --list

EACH GATE IS A SUBPROCESS, INVOKED BY FILE PATH, and that is a requirement
rather than a style. Four gate filenames collide between suites; two copies of
the mollified kernels exist on purpose and must not contend for one module
identity; and several gates rebind ``defaults.X`` around a clause. Importing
them into one interpreter would let all three leak across gates. Invoking by
path also keeps ``sys.path[0]`` the gate's own directory, which is how
``verify_fmm`` reaches ``verify_hbackend`` and how three gates reach
``_sphere``.

THE COUNT IS ASSERTED, because the discovery is a glob: pointed at the wrong
directory it finds nothing, every "all passed" check trivially holds, and the
exit code is 0. A green empty suite is the one failure a test runner must not
be able to report, so an unexpected count is itself a failure.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# suite -> (directory, expected gate count). The count is a pin, not a hint.
SUITES = {
    "mbem": (HERE / "gates" / "mbem", 22),
    "clq": (HERE / "gates" / "clq", 16),
    # moss's own gates on the INDEPENDENT kernel copy -- the only
    # thing that has ever gated the copy holding the eigenstress oracle.
    "moss_kernel": (HERE / "gates" / "moss_kernel", 7),
}

# Gates over 60 s, skipped by --fast. verify_fmm's reference evaluates every
# M2L where it is used, in numpy, which is the whole 900 s.
SLOW = {"mbem/verify_fmm"}


def discover(suite: str) -> list:
    d, _n = SUITES[suite]
    return sorted(p for p in d.glob("verify_*.py") if p.name != "run_all.py")


def verdict(stdout: str) -> str:
    """The LAST line starting at column 0 with PASS or FAIL.

    Column 0 and last: every gate indents its per-clause lines, so an indented
    'PASS' is a clause and not the verdict, and gates that print a mid-run
    verdict are summarised by their final one.

    This is no longer the authority -- every gate now returns its verdict and
    exits on it, so the exit code decides. It is kept as a CROSS-CHECK: a gate
    whose printed verdict disagrees with its exit code is reported as MISMATCH
    rather than quietly trusted, because the two are meant to be one fact and a
    disagreement means one of them is lying. Scraping was the only signal until
    this run; 24 of the then-43 gates printed FAIL and exited 0.
    """
    for ln in reversed(stdout.splitlines()):
        if ln.startswith(("PASS", "FAIL")):
            return ln.split(":")[0]
    return "NOVERDICT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--only", default=None,
                    help="substring match over <suite>/<gate>")
    ap.add_argument("--suite", choices=sorted(SUITES), default=None)
    ap.add_argument("--fast", action="store_true", help="skip the slow gates")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--timeout", type=float, default=None, help="seconds/gate")
    a = ap.parse_args()

    suites = [a.suite] if a.suite else sorted(SUITES)
    rows, problems = [], []
    for suite in suites:
        found = discover(suite)
        want = SUITES[suite][1]
        if len(found) != want:
            problems.append(
                f"{suite}: discovered {len(found)} gates, expected {want} "
                f"(update SUITES, or the directory is wrong)")
        for p in found:
            gid = f"{suite}/{p.stem}"
            if a.only and a.only not in gid:
                continue
            if a.fast and gid in SLOW:
                continue
            rows.append((gid, p))

    if a.list:
        for gid, _p in rows:
            print(f"{gid}{'  (slow)' if gid in SLOW else ''}")
        print(f"{len(rows)} gates")
        for p in problems:                 # never exit non-zero in silence
            print(f"PROBLEM: {p}")
        return 1 if problems else 0

    if not rows and not problems:
        problems.append("no gates selected: a glob that matches nothing must "
                        "not report success")

    width = max((len(g) for g, _ in rows), default=10)
    n_fail = 0
    for gid, path in rows:
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT),
                                  capture_output=True, text=True,
                                  timeout=a.timeout)
            printed = verdict(proc.stdout)
            v = "PASS" if proc.returncode == 0 else "FAIL"
            if printed != v:
                v = f"MISMATCH({printed}/rc{proc.returncode})"
            tail = proc.stdout.strip().splitlines()[-1:] if v != "PASS" else []
        except subprocess.TimeoutExpired:
            v, tail = "TIMEOUT", []
        dt = time.perf_counter() - t0
        if v != "PASS":
            n_fail += 1
        print(f"{v:9s} {gid:{width}s} {dt:8.1f} s", flush=True)
        for ln in tail:
            print(f"          {ln[:160]}")

    for p in problems:
        print(f"PROBLEM: {p}")
    n_fail += len(problems)
    print(f"\n{len(rows) - (n_fail - len(problems))} / {len(rows)} gates PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
