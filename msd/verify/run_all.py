"""Run every verify_*.py in this directory, sequentially, and summarise.

    python verify/run_all.py            # from the msd root
Each gate prints one final ``PASS: <title>`` / ``FAIL: <title>`` line and
exits 1 on FAIL; a non-zero exit with no such line also counts as FAIL.
Exit status 1 if any script fails.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    scripts = sorted(HERE.glob("verify_*.py"))
    results = []
    for s in scripts:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(s)], cwd=str(ROOT),
                              capture_output=True, text=True)
        dt = time.perf_counter() - t0
        verdict = "FAIL"
        for ln in reversed(proc.stdout.splitlines()):
            if ln.startswith("PASS") or ln.startswith("FAIL"):
                verdict = ln.split(":")[0]
                break
        if proc.returncode != 0:
            verdict = "FAIL"
        results.append((s.name, verdict, dt, proc.returncode))
        tail = (proc.stderr.strip().splitlines()[-1]
                if proc.returncode != 0 and proc.stderr.strip() else "")
        print(f"{verdict:4s}  {s.name:40s} {dt:7.1f} s  exit {proc.returncode}  {tail}",
              flush=True)
    n_fail = sum(1 for r in results if r[1] != "PASS")
    print("-" * 72)
    print(f"{len(results) - n_fail} / {len(results)} scripts PASS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
