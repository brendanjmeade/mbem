"""Run every verify_*.py in this directory and summarise PASS/FAIL.

    python verify/run_all.py            # from the clq root
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
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        verdict = "FAIL"
        for ln in reversed(lines):
            if ln.startswith("PASS") or ln.startswith("FAIL"):
                verdict = ln.split(":")[0]
                break
        if proc.returncode != 0 and verdict == "PASS":
            verdict = "FAIL"
        results.append((s.name, verdict, dt))
        tail = proc.stderr.strip().splitlines()[-1] if proc.returncode != 0 and proc.stderr.strip() else ""
        print(f"{verdict:4s}  {s.name:40s} {dt:7.1f} s  {tail}")
    n_fail = sum(1 for r in results if r[1] != "PASS")
    print("-" * 72)
    print(f"{len(results) - n_fail} / {len(results)} scripts PASS")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
