"""Run every fbem gate and print one PASS/FAIL line each, then a summary.

Mirrors clq/verify/run_all.py: each gate is a standalone script that ends with
a single line "PASS <name>" or "FAIL <name>", and is launched with the same
interpreter that runs this one, so nothing assumes a particular python path.

    /Users/meade/micromamba/bin/python verify/run_all.py
    /Users/meade/micromamba/bin/python verify/run_all.py l0 l1   # a subset
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def gates(patterns):
    found = sorted(p for p in HERE.glob("verify_*.py"))
    if not patterns:
        return found
    return [p for p in found if any(s in p.stem for s in patterns)]


def main(patterns):
    todo = gates(patterns)
    if not todo:
        print("no gates matched")
        return 1
    width = max(len(p.stem) for p in todo)
    results = []
    for p in todo:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT,
                           capture_output=True, text=True)
        out = (r.stdout or "").strip().splitlines()
        last = out[-1].strip() if out else ""
        if last.startswith("PASS"):
            verdict = "PASS"
        elif last.startswith("FAIL"):
            verdict = "FAIL"
        else:
            verdict = "ERROR"
        results.append((p.stem, verdict, time.time() - t0, r, last))
        print(f"  {p.stem:<{width}}  {verdict:<5} {time.time()-t0:6.1f}s  {last[:60]}",
              flush=True)

    bad = [x for x in results if x[1] != "PASS"]
    print(f"\n{len(results) - len(bad)}/{len(results)} gates passed")
    for stem, verdict, _, r, _ in bad:
        print(f"\n--- {stem} ({verdict}) ---")
        tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-25:]
        print("\n".join("    " + t for t in tail))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
