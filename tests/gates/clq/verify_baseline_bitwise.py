"""Gate: the dislocation kernels are BITWISE unchanged from the baseline.

`verify_regressions.py` checks that `want=("U",)` and `want=("U","S")` agree --
want-INDEPENDENCE, within whatever the current code happens to compute.  It
cannot see a change that moves both equally, which is exactly what a refactor
of the shared moment machinery does.  This gate closes that hole: it pins
`U`, `H` and `E` against a stored manifest of byte hashes taken from the
pre-force-element tree (commit 72c2840, 2026-09-17).

Why bitwise and not a tolerance: the force element reuses `MomentTable`, the
edge primitives and `lift`, so every future force-side change is a chance to
perturb the slip kernels.  The perturbations are ~1e-16 and no published result
would move -- but a tolerance gate would let them accumulate silently, and
"the slip kernels are frozen" is the property `clq/CLAUDE.md` actually claims.
One real instance has already occurred and been reverted: regrouping
`dp ** pw` out of `coeff` in `moments.py` moved 40 of 48 arrays by up to
1.9e-15, and nothing caught it.

Regenerating the manifest defeats the gate.  Do it ONLY when a change to the
slip kernels is intended, and say so in the commit message:

    python verify/verify_baseline_bitwise.py --regenerate

BYTES ARE PER PLATFORM.  The hashes are of float64 bytes, and those depend on
the CPU and on numpy's SIMD kernels as well as on the code: on x86-64 Linux
576 of the 648 arrays differ from the manifest recorded on the Mac, by
rounding, with identical sources.  A tolerance is still the wrong answer --
the gate would stop being bitwise everywhere -- so a platform may carry its
OWN pin under ``platforms[<sys.platform>-<machine>]``, compared bitwise in
place of the reference entries.  A platform with no pin of its own is compared
against the reference, exactly as before.

    python verify/verify_baseline_bitwise.py --record-platform

records one, and refuses unless the clq closed-form sources are byte-identical
to the frozen-oracle manifest (``tests/gates/mbem/oracle_manifest.json``, the
one statement of what the oracle IS): a platform pin is a statement about the
oracle on that platform, not about whatever tree happens to be checked out.
``--regenerate`` rewrites this platform's pin if it has one and the reference
entries otherwise, so on the reference platform it does what it always did;
other platforms' pins then fail there until recorded again, which is right.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys

import numpy as np

import clq                                                       # noqa: E402

MANIFEST = pathlib.Path(__file__).resolve().parent / "baseline_uhe.json"
BASELINE_COMMIT = "72c2840"
PLATFORM = f"{sys.platform}-{platform.machine()}"
ORACLES = (pathlib.Path(__file__).resolve().parents[1] / "mbem"
           / "oracle_manifest.json")


def _drifted_oracles() -> list:
    """clq sources that are NOT the frozen oracle's, by the oracle manifest."""
    root = ORACLES.resolve().parents[3]
    pins = json.loads(ORACLES.read_text())
    return sorted(
        name for name, pin in pins.items() if name.startswith("clq.")
        and hashlib.sha256((root / pin["path"]).read_bytes()).hexdigest()
        != pin["sha256"])


def _write(man: dict) -> None:
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))

TRIS = {
    "equilateral": [[0., 0., 0.], [1., 0., 0.], [0.5, np.sqrt(3) / 2, 0.]],
    "scalene":     [[0.1, -0.2, 0.05], [1.3, 0.2, 0.4], [0.3, 1.1, -0.2]],
    "sliver":      [[0., 0., 0.], [2., 0., 0.], [1., 0.04, 0.]],
}
ORDERS = (0, 1, 2, 3)
NUS = (0.25, 0.30, 0.45)
EPSILONS = (0.0, 0.3, 1.1)
FAR = ("analytic", "hybrid")


def observations():
    rng = np.random.default_rng(12345)
    return rng.normal(size=(40, 3)) * 1.7 + np.array([0.3, 0.2, 0.5])


def sweep():
    """{key: sha256 of the array bytes} over the full configuration sweep."""
    obs = observations()
    out = {}
    for tn, tri in TRIS.items():
        tri = np.asarray(tri, float)
        for p in ORDERS:
            for nu in NUS:
                for eps in EPSILONS:
                    for ff in FAR:
                        try:
                            r = clq.influence(obs, tri, 1.0, nu, eps, order=p,
                                              want=("U", "H", "E"), far_field=ff)
                        except Exception:
                            continue          # configurations that legitimately raise
                        for k in ("U", "H", "E"):
                            v = getattr(r, k)
                            if v is None:
                                continue
                            a = np.ascontiguousarray(v, dtype=np.float64)
                            key = f"{tn}|{p}|{nu}|{eps}|{ff}|{k}"
                            out[key] = (hashlib.sha256(a.tobytes()).hexdigest(),
                                        list(a.shape))
    return out


def _platform_pin(cur: dict) -> dict:
    return {"entries": {k: v[0] for k, v in cur.items()},
            "recorded": {"numpy": np.__version__,
                         "python": platform.python_version(),
                         "note": "clq closed-form sources byte-identical to "
                                 "oracle_manifest.json when recorded"}}


def main(argv):
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    pins = man.get("platforms", {})

    if "--record-platform" in argv:
        drift = _drifted_oracles()
        if drift:
            print(f"FAIL: not recording a pin for {PLATFORM}: {drift} differ "
                  f"from the frozen-oracle manifest, so this tree is not the "
                  f"oracle")
            return False
        if not man:
            print(f"FAIL: no reference manifest at {MANIFEST} to add to")
            return False
        cur = sweep()
        man.setdefault("platforms", {})[PLATFORM] = _platform_pin(cur)
        _write(man)
        print(f"recorded a bitwise pin for {PLATFORM} ({len(cur)} arrays); "
              f"the reference entries are untouched")
        return True

    if "--regenerate" in argv:
        cur = sweep()
        if PLATFORM in pins:
            pins[PLATFORM] = _platform_pin(cur)
            _write(man)
            print(f"regenerated the {PLATFORM} pin with {len(cur)} entries")
            return True
        man.update({"baseline_commit": BASELINE_COMMIT,
                    "note": "sha256 of float64 bytes; see "
                            "verify_baseline_bitwise.py",
                    "entries": {k: v[0] for k, v in cur.items()},
                    "shapes": {k: v[1] for k, v in cur.items()}})
        _write(man)
        print(f"regenerated {MANIFEST.name} with {len(cur)} entries -- "
              "the bitwise gate is now pinned to the CURRENT tree")
        return True

    if not man:
        print(f"FAIL: baseline bitwise -- no manifest at {MANIFEST}")
        return False
    own = PLATFORM in pins
    want = pins[PLATFORM]["entries"] if own else man["entries"]
    shapes = man.get("shapes", {})
    cur = sweep()
    print(f"reference: {'the ' + PLATFORM + ' pin' if own else 'the baseline entries (no pin for ' + PLATFORM + ')'}")

    missing = sorted(set(want) - set(cur))
    extra = sorted(set(cur) - set(want))
    changed = sorted(k for k in set(want) & set(cur) if cur[k][0] != want[k])
    reshaped = sorted(k for k in set(want) & set(cur)
                      if k in shapes and list(cur[k][1]) != list(shapes[k]))

    print(f"baseline {man['baseline_commit']}: {len(want)} arrays; "
          f"recomputed {len(cur)}")
    print(f"  bitwise identical {len(set(want) & set(cur)) - len(changed)}")
    print(f"  changed           {len(changed)}")
    print(f"  missing now       {len(missing)}   (config raises but used to work)")
    print(f"  new              {len(extra)}   (config works but is unpinned)")
    if reshaped:
        print(f"  reshaped          {len(reshaped)}")
    for k in changed[:8]:
        print(f"    changed: {k}")
    if len(changed) > 8:
        print(f"    ... and {len(changed) - 8} more")
    for k in missing[:5]:
        print(f"    missing: {k}")

    bad = bool(changed or missing or reshaped)
    if changed and not own:
        print(f"  note: byte hashes are per platform, and {PLATFORM} has no "
              f"pin of its own. If verify_oracle_provenance passes, the "
              f"sources are the oracle's and --record-platform pins them here")
    if extra and not bad:
        print("  note: new configurations are not a failure, but run "
              "--regenerate to pin them once the change is intended")
    n = len(set(want) & set(cur))
    print(f"{'FAIL' if bad else 'PASS'}: baseline bitwise, U/H/E vs "
          f"{man['baseline_commit']} ({n} arrays)")
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
