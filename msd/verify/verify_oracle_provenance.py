"""Gate: every frozen oracle resolves to the file it is supposed to, unchanged.

WHY THIS EXISTS. The oracles are frozen and the live kernels are gated against
them by entrywise parity, so a parity clause is only worth what its reference is
worth. Two failure modes make a reference worthless WITHOUT making any gate red:

  * A SWAP. ``msd/mollified_kernel/`` and ``moss/mollified_kernel/`` are two
    independent implementations of the same analytic kernels, installed as
    ``mollified_kernel`` and ``moss_kernel``. If an import ever resolves one to
    the other, the parity clause compares a copy against ITSELF and still prints
    PASS -- the residual merely drops from ~1e-12 to ~1e-16, which no tolerance
    rejects. That is not hypothetical: three files in moss's copy hardcoded the
    prefix ``mollified_kernel.``, which resolved to msd's file the moment msd's
    copy became an installed package, and resolved SUCCESSFULLY because msd's
    copy defines the same two function names.
  * AN "IMPROVEMENT". An oracle edited to agree with the code it gates turns
    every clause that uses it into a tautology.

So this gate pins both the identity (which file each module name resolves to)
and the contents (sha256) of every frozen oracle, and asserts the one asymmetry
that cannot be faked: msd's copy does NOT define the eigenstress oracle and
moss's does, which is why ``verify_eigenstress_exact`` needs moss's and has no
substitute for it.

A deliberate edit to a frozen oracle -- an import prefix, say -- is meant to
require updating ``oracle_manifest.json`` in the same commit. That is the point:
it makes the edit visible in review instead of silent. Regenerate with

    python verify/verify_oracle_provenance.py --write

and say in the commit message why the hash moved.

Run from msd/. Prints PASS:/FAIL:, exits 1 on FAIL.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "oracle_manifest.json"

# module name -> the path tail its __file__ must end with. The tail is relative
# to the repo root, so it pins WHICH COPY the name resolves to, not merely that
# something importable exists.
ORACLES = {
    "mollified_bem": "msd/mollified_bem.py",
    "anelastic": "msd/anelastic.py",
    "tde_reference": "msd/tde_reference.py",
    "local_box_mesh": "msd/local_box_mesh.py",
    "local_box_mesh_eq": "msd/local_box_mesh_eq.py",
    "inclusion_mesh": "msd/inclusion_mesh.py",
    "mollified_kernel.analytical_kernels":
        "msd/mollified_kernel/analytical_kernels.py",
    "mollified_kernel.analytical_batch":
        "msd/mollified_kernel/analytical_batch.py",
    "mollified_kernel.mollified_elastic_kernels":
        "msd/mollified_kernel/mollified_elastic_kernels.py",
    "moss_kernel.analytical_kernels":
        "moss/mollified_kernel/analytical_kernels.py",
    "moss_kernel.analytical_batch":
        "moss/mollified_kernel/analytical_batch.py",
    "moss_kernel.mollified_elastic_kernels":
        "moss/mollified_kernel/mollified_elastic_kernels.py",
    "moss_kernel.mindlin_kernels":
        "moss/mollified_kernel/mindlin_kernels.py",
    "moss_kernel.mindlin_triangle":
        "moss/mollified_kernel/mindlin_triangle.py",
    "clq.kernels": "clq/clq/kernels.py",
    "clq.primitives": "clq/clq/primitives.py",
    "clq.moments": "clq/clq/moments.py",
}

# Names present in exactly one of the two kernel copies, per MODULE -- the
# eigenstress oracle is split across two files, the pointwise kernel in
# analytical_kernels and the batch driver in analytical_batch. This asymmetry is
# what a swap cannot preserve, so it is checked directly rather than inferred
# from hashes. {module suffix: (moss-only names, names both must have)}
ASYMMETRY = {
    "analytical_kernels": (("analytical_eigenstress_kernel",),
                           ("analytical_dd_displacement",
                            "analytical_kelvin_G")),
    "analytical_batch": (("eigenstress_batch",),
                         ("dd_displacement_batch",)),
}


def _resolve() -> tuple[dict, list]:
    """``({name: (path, sha256)}, problems)`` for every pinned oracle."""
    got, problems = {}, []
    for name, tail in ORACLES.items():
        try:
            mod = importlib.import_module(name)
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        path = pathlib.Path(getattr(mod, "__file__", "") or "")
        if not path.as_posix().endswith(tail):
            problems.append(f"{name} resolved to {path}, expected .../{tail}")
            continue
        got[name] = (tail, hashlib.sha256(path.read_bytes()).hexdigest())
    return got, problems


def _asymmetry() -> list:
    """The two kernel copies must be distinct, and differently populated."""
    out = []
    for suffix, (moss_only, both) in ASYMMETRY.items():
        try:
            msd = importlib.import_module(f"mollified_kernel.{suffix}")
            moss = importlib.import_module(f"moss_kernel.{suffix}")
        except Exception as exc:                      # noqa: BLE001
            out.append(f"cannot import both copies of {suffix}: {exc}")
            continue
        if msd is moss:
            out.append(f"mollified_kernel.{suffix} and moss_kernel.{suffix} "
                       "are the SAME module object -- one shadows the other")
        for fn in moss_only:
            if not hasattr(moss, fn):
                out.append(f"moss_kernel.{suffix} lost {fn} "
                           "(the eigenstress oracle)")
            if hasattr(msd, fn):
                out.append(f"mollified_kernel.{suffix} gained {fn}: the copies "
                           "have converged, so the parity clause that compares "
                           "them is now a tautology")
        for fn in both:
            for tag, m in ((f"mollified_kernel.{suffix}", msd),
                           (f"moss_kernel.{suffix}", moss)):
                if not hasattr(m, fn):
                    out.append(f"{tag} lost {fn}")
    return out


def main(write: bool = False) -> bool:
    got, problems = _resolve()
    print(f"    {len(got)} of {len(ORACLES)} oracles resolved")
    for name in sorted(got):
        tail, _h = got[name]
        print(f"      {name:44s} -> {tail}")

    if write:
        MANIFEST.write_text(json.dumps(
            {n: {"path": t, "sha256": h} for n, (t, h) in sorted(got.items())},
            indent=1) + "\n")
        print(f"    wrote {MANIFEST.name} with {len(got)} entries")
        return not problems

    if not MANIFEST.exists():
        print(f"FAIL: oracle provenance ({MANIFEST.name} missing; "
              f"regenerate with --write)")
        return False
    pinned = json.loads(MANIFEST.read_text())

    for name, rec in sorted(pinned.items()):
        if name not in got:
            problems.append(f"{name}: pinned but did not resolve")
            continue
        tail, h = got[name]
        if tail != rec["path"]:
            problems.append(f"{name}: path {tail} != pinned {rec['path']}")
        if h != rec["sha256"]:
            problems.append(f"{name}: CONTENTS CHANGED\n"
                            f"        pinned {rec['sha256']}\n"
                            f"        now    {h}")
    extra = sorted(set(got) - set(pinned))
    if extra:
        problems.append(f"resolved but not pinned: {extra}")

    asym = _asymmetry()
    print(f"    the two kernel copies: "
          f"{'distinct, differently populated' if not asym else 'PROBLEM'}")
    problems += asym

    for p in problems:
        print(f"      !! {p}")
    ok = not problems
    print(f"{'PASS' if ok else 'FAIL'}: oracle provenance "
          f"({len(pinned)} files pinned by path and sha256)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(write="--write" in sys.argv) else 1)
