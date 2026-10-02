"""Turn a sampled volume run into the website's data payload.

    python website/scripts/make_payload.py runs/<volume-dir> [--fields a,b]

A thin wrapper: the encoding lives in ``mbem.webexport`` so it is importable and
gated (``verify_volume`` clause [g]), and this only locates things and reports.
Writes into ``website/public/data/``, which the site fetches through
``import.meta.env.BASE_URL`` -- a hardcoded ``/data/...`` works in dev and 404s
under the ``/mbem`` base in production.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
WEB = HERE.parent
REPO = WEB.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("volume_dir", help="a run directory written by mbem sample")
    ap.add_argument("--fields", default=None,
                    help="comma-separated; default u_mag,von_mises,max_shear")
    ap.add_argument("--out", default=str(WEB / "public" / "data"))
    a = ap.parse_args()

    from mbem import webexport as W

    vol = pathlib.Path(a.volume_dir)
    if not (vol / "resolved.json").exists():
        print(f"not a volume run (no resolved.json): {vol}")
        return 2
    fields = tuple(a.fields.split(",")) if a.fields else W.DEFAULT_FIELDS

    # The geometry outlines need the MODEL, which means rebuilding it from the
    # run the volume was sampled from -- the same recipe `mbem sample` uses, and
    # for the same reason: meshes.npz has no fault in it.
    model = None
    src = json.loads((vol / "resolved.json").read_text()).get("source_run")
    if src:
        from mbem import figures as F
        from mbem.cli import _rebuild
        d = pathlib.Path(src)
        if not d.is_absolute():
            d = REPO / src
        child = d
        if (d / "study.json").exists():
            rows = json.loads((d / "study.json").read_text())["rows"]
            child = d / rows[0]["dir"]
        if (child / "resolved.json").exists():
            _, model, _ = _rebuild(F.load_run(child).resolved)
            print(f"  geometry from {child.name}")
        else:
            print(f"  source run missing ({src}); geometry.json skipped")

    out = pathlib.Path(a.out)
    man = W.export(vol, out, fields=fields, model=model)

    total = sum(m["bytes"] for m in man["arrays"].values())
    print(f"  {len(man['states'])} states, {len(man['arrays'])} arrays, "
          f"{total / 1024:.0f} KB total")
    for name, st in sorted(man["states"].items()):
        print(f"    {name:34s} {st['kind']:10s} "
              f"{', '.join(sorted(st['fields']))}")
    print(f"  -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
