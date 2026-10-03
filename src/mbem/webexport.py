"""Quantise a sampled volume into a payload a browser can hold.

Lives in the package rather than in ``website/`` so the encoder is importable
and therefore gateable: it is the one piece of the website that can silently
corrupt data, and a wrong quantisation looks like a plausible picture.

WHY log10, NOT LINEAR. The fields span **5.2 decades** on the showcase model,
and linear quantisation cannot hold that. Measured max relative error on the
value itself:

    field        linear uint8   linear uint16   log10 -> uint8
    u_mag             99.7 %         39.1 %          2.37 %
    von_mises         99.7 %         54.0 %          2.39 %

Linear uint16 fails too, because 65,536 levels is fewer than the 10^5.2 of
dynamic range being asked for and every small value collapses to the bottom
bucket. So a positive field is encoded as log10 and a signed or bounded one
linearly, and which was used is recorded per array in the manifest.

WHY uint8 AND NOT float. A float32 3-D texture cannot be linearly filtered on
roughly half of iOS devices (``OES_texture_float_linear`` is at 54 % there), and
the failure is a silently black volume, not an error. uint8 is universal. 256
levels is also more than a colour map can show.

Index 0 is RESERVED for "no data", so values map to 1..255: the grid covers the
bounding box and the body does not fill it, and a reader must be able to tell an
absent voxel from a small one.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

SCHEMA = 1
NODATA = 0
LEVELS = 254                       # 1..255 inclusive
# Fields worth shipping: what a colour map can show, not the raw tensor. Six
# components of sigma are for ParaView; a web viewer wants scalars.
DEFAULT_FIELDS = ("u_mag", "von_mises", "max_shear")
CLEARANCE_CLAMP = 3.0              # clearance_h above this is "well clear"


def encode_log_u8(v: np.ndarray) -> tuple[bytes, dict]:
    """Positive, many-decade field -> uint8 over log10. 0 means no data."""
    v = np.asarray(v, float)
    good = np.isfinite(v) & (v > 0.0)
    if not good.any():
        raise ValueError("no positive finite samples to encode")
    lo, hi = float(np.log10(v[good].min())), float(np.log10(v[good].max()))
    if hi <= lo:
        hi = lo + 1e-12
    q = np.zeros(v.shape, np.uint8)
    t = (np.log10(np.where(good, v, 1.0)) - lo) / (hi - lo)
    q[good] = 1 + np.clip(np.round(t[good] * (LEVELS - 1)), 0,
                          LEVELS - 1).astype(np.uint8)
    return q.tobytes(), {"encoding": "log10_u8", "log_min": lo, "log_max": hi,
                         "min": float(v[good].min()),
                         "max": float(v[good].max())}


def decode_log_u8(buf, meta: dict, shape=None) -> np.ndarray:
    """Inverse of :func:`encode_log_u8`; absent voxels come back NaN."""
    q = np.frombuffer(buf, np.uint8)
    if shape is not None:
        q = q.reshape(shape)
    lo, hi = meta["log_min"], meta["log_max"]
    t = (q.astype(float) - 1.0) / (LEVELS - 1)
    out = 10.0 ** (lo + t * (hi - lo))
    return np.where(q == NODATA, np.nan, out)


def encode_linear_u8(v: np.ndarray, lo: float, hi: float) -> tuple[bytes, dict]:
    """Bounded field -> uint8, clamped to ``[lo, hi]``. 0 means no data."""
    v = np.asarray(v, float)
    good = np.isfinite(v)
    q = np.zeros(v.shape, np.uint8)
    t = (np.clip(np.where(good, v, lo), lo, hi) - lo) / max(hi - lo, 1e-300)
    q[good] = 1 + np.clip(np.round(t[good] * (LEVELS - 1)), 0,
                          LEVELS - 1).astype(np.uint8)
    return q.tobytes(), {"encoding": "linear_u8", "min": lo, "max": hi}


def decode_linear_u8(buf, meta: dict, shape=None) -> np.ndarray:
    q = np.frombuffer(buf, np.uint8)
    if shape is not None:
        q = q.reshape(shape)
    lo, hi = meta["min"], meta["max"]
    out = lo + (q.astype(float) - 1.0) / (LEVELS - 1) * (hi - lo)
    return np.where(q == NODATA, np.nan, out)


def encode_codes_u8(v: np.ndarray) -> tuple[bytes, dict]:
    """A small integer code, stored as itself -- no scaling, no interpolation.

    The reader must bind this to a NEAREST sampler: linearly filtering a
    categorical code invents regions that do not exist between real ones.
    """
    q = np.nan_to_num(np.asarray(v, float), nan=0.0).astype(np.uint8)
    return q.tobytes(), {"encoding": "codes_u8", "filter": "nearest",
                         "values": sorted(int(x) for x in np.unique(q))}


def geometry(model, grid_meta: dict) -> dict:
    """Outlines a viewer draws over the slices, in km.

    Taken from the REBUILT model, because the runner drops ``bundle.scalars``
    and the fault is therefore not in ``meshes.npz`` at all. Outlines rather
    than meshes: the fault is 633 triangles and the inclusion rim 1,619, which
    is a lot of bytes to draw four lines and a circle.
    """
    out = {"box": {"origin": grid_meta["origin"],
                   "spacing": grid_meta["spacing"],
                   "dims": grid_meta["dims"]}}
    for r in model.regions:
        for p in list(r.patches) + list(r.faults):
            v = np.asarray(p.mesh.vertices, float)
            lo, hi = v.min(axis=0), v.max(axis=0)
            out.setdefault("patches", {})[p.name] = {
                "lo": [float(x) for x in lo], "hi": [float(x) for x in hi],
                "n_triangles": int(p.mesh.n_triangles),
                "is_fault": p in list(r.faults), "region": r.name}
    # The top surface as a height field on the grid's own x, y, so a viewer can
    # draw the topography without the 2,755-triangle mesh.
    top = next((p for r in model.regions for p in r.patches
                if p.name.endswith("host_top") or p.name == "top"), None)
    if top is not None:
        nx, ny, _ = grid_meta["dims"]
        ox, oy, _ = grid_meta["origin"]
        sx, sy, _ = grid_meta["spacing"]
        gx = ox + sx * np.arange(nx)
        gy = oy + sy * np.arange(ny)
        from scipy.interpolate import griddata
        c = np.asarray(top.mesh.vertices, float)
        X, Y = np.meshgrid(gx, gy, indexing="xy")
        Z = griddata(c[:, :2], c[:, 2], (X, Y), method="linear")
        out["topography"] = {
            "nx": int(nx), "ny": int(ny),
            "x0": float(ox), "y0": float(oy), "dx": float(sx), "dy": float(sy),
            "z": [None if not np.isfinite(z) else round(float(z), 4)
                  for z in Z.ravel()]}
    return out


def export(volume_dir, out_dir, fields=DEFAULT_FIELDS, model=None) -> dict:
    """Write every ``.vti`` in ``volume_dir`` as a uint8 payload + manifest.

    Returns the manifest. Arrays are separate files so a page fetches only the
    state and field on screen; the whole set is ~1 MB but first paint is ~230 kB.
    """
    from mbem import vti

    volume_dir = pathlib.Path(volume_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = json.loads((volume_dir / "resolved.json").read_text())
    grid = src["grid"]
    shape = (grid["dims"][2], grid["dims"][1], grid["dims"][0])

    man = {"schema": SCHEMA, "grid": grid, "nodata": NODATA, "levels": LEVELS,
           "source": {"volume_run": volume_dir.name,
                      "sampled_from": src.get("source_run"),
                      "git": src.get("env", {}).get("git")},
           "states": {}, "arrays": {}}

    def put(name: str, payload: bytes, meta: dict) -> None:
        (out_dir / f"{name}.bin").write_bytes(payload)
        meta = dict(meta)
        meta["bytes"] = len(payload)
        meta["sha256"] = hashlib.sha256(payload).hexdigest()[:16]
        man["arrays"][name] = meta

    wrote_shared = set()
    for p in sorted(volume_dir.glob("*.vti")):
        label = p.stem.replace("volume_", "")
        A = vti.read(p)["arrays"]
        state = {"kind": "difference" if p.stem.startswith("diff_") else "state",
                 "fields": {}}
        for f in fields:
            if f not in A:
                continue
            v = np.asarray(A[f], float)
            if state["kind"] == "difference":
                v = np.abs(v)           # a difference is signed; show magnitude
            buf, meta = encode_log_u8(v)
            key = f"{label}.{f}"
            put(key, buf, meta)
            state["fields"][f] = key
        # region and clearance belong to the GEOMETRY, which topo and flat do
        # not share -- a point under the hill is inside one body and outside the
        # other -- so they are per state, deduplicated by content.
        for name, enc in (("region", lambda a: encode_codes_u8(a)),
                          ("clearance_h",
                           lambda a: encode_linear_u8(a, 0.0, CLEARANCE_CLAMP))):
            buf, meta = enc(np.asarray(A[name], float))
            h = hashlib.sha256(buf).hexdigest()[:16]
            key = f"{name}.{h}"
            if key not in wrote_shared:
                put(key, buf, meta)
                wrote_shared.add(key)
            state[name] = key
        man["states"][label] = state

    if model is not None:
        geo = geometry(model, grid)
        (out_dir / "geometry.json").write_text(json.dumps(geo))
        man["geometry"] = "geometry.json"
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1,
                                                      sort_keys=True) + "\n")
    # Prune arrays this export did not write. Keys for region and clearance are
    # CONTENT HASHES, so changing what they mean renames them and the old files
    # would otherwise linger -- dead weight in every clone of the site, and
    # indistinguishable from live data.
    keep = {f"{k}.bin" for k in man["arrays"]} | {"manifest.json",
                                                  "geometry.json"}
    stale = [q for q in out_dir.iterdir() if q.is_file() and q.name not in keep]
    for q in stale:
        q.unlink()
    man["pruned"] = sorted(q.name for q in stale)
    return man
