"""Quantise a sampled volume into a payload a browser can hold; and, below,
a Laplace run into fields a browser can move through time.

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


# ---------------------------------------------------------------------
# a Laplace run through time
# ---------------------------------------------------------------------
#
# A run of the Laplace route stores one solution per state, and time is a
# linear map of that stack (``laplace.Prony``). Its weights are large and of
# both signs -- they sum to ~200 in magnitude, ~800 for a rate -- so the
# states themselves cannot be quantised and combined by a reader: the error
# of each is multiplied by the weights that cancel. But the stack has few
# rows and its singular values fall by a decade apiece, so it is shipped as
# an ORTHONORMAL BASIS of its rows, ``stack = C @ modes``, and every quantity
# of time as a small table ``weights @ C`` made here, where the weights are.
# A reader adds a handful of stored fields with one row of a table; the
# coefficient of a mode is then bounded by the field it is a part of, and the
# inversion is stated once, in ``laplace``.
#
# The modes are float32, not uint8: they go to a vertex buffer, never to a
# texture, so nothing has to be filtered, and a stress keeps its five decades.


def node_sheet(vertices, triangles, faults=(), on_fault: float = None):
    """``(nodes (M, 3), triangles (T, 3), A)``: a P0 mesh as a sheet a viewer
    can warp, ``A`` the sparse ``(M, T)`` map from a value per element to a
    value per node.

    A node takes the mean of the elements around it, weighted by the inverse
    of the distance to their centroids. It is a DISPLAY smoothing and says
    nothing the elements do not; what it must not do is average across a
    fault, where the displacement jumps. A node lying on one of ``faults``
    (``TriMesh``) is therefore split, one copy per side, each averaging the
    elements of its own side only. Vertices that COINCIDE are one node: a
    patch built of pieces that share no vertices (an interface that wraps a
    slab) is still one surface, and would otherwise tear along its seams.
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    from . import defaults
    from .geometry import distance_to_mesh

    v = np.asarray(vertices, float)
    tri = np.asarray(triangles, np.int64)
    nt = tri.shape[0]
    corners = v[tri]
    cen = corners.mean(axis=1)
    edge = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    tol = (defaults.WEB_ON_FAULT if on_fault is None else on_fault) * edge.min()
    pairs = cKDTree(v).query_pairs(tol, output_type="ndarray")
    if pairs.size:
        _n, same = sparse.csgraph.connected_components(sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(v.shape[0],) * 2), directed=False)
        tri = same[tri]                  # a label per group of coincident vertices
    side = np.zeros((nt, 3), np.int64)
    for f in faults:
        d, near = distance_to_mesh(v, f)
        on = d < tol
        if not on.any():
            continue
        fv = np.asarray(f.vertices, float)[np.asarray(f.triangles)]
        normal = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
        for c in range(3):
            sel = on[tri[:, c]]
            k = near[tri[sel, c]]
            s = np.sign(np.einsum("ij,ij->i", cen[sel] - fv[k, 0], normal[k]))
            side[sel, c] = 3 * side[sel, c] + s.astype(np.int64) + 2
    _keys, index = np.unique(np.column_stack([tri.ravel(), side.ravel()]),
                             axis=0, return_inverse=True)
    index = np.asarray(index).ravel()
    nodes = np.empty((index.max() + 1, 3))
    nodes[index] = corners.reshape(-1, 3)
    w = 1.0 / np.linalg.norm(cen[:, None, :] - corners, axis=2)
    A = sparse.csr_matrix((w.ravel(), (index, np.repeat(np.arange(nt), 3))),
                          shape=(nodes.shape[0], nt))
    A = sparse.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A
    return nodes, index.reshape(nt, 3), A


def boundary_segments(vertices, triangles) -> np.ndarray:
    """``(n, 2, 3)``: the edges of a mesh that belong to one triangle only --
    a fault's outline, as the segments a viewer draws."""
    tri = np.asarray(triangles, np.int64)
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                tri[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    return np.asarray(vertices, float)[uniq[count == 1]]


def encode_modes(stack, floor: float = None) -> tuple:
    """``(modes, C)`` with ``stack = C @ modes`` up to the modes dropped.

    ``stack`` is ``(n_states, ...)``; ``modes`` is float32 ``(r, ...)``,
    orthonormal over everything but the first axis, and ``C`` is
    ``(n_states, r)`` in float64. Modes below ``floor`` of the largest
    singular value are dropped (``defaults.WEB_MODE_FLOOR``).
    """
    from . import defaults

    a = np.asarray(stack, float)
    if not np.isfinite(a).all():
        raise ValueError("a stack to encode must be finite everywhere: drop "
                         "the points outside the body first")
    u, sv, vt = np.linalg.svd(a.reshape(a.shape[0], -1), full_matrices=False)
    floor = defaults.WEB_MODE_FLOOR if floor is None else floor
    r = max(1, int((sv > floor * sv[0]).sum()))
    return (vt[:r].astype(np.float32).reshape((r,) + a.shape[1:]),
            u[:, :r] * sv[:r])


def decode_modes(modes, rows) -> np.ndarray:
    """What a reader computes: ``rows @ modes``, ``(n_rows, ...)``."""
    return np.tensordot(np.asarray(rows, float),
                        np.asarray(modes, np.float64), axes=1)


def time_weights(s, periods=(), per_decade: int = None,
                 phases: int = None) -> dict:
    """``{quantity: {"axis", "at", "w"}}``: every quantity of time a viewer
    shows, as the weights ``(n, n_states)`` that take a stack to it.

    ``s`` are the finite collocation points of the run. The times are
    log-spaced over the fit's own window (``laplace.window``), never beyond
    it:

    * ``total``: the response at ``t``, ``t = 0`` the elastic state;
    * ``since``: the same less the elastic state -- what moved after it;
    * ``rate``: its rate, inside the window only;
    * ``locked:T``: the velocity of a fault locked between earthquakes that
      repeat every ``T``, per unit slip rate (``Prony.locked_rate_weights``),
      at phases of the cycle.
    """
    from . import defaults, laplace

    per_decade = (defaults.WEB_TIMES_PER_DECADE if per_decade is None
                  else per_decade)
    phases = defaults.WEB_CYCLE_PHASES if phases is None else phases
    pr = laplace.Prony(s)
    t0, t1 = laplace.window(s)
    t = np.geomspace(t0, t1, 1 + int(round(np.log10(t1 / t0) * per_decade)))
    elastic = np.eye(1 + len(pr.s))[0]
    w = np.vstack([elastic, pr.weights(t)])
    at = np.concatenate([[0.0], t])
    out = {"total": {"axis": "time", "at": at, "w": w},
           "since": {"axis": "time", "at": at, "w": w - elastic},
           "rate": {"axis": "time", "at": t, "w": pr.rate_weights(t)}}
    ph = (np.arange(phases) + 0.5) / phases
    for period in periods:
        out[f"locked:{period:g}"] = {
            "axis": "phase", "period": float(period), "at": ph,
            "w": pr.locked_rate_weights(ph * period, period)}
    return out


def time_tables(s, C, periods=(), per_decade: int = None,
                phases: int = None) -> dict:
    """``time_weights`` applied to the coefficients ``encode_modes`` returned
    for a stack: ``rows = w @ C``, what a reader multiplies into the modes.
    A stack too small to need modes is shipped whole and takes ``w`` itself.
    """
    return {name: {**{k: v for k, v in q.items() if k != "w"},
                   "rows": q["w"] @ C}
            for name, q in time_weights(s, periods, per_decade, phases).items()}
