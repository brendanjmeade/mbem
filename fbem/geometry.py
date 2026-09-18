"""Rebuild (and cache) the topo_inclusion meshes used by the cached solution.

The cache `topo_inclusion_fields_mu10.npz` stores only the two OUTPUT surfaces
(host_top, inclusion_top).  The sides, base, interfaces and the fault have to
be regenerated from the same builder with the same parameters, so this module
imports `make_topo_inclusion.build()` itself rather than re-implementing it,
and caches it under `fbem/cache/` (git-ignored, regenerable).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                         # the moss-org tree
TOPO = ROOT / "medt_paper" / "topo_inclusion"
CACHE = HERE / "cache" / "geometry.npz"    # git-ignored, regenerable
PATCHES = ("host_top", "host_sides", "host_base", "inclusion_top",
           "interface_side", "interface_bot")


def _build_fresh():
    sys.path.insert(0, str(TOPO))
    sys.path.insert(0, str(TOPO.parent))
    import make_topo_inclusion as mt
    (meshes, top_flat, top_topo, fault_flat, fault_topo,
     s_hat, bump) = mt.build()
    out = {}
    for k in PATCHES:
        m = top_flat if k == "host_top" else meshes[k]
        out[f"{k}_v"], out[f"{k}_t"] = m.vertices, m.triangles
    out["host_top_topo_v"] = top_topo.vertices
    out["host_top_topo_t"] = top_topo.triangles
    out["fault_v"], out["fault_t"] = fault_flat.vertices, fault_flat.triangles
    out["s_hat"] = np.asarray(s_hat, float)
    out["bump"] = np.array([bump.center_xy[0], bump.center_xy[1], bump.height,
                            bump.sigma, bump.support_radius])
    out["eps"] = np.array(mt.EPS)
    return out


def load(rebuild=False):
    if CACHE.exists() and not rebuild:
        return {k: v for k, v in np.load(CACHE).items()}
    out = _build_fresh()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


if __name__ == "__main__":
    g = load(rebuild="--rebuild" in sys.argv)
    tot = 0
    for k in PATCHES + ("fault",):
        n = len(g[f"{k}_t"])
        tot += n
        print(f"  {k:16s} {n:6d} tri")
    print(f"  {'total (boundary)':16s} {tot - len(g['fault_t']):6d} tri "
          f"-> {3*(tot - len(g['fault_t']))} unknowns")
    print("  s_hat", g["s_hat"], " eps", g["eps"])
