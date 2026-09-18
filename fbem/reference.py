"""Load the cached matching-BC solution and define the comparison metrics."""
import numpy as np

import pathlib

CACHE = (pathlib.Path(__file__).resolve().parent.parent
         / "medt_paper" / "cache" / "topo_inclusion_fields_mu10.npz")


def load(cache=CACHE):
    d = np.load(cache)
    out = {k: d[k] for k in d.files}
    for p in ("host_top", "inclusion_top"):
        v, t = out[f"{p}_vertices"], out[f"{p}_triangles"]
        out[f"{p}_centroids"] = v[t].mean(axis=1)
    return out


def states(ref):
    """The four solved states, as (surface, material) -> u at centroids (m)."""
    return {(s, m): np.vstack([ref[f"u_host_top_{s}_{m}"], ref[f"u_inclusion_top_{s}_{m}"]])
            for s in ("topo", "flat") for m in ("het", "hom")}


def centroids(ref):
    return np.vstack([ref["host_top_centroids"], ref["inclusion_top_centroids"]])


def panels(ref):
    """The four rows of paper Figure 10, in mm."""
    S = states(ref)
    return {
        "full model (topo, het)": 1e6 * S[("topo", "het")],
        "inclusion only": 1e6 * (S[("flat", "het")] - S[("flat", "hom")]),
        "topography only": 1e6 * (S[("topo", "hom")] - S[("flat", "hom")]),
        "topography + inclusion": 1e6 * (S[("topo", "het")] - S[("flat", "hom")]),
    }


def compare(u_new, u_ref, label="", mask=None):
    """Relative RMS and max difference of two (N,3) fields, in the same units."""
    if mask is None:
        mask = np.ones(len(u_ref), bool)
    d = u_new[mask] - u_ref[mask]
    rms = np.sqrt(np.mean(np.sum(d ** 2, axis=1)))
    ref = np.sqrt(np.mean(np.sum(u_ref[mask] ** 2, axis=1)))
    return {"label": label, "rel_rms": rms / ref, "max_abs": np.abs(d).max(),
            "max_ref": np.abs(u_ref[mask]).max(), "n": int(mask.sum())}
