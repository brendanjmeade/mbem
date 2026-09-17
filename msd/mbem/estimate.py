"""Memory estimation for a BlockSystem before assembling it.

``estimate_memory(system, ...)`` predicts the dominant allocations of
each backend/mode from mesh sizes alone (dense modes are exact; the H
estimate builds the real cluster partitions -- cheap -- and assumes a
per-basis rank for admissible blocks), so callers can pick a backend
before committing tens of GB. ``fgmres_workspace_bytes`` covers the
Krylov basis (V and Z, (restart+1)+restart vectors), which at 3e6 DOFs
and the default restart=200 is ~10 GB -- lower ``restart`` for very
large models.
"""

from __future__ import annotations

import numpy as np

from . import defaults
from .kernels import basis as kb
from .la.cluster import build_cluster_tree, build_partition

ASSUMED_BASIS_RANK = 40      # typical measured per-basis ACA rank


def total_ram_bytes() -> int | None:
    """Physical RAM, or None if the platform will not say."""
    try:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def fgmres_workspace_bytes(n_unknowns: int,
                           restart: int = defaults.GMRES_RESTART) -> int:
    return (2 * restart + 1) * n_unknowns * 8


def _pair_keys(system):
    keys = {}
    for t in list(system.terms) + list(system.rhs_terms):
        keys[(id(t.field_patch), id(t.source_patch), t.kernel)] = \
            (t.field_patch, t.source_patch, t.kernel)
    return keys


def estimate_memory(system, mode: str = "direct",
                    storage: str = "basis",
                    min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                    eta: float = defaults.ADMISSIBILITY_ETA,
                    max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK) -> dict:
    """Predicted peak bytes for assembling ``system``.

    mode: "direct" | "basis" (dense backends) | "hmat".
    Returns a dict with ``total_bytes`` plus a breakdown, and
    ``fraction_of_ram`` when the platform reports RAM.
    """
    n = system.layout.n_unknowns
    out: dict = {"mode": mode, "n_unknowns": n}

    if mode in ("direct", "basis"):
        a_bytes = n * n * 8
        out["A_bytes"] = a_bytes
        out["lu_bytes"] = a_bytes            # lu_factor copies A
        basis_bytes = 0
        if mode == "basis":
            for fp, sp, kern in _pair_keys(system).values():
                B = 6 if kern == "H" else 3
                basis_bytes += B * (3 * fp.n_triangles) \
                    * (3 * sp.n_triangles) * 8
        out["basis_bytes"] = basis_bytes
        out["total_bytes"] = a_bytes * 2 + basis_bytes
    elif mode == "hmat":
        arrays = kb.MeshArrays()
        tree_cache: dict = {}

        def _tree(mesh):
            t = tree_cache.get(id(mesh))
            if t is None:
                t = build_cluster_tree(arrays.field_points(mesh), min_leaf)
                tree_cache[id(mesh)] = t
            return t

        dense_bytes = 0
        lowrank_bytes = 0
        for fp, sp, kern in _pair_keys(system).values():
            B = 6 if kern == "H" else 3
            mult = B if storage == "basis" else 1
            part = build_partition(_tree(fp.mesh), _tree(sp.mesh),
                                   eta=eta, max_admissible=max_admissible)
            for rows, cols in part.dense:
                dense_bytes += mult * (3 * len(rows)) * (3 * len(cols)) * 8
            for rows, cols in part.admissible:
                k = min(ASSUMED_BASIS_RANK, len(rows), len(cols))
                lowrank_bytes += mult * 3 * (len(rows) + len(cols)) * k * 8
        out["dense_leaf_bytes"] = dense_bytes
        out["lowrank_bytes"] = lowrank_bytes
        out["storage"] = storage
        out["total_bytes"] = dense_bytes + lowrank_bytes
    else:
        raise ValueError(mode)

    out["fgmres_workspace_bytes"] = fgmres_workspace_bytes(n)
    ram = total_ram_bytes()
    if ram:
        out["fraction_of_ram"] = out["total_bytes"] / ram
    return out


def choose_dense_mode(system, budget_fraction: float = 0.5) -> str:
    """"basis" if its stacks fit within ``budget_fraction`` of RAM
    (cheap material rebuilds), else "direct" (memory-light). Falls back
    to "direct" when RAM cannot be determined and the stacks exceed
    16 GB."""
    est = estimate_memory(system, mode="basis")
    ram = total_ram_bytes()
    limit = budget_fraction * ram if ram else 16e9
    if est["total_bytes"] <= limit:
        return "basis"
    import warnings
    warnings.warn(
        f"basis-mode stacks would need {est['total_bytes']/1e9:.1f} GB "
        f"(> {limit/1e9:.1f} GB budget); using mode='direct'")
    return "direct"
