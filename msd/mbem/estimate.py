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
from .kernels import kernel_n_basis
from .kernels import basis as kb
from .la.cluster import build_cluster_tree, build_partition
from .la.hop import require_order0

# Per-basis ACA rank assumed for every admissible block, at
# BLOCK_COMPRESSION_TOL: a far-field T block holds 16 at 1e-4 (42 at
# 1e-8); a 10k-element plate self-pair averages 9-11 over its blocks.
ASSUMED_BASIS_RANK = 16
# What storage="basis" holds on top of the material view: one shared
# subspace per block (la/aca.shared_subspace), whose rank is this
# fraction of the B per-basis ranks it replaces -- 0.41 measured on
# 1024-element far-field T blocks, 0.59 on U blocks of the same pair, so
# the geometry-only payload is ~2.5x one T view rather than 6x.
ASSUMED_JOINT_FRACTION = 0.41


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


def _pair_keys(system, rhs: bool = True):
    keys = {}
    terms = list(system.terms) + (list(system.rhs_terms) if rhs else [])
    for t in terms:
        keys[(id(t.field_patch), id(t.source_patch), t.kernel)] = \
            (t.field_patch, t.source_patch, t.kernel)
    return keys


def estimate_memory(system, mode: str = "direct",
                    storage: str = "combined",
                    sweep: bool = False,
                    min_leaf: int = defaults.CLUSTER_MIN_LEAF,
                    eta: float = defaults.ADMISSIBILITY_ETA,
                    max_admissible: int = defaults.MAX_ADMISSIBLE_BLOCK,
                    min_aca: int = defaults.ACA_MIN_BLOCK) -> dict:
    """Predicted peak bytes for assembling ``system``.

    mode: "direct" | "basis" (dense backends) | "hmat". ``storage`` and
    ``sweep`` mirror ``HBackend``'s (combined payloads; RHS pairs built
    only for a declared material sweep, else applied matrix-free).
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
                B = kernel_n_basis(kern)
                basis_bytes += B * (3 * fp.n_nodes) * (3 * sp.n_nodes) * 8
        out["basis_bytes"] = basis_bytes
        out["total_bytes"] = a_bytes * 2 + basis_bytes
    elif mode == "hmat":
        require_order0(system.model)     # the trees below are on centroids:
        arrays = kb.MeshArrays()         # a P1/P2 model would get P0's number
        tree_cache: dict = {}

        def _tree(mesh):
            t = tree_cache.get(id(mesh))
            if t is None:
                t = build_cluster_tree(arrays.field_points(mesh), min_leaf)
                tree_cache[id(mesh)] = t
            return t

        dense_bytes = 0
        lowrank_bytes = 0
        for fp, sp, kern in _pair_keys(system, rhs=sweep).values():
            B = kernel_n_basis(kern)
            part = build_partition(_tree(fp.mesh), _tree(sp.mesh),
                                   eta=eta, max_admissible=max_admissible,
                                   min_aca=min_aca)
            # The near field is per material in BOTH storage modes (it is
            # re-evaluated from the kernels, never stored per basis).
            for rows, cols in part.dense:
                dense_bytes += (3 * len(rows)) * (3 * len(cols)) * 8
            for rows, cols in part.admissible:
                k = min(ASSUMED_BASIS_RANK, len(rows), len(cols))
                lowrank_bytes += 3 * (len(rows) + len(cols)) * k * 8
                if storage == "basis":      # + the block's shared subspace
                    kj = min(int(ASSUMED_JOINT_FRACTION * B * k) + 1,
                             3 * len(rows), 3 * len(cols))
                    lowrank_bytes += (3 * (len(rows) + len(cols)) * kj
                                      + B * kj * kj) * 8
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


def choose_dense_mode(system, budget_fraction: float = 0.5,
                      need_rebuild: bool = False) -> str:
    """"direct" (memory-light, the default everywhere) unless the caller
    needs cheap material rebuilds (``need_rebuild``) AND the "basis" stacks
    fit within ``budget_fraction`` of RAM (16 GB when RAM is unknown); a
    rebuild that does not fit falls back to "direct" with a warning."""
    if not need_rebuild:
        return "direct"
    est = estimate_memory(system, mode="basis")
    ram = total_ram_bytes()
    limit = budget_fraction * ram if ram else 16e9
    if est["total_bytes"] <= limit:
        return "basis"
    import warnings
    warnings.warn(
        f"basis-mode stacks would need {est['total_bytes']/1e9:.1f} GB "
        f"(> {limit/1e9:.1f} GB budget); using mode='direct' (material "
        f"rebuilds will re-assemble)")
    return "direct"
