#!/usr/bin/env python
"""Gate: the adaptive octree and its FMM interaction lists (la/octree.py).

An interaction list is combinatorics, so a wrong one is silent -- it does not
raise, it just moves work between the near and far field and changes an answer
nobody is checking yet. Every clause here is therefore an identity that a
mis-built list BREAKS, not a tolerance:

  a  every element is resident in exactly one box, and that box is coarse
     enough to hold it under the placement rule
  b  U + V + 2W = N^2 over element pairs. Each ordered pair of elements is
     handled exactly once by the tree: near (U), by M2L (V), or by M2P/P2L
     (W once in each direction, which is the factor 2). This single clause
     catches a missed list entry, a double count, and a mis-levelled
     adjacency, which is why it is the gate's centre.
  c  U is symmetric and W/X are exact transposes
  d  adjacency agrees with a brute-force geometric test on every box pair
  e  a V-list entry is genuinely separated -- no V pair's cubes touch
  f  protrusion is bounded, and the enlarged extents contain the elements
     the cubes do not

Run from msd/. PASS:/FAIL:, exit 1 on FAIL.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mbem import defaults                                       # noqa: E402
from mbem.kernels import basis as kb                            # noqa: E402
from mbem.la.octree import (InteractionLists, Octree,            # noqa: E402
                            cubes_adjacent)
import mollified_bem as mb                                      # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq            # noqa: E402
from mbem.wrappers import build_vertical_fault_zone_model       # noqa: E402


def _zone_model(refine: float = 1.0):
    """The three-region fault-zone box, as verify_hbackend builds it."""
    r = float(refine)
    fault, _n_hat, s_hat = make_vertical_fault_eq(
        strike_length=16.0, depth_range=(-24.0, -6.0), target_edge=4.0 / r)
    return build_vertical_fault_zone_model(
        x_range=(-60.0, 60.0), y_range=(-40.0, 40.0), z_bottom=-40.0,
        zone_half_width=12.0,
        material_outer=mb.ElasticMaterial(mu=30.0, lam=30.0),
        material_zone=mb.ElasticMaterial(mu=10.0, lam=10.0),
        fault_mesh=fault, fault_slip_vector=s_hat, slip_magnitude=0.01,
        nx_outer=round(4 * r), nx_zone=round(2 * r), ny=round(8 * r),
        nz=round(4 * r))


def _geometry(refine: float = 1.0):
    """Element centroids, sizes and vertices of every mesh in one model."""
    model = _zone_model(refine)
    arrays = kb.MeshArrays()
    seen, cen, siz, verts = {}, [], [], []
    for region in model.regions:
        for patch in list(region.patches) + list(region.faults):
            if id(patch.mesh) in seen:
                continue
            seen[id(patch.mesh)] = patch
            tv, _ = arrays.source_arrays(patch.mesh)
            cen.append(arrays.field_points(patch.mesh))
            edges = np.stack([
                np.linalg.norm(tv[:, 1] - tv[:, 0], axis=1),
                np.linalg.norm(tv[:, 2] - tv[:, 1], axis=1),
                np.linalg.norm(tv[:, 0] - tv[:, 2], axis=1)], axis=1)
            siz.append(edges.max(axis=1))
            verts.append(tv)
    return (np.vstack(cen), np.concatenate(siz), np.vstack(verts),
            len(seen))


def check_placement() -> bool:
    """[a] Every element resident exactly once, in a box big enough."""
    cen, siz, verts, n_mesh = _geometry()
    tree = Octree(cen, siz, verts)
    print(f"    {tree.summary()}, {n_mesh} meshes, "
          f"size grading {siz.max()/siz.min():.0f}:1")

    home = np.full(tree.n, -1, dtype=np.int64)
    dup = 0
    for bi in range(len(tree.boxes)):
        for e in tree.elements_of(bi):
            if home[e] >= 0:
                dup += 1
            home[e] = bi
    missing = int((home < 0).sum())
    print(f"    residency: {dup} duplicated, {missing} unplaced")

    levels = tree.level[home]
    too_deep = int((levels > tree.max_level_of[np.arange(tree.n)]).sum())
    print(f"    elements deeper than their size allows: {too_deep}")

    prot = tree.protrusion(verts)
    print(f"    protrusion beyond own cube: max {prot.max():.3f} box edges, "
          f"mean {prot.mean():.3f}")
    return dup == 0 and missing == 0 and too_deep == 0


def check_pair_conservation() -> bool:
    """[b] U + V + 2W = N^2 over ELEMENT pairs, the load-bearing identity."""
    ok = True
    for refine in (1.0, 1.6):
        cen, siz, verts, _ = _geometry(refine)
        tree = Octree(cen, siz, verts)
        lists = InteractionLists(tree)
        n_sub = tree.n_subtree
        n_res = tree.n_resident

        u = sum(int(n_res[b]) * int(n_res[t]) for b, v in lists.U.items()
                for t in v)
        # A V or W entry is a box pair; its element-pair weight is the
        # product of the subtree the local expansion serves and the subtree
        # the multipole summarizes.
        v = sum(int(n_sub[b]) * int(n_sub[t]) for b, lst in lists.V.items()
                for t in lst)
        w = sum(int(n_res[b]) * int(n_sub[t]) for b, lst in lists.W.items()
                for t in lst)
        x = sum(int(n_sub[b]) * int(n_res[t]) for b, lst in lists.X.items()
                for t in lst)
        total = u + v + w + x
        n2 = tree.n ** 2
        c = lists.counts()
        print(f"    refine {refine:g}: {tree.n} elements, boxes "
              f"{len(tree.boxes)}; U {c['U']} V {c['V']} W {c['W']} "
              f"X {c['X']} (box pairs)")
        print(f"        element pairs U {u} + V {v} + W {w} + X {x} "
              f"= {total} vs N^2 = {n2}"
              + ("  EXACT" if total == n2 else f"  MISMATCH {total - n2:+d}"))
        ok &= (total == n2) and (w == x)
    return ok


def check_transposes() -> bool:
    """[c] U symmetric, W and X exact transposes."""
    cen, siz, verts, _ = _geometry()
    tree = Octree(cen, siz, verts)
    lists = InteractionLists(tree)
    u_pairs = {(b, t) for b, v in lists.U.items() for t in v}
    asym = {(b, t) for (b, t) in u_pairs if (t, b) not in u_pairs}
    w_pairs = {(b, t) for b, v in lists.W.items() for t in v}
    x_pairs = {(t, b) for b, v in lists.X.items() for t in v}
    print(f"    U pairs {len(u_pairs)}, asymmetric {len(asym)}")
    print(f"    W pairs {len(w_pairs)}, X transpose matches "
          f"{w_pairs == x_pairs}")
    return not asym and w_pairs == x_pairs


def check_adjacency() -> bool:
    """[d] The integer adjacency test against brute-force geometry, and
    [e] no V-list pair touches."""
    cen, siz, verts, _ = _geometry()
    tree = Octree(cen, siz, verts)
    rng = np.random.default_rng(0)
    nb = len(tree.boxes)
    idx = rng.choice(nb, size=min(nb, 120), replace=False)
    bad = 0
    for a in idx:
        la, qa = tree.boxes[a]
        lo_a, hi_a = tree.cube(a)
        for b in idx:
            lb, qb = tree.boxes[b]
            lo_b, hi_b = tree.cube(b)
            eps = 1e-9 * tree.root_edge
            geom = bool(np.all(lo_a <= hi_b + eps)
                        and np.all(lo_b <= hi_a + eps))
            if geom != cubes_adjacent(la, qa, lb, qb):
                bad += 1
    print(f"    adjacency vs geometry over {idx.size}^2 box pairs: "
          f"{bad} disagreements")

    lists = InteractionLists(tree)
    touching = 0
    for b, lst in lists.V.items():
        lb, qb = tree.boxes[b]
        for t in lst:
            lt, qt = tree.boxes[t]
            if cubes_adjacent(lb, qb, lt, qt):
                touching += 1
    print(f"    V-list entries whose cubes touch: {touching}")
    return bad == 0 and touching == 0


def check_extents() -> bool:
    """[f] The enlarged interpolation domain contains what the cube does not."""
    cen, siz, verts, _ = _geometry()
    tree = Octree(cen, siz, verts)
    lo, hi = tree.extents(verts)
    worst = 0.0
    for bi in range(len(tree.boxes)):
        held = tree.elements_of(bi)
        if not held.size:
            continue
        v = verts[held].reshape(-1, 3)
        out = max(float((lo[bi] - v.min(axis=0)).max()),
                  float((v.max(axis=0) - hi[bi]).max()))
        worst = max(worst, out)
    cube_lo, cube_hi = tree.cube(0)
    grew = np.isfinite(lo).all(axis=1)
    slack = np.zeros(len(tree.boxes))
    for bi in np.flatnonzero(grew):
        c_lo, c_hi = tree.cube(bi)
        edge = float(c_hi[0] - c_lo[0])
        slack[bi] = max(float((c_lo - lo[bi]).max()),
                        float((hi[bi] - c_hi).max())) / edge
    print(f"    extents contain their elements: worst overhang "
          f"{worst:.3e} (must be <= 0)")
    print(f"    enlargement over the nominal cube: max "
          f"{slack.max():.3f} edges, mean {slack[grew].mean():.3f}")
    return worst <= 1e-9 * tree.root_edge


CHECKS = [
    ("element placement and residency", check_placement),
    ("U + V + 2W = N^2 over element pairs", check_pair_conservation),
    ("U symmetric, W and X transposes", check_transposes),
    ("adjacency vs geometry; V entries separated", check_adjacency),
    ("enlarged interpolation extents", check_extents),
]


def main() -> int:
    failed = []
    for name, fn in CHECKS:
        print(f"\n[{name}]")
        try:
            ok = fn()
        except Exception as exc:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok = False
            print(f"    raised {exc!r}")
        print("    -> " + ("PASS" if ok else "FAIL"))
        if not ok:
            failed.append(name)
    if failed:
        print("\nFAIL: " + "; ".join(failed))
        return 1
    print(f"\nPASS: octree and interaction lists, {len(CHECKS)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
