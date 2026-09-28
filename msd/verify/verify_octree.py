#!/usr/bin/env python
"""Gate: the adaptive octree and its FMM interaction lists (la/octree.py).

An interaction list is combinatorics, so a wrong one is silent -- it does not
raise, it just moves work between the near and far field and changes an answer
nobody is checking yet. Every clause here is therefore an identity that a
mis-built list BREAKS, not a tolerance:

  a  every element is resident in exactly one box, and that box is coarse
     enough to hold it under the placement rule
  b  U + V + 2W = N^2 over element pairs, WITH the X rule and without. Each
     ordered pair of elements is handled exactly once by the tree: near (U),
     by M2L (V), or by M2P/P2L (W once in each direction, which is the
     factor 2). This single clause catches a missed list entry, a double
     count, and a mis-levelled adjacency, which is why it is the gate's
     centre -- and it is the only thing that would notice the X rule losing
     or duplicating a pair as it moves one from X into U.
  c  U is symmetric and W/X are exact transposes, of the DOMAIN-BLIND lists
  d  adjacency agrees with a brute-force geometric test on every box pair
  e  a V-list entry is genuinely separated -- no V pair's cubes touch
  f  protrusion is bounded, and the enlarged extents contain the elements
     the cubes do not
  g  the X rule: its two endpoints (threshold 0 reproduces the domain-blind
     lists exactly, threshold infinity empties X into U over exactly the
     pairs X held), what it selects in between, and what that costs

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
from mbem.la.fmm import FmmTree                                 # noqa: E402
from mbem.la.octree import (InteractionLists, Octree,            # noqa: E402
                            XMargin, cubes_adjacent)
import mollified_bem as mb                                      # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq            # noqa: E402
from mbem.wrappers import build_vertical_fault_zone_model       # noqa: E402

# Target interpolation domains the X rule is stated against. They come from
# ``la/fmm.FmmTree`` rather than from a copy here, because the rule reads the
# domain the caller will actually interpolate on and FmmTree is what states
# it; a second implementation of the extent, its half-width floor and the
# inflation would be the same convention written twice.
#
# The plain cube and the contents extent are both clear of the threshold on
# this model (worst X margin 2.20 and 3.00), so a gate on them alone would
# never exercise the rule. INFLATE is verify_fmm [g]'s uniform cube inflation
# -- its containment factor here, 1.667, rounded up -- the configuration that
# pays for containment with the target domain's own margin, which drops to
# 1.32. It is the cheapest geometry in which the rule bites, and it bites
# hard: 89 of 144 entries at refine 1.
INFLATE = 1.67
DOMAINS = (("cube", dict(domain="cube")),
           (f"cube x{INFLATE:g}", dict(domain="cube", inflate=INFLATE)),
           ("extent", {}))
# Refinements of the fault-zone model the list clauses run on. 2.0 is here for
# the X rule alone: at 1.0 and 1.6 every refused entry's target box is a LEAF,
# so the rule's descent never runs and only its U fallback does. At 2.0 the
# 161 refusals on the inflated cube recover 88 deeper X entries and, at a
# larger threshold, cascade -- which is the path the conservation identity has
# to police.
REFINES = (1.0, 1.6, 2.0)
# Placement safety these trees are built at, PINNED rather than read from
# OCTREE_PLACEMENT_SAFETY. The shipping value is chosen on topo_inclusion at
# 31k-261k unknowns (basis beside the constant); this model has 2,592, and on
# it safety 1.5 empties the W and X lists outright and 2.5 leaves U at 100 %.
# A gate that followed the default would stop exercising the lists it exists
# to check, so it states its own and the two are measured separately.
SAFETY = 1.0


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


def _meshes(model) -> list:
    """Every distinct mesh of one model, once, in declaration order."""
    out, seen = [], set()
    for region in model.regions:
        for patch in list(region.patches) + list(region.faults):
            if id(patch.mesh) not in seen:
                seen.add(id(patch.mesh))
                out.append(patch.mesh)
    return out


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
    """[a] Every element resident exactly once, in a box big enough, and
    protrusion bounded by the placement rule rather than by luck.

    Protrusion -- how far an element hangs outside its own box, in box edges
    -- is what the FMM's accuracy turns on (``defaults`` states why beside
    ``OCTREE_PLACEMENT_SAFETY``), and the safety factor's whole claim is the
    sawtooth bound ``prot <= OCTREE_PROTRUSION_C / safety``. The bound is
    checked over a sweep INCLUDING the shipping default, so a default chosen
    on one mesh cannot quietly stop controlling protrusion on another, and
    the sweep is the clause's own because a single value would confirm
    nothing about the rule.
    """
    cen, siz, verts, n_mesh = _geometry()
    tree = Octree(cen, siz, verts, placement_safety=SAFETY)
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

    c_max, bounded = 0.0, True
    safeties = sorted({SAFETY, 1.5, defaults.OCTREE_PLACEMENT_SAFETY, 2.5})
    for s in safeties:
        p = Octree(cen, siz, verts, placement_safety=s).protrusion(verts)
        c = float(p.max()) * s
        c_max = max(c_max, c)
        bounded &= c <= defaults.OCTREE_PROTRUSION_C
        mark = "  <- default" if s == defaults.OCTREE_PLACEMENT_SAFETY else ""
        print(f"    safety {s:<4g} protrusion max {p.max():.3f} <= "
              f"{defaults.OCTREE_PROTRUSION_C / s:.3f} (c = {c:.3f}){mark}")
    print(f"    sawtooth constant c <= {defaults.OCTREE_PROTRUSION_C:g}: "
          f"worst {c_max:.3f}")
    return dup == 0 and missing == 0 and too_deep == 0 and bounded


def _pair_totals(tree, lists) -> dict:
    """Element-pair weight of each list. A V or W entry is a box pair; its
    weight is the product of the subtree the local expansion serves and the
    subtree the multipole summarizes."""
    n_sub, n_res = tree.n_subtree, tree.n_resident
    u = sum(int(n_res[a]) * int(n_res[b]) for a, lst in lists.U.items()
            for b in lst)
    v = sum(int(n_sub[a]) * int(n_sub[b]) for a, lst in lists.V.items()
            for b in lst)
    w = sum(int(n_res[a]) * int(n_sub[b]) for a, lst in lists.W.items()
            for b in lst)
    x = sum(int(n_sub[a]) * int(n_res[b]) for a, lst in lists.X.items()
            for b in lst)
    return {"U": u, "V": v, "W": w, "X": x, "total": u + v + w + x}


def _list_sets(refine: float) -> list:
    """``(label, FmmTree, lists)`` for every list set the identity must hold
    for: the domain-blind traversal, and the X rule on each target
    interpolation domain. All of them share one tree -- the rule moves work
    between lists, never a box."""
    arrays = kb.MeshArrays()
    meshes = _meshes(_zone_model(refine))
    trees = [(label, FmmTree(meshes, arrays=arrays,
                             placement_safety=SAFETY, **kw))
             for label, kw in DOMAINS]
    out = [("no rule", trees[0][1], InteractionLists(trees[0][1].tree))]
    out += [(f"rule {defaults.FMM_X_MARGIN:g} / {label}", gm, gm.lists)
            for label, gm in trees]
    return out


def check_pair_conservation() -> bool:
    """[b] U + V + 2W = N^2 over ELEMENT pairs, the load-bearing identity,
    with the X rule and without.

    A demoted X entry moves into the same traversal's U list and deeper X
    entries, so the identity is exactly as binding with the rule on as off:
    if the rule dropped a pair or emitted one twice, nothing else here would
    see it. ``W == X`` is asserted only with the rule OFF -- the rule is
    one-sided, and clause [c] says why.
    """
    ok = True
    for refine in REFINES:
        for label, gm, lists in _list_sets(refine):
            tree = gm.tree
            t = _pair_totals(tree, lists)
            n2 = tree.n ** 2
            c = lists.counts()
            d = lists.demotions()
            if label == "no rule":
                print(f"    refine {refine:g}: {tree.n} elements, boxes "
                      f"{len(tree.boxes)}")
            print(f"      {label:22s} U {c['U']:5d} V {c['V']:5d} "
                  f"W {c['W']:4d} X {c['X']:4d} box pairs; refused "
                  f"{d['top']:3d} (+{d['entries'] - d['top']} cascaded), "
                  f"{d['top_pairs']} element pairs, {d['direct']} to U")
            print(f"      {'':22s} element pairs U {t['U']} + V {t['V']} "
                  f"+ W {t['W']} + X {t['X']} = {t['total']} vs N^2 = {n2}"
                  + ("  EXACT" if t["total"] == n2
                     else f"  MISMATCH {t['total'] - n2:+d}"))
            ok &= t["total"] == n2
            if label == "no rule":
                ok &= t["W"] == t["X"]
    return ok


def check_transposes() -> bool:
    """[c] U symmetric, W and X exact transposes -- of the DOMAIN-BLIND lists.

    This is a property of the traversal, not of the operator: W is M2P, whose
    target is a field point with no interpolation domain to sit inside, so
    the X rule tests X alone and the moment it bites U stops being symmetric
    and X stops being W's transpose. Clause [g] states what does survive.
    """
    cen, siz, verts, _ = _geometry()
    tree = Octree(cen, siz, verts, placement_safety=SAFETY)
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
    tree = Octree(cen, siz, verts, placement_safety=SAFETY)
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
    tree = Octree(cen, siz, verts, placement_safety=SAFETY)
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


def _subtree_elements(tree) -> list:
    """Per box, the elements resident at or below it -- the rows a local
    expansion built there reaches, and so an X entry's target set."""
    out = [None] * len(tree.boxes)
    for bi in range(len(tree.boxes) - 1, -1, -1):     # children first
        parts = [tree.elements_of(bi)] + [out[c] for c in tree.children[bi]]
        parts = [p for p in parts if p.size]
        out[bi] = (np.concatenate(parts) if parts
                   else np.empty(0, dtype=np.int64))
    return out


def _entries(lst) -> set:
    return {(a, b) for a, v in lst.items() for b in v}


def check_x_rule() -> bool:
    """[g] The X-list admissibility rule: its endpoints, and what it selects.

    The rule is the one place the lists are not domain-blind, so it is gated
    against the domain it is stated on, on each of the three the FMM offers.
    Two endpoints prove the plumbing and nothing else can:

      threshold 0    every rho >= 0, so no entry can be refused and the lists
                     must be the domain-blind ones ENTRY FOR ENTRY -- which
                     is what makes the operator bitwise unchanged
      threshold inf  every entry is refused all the way down, so X empties
                     and the element pairs it held must reappear in U
                     EXACTLY -- the "all X direct" case, which was measured
                     as a control and costs +54.3 % near field

    In between, the domain-blind X list is partitioned: an entry survives iff
    its own margin clears the threshold, and every entry that does not is
    recorded, so nothing is silently dropped. The deeper entries the descent
    recovers are the difference between this rule and moving the whole entry
    direct, and they are counted here.
    """
    ok = True
    tau = defaults.FMM_X_MARGIN
    for refine in REFINES:
        arrays = kb.MeshArrays()
        meshes = _meshes(_zone_model(refine))
        for label, kw in DOMAINS:
            gm = FmmTree(meshes, arrays=arrays,
                         placement_safety=SAFETY, **kw)
            tree, on = gm.tree, gm.lists
            m = XMargin(gm.verts, *gm.tgt_dom, tau)
            base = InteractionLists(tree)
            off = InteractionLists(tree, XMargin(gm.verts, *gm.tgt_dom, 0.0))
            allx = InteractionLists(tree,
                                    XMargin(gm.verts, *gm.tgt_dom, np.inf))

            base_x, on_x = _entries(base.X), _entries(on.X)
            refused = _entries(on.X_demoted)
            keeps = {e for e in base_x if m.rho(tree, *e) >= tau}
            worst = min((m.rho(tree, *e) for e in base_x), default=np.inf)

            # threshold 0: the domain-blind lists, entry for entry.
            no_op = all(dict(getattr(off, k)) == dict(getattr(base, k))
                        for k in ("U", "V", "W", "X")) and not off.X_demoted
            # threshold inf: X empty, and its element pairs now in U.
            sub = _subtree_elements(tree)
            want = np.zeros((tree.n, tree.n), dtype=bool)
            got = np.zeros((tree.n, tree.n), dtype=bool)
            for a, b in _entries(base.U):
                want[np.ix_(tree.elements_of(a), tree.elements_of(b))] = True
            for a, b in base_x:
                want[np.ix_(sub[a], tree.elements_of(b))] = True
            for a, b in _entries(allx.U):
                got[np.ix_(tree.elements_of(a), tree.elements_of(b))] = True
            drained = (not allx.X) and bool((want == got).all())

            # the identity again, at both endpoints, because they are the two
            # list sets [b] does not build and the descent runs hardest at
            # tau = infinity.
            n2 = tree.n ** 2
            ends = {k: _pair_totals(tree, ls)["total"] == n2
                    for k, ls in (("tau=0", off), ("tau=inf", allx))}

            # The domain-blind list is partitioned by the margin: an entry
            # survives iff it clears the threshold, every entry that does not
            # is recorded, and nothing the rule emits fails its own test.
            selection = ((base_x & on_x) == keeps
                         and base_x <= (on_x | refused)
                         and all(m.rho(tree, *e) >= tau for e in on_x)
                         and all(m.rho(tree, *e) < tau for e in refused))

            d = on.demotions()
            tag = f"refine {refine:g} {label}"
            print(f"    {tag:26s}: worst X margin {worst:.4f}; rule refuses "
                  f"{d['top']} of {len(base_x)} entries ({d['top_pairs']} "
                  f"element pairs, "
                  f"{100 * d['top_pairs'] / tree.n ** 2:.4f} % of N^2)")
            print(f"    {'':26s}  {d['direct']} element pairs to U, "
                  f"{len(on_x) - len(base_x & on_x)} deeper X entries "
                  f"recovered, {d['entries'] - d['top']} cascaded; selection "
                  f"{selection}, tau=0 no-op {no_op}, tau=inf drains X "
                  f"{drained}, identity exact at both {all(ends.values())}")
            ok &= selection and no_op and drained and all(ends.values())
    return ok


CHECKS = [
    ("element placement and residency", check_placement),
    ("U + V + 2W = N^2 over element pairs", check_pair_conservation),
    ("U symmetric, W and X transposes", check_transposes),
    ("adjacency vs geometry; V entries separated", check_adjacency),
    ("enlarged interpolation extents", check_extents),
    ("the X-list admissibility rule", check_x_rule),
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
