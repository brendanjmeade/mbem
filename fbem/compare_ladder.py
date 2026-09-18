"""Matched-eps comparison: force element vs direct BIE at the SAME eps.

The eps-ladder in `eps_ladder.py` compares the force element at several eps
against the cached direct-BIE solution, which is pinned at eps = 3 km.  That
shows only that the FORCE ELEMENT moves with eps -- it cannot distinguish

  (a) the two formulations converge to a common sharp limit, their O(eps)
      errors simply differing in size, from
  (b) the direct BIE is already near the sharp answer at eps = 3 (its
      `jump="calibrated"` free term is designed to remove exactly that O(eps)
      error) while the force element's analytic 1/2 leaves one.

Distinguishing them needs the direct BIE re-solved at the same eps values,
which `regen_reference.py` does.  If the matched-eps slope is flat in eps, the
difference is a genuine formulation gap; if it shrinks with eps, both converge.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SCR = HERE / "cache"     # regenerable solver output, git-ignored
sys.path.insert(0, str(HERE))
import reference                                              # noqa: E402

INC_AXIS = (-100.0, 100.0)
EXCLUDE = 6.0


def stats(u, v, keep):
    """Compare field u against reference v.

    ``amp`` is the honest amplitude ratio ||u|| / ||v||: < 1 means u is too
    SMALL.  ``slope`` is the least-squares fit of v onto u (v ~ slope * u), so
    slope > 1 also means u is too small -- the opposite reading to the naive
    one, which is exactly why ``amp`` is reported next to it.
    """
    a, b = u[keep].ravel(), v[keep].ravel()
    d = u[keep] - v[keep]
    return dict(
        rel_rms=float(np.sqrt((d ** 2).sum(1).mean()) /
                      np.sqrt((v[keep] ** 2).sum(1).mean())),
        corr=float(a @ b / np.sqrt((a @ a) * (b @ b))),
        slope=float(a @ b / (a @ a)),
        amp=float(np.sqrt((a @ a) / (b @ b))),
        max_abs=float(np.abs(d).max()),
    )


def main():
    R = np.load(SCR / "reference_regen.npz")
    cen = reference.centroids(reference.load())
    rim = np.linalg.norm(cen[:, :2] - np.array(INC_AXIS), axis=1)
    keep = np.abs(rim - 75.0) > EXCLUDE

    eps_bie = sorted(float(str(k).split("_eps")[1])
                     for k in R.files if k.startswith("u_host_top_flat_het_eps"))
    print(f"direct BIE solved at eps = {eps_bie}")
    print(f"{keep.sum()}/{len(keep)} centroids scored "
          f"(excluding |r-75| < {EXCLUDE} km of the outcrop rim)\n")

    # The direct BIE's own eps-dependence, referenced to its smallest eps.
    ref_small = None
    print("== direct BIE 'inclusion only' panel, its own eps-dependence ==")
    print(f"{'eps':>6} {'|u|max mm':>10} {'slope vs eps_min':>18} {'rel_rms':>9}")
    bie = {}
    for e in eps_bie:
        u = 1e6 * np.vstack([R[f"u_host_top_flat_het_eps{e:g}"],
                             R[f"u_inclusion_top_flat_het_eps{e:g}"]]) \
            - 1e6 * np.vstack([R[f"u_host_top_flat_hom_eps{e:g}"],
                               R[f"u_inclusion_top_flat_hom_eps{e:g}"]])
        bie[e] = u
        if ref_small is None or e == min(eps_bie):
            ref_small = bie[min(eps_bie)] if min(eps_bie) in bie else None
    small = bie[min(eps_bie)]
    for e in eps_bie:
        s = stats(bie[e], small, keep)
        print(f"{e:6.2f} {np.abs(bie[e][keep]).max():10.1f} "
              f"{s['slope']:18.5f} {s['rel_rms']:9.4f}")

    print("\n== force element vs direct BIE at the SAME eps ==")
    print(f"{'eps':>6} {'eps-corr':>9} {'amp FE/BIE':>11} {'rel_rms':>9} "
          f"{'corr':>9} {'slope':>9} {'max mm':>9} {'BIE max':>9}")
    rows = []
    for e in eps_bie:
        f = SCR / f"topo_inclusion_fe_eps{e:g}.npz"
        if not f.exists():
            print(f"{e:6.2f}   (no force-element run at this eps)")
            continue
        F = np.load(f)
        for key, tag in (("u_fe_corrected", "True"), ("u_fe_raw", "False")):
            s = stats(F[key], bie[e], keep)
            rows.append((e, tag, s))
            print(f"{e:6.2f} {tag:>9} {s['amp']:11.5f} {s['rel_rms']:9.4f} "
                  f"{s['corr']:+9.5f} {s['slope']:+9.5f} {s['max_abs']:9.2f} "
                  f"{np.abs(bie[e][keep]).max():9.1f}")

    # Richardson in eps, on the matched-eps slopes.
    print("\n  amp < 1 means the FORCE ELEMENT is too small.")
    for tag in ("True", "False"):
        r = [(e, 1.0 / s["amp"]) for e, t, s in rows if t == tag]
        if len(r) >= 2:
            r.sort()
            (e1, s1), (e2, s2) = r[0], r[1]
            p = np.log(abs(s2 - 1) / abs(s1 - 1)) / np.log(e2 / e1) \
                if abs(s1 - 1) > 0 else float("nan")
            rich = s1 - (s2 - s1) * e1 / (e2 - e1)
            print(f"  eps-corrected={tag}: (BIE/FE amplitude)-1 scales as "
                  f"eps^{p:.2f}; linear extrapolation to eps=0 gives "
                  f"{rich:.5f} (1.0 = exact agreement)")


if __name__ == "__main__":
    main()
