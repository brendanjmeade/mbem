"""eps-ladder: is the amplitude gap against the cached solution driven by
mollification, or is it a formulation error?

Both formulations converge to the same sharp (eps -> 0) answer, but they
mollify DIFFERENT operators -- the direct BIE smooths the double layer, the
force element smooths the single layer -- so their O(eps) errors differ and
need not agree at the eps = 3 km the cache was run at.  If the slope against
the cache moves systematically with eps, the gap is mollification; if it is
flat, it is a defect in the rows.

The comparison mask is held FIXED (6 km from the outcrop rim) at every eps so
the same 3951 centroids are scored throughout.
"""
import sys

import numpy as np

from topo_inclusion import run

EPSILONS = [1.5, 3.0, 6.0]

if __name__ == "__main__":
    eps_list = [float(x) for x in sys.argv[1:]] or EPSILONS
    rows = []
    for e in eps_list:
        print(f"\n{'='*70}\n=== eps = {e} km\n{'='*70}", flush=True)
        st = run(alpha=0.1, eps=e, exclude=6.0, out=f"topo_inclusion_fe_eps{e:g}.npz")
        for k, v in st.items():
            rows.append((e, k, v))
    print(f"\n{'='*70}\n=== eps ladder, 'inclusion only' panel vs cached matching-BC\n{'='*70}")
    print(f"{'eps':>6} {'eps-corr':>9} {'rel_rms':>9} {'corr':>9} {'slope':>9} {'max mm':>9}")
    for e, k, v in rows:
        print(f"{e:6.2f} {str(k)[-5:]:>9} {v['rel_rms']:9.4f} {v['corr']:+9.5f} "
              f"{v['slope']:+9.5f} {v['max_abs']:9.2f}")
