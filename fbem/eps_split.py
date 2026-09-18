"""Which surfaces carry the force element's O(eps) error?

`eps` is per source element, so the mollification can be varied on the
INTERFACE (the inclusion's boundary, where the material contrast lives) and on
the OUTER boundary (free surface, sides, base) independently.  If the
eps-sensitivity of the "inclusion only" panel sits entirely in the interface,
the error is in how the mollified single layer represents a curved material
jump; if it sits in the outer boundary, it is a free-surface/corner artifact
that the het - hom difference was supposed to cancel.

Sweeps eps_interface at a fixed production eps_outer = 3.0 (edit the loop at
the bottom for the 2x2 eps_int x eps_out factorial reported in RESULTS.md).
"""
from __future__ import annotations

import gc
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry                                              # noqa: E402
import reference                                             # noqa: E402
from model import fault_source                               # noqa: E402
from topo_inclusion import INC_AXIS, OUTPUT, build           # noqa: E402

INTERFACE_PATCHES = ("interface_side", "interface_bot")
EXCLUDE = 6.0


def solve_at(eps_int, eps_out, u_F, t_F, out_idx):
    """Two solves (het, hom) with a split eps, returning the panel in mm."""
    sols = {}
    B0 = None
    for label, a in (("het", 0.1), ("hom", 1.0)):
        m = build(a, eps=eps_out)
        m.eps_arr[m.sel(*INTERFACE_PATCHES)] = eps_int
        m.eps = eps_out                        # used only by the (eps/4) term
        if B0 is None:
            B0 = m.traction_block(verbose=False)
            B = B0.copy(order="F")
        else:
            B = B0
        m.assemble(t_F, u_F, B=B, verbose=False)
        m.solve(verbose=False)
        sols[label] = m.displacement(m.centroids[out_idx], self_elems=out_idx,
                                     correct_eps=True) + u_F[out_idx]
        cond = m.cond_estimate
        del m
        gc.collect()
    del B0
    gc.collect()
    return 1e6 * (sols["het"] - sols["hom"]), cond


def main():
    ref = reference.load()
    R = np.load(pathlib.Path(geometry.CACHE).parent / "reference_regen.npz")
    cen = reference.centroids(ref)
    rim = np.linalg.norm(cen[:, :2] - np.array(INC_AXIS), axis=1)
    keep = np.abs(rim - 75.0) > EXCLUDE

    m0 = build(0.1, eps=3.0)
    out_idx = m0.sel(*OUTPUT)
    del m0

    print(f"{'eps_int':>8} {'eps_out':>8} {'|panel|max mm':>14} {'slope vs BIE':>13} "
          f"{'rel_rms':>9} {'cond':>10}")
    for eps_int in (0.1875, 0.375, 0.75, 1.5, 3.0, 6.0):
        for eps_out in (3.0,):
            fc = pathlib.Path(geometry.CACHE).parent / f"fault_eps{eps_out:g}.npz"
            d = np.load(fc)
            u_F, t_F = d["u_F"], d["t_F"]
            panel, cond = solve_at(eps_int, eps_out, u_F, t_F, out_idx)
            # score against the direct BIE at the OUTER eps (the fault and the
            # free surface are what that eps controls)
            bie = 1e6 * (np.vstack([R[f"u_host_top_flat_het_eps{eps_out:g}"],
                                    R[f"u_inclusion_top_flat_het_eps{eps_out:g}"]])
                         - np.vstack([R[f"u_host_top_flat_hom_eps{eps_out:g}"],
                                      R[f"u_inclusion_top_flat_hom_eps{eps_out:g}"]]))
            a, b = panel[keep].ravel(), bie[keep].ravel()
            d_ = panel[keep] - bie[keep]
            print(f"{eps_int:8.2f} {eps_out:8.2f} {np.abs(panel[keep]).max():14.1f} "
                  f"{a @ b / (a @ a):13.5f} "
                  f"{np.sqrt((d_**2).sum(1).mean())/np.sqrt((bie[keep]**2).sum(1).mean()):9.4f} "
                  f"{cond:10.2e}", flush=True)


if __name__ == "__main__":
    main()
