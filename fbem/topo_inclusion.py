"""Force-element solve of the medt_paper topo_inclusion model.

Reproduces the "inclusion only" panel of paper Figure 10 -- the difference
between the heterogeneous and homogeneous solves on the FLAT surface --
with equivalent body forces instead of matching boundary conditions, and
compares it against the cached matching-BC solution.

    u_inc = u(flat, het) - u(flat, hom)

The difference is the right target: it isolates the material contrast, and
each formulation is differenced against its OWN homogeneous state, so the
truncation and free-surface errors common to both states cancel.

    python topo_inclusion.py --check      # conventions + fault sign only
    python topo_inclusion.py              # the full two-solve trial
"""
from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry                                              # noqa: E402
import reference                                             # noqa: E402
from model import (FIXED, FREE, INTERFACE, ForceElementModel,  # noqa: E402
                   Patch, fault_source)

MU_HOST, LAM_HOST = 30.0, 30.0
NU = LAM_HOST / (2.0 * (LAM_HOST + MU_HOST))                 # = 0.25
SLIP_MAG = 0.01                                              # km, mbem's `value`
INC_AXIS = (-100.0, 100.0)
BOX_AXIS = (0.0, 0.0)

PATCH_SPEC = [
    ("host_top",       FREE,      ("up",)),
    ("host_sides",     FREE,      ("radial", BOX_AXIS)),
    ("host_base",      FIXED,     ("down",)),
    ("inclusion_top",  FREE,      ("up",)),
    ("interface_side", INTERFACE, ("radial", INC_AXIS)),
    ("interface_bot",  INTERFACE, ("down",)),
]
OUTPUT = ("host_top", "inclusion_top")


def build(alpha, eps=None, topo=False):
    g = geometry.load()
    eps = float(g["eps"]) if eps is None else eps
    patches = []
    for name, row, orient in PATCH_SPEC:
        key = "host_top_topo" if (name == "host_top" and topo) else name
        patches.append(Patch(name, g[f"{key}_v"], g[f"{key}_t"], row, orient))
    m = ForceElementModel(patches, MU_HOST, NU, eps, alpha)
    m.fault_tv = np.ascontiguousarray(g["fault_v"][g["fault_t"]])
    m.s_hat = g["s_hat"]
    return m


# --------------------------------------------------------------------------
# L0: conventions
# --------------------------------------------------------------------------
def check_normals(m, verbose=True):
    """Every normal must point out of the solid (out of the INCLUSION on the
    interfaces).  Checked by the divergence theorem on both closed surfaces:
    sum(area * n) must vanish, and sum(area * n.x)/3 must be the ENCLOSED
    VOLUME with the right sign."""
    ok = True
    box = m.sel("host_top", "inclusion_top", "host_sides", "host_base")
    inc = m.sel("inclusion_top", "interface_side", "interface_bot")
    exp_inc = np.pi * 75.0 ** 2 * 50.0
    exp_box = 400.0 * 400.0 * 200.0
    for label, idx, expect in (("box", box, exp_box), ("inclusion", inc, exp_inc)):
        a, n, c = m.area[idx], m.normals[idx], m.centroids[idx]
        closure = np.abs((a[:, None] * n).sum(axis=0)).max() / a.sum()
        vol = (a * np.einsum("mi,mi->m", n, c)).sum() / 3.0
        rel = abs(vol - expect) / expect
        good = closure < 1e-10 and rel < 2e-3
        ok &= good
        if verbose:
            print(f"  {label:10s} closure {closure:8.2e}  volume {vol:12.1f} "
                  f"(expect {expect:12.1f}, rel {rel:7.1e})  "
                  f"{'ok' if good else 'FAIL'}")
    if verbose:
        for p in m.patches:
            print(f"    {p.name:16s} {p.n:5d} tri, {p.flipped:5d} winding flips")
    return ok


def check_fault_sign(m, u_F_top, verbose=True):
    """Regress the full-space fault field at the free surface against the
    cached (flat, hom) solution.  A surface-breaking fault in a half space
    gives about twice the full-space displacement, so the slope must be near
    +2; a sign error shows up as -2."""
    ref = reference.load()
    u_ref = np.vstack([ref["u_host_top_flat_hom"], ref["u_inclusion_top_flat_hom"]])
    a, b = u_F_top.ravel(), u_ref.ravel()
    slope = float(a @ b / (a @ a))
    corr = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    good = 1.5 < slope < 2.6 and corr > 0.95
    if verbose:
        print(f"  fault vs cached (flat,hom): slope {slope:+.5f}  "
              f"corr {corr:+.5f}  {'ok' if good else 'FAIL (sign?)'}")
    return good, slope, corr


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(alpha=0.1, eps=None, check_only=False, verbose=True, exclude=6.0,
        out=None):
    t00 = time.time()
    ref = reference.load()
    assert abs(float(ref["mu_inc"]) - 3.0) < 1e-12, (
        f"cache is mu_inc = {float(ref['mu_inc'])}, expected 3.0 (alpha = 0.1); "
        "regenerate with --mu-inc 3.0 or point CACHE elsewhere")

    m = build(alpha, eps=eps)
    print(f"[geometry] {m.N} triangles -> {3*m.N} unknowns, eps = {m.eps} km")
    if not check_normals(m):
        raise SystemExit("normal-orientation gate FAILED")

    out_idx = m.sel(*OUTPUT)
    slip = -SLIP_MAG * np.asarray(m.s_hat, float)        # see fault_source
    print(f"[source] {len(m.fault_tv)} fault triangles, slip {slip} km")
    t0 = time.time()
    fc = pathlib.Path(geometry.CACHE).parent / f"fault_eps{m.eps:g}.npz"
    if fc.exists():
        d = np.load(fc)
        u_F, t_F = d["u_F"], d["t_F"]
        print(f"[source] loaded {fc.name}")
    else:
        u_F, t_F = fault_source(m.centroids, m.normals, m.fault_tv, slip,
                                m.mu, m.nu, m.eps)
        np.savez(fc, u_F=u_F, t_F=t_F, slip=slip)
    print(f"[source] done in {time.time()-t0:.1f}s; "
          f"|u_F|max {np.abs(u_F).max():.4e} km, |t_F|max {np.abs(t_F).max():.4e} GPa")
    good, slope, corr = check_fault_sign(m, u_F[out_idx])
    if not good:
        raise SystemExit("fault-sign gate FAILED")
    if check_only:
        print(f"[check] all conventions ok ({time.time()-t00:.1f}s)")
        return

    # One traction block, two solves.
    B0 = m.traction_block()
    sols = {}
    for label, a in (("het", alpha), ("hom", 1.0)):
        print(f"[{label}] alpha = {a}")
        mm = build(a, eps=eps)
        # .copy() would give C order and lose the in-place LU; keep F.
        mm.assemble(t_F, u_F, B=B0.copy(order="F") if label == "het" else B0)
        mm.solve()
        qi = mm.q[mm.sel("interface_side", "interface_bot")]
        print(f"[{label}] |q| interface max {np.abs(qi).max():.4e}, "
              f"outer max {np.abs(mm.q[mm.sel('host_top')]).max():.4e}")
        for corr_eps in (True, False):
            u = mm.displacement(mm.centroids[out_idx], self_elems=out_idx,
                                correct_eps=corr_eps) + u_F[out_idx]
            sols[(label, corr_eps)] = u
        sols[(label, "q")] = mm.q
        del mm
        gc.collect()
    del B0
    gc.collect()

    # --- comparison
    S = reference.states(ref)
    u_ref = 1e6 * (S[("flat", "het")] - S[("flat", "hom")])      # mm
    cen = reference.centroids(ref)
    rim = np.linalg.norm(cen[:, :2] - np.array(INC_AXIS), axis=1)
    keep = np.abs(rim - 75.0) > exclude                           # outcrop rim
    print(f"\n[compare] 'inclusion only' panel, {keep.sum()}/{len(keep)} "
          f"centroids (excluding |r - 75| < {exclude} km of the outcrop rim)")
    print(f"          reference |u|max {np.abs(u_ref[keep]).max():.1f} mm")
    res = {}
    for corr_eps in (True, False):
        u_fe = 1e6 * (sols[("het", corr_eps)] - sols[("hom", corr_eps)])
        st = reference.compare(u_fe, u_ref, mask=keep)
        a_, b_ = u_fe[keep].ravel(), u_ref[keep].ravel()
        st["slope"] = float(a_ @ b_ / (a_ @ a_))
        st["corr"] = float(a_ @ b_ / np.sqrt((a_ @ a_) * (b_ @ b_)))
        st["label"] = f"eps-corrected={corr_eps}"
        res[corr_eps] = (u_fe, st)
        print(f"  {st['label']:22s} rel_rms {st['rel_rms']:7.4f}  "
              f"corr {st['corr']:+.5f}  slope {st['slope']:+.5f}  "
              f"max {st['max_abs']:8.2f} mm (ref max {st['max_ref']:.1f})")
    np.savez(HERE / (out or "topo_inclusion_fe.npz"),
             centroids=cen, keep=keep, u_ref=u_ref,
             u_fe_corrected=res[True][0], u_fe_raw=res[False][0],
             q_het=sols[("het", "q")], q_hom=sols[("hom", "q")],
             alpha=alpha, eps=m.eps, slope_fault=slope, corr_fault=corr)
    print(f"\n[done] {time.time()-t00:.1f}s")
    return {k: v[1] for k, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    run(alpha=a.alpha, eps=a.eps, check_only=a.check)
