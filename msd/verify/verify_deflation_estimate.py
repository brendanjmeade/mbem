"""Verify rigid-body deflation (roadmap C4) and the memory estimator (C5).

Deflation: on an ALL-NEUMANN model (every boundary patch free traction)
with jump="calibrated", rigid translations are an EXACT null space of A.
The bordered dense solve and the projected FGMRES solve must (a) return
finite solutions with zero mean translation, (b) agree with each other,
and (c) agree with the classical half-jump solution up to a rigid
translation (the two jump treatments discretize the same physics).

Estimator: dense-mode predictions must match the actual allocations;
the H-mode prediction must land within a small factor of the measured
compressed size (rank is assumed, not computed).

All checks PASS/FAIL.
"""
import pathlib
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import mollified_bem as mb                                        # noqa: E402
from mbem.backends import AssembledH, HBackend                    # noqa: E402
from mbem.backends.dense import (                                 # noqa: E402
    AssembledDense,
    DenseBackend,
    translation_basis,
)
from mbem.estimate import choose_dense_mode, estimate_memory      # noqa: E402
from mbem.model import BCType, Patch, Region, RegionModel         # noqa: E402
from mbem.model import generate_system                            # noqa: E402

EPS = 3.0
MAT = mb.ElasticMaterial(mu=30.0, lam=30.0)


def _all_neumann_model():
    """Fault box with a FREE base: a genuinely all-Neumann model."""
    from _fault_box import build_fault_box
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    fault = meshes["fault"]
    s_hat = np.asarray(meshes["s_hat"], float)
    top = Patch("top", meshes["top"], BCType.FREE_TRACTION)
    sides = Patch("sides", meshes["sides"], BCType.FREE_TRACTION)
    base = Patch("base", meshes["base"], BCType.FREE_TRACTION)
    fpatch = Patch("fault", fault, BCType.FAULT,
                   value=np.broadcast_to(0.01 * s_hat,
                                         (fault.n_triangles, 3)))
    region = Region("crust", MAT, [top, sides, base],
                    probe_point=np.array([45.0, 45.0, -30.0]),
                    faults=[fpatch])
    return RegionModel([region])


def _strip_translation(sol, layout):
    """Remove the mean rigid translation from the u-slots of a solution."""
    out = {}
    means = []
    for name, v in sol.items():
        if name.startswith("u:"):
            means.append(v.mean(axis=0) * v.shape[0])
    tot = sum(s.stop - s.offset for s in layout.slots
              if s.kind == "u") // 3
    mean = np.sum(means, axis=0) / tot
    for name, v in sol.items():
        out[name] = v - mean if name.startswith("u:") else v
    return out


def check_deflation():
    model = _all_neumann_model()
    system = generate_system(model)
    layout = system.layout
    Z = translation_basis(layout)

    # un-deflated calibrated all-Neumann: every entry point must REFUSE (a
    # warning on the front class alone is bypassed by direct construction)
    try:
        DenseBackend("direct", jump="calibrated").assemble(system, EPS)
        refused_d = False
    except ValueError as exc:
        refused_d = "null space" in str(exc)
    try:
        AssembledDense(system, EPS, "direct", jump="calibrated")
        refused_a = False
    except ValueError as exc:
        refused_a = "null space" in str(exc)
    hb = HBackend(eta=0.8, tol=1e-6, jump="calibrated")
    try:
        hb.assemble(system, EPS)
        refused_h = False
    except ValueError as exc:
        refused_h = "null space" in str(exc)
    try:
        AssembledH(system, EPS, hb.opts, False, jump="calibrated")
        refused_ah = False
    except ValueError as exc:
        refused_ah = "null space" in str(exc)
    warned = refused_d and refused_a and refused_h and refused_ah
    print(f"    all-Neumann calibrated REFUSED without deflate: dense front "
          f"{refused_d}, AssembledDense {refused_a}, H front {refused_h}, "
          f"AssembledH {refused_ah}")

    dense = AssembledDense(system, EPS, "direct", jump="calibrated",
                           deflate=True)
    sol_d = dense.solve()
    x_d = np.concatenate([sol_d[s.name].ravel() for s in layout.slots])
    zt = float(np.max(np.abs(Z.T @ x_d)))
    finite = all(np.all(np.isfinite(v)) for v in sol_d.values())
    print(f"    bordered dense: finite={finite}, |Z^T x| = {zt:.2e}")

    hasm = HBackend(eta=0.8, tol=1e-6, jump="calibrated",
                    deflate=True).assemble(system, EPS)
    sol_h = hasm.solve(rtol=1e-9)
    rep = hasm.report
    worst_hd = max(float(np.max(np.abs(sol_h[k] - sol_d[k]))
                         / max(np.max(np.abs(sol_d[k])), 1e-30))
                   for k in sol_d)
    print(f"    projected FGMRES vs bordered dense: rel = {worst_hd:.2e} "
          f"(iters {rep.iterations}, converged {rep.converged})")

    # physics: calibrated+deflated == half-jump up to a rigid translation
    sol_half = AssembledDense(system, EPS, "direct", jump="half").solve()
    a = _strip_translation(sol_d, layout)
    b = _strip_translation(sol_half, layout)
    scale = max(np.max(np.abs(b[k])) for k in b)
    worst_ph = max(float(np.max(np.abs(a[k] - b[k]))) / scale for k in a)
    print(f"    calibrated+deflated vs half-jump (translation-free): "
          f"rel = {worst_ph:.2e}")

    return (warned and finite and zt < 1e-8 and worst_hd < 1e-5
            and rep.converged and worst_ph < 0.05)


def check_estimator():
    from _fault_box import build_fault_box, build_model
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=20.0, fault_depth=18.0,
                             edge_fault=3.0, edge_near=24.0, edge_far=50.0,
                             edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, MAT)
    system = generate_system(model)

    est_d = estimate_memory(system, mode="direct")
    dense = AssembledDense(system, EPS, "direct", jump="half")
    ok = est_d["A_bytes"] == dense.A.nbytes
    print(f"    direct: est A {est_d['A_bytes']/1e6:.1f} MB == actual "
          f"{dense.A.nbytes/1e6:.1f} MB: {ok}")

    est_b = estimate_memory(system, mode="basis")
    basis = AssembledDense(system, EPS, "basis", jump="half")
    actual_basis = sum(v.nbytes() for v in basis._basis.values())
    rel_b = abs(est_b["basis_bytes"] - actual_basis) / actual_basis
    print(f"    basis stacks: est {est_b['basis_bytes']/1e9:.2f} GB vs "
          f"actual {actual_basis/1e9:.2f} GB (rel {rel_b:.2f})")
    ok &= rel_b < 0.01

    est_h = estimate_memory(system, mode="hmat")
    hasm = HBackend(eta=0.8, tol=1e-6).assemble(system, EPS)
    ratio = est_h["total_bytes"] / hasm.nbytes()
    print(f"    hmat: est {est_h['total_bytes']/1e6:.0f} MB vs actual "
          f"{hasm.nbytes()/1e6:.0f} MB (x{ratio:.2f})")
    ok &= 0.3 < ratio < 3.0

    mode = choose_dense_mode(system)
    mode_rb = choose_dense_mode(system, need_rebuild=True)
    print(f"    choose_dense_mode -> {mode!r}; with need_rebuild -> "
          f"{mode_rb!r} (small model)")
    ok &= mode == "direct" and mode_rb == "basis"
    print(f"    fgmres workspace at 3e6 DOFs, restart=200: "
          f"{est_h['fgmres_workspace_bytes'] * 3e6 / est_h['n_unknowns'] / 3 / 1e9:.1f}"
          f" GB-scale guidance available")
    return ok


def main():
    checks = [
        ("rigid-body deflation (all-Neumann + calibrated)", check_deflation),
        ("memory estimator + mode chooser", check_estimator),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: deflation and memory estimation verified.")
    else:
        print("\nFAIL: deflation/estimator check failed.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
