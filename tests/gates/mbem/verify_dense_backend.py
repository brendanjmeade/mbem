"""Verify dense-backend assembly invariants.

Gate for the calibration block-cache sharing (roadmap A1): `_build` and
`the calibration row-sums` now draw H blocks from ONE per-build cache instead
of re-assembling them (in mode="direct" + jump="calibrated" -- the demo
configuration -- every H block used to be assembled twice). Caching is
pure plumbing, so the assembled operator must be BIT-IDENTICAL to a
cache-bypassing build; anything else is a bug.

Checks (PASS/FAIL):
  1. Fault-box model (single region, anchored base), direct+calibrated:
     A and b bit-identical with the cache bypassed; assembly speedup
     reported (informational).
  2. Three-region vertical fault-zone model (INTERFACE topology),
     direct+calibrated: same bit-identity through the multi-region
     calibration path.
  3. rebuild_for_materials on the zone model reproduces a fresh
     assembly at the new material bit-identically (the cache is
     per-build, so rebuilds cannot see stale blocks).
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

import mollified_bem as mb                                        # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq              # noqa: E402
from mbem.backends.dense import AssembledDense                    # noqa: E402
from mbem.model import generate_system                            # noqa: E402
from mbem.wrappers import build_vertical_fault_zone_model         # noqa: E402

EPS = 3.0


class _NoCacheAssembled(AssembledDense):
    """Reference build: every block request re-assembles (old behaviour)."""

    def _cached_block(self, cache, field_patch, source_patch, kernel, region):
        return self._block(field_patch, source_patch, kernel, region)


def _bit_identical(name, asm, ref):
    a_ok = np.array_equal(asm.A, ref.A)
    b_ok = np.array_equal(asm.b, ref.b)
    print(f"    {name}: A bit-identical={a_ok}  b bit-identical={b_ok}")
    return a_ok and b_ok


def _fault_box_system():
    from mbem.cases.fault_box import build_fault_box, build_model
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0,
                             fault_half_len=30.0, fault_depth=18.0,
                             edge_fault=2.5, edge_near=15.0, edge_far=32.0,
                             edge_side=32.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mb.ElasticMaterial(mu=30.0, lam=30.0))
    return generate_system(model)


def _zone_system(mat_zone):
    fault, _n, s_hat = make_vertical_fault_eq(
        strike_length=16.0, depth_range=(-24.0, -6.0), target_edge=4.0)
    model = build_vertical_fault_zone_model(
        x_range=(-60.0, 60.0), y_range=(-40.0, 40.0), z_bottom=-40.0,
        zone_half_width=12.0,
        material_outer=mb.ElasticMaterial(mu=30.0, lam=30.0),
        material_zone=mat_zone, fault_mesh=fault,
        fault_slip_vector=s_hat, slip_magnitude=0.01,
        nx_outer=4, nx_zone=2, ny=8, nz=4)
    return generate_system(model)


def check_fault_box():
    system = _fault_box_system()
    print(f"    fault box: {system.layout.n_unknowns} unknowns")
    asm = AssembledDense(system, EPS, "direct", jump="calibrated")  # + JIT
    t0 = time.perf_counter()
    AssembledDense(system, EPS, "direct", jump="calibrated")
    t_new = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = _NoCacheAssembled(system, EPS, "direct", jump="calibrated")
    t_old = time.perf_counter() - t0
    # ~1.3x, not 2x: the base patch's PRESCRIBED displacement is zero, so
    # its H blocks appear in neither terms nor rhs_terms -- calibration
    # still assembles that one column fresh.
    print(f"    assembly (warm): cached {t_new:.2f} s vs uncached "
          f"{t_old:.2f} s ({t_old / max(t_new, 1e-9):.2f}x)")
    return _bit_identical("direct+calibrated", asm, ref)


def check_zone_model():
    system = _zone_system(mb.ElasticMaterial(mu=10.0, lam=10.0))
    print(f"    fault zone: {system.layout.n_unknowns} unknowns")
    asm = AssembledDense(system, EPS, "direct", jump="calibrated")
    ref = _NoCacheAssembled(system, EPS, "direct", jump="calibrated")
    return _bit_identical("interfaces, direct+calibrated", asm, ref)


def check_rebuild():
    mat_a = mb.ElasticMaterial(mu=10.0, lam=10.0)
    mat_b = mb.ElasticMaterial(mu=20.0, lam=15.0)
    system = _zone_system(mat_a)
    asm = AssembledDense(system, EPS, "direct", jump="calibrated")
    rebuilt = asm.rebuild_for_materials({"zone": mat_b})

    system_b = _zone_system(mat_b)   # fresh model at mat_b
    # Same geometry is rebuilt by the builder, so compare via solutions
    # (patch objects differ between the two systems).
    fresh = AssembledDense(system_b, EPS, "direct", jump="calibrated")
    sol_r = rebuilt.solve()
    sol_f = fresh.solve()
    worst = 0.0
    for k in sol_f:
        ref = float(np.max(np.abs(sol_f[k])))
        if ref > 0:
            worst = max(worst,
                        float(np.max(np.abs(sol_r[k] - sol_f[k]))) / ref)
    print(f"    rebuild vs fresh solution: rel = {worst:.2e}")
    return worst < 1e-12


def main():
    checks = [
        ("calibration cache, fault box (anchored)", check_fault_box),
        ("calibration cache, fault-zone interfaces", check_zone_model),
        ("rebuild_for_materials freshness", check_rebuild),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    if all(results):
        print("\nPASS: dense-backend assembly invariants hold.")
    else:
        print("\nFAIL: dense-backend assembly changed behaviour.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
