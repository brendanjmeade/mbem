"""Verify the numba batched stress assemblers against the scalar oracle.

The numba parallel assemblers ``dd_stress_contract`` (slip/displacement ->
stress) and ``kelvin_stress_contract`` (traction/force -> stress) in
``mbem.kernels.tri_kernels`` must reproduce the scalar analytic oracles
``analytical_stress_kernel`` / ``analytical_kelvin_stress`` (summed over the
triangulated source and contracted with a per-triangle density) to MACHINE
PRECISION -- they integrate the identical mollified moment recursion, just
restructured to be nopython-legal and parallel over observation points.

Three checks, each PASS/FAIL:
  1. dd_stress_contract  == scalar analytical_stress_kernel   (rel < 1e-10)
  2. kelvin_stress_contract == scalar analytical_kelvin_stress (rel < 1e-10)
  3. per-element eps array honored (a non-constant eps reproduces the scalar
     oracle called with that triangle's eps).
Plus a (non-gating) speed comparison numba-parallel vs the scalar loop.
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

from local_box_mesh_eq import make_vertical_fault_eq                 # noqa: E402
from mbem.kernels.tri_kernels import (                               # noqa: E402
    dd_stress_contract,
    kelvin_stress_contract,
)
from mollified_kernel.analytical_kernels import (                    # noqa: E402
    analytical_kelvin_stress,
    analytical_stress_kernel,
)

MU, NU = 30.0, 0.25


def _lam(nu):
    """The test material is stated by nu (the scalar oracles take nu); the
    mbem drivers take (mu, lam)."""
    return 2.0 * MU * nu / (1.0 - 2.0 * nu)


def _scalar(obs_pts, verts, normals, density, eps_arr, kernel):
    """Scalar oracle: sum over triangles, contract with density -> (N,3,3)."""
    sig = np.zeros((len(obs_pts), 3, 3))
    for i, x in enumerate(obs_pts):
        for t in range(len(verts)):
            d = density[t]
            if not d.any():
                continue
            A, B, C = verts[t]
            if kernel == "dd":
                K = analytical_stress_kernel(x, A, B, C, normals[t],
                                             MU, NU, float(eps_arr[t]))
            else:
                K = analytical_kelvin_stress(x, A, B, C,
                                             MU, NU, float(eps_arr[t]))
            sig[i] += K @ d
    return sig


def _relmax(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _geometry(target_edge=4.0):
    fault, _n, _s = make_vertical_fault_eq(
        strike_length=16.0, depth_range=(-8.0, 0.0), target_edge=target_edge)
    verts = np.ascontiguousarray(
        np.asarray(fault.vertices, float)[np.asarray(fault.triangles)])
    normals, _ = fault.normals_and_areas()
    normals = np.ascontiguousarray(np.asarray(normals, float))
    return verts, normals


def _obs_and_density(verts, n_obs=8, seed=0):
    rng = np.random.default_rng(seed)
    nt = verts.shape[0]
    density = rng.normal(size=(nt, 3))
    # a spread of obs points off (and through) the fault plane
    obs = np.column_stack([
        rng.uniform(-6, 6, n_obs),
        rng.uniform(-4, 4, n_obs),
        rng.uniform(-7, -1, n_obs),
    ])
    return np.ascontiguousarray(obs), np.ascontiguousarray(density)


def check_kernel(kernel):
    verts, normals = _geometry()
    obs, density = _obs_and_density(verts)
    eps = np.full(verts.shape[0], 0.5)
    if kernel == "dd":
        num = dd_stress_contract(obs, verts, normals, eps, density, MU, _lam(NU))
    else:
        num = kelvin_stress_contract(obs, verts, eps, density, MU, _lam(NU))
    ref = _scalar(obs, verts, normals, density, eps, kernel)
    err = _relmax(num, ref)
    print(f"    {kernel:>5} numba vs scalar oracle: rel = {err:.2e}")
    return err < 1e-10


def check_per_element_eps():
    verts, normals = _geometry()
    obs, density = _obs_and_density(verts, seed=3)
    rng = np.random.default_rng(7)
    eps = rng.uniform(0.3, 1.2, verts.shape[0])     # non-constant eps
    num = dd_stress_contract(obs, verts, normals, eps, density, MU, _lam(NU))
    ref = _scalar(obs, verts, normals, density, eps, "dd")
    err = _relmax(num, ref)
    print(f"    per-element eps (dd): rel = {err:.2e}")
    return err < 1e-10


def speed_report():
    verts, normals = _geometry(target_edge=3.0)
    obs, density = _obs_and_density(verts, n_obs=60, seed=1)
    eps = np.full(verts.shape[0], 0.5)
    dd_stress_contract(obs[:1], verts, normals, eps, density, MU, _lam(NU))  # warm JIT
    t0 = time.time()
    dd_stress_contract(obs, verts, normals, eps, density, MU, _lam(NU))
    t_numba = time.time() - t0
    t0 = time.time()
    _scalar(obs, verts, normals, density, eps, "dd")
    t_scalar = time.time() - t0
    nt = verts.shape[0]
    print(f"    {len(obs)} obs x {nt} tris: scalar {t_scalar*1e3:7.1f} ms, "
          f"numba {t_numba*1e3:6.1f} ms  ->  {t_scalar/max(t_numba,1e-6):.0f}x")


def check_kernel_at(kernel, nu):
    """Same as check_kernel at another Poisson ratio (lam != mu makes any
    lam/mu index-pairing slip visible; see verify_dd_pairing.py)."""
    global NU
    old = NU
    NU = nu
    try:
        return check_kernel(kernel)
    finally:
        NU = old


def main():
    checks = [
        ("dd_stress_contract vs scalar", lambda: check_kernel("dd")),
        ("kelvin_stress_contract vs scalar", lambda: check_kernel("force")),
        ("dd_stress_contract vs scalar, nu = 0.30", lambda: check_kernel_at("dd", 0.30)),
        ("kelvin_stress_contract vs scalar, nu = 0.30", lambda: check_kernel_at("force", 0.30)),
        ("per-element eps honored", check_per_element_eps),
    ]
    results = []
    for name, fn in checks:
        print(f"\n[{name}]")
        ok = fn()
        results.append(ok)
        print(f"    -> {'PASS' if ok else 'FAIL'}")

    print("\n[speed (informational)]")
    speed_report()

    if all(results):
        print("\nPASS: numba stress assemblers match the scalar oracle.")
    else:
        print("\nFAIL: a numba stress assembler disagrees with the oracle.")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
