"""Topography + inclusion showcase: solve the four-state decomposition.

The fig06 soft-inclusion model (host + mu_host/10 cylindrical inclusion
+ surface-breaking strike-slip fault) gains a compactly supported
Gaussian hill ON the fault trace:

    h(x,y) = 2 km * exp(-r^2 / (2 * 30^2)) * S(r),  r about (0, -50),
    S = quintic taper on [75, 90] km, EXACTLY zero beyond 90 km

so the topography never touches the inclusion, and the fault mesh is
warped with the surface it breaks (``build`` enforces the equal-warp rule
on the trace and ``assert_zero_clearance`` the inclusion rim and box
edge). ``--bump-center`` moves the hill off the trace.

Four solves on the SAME refined triangulation (warped vs flat, het vs
homogeneous inclusion), all with the calibrated formulation and the
``eps="auto"`` (0.1 h on every boundary and interface patch, one
0.07 min h on the fault):

    (topo, het)   (topo, hom)   (flat, het)   (flat, hom)

enabling same-connectivity decompositions:
    inclusion effect  Du_inc  = u(topo,het) - u(topo,hom)
    topography effect Du_topo = u(topo,het) - u(flat,het)

``--backend`` picks the operator, on the same system and the same four
states: ``dense`` (LU, the reference; the only one whose report carries a
condition estimate, so ``cond_*`` is saved only for it), ``hmat``
(``HBackend(far="aca")``) or ``fmm`` (``HBackend(far="fmm")``). The
second material state exercises ``rebuild_for_materials`` on whichever
far field was built, which is the one path a single-solve benchmark
never reaches.

Writes fields to examples/topo_inclusion_fields_mu10.npz (the input to
render_topo_inclusion.py) for ``dense``, and to a ``_<backend>``-suffixed
file otherwise, so a cross-backend run never overwrites the reference.

Run from msd/:  python examples/make_topo_inclusion.py [--backend hmat]
"""

import gc
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

from inclusion_mesh import (_circle_boundary, _rectangle_boundary,
                            make_inclusion_geometry)
from local_box_mesh_eq import make_vertical_fault_eq

from mbem.cases.inclusion import MAT_HOST, build_model
from mbem.backends.dense import AssembledDense
from mbem.model import generate_system
from mbem.topography import (apply_topography, assert_zero_clearance,
                             gaussian_bump)

BUMP_CENTER = (0.0, -50.0)     # on the fault trace ("on-fault" hill)
BUMP_HEIGHT = 2.0          # km of relief
BUMP_SIGMA = 30.0          # km
BUMP_EDGE = 9.0            # km, target triangle edge inside the support
FAULT_DEPTH = 20.0         # km (taper depth for fault-conforming warps)
OUT = pathlib.Path(__file__).parent / "topo_inclusion_fields_mu10.npz"


def build(bump_center=BUMP_CENTER, bump_sigma=BUMP_SIGMA,
          bump_height=BUMP_HEIGHT, scale=1.0):
    """fig06 geometry + refinement disk under the bump + the warp.

    If the bump's support reaches the fault trace, the fault mesh is
    warped consistently (same h, depth-tapered) so its top edge conforms
    to the deformed free surface — topography may overlap the fault.
    The inclusion rim and box edges must always stay clear.

    ``scale`` divides every target edge (the same geometry ~scale^2 times
    finer): the mesh ladder of ``bench_scaling.py --model topo_inclusion``.
    """
    s = float(scale)
    bump = gaussian_bump(bump_center, bump_height, bump_sigma,
                         taper=(2.5, 3.0))
    fault_trace = np.array([[0.0, -100.0], [0.0, 100.0]])
    meshes = make_inclusion_geometry(
        x_range=(-200.0, 200.0), y_range=(-200.0, 200.0), z_bottom=-200.0,
        inclusion_center_xy=(-100.0, 100.0), inclusion_radius=75.0,
        inclusion_depth=50.0,
        target_edge_inclusion=6.0 / s, target_edge_top=9.0 / s,
        target_edge_far=80.0 / s, target_edge_side=80.0 / s,
        fault_trace=fault_trace, fault_edge=4.5 / s,
        host_top_max_edge=20.0 / s,
        top_refine_disks=[(bump_center, bump.support_radius, BUMP_EDGE / s,
                           0.5 * (BUMP_EDGE / s) ** 2)])
    fault_mesh, n_hat, s_hat = make_vertical_fault_eq(
        strike_length=200.0, depth_range=(-20.0, 0.0), target_edge=4.5 / s)

    # Hard watertightness guards (curves shared with UNWARPED patches).
    rim, _ = _circle_boundary((-100.0, 100.0), 75.0, 6.0 / s)
    rect = _rectangle_boundary((-200.0, 200.0), (-200.0, 200.0), 9.0 / s)
    for label, pts in (("inclusion rim", rim), ("box edge", rect)):
        assert_zero_clearance(bump, pts, tol=0.0, label=label)

    # Fault: equal-warp rule. If topography reaches the trace, warp the
    # fault mesh with the same h (full at z=0, zero at the fault bottom).
    trace = fault_mesh.vertices[np.isclose(fault_mesh.vertices[:, 2], 0.0),
                                :2]
    h_trace = float(np.abs(bump(trace[:, 0], trace[:, 1])).max())
    if h_trace > 0.0:
        fault_topo = apply_topography(fault_mesh, bump,
                                      taper_depth=FAULT_DEPTH)
        print(f"  bump overlaps fault trace (max h on trace "
              f"{h_trace*1000:.0f} m): fault mesh warped consistently",
              flush=True)
    else:
        fault_topo = fault_mesh

    host_top_flat = meshes["host_top"]
    host_top_topo = apply_topography(host_top_flat, bump)
    return (meshes, host_top_flat, host_top_topo, fault_mesh, fault_topo,
            s_hat, bump)


BACKENDS = ("dense", "hmat", "fmm")


def _assemble(backend: str, system):
    """The chosen far field on this BlockSystem, all three calibrated.

    ``eps="auto"`` and ``jump="calibrated"`` are the model's, not the
    backend's, so the only thing that varies here is how the far field is
    represented -- which is the point of comparing them on this model.
    """
    if backend == "dense":
        return AssembledDense(system, "auto", "direct", jump="calibrated")
    from mbem.backends import HBackend
    return HBackend(jump="calibrated",
                    far="aca" if backend == "hmat" else "fmm"
                    ).assemble(system, "auto")


def _release(asm) -> None:
    """Drop the dense factors before the rebuild allocates its own.

    The compressed backends share their compressed geometry ACROSS the
    rebuild -- that is what makes a material step cheap there -- so they
    have nothing to release, and setting ``A``/``_lu`` on one would create
    two unused attributes rather than free anything.
    """
    if isinstance(asm, AssembledDense):
        asm.A = None
        asm._lu = None
    gc.collect()


def _record(asm, state: str, conds: dict, iters: dict) -> str:
    """Whatever this backend's report actually carries, and a line to print.

    The dense report has a condition estimate and no iteration count; the
    compressed ones have the reverse. Recording only what ran is why the
    saved npz schema differs by backend, and the renderer prints the
    ``cond_*`` keys when they are there.
    """
    rep = asm.report
    if getattr(rep, "cond_estimate", None) is not None:
        conds[state] = rep.cond_estimate
        return f"cond {rep.cond_estimate:.3e}"
    iters[state] = (int(rep.iterations), float(rep.true_relres))
    return (f"{rep.iterations} iterations, true relres {rep.true_relres:.2e}"
            f"{'' if rep.converged else '  NOT CONVERGED'}")


def main(mu_inc: float = 3.0, lam_inc: float | None = None,
         out: pathlib.Path | None = None, bump_center=BUMP_CENTER,
         bump_sigma=BUMP_SIGMA, bump_height=BUMP_HEIGHT,
         backend: str = "dense"):
    import mollified_bem as mb
    if backend not in BACKENDS:
        raise ValueError(backend)
    if out is None:                  # never overwrite the dense reference
        out = OUT if backend == "dense" else OUT.with_name(
            f"{OUT.stem}_{backend}{OUT.suffix}")
    mat_inc = mb.ElasticMaterial(mu=mu_inc,
                                 lam=mu_inc if lam_inc is None else lam_inc)
    (meshes, top_flat, top_topo, fault_flat, fault_topo,
     s_hat, bump) = build(bump_center, bump_sigma, bump_height)
    for k, m in meshes.items():
        print(f"  {k:16s}: {m.n_triangles:5d} tri", flush=True)
    print(f"  bump: H={bump.height} km, sigma={bump.sigma} km, "
          f"support r={bump.support_radius} km at {bump.center_xy}")
    print(f"  inclusion: mu={mat_inc.mu} GPa (host 30)", flush=True)
    print(f"  backend: {backend}", flush=True)

    fields = {}
    conds: dict = {}
    iters: dict = {}
    for surface, host_top, fault_mesh in (("topo", top_topo, fault_topo),
                                          ("flat", top_flat, fault_flat)):
        meshes["host_top"] = host_top
        model = build_model(meshes, fault_mesh, s_hat, mat_inc=mat_inc)
        system = generate_system(model)
        print(f"[{surface}] unknowns: {system.layout.n_unknowns}",
              flush=True)

        t0 = time.time()
        asm = _assemble(backend, system)
        build_s = time.time() - t0
        sol = asm.solve()
        line = _record(asm, f"{surface}_het", conds, iters)
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_{surface}_het"] = sol[f"u:{p}"]
        print(f"[{surface}] het: {line}  (build {build_s:.1f}s, "
              f"total {time.time() - t0:.1f}s)", flush=True)

        _release(asm)
        t0 = time.time()
        asm_h = asm.rebuild_for_materials({"inclusion": MAT_HOST})
        del asm
        gc.collect()
        sol_h = asm_h.solve()
        line = _record(asm_h, f"{surface}_hom", conds, iters)
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_{surface}_hom"] = sol_h[f"u:{p}"]
        print(f"[{surface}] hom: {line}  "
              f"(rebuild + solve {time.time() - t0:.1f}s)", flush=True)
        del asm_h, sol, sol_h, model, system
        gc.collect()

    v = top_flat.vertices
    np.savez(
        out,
        host_top_vertices=v,
        host_top_triangles=top_flat.triangles,
        host_top_h=bump(v[:, 0], v[:, 1]),
        inclusion_top_vertices=meshes["inclusion_top"].vertices,
        inclusion_top_triangles=meshes["inclusion_top"].triangles,
        bump_center=np.array(bump.center_xy),
        bump_height=bump.height, bump_sigma=bump.sigma,
        bump_support=bump.support_radius,
        mu_inc=mat_inc.mu, lam_inc=mat_inc.lam,
        backend=backend,
        **fields,
        **{f"cond_{k}": val for k, val in conds.items()},
        **{f"iters_{k}": np.array(v) for k, v in iters.items()},
    )
    print(f"saved {out}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-inc", type=float, default=3.0,
                    help="inclusion shear modulus, GPa (host is 30; mu/10)")
    ap.add_argument("--lam-inc", type=float, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="default: topo_inclusion_fields_mu10[_BACKEND].npz")
    ap.add_argument("--backend", choices=BACKENDS, default="dense",
                    help="far field: dense LU, flat H + ACA, or bbFMM")
    ap.add_argument("--bump-center", type=float, nargs=2,
                    default=list(BUMP_CENTER), metavar=("X", "Y"))
    ap.add_argument("--bump-sigma", type=float, default=BUMP_SIGMA)
    ap.add_argument("--bump-height", type=float, default=BUMP_HEIGHT)
    a = ap.parse_args()
    main(mu_inc=a.mu_inc, lam_inc=a.lam_inc, out=a.out,
         bump_center=tuple(a.bump_center), bump_sigma=a.bump_sigma,
         bump_height=a.bump_height, backend=a.backend)
