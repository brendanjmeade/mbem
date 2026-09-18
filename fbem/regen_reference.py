"""Re-solve the matching-BC (direct BIE) FLAT pair with the CURRENT medt_paper
kernels, so the force-element comparison is against a freshly computed
reference rather than a cache of unknown vintage.

The cache `medt_paper/cache/topo_inclusion_fields_mu10.npz` is dated
2026-09-11; `medt_paper/topo_inclusion/mbem/kernels/{tri_kernels,basis}.py`
and `mbem/defaults.py` were rewritten on 2026-09-17 (the lam/mu
traction-pairing port).  That port is documented as invisible at nu = 1/4,
which is exactly this model -- but "documented as invisible" is not a
measurement, so this script makes one.

Only the FLAT pair (het, hom) is recomputed: that is the "inclusion only"
panel, the target of the force-element trial.  Nothing in medt_paper is
modified; its modules are imported read-only.
"""
from __future__ import annotations

import gc
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TOPO = ROOT / "medt_paper" / "topo_inclusion"
OUT = HERE / "cache" / "reference_regen.npz"
MU_INC = 3.0


def main(eps_list=(3.0,)):
    sys.path.insert(0, str(TOPO))
    sys.path.insert(0, str(TOPO.parent))
    import mollified_bem as mb
    import make_topo_inclusion as mt
    from assess_fig06_inclusion import MAT_HOST, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system

    mat_inc = mb.ElasticMaterial(mu=MU_INC, lam=MU_INC)
    (meshes, top_flat, top_topo, fault_flat, fault_topo,
     s_hat, bump) = mt.build()
    meshes["host_top"] = top_flat                      # FLAT pair only

    fields = {}
    for eps in eps_list:
        print(f"\n[regen] EPS = {eps}, mu_inc = {MU_INC} (host {MAT_HOST.mu})",
              flush=True)
        model = build_model(meshes, fault_flat, s_hat, mat_inc=mat_inc)
        system = generate_system(model)
        print(f"[regen] unknowns {system.layout.n_unknowns}", flush=True)
        t0 = time.time()
        asm = AssembledDense(system, eps, "direct", jump="calibrated")
        sol = asm.solve()
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_flat_het_eps{eps:g}"] = sol[f"u:{p}"]
        print(f"[regen] het done {time.time()-t0:.0f}s, "
              f"cond {asm.cond_estimate:.3e}", flush=True)

        asm.A = None
        asm._lu = None
        gc.collect()
        t0 = time.time()
        asm_h = asm.rebuild_for_materials({"inclusion": MAT_HOST})
        del asm
        gc.collect()
        sol_h = asm_h.solve()
        for p in ("host_top", "inclusion_top"):
            fields[f"u_{p}_flat_hom_eps{eps:g}"] = sol_h[f"u:{p}"]
        print(f"[regen] hom done {time.time()-t0:.0f}s, "
              f"cond {asm_h.cond_estimate:.3e}", flush=True)
        del asm_h, sol, sol_h, model, system
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, mu_inc=MU_INC, eps=np.array(list(eps_list)),
             host_top_vertices=top_flat.vertices,
             host_top_triangles=top_flat.triangles,
             inclusion_top_vertices=meshes["inclusion_top"].vertices,
             inclusion_top_triangles=meshes["inclusion_top"].triangles,
             **fields)
    print(f"[regen] saved {OUT}")

    # --- compare the eps = 3 pair against the Sep-11 cache
    import reference
    old = reference.load()
    worst = 0.0
    for k in fields:
        if not k.endswith("_eps3"):
            continue
        a, b = fields[k], old[k[:-5]]
        rel = np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)
        worst = max(worst, rel)
        print(f"  {k:32s} max|new-old| {np.abs(a-b).max():.4e} km  rel {rel:.3e}")
    print(f"\n[regen] worst relative change vs the Sep-11 cache: {worst:.3e}")
    print("        -> cache is CURRENT" if worst < 1e-10 else
          "        -> cache is STALE; use reference_regen.npz")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(tuple(float(x) for x in sys.argv[1:]) or (3.0,))
