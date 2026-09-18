"""Gate: the convergence study's HARNESS (``../convergence.py``), not its result.

    python verify/verify_convergence.py          # from the ddbem root

A convergence study is the one kind of code where a bug is invisible in the
output: every number it prints is "about what you would expect", and a rate is
a two-point slope of two numbers nobody can check by eye.  So the harness is
gated the way the kernels are -- against independent statements, with tripwires
that must break when the checked thing is broken.

WHAT IS PINNED, AND WHY IT IS NOT VACUOUS
=========================================
1. THE EXACT SOLUTION IS A SOLUTION, AND IT IS EXTERIOR.  ``div sigma`` of the
   Kelvin field is zero at the observation cloud (central differences of the
   reference's own stress), and the point force sits OUTSIDE both domains by a
   margin -- if it were inside, no boundary-only representation could reproduce
   the field and every "rate" below would be measuring the wrong thing.

2. THE RATE FIT.  ``fit_rate`` recovers a planted exponent to machine precision
   on synthetic data, is invariant to the units of x and y (a rate that moved
   when h was measured in metres would be a real bug), and returns NaN rather
   than a number when it has fewer than two usable points.

3. THE ERROR MEASURES.  Fed the EXACT field they return 0; fed the exact field
   plus a rigid motion, the Neumann (rigid-removed) measure still returns ~0
   while the Dirichlet measure returns the size of the rigid motion.  Both
   halves matter: the first says the measure is not blind, the second says the
   rigid removal is doing what the interior Neumann problem requires and not
   quietly absorbing real error (checked by planting a NON-rigid perturbation,
   which must survive).

4. THE CUBE EDGE CLASSIFIER.  ``cube_edge_distance`` against hand-computed
   distances at a corner, an edge midpoint, a face centre and an interior-face
   point.  The edge-concentration statistic is the study's only edge-specific
   measure, so a wrong classifier would silently answer the study's central
   question with noise.

5. THE HARNESS IS THE SOLVER.  ``solve_ddbem`` is re-run inline, model built by
   hand, and must agree to machine precision -- i.e. the study adds nothing to
   the solve it is measuring.

6. THE msd ANCHOR SOLVES THE SAME PROBLEM.  msd's frozen P0 solver on the same
   mesh and data converges (20 -> 80 triangles) and lands within a factor of a
   few of ddbem P0.  It is NOT gated to agree entrywise: they are different
   formulations (``verify/verify_msd_parity.py`` is the entrywise gate, on the
   one problem where they coincide).

7. THE CACHE.  A record round-trips through JSONL unchanged, and a re-run of an
   already-cached case is skipped rather than recomputed (the study is hours
   long; a resume that silently recomputed or silently duplicated would corrupt
   a ladder).

TOLERANCES.  1e-12..1e-14 everywhere an exact identity is claimed.  The two
physics checks (6) carry explicit ceilings, printed next to the value.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from _common import Report                                       # noqa: E402
from _exact import kelvin_displacement, kelvin_stress            # noqa: E402

import ddbem                                                     # noqa: E402
import convergence as C                                          # noqa: E402


def gate_exact_solution(rep):
    """The planted exact solution is a homogeneous solution, and is exterior."""
    # div sigma = 0 at the observation cloud (central differences)
    step = 1e-5
    div = np.zeros((C.OBS.shape[0], 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = step
        sp = kelvin_stress(C.OBS + d, C.X0, C.FORCE, C.MU, C.NU)
        sm = kelvin_stress(C.OBS - d, C.X0, C.FORCE, C.MU, C.NU)
        div += (sp[:, :, j] - sm[:, :, j]) / (2 * step)
    scale = np.max(np.abs(kelvin_stress(C.OBS, C.X0, C.FORCE, C.MU, C.NU)))
    rep.check("exact field: div sigma = 0 on the observation cloud",
              float(np.max(np.abs(div))) / scale, 1e-6)

    # the point force is OUTSIDE both domains, with margin
    for geom, level in (("sphere", 1), ("cube", 2)):
        tv, _, _ = C.geometry(geom, level)
        v = tv.reshape(-1, 3)
        d = float(np.min(np.linalg.norm(v - C.X0, axis=1)))
        h = float(ddbem.element_h(tv).mean())
        rep.check_bool(f"{geom}: point force is outside, {d / h:.2f} h clear of "
                       f"the surface", d > 1.5 * h, f"d = {d:.3f}, h = {h:.3f}")
        # ... and the observation cloud is well inside
        dmin = float(np.min(np.linalg.norm(
            C.OBS[:, None, :] - v[None, :, :], axis=2)))
        rep.check_bool(f"{geom}: observation cloud is {dmin:.2f} from the "
                       f"surface ({dmin / h:.2f} h)", dmin > 0.6,
                       "(the representation is mesh-limited within ~1 h)")


def gate_rate_fit(rep):
    """A planted exponent must come back exactly, in any units."""
    rng = np.random.default_rng(3)
    for slope in (-1.7, 0.9, 2.0):
        x = np.array([1.0, 0.5, 0.25, 0.125])
        y = 3.1 * x ** slope
        rep.check(f"fit_rate recovers a planted exponent {slope:+.1f}",
                  abs(C.fit_rate(x, y) - slope), 1e-12)
        rep.check(f"  and is invariant to the units of x and y ({slope:+.1f})",
                  abs(C.fit_rate(1000.0 * x, 1e-6 * y) - slope), 1e-12)
    noisy = 3.1 * np.array([1.0, 0.5, 0.25, 0.125]) ** -1.7 * (
        1.0 + 0.01 * rng.normal(size=4))
    rep.check_bool("  a 1 % perturbation moves the fitted rate by < 0.05",
                   abs(C.fit_rate([1.0, .5, .25, .125], noisy) + 1.7) < 0.05,
                   f"(rate {C.fit_rate([1.0, .5, .25, .125], noisy):+.4f})")
    rep.check_bool("  fewer than 2 usable points -> NaN, never a number",
                   np.isnan(C.fit_rate([1.0], [2.0]))
                   and np.isnan(C.fit_rate([1.0, 2.0], [0.0, -1.0])))


def gate_error_measures(rep):
    """Zero on the exact field; rigid-blind only where it must be."""
    ue = kelvin_displacement(C.OBS, C.X0, C.FORCE, C.MU, C.NU)
    se = ddbem.tensor_to_voigt(kelvin_stress(C.OBS, C.X0, C.FORCE, C.MU, C.NU))
    for bc in ("dirichlet", "neumann"):
        eu, eu2, es = C.field_errors(ue, se, bc)
        rep.check(f"field_errors({bc}) is 0 on the exact field",
                  max(eu, eu2, es), 1e-14)

    # a rigid motion: invisible to the Neumann measure, visible to Dirichlet
    rigid = np.array([0.02, -0.01, 0.03]) + np.cross(
        np.array([0.005, -0.004, 0.002]), C.OBS)
    amp = float(np.max(np.abs(rigid)) / np.max(np.abs(ue)))
    e_n = C.field_errors(ue + rigid, se, "neumann")[0]
    e_d = C.field_errors(ue + rigid, se, "dirichlet")[0]
    rep.check("a rigid motion is invisible to the Neumann measure", e_n, 1e-13)
    rep.check_bool("  and fully visible to the Dirichlet measure",
                   abs(e_d / amp - 1.0) < 1e-9, f"({e_d:.3e} vs planted {amp:.3e})")

    # a NON-rigid perturbation must survive the rigid removal
    pert = np.zeros_like(ue)
    pert[0] = 0.01 * np.max(np.abs(ue))
    e_n2 = C.field_errors(ue + pert, se, "neumann")[0]
    rep.check_bool("  TRIPWIRE a non-rigid perturbation SURVIVES the removal",
                   e_n2 > 0.3 * 0.01, f"(planted 1.0e-2 of scale, kept {e_n2:.3e})")

    # surface measures, on real collocation points
    tv, _, _ = C.geometry("cube", 3)
    pts = ddbem.collocation_points(tv, 1, 0.5)
    ub = kelvin_displacement(pts, C.X0, C.FORCE, C.MU, C.NU)
    s = C.surface_errors(ub, pts, "neumann", "cube")
    rep.check("surface_errors is 0 on the exact boundary trace",
              max(s["u_surf"], s["u_surf_rms"]), 1e-13)
    # a uniform error gives a concentration of exactly 1
    unit = np.ones_like(ub) * 1e-3
    s1 = C.surface_errors(ub + unit, pts, "dirichlet", "cube")
    rep.check("edge concentration of a UNIFORM error is exactly 1",
              abs(s1["edge_conc"] - 1.0), 1e-13,
              f"({s1['frac_near_edge']:.3f} of points are in the edge band)")
    # an error placed only near the edges gives 1/frac
    near = C.cube_edge_distance(pts) < C.EDGE_BAND
    e_only = np.zeros_like(ub)
    e_only[near] = 1e-3
    s2 = C.surface_errors(ub + e_only, pts, "dirichlet", "cube")
    rep.check("  and of an edge-ONLY error is exactly 1 / (edge fraction)",
              abs(s2["edge_conc"] - 1.0 / s2["frac_near_edge"]), 1e-12,
              f"({s2['edge_conc']:.3f})")


def gate_edge_distance(rep):
    """The cube-edge classifier, against hand-computed distances."""
    a = 0.5 * C.CUBE_SIDE
    pts = np.array([[a, a, 0.0],        # ON an edge                     -> 0
                    [a, a, a],          # a corner                       -> 0
                    [0.0, 0.0, a],      # face centre                    -> a
                    [0.3, 0.0, a],      # on the +z face                 -> a - 0.3
                    [a, 0.0, 0.0],      # centre of the +x face          -> a
                    [a, 0.4 * a, 0.2 * a]])  # on the +x face, 0.6a from its
                                             # nearest edge (y = a)
    want = np.array([0.0, 0.0, a, a - 0.3, a, 0.6 * a])
    got = C.cube_edge_distance(pts)
    rep.check("cube_edge_distance matches hand-computed values",
              float(np.max(np.abs(got - want))), 1e-13,
              f"got {np.round(got, 4).tolist()}")
    tv, _, _ = C.geometry("cube", 4)
    d = C.cube_edge_distance(ddbem.collocation_points(tv, 0, 0.0))
    rep.check_bool("every P0 centroid of the cube is inside the cube's edge web",
                   bool(np.all(d >= -1e-12) and np.all(d <= a * np.sqrt(2) + 1e-12)),
                   f"range {d.min():.4f}..{d.max():.4f}")


def gate_harness_is_the_solver(rep):
    """``solve_ddbem`` must add nothing to the solve it measures."""
    from _exact import best_fit_rigid, kelvin_traction
    geom, level, order, bc = "sphere", 1, 0, "neumann"
    tv, _, _ = C.geometry(geom, level)
    h = float(ddbem.element_h(tv).mean())
    eps = 0.3 * h
    rec = C.solve_ddbem(geom, level, order, eps, bc)

    p = ddbem.Patch("bdy", tv, ddbem.BCType.FREE_TRACTION,
                    value=lambda x, n: kelvin_traction(x, n, C.X0, C.FORCE,
                                                      C.MU, C.NU), eps=eps)
    m = ddbem.Model([p], C.MU, C.NU, order=order, jump="calibrated")
    sol = m.solve()
    ue = kelvin_displacement(C.OBS, C.X0, C.FORCE, C.MU, C.NU)
    eu = float(np.max(np.abs(best_fit_rigid(sol.displacement(C.OBS) - ue, C.OBS)))
               / np.max(np.abs(ue)))
    rep.check("solve_ddbem == an inline hand-built solve", abs(rec["u_int"] - eu),
              1e-14, f"({rec['u_int']:.6e})")
    rep.check_bool("  and reports the right unknown count",
                   rec["n_dof"] == 3 * ddbem.n_nodes(order) * tv.shape[0])
    rep.check("  and the right eps/h", abs(rec["eps_over_h"] - 0.3), 1e-12)


def gate_trace_at_bary(rep):
    """``trace_at_bary`` must BE the model's own trace, at the same points."""
    from _exact import kelvin_traction
    for order in (0, 1, 2):
        tv, _, _ = C.geometry("sphere", 0)
        eps = 0.3 * float(ddbem.element_h(tv).mean())
        p = ddbem.Patch("bdy", tv, ddbem.BCType.FREE_TRACTION,
                        value=lambda x, n: kelvin_traction(x, n, C.X0, C.FORCE,
                                                           C.MU, C.NU), eps=eps)
        for jump in ("calibrated", "half"):
            m = ddbem.Model([p], C.MU, C.NU, order=order, jump=jump)
            sol = m.solve()
            sh = m.shrink_of(p)
            lam = ddbem.mesh.barycentric_nodes(order, sh)
            x, u = C.trace_at_bary(m, p, sol.q, lam)
            ref = sol.trace_displacement("bdy")
            xc = p.collocation_points(order, sh)
            rep.check(f"P{order} {jump:11s}: trace_at_bary == "
                      f"Solution.trace_displacement",
                      float(np.max(np.abs(u - ref))) / float(np.max(np.abs(ref))),
                      1e-14, f"(points agree to {np.max(np.abs(x - xc)):.1e})")
    # and it really does move: the centroid trace differs from the P1 nodal one
    m = ddbem.Model([p], C.MU, C.NU, order=1, jump="calibrated")
    sol = m.solve()
    _, u_cen = C.trace_at_bary(m, p, sol.q, np.full((1, 3), 1 / 3))
    u_nod = sol.trace_displacement("bdy")
    rep.check_bool("  TRIPWIRE the centroid trace is NOT the nodal trace",
                   float(np.max(np.abs(u_cen.mean(axis=0) - u_nod.mean(axis=0))))
                   > 0.0 and u_cen.shape[0] * 3 == u_nod.shape[0],
                   f"({u_cen.shape[0]} centroids vs {u_nod.shape[0]} nodes)")


def gate_msd_anchor(rep):
    """msd's frozen P0 solver on the same problem converges and is comparable."""
    errs, dd = [], []
    for level in (0, 1):
        tv, _, _ = C.geometry("sphere", level)
        eps = 0.3 * float(ddbem.element_h(tv).mean())
        errs.append(C.solve_msd("sphere", level, eps, "neumann")["u_int"])
        dd.append(C.solve_ddbem("sphere", level, 0, eps, "neumann")["u_int"])
    rep.check_bool("msd P0 anchor converges on the sphere ladder",
                   errs[1] < errs[0], f"{errs[0]:.3e} -> {errs[1]:.3e}")
    rep.check("msd P0 anchor error on the 80-triangle sphere", errs[1], 0.05,
              f"(ddbem P0 {dd[1]:.3e})")
    ratio = max(errs[1] / dd[1], dd[1] / errs[1])
    rep.check_bool("  ddbem P0 and msd P0 land within 3x of each other",
                   ratio < 3.0, f"({ratio:.2f}x -- they are DIFFERENT "
                                f"formulations, so this is a sanity bound, "
                                f"not a parity check)")


def gate_cache(rep):
    """Round-trip, and resume-skips rather than recompute or duplicate."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    old = C.CACHE
    try:
        C.CACHE = tmp
        rec = {"solver": "ddbem", "geom": "sphere", "bc": "neumann", "order": 2,
               "level": 1, "eps": 0.1234567890123, "u_int": 1.5e-3,
               "case": "x", "nan_field": float("nan")}
        C.append("t", rec)
        back = C.load(["t"])
        rep.check_bool("cache round-trips a record unchanged",
                       len(back) == 1
                       and all(back[0][k] == rec[k] for k in rec if k != "nan_field")
                       and np.isnan(back[0]["nan_field"]))
        rep.check("  float precision survives JSON",
                  abs(back[0]["eps"] - rec["eps"]), 0.0 if False else 1e-15)
        C.run_plan("headline", tag="t2", geoms=["sphere"], bcs=["neumann"],
                   orders=[0], levels=[0], dry=True)
        n0 = len(C.load(["t2"]))
        C.run_plan("headline", tag="t2", geoms=["sphere"], bcs=["neumann"],
                   orders=[0], levels=[0])
        n1 = len(C.load(["t2"]))
        C.run_plan("headline", tag="t2", geoms=["sphere"], bcs=["neumann"],
                   orders=[0], levels=[0])
        n2 = len(C.load(["t2"]))
        rep.check_bool("--dry writes nothing; a run writes once; a re-run skips",
                       (n0, n1, n2) == (0, 1, 1), f"(counts {n0}, {n1}, {n2})")
    finally:
        C.CACHE = old
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    rep = Report("ddbem convergence study: the harness")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_exact_solution(rep)
        gate_rate_fit(rep)
        gate_error_measures(rep)
        gate_edge_distance(rep)
        gate_harness_is_the_solver(rep)
        gate_trace_at_bary(rep)
        gate_msd_anchor(rep)
        gate_cache(rep)
    return 0 if rep.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
