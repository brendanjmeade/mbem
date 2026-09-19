"""Convergence of P0/P1/P2 DD collocation -- in h, per unknown, and in eps.

    python convergence.py run    --plan headline          # compute, cache
    python convergence.py report                          # tables + rates
    python convergence.py plans                           # what plans exist

This is what the sub-project is for.  ``../fbem/FINDINGS.md`` closed the
indirect (force-element) BEM with two numbers -- O(h^0.31) on free-traction rows
over a polyhedron against the direct formulation's O(h^0.90), and p-refinement
buying *nothing* (P0 +0.31, P1 +0.32, P2 +0.28) -- and with one transferable
warning:

> eps is not only a regularisation.  It is a FLOOR on the resolvable structure
> of the unknown.  No basis can represent detail below eps.

At ``eps = 0.3 h`` no collocation point of a P1/P2 element is even one
mollification length clear of the element boundary, and higher order bought
nothing there; holding eps FIXED in absolute terms inverted the ordering and P1
beat P0 by 1.37x.  So the question this script exists to answer is not "does DD
converge" (``verify/verify_solver.py`` already gates that) but:

    does higher-order DD hit the same wall, and is that wall eps or the edge?

WHAT IS MEASURED
================
Exact solution: a Kelvin point force OUTSIDE the body (``verify/_exact.py`` for
why that is the only exact solution a boundary-only representation can be asked
to reproduce).  Two row types, which are two different operators:

* ``dirichlet`` -- ``u_int(x_c) = u_bar``, the second-kind displacement row;
* ``neumann``   -- ``t(x_c) = t_bar``, the HYPERSINGULAR row, which is the row
  fbem's stall lived on.

Two domains, because fbem's stall was EDGE-driven and the sphere is what
separated the two:

* ``sphere`` -- icosphere, radius 1, 20/80/320/1280 triangles.  Smooth; the
  only edges are the O(h) dihedral creases of the polyhedral approximation,
  which flatten under refinement.
* ``cube``   -- side 2, 12 m^2 triangles at m = 2/3/4/6.  Twelve permanent
  90-degree edges and eight corners, whose exterior wedge is exactly the
  270-degree geometry that gave the single-layer density its rho^(-1/3)
  singularity.  A DD density is a displacement jump and should be BOUNDED
  there -- that is the whole claim of the formulation, and this is where it is
  tested.

Four error measures per solve, all relative to ``max|u_exact|``:

* ``u_int``   interior displacement on a FIXED cloud (|x| <= 0.3, i.e. never
  closer than ~0.7 to the surface, because a field read from the
  representation is mesh-limited within ~1 h of the boundary -- msd says the
  same in ``mbem/evaluate.py::_warn_near_boundary``).  The headline number.
* ``sigma``   interior stress on the same cloud.  Reported, never used for a
  rate: it is measurably non-monotone in h (documented in verify_solver.py).
* ``u_surf``  the BOUNDARY trace at the collocation points
  (``Solution.trace_displacement``) against the exact field there.  This is the
  honest place to look for an edge effect: the interior field is an integral of
  the density and smooths it, while the trace is the density.  Defined for the
  Neumann rows; on a Dirichlet row the trace IS the prescribed data (the row
  states it), so it is reported as a residual and is not an accuracy measure.
* ``u_surf_cen`` the SAME trace read at the element CENTROID for every order
  (``trace_at_bary``).  The collocation point moves with the element order -- a
  P0 centroid, a shrunk P1 vertex, a shrunk P2 midpoint -- and since the whole
  question is how far a collocation point is (in eps) from the element
  boundary, "where you look" is exactly the confound that has to be removed
  before any p-comparison means anything.  It is large: on the cube at
  eps/h = 0.3 the P1 surface error is 3.9e-1 read at its own node and 1.7e-1
  read at the centroid.
* ``edge_conc`` (cube only) the fraction of the squared surface error carried by
  collocation points within 0.3 of a cube edge, divided by the fraction of
  points there -- fbem's "error-energy concentration near edges", which it
  measured to be identical to within 4 % across P0/P1/P2.  Here it GROWS under
  refinement (1.39 -> 1.59 at P0, 1.65 -> 1.98 at P1), which is the signature of
  an edge feature the mesh is not resolving.

Rates are least-squares fits of ``log(err)`` on ``log(h)`` AND on
``log(n_unknowns)``.  **Error vs unknowns is the honest axis**: discontinuous P2
costs 18 unknowns per triangle against P0's 3, so a p-refinement that wins in h
can lose outright at matched cost.

THE ANCHOR
==========
``--plan msd`` runs msd's FROZEN P0 solver (``mbem``, numba) on the SAME
problem, same meshes, same eps.  It is a different formulation -- the direct
Somigliana BIE with u and t as unknowns, not one DD density -- so this is not a
parity check (``verify/verify_msd_parity.py`` is, on the one problem where the
two coincide).  It is the reference rate: whatever ddbem P0 does, msd P0 has
been doing for a year, and if the two disagree about the RATE then this script
is measuring the harness and not the method.

CAVEATS THAT APPLY TO EVERY NUMBER BELOW
========================================
* Pure Python assembly, 470-930x slower than msd's numba kernels (Stage 1).
  The largest case here is ~7800 unknowns; a production-scale study needs the
  numba port first.
* The DISCONTINUOUS layout is the only one the solver supports
  (``ddbem/model.py``, "WHAT IS NOT HERE"), so the per-unknown axis is as
  unfavourable to higher order as it can possibly be.  Continuous P1 would be
  ~1.5 unknowns per triangle against P0's 3 -- a 6x change in the x-axis that
  this study CANNOT measure.  Read every per-unknown rate with that in mind.
* Collocation shrink is the fixed default (0.5 at P1/P2), not the measured
  1-eps rule, except in ``--plan shrink``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify"))

import ddbem                                                     # noqa: E402
from _exact import (best_fit_rigid, kelvin_displacement,         # noqa: E402
                    kelvin_stress, kelvin_traction)

# --------------------------------------------------------------------------
# constants of the study (defaults module is for LIBRARY constants; these are
# the study's own, and they are here so one file defines the experiment)
# --------------------------------------------------------------------------
MU, NU = 1.0, 0.30
X0 = np.array([3.0, 1.2, -0.7])          # point force, outside both domains
FORCE = np.array([1.0, -0.5, 0.8])
CACHE = ROOT / "convergence_cache"
EDGE_BAND = 0.3                          # cube-edge band for the concentration
SPHERE_LEVELS = (0, 1, 2, 3)             # 20, 80, 320, 1280 triangles
CUBE_LEVELS = (2, 3, 4, 6)               # 48, 108, 192, 432 triangles
CUBE_SIDE = 2.0

#: interior observation cloud.  |x| <= 0.3 for BOTH domains (the cube's
#: inradius is 1 and the sphere's radius is 1), so the two are read at the same
#: standoff from the surface and their errors are comparable.
OBS = np.array([[0.00, 0.00, 0.00],
                [0.25, 0.10, -0.10],
                [-0.15, 0.20, 0.10],
                [0.05, -0.28, 0.12],
                [-0.20, -0.10, -0.22],
                [0.12, 0.22, 0.18],
                [-0.26, 0.06, -0.08]])


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def geometry(geom: str, level: int):
    """``(tri_verts, vertices, triangles)`` for one rung of a mesh ladder."""
    if geom == "sphere":
        v, t = ddbem.icosphere(int(level), radius=1.0)
    elif geom == "cube":
        m = int(level)
        v, t = ddbem.box((CUBE_SIDE,) * 3, (m, m, m))
    else:
        raise ValueError(f"unknown geometry {geom!r}")
    return ddbem.tri_verts(v, t), v, t


def cube_edge_distance(points) -> np.ndarray:
    """Distance from each point to the nearest of the cube's 12 edges."""
    p = np.asarray(points, float)
    a = 0.5 * CUBE_SIDE
    d = np.full(p.shape[0], np.inf)
    for axis in range(3):
        b, c = (axis + 1) % 3, (axis + 2) % 3
        for sb in (-a, a):
            for sc in (-a, a):
                # the edge is the line {x_b = sb, x_c = sc}, |x_axis| <= a
                t = np.clip(p[:, axis], -a, a)
                dd = np.sqrt((p[:, b] - sb) ** 2 + (p[:, c] - sc) ** 2
                             + (p[:, axis] - t) ** 2)
                d = np.minimum(d, dd)
    return d


# --------------------------------------------------------------------------
# error measures
# --------------------------------------------------------------------------

def _rel(num, den):
    return float(num) / float(den)


def field_errors(u, sig, bc):
    """Interior displacement and stress errors on :data:`OBS`."""
    ue = kelvin_displacement(OBS, X0, FORCE, MU, NU)
    se = ddbem.tensor_to_voigt(kelvin_stress(OBS, X0, FORCE, MU, NU))
    scale_u = float(np.max(np.abs(ue)))
    du = u - ue
    if bc == "neumann":                 # defined only up to a rigid motion
        du = best_fit_rigid(du, OBS)
    return (_rel(np.max(np.abs(du)), scale_u),
            _rel(np.sqrt(np.mean(du ** 2)), np.sqrt(np.mean(ue ** 2))),
            _rel(np.max(np.abs(sig - se)), np.max(np.abs(se))))


def surface_errors(trace, pts, bc, geom):
    """Boundary-trace error, and (cube) its concentration near the edges."""
    ue = kelvin_displacement(pts, X0, FORCE, MU, NU)
    scale = float(np.max(np.abs(ue)))
    d = trace - ue
    if bc == "neumann":
        d = best_fit_rigid(d, pts)
    e2 = np.sum(d ** 2, axis=1)
    out = {"u_surf": _rel(np.max(np.abs(d)), scale),
           "u_surf_rms": _rel(np.sqrt(np.mean(e2)), np.sqrt(np.mean(ue ** 2))),
           "edge_conc": float("nan")}
    if geom == "cube":
        near = cube_edge_distance(pts) < EDGE_BAND
        f_pts = float(near.mean())
        if 0.0 < f_pts < 1.0 and e2.sum() > 0:
            out["edge_conc"] = float((e2[near].sum() / e2.sum()) / f_pts)
        out["frac_near_edge"] = f_pts
    return out


# --------------------------------------------------------------------------
# the interior trace at an ARBITRARY point of every element
# --------------------------------------------------------------------------

def trace_at_bary(model, patch, q, lam):
    """Interior-side displacement at the barycentric points ``lam`` (M, 3).

    WHY THIS EXISTS.  ``Solution.trace_displacement`` reads the trace at the
    patch's own COLLOCATION points, which move with the element order (a P0
    centroid, a shrunk P1 vertex, a shrunk P2 midpoint).  Comparing the surface
    error across orders therefore compares three different places on the
    element -- and since the whole question is whether a collocation point is
    far enough (in eps) from the element boundary, "where you look" is exactly
    the confound that has to be removed.  This evaluates the SAME functional at
    the SAME place for every order.

    It mirrors ``ddbem.Model._trace_displacement`` line for line with the
    barycentric point generalised; ``verify/verify_convergence.py`` gates the
    two against each other at the collocation barycentrics, so it cannot drift
    from the model's own convention.
    """
    lam = np.atleast_2d(np.asarray(lam, float))
    o = model.order_of(patch)
    K = ddbem.n_nodes(o)
    x = np.einsum("mv,tvc->tmc", lam, patch.tri_verts).reshape(-1, 3)
    cols, n = {}, 0
    for p in model.boundary:
        d = p.n_dof(model.order_of(p))
        cols[p.name] = slice(n, n + d)
        n += d
    u = np.zeros(3 * x.shape[0])
    R = np.zeros((x.shape[0], 3, 3))
    for p in model.boundary:
        B = ddbem.displacement_matrix(x, p.tri_verts, model.eps_of(p), model.mu,
                                      model.nu, model.order_of(p),
                                      far_field=model.far_field)
        u += B @ q[cols[p.name]]
        for j in range(3):
            R[:, :, j] -= float(p.orientation) * B[:, j::3].sum(axis=1).reshape(-1, 3)
    for f in model.faults:
        B = ddbem.displacement_matrix(x, f.tri_verts, model.eps_of(f), model.mu,
                                      model.nu, model.order_of(f),
                                      far_field=model.far_field)
        u += B @ f.nodal_density(model.order_of(f))
    if model.jump == "calibrated":
        F = float(patch.orientation) * (R - np.eye(3))
    else:
        F = np.broadcast_to(-0.5 * float(patch.orientation) * np.eye(3),
                            (x.shape[0], 3, 3))
    N = ddbem.shape_at(o, lam)                       # (M, K)
    qp = q[cols[patch.name]].reshape(patch.n_tri, K, 3)
    at_c = np.einsum("mk,tkj->tmj", N, qp).reshape(-1, 3)
    return x, u.reshape(-1, 3) + np.einsum("cij,cj->ci", F, at_c)


# --------------------------------------------------------------------------
# one ddbem solve
# --------------------------------------------------------------------------

def solve_ddbem(geom, level, order, eps, bc, jump="calibrated", shrink=None,
                constrain=None, eps_label=""):
    tv, _, _ = geometry(geom, level)
    h = float(ddbem.element_h(tv).mean())
    if bc == "dirichlet":
        patch = ddbem.Patch("bdy", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                            value=lambda x: kelvin_displacement(x, X0, FORCE, MU, NU),
                            eps=eps)
    elif bc == "neumann":
        patch = ddbem.Patch("bdy", tv, ddbem.BCType.FREE_TRACTION,
                            value=lambda x, n: kelvin_traction(x, n, X0, FORCE, MU, NU),
                            eps=eps)
    else:
        raise ValueError(bc)
    m = ddbem.Model([patch], MU, NU, order=order, jump=jump, shrink=shrink)
    t0 = time.perf_counter()
    system = m.assemble()
    t1 = time.perf_counter()
    sol = m.solve(system, constrain=constrain)
    t2 = time.perf_counter()
    eu, eu2, es = field_errors(sol.displacement(OBS), sol.stress(OBS), bc)
    sh = m.shrink_of(patch)
    # The boundary trace is an ACCURACY measure only for the Neumann rows.  On
    # a Dirichlet row ``u_int(x_c) = u_bar`` is the equation, so the trace is
    # the solver residual, not an error -- and it costs a second full assembly.
    if bc == "neumann":
        pts = patch.collocation_points(order, sh)
        surf = surface_errors(sol.trace_displacement("bdy"), pts, bc, geom)
        # ... and again at the element CENTROID, the one place every order can
        # be read at, so the p-comparison is not confounded by the collocation
        # point moving with the order.
        cen, u_cen = trace_at_bary(m, patch, sol.q, np.full((1, 3), 1.0 / 3.0))
        sc = surface_errors(u_cen, cen, bc, geom)
        surf["u_surf_cen"] = sc["u_surf"]
        surf["u_surf_cen_rms"] = sc["u_surf_rms"]
        surf["edge_conc_cen"] = sc["edge_conc"]
    else:
        surf = {"u_surf": float("nan"), "u_surf_rms": float("nan"),
                "edge_conc": float("nan"), "u_surf_cen": float("nan"),
                "u_surf_cen_rms": float("nan"), "edge_conc_cen": float("nan")}
    clr = float(np.median(ddbem.node_clearance(tv, order, sh).min(axis=1)))
    rec = {"solver": "ddbem", "geom": geom, "level": int(level),
           "order": int(order), "bc": bc, "jump": jump, "shrink": sh,
           "n_tri": int(tv.shape[0]), "n_dof": int(system.n_dof), "h": h,
           "eps": float(np.mean(m.eps_of(patch))), "eps_over_h": float(np.mean(m.eps_of(patch)) / h),
           "clearance_over_eps": clr / float(np.mean(m.eps_of(patch))),
           "u_int": eu, "u_int_rms": eu2, "sigma": es,
           "cond": float(sol.cond), "residual": float(sol.residual),
           "rigid_defect": float(system.rigid_defect()),
           "t_assemble": t1 - t0, "t_solve": t2 - t1,
           "constrain": constrain or "auto", "eps_label": eps_label}
    rec.update(surf)
    return rec


# --------------------------------------------------------------------------
# the msd P0 anchor (frozen; read-only)
# --------------------------------------------------------------------------

def solve_msd(geom, level, eps, bc, jump=None, eps_label=""):
    """msd's frozen direct-BIE P0 solver on the same problem.

    Different formulation (u and t as boundary unknowns, Somigliana), same
    mesh, same eps, same data.  ``jump`` defaults to msd's best setting for the
    problem: "calibrated" + 3-translation deflation for the all-Neumann case,
    "half" for the Dirichlet case (whose unknown is the traction, so there is
    no u diagonal to calibrate).
    """
    sys.path.insert(0, str(ROOT.parent / "msd"))
    import mollified_bem as mb
    from mbem.backends.dense import DenseBackend
    from mbem.evaluate import evaluate_displacement, evaluate_stress
    from mbem.model.core import BCType as MBC
    from mbem.model.core import Patch as MPatch
    from mbem.model.core import Region as MRegion
    from mbem.model.core import RegionModel
    from mbem.model.equations import generate_system

    tv, v, t = geometry(geom, level)
    h = float(ddbem.element_h(tv).mean())
    mesh = mb.TriMesh(vertices=np.ascontiguousarray(v, float),
                      triangles=np.ascontiguousarray(t))
    c = mesh.centroids()
    n, _ = mesh.normals_and_areas()
    if bc == "dirichlet":
        patch = MPatch("bdy", mesh, MBC.PRESCRIBED_DISPLACEMENT,
                       value=kelvin_displacement(c, X0, FORCE, MU, NU))
        jump = jump or "half"
        deflate = False
    else:
        patch = MPatch("bdy", mesh, MBC.FREE_TRACTION,
                       value=kelvin_traction(c, n, X0, FORCE, MU, NU))
        jump = jump or "calibrated"
        deflate = True
    lam = 2.0 * MU * NU / (1.0 - 2.0 * NU)
    region = MRegion("body", mb.ElasticMaterial(mu=MU, lam=lam), [patch],
                     np.zeros(3))
    model = RegionModel([region])
    model.validate()
    eps_map = {"bdy": eps}
    t0 = time.perf_counter()
    back = DenseBackend(mode="direct", jump=jump,
                        deflate=deflate).assemble(generate_system(model), eps_map)
    t1 = time.perf_counter()
    out = back.solve()
    t2 = time.perf_counter()
    u = evaluate_displacement(model, region, out, OBS, eps_map, warn_near=False)
    sig = evaluate_stress(model, region, out, OBS, eps_map, warn_near=False)
    eu, eu2, es = field_errors(u, ddbem.tensor_to_voigt(sig), bc)
    rec = {"solver": "msd", "geom": geom, "level": int(level), "order": 0,
           "bc": bc, "jump": jump, "shrink": 0.0, "n_tri": int(tv.shape[0]),
           "n_dof": int(back.A.shape[0]), "h": h,
           "eps": float(np.mean(eps)), "eps_over_h": float(np.mean(eps)) / h,
           "clearance_over_eps": float("nan"),
           "u_int": eu, "u_int_rms": eu2, "sigma": es,
           "cond": float(back.report.cond_estimate),
           "residual": float("nan"), "rigid_defect": float("nan"),
           "t_assemble": t1 - t0, "t_solve": t2 - t1, "constrain": "-",
           "u_surf": float("nan"), "u_surf_rms": float("nan"),
           "edge_conc": float("nan"), "eps_label": eps_label}
    if bc == "neumann":         # msd's Neumann unknown IS the boundary u
        pts = c
        ub = out["u:bdy"]
        s = surface_errors(ub, pts, bc, geom)
        rec.update(s)
    return rec


# --------------------------------------------------------------------------
# rate fitting
# --------------------------------------------------------------------------

def fit_rate(x, y):
    """Least-squares slope of ``log y`` on ``log x`` (nan with < 2 valid points)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

def cache_path(tag: str) -> pathlib.Path:
    CACHE.mkdir(exist_ok=True)
    return CACHE / f"{tag}.jsonl"


def append(tag: str, rec: dict) -> None:
    with cache_path(tag).open("a") as f:
        f.write(json.dumps(rec) + "\n")


def load(tags=None, dedupe: bool = True) -> list:
    """Every cached record, newest-and-richest wins on a repeated case.

    Plans deliberately overlap (``ptest`` re-solves cases ``eps_ratio`` already
    has, to add the common-point trace), so the same ``case`` id can appear in
    two tag files.  They are the SAME solve -- deterministic, same mesh, same
    eps -- so the duplicate is dropped, preferring the record that carries the
    extra measurement.
    """
    CACHE.mkdir(exist_ok=True)
    files = ([cache_path(t) for t in tags] if tags
             else sorted(CACHE.glob("*.jsonl")))
    out = []
    for f in files:
        if f.exists():
            out += [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    if not dedupe:
        return out
    best = {}
    for r in out:
        k = r.get("case") or json.dumps(sorted(r.items()), default=str)
        old = best.get(k)
        if old is None or (not np.isfinite(old.get("u_surf_cen", np.nan))
                           and np.isfinite(r.get("u_surf_cen", np.nan))):
            best[k] = r
    return list(best.values())


def case_id(solver, geom, bc, order, level, eps_label, jump, shrink) -> str:
    """Canonical name of one solve -- the cache key, so a rerun resumes."""
    return (f"{solver}|{geom}|{bc}|P{order}|L{level}|{eps_label}|{jump}|"
            f"t{float(shrink):.3f}")


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------

def _eps_values(geom, mode, value, h):
    return value * h if mode == "ratio" else value


PLANS = {
    # the headline ladders: both geometries, both row types, three orders,
    # eps/h = 0.3 (the tree's default and fbem's stall point)
    "headline": dict(geoms=("sphere", "cube"), bcs=("dirichlet", "neumann"),
                     orders=(0, 1, 2), eps_mode="ratio", eps_values=(0.3,)),
    # the eps sweep: the question the study exists for
    "eps_ratio": dict(geoms=("sphere", "cube"), bcs=("dirichlet", "neumann"),
                      orders=(0, 1, 2), eps_mode="ratio", eps_values=(0.5, 0.15)),
    # eps FIXED in absolute terms (fbem's inversion experiment)
    "eps_abs": dict(geoms=("sphere", "cube"), bcs=("dirichlet", "neumann"),
                    orders=(0, 1, 2), eps_mode="absolute", eps_values=(0.25, 0.12)),
    # does the free-term calibration change the RATE?
    "jump": dict(geoms=("sphere", "cube"), bcs=("dirichlet",), orders=(0, 1, 2),
                 eps_mode="ratio", eps_values=(0.3,), jumps=("half",)),
    # the msd P0 anchor, every eps the ddbem ladders use
    "msd": dict(geoms=("sphere", "cube"), bcs=("dirichlet", "neumann"),
                orders=(0,), eps_mode="ratio", eps_values=(0.5, 0.3, 0.15),
                solver="msd"),
    "msd_abs": dict(geoms=("sphere", "cube"), bcs=("dirichlet", "neumann"),
                    orders=(0,), eps_mode="absolute", eps_values=(0.25, 0.12),
                    solver="msd"),
    # the p-refinement test read at a COMMON point (the element centroid), so
    # the comparison across orders is not confounded by the collocation point
    # moving with the order.  Neumann rows only: a Dirichlet row states the
    # trace, so there is nothing to measure there.
    "ptest": dict(geoms=("sphere", "cube"), bcs=("neumann",), orders=(0, 1, 2),
                  eps_mode="ratio", eps_values=(0.5, 0.3, 0.15)),
    # the collocation shrink, on the production-relevant row
    "shrink": dict(geoms=("cube",), bcs=("neumann",), orders=(1, 2),
                   eps_mode="ratio", eps_values=(0.15,), shrinks=(0.3, 0.7)),
}


def levels_for(geom):
    return SPHERE_LEVELS if geom == "sphere" else CUBE_LEVELS


def run_plan(name, tag=None, max_dof=8000, levels=None, dry=False,
             geoms=None, bcs=None, orders=None):
    """Compute every case of one plan that is not already cached.

    ``geoms``/``bcs``/``orders`` narrow the plan (so one plan can be split
    across several processes, each with its own ``tag``; the cache is one
    append-only JSONL per tag and ``report`` merges them).
    """
    plan = dict(PLANS[name])
    if geoms:
        plan["geoms"] = tuple(geoms)
    if bcs:
        plan["bcs"] = tuple(bcs)
    if orders is not None:
        plan["orders"] = tuple(o for o in plan["orders"] if o in set(orders))
    tag = tag or name
    done = {r["case"] for r in load([tag]) if "case" in r}
    solver = plan.get("solver", "ddbem")
    n_run = n_skip = 0
    for geom in plan["geoms"]:
        lv = list(levels) if levels is not None else list(levels_for(geom))
        for bc in plan["bcs"]:
            for ev in plan["eps_values"]:
                for order in plan["orders"]:
                    for jump in plan.get("jumps", ("calibrated",)):
                        for shrink in plan.get("shrinks", (None,)):
                            for level in lv:
                                tv, _, _ = geometry(geom, level)
                                h = float(ddbem.element_h(tv).mean())
                                eps = ev * h if plan["eps_mode"] == "ratio" else ev
                                lab = (f"eps/h={ev:g}" if plan["eps_mode"] == "ratio"
                                       else f"eps={ev:g}")
                                sh = (shrink if shrink is not None else
                                      ddbem.defaults.COLLOCATION_SHRINK_BY_ORDER[order])
                                jm = (jump if solver == "ddbem" else
                                      ("half" if bc == "dirichlet" else "calibrated"))
                                cid = case_id(solver, geom, bc, order, level, lab,
                                              jm, 0.0 if solver == "msd" else sh)
                                if cid in done:
                                    n_skip += 1
                                    continue
                                dof = 3 * ddbem.n_nodes(order) * tv.shape[0]
                                if solver == "ddbem" and dof > max_dof:
                                    print(f"  skip {cid}: {dof} dof > max_dof",
                                          flush=True)
                                    continue
                                if dry:
                                    print(f"  would run {cid} ({dof} dof)")
                                    n_run += 1
                                    continue
                                t0 = time.perf_counter()
                                if solver == "msd":
                                    rec = solve_msd(geom, level, eps, bc,
                                                    eps_label=lab)
                                else:
                                    rec = solve_ddbem(geom, level, order, eps, bc,
                                                      jump=jump, shrink=shrink,
                                                      eps_label=lab)
                                rec["plan"] = name
                                rec["case"] = cid
                                append(tag, rec)
                                n_run += 1
                                print(f"  {cid:52s} dof={rec['n_dof']:5d} "
                                      f"eps/h={rec['eps_over_h']:.3f} "
                                      f"u={rec['u_int']:.3e} "
                                      f"surf={rec.get('u_surf', float('nan')):.3e} "
                                      f"[{time.perf_counter() - t0:.1f}s]", flush=True)
    print(f"plan {name}: {n_run} new, {n_skip} cached", flush=True)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _ladder(rows, key):
    rows = sorted(rows, key=lambda r: r["n_tri"])
    return (np.array([r["h"] for r in rows]),
            np.array([r["n_dof"] for r in rows]),
            np.array([r.get(key, np.nan) for r in rows]), rows)


def report(tags=None, keys=("u_int", "u_surf")):
    recs = load(tags)
    if not recs:
        print("no cached results; run `python convergence.py run --plan headline`")
        return
    # group: (geom, bc, eps-label, jump, shrink, solver, order)
    groups = {}
    for r in recs:
        lab = r.get("eps_label") or f"eps={r['eps']:.3g}"
        # group P0/P1/P2 TOGETHER when each is at its own default shrink (the
        # collocation point moves with the order by design); a hand-set shrink
        # gets its own group so the `shrink` plan does not contaminate them.
        dflt = ddbem.defaults.COLLOCATION_SHRINK_BY_ORDER[r["order"]]
        st = "t=default" if abs(r["shrink"] - dflt) < 1e-9 else f"t={r['shrink']:g}"
        g = (r["geom"], r["bc"], lab, r["jump"], st)
        groups.setdefault(g, {}).setdefault((r["solver"], r["order"]), []).append(r)
    for g in sorted(groups):
        geom, bc, lab, jump, shrink = g
        print("\n" + "=" * 100)
        print(f"{geom}  |  {bc} rows  |  {lab}  |  jump={jump}  shrink {shrink}")
        print("=" * 100)
        print(f"{'':10s} {'N_tri':>6s} {'dof':>6s} {'h':>7s} {'eps/h':>6s} "
              f"{'clr/eps':>7s} {'u_int':>10s} {'u_surf':>10s} {'sigma':>10s} "
              f"{'conc':>6s} {'cond':>9s} {'t(s)':>7s}")
        for so in sorted(groups[g]):
            solver, order = so
            rows = sorted(groups[g][so], key=lambda r: r["n_tri"])
            name = f"{solver} P{order}"
            for r in rows:
                print(f"{name:10s} {r['n_tri']:6d} {r['n_dof']:6d} {r['h']:7.4f} "
                      f"{r['eps_over_h']:6.3f} {r['clearance_over_eps']:7.3f} "
                      f"{r['u_int']:10.3e} {r.get('u_surf', float('nan')):10.3e} "
                      f"{r['sigma']:10.3e} {r.get('edge_conc', float('nan')):6.2f} "
                      f"{r['cond']:9.2e} {r['t_assemble'] + r['t_solve']:7.1f}")
                name = ""
            h, n, _, _ = _ladder(rows, "u_int")
            line = f"{'':10s} rates:"
            for k in keys:
                y = np.array([r.get(k, np.nan) for r in rows])
                line += (f"   {k}: h {fit_rate(h, y):+.2f}, "
                         f"dof {fit_rate(n, y):+.2f}")
            print(line)
    _cross_order(groups)


def _cross_order(groups):
    """The comparison the study is for: P0 vs P1 vs P2 at MATCHED unknowns."""
    print("\n" + "=" * 100)
    print("HIGHER ORDER AT MATCHED UNKNOWN COUNT  (error interpolated to the "
          "P0 ladder's dof range)")
    print("=" * 100)
    print(f"{'geom':7s} {'bc':10s} {'eps':13s} {'dof':>7s} "
          f"{'P0':>10s} {'P1':>10s} {'P2':>10s}   {'P1/P0':>6s} {'P2/P0':>6s}")
    for g in sorted(groups):
        geom, bc, lab, jump, shrink = g
        if jump != "calibrated" or shrink != "t=default":
            continue
        by = {}
        for (solver, order), rows in groups[g].items():
            if solver != "ddbem":
                continue
            rows = sorted(rows, key=lambda r: r["n_dof"])
            by[order] = (np.array([r["n_dof"] for r in rows], float),
                         np.array([r["u_int"] for r in rows], float))
        if 0 not in by:
            continue
        n0, e0 = by[0]
        for target in n0[1:]:
            vals = {}
            for p, (n, e) in by.items():
                if n.min() <= target <= n.max():
                    vals[p] = float(np.exp(np.interp(np.log(target), np.log(n),
                                                     np.log(e))))
            if 0 not in vals:
                continue
            f = lambda p: (f"{vals[p]:10.3e}" if p in vals else f"{'-':>10s}")  # noqa: E731
            r1 = f"{vals[1] / vals[0]:6.2f}" if 1 in vals else f"{'-':>6s}"
            r2 = f"{vals[2] / vals[0]:6.2f}" if 2 in vals else f"{'-':>6s}"
            print(f"{geom:7s} {bc:10s} {lab:13s} {int(target):7d} "
                  f"{f(0)} {f(1)} {f(2)}   {r1} {r2}")


def collapse(tags=None, metric="u_surf", geom="cube", bc="neumann",
             solver="ddbem", jump="calibrated"):
    """Is the error set by the MESH or by the MOLLIFICATION?

    Pools every cached case of one (geometry, row type) and regresses

        log err  =  a  +  b log h  +  c log(clearance / eps)

    against the two single-variable models.  ``clearance / eps`` is the
    in-plane distance from a collocation point to its own element's boundary,
    measured in mollification lengths -- ``../fbem/FINDINGS.md`` sec.2 says this
    is the quantity that decides whether extra nodes buy anything, and this is
    the test of that claim on the DD formulation.  R^2 is the fraction of the
    variance of ``log err`` each model explains; the honest comparison is
    between the two one-variable fits, since the two-variable fit cannot be
    worse than either.
    """
    recs = [r for r in load(tags)
            if r["solver"] == solver and r["geom"] == geom and r["bc"] == bc
            and r["jump"] == jump and np.isfinite(r.get(metric, np.nan))
            and r.get(metric, 0) > 0 and np.isfinite(r["clearance_over_eps"])]
    if len(recs) < 6:
        print(f"collapse: only {len(recs)} usable records for "
              f"{geom}/{bc}/{metric}; run more of the study first")
        return
    print("\n" + "=" * 100)
    print(f"WHAT SETS THE ERROR: {metric} on the {geom}, {bc} rows "
          f"({len(recs)} cases, {jump} free term)")
    print("=" * 100)
    print(f"{'order':>5s} {'n':>4s} {'model':28s} {'coeff(s)':>26s} {'R^2':>7s}")
    for order in sorted({r["order"] for r in recs}):
        sub = [r for r in recs if r["order"] == order]
        if len(sub) < 4:
            continue
        y = np.log(np.array([r[metric] for r in sub]))
        lh = np.log(np.array([r["h"] for r in sub]))
        lc = np.log(np.array([r["clearance_over_eps"] for r in sub]))
        for label, cols in (("err ~ h", [lh]),
                            ("err ~ clearance/eps", [lc]),
                            ("err ~ h AND clearance/eps", [lh, lc])):
            M = np.column_stack([np.ones_like(y)] + cols)
            beta, *_ = np.linalg.lstsq(M, y, rcond=None)
            resid = y - M @ beta
            r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
            coef = "  ".join(f"{b:+.2f}" for b in beta[1:])
            print(f"P{order:<4d} {len(sub):4d} {label:28s} {coef:>26s} {r2:7.3f}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", nargs="+", default=["headline"])
    r.add_argument("--tag", default=None)
    r.add_argument("--max-dof", type=int, default=8000)
    r.add_argument("--levels", type=int, nargs="+", default=None)
    r.add_argument("--dry", action="store_true")
    r.add_argument("--geoms", nargs="+", default=None)
    r.add_argument("--bcs", nargs="+", default=None)
    r.add_argument("--orders", type=int, nargs="+", default=None)
    p = sub.add_parser("report")
    p.add_argument("--tags", nargs="+", default=None)
    cl = sub.add_parser("collapse")
    cl.add_argument("--tags", nargs="+", default=None)
    cl.add_argument("--metric", default="u_surf")
    cl.add_argument("--geom", default="cube")
    cl.add_argument("--bc", default="neumann")
    sub.add_parser("plans")
    a = ap.parse_args()
    if a.cmd == "plans":
        for k, v in PLANS.items():
            print(f"{k:12s} {v}")
        return
    if a.cmd == "report":
        report(a.tags)
        return
    if a.cmd == "collapse":
        collapse(a.tags, metric=a.metric, geom=a.geom, bc=a.bc)
        return
    for name in a.plan:
        print(f"--- plan {name} ---", flush=True)
        run_plan(name, tag=a.tag, max_dof=a.max_dof, levels=a.levels, dry=a.dry,
                 geoms=a.geoms, bcs=a.bcs, orders=a.orders)


if __name__ == "__main__":
    main()
