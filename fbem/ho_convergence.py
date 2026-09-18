"""Does a HIGHER-ORDER force density fix the O(h^0.3) stall that gate L2 found?

This is a STUDY, not a PASS/FAIL gate.  It reuses ``verify/verify_l2_head_to
_head.py``'s problem verbatim -- same box, same exterior slip-triangle exact
solution, same ``eps = 0.3 h``, same surface-displacement metric, same
``2*eps`` edge exclusion -- and sweeps the polynomial order of the force
density, ``p = 0`` (constant), ``1`` (linear) and ``2`` (quadratic), across a
mesh ladder.  The direct BIE is run on the same meshes so the comparison is
anchored to the same baseline L2 used.

THE PREDICTION UNDER TEST
-------------------------
Standard approximation theory says that on a QUASI-UNIFORM mesh the rate for a
solution with an edge singularity ``r^s`` is governed by ``s``, not by the
polynomial degree: raising ``p`` buys a smaller CONSTANT, not a better RATE,
and only mesh GRADING restores the rate.  The density of an indirect
single-layer Neumann formulation on a Lipschitz polyhedron has exactly such an
edge singularity.  If the prediction holds, P1/P2 on uniform meshes lower the
absolute error while leaving the ~h^0.3 rate roughly intact.  Measured below.

THE DECISIVE AXIS IS UNKNOWNS, NOT h
------------------------------------
P1 triples and P2 sextuples the unknown count on the same mesh (the density is
DISCONTINUOUS -- every element owns its own nodes, see ``assembly_ho``), and
the whole appeal of the force element was spending FEWER unknowns than the
direct BIE.  So error-vs-h flatters the higher orders and is reported only for
the rate fit; error-vs-unknowns is the comparison that decides.

THE ROWS (the piece that did not exist before this file)
--------------------------------------------------------
``assembly_ho`` gives the operators; the rows are written here.  With a nodal
density the free term is ``(1/2) sum_k N_k(x_c) q_{e,k}``, not ``(1/2) q``, so
the identity block is the matrix of shape-function values at the collocation
point.  Collocating at the nodes themselves would make that block exactly
``I``, but the P1/P2 nodes are vertices and edge midpoints -- they sit ON
element edges, where the free term is not 1/2, and on the polyhedron edges
where L2 says the formulation already stalls.  So this file collocates at
SHRUNK nodes, pulled toward the centroid in barycentric coordinates,

    lam_c = (1 - t) lam_node + t/3,      t = SHRINK (default 0.5)

which keeps every collocation point strictly inside a flat element at the price
of a dense ``K x K`` identity block ``N_k(x_c)``.  ``t = 0.5`` is the standard
discontinuous-element placement -- it puts the P1 nodes at barycentric
``(2/3, 1/6, 1/6)`` -- and it is also, measured, the smallest shrink at which
the higher orders stop being badly hurt by the free term, so the negative
conclusion below is not an artifact of handicapping them.  Section [F] sweeps
``t`` over 0.05 .. 0.85; the error falls monotonically toward P0's as ``t``
grows, but only because ``t -> 1`` collapses every collocation point onto the
centroid and the scheme degenerates into P0 (``cond(N)`` is printed so that
collapse is visible).  A separate check at ``t = 0.15`` and ``t = 0.5`` over
four meshes gave rates ``P1 +0.69 / +0.29`` and ``P2 +0.29 / +0.26`` -- the
absolute error moves a lot with ``t``, the verdict does not.

Section [G] is the one that explains the result, and [G3] is the one that
turns it into something actionable: with ``eps = 0.3 h`` the
mollification length is a sizeable fraction of an element, so (a) the
sub-element structure a P1/P2 density exists to represent is smoothed away by
the kernel, and (b) there is nowhere inside a triangle to put more than one
collocation point that is a full ``eps`` clear of the element boundary, where
the analytic free term ``1/2`` stops being right.  The printed table gives the
collocation-to-edge distance in units of ``eps`` for each order.  Shrink ``eps``
and the picture reverses: [G3] holds ``eps`` fixed in absolute terms, so every
run approximates the same mollified problem, and there P1 beats P0 at matched
unknown count.  So higher order is not useless for this formulation -- it is
specifically defeated by the production ``eps/h``.

    R1 free traction:   (1/2) N q_e + (B q)(x_c) = -t_F(x_c)
    R3 prescribed u:    (G q)(x_c) - (eps/4) M_e N q_e = -u_F(x_c)

with ``N`` the ``K x K`` shape-function matrix at the collocation points and
``M_e`` the per-element ``(eps/4)`` on-surface bias operator.  Rows are ordered
``3*K*e + 3*k + i`` so they line up with ``assembly_ho``'s columns, and the
five free faces come before the Dirichlet base, so both row groups are
contiguous slices.  R2 (interface) is not exercised here: L2's box has none.

``order = 0`` collapses to exactly ``model.ForceElementModel`` -- the shrink is
a no-op on a single centroid node, and the shape-function matrix is ``[[1]]``.
Check ``[A]`` below asserts that numerically against the numba path rather than
on the docstring's word, so the P0 column of this study is the same P0 that L2
measured.

Run (from the fbem root):

    /Users/meade/micromamba/bin/python ho_convergence.py --workers 8
    /Users/meade/micromamba/bin/python ho_convergence.py --quick --workers 8

Nothing here modifies ``assembly.py``, ``assembly_ho.py``, ``model.py``,
``verify/`` or anything in ``clq/``, ``msd/``, ``moss/`` or ``medt_paper/``.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (_ROOT / "fbem", _ROOT / "fbem" / "verify", _ROOT / "clq",
           _ROOT / "msd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import clq                                                          # noqa: E402
import assembly_ho as ah                                            # noqa: E402
import verify_l2_head_to_head as L2                                 # noqa: E402
from assembly import eps_bias_operator                              # noqa: E402
from model import Patch as FPatch, FREE, FIXED                      # noqa: E402
from mbem.backends.dense import AssembledDense                      # noqa: E402
from mbem.model import (BCType, Patch as MPatch, Region,            # noqa: E402
                        RegionModel, generate_system)
import mollified_bem as mb                                          # noqa: E402

SHRINK = 0.50          # barycentric pull of the collocation points to centroid
CACHE = _ROOT / "fbem" / "cache" / "ho_convergence.npz"

# mesh ladders, as nxy (the box has N_tri = 8 * nxy^2 and h = 40/nxy).
# P0 spans the whole unknown range so that every P1/P2 point can be compared
# against an INTERPOLATED P0 error, never an extrapolated one.
P0_NXY = (6, 8, 10, 12, 14, 16, 20, 24, 28, 30, 34)
P1_NXY = (6, 8, 10, 12, 14, 16)
P2_NXY = (6, 8, 10, 12, 14)
QUICK = {0: (6, 8, 10, 12, 14, 16, 20, 24), 1: (6, 8, 10, 12), 2: (6, 8, 10)}


# --------------------------------------------------------------------------
# collocation
# --------------------------------------------------------------------------
def shrunk_bary(order, t):
    """(K, 3) barycentric collocation points: nodes pulled toward the centroid."""
    p = int(order)
    if p == 0:
        return np.full((1, 3), 1.0 / 3.0)
    from clq.shape import lattice
    lam = lattice(p) / p
    return (1.0 - t) * lam + t / 3.0


def collocation(tv, order, t):
    """Collocation points (N, K, 3) and the K x K shape-function matrix.

    On flat (affine) triangles the Lagrange basis is a function of barycentric
    coordinates alone, so ``N_k(x_c)`` is the SAME matrix on every element.
    That is asserted here against ``clq.shape_functions`` on real elements
    rather than assumed.
    """
    lam = shrunk_bary(order, t)                       # (K, 3)
    x_c = np.einsum("kb,ebi->eki", lam, tv)           # (N, K, 3)
    mats = [clq.shape_functions(tv[e], order, x_c[e])
            for e in (0, len(tv) // 3, len(tv) - 1)]
    spread = max(float(np.abs(m - mats[0]).max()) for m in mats)
    if spread > 1e-11:
        raise AssertionError(f"shape-function matrix is element-dependent "
                             f"({spread:.2e}); the affine assumption is wrong")
    return x_c, np.ascontiguousarray(mats[0], float), spread


def edge_clearance(tv, order, t):
    """Distance from each collocation point to the nearest element edge.

    In barycentric terms the distance to edge ``i`` (the edge OPPOSITE vertex
    ``i``) is ``lam_i * height_i``, so the clearance is
    ``min_i lam_c[k, i] * 2 A / |edge_i|``.  Returns the (N, K) array.
    """
    lam = shrunk_bary(order, t)                                  # (K, 3)
    e = np.stack([tv[:, 2] - tv[:, 1], tv[:, 0] - tv[:, 2],
                  tv[:, 1] - tv[:, 0]], axis=1)                  # (N, 3, 3)
    twoA = np.linalg.norm(np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]),
                          axis=1)                                # (N,)
    height = twoA[:, None] / np.linalg.norm(e, axis=2)           # (N, 3)
    return (lam[None, :, :] * height[:, None, :]).min(axis=2)    # (N, K)


def centroid_shapes(tv, order):
    """(K,) shape-function values at the element centroid (element independent)."""
    c = tv[0].mean(axis=0)[None, :]
    return np.ascontiguousarray(clq.shape_functions(tv[0], order, c)[0], float)


# --------------------------------------------------------------------------
# the higher-order force-element solve
# --------------------------------------------------------------------------
def build_patches(faces, free_keys=L2.FREE_KEYS):
    pats = [FPatch(k, *faces[k], FREE if k in free_keys else FIXED,
                   L2.ORIENT[k]) for k in L2.KEYS]
    assert sum(p.flipped for p in pats) == 0, "orientation disagreement"
    # the free faces must precede the Dirichlet ones so both row groups are
    # contiguous slices of A.
    rows = [p.row for p in pats]
    assert rows == sorted(rows, key=lambda r: r != FREE), rows
    return pats


def ho_solve(faces, eps, order, t=SHRINK, workers=1, verbose=True,
             free_keys=L2.FREE_KEYS, eval_pts=None):
    """Solve rows R1 + R3 for a discontinuous Pp nodal force density.

    Returns a dict with the nodal density ``q`` (N, K, 3), the surface
    displacement at the FREE-element centroids (eps-bias corrected and raw),
    timings and the unknown count.
    """
    pats = build_patches(faces, free_keys)
    tv = np.ascontiguousarray(np.concatenate([p.tv for p in pats]))
    nrm = np.concatenate([p.normals for p in pats])
    cen = np.concatenate([p.centroids for p in pats])
    n_free = sum(p.n for p in pats if p.row == FREE)
    N, K = len(tv), ah.n_nodes(order)
    ndof = 3 * K * N
    x_c, Nmat, spread = collocation(tv, order, t)
    n_c = np.repeat(nrm, K, axis=0)

    t0 = time.time()
    A = np.empty((ndof, ndof), order="F")
    r0 = 3 * K * n_free
    if n_free:
        T = ah.traction_matrix_ho(x_c[:n_free].reshape(-1, 3), n_c[:K * n_free],
                                  tv, eps, L2.MU, L2.NU, order, workers=workers)
        A[:r0, :] = T
        del T
    if n_free < N:
        G = ah.displacement_matrix_ho(x_c[n_free:].reshape(-1, 3), tv, eps,
                                      L2.MU, L2.NU, order, workers=workers)
        A[r0:, :] = G
        del G
    t_asm = time.time() - t0
    b = np.empty(ndof)

    # R1: free term (1/2) N_k(x_c) q_{e,k}, and RHS -t_F = +t_exact
    Fblk = 0.5 * np.kron(Nmat, np.eye(3))
    for e in range(n_free):
        s = slice(3 * K * e, 3 * K * (e + 1))
        A[s, s] += Fblk
    if n_free:
        b[:r0] = L2.t_exact(x_c[:n_free].reshape(-1, 3),
                            n_c[:K * n_free]).ravel()

    # R3: subtract the (eps/4) on-surface self bias, then row-equilibrate
    if n_free < N:
        M = eps_bias_operator(nrm[n_free:], L2.MU, L2.NU)
        for j, e in enumerate(range(n_free, N)):
            s = slice(3 * K * e, 3 * K * (e + 1))
            A[r0 + 3 * K * j:r0 + 3 * K * (j + 1), s] -= (
                0.25 * eps * np.kron(Nmat, M[j]))
        b[r0:] = L2.u_exact(x_c[n_free:].reshape(-1, 3)).ravel()
        scale = 1.0 / np.abs(A[r0:, :]).max(axis=1)
        A[r0:, :] *= scale[:, None]
        b[r0:] *= scale

    t1 = time.time()
    anorm = float(np.linalg.norm(A, 1))
    lu, piv = lu_factor(A, overwrite_a=True)
    del A
    from scipy.linalg import get_lapack_funcs
    gecon = get_lapack_funcs(("gecon",), (lu,))[0]
    rcond, info = gecon(lu, anorm, norm="1")
    cond = (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
    q = lu_solve((lu, piv), b).reshape(N, K, 3)
    del lu
    t_solve = time.time() - t1

    # surface displacement at the FREE-element centroids (L2's metric points),
    # or at caller-supplied interior points (no on-surface bias there).
    t2 = time.time()
    Nc = centroid_shapes(tv, order)
    if eval_pts is None:
        ev = cen[:n_free]
        Gev = ah.displacement_matrix_ho(ev, tv, eps, L2.MU, L2.NU, order,
                                        workers=workers)
        u_raw = (Gev @ q.ravel()).reshape(-1, 3)
        del Gev
        q_cen = np.einsum("k,ekc->ec", Nc, q[:n_free])
        Mf = eps_bias_operator(nrm[:n_free], L2.MU, L2.NU)
        u_fe = u_raw - 0.25 * eps * np.einsum("eij,ej->ei", Mf, q_cen)
    else:
        ev = np.ascontiguousarray(eval_pts, float)
        Gev = ah.displacement_matrix_ho(ev, tv, eps, L2.MU, L2.NU, order,
                                        workers=workers)
        u_fe = u_raw = (Gev @ q.ravel()).reshape(-1, 3)
        del Gev
        q_cen = np.einsum("k,ekc->ec", Nc, q)
    t_eval = time.time() - t2
    if verbose:
        print(f"      p={order} N_tri={N} ndof={ndof} asm {t_asm:.1f}s "
              f"solve {t_solve:.1f}s eval {t_eval:.1f}s cond {cond:.2e}",
              flush=True)
    return dict(q=q, q_cen=q_cen, u=u_fe, u_raw=u_raw, c=ev,
                ndof=ndof, N=N, K=K, cond=cond, t_asm=t_asm,
                t_solve=t_solve, t_eval=t_eval, shape_spread=spread,
                Nmat=Nmat)


def p0_solve(faces, eps):
    """The P0 force element through the existing numba path (model.py)."""
    t0 = time.time()
    m, cond = L2.fe_solve(faces, eps)
    free = m.rows(FREE)
    c = m.centroids[free]
    u = m.displacement(c, self_elems=free, correct_eps=True)
    u_raw = m.displacement(c, self_elems=free, correct_eps=False)
    return dict(q=m.q[free][:, None, :], q_cen=m.q[free], u=u, u_raw=u_raw,
                c=c, ndof=3 * m.N, N=m.N, K=1, cond=cond,
                t_asm=time.time() - t0, t_solve=0.0, t_eval=0.0,
                shape_spread=0.0, Nmat=np.ones((1, 1)))


def direct_solve(faces, eps):
    """msd's direct BIE on the same mesh; also returns its matrix dimension."""
    t0 = time.time()
    meshes = {k: mb.TriMesh(v.copy(), t.copy()) for k, (v, t) in faces.items()}
    pats = {}
    for k, msh in meshes.items():
        c = msh.centroids()
        ns, _ = msh.normals_and_areas()
        if k in L2.FREE_KEYS:
            pats[k] = MPatch(k, msh, BCType.FREE_TRACTION,
                             value=L2.t_exact(c, ns))
        else:
            pats[k] = MPatch(k, msh, BCType.PRESCRIBED_DISPLACEMENT,
                             value=L2.u_exact(c))
    host = Region("host", mb.ElasticMaterial(mu=L2.MU, lam=L2.LAM),
                  [pats[k] for k in L2.KEYS],
                  probe_point=np.array([1.3, -2.1, -7.7]))
    model = RegionModel([host])
    for k in faces:
        assert model.orientation(host, pats[k]) == 1, f"{k}: sigma != +1"
    asm = AssembledDense(generate_system(model), eps, "direct",
                         jump="calibrated")
    ndof = int(asm.A.shape[0])
    sol = asm.solve()
    c = np.vstack([meshes[k].centroids() for k in L2.FREE_KEYS])
    u = np.vstack([sol[f"u:{k}"] for k in L2.FREE_KEYS])
    return dict(u=u, c=c, ndof=ndof, cond=float(asm.cond_estimate),
                t_asm=time.time() - t0, t_solve=0.0, t_eval=0.0)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def edge_concentration(u, ue, c, h, band=0.6):
    """(area fraction, error-energy fraction, concentration) within band*h."""
    d = L2.box_edge_distance(c)
    m = d < band * h
    e = ((u - ue) ** 2).sum(axis=1)
    af, ef = float(m.mean()), float(e[m].sum() / e.sum())
    return af, ef, (ef / af if af > 0 else np.nan)


def loglog_interp(x, y, xq):
    """log-log linear interpolation of y(x) at xq; None outside the range."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    o = np.argsort(lx)
    lx, ly = lx[o], ly[o]
    lq = np.log(float(xq))
    if lq < lx[0] or lq > lx[-1]:
        return None
    return float(np.exp(np.interp(lq, lx, ly)))


def fit(errs, xs):
    return float(np.polyfit(np.log(xs), np.log(errs), 1)[0])


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shrink", type=float, default=SHRINK)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-shrink-sweep", action="store_true")
    ap.add_argument("--orders", type=str, default="0,1,2")
    args = ap.parse_args()
    orders = tuple(int(s) for s in args.orders.split(","))
    ladder = {0: P0_NXY, 1: P1_NXY, 2: P2_NXY}
    if args.quick:
        ladder = QUICK
    ladder = {p: ladder[p] for p in orders}

    t_start = time.time()
    print(__doc__.split("Run (from")[0])
    print("=" * 78)
    print(f"shrink t = {args.shrink}, workers = {args.workers}, "
          f"eps = {L2.EPS_OVER_H} h, ladders " +
          ", ".join(f"p{p}: nxy {list(v)}" for p, v in ladder.items()))

    # ---- [A] the order-0 path really is model.ForceElementModel ----------
    print("\n[A] self-check: order = 0 through assembly_ho vs the numba "
          "model.py path")
    fa = L2.box_faces(40.0 / 8)
    ep = L2.EPS_OVER_H * 40.0 / 8
    a0 = ho_solve(fa, ep, 0, args.shrink, args.workers, verbose=False)
    b0 = p0_solve(fa, ep)
    du = np.abs(a0["u"] - b0["u"]).max() / np.abs(b0["u"]).max()
    dq = np.abs(a0["q_cen"] - b0["q_cen"]).max() / np.abs(b0["q_cen"]).max()
    print(f"    nxy = 8: surface u differs by {du:.2e} (relative), "
          f"density q by {dq:.2e}")
    print(f"    shape-function matrix element-independence: "
          f"{a0['shape_spread']:.2e}")
    if du > 1e-9:
        print("    *** the two P0 paths DISAGREE -- everything below is "
              "suspect ***")
    del a0, b0

    # ---- [B] the ladder ---------------------------------------------------
    print("\n[B] convergence ladder -- surface-displacement L2 at the "
          "free-element centroids,")
    print("    2*eps edge exclusion (the L2 gate metric), same exact solution.")
    print("    P0 runs through model.ForceElementModel's numba assembler, so "
          "its t_asm is\n    assemble + solve + evaluate lumped together and "
          "t_sol reads 0.  P1/P2 go\n    through clq's pure-numpy path (~50x "
          "slower per entry) and are split; 'raw' is\n    the same error "
          "without the removable (eps/4) on-surface bias correction.")
    res = {p: [] for p in ladder}
    dres = []
    done_direct = set()
    for p in sorted(ladder):
        print(f"\n    --- order p = {p} ---", flush=True)
        print(f"    {'nxy':>4s} {'N_tri':>6s} {'h':>5s} {'ndof':>7s} "
              f"{'L2 err':>9s} {'raw':>9s} {'cond':>9s} {'t_asm':>7s} "
              f"{'t_sol':>7s}")
        for nxy in ladder[p]:
            h = 2 * L2.L / nxy
            eps = L2.EPS_OVER_H * h
            faces = L2.box_faces(h)
            r = (p0_solve(faces, eps) if p == 0 else
                 ho_solve(faces, eps, p, args.shrink, args.workers,
                          verbose=False))
            ue = L2.u_exact(r["c"])
            excl = L2.box_edge_distance(r["c"]) > 2 * eps
            r.update(nxy=nxy, h=h, eps=eps,
                     err=L2.rel_l2(r["u"], ue, excl),
                     err_all=L2.rel_l2(r["u"], ue),
                     err_raw=L2.rel_l2(r["u_raw"], ue, excl))
            af, ef, conc = edge_concentration(r["u"], ue, r["c"], h)
            r.update(edge_area=af, edge_energy=ef, edge_conc=conc)
            qn = np.linalg.norm(r["q_cen"], axis=1)
            de = L2.box_edge_distance(r["c"])
            bands = [de < 0.6 * h, (de >= 0.6 * h) & (de < 2 * h), de >= 2 * h]
            r["q_bands"] = [float(np.sqrt((qn[b] ** 2).mean())
                                  / np.sqrt((qn ** 2).mean())) for b in bands]
            r["q_max"] = float(np.abs(r["q"]).max())
            for k in ("q", "u", "u_raw", "c", "Nmat", "q_cen"):
                r.pop(k, None)
            res[p].append(r)
            print(f"    {nxy:4d} {r['N']:6d} {h:5.2f} {r['ndof']:7d} "
                  f"{r['err']:9.4f} {r['err_raw']:9.4f} {r['cond']:9.2e} "
                  f"{r['t_asm']:7.1f} {r['t_solve']:7.1f}", flush=True)
            if nxy not in done_direct:
                d = direct_solve(faces, eps)
                ued = L2.u_exact(d["c"])
                ex = L2.box_edge_distance(d["c"]) > 2 * eps
                d.update(nxy=nxy, h=h, err=L2.rel_l2(d["u"], ued, ex),
                         err_all=L2.rel_l2(d["u"], ued))
                af, ef, conc = edge_concentration(d["u"], ued, d["c"], h)
                d.update(edge_area=af, edge_energy=ef, edge_conc=conc)
                d.pop("u"); d.pop("c")
                dres.append(d)
                done_direct.add(nxy)

    dres.sort(key=lambda d: d["nxy"])
    print("\n    --- direct BIE baseline, same meshes and metric ---")
    print(f"    {'nxy':>4s} {'h':>5s} {'ndof':>7s} {'L2 err':>9s} "
          f"{'cond':>9s} {'t_asm+sol':>10s}")
    for d in dres:
        print(f"    {d['nxy']:4d} {d['h']:5.2f} {d['ndof']:7d} "
              f"{d['err']:9.4f} {d['cond']:9.2e} {d['t_asm']:10.1f}")

    # ---- [C] rates --------------------------------------------------------
    print("\n[C] fitted convergence rates (least squares on log-log)")
    print(f"    {'config':>12s} {'meshes':>7s} {'rate in h':>10s} "
          f"{'rate in ndof':>13s} {'err range':>22s}")
    rates = {}
    for p in sorted(res):
        e = [r["err"] for r in res[p]]
        hs = [r["h"] for r in res[p]]
        nd = [r["ndof"] for r in res[p]]
        rates[f"P{p}"] = (fit(e, hs), fit(e, nd))
        print(f"    {'force P' + str(p):>12s} {len(e):7d} "
              f"{fit(e, hs):+10.2f} {fit(e, nd):+13.2f} "
              f"{e[0]:.4f} -> {e[-1]:.4f}")
    e = [d["err"] for d in dres]
    hs = [d["h"] for d in dres]
    nd = [d["ndof"] for d in dres]
    rates["direct"] = (fit(e, hs), fit(e, nd))
    print(f"    {'direct BIE':>12s} {len(e):7d} {fit(e, hs):+10.2f} "
          f"{fit(e, nd):+13.2f} {e[0]:.4f} -> {e[-1]:.4f}")
    print("    (rate in ndof = rate in h / -2 for a fixed order, because "
          "ndof ~ h^-2)")

    # ---- [D] THE DECISIVE TABLE: error at matched unknown count ----------
    print("\n[D] ERROR AT MATCHED UNKNOWN COUNT -- the decisive comparison.")
    print("    For each P1/P2 mesh: the P0 error INTERPOLATED (log-log, "
          "between the two\n    bracketing P0 meshes -- never extrapolated) at "
          "the SAME ndof, and the same\n    for the direct BIE.  ratio < 1 "
          "means higher order wins at equal cost.")
    print(f"    {'config':>8s} {'nxy':>4s} {'ndof':>7s} {'err':>9s} | "
          f"{'P0 @ ndof':>10s} {'ratio':>7s} | {'dir @ ndof':>10s} "
          f"{'ratio':>7s}")
    p0nd = [r["ndof"] for r in res.get(0, [])]
    p0er = [r["err"] for r in res.get(0, [])]
    matched = []
    for p in sorted(res):
        if p == 0:
            continue
        for r in res[p]:
            i0 = loglog_interp(p0nd, p0er, r["ndof"]) if p0nd else None
            id_ = loglog_interp(nd, e, r["ndof"])
            s0 = f"{i0:10.4f} {r['err']/i0:7.3f}" if i0 else f"{'--':>10s} {'--':>7s}"
            sd = f"{id_:10.4f} {r['err']/id_:7.3f}" if id_ else f"{'--':>10s} {'--':>7s}"
            print(f"    {'P' + str(p):>8s} {r['nxy']:4d} {r['ndof']:7d} "
                  f"{r['err']:9.4f} | {s0} | {sd}")
            matched.append((p, r["nxy"], r["ndof"], r["err"], i0, id_))

    # ---- [E] edge concentration ------------------------------------------
    print("\n[E] does the error still pile up at the box edges?")
    print("    error-energy density within 0.6h of an edge, divided by its "
          "area fraction\n    (the P0 diagnosis in RESULTS.md was 3.4x; "
          "direct BIE 2.2x)")
    print(f"    {'config':>8s} " + " ".join(f"{'nxy=' + str(n):>9s}"
                                            for n in sorted(done_direct)))
    for p in sorted(res):
        row = {r["nxy"]: r["edge_conc"] for r in res[p]}
        print(f"    {'P' + str(p):>8s} " +
              " ".join(f"{row.get(n, float('nan')):9.2f}"
                       for n in sorted(done_direct)))
    row = {d["nxy"]: d["edge_conc"] for d in dres}
    print(f"    {'direct':>8s} " +
          " ".join(f"{row.get(n, float('nan')):9.2f}"
                   for n in sorted(done_direct)))
    print("\n    force density |q(centroid)| by distance to an edge, rms over "
          "the band /\n    rms over all elements  [d<0.6h, 0.6-2h, >2h]:")
    for p in sorted(res):
        for r in res[p]:
            print(f"      P{p} nxy={r['nxy']:3d}: " +
                  "  ".join(f"{v:.2f}" for v in r["q_bands"]) +
                  f"   max|q_nodal| = {r['q_max']:.3e}")

    # ---- [F] shrink sensitivity ------------------------------------------
    sweep = []
    hi_orders = sorted(x for x in orders if x > 0)
    if not args.no_shrink_sweep and hi_orders:
        print("\n[F] collocation-shrink sensitivity (nxy = 10, the middle of "
              "the ladder).")
        print("    t -> 1 collapses every collocation point onto the centroid, "
              "so the K x K\n    shape matrix N becomes singular and the scheme "
              "degenerates toward P0;\n    cond(N) is printed so that collapse "
              "is visible rather than mistaken for a win.")
        h = 2 * L2.L / 10
        eps = L2.EPS_OVER_H * h
        faces = L2.box_faces(h)
        b0 = p0_solve(faces, eps)
        ue = L2.u_exact(b0["c"])
        excl = L2.box_edge_distance(b0["c"]) > 2 * eps
        print(f"    P0 reference on this mesh: L2 = "
              f"{L2.rel_l2(b0['u'], ue, excl):.4f}")
        for p in hi_orders:
            for t in (0.05, 0.15, 0.30, 0.50, 0.70, 0.85):
                r = ho_solve(faces, eps, p, t, args.workers, verbose=False)
                err = L2.rel_l2(r["u"], ue, excl)
                cN = float(np.linalg.cond(r["Nmat"]))
                clr = edge_clearance(np.concatenate([q.tv for q in
                                                     build_patches(faces)]),
                                     p, t).min() / eps
                print(f"    P{p} t = {t:4.2f}: L2 = {err:.4f}  "
                      f"cond(A) = {r['cond']:.2e}  cond(N) = {cN:7.2f}  "
                      f"min edge clearance = {clr:.2f} eps", flush=True)
                sweep.append((p, t, err, r["cond"], cN, clr))

    # ---- [G] WHY: the mollification length is the limiter ------------------
    print("\n[G] mechanism -- is eps/h what stops higher order from paying?")
    print("    (G1) collocation-point clearance from the element boundary, in "
          "units of eps,\n    at the default t; the analytic free term 1/2 is "
          "only right well clear of it.")
    h10 = 2 * L2.L / 10
    faces = L2.box_faces(h10)
    tv10 = np.concatenate([p.tv for p in build_patches(faces)])
    for p in sorted(orders):
        c = edge_clearance(tv10, p, args.shrink) / (L2.EPS_OVER_H * h10)
        print(f"      P{p}: min {c.min():.2f} eps, median {np.median(c):.2f} "
              f"eps  ({ah.n_nodes(p)} points per element)")
    print("    (G2) eps/h sweep at FIXED h = 5.0 (nxy = 8), same metric and "
          "the same\n    2*eps_ref edge exclusion at every eps so the scored "
          "points do not move.")
    eps_rows = []
    nxy8, h8 = 8, 2 * L2.L / 8
    faces = L2.box_faces(h8)
    ex_ref = None
    print(f"      {'eps/h':>7s} " +
          " ".join(f"{'P' + str(p):>9s}" for p in sorted(orders)))
    for r in (0.30, 0.15, 0.05, 0.02):
        row = [r]
        for p in sorted(orders):
            s = (p0_solve(faces, r * h8) if p == 0 else
                 ho_solve(faces, r * h8, p, args.shrink, args.workers,
                          verbose=False))
            if ex_ref is None:
                ue8 = L2.u_exact(s["c"])
                ex_ref = L2.box_edge_distance(s["c"]) > 2 * L2.EPS_OVER_H * h8
            row.append(L2.rel_l2(s["u"], ue8, ex_ref))
        eps_rows.append(row)
        print(f"      {r:7.3f} " + " ".join(f"{v:9.4f}" for v in row[1:]),
              flush=True)

    print("    (G3) the matched-unknown question again, but at a SMALL "
          "mollification:\n    eps is held at 0.25 in ABSOLUTE terms for every "
          "run, so all of them\n    approximate the SAME mollified problem and "
          "only the discretisation differs.\n    P1/P2 stay on the coarse "
          "nxy = 8 mesh; P0 is refined to match their ndof.")
    g3 = []
    ue8 = ex8 = None
    print(f"      {'cfg':>4s} {'nxy':>4s} {'eps/h':>6s} {'ndof':>6s} "
          f"{'err':>8s}")
    for p, nxys in ((0, (8, 10, 14, 16, 20, 24, 28)), (1, (8,)), (2, (8,))):
        if p not in orders:
            continue
        for nxy in nxys:
            h = 2 * L2.L / nxy
            fa = L2.box_faces(h)
            s = (p0_solve(fa, 0.25) if p == 0 else
                 ho_solve(fa, 0.25, p, args.shrink, args.workers,
                          verbose=False))
            uu = L2.u_exact(s["c"])
            ex = L2.box_edge_distance(s["c"]) > 2 * L2.EPS_OVER_H * h8
            err = L2.rel_l2(s["u"], uu, ex)
            g3.append((p, nxy, s["ndof"], err))
            print(f"      P{p:<3d} {nxy:4d} {0.25 / h:6.3f} {s['ndof']:6d} "
                  f"{err:8.4f}", flush=True)
    print("      CAVEAT: eps = 0.25 puts an O(eps) floor under every row here "
          "(P0 is still\n      falling at nxy = 28, so the floor is near but "
          "not reached); a comparison\n      made close to a common floor is "
          "compressed, not inflated.")

    # ---- [H] all-Dirichlet control ---------------------------------------
    print("\n[H] all-Dirichlet control (R3 rows only -- no free term, no "
          "Neumann row),")
    print("    interior-grid L2 on L2's 144 points.  This is where the force "
          "element is\n    4-8x BETTER than the direct BIE, so it asks whether "
          "higher order pays\n    anywhere in this formulation.")
    pts = L2.interior_grid()
    ue_int = L2.u_exact(pts)
    dir_rows = []
    print(f"      {'nxy':>4s} {'h':>5s} " +
          " ".join(f"{'P' + str(p):>11s} {'ndof':>7s}" for p in sorted(orders)))
    d_nxy = (6, 8, 10, 12) if not args.quick else (6, 8, 10)
    for nxy in d_nxy:
        h = 2 * L2.L / nxy
        eps = L2.EPS_OVER_H * h
        fa = L2.box_faces(h)
        row = [nxy, h]
        for p in sorted(orders):
            s = ho_solve(fa, eps, p, args.shrink, args.workers, verbose=False,
                         free_keys=(), eval_pts=pts)
            row += [L2.rel_l2(s["u"], ue_int), s["ndof"]]
        dir_rows.append(row)
        print(f"      {nxy:4d} {h:5.2f} " +
              " ".join(f"{row[2 + 2 * i]:11.3e} {row[3 + 2 * i]:7d}"
                       for i in range(len(orders))), flush=True)
    for i, p in enumerate(sorted(orders)):
        e = [r[2 + 2 * i] for r in dir_rows]
        print(f"      P{p} rate in h {fit(e, [r[1] for r in dir_rows]):+.2f}, "
              f"in ndof {fit(e, [r[3 + 2 * i] for r in dir_rows]):+.2f}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    for p in sorted(res):
        rh, rn = rates[f"P{p}"]
        print(f"  force element P{p}: rate O(h^{rh:+.2f}) over "
              f"{len(res[p])} meshes, error "
              f"{res[p][0]['err']:.4f} -> {res[p][-1]['err']:.4f}")
    rh, rn = rates["direct"]
    print(f"  direct BIE       : rate O(h^{rh:+.2f}) over {len(dres)} meshes, "
          f"error {dres[0]['err']:.4f} -> {dres[-1]['err']:.4f}")
    if matched:
        avail = [m for m in matched if m[4] is not None]
        good = [m for m in avail if m[3] < m[4]]
        worst = max((m[3] / m[4] for m in avail), default=float("nan"))
        best = min((m[3] / m[4] for m in avail), default=float("nan"))
        print(f"  at MATCHED unknown count, higher order beats P0 in "
              f"{len(good)} of {len(avail)} comparisons; "
              f"err(HO)/err(P0 @ same ndof) spans {best:.2f} - {worst:.2f}")
    print(f"\n  runtime {time.time() - t_start:.0f} s")

    np.savez(CACHE,
             **{f"p{p}_{k}": np.array([r[k] for r in res[p]])
                for p in res for k in ("nxy", "N", "h", "ndof", "err",
                                       "err_all", "err_raw", "cond",
                                       "edge_conc", "t_asm", "t_solve")},
             **{f"direct_{k}": np.array([d[k] for d in dres])
                for k in ("nxy", "h", "ndof", "err", "err_all", "cond",
                          "edge_conc", "t_asm")},
             shrink=args.shrink,
             sweep=np.array(sweep) if sweep else np.zeros((0, 6)),
             eps_sweep=np.array(eps_rows),
             fixed_eps=np.array(g3) if g3 else np.zeros((0, 4)),
             dirichlet=np.array(dir_rows),
             matched=np.array([[m[0], m[1], m[2], m[3],
                                np.nan if m[4] is None else m[4],
                                np.nan if m[5] is None else m[5]]
                               for m in matched]) if matched
             else np.zeros((0, 6)))
    print(f"  numbers cached to {CACHE}")


if __name__ == "__main__":
    main()
