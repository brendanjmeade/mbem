"""Does algebraic mesh grading toward the box EDGES repair the O(h^0.31) stall?

Scratch experiment (nothing under moss-org/ is modified).  Reuses
fbem/verify/verify_l2_head_to_head.py's problem verbatim -- same box, same
exterior slip-triangle exact solution, same metric region -- but replaces the
uniform structured face meshes with TENSOR-PRODUCT ALGEBRAICALLY GRADED ones:
each face parameter t in [0, 1] is mapped two-sidedly by t -> (2t)^beta / 2,
so nodes cluster toward every box edge (and, by the product, more strongly
toward every box corner).  beta = 1 reproduces the uniform mesh exactly.

Because grading makes the elements wildly non-uniform, three things change
relative to L2:
  * eps is per element (eps_arr), with several policies;
  * the R3 (eps/4) self-bias and the on-surface bias correction use the
    per-element eps, not the scalar (model.py uses the scalar);
  * the score is AREA-WEIGHTED over a FIXED physical band (d > 3.0 from every
    edge), so refining the edge region cannot flatter or punish the metric by
    changing the point density.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve, get_lapack_funcs

ROOT = pathlib.Path("/Users/meade/Desktop/moss-org")
for _p in (ROOT / "fbem", ROOT / "fbem" / "verify", ROOT / "clq", ROOT / "msd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import verify_l2_head_to_head as L2                                # noqa: E402
from model import ForceElementModel, Patch as FPatch, FREE, FIXED  # noqa: E402
from assembly import (displacement_matrix, eps_bias_operator,      # noqa: E402
                      apply_displacement)

L, D = L2.L, L2.D
BAND = 3.0          # fixed physical exclusion band from every box edge


# --------------------------------------------------------------------- mesh
def graded_nodes(n, beta):
    """n+1 nodes on [0,1], clustered toward BOTH ends; beta=1 is uniform."""
    assert n % 2 == 0, "n must be even for a symmetric two-sided grading"
    half = n // 2
    s = np.arange(half + 1) / half
    t = 0.5 * s ** beta
    return np.concatenate([t, 1.0 - t[::-1][1:]])


def bl_nodes(n, nlayer, sigma):
    """Uniform n intervals on [0,1], plus a geometric BOUNDARY LAYER at each
    end: the first and last uniform interval gain `nlayer` extra nodes at
    H*sigma^j.  Interior resolution is untouched, so this ADDS unknowns at the
    edges rather than moving them there."""
    u = np.arange(n + 1) / n
    if nlayer <= 0:
        return u
    H = 1.0 / n
    extra = np.array([H * sigma ** j for j in range(1, nlayer + 1)])
    return np.unique(np.concatenate([u, extra, 1.0 - extra]))


def quad_face_g(origin, e1, e2, a, b):
    """L2.quad_face with arbitrary parameter node vectors a, b."""
    A, B = np.meshgrid(a, b, indexing="ij")
    V = (np.asarray(origin, float)[None, None, :]
         + A[..., None] * np.asarray(e1, float)[None, None, :]
         + B[..., None] * np.asarray(e2, float)[None, None, :])
    n1, n2 = len(a) - 1, len(b) - 1
    idx = lambda i, j: i * (n2 + 1) + j                          # noqa: E731
    tris = [t for i in range(n1) for j in range(n2)
            for t in ([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)],
                      [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])]
    return V.reshape(-1, 3), np.array(tris, int)


def box_faces_g(nxy, nz, beta, bl=None):
    """The same six outward-wound faces as L2.box_faces, graded to the edges.

    `bl = (nlayer, sigma)` switches to boundary-layer refinement instead of
    global algebraic grading."""
    if bl is not None:
        gxy, gz = bl_nodes(nxy, *bl), bl_nodes(nz, *bl)
    else:
        gxy, gz = graded_nodes(nxy, beta), graded_nodes(nz, beta)
    X = 2 * L
    return {
        "top":  quad_face_g((-L, -L, 0.0), (X, 0, 0), (0, X, 0), gxy, gxy),
        "base": quad_face_g((-L, -L, -D), (0, X, 0), (X, 0, 0), gxy, gxy),
        "xp":   quad_face_g((L, -L, -D), (0, X, 0), (0, 0, D), gxy, gz),
        "xm":   quad_face_g((-L, -L, -D), (0, 0, D), (0, X, 0), gz, gxy),
        "yp":   quad_face_g((-L, L, -D), (0, 0, D), (X, 0, 0), gz, gxy),
        "ym":   quad_face_g((-L, -L, -D), (X, 0, 0), (0, 0, D), gxy, gz)}


def elem_size(tv):
    """Per-element length scale: equals h on L2's uniform right-isoceles mesh.

    Uses sqrt(2*area), which for legs (h, h) gives exactly h.  Also returns the
    smallest altitude (the dimension eps must fit inside).
    """
    a = tv[:, 1] - tv[:, 0]
    b = tv[:, 2] - tv[:, 0]
    cr = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    e = np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 1], tv[:, 0] - tv[:, 2]],
                 axis=1)
    lmax = np.linalg.norm(e, axis=2).max(axis=1)
    return np.sqrt(2.0 * area), 2.0 * area / lmax, area


# ------------------------------------------------------------------- solver
def assemble_var_eps(m, t_F, u_F):
    """model.ForceElementModel.assemble, but the R3 bias uses eps PER ELEMENT."""
    N = m.N
    A = m.traction_block(verbose=False)
    assert A.flags.f_contiguous
    b = np.zeros(3 * N)
    free, fixed = m.rows(FREE), m.rows(FIXED)
    d = m._dof(free)
    A[d, d] += 0.5
    b[d] = -np.asarray(t_F)[free].ravel()
    if len(fixed):
        df = m._dof(fixed)
        G = displacement_matrix(m.centroids[fixed], m.tv, m.eps_arr, m.mu, m.nu)
        M = eps_bias_operator(m.normals[fixed], m.mu, m.nu)
        for k, e in enumerate(fixed):
            G[3 * k:3 * k + 3, 3 * e:3 * e + 3] -= 0.25 * m.eps_arr[e] * M[k]
        s = 1.0 / np.abs(G).max(axis=1)
        A[df, :] = G * s[:, None]
        b[df] = -np.asarray(u_F)[fixed].ravel() * s
        del G
    m.A, m.b = A, b
    return A, b


def displacement_var_eps(m, obs, self_elems):
    u = apply_displacement(obs, m.tv, m.eps_arr, m.mu, m.nu, m.q, chunk=2048)
    e = np.asarray(self_elems)
    M = eps_bias_operator(m.normals[e], m.mu, m.nu)
    u -= 0.25 * m.eps_arr[e][:, None] * np.einsum("mij,mj->mi", M, m.q[e])
    return u


def solve_graded(nxy, nz, beta, eps_policy, eps_const=0.5, eps_fac=0.3):
    faces = box_faces_g(nxy, nz, beta)
    patches = [FPatch(k, *faces[k], FREE if k in L2.FREE_KEYS else FIXED,
                      L2.ORIENT[k]) for k in L2.KEYS]
    assert sum(p.flipped for p in patches) == 0
    m = ForceElementModel(patches, L2.MU, L2.NU, 1.0, 1.0)
    hloc, halt, area = elem_size(m.tv)
    if eps_policy == "local":
        m.eps_arr = eps_fac * hloc
    elif eps_policy == "global":
        m.eps_arr = np.full(m.N, eps_fac * hloc.max())
    elif eps_policy == "fixed":
        m.eps_arr = np.full(m.N, eps_const)
    elif eps_policy == "capped":      # fixed physical eps, but never > 0.6*h
        m.eps_arr = np.minimum(eps_const, 0.6 * hloc)
    else:
        raise ValueError(eps_policy)
    c, n = m.centroids, m.normals
    t0 = time.time()
    assemble_var_eps(m, -L2.t_exact(c, n), -L2.u_exact(c))
    anorm = float(np.linalg.norm(m.A, 1))
    lu, piv = lu_factor(m.A, overwrite_a=True)
    m.A = None
    gecon = get_lapack_funcs(("gecon",), (lu,))[0]
    rcond, info = gecon(lu, anorm, norm="1")
    cond = (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
    m.q = lu_solve((lu, piv), m.b).reshape(m.N, 3)
    del lu
    free = m.rows(FREE)
    u = displacement_var_eps(m, m.centroids[free], free)
    return dict(m=m, u=u, free=free, cond=cond, hloc=hloc, halt=halt,
                area=area, t=time.time() - t0,
                h_max=hloc.max(), h_min=hloc.min(), N=m.N)


def score(r, band=None):
    """Area-weighted and plain relative L2 on the FIXED band d > BAND."""
    m, free, u = r["m"], r["free"], r["u"]
    c = m.centroids[free]
    ue = L2.u_exact(c)
    de = L2.box_edge_distance(c)
    w = r["area"][free]
    sel = de > (BAND if band is None else band)
    e2 = ((u - ue) ** 2).sum(axis=1)
    n2 = (ue ** 2).sum(axis=1)
    aw = float(np.sqrt((w[sel] * e2[sel]).sum() / (w[sel] * n2[sel]).sum()))
    pl = float(np.sqrt(e2[sel].sum() / n2[sel].sum()))
    return aw, pl, int(sel.sum())


BANDS = (0.0, 0.5, 1.0, 3.0)


def bands_table(betas, ladders, eps_policy, eps_const=0.5, label=""):
    """Area-weighted L2 on several FIXED physical bands, vs ndof."""
    print(f"\n=== {label} (policy {eps_policy}"
          + (f", eps={eps_const}" if eps_policy in ("fixed", "capped") else "")
          + ") ===")
    hdr = " ".join(f"{'d>' + str(b):>8s}" for b in BANDS)
    print(f"  {'beta':>4s} {'ndof':>6s} {'h_max':>6s} {'h_min':>7s} "
          f"{'eps_mn':>7s} {'eps_mx':>7s} {hdr} {'cond':>9s}")
    res = {}
    for beta in betas:
        rows = []
        for nxy, nz in ladders:
            r = solve_graded(nxy, nz, beta, eps_policy, eps_const)
            e = [score(r, b)[0] for b in BANDS]
            ea = r["m"].eps_arr
            print(f"  {beta:4.1f} {3*r['N']:6d} {r['h_max']:6.2f} "
                  f"{r['h_min']:7.4f} {ea.min():7.4f} {ea.max():7.3f} "
                  + " ".join(f"{v:8.4f}" for v in e)
                  + f" {r['cond']:9.2e}", flush=True)
            rows.append((3 * r["N"], r["h_max"], e))
            del r
        res[beta] = rows
        nd = [x[0] for x in rows]
        hm = [x[1] for x in rows]
        rt_n = [fit([x[2][i] for x in rows], nd) for i in range(len(BANDS))]
        rt_h = [fit([x[2][i] for x in rows], hm) for i in range(len(BANDS))]
        print(f"        rate vs ndof: " + " ".join(f"{v:+8.2f}" for v in rt_n)
              + f"   (vs h_max: " + " ".join(f"{v:+.2f}" for v in rt_h) + ")")
    return res


def fit(err, h):
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


# ------------------------------------------------------------------- driver
def ladder(betas, ladders, eps_policy, eps_const=0.5, label=""):
    print(f"\n=== {label}  (eps policy = {eps_policy}"
          + (f", eps = {eps_const}" if eps_policy in ("fixed", "capped") else "")
          + ") ===")
    print(f"  {'beta':>4s} {'nxy':>4s} {'N':>6s} {'ndof':>6s} {'h_max':>6s} "
          f"{'h_min':>7s} {'eps_mn':>7s} {'eps_mx':>7s} {'L2_area':>8s} "
          f"{'L2_pts':>8s} {'cond':>9s} {'s':>5s}")
    out = {}
    for beta in betas:
        rows = []
        for nxy, nz in ladders:
            r = solve_graded(nxy, nz, beta, eps_policy, eps_const)
            aw, pl, npts = score(r)
            ea = r["m"].eps_arr
            print(f"  {beta:4.1f} {nxy:4d} {r['N']:6d} {3*r['N']:6d} "
                  f"{r['h_max']:6.2f} {r['h_min']:7.4f} {ea.min():7.4f} "
                  f"{ea.max():7.3f} {aw:8.4f} {pl:8.4f} {r['cond']:9.2e} "
                  f"{r['t']:5.0f}", flush=True)
            rows.append((r["h_max"], 3 * r["N"], aw, pl, r))
        out[beta] = rows
        if len(rows) > 1:
            h = [x[0] for x in rows]; nd = [x[1] for x in rows]
            print(f"        beta = {beta}: rate in h_max {fit([x[2] for x in rows], h):+.2f}"
                  f"  |  in ndof {fit([x[2] for x in rows], nd):+.2f}"
                  f"  (unweighted in h_max {fit([x[3] for x in rows], h):+.2f})")
    return out


def density_exponent(r, face="top"):
    """Fit log|q| vs log(distance to the nearest box edge) on one free face."""
    m = r["m"]
    sl = m.slices[face]
    c = m.centroids[sl]
    q = np.linalg.norm(m.q[sl], axis=1)
    de = L2.box_edge_distance(c)
    # stay away from the corners: use only points whose distance to the
    # nearest CORNER is > 3x their distance to the nearest edge
    cor = np.array([(x, y, 0.0) for x in (-L, L) for y in (-L, L)])
    dc = np.linalg.norm(c[:, None, :] - cor[None, :, :], axis=2).min(axis=1)
    ok = (dc > 3 * de) & (de > 0)
    lo, hi = np.percentile(de[ok], [0, 60])
    sel = ok & (de <= hi)
    s = float(np.polyfit(np.log(de[sel]), np.log(q[sel]), 1)[0])
    return s, int(sel.sum()), de[sel].min(), de[sel].max()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick")
    a = ap.parse_args()
    t0 = time.time()
    print(__doc__.split("Because")[0])

    if a.mode == "quick":
        r = solve_graded(8, 8, 1.0, "local")
        aw, pl, npts = score(r)
        print(f"sanity beta=1 nxy=8: N={r['N']} h={r['h_max']:.2f} "
              f"L2_area={aw:.4f} L2_pts={pl:.4f} cond={r['cond']:.2e} "
              f"t={r['t']:.0f}s  (L2 gate FIX column at h=5: FE 0.0466)")
        r2 = solve_graded(8, 8, 2.0, "local")
        aw2, pl2, _ = score(r2)
        print(f"        beta=2 nxy=8: N={r2['N']} h_max={r2['h_max']:.2f} "
              f"h_min={r2['h_min']:.3f} L2_area={aw2:.4f} t={r2['t']:.0f}s")
    print(f"\ntotal {time.time()-t0:.0f}s")
