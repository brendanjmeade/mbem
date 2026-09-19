"""Sharpest manufactured solution: a UNIFORM strain field.

u(x) = A x with A symmetric => sigma = C:A is constant, so the exact boundary
traction is piecewise constant (exactly representable by P0) and the exact
interior field is linear.  Any error is the method's, not the density space's.

Dirichlet: prescribe u = A x, solve for t, compare with the constant sigma.n.
Neumann (all free): prescribe t = sigma.n, solve for u with
jump='calibrated', deflate=True; compare with A x up to a rigid motion.
"""
import sys
import numpy as np
SB = ("/private/tmp/claude-501/-Users-meade-Desktop-moss-org/"
      "bb9d3cdf-84ba-4d36-8fdb-70f953c90d80/scratchpad/sb/base/msd")
sys.path.insert(0, SB)
import mollified_bem as mb
from mbem.model import BCType, Patch, Region, RegionModel, generate_system
from mbem.backends.dense import AssembledDense
from mbem.evaluate import evaluate_displacement
from mbem.kernels import basis as kb

MAT = mb.ElasticMaterial(mu=30.0, lam=20.0)
MU, LAM = MAT.mu, MAT.lam
A = np.array([[2e-4, 1e-4, 0.0], [1e-4, -1e-4, 5e-5], [0.0, 5e-5, 3e-4]])
SIG = LAM * np.trace(A) * np.eye(3) + 2 * MU * A


def icosphere(R=1.0, n_sub=2):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    for _ in range(n_sub):
        mid, newF, Vl = {}, [], list(V)

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                p = (Vl[a] + Vl[b]) / 2
                Vl.append(p / np.linalg.norm(p)); mid[k] = len(Vl) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.array(Vl); F = np.array(newF)
    return mb.TriMesh(vertices=R * V, triangles=F)


def rigid_fit(u, x):
    """Remove the best-fit rigid motion (translation + infinitesimal rotation)."""
    M = np.zeros((u.size, 6))
    for k in range(3):
        M[k::3, k] = 1.0
    # rotation generators
    gens = [np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float),
            np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], float)]
    for j, G in enumerate(gens):
        M[:, 3 + j] = (x @ G.T).ravel()
    c, *_ = np.linalg.lstsq(M, u.ravel(), rcond=None)
    return (u.ravel() - M @ c).reshape(-1, 3)


obs = np.array([[3.0, 2.0, 1.0], [-4.0, 1.0, -2.0], [0.0, 5.0, 0.0]])
u_int_ex = obs @ A.T

print("  uniform-strain manufactured solution on a sphere R = 10 "
      "(exact traction is piecewise constant)")
print(f"{'N':>5s} {'eps/h':>6s} {'jump':>11s} {'Dir t med':>10s} "
      f"{'Dir int u':>10s} | {'Neu u med':>10s} {'Neu int u':>10s}")
for n_sub in (1, 2, 3):
    sph = icosphere(10.0, n_sub); sph.ensure_outward_normals()
    N = sph.n_triangles
    h = float(kb.element_sizes(sph).mean())
    cb = np.asarray(sph.centroids(), float)
    nb, _ = sph.normals_and_areas()
    u_b = cb @ A.T
    t_b = nb @ SIG.T
    for r in (1.25, 0.5, 0.3, 0.15):
        eps = r * h
        out = {}
        for tag, bt, val, jump, defl in (
                ("Dh", BCType.PRESCRIBED_DISPLACEMENT, u_b, "half", False),
                ("Dc", BCType.PRESCRIBED_DISPLACEMENT, u_b, "calibrated", False),
                ("Nc", BCType.FREE_TRACTION, t_b, "calibrated", True)):
            bnd = Patch("sphere", sph, bt, value=val)
            reg = Region("body", MAT, [bnd], probe_point=np.zeros(3))
            model = RegionModel([reg]); sysm = generate_system(model)
            try:
                asm = AssembledDense(sysm, eps, "direct", jump=jump, deflate=defl)
                sol = asm.solve()
            except Exception as e:
                out[tag] = (np.nan, np.nan); continue
            if bt is BCType.PRESCRIBED_DISPLACEMENT:
                got, ref = sol["t:sphere"], t_b
                d = np.linalg.norm(got - ref, axis=1)
            else:
                got = rigid_fit(sol["u:sphere"], cb)
                ref = rigid_fit(u_b, cb)
                d = np.linalg.norm(got - ref, axis=1)
            rel = np.median(d) / np.sqrt((np.linalg.norm(ref, axis=1) ** 2).mean())
            u = evaluate_displacement(model, reg, sol, obs, eps, warn_near=False)
            if bt is BCType.FREE_TRACTION:
                ru = np.abs(rigid_fit(u, obs) - rigid_fit(u_int_ex, obs)).max() \
                    / np.abs(u_int_ex).max()
            else:
                ru = np.abs(u - u_int_ex).max() / np.abs(u_int_ex).max()
            out[tag] = (rel, ru)
        print(f"{N:5d} {r:6.2f} {'half':>11s} {out['Dh'][0]:10.2e} "
              f"{out['Dh'][1]:10.2e} | {out['Nc'][0]:10.2e} {out['Nc'][1]:10.2e}")
        print(f"{'':5s} {'':6s} {'calibrated':>11s} {out['Dc'][0]:10.2e} "
              f"{out['Dc'][1]:10.2e} |")
