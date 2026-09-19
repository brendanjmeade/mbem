"""
Eigenstress audit.

(A) Are the five anelastic.py copies numerically identical?
(B) How far is anelastic.py's nearest-triangle / infinite-plane marginal
    from the EXACT finite-triangle eigenstress (sum over ALL elements of
    analytical_eigenstress_kernel) at production eps/h = 1.25?
    This is the correction msd's evaluate_stress applies to get ON-FAULT
    ELASTIC STRESS -- the quantity the user says matters most.
"""
import importlib.util
import sys
import numpy as np

ROOT = "/Users/meade/Desktop/moss-org"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ANEL = {
    "msd/anelastic.py (REFERENCE)": "msd/anelastic.py",
    "moss/anelastic.py": "moss/anelastic.py",
    "moss/mh_deploy/anelastic.py": "moss/mh_deploy/anelastic.py",
    "moss/mhf/anelastic.py": "moss/mhf/anelastic.py",
    "medt_paper/mhf/anelastic.py": "medt_paper/mhf/anelastic.py",
}

MU, NU = 30.0, 0.30          # nu != 1/4 on purpose
SLIP = np.array([1.0, 0.0, 0.0]) * 0.001   # 1 m in km


def rect_fault(nx, ny, L=10.0, W=6.0):
    """Planar fault in the x-z plane (normal = +y), nx x ny cells, 2 tris each."""
    xs = np.linspace(-L / 2, L / 2, nx + 1)
    zs = np.linspace(-W, 0.0, ny + 1)
    verts, tris = [], []
    idx = {}
    for i, x in enumerate(xs):
        for j, z in enumerate(zs):
            idx[(i, j)] = len(verts)
            verts.append([x, 0.0, z])
    for i in range(nx):
        for j in range(ny):
            a, b = idx[(i, j)], idx[(i + 1, j)]
            c, d = idx[(i + 1, j + 1)], idx[(i, j + 1)]
            tris.append([a, b, c]); tris.append([a, c, d])
    return np.asarray(verts, float), np.asarray(tris, int)


def main():
    print("=" * 78)
    print("(A) anelastic.py copies -- numerical identity check")
    print("=" * 78)
    mods = {k: load(f"{ROOT}/{v}", "an_" + k.split("/")[0] + str(i))
            for i, (k, v) in enumerate(ANEL.items())}
    V, T = rect_fault(6, 4)
    verts = V[T]
    obs = np.array([[0.3, 0.0, -2.0], [0.3, 0.4, -2.0], [5.4, 0.1, -0.1],
                    [0.0, 2.0, -3.0]])
    eps = 1.25 * 1.0
    ref = None
    for name, m in mods.items():
        out = m.eigenstress_at_points(obs, verts, SLIP, MU, NU, eps)
        if ref is None:
            ref = out
            print(f"    {name:32s} (reference)")
            continue
        d = np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-300)
        print(f"    {name:32s} rel_max_diff = {d:.3e}")

    print()
    print("=" * 78)
    print("(B) anelastic.py (nearest-tri, infinite-plane marginal)")
    print("    vs EXACT finite-triangle eigenstress (moss analytical_eigenstress_kernel)")
    print("    on a planar rectangular fault, evaluated AT ELEMENT CENTROIDS")
    print("=" * 78)
    ak = load(f"{ROOT}/moss/mollified_kernel/analytical_kernels.py", "ak_m")
    an = mods["msd/anelastic.py (REFERENCE)"]

    for nx, ny in ((6, 4), (12, 8), (20, 14)):
        V, T = rect_fault(nx, ny)
        verts = V[T]
        Nt = verts.shape[0]
        cen = verts.mean(axis=1)
        e1 = verts[:, 1] - verts[:, 0]
        e2 = verts[:, 2] - verts[:, 0]
        cr = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        nrm = cr / np.linalg.norm(cr, axis=1, keepdims=True)
        h = np.sqrt(2.0 * area.mean())          # nominal element size
        eps = 1.25 * h

        approx = an.eigenstress_at_points(cen, verts, SLIP, MU, NU, eps)

        exact = np.zeros((Nt, 3, 3))
        for s in range(Nt):
            H = np.array([ak.analytical_eigenstress_kernel(
                cen[o], verts[s, 0], verts[s, 1], verts[s, 2],
                nrm[s], MU, NU, eps) for o in range(Nt)])
            exact += np.einsum("omnk,k->omn", H, SLIP)

        # magnitude scale: the analytic on-fault shear the eigenstress carries
        scale = np.abs(exact).max()
        diff = np.abs(approx - exact)
        # distance of each centroid to the fault boundary (in units of eps)
        dx = np.minimum(cen[:, 0] + 5.0, 5.0 - cen[:, 0])
        dz = np.minimum(cen[:, 2] + 6.0, 0.0 - cen[:, 2])
        dedge = np.minimum(dx, dz)
        interior = dedge > 2.0 * eps
        print(f"\n  mesh {nx}x{ny} ({Nt} tri)  h={h:.3f}  eps={eps:.3f} (eps/h=1.25)")
        print(f"    peak |exact eigenstress|          = {scale:.4f} GPa")
        print(f"    max  |approx - exact| / peak      = {diff.max()/scale:.3e}   "
              f"(all {Nt} centroids)")
        if interior.any():
            print(f"    max  rel diff, centroids > 2 eps from fault edge "
                  f"({interior.sum()} of {Nt}) = {diff[interior].max()/scale:.3e}")
        else:
            print("    (no centroid is more than 2 eps from a fault edge)")
        print(f"    mean rel diff                      = "
              f"{diff.mean()/scale:.3e}")
        # where is it worst?
        k = np.unravel_index(np.argmax(diff.sum(axis=(1, 2))), (Nt,))[0]
        print(f"    worst centroid {cen[k]} : dist-to-edge = {dedge[k]:.3f} "
              f"= {dedge[k]/eps:.2f} eps")
        print(f"      approx sig_xy = {approx[k,0,1]:+.5f}  "
              f"exact sig_xy = {exact[k,0,1]:+.5f}  "
              f"ratio = {approx[k,0,1]/exact[k,0,1]:.4f}")
        # interior representative
        if interior.any():
            ki = np.where(interior)[0][interior.sum() // 2]
            print(f"    interior centroid: approx sig_xy = {approx[ki,0,1]:+.5f}  "
                  f"exact = {exact[ki,0,1]:+.5f}  "
                  f"ratio = {approx[ki,0,1]/exact[ki,0,1]:.4f}")


if __name__ == "__main__":
    main()
