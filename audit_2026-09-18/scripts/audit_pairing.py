"""
Independent numerical audit of the lam/mu traction pairing in EVERY copy
of the mollified DD displacement kernel in moss-org.

Reference is written from scratch here (moment-tensor / Volterra form),
NOT taken from any copy:

    G^eps_ij = C1 [ (3-4nu) d_ij / Re + d_i d_j / Re^3 + 2(1-nu) eps^2 d_ij / Re^3 ]
    C1 = 1/(16 pi mu (1-nu)),  Re = sqrt(|d|^2 + eps^2)

    M_pq = A [ lam d_pq (s.n) + mu (s_p n_q + s_q n_p) ]     (seismic moment tensor)
    u_i  = M_pq dG_ip/dy_q = -M_pq dG_ip/dx_q

so U[i,j] (u_i = U[i,j] s_j) = -[ mu n_m G1[i,j,m] + lam n_j G1[i,m,m] + mu n_k G1[i,k,j] ]
with G1[i,p,q] = int_T dG_ip/dx_q dA.

The SWAPPED (buggy) form exchanges lam and mu on the first two terms.
Both agree exactly at nu = 1/4.
"""
import importlib.util
import sys
import numpy as np

ROOT = "/Users/meade/Desktop/moss-org"


# ---------------------------------------------------------------- reference
def dG_mollified(d, mu, nu, eps):
    """dG_ij/dx_m of the Cortez-mollified Kelvin tensor, coded here from scratch."""
    r2 = float(d @ d)
    Re2 = r2 + eps * eps
    Re = np.sqrt(Re2)
    i3 = 1.0 / (Re * Re2)
    i5 = i3 / Re2
    C1 = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    c34 = 3.0 - 4.0 * nu
    I = np.eye(3)
    out = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for m in range(3):
                out[i, j, m] = C1 * (
                    -c34 * I[i, j] * d[m] * i3
                    + I[i, m] * d[j] * i3
                    + I[j, m] * d[i] * i3
                    - 3.0 * d[i] * d[j] * d[m] * i5
                    - 6.0 * (1.0 - nu) * eps * eps * I[i, j] * d[m] * i5
                )
    return out


def _tri_quad(n):
    """Symmetric-ish tensor-product Duffy quadrature on the unit triangle."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    pts, wts = [], []
    for a, wa in zip(x, w):
        for b, wb in zip(x, w):
            u = a
            v = b * (1.0 - a)
            pts.append((u, v))
            wts.append(wa * wb * (1.0 - a))
    return np.array(pts), np.array(wts)


def G1_numeric(obs, v1, v2, v3, mu, nu, eps, n=40):
    """int_T dG/dx dA by quadrature (independent of every repo copy)."""
    pts, wts = _tri_quad(n)
    area2 = np.cross(v2 - v1, v3 - v1)
    A = 0.5 * np.linalg.norm(area2)
    acc = np.zeros((3, 3, 3))
    for (u, v), w in zip(pts, wts):
        y = v1 + u * (v2 - v1) + v * (v3 - v1)
        acc += w * dG_mollified(obs - y, mu, nu, eps)
    return acc * (2.0 * A)


def U_reference(obs, v1, v2, v3, normal, mu, nu, eps, n=40, swapped=False):
    G1 = G1_numeric(obs, v1, v2, v3, mu, nu, eps, n)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    a, b = (lam, mu) if swapped else (mu, lam)
    nvec = np.asarray(normal, float)
    U = np.zeros((3, 3))
    for i in range(3):
        tr = G1[i, 0, 0] + G1[i, 1, 1] + G1[i, 2, 2]
        for j in range(3):
            U[i, j] = -(
                a * sum(nvec[m] * G1[i, j, m] for m in range(3))
                + b * nvec[j] * tr
                + mu * sum(nvec[k] * G1[i, k, j] for k in range(3))
            )
    return U


# ---------------------------------------------------------------- loader
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


COPIES = {
    "msd/mollified_kernel":            "msd/mollified_kernel/analytical_kernels.py",
    "moss/mollified_kernel":           "moss/mollified_kernel/analytical_kernels.py",
    "moss/mh_deploy/mollified_kernel": "moss/mh_deploy/mollified_kernel/analytical_kernels.py",
    "medt_paper/mollified_kernel":     "medt_paper/mollified_kernel/analytical_kernels.py",
    "moss/mhf":                        "moss/mhf/analytical_kernels.py",
    "medt_paper/mhf":                  "medt_paper/mhf/analytical_kernels.py",
}

# geometry: a tilted, irregular triangle (no accidental symmetry)
v1 = np.array([0.0, 0.0, 0.0])
v2 = np.array([1.3, 0.2, 0.1])
v3 = np.array([0.4, 1.1, -0.3])
nrm = np.cross(v2 - v1, v3 - v1)
nrm = nrm / np.linalg.norm(nrm)
OBS = [np.array([0.6, 0.3, 1.7]),      # a couple of eps away, off-plane
       np.array([-1.2, 2.4, 0.9]),     # farther field
       np.array([0.5, 0.45, 0.05])]    # essentially ON the element
MU = 30.0
EPS = 0.35


def main():
    print("=" * 78)
    print("PART 1  lam/mu DD pairing: every scalar copy vs independent reference")
    print("=" * 78)
    mods = {k: load(f"{ROOT}/{v}", "ak_" + k.replace("/", "_")) for k, v in COPIES.items()}

    for nu in (0.25, 0.30, 0.45):
        print(f"\n--- nu = {nu}")
        # how different are correct and swapped, to show the test has teeth
        Uc = U_reference(OBS[0], v1, v2, v3, nrm, MU, nu, EPS)
        Us = U_reference(OBS[0], v1, v2, v3, nrm, MU, nu, EPS, swapped=True)
        sep = np.abs(Uc - Us).max() / np.abs(Uc).max()
        print(f"    [test sensitivity] correct vs swapped reference differ by "
              f"{sep:.3e} (relative)")
        for name, m in mods.items():
            worst = 0.0
            worst_sw = 1e99
            for obs in OBS:
                U = m.analytical_dd_displacement(obs, v1, v2, v3, nrm, MU, nu, EPS)
                R = U_reference(obs, v1, v2, v3, nrm, MU, nu, EPS)
                S = U_reference(obs, v1, v2, v3, nrm, MU, nu, EPS, swapped=True)
                scale = np.abs(R).max()
                worst = max(worst, np.abs(U - R).max() / scale)
                worst_sw = min(worst_sw, np.abs(U - S).max() / scale)
            verdict = "CORRECT" if worst < 2e-3 else (
                "*** SWAPPED ***" if worst_sw < 2e-3 else "*** UNKNOWN FORM ***")
            print(f"    {name:34s} err_vs_correct={worst:.2e}  "
                  f"err_vs_swapped={worst_sw:.2e}  {verdict}")

    print()
    print("=" * 78)
    print("PART 2  copy-vs-copy entrywise agreement (same inputs, nu = 0.30/0.45)")
    print("=" * 78)
    ref_name = "msd/mollified_kernel"
    for nu in (0.30, 0.45):
        print(f"\n--- nu = {nu}")
        base = {}
        for fn in ("analytical_dd_displacement", "analytical_stress_kernel"):
            vals = {}
            for name, m in mods.items():
                if not hasattr(m, fn):
                    vals[name] = None
                    continue
                out = [np.asarray(getattr(m, fn)(o, v1, v2, v3, nrm, MU, nu, EPS))
                       for o in OBS]
                vals[name] = np.concatenate([o.ravel() for o in out])
            r = vals[ref_name]
            for name, a in vals.items():
                if a is None:
                    print(f"    {fn:28s} {name:34s} MISSING")
                    continue
                d = np.abs(a - r).max() / max(np.abs(r).max(), 1e-300)
                print(f"    {fn:28s} {name:34s} rel_max_diff_vs_msd={d:.3e}")
        # kelvin G / kelvin stress (no normal argument)
        for fn in ("analytical_kelvin_G", "analytical_kelvin_stress"):
            vals = {}
            for name, m in mods.items():
                out = [np.asarray(getattr(m, fn)(o, v1, v2, v3, MU, nu, EPS))
                       for o in OBS]
                vals[name] = np.concatenate([o.ravel() for o in out])
            r = vals[ref_name]
            for name, a in vals.items():
                d = np.abs(a - r).max() / max(np.abs(r).max(), 1e-300)
                print(f"    {fn:28s} {name:34s} rel_max_diff_vs_msd={d:.3e}")

    print()
    print("=" * 78)
    print("PART 3  which copies have the eigenstress kernel at all")
    print("=" * 78)
    for name, m in mods.items():
        print(f"    {name:34s} analytical_eigenstress_kernel: "
              f"{'yes' if hasattr(m, 'analytical_eigenstress_kernel') else 'NO'}")


if __name__ == "__main__":
    main()
