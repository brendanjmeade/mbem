"""
Verify the batch (vectorized) kernel functions match the scalar versions
after the Cortez-blob fix, and that the batched eigenstress kernel matches
the scalar ``analytical_eigenstress_kernel`` (incl. observers on the
triangle plane: inside, on an edge, at a vertex, just off an edge).
"""

import numpy as np

from moss_kernel.analytical_kernels import (
    analytical_dd_displacement,
    analytical_eigenstress_kernel,
    analytical_kelvin_G,
)
from moss_kernel.analytical_batch import (
    dd_displacement_batch,
    eigenstress_batch,
    kelvin_G_batch,
)
import sys


def main():
    # Triangle and a panel of obs points
    v1 = np.array([0.1, -0.2, 0.0])
    v2 = np.array([1.3, 0.0, 0.0])
    v3 = np.array([0.6, 1.1, 0.05])
    normal = np.cross(v2 - v1, v3 - v1)
    normal = normal / np.linalg.norm(normal)

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(8, 3))

    mu = 1.0
    eps = 0.3

    # Observers on the (tilted) triangle plane, plus one just above it
    e1, e2 = v2 - v1, v3 - v1
    cen = (v1 + v2 + v3) / 3.0
    out12 = np.cross(e1, normal)
    out12 = out12 / np.linalg.norm(out12)
    if np.dot(out12, cen - v1) > 0:
        out12 = -out12                      # in-plane outward normal of edge v1-v2
    obs_plane = np.array([
        cen,                                # inside
        v1 + 0.3 * e1 + 0.3 * e2,           # inside
        0.5 * (v1 + v2),                    # on an edge
        v2,                                 # at a vertex
        v1 + 0.4 * e1 + 1e-9 * out12,       # just outside an edge
        v1 + 0.4 * e1 - 1e-9 * out12,       # just inside an edge
        v1 + 1.2 * e1 + 1e-9 * out12,       # on the edge's extension
        v1 + 0.5 * e1 + 0.5 * out12,        # outside
        cen + 1e-3 * normal,                # just above the plane
    ])
    obs_eig = np.vstack([obs, obs_plane])

    # nu = 0.3 as well as 1/4: at nu = 1/4 lam == mu, which hid a lam/mu
    # swap in the batch DD kernel until 2026-09-17 (see verify_dd_pairing.py).
    worst = 0.0
    worst_eig = 0.0
    for nu in (0.25, 0.3):
        # Batch
        G_b = kelvin_G_batch(v1, v2, v3, obs, mu, nu, eps)             # (8,3,3)
        U_b = dd_displacement_batch(v1, v2, v3, normal, obs, mu, nu, eps)

        # Scalar
        G_s = np.zeros_like(G_b)
        U_s = np.zeros_like(U_b)
        for k in range(obs.shape[0]):
            G_s[k] = analytical_kelvin_G(obs[k], v1, v2, v3, mu, nu, eps)
            U_s[k] = analytical_dd_displacement(obs[k], v1, v2, v3, normal, mu, nu, eps)

        g_diff = np.max(np.abs(G_b - G_s))
        u_diff = np.max(np.abs(U_b - U_s))
        g_ref = np.max(np.abs(G_s))
        u_ref = np.max(np.abs(U_s))

        print(f"nu={nu:.2f} kelvin_G:        batch vs scalar max|diff|={g_diff:.3e}  rel={g_diff/g_ref:.3e}")
        print(f"nu={nu:.2f} dd_displacement: batch vs scalar max|diff|={u_diff:.3e}  rel={u_diff/u_ref:.3e}")
        worst = max(worst, g_diff / g_ref, u_diff / u_ref)

        # Eigenstress: per-observer relative error (values span ~5 decades)
        H_b = eigenstress_batch(v1, v2, v3, normal, obs_eig, mu, nu, eps)
        H_s = np.array([analytical_eigenstress_kernel(o, v1, v2, v3, normal,
                                                      mu, nu, eps)
                        for o in obs_eig])
        h_rel = (np.max(np.abs(H_b - H_s), axis=(1, 2, 3))
                 / np.max(np.abs(H_s), axis=(1, 2, 3)))
        h_rel_plane = h_rel[len(obs):].max()
        print(f"nu={nu:.2f} eigenstress:     batch vs scalar worst per-obs rel={h_rel.max():.3e}"
              f"  (on-plane obs {h_rel_plane:.3e})")
        worst_eig = max(worst_eig, h_rel.max())
        if np.any(eigenstress_batch(v1, v2, v3, normal, obs_eig, mu, nu, 0.0)):
            print(f"nu={nu:.2f} eigenstress:     nonzero for eps = 0")
            worst_eig = np.inf

    # The eigenstress recursion loses relative (not absolute) accuracy in
    # its far tail, identically in both versions, hence the looser bound.
    ok = worst < 1e-12 and worst_eig < 1e-10
    if ok:
        print("\nPASS: batch and scalar agree to machine precision.")
    else:
        print("\nFAIL: batch and scalar disagree.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
