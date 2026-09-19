"""
Verify the batch (vectorized) kernel functions match the scalar versions
after the Cortez-blob fix.
"""

import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "mollified_kernel"))

from analytical_kernels import (
    analytical_dd_displacement,
    analytical_kelvin_G,
)
from analytical_batch import (
    dd_displacement_batch,
    kelvin_G_batch,
)


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
    worst = 0.0
    for nu in (0.25, 0.30):          # nu = 0.3 makes a lam/mu swap visible (lam != mu)
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

        print(f"nu={nu}: kelvin_G:        batch vs scalar  max|diff|={g_diff:.3e}  rel={g_diff/g_ref:.3e}")
        print(f"nu={nu}: dd_displacement: batch vs scalar  max|diff|={u_diff:.3e}  rel={u_diff/u_ref:.3e}")
        worst = max(worst, g_diff / g_ref, u_diff / u_ref)

    ok = worst < 1e-12
    if ok:
        print("\nPASS: batch and scalar agree to machine precision.")
    else:
        print("\nFAIL: batch and scalar disagree.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
