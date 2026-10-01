"""Classical full-space triangular-dislocation (TDE) stress reference.

Independent check for the mollified on-fault elastic stress.  ``cutde``'s
artifact-free full-space TDE gives the classical (singular-kernel) elastic
stress of the SAME triangulated fault, so:

  * OFF the fault it equals the eps -> 0 limit of the mollified DD stress
    (the mollification error is O(eps^2));
  * ON the fault the classical kernel jumps/diverges, but its two-sided
    *finite part* (the average of the +-perp limits) is the constant the
    mollified-minus-eigenstress on-fault stress converges to.

The codebase carries Cartesian slip; cutde wants slip in the per-triangle
strike/dip/tensile (TDCS) frame, so we rotate with
``compute_efcs_to_tdcs_rotations``.  Reconciled to the mollified DD kernel
(no sign flip) at machine-vs-eps accuracy: ``slip_cart`` is the Burgers
vector b = u(+n) - u(-n), the same quantity a FAULT ``Patch.value`` holds,
so a patch's value passes through unchanged.
"""
from __future__ import annotations

import numpy as np


def _fault_tris(fault) -> np.ndarray:
    if hasattr(fault, "vertices"):
        tris = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    else:
        tris = np.asarray(fault, float).reshape(-1, 3, 3)
    return np.ascontiguousarray(tris)


def classical_tde_stress(obs, fault, slip_cart, mu, nu,
                         halfspace: bool = False) -> np.ndarray:
    """Classical TDE stress (N,3,3) at ``obs``.

    ``fault`` is a TriMesh-like mesh or an (Nt,3,3) vertex array; ``slip_cart``
    is a Cartesian slip, either (3,) broadcast to every triangle or (Nt,3).
    ``halfspace=True`` uses cutde's half-space solution (free surface at z=0,
    sources/obs at z<=0) -- the right reference for a free-surface BEM box;
    the default full-space solution matches an unbounded mollified fault.
    """
    import cutde.geometry as cg
    if halfspace:
        import cutde.halfspace as ts
    else:
        import cutde.fullspace as ts

    obs = np.ascontiguousarray(np.asarray(obs, float))
    tris = _fault_tris(fault)
    nt = tris.shape[0]
    slip_cart = np.asarray(slip_cart, float)
    if slip_cart.ndim == 1:
        slip_cart = np.broadcast_to(slip_cart, (nt, 3))

    R = cg.compute_efcs_to_tdcs_rotations(tris)               # EFCS->TDCS (nt,3,3)
    slip_tdcs = np.ascontiguousarray(np.einsum("sij,sj->si", R, slip_cart))
    M = ts.strain_matrix(obs, tris, nu)                      # (No,6,Ns,3)
    strain = np.einsum("ovsc,sc->ov", M, slip_tdcs)          # (No,6) voigt
    sv = cg.strain_to_stress(strain, mu, nu)                  # (No,6) voigt

    sig = np.zeros((obs.shape[0], 3, 3))
    sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2] = sv[:, 0], sv[:, 1], sv[:, 2]
    sig[:, 0, 1] = sig[:, 1, 0] = sv[:, 3]
    sig[:, 0, 2] = sig[:, 2, 0] = sv[:, 4]
    sig[:, 1, 2] = sig[:, 2, 1] = sv[:, 5]
    return sig
