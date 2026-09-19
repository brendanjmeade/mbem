"""Interior-field evaluation from a solved RegionModel.

Representation formula for x strictly inside region R (c(x) = 1):

    u(x) = sum_p sigma(R,p) * G_xp @ t_p
         - sum_p sigma(R,p) * H_xp @ u_p
         - sum_f sigma(R,f) * H_xf @ slip_f

with all kernels at region R's material; sigma and prescribed-value
handling identical to the boundary equations -- faults included, since
``RegionModel.orientation`` answers for them too (``FAULT_ORIENTATION``,
the single statement of the fault sign convention; see
``mbem/model/core.py``). Evaluation uses the
numba direct assemblers (fast dense N_obs x N_src blocks); a compressed
evaluation operator for very large observation sets is future work.

``evaluate_stress`` applies the stress operator C:grad to the SAME
representation, term by term:

    sigma(x) = sum_p sigma(R,p) * SG_xp @ t_p     (single layer, force->stress)
             - sum_p sigma(R,p) * SH_xp @ u_p     (double layer, slip->stress)
             - sum_f sigma(R,f) * SH_xf @ slip_f  (fault slip is a double layer)

where SG is the integrated Kelvin force-stress kernel
(``analytical_kelvin_stress``) and SH is the integrated displacement-
discontinuity stress kernel (``analytical_stress_kernel``) -- the SAME
kernel that maps a fault slip to stress. Inside the ~eps fault zone the
slip term returns the TOTAL stress (elastic + the anelastic eigenstress
C:eps_star, which peaks at (3/4) mu s / eps and diverges as eps -> 0);
with ``subtract_anelastic`` the eigenstress of each region fault is
removed, leaving the genuine ELASTIC stress -- the on-fault,
Coulomb-relevant field. The subtraction is a no-op away from a fault, so
it is safe to apply for off-fault observation points too.

The eigenstress used here is the EXACT finite-triangle form
(``tri_kernels.eigenstress_contract``: (15 eps^4/8pi) I7 per element,
summed over every fault element, each with its own eps), NOT the
infinite-plane / nearest-triangle approximation of the frozen
``anelastic.py``. The two agree deep inside a large element (the
d/L -> 0 limit) and differ by up to ~2x near element edges -- i.e. over
the whole fault RIM, where crack-tip stress and stress-drop diagnostics
live. Gate: ``verify/verify_eigenstress_exact.py``.
"""

from __future__ import annotations

import numpy as np

from .kernels import basis as kb
from .model import BCType, Region, RegionModel
from .selfcheck import ensure_fault_convention


def _warn_near_boundary(points, region, model=None):
    """Warn when observation points sit closer than ~0.5 local h to a
    BOUNDARY patch: the volume representation is mesh-limited there (the
    piecewise-constant boundary density cannot resolve the c=1/2 -> 1
    boundary-jump transition closer than ~h; shrinking eps does NOT help
    and inflates the traction components -- session-verified). Faults
    are exempt: on-fault evaluation is legitimate (the eigenstress
    subtraction handles the fault's own near field).
    """
    from scipy.spatial import cKDTree

    n_close = 0
    worst = np.inf
    for p in region.patches:
        c = np.asarray(p.mesh.centroids(), float)
        h = kb.element_sizes(p.mesh)
        d, idx = cKDTree(c).query(points, k=1)
        ratio = d / h[idx]
        n_close += int(np.sum(ratio < 0.5))
        if ratio.size:
            worst = min(worst, float(ratio.min()))
    if n_close:
        import warnings
        warnings.warn(
            f"{n_close} observation point(s) lie within 0.5*local-h of a "
            f"boundary patch (min d/h = {worst:.2f}): the volume "
            f"representation is MESH-limited there (expect the c=1/2 "
            f"boundary-jump error; refine the patch or evaluate deeper -- "
            f"smaller eps does not help)")


def _disp_from_source(points, src_mesh, density, kernel, material, eps_arr):
    """Displacement (N,3) at ``points`` from a triangulated source
    carrying a per-triangle ``density`` (N_src,3), via the matrix-free
    numba contraction drivers (``t_disp_contract`` / ``u_disp_contract``)
    -- O(N_obs) memory at any source count; the (3N_obs, 3N_src)
    influence matrix is never materialized.
    """
    from .kernels import tri_kernels as tk

    tri_verts, normals = kb._source_arrays(src_mesh)
    density = np.ascontiguousarray(np.asarray(density, float).reshape(-1, 3))
    points = np.ascontiguousarray(np.asarray(points, float))
    if kernel == "t":
        c = kb.t_coeffs(material.mu, material.lam)
        return tk.t_disp_contract(points, tri_verts, normals, eps_arr,
                                  density, *c)
    g = kb.u_coeffs(material.mu, material.lam)
    return tk.u_disp_contract(points, tri_verts, eps_arr, density, *g)


def evaluate_displacement(model: RegionModel, region: Region | str,
                          solution: dict, points: np.ndarray,
                          eps, warn_near: bool = True) -> np.ndarray:
    """Displacement at interior ``points`` (N,3) of ``region``.

    ``solution`` is the slot dict returned by a backend solve.
    """
    if isinstance(region, str):
        region = next(r for r in model.regions if r.name == region)
    if region.faults:
        ensure_fault_convention()
    points = np.asarray(points, dtype=float)
    if warn_near:
        _warn_near_boundary(points, region)
    u = np.zeros((points.shape[0], 3))

    def eps_for(patch):
        e = eps[patch.name] if isinstance(eps, dict) else eps
        return kb.resolve_eps(e, patch.mesh)

    mat = region.material
    for p in region.patches:
        sigma = float(model.orientation(region, p))
        # u_p term
        if p.bc is BCType.PRESCRIBED_DISPLACEMENT:
            u_p = p.value_array()
        else:
            u_p = solution[f"u:{p.name}"]
        if np.any(u_p):
            u -= sigma * _disp_from_source(points, p.mesh, u_p, "t",
                                           mat, eps_for(p))
        # t_p term
        if p.bc is BCType.FREE_TRACTION:
            t_p = p.value_array()
        else:
            t_p = solution[f"t:{p.name}"]
        if np.any(t_p):
            u += sigma * _disp_from_source(points, p.mesh, t_p, "u",
                                           mat, eps_for(p))

    # Faults carry an orientation too (FAULT_ORIENTATION), so this is
    # literally the u_p branch above with sigma supplied by the same
    # accessor -- no second statement of the fault sign convention.
    for f in region.faults:
        sigma = float(model.orientation(region, f))
        slip = f.value_array()
        if np.any(slip):
            u -= sigma * _disp_from_source(points, f.mesh, slip, "t",
                                           mat, eps_for(f))

    return u


def _stress_from_source(points, src_mesh, density, kernel, mu, nu, eps_arr):
    """Stress (N,3,3) at ``points`` from a triangulated source ``src_mesh``
    carrying a per-triangle ``density`` (N_src,3).

    ``kernel="dd"`` uses the displacement-discontinuity stress kernel
    (slip / boundary displacement -> stress); ``kernel="force"`` uses the
    Kelvin force-stress kernel (boundary traction -> stress); ``kernel="eigen"``
    returns the EXACT finite-triangle anelastic eigenstress +C:eps_star of
    the smeared slip (the divergent on-fault part of the "dd" term), summed
    over ALL source elements. All three route through the numba parallel
    stress assemblers (``dd_stress_contract`` / ``kelvin_stress_contract`` /
    ``eigenstress_contract``), which reproduce the scalar
    ``analytical_stress_kernel`` / ``analytical_kelvin_stress`` /
    ``analytical_eigenstress_kernel`` oracles to machine precision (see
    ``verify/verify_stress_assembler.py``,
    ``verify/verify_eigenstress_exact.py``) while parallelising over
    observation points. All three take a PER-ELEMENT ``eps_arr``.
    """
    from .kernels.tri_kernels import (dd_stress_contract,
                                      eigenstress_contract,
                                      kelvin_stress_contract)

    verts = np.ascontiguousarray(
        np.asarray(src_mesh.vertices, float)[np.asarray(src_mesh.triangles)])
    density = np.ascontiguousarray(np.asarray(density, float).reshape(-1, 3))
    points = np.ascontiguousarray(np.asarray(points, float))
    eps_arr = np.ascontiguousarray(np.asarray(eps_arr, float))

    if kernel in ("dd", "eigen"):
        normals, _ = src_mesh.normals_and_areas()
        normals = np.ascontiguousarray(np.asarray(normals, float))
        if kernel == "eigen":
            return eigenstress_contract(points, verts, normals, eps_arr,
                                        density, mu, nu)
        return dd_stress_contract(points, verts, normals, eps_arr,
                                  density, mu, nu)
    return kelvin_stress_contract(points, verts, eps_arr, density, mu, nu)


class PointCloud:
    """Adapter presenting raw observation points as a 'mesh' whose
    centroids are the points -- the field side of PairCompressed only
    ever asks for ``centroids()`` and ``n_triangles``."""

    def __init__(self, points):
        self.points = np.ascontiguousarray(np.asarray(points, dtype=float))

    @property
    def n_triangles(self) -> int:
        return self.points.shape[0]

    def centroids(self) -> np.ndarray:
        return self.points


class DisplacementEvaluator:
    """Block-compressed evaluation operator for a FIXED observation grid.

    Compresses the (obs x source) influence of every patch/fault of one
    region ONCE (ACA low-rank + dense leaves, memory O(k(N_obs+N_src))
    instead of the dense (3N_obs, 3N_src)), then evaluates any number of
    solutions / materials at matvec cost. Worth building only for
    REPEATED evaluation on the same grid (Laplace sweeps, time series);
    one-shot maps are cheaper through ``evaluate_displacement`` (the
    matrix-free contraction kernels).
    """

    def __init__(self, model: RegionModel, region: Region | str,
                 points: np.ndarray, eps,
                 tol: float = None, n_workers: int | None = None):
        from . import defaults
        if isinstance(region, str):
            region = next(r for r in model.regions if r.name == region)
        if region.faults:
            ensure_fault_convention("compressed")
        self.model = model
        self.region = region
        self.cloud = PointCloud(points)
        tol = defaults.BLOCK_COMPRESSION_TOL if tol is None else tol

        from .la.hop import PairCompressed
        arrays = kb.MeshArrays()
        tree_cache: dict = {}

        def eps_for(patch):
            e = eps[patch.name] if isinstance(eps, dict) else eps
            return kb.resolve_eps(e, patch.mesh)

        self._terms = []       # (sign, patch-or-fault, kernel, pair)
        for p in region.patches:
            sigma = float(model.orientation(region, p))
            self._terms.append(
                ("u", p, -sigma, PairCompressed(
                    self.cloud, p.mesh, "H", eps_for(p), tol=tol,
                    tree_cache=tree_cache, arrays=arrays,
                    n_workers=n_workers)))
            self._terms.append(
                ("t", p, +sigma, PairCompressed(
                    self.cloud, p.mesh, "U", eps_for(p), tol=tol,
                    tree_cache=tree_cache, arrays=arrays,
                    n_workers=n_workers)))
        for f in region.faults:
            self._terms.append(
                ("slip", f, -float(model.orientation(region, f)),
                 PairCompressed(
                    self.cloud, f.mesh, "H", eps_for(f), tol=tol,
                    tree_cache=tree_cache, arrays=arrays,
                    n_workers=n_workers)))

    def __call__(self, solution: dict) -> np.ndarray:
        mat = self.region.material
        tc = np.asarray(kb.t_coeffs(mat.mu, mat.lam))
        uc = np.asarray(kb.u_coeffs(mat.mu, mat.lam))
        u = np.zeros(3 * self.cloud.n_triangles)
        for kind, src, sign, pair in self._terms:
            if kind == "u":
                dens = (src.value_array()
                        if src.bc is BCType.PRESCRIBED_DISPLACEMENT
                        else solution[f"u:{src.name}"])
                coeffs = tc
            elif kind == "t":
                dens = (src.value_array()
                        if src.bc is BCType.FREE_TRACTION
                        else solution[f"t:{src.name}"])
                coeffs = uc
            else:
                dens = src.value_array()
                coeffs = tc
            dens = np.asarray(dens, float).ravel()
            if np.any(dens):
                u += sign * pair.matvec(coeffs, dens)
        return u.reshape(-1, 3)

    def nbytes(self) -> int:
        return sum(pair.nbytes() for _, _, _, pair in self._terms)


def evaluate_stress(model: RegionModel, region: Region | str,
                    solution: dict, points: np.ndarray, eps,
                    subtract_anelastic: bool = True,
                    warn_near: bool = True) -> np.ndarray:
    """Stress tensor (N,3,3) at ``points`` of ``region``.

    Mirrors :func:`evaluate_displacement` with the stress operator applied
    to every term. With ``subtract_anelastic`` (default) the EXACT
    finite-triangle eigenstress of each region fault is removed, so the
    returned field is the ELASTIC stress -- finite and eps-independent on
    the fault. Set it False to get the raw TOTAL stress (which diverges
    like 1/eps on the fault). A graded / per-element fault eps is fine:
    the eigenstress is summed element by element with each element's own
    eps (this restriction existed only for the frozen scalar-eps
    ``anelastic.py`` approximation).

    ``solution`` is the slot dict returned by a backend solve.
    """
    if isinstance(region, str):
        region = next(r for r in model.regions if r.name == region)
    if region.faults:
        ensure_fault_convention("stress")
    points = np.asarray(points, dtype=float)
    if warn_near:
        _warn_near_boundary(points, region)
    sig = np.zeros((points.shape[0], 3, 3))

    def eps_for(patch):
        e = eps[patch.name] if isinstance(eps, dict) else eps
        return kb.resolve_eps(e, patch.mesh)

    mat = region.material
    mu, nu = mat.mu, mat.nu
    for p in region.patches:
        sigma = float(model.orientation(region, p))
        # u_p term (double layer): - sigma * SH @ u_p
        if p.bc is BCType.PRESCRIBED_DISPLACEMENT:
            u_p = p.value_array()
        else:
            u_p = solution[f"u:{p.name}"]
        if np.any(u_p):
            sig -= sigma * _stress_from_source(points, p.mesh, u_p, "dd",
                                               mu, nu, eps_for(p))
        # t_p term (single layer): + sigma * SG @ t_p
        if p.bc is BCType.FREE_TRACTION:
            t_p = p.value_array()
        else:
            t_p = solution[f"t:{p.name}"]
        if np.any(t_p):
            sig += sigma * _stress_from_source(points, p.mesh, t_p, "force",
                                               mu, nu, eps_for(p))

    # As in evaluate_displacement: the fault term is the u_p branch with
    # sigma = FAULT_ORIENTATION, read through the same accessor.
    for f in region.faults:
        sigma = float(model.orientation(region, f))
        slip = f.value_array()
        if np.any(slip):
            sig -= sigma * _stress_from_source(points, f.mesh, slip, "dd",
                                               mu, nu, eps_for(f))
            if subtract_anelastic:
                # The fault stress term above is -sigma*Sdd@slip (mirroring
                # the -sigma*H@slip displacement term), so its divergent
                # on-fault part is -sigma * C:eps_star; removing it adds
                # +sigma * C:eps_star -- the SAME sigma, so the eigenstress
                # never states the convention a second time either.
                # (kernel="eigen" returns +C:eps_star, the divergent part of
                # +Sdd@slip.) Off a fault this is a no-op.
                # Sign verified by finiteness as eps->0
                # (verify/verify_eigenstress_exact.py, check [e]).
                #
                # EXACT finite-triangle eigenstress: the Cortez blob is
                # integrated over each actual triangle ((15 eps^4/8pi) I7)
                # and summed over ALL fault elements, with each element's
                # OWN eps. This replaces the frozen anelastic.py
                # approximation (nearest-triangle assignment + the
                # infinite-plane marginal), which is the d/L -> 0 limit --
                # right deep inside a large element, ~2x too large over the
                # whole fault rim. A per-element sum has no near-uniform-eps
                # restriction, so the former graded-eps raise is gone.
                sig += sigma * _stress_from_source(points, f.mesh, slip,
                                                   "eigen", mu, nu,
                                                   eps_for(f))

    return sig
