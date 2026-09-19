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
kernel that maps a fault slip to stress. Every SH term is a mollified
DOUBLE LAYER: within ~3 eps of its surface it returns the TOTAL stress,
elastic plus the anelastic eigenstress C:eps_star of the smeared jump
((3/4) mu |jump| / eps at the surface, divergent as eps -> 0). For a fault
the jump is the slip; for a boundary patch it is u_p itself (the field
inside R against zero outside), and its smeared eigenstress is non-physical
inside the body. ``subtract_anelastic`` (default) removes the eigenstress of
EVERY double layer, boundary patches and faults alike, leaving the ELASTIC
stress: the on-fault Coulomb field, and an interior field near boundaries
that converges under refinement (``verify/verify_boundary_eigenstress.py``).
It is a no-op farther than a few eps from every surface.

The eigenstress used here is the EXACT finite-triangle form
(``tri_kernels.eigenstress_contract``: (15 eps^4/8pi) I7 per element,
summed over every source element, each with its own eps), NOT the
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


def _warn_near_boundary(points, region):
    """Warn when observation points lie within NEAR_BOUNDARY_H_RATIO local
    h (mean edge) of a BOUNDARY patch (exact point-to-triangle distance):
    the piecewise-constant density limits the volume representation there
    (see ``defaults``). Faults are exempt: on-fault evaluation is legitimate
    (the eigenstress subtraction handles the fault's own near field).
    """
    from . import defaults
    from .geometry import distance_to_mesh

    ratio_max = defaults.NEAR_BOUNDARY_H_RATIO
    points = np.asarray(points, float).reshape(-1, 3)
    close = np.zeros(points.shape[0], bool)
    worst, names = np.inf, []
    for p in region.patches:
        d, idx = distance_to_mesh(points, p.mesh)
        ratio = d / kb.element_sizes(p.mesh)[idx]
        if ratio.size and ratio.min() < ratio_max:
            close |= ratio < ratio_max
            worst = min(worst, float(ratio.min()))
            names.append(p.name)
    if close.any():
        import warnings
        warnings.warn(
            f"{int(close.sum())} of {close.size} observation point(s) lie "
            f"within {ratio_max}*local-h of boundary patch(es) "
            f"{', '.join(names)} (min d/h = {worst:.2f}): the "
            f"piecewise-constant boundary density limits the volume "
            f"representation there (~2e-1 relative stress error at "
            f"d/h = 0.25 measured); refine the patch or evaluate deeper")


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


def _double_layer_stress(points, src_mesh, jump, sigma, mu, nu, eps_arr,
                         subtract_anelastic):
    """``-sigma * Sdd @ jump`` for one mollified double layer (boundary
    ``u_p`` or fault ``slip``); with ``subtract_anelastic`` its divergent
    on-surface part ``-sigma * C:eps_star`` is removed by adding
    ``+sigma * C:eps_star`` (``kernel="eigen"`` returns +C:eps_star). The
    one place this pairing is written; the sign is pinned by finiteness as
    eps -> 0 (``verify_eigenstress_exact.py`` [e]).
    """
    sig = -sigma * _stress_from_source(points, src_mesh, jump, "dd",
                                       mu, nu, eps_arr)
    if subtract_anelastic:
        sig += sigma * _stress_from_source(points, src_mesh, jump, "eigen",
                                           mu, nu, eps_arr)
    return sig


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
                 tol: float = None, n_workers: int | None = None,
                 warn_near: bool = True):
        from . import defaults
        if isinstance(region, str):
            region = next(r for r in model.regions if r.name == region)
        if region.faults:
            ensure_fault_convention("compressed")
        if warn_near:
            _warn_near_boundary(points, region)
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
    finite-triangle eigenstress of EVERY double layer -- each boundary
    patch's u_p and each fault's slip -- is removed, so the returned field
    is the ELASTIC stress: finite and eps-independent on a fault, and
    convergent under refinement near a boundary patch. Set it False to get
    the raw TOTAL stress (which diverges like 1/eps on the fault and
    carries a spurious ~mu |u_p| Phi_eps(d) within ~3 eps of every patch).
    A graded / per-element eps is fine: the eigenstress is summed element
    by element with each element's own eps (this restriction existed only
    for the frozen scalar-eps ``anelastic.py`` approximation).

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
        # u_p term (double layer): - sigma * SH @ u_p, minus its eigenstress;
        # the boundary u_p is a jump exactly as a fault slip is.
        if p.bc is BCType.PRESCRIBED_DISPLACEMENT:
            u_p = p.value_array()
        else:
            u_p = solution[f"u:{p.name}"]
        if np.any(u_p):
            sig += _double_layer_stress(points, p.mesh, u_p, sigma, mu, nu,
                                        eps_for(p), subtract_anelastic)
        # t_p term (single layer): + sigma * SG @ t_p
        if p.bc is BCType.FREE_TRACTION:
            t_p = p.value_array()
        else:
            t_p = solution[f"t:{p.name}"]
        if np.any(t_p):
            sig += sigma * _stress_from_source(points, p.mesh, t_p, "force",
                                               mu, nu, eps_for(p))

    # As in evaluate_displacement: the fault term is the u_p branch with
    # sigma = FAULT_ORIENTATION through the same accessor and the same
    # double-layer helper, so the convention is never stated twice.
    for f in region.faults:
        sigma = float(model.orientation(region, f))
        slip = f.value_array()
        if np.any(slip):
            sig += _double_layer_stress(points, f.mesh, slip, sigma, mu, nu,
                                        eps_for(f), subtract_anelastic)

    return sig
