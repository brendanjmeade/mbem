"""Adapters giving the reference cases one shape a config can drive.

The builders themselves are untouched and keep their own signatures: they are
what the gates call, and the point of routing a config through them is that the
config cannot grow a second statement of the boundary conditions or the fault
sign. These adapters only normalise the plumbing -- one returns a dict, one
returns a 7-tuple -- and apply ``scale`` to the edge lengths, which is where
``bench_scaling`` already did it.

    mesh builder:   f(*, scale=1.0, **params) -> MeshBundle
    model builder:  f(bundle, **params)       -> RegionModel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MeshBundle:
    """``(meshes, arrays, scalars)`` -- the triple bench_scaling's mesh cache
    already persists, plus the scalars a model builder needs (s_hat, x_range,
    z_bottom, the bump parameters)."""
    meshes: dict
    arrays: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)

    def flat(self) -> dict:
        """One dict, the shape the existing build_model functions expect."""
        return dict(self.meshes, **self.arrays, **self.scalars)


# ---------------------------------------------------------------------
# fault_box
# ---------------------------------------------------------------------

def fault_box_meshes(*, scale: float = 1.0, half_x: float = 100.0,
                     z_bottom: float = -60.0, fault_half_len: float = 30.0,
                     fault_depth: float = 18.0, near_field_radius: float = 50.0,
                     edge_fault: float = 4.0, edge_near: float = 25.0,
                     edge_far: float = 50.0,
                     edge_side: float = 50.0) -> MeshBundle:
    """The homogeneous box + buried strike-slip fault, at ``scale``.

    The defaults are bench_scaling's scale-1 rung, so a ladder written as a
    config reproduces the historical geometry. Only the EDGES are divided by
    scale -- the box does not shrink, it refines.
    """
    from mbem.cases.fault_box import build_fault_box
    d = build_fault_box(half_x=half_x, z_bottom=z_bottom,
                        fault_half_len=fault_half_len,
                        fault_depth=fault_depth,
                        near_field_radius=near_field_radius,
                        edge_fault=edge_fault / scale,
                        edge_near=edge_near / scale,
                        edge_far=edge_far / scale,
                        edge_side=edge_side / scale)
    meshes = {k: v for k, v in d.items() if hasattr(v, "triangles")}
    arrays = {k: v for k, v in d.items()
              if isinstance(v, np.ndarray) and k not in meshes}
    scalars = {k: v for k, v in d.items()
               if k not in meshes and k not in arrays}
    return MeshBundle(meshes, arrays, scalars)


def fault_box_model(bundle: MeshBundle, *, slip_mag: float = 0.01,
                    mu: float = 30.0, lam: float = 30.0, order_top: int = 0):
    from mbem import ElasticMaterial
    from mbem.cases.fault_box import build_model
    return build_model(bundle.flat(), slip_mag,
                       ElasticMaterial(mu=mu, lam=lam), order_top=order_top)


# ---------------------------------------------------------------------
# layered_box
# ---------------------------------------------------------------------

def layered_box_meshes(*, scale: float = 1.0, half_x: float = 400.0,
                       z_interface: float = -30.0, z_bottom: float = -300.0,
                       fault_half_len: float = 150.0, fault_depth: float = 15.0,
                       near_field_radius: float = 90.0,
                       edge_fault: float = 5.0, edge_near: float = 20.0,
                       edge_far: float = 80.0, edge_side: float = 80.0,
                       edge_interface: float = 15.0) -> MeshBundle:
    """An elastic plate over a substrate with a strike-slip fault in the
    plate, at ``scale``. As for the fault box, only the EDGES are divided by
    scale: the box does not shrink, it refines."""
    from mbem.cases.layered_box import build_layered_box
    d = build_layered_box(half_x=half_x, z_interface=z_interface,
                          z_bottom=z_bottom, fault_half_len=fault_half_len,
                          fault_depth=fault_depth,
                          near_field_radius=near_field_radius,
                          edge_fault=edge_fault / scale,
                          edge_near=edge_near / scale,
                          edge_far=edge_far / scale,
                          edge_side=edge_side / scale,
                          edge_interface=edge_interface / scale)
    meshes = {k: v for k, v in d.items() if hasattr(v, "triangles")}
    arrays = {k: v for k, v in d.items()
              if isinstance(v, np.ndarray) and k not in meshes}
    scalars = {k: v for k, v in d.items()
               if k not in meshes and k not in arrays}
    return MeshBundle(meshes, arrays, scalars)


def layered_box_model(bundle: MeshBundle, *, slip_mag: float = 0.001,
                      mu_plate: float = 30.0, lam_plate: float = 30.0,
                      mu_substrate: float = 30.0, lam_substrate: float = 30.0,
                      sides: str = "free", order_top: int = 0):
    from mbem import ElasticMaterial
    from mbem.cases.layered_box import build_model
    return build_model(bundle.flat(), slip_mag,
                       ElasticMaterial(mu=mu_plate, lam=lam_plate),
                       ElasticMaterial(mu=mu_substrate, lam=lam_substrate),
                       sides=sides, order_top=order_top)


# ---------------------------------------------------------------------
# topo_inclusion
# ---------------------------------------------------------------------

def topo_inclusion_meshes(*, scale: float = 1.0,
                          bump_center=(0.0, -50.0), bump_sigma: float = 30.0,
                          bump_height: float = 2.0,
                          surface: str = "topo",
                          edge_side: float = 80.0,
                          edge_far: float = 80.0) -> MeshBundle:
    """Host + soft inclusion + surface-breaking fault, with a Gaussian hill.

    ``surface`` selects the warped or flat embedding. It is a GEOMETRY
    parameter, not a state: it changes the mesh and therefore the operator, so
    topo and flat are two runs. (het/hom are two STATES of one run, because
    those share the assembly.)
    """
    if surface not in ("topo", "flat"):
        raise ValueError(f"surface {surface!r} must be 'topo' or 'flat'")
    from mbem.cases.topo_inclusion import build
    (meshes, top_flat, top_topo, fault_flat, fault_topo,
     s_hat, bump) = build(tuple(bump_center), bump_sigma, bump_height, scale,
                          edge_side=edge_side, edge_far=edge_far)
    meshes = dict(meshes)
    warped = surface == "topo"
    meshes["host_top"] = top_topo if warped else top_flat
    # The figure plots in the FLAT frame (a map view) and colours the relief, so
    # both travel with the bundle: a solved field is just numbers until you know
    # which triangles it sits on and how high they are.
    v = top_flat.vertices
    return MeshBundle(
        meshes,
        {"s_hat": s_hat,
         "host_top_flat_v": v, "host_top_flat_t": top_flat.triangles,
         "host_top_h": bump(v[:, 0], v[:, 1])},
        {"fault_mesh": fault_topo if warped else fault_flat,
         "surface": surface, "bump_center": tuple(bump_center),
         "bump_sigma": bump_sigma, "bump_height": bump_height,
         "bump_support": bump.support_radius})


def topo_inclusion_model(bundle: MeshBundle, *, mu_inc: float = 3.0,
                         lam_inc: float | None = None):
    from mbem import ElasticMaterial
    from mbem.cases.inclusion import build_model
    mat = ElasticMaterial(mu=mu_inc,
                          lam=mu_inc if lam_inc is None else lam_inc)
    return build_model(bundle.meshes, bundle.scalars["fault_mesh"],
                       bundle.arrays["s_hat"], mat_inc=mat)


# ---------------------------------------------------------------------
# penny
# ---------------------------------------------------------------------

def penny_meshes(*, scale: float = 1.0, a: float = 1.0, a_over_h: float = 8.0,
                 kind: str = "cdt", jitter: float = 0.25,
                 seed: int = 0) -> MeshBundle:
    """The penny-shaped crack of radius ``a`` at ``a_over_h`` elements per
    radius (``scale`` refines further), ``kind`` the mesher
    (``mbem.cases.penny.penny_mesh``)."""
    from mbem.cases.penny import penny_mesh
    h = a / (a_over_h * scale)
    mesh = penny_mesh(a, h, kind, jitter=jitter, seed=seed)
    return MeshBundle({"penny": mesh}, {}, {"a": a, "h": h, "kind": kind})


def penny_model(bundle: MeshBundle, *, mu: float = 30.0, lam: float = 30.0,
                delta0: float = 1e-3, order: int = 2,
                allow_opening: bool = False, prescribed: bool = False):
    """The crack as a FAULT_TRACTION fault whose drop is Eshelby's for the
    peak slip ``delta0`` (so the exact answer of the run is the Eshelby slip),
    or, ``prescribed``, with that slip imposed (the readout fixture)."""
    from mbem import ElasticMaterial
    from mbem.cases.penny import build_model, build_slip_model, stress_drop
    mat = ElasticMaterial(mu=mu, lam=lam)
    a = bundle.scalars["a"]
    mesh = bundle.meshes["penny"]
    if prescribed:
        return build_slip_model(mesh, mat, delta0=delta0, order=order, a=a)
    return build_model(mesh, mat, drop=stress_drop(mu, lam, delta0, a),
                       order=order, allow_opening=allow_opening)
