"""Fault-only mollified BEM: surface displacement and elastic surface stress.

The figure differentiates the free-surface displacement to get stress, then
removes the anelastic eigenstress of the fault; the thin strip hugging the
surface-breaking trace is masked, because slip is DISCONTINUOUS there and
grid-differentiating it renders the jump rather than stress.

    python -m mbem run configs/fault_only.py
"""
from mbem.config import Backend, Geometry, Model, Output, Run

# The demo's own geometry: a shallower, larger box than the bench rung, chosen so
# the free surface is resolved out to the 100 km plotting window.
GEOM = dict(half_x=200.0, z_bottom=-100.0, fault_half_len=50.0,
            fault_depth=25.0, near_field_radius=120.0,
            edge_fault=6.0, edge_near=15.0, edge_far=35.0, edge_side=35.0)


def run_spec(scale: float = 1.0, backend: str = "dense",
             slip_mag: float = 0.01) -> Run:
    return Run(
        name="fault_only",
        model=Model(geometry=Geometry("fault_box", scale=scale, params=GEOM),
                    builder="fault_box",
                    params=dict(slip_mag=slip_mag, mu=30.0, lam=30.0),
                    eps="auto"),
        backend=Backend(kind=backend),
        outputs=Output(save_meshes=True, figures=("fault_only",)),
        notes="surface displacement + elastic surface stress, fault only",
        tags=("figure", "fault_box"))
