"""On-fault elastic shear: against the classical TDE, and down the dip.

Two axes, because the figure asks two questions:

  eps        does the mollified on-fault shear approach the half-space TDE as
             eps falls?
  order_top  what does the FIRST element row below the free surface do? A P0
             staircase is the known failure there; a P1 top is the cure.

    python -m mbem run configs/onfault_stress.py \
        --sweep eps=4,3,2,1.5,1 --sweep order_top=0,1

The stress is the ELASTIC part: the eigenstress of every mollified double layer
is subtracted in the readout, never in the solve.
"""
from mbem.config import Backend, Geometry, Model, Output, Run

GEOM = dict(half_x=120.0, z_bottom=-80.0, fault_half_len=25.0,
            fault_depth=15.0, edge_fault=2.5, edge_near=20.0,
            edge_far=60.0, edge_side=60.0, near_field_radius=60.0)


def run_spec(eps: float = 1.0, order_top: int = 1, backend: str = "dense",
             slip_mag: float = 0.01) -> Run:
    return Run(
        name=f"onfault_eps{eps:g}_P{order_top}",
        model=Model(geometry=Geometry("fault_box", params=GEOM),
                    builder="fault_box",
                    params=dict(slip_mag=slip_mag, order_top=order_top),
                    eps=eps),
        backend=Backend(kind=backend),
        outputs=Output(save_meshes=True, figures=("onfault_stress",)),
        notes="on-fault elastic shear vs eps, and the first-row P0/P1 contrast",
        tags=("figure", "onfault"))
