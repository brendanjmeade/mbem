"""Surface field convergence as the mollification width falls.

A sweep over EPS, because eps is a per-source-element parameter of the kernel:
changing it changes the operator, and a different operator is a different run.
The reference is the smallest rung, so what the figure shows is
self-convergence against the eps^2 guide.

The geometry refines the fault (edge 6 km) so that eps stays above ~h/3 across
the whole ladder -- below that the mesh no longer resolves the mollification and
the ladder flattens for a reason that has nothing to do with the kernel.

    python -m mbem run configs/eps_convergence.py --sweep eps=12,8,6,4,3,2
"""
from mbem.config import Backend, Geometry, Model, Output, Run

GEOM = dict(edge_fault=6.0)


def run_spec(eps: float = 4.0, backend: str = "dense",
             slip_mag: float = 0.01) -> Run:
    return Run(
        name=f"eps_{eps:g}",
        model=Model(geometry=Geometry("fault_box", params=GEOM),
                    builder="fault_box",
                    params=dict(slip_mag=slip_mag),
                    eps=eps),
        backend=Backend(kind=backend),
        outputs=Output(slots=("u:top",), save_meshes=True,
                       figures=("eps_convergence",)),
        notes="self-convergence of the surface field in eps",
        tags=("figure", "convergence"))
