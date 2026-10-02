"""Does the mesh-size jump at the top rim cost top-surface accuracy?

``docs/fbem-findings.md`` records an unclaimed result from the closed
force-element line: removing the size discontinuity between the fine top and the
coarse box sides costs little and buys a lot (+2.4 % unknowns, -22 % top-surface
L2, 2.2x better conditioning), is a property of the MESH rather than of the
formulation, and has never been tried with the direct BIE.

This is that test. The top mesh is held FIXED at 9 km and only ``edge_side``
varies, so any change in the top-surface field is caused by the sides alone --
which is what makes the comparison clean. The finest-sides run is the reference,
because there is no analytic solution here; what is being measured is how much
side-induced error the coarse default carries.

    python -m mbem run configs/side_grading.py --sweep edge_side=80,40,20,10
"""
from mbem.config import Backend, Geometry, Model, Output, Run, Solve


def run_spec(edge_side: float = 80.0, edge_far: float = 80.0,
             backend: str = "hmat", mu_inc: float = 3.0) -> Run:
    return Run(
        name=f"s{edge_side:g}_f{edge_far:g}",
        model=Model(
            geometry=Geometry("topo_inclusion",
                              params=dict(surface="topo", edge_side=edge_side,
                                          edge_far=edge_far)),
            builder="topo_inclusion", params=dict(mu_inc=mu_inc),
            eps="auto"),
        backend=Backend(kind=backend),
        solve=Solve(rtol=1e-10),      # tighter than the effect being measured
        outputs=Output(slots=("u:host_top", "u:inclusion_top"),
                       save_meshes=True),
        notes="top mesh fixed at 9 km; only the side/far edge varies",
        tags=("experiment", "mesh"))
