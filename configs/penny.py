"""The penny-shaped crack solved as a traction-BC fault: the convergence ladder.

The crack of radius 1 km in the full space, Eshelby's shear drop for a peak
slip ``delta0``, so the exact answer of every rung is the Eshelby slip
delta0 sqrt(1 - r^2/a^2) and the exact moment 2 pi delta0 a^2 / 3. Two axes:

  a_over_h   elements per radius (8, 16, 24: ~490, ~1900, ~4300 elements)
  order      0 = P0 centroid collocation (the control), 2 = continuous P2
             (shared nodes, collocated at the nodes)

and ``eps_over_h`` (None = the fault's "auto" rule, 0.07 min h) sets the
mollification width in units of a / a_over_h. The gate ``verify_penny_crack``
pins a/h 8 and 16; this config is for the rungs and ladders too large for a
gate (a/h 24 is 26k unknowns, 5.5 GB dense).

    python -m mbem run configs/penny.py --sweep a_over_h=8,16,24 --sweep order=0,2
    python -m mbem run configs/penny.py --set a_over_h=16 --set eps_over_h=0.05
"""
from mbem.config import Backend, Geometry, Model, Output, Run


def run_spec(a_over_h: float = 8.0, order: int = 2, eps_over_h: float | None = None,
             kind: str = "cdt", delta0: float = 1e-3, backend: str = "dense") -> Run:
    a = 1.0
    eps = "auto" if eps_over_h is None else float(eps_over_h) * a / float(a_over_h)
    return Run(
        name=f"penny_ah{a_over_h:g}_P{order}" + ("" if eps_over_h is None
                                                 else f"_eps{eps_over_h:g}"),
        model=Model(geometry=Geometry("penny", params=dict(a=a, a_over_h=a_over_h, kind=kind)),
                    builder="penny",
                    params=dict(order=order, delta0=delta0),
                    eps=eps),
        backend=Backend(kind=backend),
        outputs=Output(save_fields=True, save_meshes=True),
        notes="penny crack, traction-BC fault: slip b:penny vs the Eshelby profile",
        tags=("penny", "convergence"))
