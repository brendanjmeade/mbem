"""Does the fast operator reproduce the reference, on the inclusion model?

A sweep over BACKEND, because two backends are two operators and a different
operator is a different run. Each gets its own resolved.json, so the comparison
has provenance on both sides -- the old demo assembled both in one process and
nothing afterwards could say which tolerance or eps either side used.

    python -m mbem run configs/backend_agreement.py --sweep backend=hmat,dense
    python -m mbem run configs/backend_agreement.py --sweep backend=fmm,hmat
"""
from mbem.config import Backend, Geometry, Model, Output, Run, Solve


def run_spec(backend: str = "hmat", scale: float = 1.0,
             mu_inc: float = 3.0) -> Run:
    return Run(
        name=f"agreement_{backend}",
        model=Model(
            geometry=Geometry("topo_inclusion", scale=scale,
                              params=dict(surface="topo")),
            builder="topo_inclusion", params=dict(mu_inc=mu_inc),
            eps="auto"),
        # jump="half" on BOTH sides. Apples-to-apples is the whole point:
        # the calibrated diagonal is built FROM the far field, so comparing a
        # calibrated fast operator against a calibrated dense one lets each
        # one's diagonal partly absorb its own far-field error. Mixing the two
        # is worse still -- it compares different PROBLEMS, and reads as a 3%
        # backend disagreement when nothing is wrong with either backend.
        backend=Backend(kind=backend, jump="half"),
        solve=Solve(rtol=1e-8),
        outputs=Output(slots=("u:host_top", "u:inclusion_top"),
                       save_meshes=True, figures=("backend_agreement",)),
        notes="fast operator against the reference, same model and eps",
        tags=("figure", "agreement"))
