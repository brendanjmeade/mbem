"""Topography + soft inclusion: the four-state decomposition, as two runs.

``surface`` is a GEOMETRY parameter -- it changes the mesh and so the operator
-- so topo and flat are two runs:

    python -m mbem run configs/topo_inclusion.py --sweep surface=topo,flat

which is one study of two child runs, because the showcase figure differences
them. A single ``--set surface=topo`` run still solves and saves its fields; it
reports the showcase as deferred, since one surface cannot draw it.

het and hom are two STATES of one run, because they share the assembly through
``rebuild_for_materials``; that sharing is the whole reason states exist. The
decomposition is then u(topo,het) - u(topo,hom) for the inclusion effect and
u(topo,het) - u(flat,het) for the topography effect, taken across the npz of
the two runs.

The topography effect differences two fields that agree to 43-72x their
difference, so it inherits ~2e-4 relative error from fields accurate to ~2e-6.
That is a property of the quantity, not of the backend -- see BACKLOG.
"""
from mbem.config import (Backend, Geometry, Material, Model, Output, Run,
                         Solve, State)

HOST = Material(mu=30.0, lam=30.0)


def run_spec(surface: str = "topo", backend: str = "hmat",
             scale: float = 1.0, mu_inc: float = 3.0) -> Run:
    return Run(
        name=f"topo_inclusion_{surface}_{backend}",
        model=Model(
            geometry=Geometry("topo_inclusion", scale=scale,
                              params=dict(surface=surface,
                                          bump_center=(0.0, -50.0),
                                          bump_sigma=30.0, bump_height=2.0)),
            builder="topo_inclusion",
            params=dict(mu_inc=mu_inc),
            eps="auto"),
        backend=Backend(kind=backend),
        solve=Solve(),
        states=(State("het"), State("hom", {"inclusion": HOST})),
        # slots=() means EVERY slot, which `mbem sample` needs: an interior
        # point is evaluated from the density on all of them -- sides, base,
        # both interfaces, fault -- not just the two the showcase figure
        # draws. Output's docstring warns that all-slots is tens of GB, but
        # that is at 4M unknowns; here the whole solution is ~200 kB a state.
        outputs=Output(slots=(),
                       save_meshes=True,
                       figures=("topo_inclusion_showcase",
                                "topo_inclusion_contour")),
        notes="figure-10 showcase; het/hom share one assembly",
        tags=("figure", "showcase"))
