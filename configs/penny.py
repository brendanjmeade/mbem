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

``law`` makes it a STEPPED run instead (``mbem.stepper``): the crack's faces
resist their slip rate (``viscous``: a shear zone a radius wide with unit
relaxation time) or obey friction that strengthens with it (``friction``,
``a sigma`` being ``a_sigma_over_drop`` times the drop), from the drop at
t = 0 to ``t_max``. ``maxwell`` > 0 makes the body a Maxwell one with that
relaxation time. With ``order=0`` and ``backend=hmat`` nothing dense is
formed: the crack's own traction kernel is compressed (``la.faultkernel``),
which is what lets a/h go where n^2 does not fit:

    python -m mbem run configs/penny.py --set order=0 --set backend=hmat \
        --set law=friction --set maxwell=1 --sweep a_over_h=32,64,128
"""
from mbem import laplace, rheology
from mbem.config import (Backend, Geometry, Laplace, Model, Output, Run,
                         Stepper)


def run_spec(a_over_h: float = 8.0, order: int = 2, eps_over_h: float | None = None,
             kind: str = "cdt", delta0: float = 1e-3, backend: str = "dense",
             law: str = "", a_sigma_over_drop: float = 0.2,
             maxwell: float = 0.0, t_min: float = 0.1, t_max: float = 10.0,
             times_per_decade: int = 4, mu: float = 30.0,
             lam: float = 30.0) -> Run:
    a = 1.0
    eps = "auto" if eps_over_h is None else float(eps_over_h) * a / float(a_over_h)
    stepper = body = None
    if law:
        from mbem.cases.penny import stress_drop
        if law not in ("viscous", "friction"):
            raise ValueError(f"law {law!r}: 'viscous' or 'friction'")
        drop = stress_drop(mu, lam, delta0, a)
        zone = rheology.ViscousFault(tau=1.0, width=a)
        a_sigma = a_sigma_over_drop * drop
        faces = zone if law == "viscous" else rheology.RateStrengtheningFault(
            v0=a_sigma * zone.width / (mu * zone.tau), a_sigma=a_sigma, rake=0.0)
        n_t = max(1, round(times_per_decade * __import__("math").log10(t_max / t_min)))
        stepper = Stepper(faults={"penny": faces}, t=(0.0,) + tuple(
            t_min * (t_max / t_min) ** (i / n_t) for i in range(n_t + 1)))
        if maxwell > 0.0:
            body = Laplace(rheology={"space": rheology.Maxwell(tau=maxwell)},
                           s=laplace.s_grid(t_min, t_max))
    return Run(
        name=f"penny_ah{a_over_h:g}_P{order}" + ("" if eps_over_h is None
                                                 else f"_eps{eps_over_h:g}"),
        model=Model(geometry=Geometry("penny", params=dict(a=a, a_over_h=a_over_h, kind=kind)),
                    builder="penny",
                    params=dict(order=order, delta0=delta0,
                                **({"mu": mu, "lam": lam} if law else {})),
                    eps=eps),
        backend=Backend(kind=backend),
        laplace=body, stepper=stepper,
        outputs=Output(save_fields=True, save_meshes=True),
        notes="penny crack, traction-BC fault: slip b:penny vs the Eshelby profile",
        tags=("penny", "convergence"))
