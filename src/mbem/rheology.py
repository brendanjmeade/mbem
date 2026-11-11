"""Linear viscoelastic rheology as operational moduli (docs/viscoelastic.md).

A region's ``(mu, lam)`` is its UNRELAXED elasticity: what it does at the
instant of loading, and all it does if it has no rheology. A rheology says how
the SHEAR modulus depends on the Laplace parameter ``s``; the bulk modulus
``K = lam + 2 mu / 3`` is elastic and does not move. That is the general case
-- Poisson's ratio changes in time -- and the convention of the layered-earth
codes this is compared with.

A rheology is stated in RELAXATION TIMES and modulus RATIOS, never in
viscosities or absolute moduli, so it is a property of the material's time
dependence alone and one object serves any region and any unit of time:
viscosity enters only as ``eta / mu``.

``shear_ratio(s)`` is ``mu(s) / mu``. It is written so that ``s = inf`` returns
exactly 1 (the elastic state) and so that it is plain arithmetic, valid for a
real, a complex or an arbitrary-precision ``s`` alike -- the gates invert
closed forms on a complex contour with the same function the solver uses on
the real axis. ``s`` must be positive: the relaxed limit ``s -> 0`` of a fluid
is ``mu = 0``, which is not an elastic solid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Maxwell:
    """A spring and a dashpot in series: ``mu(s) = mu s / (s + 1/tau)``.

    ``tau = eta / mu``, the Maxwell time. A fluid: under a sustained stress it
    creeps forever.
    """
    tau: float
    kind: str = field(default="maxwell", init=False)

    def __post_init__(self):
        if not (self.tau > 0.0 and math.isfinite(self.tau)):
            raise ValueError(f"Maxwell tau {self.tau} must be positive")

    def shear_ratio(self, s):
        return 1.0 / (1.0 + 1.0 / (self.tau * s))


@dataclass(frozen=True)
class Burgers:
    """A Maxwell element in series with a Kelvin (spring || dashpot) element:

        1/mu(s) = 1/mu + 1/(mu tau_M s) + 1/(mu_K (1 + tau_K s))

    ``tau_M = eta_M / mu`` is the steady (long-term) relaxation time,
    ``tau_K = eta_K / mu_K`` the transient one, and ``kelvin = mu_K / mu`` the
    Kelvin spring over the unrelaxed modulus. The transient phase relaxes the
    modulus from ``mu`` to ``mu kelvin / (1 + kelvin)``; the steady phase takes
    it to zero. Also a fluid.
    """
    tau_M: float
    tau_K: float
    kelvin: float
    kind: str = field(default="burgers", init=False)

    def __post_init__(self):
        for name in ("tau_M", "tau_K", "kelvin"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"Burgers {name} {v} must be positive")

    def shear_ratio(self, s):
        return 1.0 / (1.0 + 1.0 / (self.tau_M * s)
                      + 1.0 / (self.kelvin * (1.0 + self.tau_K * s)))


KINDS = {"maxwell": Maxwell, "burgers": Burgers}


def from_spec(spec):
    """A rheology from itself or from the dict a resolved dump holds."""
    if not isinstance(spec, dict):
        return spec
    d = dict(spec)
    kind = d.pop("kind")
    if kind not in KINDS:
        raise KeyError(f"unknown rheology kind {kind!r}; known: {sorted(KINDS)}")
    return KINDS[kind](**d)


def moduli(rheology, mu, lam, s):
    """``(mu(s), lam(s))`` of a region whose unrelaxed moduli are ``(mu, lam)``.

    Shear relaxes and the bulk modulus is kept, so ``lam(s) = K - 2 mu(s)/3``:
    formed from ``(mu, K)`` directly and never through Poisson's ratio (rule 8),
    which goes to 1/2 as the region relaxes.
    """
    K = lam + 2.0 * mu / 3.0
    mu_s = mu * rheology.shear_ratio(s)
    return mu_s, K - 2.0 * mu_s / 3.0


def g_of(mu, lam):
    """``g = 1/(1 - nu)``, the one function of Poisson's ratio the kernels see
    (docs/viscoelastic.md section 2), from ``(mu, lam)``: 1 <= g <= 2."""
    return 2.0 * (lam + mu) / (lam + 2.0 * mu)


def laplace_states(base, rheology, s) -> tuple:
    """The states of a Laplace run: the elastic one, then one per real ``s``.

    ``base`` maps every region to its unrelaxed ``(mu, lam)``, ``rheology`` the
    relaxing ones to their rheology. Largest ``s`` first, so each state is a
    small step in the moduli from the one before -- what lets the compressed
    backend keep its preconditioner along the chain.
    """
    from mbem.config import Material, State

    out = [State("elastic", {}, s=math.inf)]
    for k, sv in enumerate(sorted((float(v) for v in s), reverse=True), 1):
        out.append(State(
            f"s{k:02d}",
            {name: Material(*moduli(from_spec(r), *base[name], sv))
             for name, r in rheology.items()},
            s=sv))
    return tuple(out)


def require_one_eps(model, eps, regions) -> None:
    """Refuse a relaxing region whose closed boundary is not on ONE eps.

    With a single mollification width the kernel is one exact elastic field,
    and a uniform pressure on a closed surface then moves nothing in an
    incompressible body, on the faceted mesh, to rounding. A per-element width
    breaks that at 1e-4 .. 1e-3, and the defect is amplified by K / mu(s): on
    the Eshelby sphere with a relaxing host the hydrostatic response is 9 %
    off at mu(s)/K = 5e-4 and 54 % at 5e-5 with ``eps="auto"``, and within
    0.2 % at every step with one scalar (docs/viscoelastic.md, section 3).
    """
    from mbem.kernels import basis as kb

    for r in model.regions:
        if r.name not in regions or not r.patches:
            continue
        lo = min(float(kb.resolve_patch_eps(eps, p).min()) for p in r.patches)
        hi = max(float(kb.resolve_patch_eps(eps, p).max()) for p in r.patches)
        if hi - lo > 1e-12 * hi:
            raise ValueError(
                f"region {r.name!r} relaxes, so its closed boundary "
                f"({', '.join(p.name for p in r.patches)}) needs ONE scalar "
                f"eps; it has eps from {lo:.4g} to {hi:.4g}. Pass eps as a "
                f"number, or as a dict giving those patches the same number "
                f"(docs/viscoelastic.md, section 3)")


def require_stable(model) -> None:
    """Refuse a Laplace run on a body with a density contrast heavy side up.

    Under gravity such a body has a GROWING mode once anything in it relaxes
    (Rayleigh-Taylor), so its response to a step is not a constant plus
    decaying exponentials, and the fit this route inverts with has nothing
    else in it: the answer would be wrong and smooth. An elastic run may keep
    the contrast; rigidity holds it.
    """
    bad = model.unstable_contrasts()
    if bad:
        raise ValueError(
            f"gravity: the density contrast on {', '.join(bad)} is heavy side "
            f"up somewhere. A relaxing body is unstable there and has no "
            f"decaying-exponential response; give the Laplace run a stable "
            f"stratification (or equal densities)")

