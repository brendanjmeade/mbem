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

import numpy as np


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


@dataclass(frozen=True)
class Andrade:
    """Transient creep that follows a power of time (Andrade 1910), on top of
    the elastic and the steady response:

        strain / (stress / mu) = 1 + (t / tau_A)^alpha + t / tau_M
        1/mu(s) = (1/mu) [1 + Gamma(1 + alpha) (tau_A s)^(-alpha)
                            + 1 / (tau_M s)]

    ``tau_M = eta / mu`` is the steady relaxation time, as in ``Burgers``;
    ``tau_A`` is the time at which the transient strain equals the elastic
    one (the usual ``beta`` is ``1 / (mu tau_A^alpha)``); ``alpha`` is the
    exponent, a third for the laboratory's olivine, and 1 is ``Maxwell`` at
    the two rates summed. A fluid.

    NO FASTEST RELAXATION: its spectrum runs on to short times as a power
    law, ``tail`` being the exponent, and a fit by exponentials that end at a
    window's short end is 8-28 % off throughout the window unless it is told
    (``laplace.Prony(tails=...)``, ``tails`` below). The elastic instant is
    exact all the same.
    """
    tau_M: float
    tau_A: float
    alpha: float
    kind: str = field(default="andrade", init=False)

    def __post_init__(self):
        for name in ("tau_M", "tau_A"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"Andrade {name} {v} must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"Andrade alpha {self.alpha} must be in (0, 1]: "
                             f"a creep that slows, at most linear in time")

    @property
    def tail(self):
        """The exponent of the power law its relaxation spectrum ends in at
        short times; None at ``alpha = 1``, which is Maxwell and has none."""
        return None if self.alpha == 1.0 else float(self.alpha)

    def shear_ratio(self, s):
        return 1.0 / (1.0 + math.gamma(1.0 + self.alpha)
                      * (self.tau_A * s) ** (-self.alpha)
                      + 1.0 / (self.tau_M * s))


@dataclass(frozen=True)
class ViscousFault:
    """A fault that resists its slip RATE: shear traction ``eta_f`` times the
    rate, the +n side dragging the other along. Stated as the shear zone it
    stands for -- ``width`` thick, of a material whose Maxwell time is
    ``tau`` at the unrelaxed shear modulus ``mu`` of the region holding the
    fault -- so ``eta_f = mu tau / width`` and, like the regions, it is a time
    and a ratio and never a viscosity. Only ``tau / width`` matters.

    In the Laplace domain the traction is ``drag(s)`` times the slip,
    ``drag = eta_f s``: infinite at ``s = inf``, so the fault is LOCKED at the
    instant of loading and slips afterwards, toward the slip that leaves it no
    traction. A patch of size L unloads in about ``tau L / width``. Friction
    that strengthens with rate, linearized about a steady rate ``V``, is the
    same law with ``eta_f = (a - b) sigma_n / V``.

    It is a law of a SOLVED fault (``BCType.FAULT_TRACTION``), whose slip is
    then driven by whatever loads it: afterslip, and the creep below a locked
    patch through its cycle. Dense backend only, as solved faults are.
    """
    tau: float
    width: float
    kind: str = field(default="viscous_fault", init=False)

    def __post_init__(self):
        for name in ("tau", "width"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"ViscousFault {name} {v} must be positive")

    def drag(self, mu, s):
        """``kappa(s)`` in GPa/km for a fault in a region of unrelaxed shear
        modulus ``mu`` (GPa)."""
        return math.inf if math.isinf(s) else mu * self.tau * s / self.width

    def rate(self, traction, mu):
        """The law in time: slip rate (km per unit of ``tau``) under
        ``traction`` (GPa), both ``(n, 2)`` in the fault's row frame
        (``equations.traction_row_frame``). What ``mbem.stepper`` integrates;
        every fault law has one, and only this law has a ``drag`` as well."""
        return np.asarray(traction, float) * (self.width / (mu * self.tau))


@dataclass(frozen=True)
class PowerLawFault:
    """A shear zone that flows as a power of its stress:

        rate = (traction / eta_f) (|traction| / stress)^(n - 1),

    ``eta_f = mu tau / width`` as in ``ViscousFault``, which this is at
    ``|traction| = stress`` and for ``n = 1``; above ``stress`` it is the
    softer of the two. Not linear, so it has no ``drag`` and no Laplace
    state: a fault with this law is stepped in time (``Run.stepper``).
    """
    tau: float
    width: float
    n: float
    stress: float
    kind: str = field(default="power_law_fault", init=False)

    def __post_init__(self):
        for name in ("tau", "width", "stress"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(f"PowerLawFault {name} {v} must be positive")
        if not (self.n >= 1.0 and math.isfinite(self.n)):
            raise ValueError(f"PowerLawFault n {self.n} must be >= 1: below "
                             f"it the rate is unbounded as the stress vanishes")

    def rate(self, traction, mu):
        t = np.asarray(traction, float)
        mag = np.linalg.norm(t, axis=-1, keepdims=True)
        return t * (self.width / (mu * self.tau)) \
            * (mag / self.stress) ** (self.n - 1.0)


@dataclass(frozen=True)
class RateStrengtheningFault:
    """Friction that strengthens with slip rate, at steady state: a fault
    sliding at ``v0`` under its ambient stress slides at
    ``v0 exp(dtau / a_sigma)`` when its shear traction along the rake changes
    by ``dtau`` (``a_sigma`` is ``(a - b) sigma_n``, GPa). Slip is measured
    FROM STEADY SLIP, as everything in a Laplace run is, so

        rate = v0 (exp(dtau / a_sigma) - 1)   along the rake,

    and none across it: the direction of sliding is the ambient traction's,
    which a change smaller than the fault's strength does not turn. ``rake``
    is in the fault's row frame, degrees from strike toward
    ``n x strike``, the convention of ``layered_box.burgers``. For
    ``dtau << a_sigma`` this is the ``ViscousFault`` of section 9 with
    ``eta_f = a_sigma / v0`` (``linearized``); above it the rate grows
    exponentially, which is why afterslip is logarithmic in time. A negative
    ``dtau`` slows the fault and never reverses it. No state variable: the
    fault is always at steady state. Stepped in time (``Run.stepper``).
    """
    v0: float
    a_sigma: float
    rake: float
    kind: str = field(default="rate_strengthening_fault", init=False)

    def __post_init__(self):
        for name in ("v0", "a_sigma"):
            v = getattr(self, name)
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(
                    f"RateStrengtheningFault {name} {v} must be positive")

    def direction(self) -> np.ndarray:
        """(2,): the rake in the row frame, quarter turns exact."""
        r = math.radians(self.rake)
        return np.array([round(math.cos(r), 15), round(math.sin(r), 15)])

    def rate(self, traction, mu=None):
        d = self.direction()
        along = np.asarray(traction, float) @ d
        return (self.v0 * np.expm1(along / self.a_sigma))[..., None] * d

    def linearized(self, mu) -> ViscousFault:
        """The linear law this one is for a small change of traction, on a
        fault in a region of unrelaxed shear modulus ``mu``."""
        return ViscousFault(tau=self.a_sigma / (self.v0 * mu), width=1.0)


KINDS = {"maxwell": Maxwell, "burgers": Burgers, "andrade": Andrade,
         "viscous_fault": ViscousFault, "power_law_fault": PowerLawFault,
         "rate_strengthening_fault": RateStrengtheningFault}


def from_spec(spec):
    """A rheology from itself or from the dict a resolved dump holds."""
    if not isinstance(spec, dict):
        return spec
    d = dict(spec)
    kind = d.pop("kind")
    if kind not in KINDS:
        raise KeyError(f"unknown rheology kind {kind!r}; known: {sorted(KINDS)}")
    return KINDS[kind](**d)


def tails(rheologies) -> tuple:
    """The exponents of the power laws a body's relaxation spectrum ends in
    at short times, sorted: one for every distinct ``tail`` among
    ``rheologies`` (rheologies, or the dicts a dump holds), and none for a
    body whose every relaxation has a fastest time. What the fit in time is
    given (``laplace.Prony(tails=...)``)."""
    found = {getattr(from_spec(r), "tail", None) for r in rheologies}
    return tuple(sorted(float(a) for a in found if a is not None))


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


def laplace_states(base, rheology, s, faults=None) -> tuple:
    """The states of a Laplace run: the elastic one, then one per real ``s``.

    ``base`` maps every region to its unrelaxed ``(mu, lam)``, ``rheology`` the
    relaxing ones to their rheology. Largest ``s`` first, so each state is a
    small step in the moduli from the one before -- what lets the compressed
    backend keep its preconditioner along the chain.

    ``faults`` maps a solved fault to ``(law, mu)``, its ``ViscousFault`` and
    the unrelaxed shear modulus of its region: each state then carries the
    fault's drag at its ``s``, infinite in the elastic state.
    """
    from mbem.config import Material, State

    faults = faults or {}

    def drag(sv):
        return {name: from_spec(law).drag(mu, sv)
                for name, (law, mu) in faults.items()}

    out = [State("elastic", {}, s=math.inf, drag=drag(math.inf))]
    for k, sv in enumerate(sorted((float(v) for v in s), reverse=True), 1):
        out.append(State(
            f"s{k:02d}",
            {name: Material(*moduli(from_spec(r), *base[name], sv))
             for name, r in rheology.items()},
            s=sv, drag=drag(sv)))
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

