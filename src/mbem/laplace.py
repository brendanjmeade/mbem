"""From elastic solves at real Laplace parameters to a function of time.

For data applied as a step, ``u~(s) = s u^(s)`` is the ELASTIC solution at the
moduli of ``s`` (docs/viscoelastic.md section 1). A relaxing body's response is
a constant plus decaying exponentials (plus a term linear in time if a fluid is
loaded by a traction), so ``u~`` is fitted by

    u~(s) = a0 + sum_j a_j s / (s + 1/tau_j)  [+ c / s]
    u(t)  = a0 + sum_j a_j exp(-t / tau_j)    [+ c t]

with the ``tau_j`` FIXED at the reciprocals of the collocation points
(Schapery's collocation). The fit is linear and involves no solution values,
so it is one matrix, the same for every unknown and every observation point:
``weights(t)`` maps the stack of state solutions to the solution at ``t``.

The elastic state (``s = inf``) is always one of the rows: it pins ``u(0+)``
exactly and makes the system square. With the secular term the slowest pole is
dropped, the two being all but collinear over the window.

Trust the result for ``t`` inside the span of the ``tau_j`` and not beyond it:
outside, nothing constrains the fit.
"""

from __future__ import annotations

import math

import numpy as np

from . import defaults


def s_grid(t_min: float, t_max: float, per_decade: int = None,
           pad: float = None) -> tuple:
    """Real collocation points for times in ``[t_min, t_max]``, largest first.

    ``s = 1/t`` log-spaced at ``per_decade`` points per decade, over the window
    widened by ``pad`` decades at each end.
    """
    per_decade = (defaults.LAPLACE_POINTS_PER_DECADE if per_decade is None
                  else per_decade)
    pad = defaults.LAPLACE_PAD_DECADES if pad is None else pad
    if not (0.0 < t_min < t_max):
        raise ValueError(f"need 0 < t_min < t_max, got {t_min}, {t_max}")
    lo = math.log10(1.0 / t_max) - pad
    hi = math.log10(1.0 / t_min) + pad
    n = int(math.ceil((hi - lo) * per_decade)) + 1
    return tuple(float(v) for v in np.logspace(hi, lo, n))


def window(s, pad: float = None) -> tuple:
    """``(t_min, t_max)`` that ``s_grid`` laid ``s`` out for: the times a fit
    on those points is trusted between."""
    pad = defaults.LAPLACE_PAD_DECADES if pad is None else pad
    return 10.0 ** pad / max(s), 10.0 ** -pad / min(s)


class Prony:
    """The fit for one set of collocation points.

    ``s`` are the FINITE points; the elastic state is implied and comes first,
    so a stack of solutions is ordered ``[elastic, s[0], s[1], ...]``.
    """

    def __init__(self, s, secular: bool = False, rcond: float = None):
        s = np.asarray(s, float)
        if s.ndim != 1 or s.size < 2 or np.any(s <= 0) or np.any(~np.isfinite(s)):
            raise ValueError("s must be at least two finite positive values")
        if np.unique(s).size != s.size:
            raise ValueError("collocation points must be distinct")
        self.s = s
        self.secular = bool(secular)
        tau = np.sort(1.0 / s)
        self.tau = tau[:-1] if secular else tau       # drop the slowest pole
        rows = np.concatenate([[np.inf], s])
        M = np.ones((rows.size, 1 + self.tau.size + int(secular)))
        with np.errstate(invalid="ignore"):
            M[:, 1:1 + self.tau.size] = 1.0 / (1.0 + 1.0 / (rows[:, None] * self.tau[None, :]))
        if secular:
            M[:, -1] = 1.0 / rows
        self.cond = float(np.linalg.cond(M))
        self._pinv = np.linalg.pinv(
            M, rcond=defaults.PRONY_RCOND if rcond is None else rcond)

    @property
    def window(self) -> tuple:
        """The span of the relaxation times: where the fit is constrained."""
        return float(self.tau.min()), float(self.tau.max())

    def _basis(self, t, kind: str, period=None):
        t = np.atleast_1d(np.asarray(t, float))
        e = np.exp(-t[:, None] / self.tau[None, :])
        B = np.zeros((t.size, 1 + self.tau.size + int(self.secular)))
        if kind == "value":
            B[:, 0] = 1.0
            B[:, 1:1 + self.tau.size] = e
            if self.secular:
                B[:, -1] = t
        elif kind == "rate":
            B[:, 1:1 + self.tau.size] = -e / self.tau
            if self.secular:
                B[:, -1] = 1.0
        else:
            if self.secular:
                raise ValueError("a creeping response has no periodic steady state")
            geo = 1.0 / (1.0 - np.exp(-float(period) / self.tau))
            if kind == "cycle":
                B[:, 1:] = (e - 1.0) * geo
            else:                                             # "cycle_rate"
                B[:, 1:] = -e / self.tau * geo
        return B

    def weights(self, t) -> np.ndarray:
        """``(n_t, n_states)``: ``u(t) = weights(t) @ stack`` for the response
        to a unit step."""
        return self._basis(t, "value") @ self._pinv

    def rate_weights(self, t) -> np.ndarray:
        """The same for ``du/dt``."""
        return self._basis(t, "rate") @ self._pinv

    def creep_rate_weights(self) -> np.ndarray:
        """``(n_states,)``: the steady creep rate ``c`` of a response fitted
        with the secular term, ``c = creep_rate_weights() @ stack``."""
        if not self.secular:
            raise ValueError("fitted without the secular term: no creep rate")
        return self._pinv[-1]

    def cycle_weights(self, t, period: float) -> np.ndarray:
        """The same step repeated every ``period`` for ever: the displacement
        accumulated since the latest one, at ``t`` after it, from the
        relaxation of all of them. ``sum_n exp(-(t + nT)/tau)`` is a geometric
        series, so this costs nothing."""
        return self._basis(t, "cycle", period) @ self._pinv

    def cycle_rate_weights(self, t, period: float) -> np.ndarray:
        """... and its rate: the interseismic velocity of the relaxation."""
        return self._basis(t, "cycle_rate", period) @ self._pinv

    def relaxed_weights(self) -> np.ndarray:
        """``(n_states,)``: the fit's constant, ``u(inf) = relaxed_weights() @
        stack``. It is where the response stood at the END of the window;
        whether the body had settled by then is the caller's to know."""
        if self.secular:
            raise ValueError("a creeping response never settles")
        return self._pinv[0]

    def locked_rate_weights(self, t, period: float) -> np.ndarray:
        """``(n_t, n_states)``: the velocity, per unit slip RATE, of a fault
        LOCKED between earthquakes that repeat every ``period``, measured
        from the same fault slipping steadily at that rate:
        ``period * sum_n du/dt(t + n period) - u(inf)``, at ``t`` after the
        latest earthquake.

        Its mean over a cycle is MINUS THE ELASTIC RESPONSE: the back-slip
        model (Savage 1983) is the cycle's average, and relaxation only moves
        velocity between early and late in it. What the response does long
        after a period enters the sum and the limit alike and all but
        cancels, so this asks less of the window than the limit itself does:
        a few periods."""
        return (float(period) * self.cycle_rate_weights(t, period)
                - self.relaxed_weights()[None, :])


def run_stack(run_dir) -> tuple:
    """``(s, {slot: (1 + len(s), ...) array})`` of a stored Laplace run.

    The solved slots of every state, stacked in the order ``Prony`` takes
    them -- the elastic state first, then the finite ``s`` as returned -- so

        s, stack = run_stack(run_dir)
        u_top = np.tensordot(Prony(s).weights(t), stack["u:top"], axes=1)

    is the top-surface displacement at the times ``t``. The states are read
    from the run's own record of what it solved, never from file names.
    """
    import json
    import pathlib

    from . import config as cfg

    d = pathlib.Path(run_dir)
    rows = [r for r in json.loads((d / "resolved.json").read_text())
            .get("states", []) if r.get("s") is not None]
    if not rows:
        raise ValueError(f"{d.name} is not a Laplace run: none of its states "
                         f"carries a Laplace parameter")
    rows.sort(key=lambda r: -float(r["s"]))            # inf, then descending
    if not math.isinf(float(rows[0]["s"])):
        raise ValueError(f"{d.name} has no elastic (s = inf) state")
    stack: dict = {}
    for r in rows:
        with np.load(d / f"fields_{cfg.slug(r['label'])}.npz") as z:
            for k in z.files:
                stack.setdefault(k, []).append(z[k])
    return (np.array([float(r["s"]) for r in rows[1:]]),
            {k: np.stack(v) for k, v in stack.items()})
