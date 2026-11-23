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

A body WITH NO FASTEST RELAXATION (``rheology.Andrade``: a spectrum that runs
on to short times as ``rate^-alpha``) breaks that fit as it stands. What has
relaxed faster than the fastest ``tau_j`` is then a fifth to two fifths of
the modulus, the elastic row pins it all the same, and the poles nearest the
short end take it up in the wrong shape: 3-21 % throughout the window, hardly
better at more points a decade, and falling only as ``t_min^alpha`` as the
window is lowered. So each exponent in ``tails`` gets one more term, in place
of the fastest pole: every pole from that one's place on, on the grid's own
spacing, their amplitudes tied to the power law,

    sum_k (rate_k / rate_0)^-alpha  s / (s + rate_k)       [one unknown]

which is what the spectrum is there to leading order. No more solves, the
elastic instant still exact, and the fit is back near where a body with a
fastest time has it: 1e-4 to 7e-3 at three points a decade, by how strong
the transient is at the window's short end (``defaults``). That holds where
the window reaches down to the transient's own time (Andrade's ``tau_A``):
above it the spectrum has bent away from the power law, and a stress is off
by 2-14 % (``rheology.warn_window``).

A POWER LAW THAT STOPS (``rheology.ExtendedBurgers.band``: a spectrum that
is ``rate^-alpha`` between two rates) is the same term with the amplitudes
of the poles outside the band left out, each pole standing for its own
share of the band. A band that runs on a few decades below the window and
is fitted as if it had no end is 0.6-3 % off; one the window reaches below
has no pole beyond the grid, and its tail is dropped: the plain fit, which
is then the right one.
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


def _tail(a) -> tuple:
    """``(exponent, fastest rate, slowest rate)`` of a tail given as an
    exponent (a power law with no end) or with the band of its rates."""
    a = (float(a), math.inf, 0.0) if np.ndim(a) == 0 else tuple(
        float(x) for x in a) + (math.inf, 0.0)[len(a) - 1:]
    if len(a) != 3 or not (0.0 < a[0] < 1.0 and 0.0 <= a[2] < a[1]):
        raise ValueError(f"a tail is an exponent in (0, 1), or that and a "
                         f"fastest and a slowest rate: {a}")
    return a


def _tail_shape(tail, tau0, step, k) -> tuple:
    """``(w, rest)``: the amplitudes a tail gives the poles at ``step^k /
    tau0`` beyond the grid, and what is left past the last of them.

    With no end they are ``step^(-alpha k)`` and a geometric series. A band
    gives each pole its share of ``tau^alpha`` between the band's ends
    within the pole's own bin (half a step either side), in the same
    measure: nothing outside the band, and the two agree inside it.
    """
    alpha, fast, slow = tail
    if fast == math.inf and slow == 0.0:
        w = step ** (-alpha * k)
        return w, float(w[-1] / (step ** alpha - 1.0))
    lo, hi = 1.0 / fast, (1.0 / slow if slow > 0.0 else math.inf)
    top = tau0 * step ** (0.5 - k)                    # each bin, in tau
    unit = tau0 ** alpha * (step ** (0.5 * alpha) - step ** (-0.5 * alpha))
    w = np.clip(np.minimum(top, hi) ** alpha
                - np.maximum(top / step, lo) ** alpha, 0.0, None) / unit
    rest = max(min(top[-1] / step, hi) ** alpha - lo ** alpha, 0.0) / unit
    return w, float(rest)


class Prony:
    """The fit for one set of collocation points.

    ``s`` are the FINITE points; the elastic state is implied and comes first,
    so a stack of solutions is ordered ``[elastic, s[0], s[1], ...]``.
    """

    def __init__(self, s, secular: bool = False, rcond: float = None,
                 tails=()):
        s = np.asarray(s, float)
        if s.ndim != 1 or s.size < 2 or np.any(s <= 0) or np.any(~np.isfinite(s)):
            raise ValueError("s must be at least two finite positive values")
        if np.unique(s).size != s.size:
            raise ValueError("collocation points must be distinct")
        self.s = s
        self.secular = bool(secular)
        tau = np.sort(1.0 / s)
        tau = tau[:-1] if secular else tau            # drop the slowest pole
        # the poles beyond the grid, on its own (geometric mean) spacing
        step = (s.max() / s.min()) ** (1.0 / (s.size - 1))
        k = np.arange(1, int(math.ceil(defaults.LAPLACE_TAIL_DECADES
                                       / math.log10(step))) + 1)
        # A tail takes the place of a fast pole. One whose power law stops
        # short of every pole beyond the grid is no tail of this fit, and
        # without it the grid's own poles reach one place further: so again
        live = sorted({_tail(a) for a in tails})
        while True:
            if tau.size - len(live) < 1:
                raise ValueError(f"{s.size} points leave no pole beside "
                                 f"{len(live)} tail(s)")
            self.tau = tau[len(live):]                # a tail for a fast one
            self._rate = step ** k / self.tau.min()   # (n_beyond,)
            shape = [_tail_shape(a, self.tau.min(), step, k) for a in live]
            keep = [bool(w.any() or rest) for w, rest in shape]
            if all(keep):
                break
            live = [a for a, ok in zip(live, keep) if ok]
        # as given: an exponent, or an exponent and the band of its rates
        self.tails = tuple(a[0] if a[1:] == (math.inf, 0.0) else a
                           for a in live)
        # (n_tails, n_beyond): each one's amplitudes, and what is left past
        # the last of them, which relaxes at once
        self._w = np.array([w for w, _ in shape]).reshape(len(live), k.size)
        self._rest = np.array([rest for _, rest in shape])
        rows = np.concatenate([[np.inf], s])
        m, nt = self.tau.size, len(self.tails)
        M = np.ones((rows.size, 1 + m + nt + int(secular)))
        with np.errstate(invalid="ignore"):
            M[:, 1:1 + m] = 1.0 / (1.0 + 1.0 / (rows[:, None] * self.tau[None, :]))
            if nt:
                beyond = 1.0 / (1.0 + self._rate[None, :] / rows[:, None])
                M[:, 1 + m:1 + m + nt] = beyond @ self._w.T
                M[0, 1 + m:1 + m + nt] += self._rest
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
        m, nt = self.tau.size, len(self.tails)
        e = np.exp(-t[:, None] / self.tau[None, :])
        B = np.zeros((t.size, 1 + m + nt + int(self.secular)))
        # a tail is exponentials too, at the rates beyond the grid, and what
        # is left past the last of them has gone by any t > 0
        f = np.exp(-t[:, None] * self._rate[None, :]) if nt else None
        tail = slice(1 + m, 1 + m + nt)
        if kind == "value":
            B[:, 0] = 1.0
            B[:, 1:1 + m] = e
            if nt:
                B[:, tail] = f @ self._w.T + np.outer(t == 0.0, self._rest)
            if self.secular:
                B[:, -1] = t
        elif kind == "rate":
            B[:, 1:1 + m] = -e / self.tau
            if nt:
                B[:, tail] = -(f * self._rate) @ self._w.T
            if self.secular:
                B[:, -1] = 1.0
        else:
            if self.secular:
                raise ValueError("a creeping response has no periodic steady state")
            geo = 1.0 / (1.0 - np.exp(-float(period) / self.tau))
            far = (1.0 / (1.0 - np.exp(-float(period) * self._rate))
                   if nt else None)
            if kind == "cycle":
                B[:, 1:1 + m] = (e - 1.0) * geo
                if nt:
                    B[:, tail] = (((f - 1.0) * far) @ self._w.T
                                  - np.outer(t > 0.0, self._rest))
            else:                                             # "cycle_rate"
                B[:, 1:1 + m] = -e / self.tau * geo
                if nt:
                    B[:, tail] = -(f * self._rate * far) @ self._w.T
        return B

    def coefficients(self) -> np.ndarray:
        """``(1 + len(tau) [+ tails] [+ 1], n_states)``: the fit itself, the
        map from a stack to ``[a0, a_1 .. a_m, (one per tail), (c)]``, each
        ``a_j`` the amplitude of ``exp(-t / tau[j - 1])``. Every method below
        is a row vector times this."""
        return self._pinv

    def memory(self, decades: float = None) -> tuple:
        """``(tau, C, gone)``: the fit as exponentials and nothing else. ``C``
        maps a stack to ``[a0, a_j]`` over ``tau``; ``gone @ stack`` is the
        part of the response that has relaxed by any lag greater than zero.
        What a consumer needs that carries the response as an equation in
        time rather than reading it at a time (``mbem.stepper``: one memory
        variable a pole).

        With no tail it is ``tau``, ``coefficients()`` and nothing gone. A
        tail is carried as its poles within ``decades`` of the fastest fitted
        one, and the rest of it is ``gone``: an instant relaxation cannot be
        stepped, and gone at once it is wrong only at lags a consumer of this
        window does not resolve (``defaults.STEPPER_TAIL_DECADES`` and what
        it was measured to cost). The elastic instant is ``C`` summed PLUS
        ``gone``.
        """
        if self.secular:
            raise ValueError("a creeping response is not a sum of exponentials")
        m, nt = self.tau.size, len(self.tails)
        if not nt:
            return self.tau, self._pinv, np.zeros(self._pinv.shape[1])
        decades = defaults.STEPPER_TAIL_DECADES if decades is None else decades
        keep = int(np.searchsorted(self._rate * self.tau.min(),
                                   10.0 ** decades * (1.0 + 1e-9)))
        d = self._pinv[1 + m:1 + m + nt]              # (n_tails, n_states)
        return (np.concatenate([self.tau, 1.0 / self._rate[:keep]]),
                np.vstack([self._pinv[:1 + m], self._w[:, :keep].T @ d]),
                (self._w[:, keep:].sum(axis=1) + self._rest) @ d)

    def weights(self, t) -> np.ndarray:
        """``(n_t, n_states)``: ``u(t) = weights(t) @ stack`` for the response
        to a unit step."""
        return self._basis(t, "value") @ self._pinv

    def rate_weights(self, t) -> np.ndarray:
        """The same for ``du/dt``."""
        return self._basis(t, "rate") @ self._pinv

    def creep_rate_weights(self) -> np.ndarray:
        """``(n_states,)``: the steady creep rate ``c`` of a response fitted
        with the secular term, ``c = creep_rate_weights() @ stack``. A body
        whose creep never becomes steady (Andrade) has none: ``c`` is then
        the fit's own, and ``rate_weights`` is what holds."""
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
        a few periods. (With a tail, what lies past its last pole relaxes
        AT the earthquake and is in no velocity after it: the mean over
        ``(0, T]`` is short of the elastic response by that much, 6e-6 of it
        at alpha = 1/3 and 8e-3 at 0.1.)"""
        return (float(period) * self.cycle_rate_weights(t, period)
                - self.relaxed_weights()[None, :])


def _run_states(run_dir) -> tuple:
    """``(directory, rows, spec)``: a stored Laplace run's states as it
    recorded them, the elastic one first, then by descending ``s``."""
    import json
    import pathlib

    d = pathlib.Path(run_dir)
    resolved = json.loads((d / "resolved.json").read_text())
    rows = [r for r in resolved.get("states", []) if r.get("s") is not None]
    if not rows:
        raise ValueError(f"{d.name} is not a Laplace run: none of its states "
                         f"carries a Laplace parameter")
    rows.sort(key=lambda r: -float(r["s"]))            # inf, then descending
    if not math.isinf(float(rows[0]["s"])):
        raise ValueError(f"{d.name} has no elastic (s = inf) state")
    return d, rows, resolved["spec"]


def run_stack(run_dir) -> tuple:
    """``(s, {slot: (1 + len(s), ...) array})`` of a stored Laplace run.

    The solved slots of every state, stacked in the order ``Prony`` takes
    them -- the elastic state first, then the finite ``s`` as returned -- so

        s, stack = run_stack(run_dir)
        u_top = np.tensordot(run_fit(run_dir).weights(t), stack["u:top"], axes=1)

    is the top-surface displacement at the times ``t``. The states are read
    from the run's own record of what it solved, never from file names.
    """
    from . import config as cfg

    d, rows, _spec = _run_states(run_dir)
    stack: dict = {}
    for r in rows:
        with np.load(d / f"fields_{cfg.slug(r['label'])}.npz") as z:
            for k in z.files:
                stack.setdefault(k, []).append(z[k])
    return (np.array([float(r["s"]) for r in rows[1:]]),
            {k: np.stack(v) for k, v in stack.items()})


def run_fit(run_dir, **kw) -> "Prony":
    """The fit of a stored Laplace run: ``Prony`` on its points AND with the
    tails its rheologies have (``rheology.tails``), both read from the run's
    own record. A consumer that built ``Prony(s)`` itself would give a body
    with no fastest relaxation the fit of one that has, and still look
    smooth. ``kw`` goes to ``Prony`` (``secular``)."""
    from . import rheology

    _d, rows, spec = _run_states(run_dir)
    return Prony([float(r["s"]) for r in rows[1:]], tails=rheology.tails(
        (spec.get("laplace") or {}).get("rheology", {}).values()), **kw)
