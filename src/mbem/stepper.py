"""A solved fault through time, under a law that need not be linear.

The Laplace route (``mbem.laplace``) needs every part of a model to be
linear. A fault whose slip rate is not proportional to its traction is not,
but it is the only part that is not, so the body is REDUCED TO THE FAULT and
the fault alone is stepped (docs/viscoelastic.md section 6).

At each state of a run the dense backend eliminates every unknown but the
solved faults' slip (``AssembledDense.condense``), leaving on the faults

    traction = S_k @ slip + L_k .

In time ``S`` is a constant plus decaying exponentials, the fit the Laplace
route makes (``Prony.coefficients``), so a body that relaxes is one decaying
vector ``p_j`` per relaxation time of the fit:

    traction(t) = sum_k S_k c_k(t) + L(t),    c = C.T @ [slip; p_1 .. p_m]
    d slip / dt = law(traction)
    d p_j  / dt = d slip / dt - p_j / tau_j

an ordinary differential equation, integrated by an embedded Runge-Kutta
pair (``scipy.integrate.solve_ivp``, Dormand-Prince by default). A body that
does not relax is one state and no ``p``. Nothing is assumed of the law but
a rate for a traction; the body's linearity is what is used, and it is exact.

Two things the fit owes a consumer that FEEDS BACK through it:

* **Its relaxed limit must not push.** The constant of the fit is what the
  faults' stiffness has relaxed to at the end of the window. It misses by
  the fit's error (1e-3 of the kernel), and where the exact limit is zero or
  nearly -- a fault in a region that relaxes completely -- it misses with
  the wrong sign for some modes, which then GROW (2-5 % by the window's
  end, measured). ``Body`` moves the growing part of that constant's
  spectrum onto the slowest pole: the elastic instant stays exact, the
  kernel changes by less than the fit's own error, and no mode grows.
* **The window is the body's, the step is the fault's.** The fit must reach
  below the body's fastest relaxation and beyond the last time asked for
  (``march`` refuses a later one); how fast the FAULT moves is the
  integrator's business, and its step is bounded by the fit's fastest pole
  (about 3.3 of its relaxation times), not by the law.

Dense backend only: a compressed operator solves no slip, and the reduction
is one right-hand side per slip unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import defaults, laplace


class Body:
    """What a body's solved faults feel through time.

    ``states`` are the ``Condensed`` operators of a run's states in the order
    ``Prony`` takes them, the elastic one first, and ``s`` their finite
    Laplace parameters; ``s=None`` is a body that does not relax (one
    state). ``passive=False`` keeps the fit's relaxed limit as fitted: for a
    gate that shows what that costs.
    """

    def __init__(self, states, s=None, passive: bool = True):
        states = list(states)
        self.states = states
        self.faults = states[0].faults
        self.layout = states[0].layout
        self.n = states[0].n
        if s is None:
            if len(states) != 1:
                raise ValueError(f"a body that does not relax is ONE state; "
                                 f"got {len(states)} and no s")
            self.tau, self.C, self.window = np.zeros(0), np.ones((1, 1)), None
        else:
            s = np.asarray(s, float)
            if len(states) != 1 + s.size:
                raise ValueError(f"{s.size} Laplace parameters want "
                                 f"{1 + s.size} states (the elastic one "
                                 f"first), got {len(states)}")
            pr = laplace.Prony(s)
            self.tau, self.C = pr.tau, pr.coefficients()
            self.window = laplace.window(s)
        K, n = len(states), self.n
        S = np.stack([c.S for c in states])
        # traction = _S @ c.ravel(), c the (K, n) slip each state is asked for
        self._S = np.ascontiguousarray(S.transpose(1, 0, 2).reshape(n, K * n))
        self._aL = self.C @ np.stack([c.L for c in states])
        self._slow = int(np.argmax(self.tau)) if self.tau.size else -1
        self._D, self.growing, self.removed = None, 0, 0.0
        if passive and self.tau.size:
            self._make_passive(np.tensordot(self.C[0], S, axes=1), S[0])

    def _make_passive(self, relaxed, elastic) -> None:
        """Take the growing part of the fit's relaxed limit out of it and
        give it to the slowest pole (module docstring): the part of
        ``relaxed`` on its invariant subspace of eigenvalues with a positive
        real part, by an ordered Schur form, which stays accurate where the
        eigenvectors of a matrix that is not symmetric do not."""
        from scipy.linalg import schur, solve_sylvester

        T, Z, k = schur(relaxed, output="real", sort="rhp")
        if k == 0:
            return
        n = relaxed.shape[0]
        top = np.zeros((k, n))
        top[:, :k] = T[:k, :k]
        if k < n:
            # the spectral projector onto that subspace is Z [[I, Y], [0, 0]] Z'
            Y = solve_sylvester(T[:k, :k], -T[k:, k:], -T[:k, k:])
            top[:, k:] = T[:k, :k] @ Y
        self._D = Z[:, :k] @ top @ Z.T
        self.growing = int(k)
        self.removed = float(np.linalg.norm(self._D, 2)
                             / np.linalg.norm(elastic, 2))

    # -- the pieces of the equation ------------------------------------

    def _basis(self, lag) -> np.ndarray:
        """``[1, exp(-lag / tau_j)]``: a unit step's fit, ``lag`` after it."""
        return np.concatenate([[1.0], np.exp(-lag / self.tau)])

    def load(self, t, events) -> np.ndarray:
        """(n,): what the prescribed data load the held faults with at ``t``,
        each of ``events = [(time, amplitude)]`` a step of them."""
        out = np.zeros(self.n)
        for te, amp in events:
            if t >= te:
                out += amp * (self._basis(t - te) @ self._aL)
        return out

    def traction(self, Y, load) -> np.ndarray:
        """(n,): the traction on the faults for the state ``Y = [slip; p]``,
        ``(1 + m, n)``."""
        out = self._S @ (self.C.T @ Y).ravel()
        if self._D is not None:
            out -= self._D @ (Y[0] - Y[1 + self._slow])
        return out + load

    def unknowns(self, t, Y, events) -> np.ndarray:
        """(N,): every unknown of the model at ``t``, in the layout's order
        (``Condensed.slots`` names them): the prescribed data's part, each
        state at the fit's weight, and the faults' slip through each state's
        own response."""
        w = np.zeros(len(self.states))
        for te, amp in events:
            if t >= te:
                w += amp * (self._basis(t - te) @ self.C)
        c = self.C.T @ Y
        x = np.zeros(self.states[0].x0.shape[0])
        for k, st in enumerate(self.states):
            x += w[k] * st.x0 + st.X @ c[k]
        return x

    def slip_scale(self) -> float:
        """The largest slip the elastic body's faults would take to shed the
        load entirely: what an absolute tolerance on slip is a fraction of."""
        el = self.states[0]
        if not np.any(el.L):
            return 0.0
        return float(np.abs(np.linalg.solve(el.S, el.L)).max())

    def growth(self, mobility) -> float:
        """The largest real part among the eigenvalues of this body with the
        LINEAR law ``d slip/dt = mobility * traction`` (a number, or one per
        coordinate): positive is a mode that grows. A diagnostic, of the size
        of the whole state -- for a gate, not a run."""
        n, m = self.n, self.tau.size
        K = len(self.states)
        G = np.tensordot(self.C, self._S.reshape(n, K, n).transpose(1, 0, 2),
                         axes=1)
        if self._D is not None:
            G[0] -= self._D
            G[1 + self._slow] += self._D
        mob = np.broadcast_to(np.asarray(mobility, float), (n,))
        J = np.zeros(((1 + m) * n, (1 + m) * n))
        for i in range(1 + m):
            for j in range(1 + m):
                J[i * n:(i + 1) * n, j * n:(j + 1) * n] = mob[:, None] * G[j]
            if i:
                J[i * n:(i + 1) * n, i * n:(i + 1) * n] -= \
                    np.eye(n) / self.tau[i - 1]
        return float(np.linalg.eigvals(J).real.max())


def law_rate(body: Body, laws, mu):
    """``traction -> slip rate`` on a body's coordinates, from a law per
    fault: ``laws`` maps EVERY solved fault to its law (``rheology``), ``mu``
    to the unrelaxed shear modulus of the region holding it. A solved fault
    with no law would be an algebraic constraint, which nothing here steps."""
    from . import rheology

    missing = [f.name for f in body.faults if f.name not in laws]
    if missing:
        raise KeyError(f"no law for the solved fault(s) {missing}: every "
                       f"FAULT_TRACTION fault of a stepped run needs one")
    parts = []
    for f in body.faults:
        law = rheology.from_spec(laws[f.name])
        if not hasattr(law, "rate"):
            raise TypeError(f"fault {f.name!r}: {law!r} is not a fault law "
                            f"(it has no rate)")
        if f.frame.shape[1] != 2:
            raise ValueError(f"fault {f.name!r} opens (allow_opening): the "
                             f"laws are of a fault that slides")
        parts.append((f.start, f.stop, law, float(mu[f.name])))

    def rate(traction):
        out = np.empty_like(traction)
        for a, b, law, m in parts:
            out[a:b] = law.rate(traction[a:b].reshape(-1, 2), m).ravel()
        return out
    return rate


@dataclass
class History:
    """What ``march`` returns: the faults at the times asked for.

    ``y[i]`` is the whole state ``[slip; p]`` at ``t[i]`` and ``traction[i]``
    the traction the law saw there, both in the body's coordinates.
    ``nfev`` counts the law's evaluations (six to a Dormand-Prince step)."""
    t: np.ndarray
    y: np.ndarray               # (n_t, 1 + m, n)
    traction: np.ndarray        # (n_t, n)
    body: Body
    events: tuple
    nfev: int

    @property
    def slip(self) -> np.ndarray:
        return self.y[:, 0]

    def fault(self, name: str, what: str = "slip") -> np.ndarray:
        """``(n_t, n_dofs, 3)``: a fault's Burgers vector, or the traction on
        it (``what="traction"``), as Cartesian vectors at its dof nodes."""
        f = next(f for f in self.body.faults if f.name == name)
        src = self.slip if what == "slip" else self.traction
        return np.stack([f.cartesian(v[f.start:f.stop]) for v in src])

    def slots(self, i: int) -> dict:
        """Every unknown at ``t[i]``, as a solve returns them."""
        return self.body.states[0].slots(
            self.body.unknowns(self.t[i], self.y[i], self.events))


def _events(events) -> tuple:
    out = sorted((float(e[0]), float(e[1])) if np.ndim(e) else (float(e), 1.0)
                 for e in events)
    if not out:
        raise ValueError("march: no event, so nothing ever loads the faults")
    return tuple(out)


def march(body: Body, rate, t, events=(0.0,), rtol: float = None,
          atol: float = None, method: str = None) -> History:
    """Integrate a body's faults under ``rate`` and return them at ``t``.

    ``rate`` maps the traction on the faults to their slip rate, both in the
    body's coordinates (``law_rate`` makes one from laws). ``events`` are the
    times at which the model's prescribed data are applied as a step -- a
    time, or ``(time, amplitude)`` -- the first of them the start: the faults
    are unslipped before it, and they do not slip AT a step, whatever the
    law, because a rate is finite. ``rtol`` is the integrator's, and ``atol``
    its absolute tolerance on slip as a fraction of ``body.slip_scale()``.

    A relaxing body is refused past the end of its fit's window, counted
    from the first event: beyond it nothing constrains the kernel.
    """
    from scipy.integrate import solve_ivp

    rtol = defaults.STEPPER_RTOL if rtol is None else rtol
    atol = defaults.STEPPER_ATOL if atol is None else atol
    method = defaults.STEPPER_METHOD if method is None else method
    events = _events(events)
    t = np.atleast_1d(np.asarray(t, float))
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) < 0.0):
        raise ValueError("march: t must be one or more non-decreasing times")
    t0 = events[0][0]
    if t[0] < t0:
        raise ValueError(f"march: t = {t[0]:g} is before the first event "
                         f"({t0:g})")
    if body.window is not None and t[-1] - t0 > body.window[1] * (1 + 1e-9):
        raise ValueError(
            f"march: t = {t[-1]:g} is {t[-1] - t0:g} after the first event, "
            f"past the end of the fit's window ({body.window[1]:g}): beyond "
            f"it nothing constrains the body's memory. Lay the Laplace "
            f"points out for a longer window")
    n, m = body.n, body.tau.size
    scale = body.slip_scale()
    atol_abs = atol * scale if scale > 0.0 else np.finfo(float).tiny
    count = [0]

    def fun_for(active):
        def fun(tt, y):
            count[0] += 1
            Y = y.reshape(1 + m, n)
            with np.errstate(over="ignore", invalid="ignore"):
                v = rate(body.traction(Y, body.load(tt, active)))
            # A TRIAL stage may overshoot far enough for a law exponential in
            # traction to overflow. That is a step too long, not a result:
            # hand the integrator a rate so large that it rejects the step.
            v = np.nan_to_num(v, nan=defaults.STEPPER_REJECT,
                              posinf=defaults.STEPPER_REJECT,
                              neginf=-defaults.STEPPER_REJECT)
            out = np.empty_like(Y)
            out[0] = v
            if m:
                out[1:] = v[None, :] - Y[1:] / body.tau[:, None]
            return out.ravel()
        return fun

    edges = [e[0] for e in events if e[0] < t[-1]] + [float(t[-1])]
    y = np.zeros((1 + m) * n)
    got = np.empty((t.size, (1 + m) * n))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        sel = np.flatnonzero((t >= a) & ((t <= b) if last else (t < b)))
        active = tuple(e for e in events if e[0] <= a)
        te = np.unique(np.concatenate([t[sel], [b]]))
        fun = fun_for(active)
        v0 = np.abs(fun(a, y)[:n])
        if v0.max() >= defaults.STEPPER_REJECT:
            raise FloatingPointError(
                f"march: at t = {a:g} the law starts at a rate beyond "
                f"{defaults.STEPPER_REJECT:g}, or at none: a law exponential "
                f"in traction does where the traction is over a hundred "
                f"times its scale, and no step is short enough")
        # the integrator's own first step probes with one that a stiff law
        # overshoots; start from a step that moves the slip by a millionth
        # of its scale, and let it grow
        first = (min(b - a, defaults.STEPPER_FIRST_STEP * scale / v0.max())
                 if scale > 0.0 and v0.max() > 0.0 else None)
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(fun, (a, b), y, method=method, t_eval=te,
                            rtol=rtol, atol=atol_abs, first_step=first)
        if sol.status != 0:
            raise RuntimeError(f"march: the integrator stopped at "
                               f"t = {sol.t[-1] if sol.t.size else a:g}: "
                               f"{sol.message}")
        got[sel] = sol.y[:, np.searchsorted(te, t[sel])].T
        y = sol.y[:, -1]
    if len(edges) == 1:                          # asked only AT the start
        got[:] = 0.0
    Y = got.reshape(t.size, 1 + m, n)
    trac = np.stack([body.traction(Y[i], body.load(t[i], events))
                     for i in range(t.size)])
    return History(t=t, y=Y, traction=trac, body=body, events=events,
                   nfev=count[0])
