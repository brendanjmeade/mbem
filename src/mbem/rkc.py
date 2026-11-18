"""A stabilised explicit integrator: Runge-Kutta-Chebyshev, order two.

An explicit Runge-Kutta pair is stable to a step of about 3.3 over the
fastest rate of what it integrates, so it takes a number of steps
proportional to that rate however slowly the answer moves. A fault's fastest
rate is its shortest wavelength's, which rises as its elements shrink, and
the slip the law is asked for does not move faster for it: a fine fault is
STIFF, and stiff along the negative real axis, the fault being dissipative.

The Chebyshev methods spend their stages on stability instead of order: a
step of ``s`` stages is stable to ``0.653 s^2`` over that rate, so a step
the accuracy allows costs the SQUARE ROOT of what it costs the pair, and
still one right-hand side per stage and no linear solve -- which is all a
compressed kernel offers. This is the method of Sommeijer, Shampine and
Verwer (RKC, J. Comput. Appl. Math. 88, 1998): the damped second-order
polynomial, its embedded error estimate and step controller, and the
spectral radius measured by a power iteration on the right-hand side
itself, so that nothing is asked of the law but its rate.

``RKC`` is a ``scipy.integrate.OdeSolver``, the ``method`` of ``solve_ivp``:
the times asked for are read from a cubic Hermite interpolant of each step,
which costs nothing, the rate at both ends being in hand.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import DenseOutput, OdeSolver

from . import defaults


class _Hermite(DenseOutput):
    """The cubic through a step's two ends and their rates."""

    def __init__(self, t_old, t, y_old, f_old, y, f):
        super().__init__(t_old, t)
        self._h = t - t_old
        self._ends = (y_old, f_old, y, f)

    def _call_impl(self, t):
        y0, f0, y1, f1 = self._ends
        x = (np.asarray(t, float) - self.t_old) / self._h
        w = ((1.0 + 2.0 * x) * (1.0 - x) ** 2, x * (1.0 - x) ** 2 * self._h,
             x * x * (3.0 - 2.0 * x), x * x * (x - 1.0) * self._h)
        if x.ndim == 0:
            return w[0] * y0 + w[1] * f0 + w[2] * y1 + w[3] * f1
        return (np.outer(y0, w[0]) + np.outer(f0, w[1])
                + np.outer(y1, w[2]) + np.outer(f1, w[3]))


def stability_bound(stages: int) -> float:
    """How far along the negative real axis a step of ``stages`` is stable,
    in units of one over the step: the damped polynomial's, less the margin
    the stage count below is chosen with (1 / 1.54 = 0.649 of ``s^2 - 1``)."""
    return (stages * stages - 1.0) / defaults.RKC_STAGES_PER_RADIUS


class RKC(OdeSolver):
    """``solve_ivp(..., method=RKC)``. ``rtol`` and ``atol`` are scipy's, the
    error of a step measured in the root-mean-square norm of its own pairs;
    ``first_step`` is taken as given, and chosen as the reference code
    chooses it otherwise.

    ``spectral_every`` is how many accepted steps an estimate of the fastest
    rate serves. It is measured again at once after a step is rejected: a
    rate that has RISEN shows as an error the stages no longer damp."""

    def __init__(self, fun, t0, y0, t_bound, rtol=1.0e-3, atol=1.0e-6,
                 first_step=None, max_step=np.inf, vectorized=False,
                 spectral_every=None):
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=False)
        if not rtol > 10.0 * np.finfo(float).eps:
            raise ValueError("RKC: rtol must exceed ten times the rounding")
        self.rtol, self.atol = float(rtol), np.asarray(atol, float)
        self.max_step = float(max_step)
        self.spectral_every = (defaults.RKC_SPECTRAL_EVERY
                               if spectral_every is None else int(spectral_every))
        # more stages than this and the recurrence's rounding, which grows
        # as their square, reaches a tenth of the tolerance
        self.stages_max = max(2, int(round(np.sqrt(
            self.rtol / (10.0 * np.finfo(float).eps)))))
        self.f = self.fun(self.t, self.y)
        self.rho = None
        self._vector = None
        self._measured_at = -1
        self._stale = True
        self._last = None                       # (error, step) of the last accepted
        self.n_accepted = self.n_rejected = self.n_spectral = 0
        self.stages = 0
        self.y_old = self.f_old = None
        self.h_abs = (self._first_step() if first_step is None
                      else float(first_step))

    # -- the fastest rate ------------------------------------------------

    def _unit(self, y) -> np.ndarray:
        """A solution's own size, a component at a time: what the error is
        relative to, over the relative tolerance."""
        return self.atol / self.rtol + np.abs(y)

    def _spectral_radius(self) -> None:
        """The largest rate of the right-hand side's Jacobian here, by the
        power iteration on differences of the right-hand side, to a part in
        a hundred and then held 20 % above."""
        t, y, f = self.t, self.y, self.f
        d = np.sqrt(np.finfo(float).eps) * np.linalg.norm(self._unit(y))
        v = f if self._vector is None else self._vector
        norm = np.linalg.norm(v)
        if not norm > 0.0:
            v, norm = np.ones_like(y), np.sqrt(y.size)
        v = y + v * (d / norm)
        sigma = 0.0
        for it in range(defaults.RKC_SPECTRAL_ITERATIONS):
            df = self.fun(t, v) - f
            norm = np.linalg.norm(df)
            last, sigma = sigma, norm / d
            if it and abs(sigma - last) <= 0.01 * sigma:
                break
            if not norm > 0.0:                  # a direction the rate ignores
                break
            v = y + df * (d / norm)
        self._vector = v - y
        self.rho = defaults.RKC_SPECTRAL_SAFETY * sigma
        self._measured_at = self.n_accepted
        self._stale = False
        self.n_spectral += 1

    def _first_step(self) -> float:
        self._spectral_radius()
        span = abs(self.t_bound - self.t)
        h = min(self.max_step, span)
        if self.rho * h > 1.0:
            h = 1.0 / self.rho
        f1 = self.fun(self.t + self.direction * h,
                      self.y + self.direction * h * self.f)
        est = h * self._norm((f1 - self.f) / self._unit(self.y)) / self.rtol
        if 0.1 * h < min(self.max_step, span) * np.sqrt(est):
            h = 0.1 * h / np.sqrt(est)
        return h

    # -- a step ----------------------------------------------------------

    @staticmethod
    def _norm(x) -> float:
        return float(np.linalg.norm(x) / np.sqrt(x.size))

    def _advance(self, t, y, f, h, s) -> np.ndarray:
        """``s`` stages of the damped second-order Chebyshev recurrence."""
        w0 = 1.0 + defaults.RKC_DAMPING / (s * s)
        root = np.sqrt(w0 * w0 - 1.0)
        arg = s * np.log(w0 + root)
        w1 = (np.sinh(arg) * (w0 * w0 - 1.0)
              / (np.cosh(arg) * s * root - w0 * np.sinh(arg)))
        b1 = b2 = 1.0 / (2.0 * w0) ** 2
        mus = w1 * b1
        y2, y1 = y, y + (h * mus) * f
        c2, c1 = 0.0, mus
        z1, z2, dz1, dz2, d2z1, d2z2 = w0, 1.0, 1.0, 0.0, 0.0, 0.0
        for j in range(2, s + 1):
            z = 2.0 * w0 * z1 - z2
            dz = 2.0 * w0 * dz1 - dz2 + 2.0 * z1
            d2z = 2.0 * w0 * d2z1 - d2z2 + 4.0 * dz1
            b = d2z / (dz * dz)
            a1 = 1.0 - z1 * b1
            mu, nu = 2.0 * w0 * b / b1, -b / b2
            mus = mu * w1 / w0
            yj = (mu * y1 + nu * y2 + (1.0 - mu - nu) * y
                  + (h * mus) * (self.fun(t + h * c1, y1) - a1 * f))
            c = mu * c1 + nu * c2 + mus * (1.0 - a1)
            y2, y1, c2, c1 = y1, yj, c1, c
            b2, b1 = b1, b
            z2, z1, dz2, dz1, d2z2, d2z1 = z1, z, dz1, dz, d2z1, d2z
        return y1

    def _step_impl(self):
        t, y, f = self.t, self.y, self.f
        h_min = 10.0 * abs(np.nextafter(t, self.direction * np.inf) - t)
        h_abs = min(self.h_abs, self.max_step)
        while True:
            if self._stale:
                self._spectral_radius()
            h_abs = min(h_abs, abs(self.t_bound - t))
            if h_abs < h_min:
                return False, self.TOO_SMALL_STEP
            s = 1 + int(np.sqrt(defaults.RKC_STAGES_PER_RADIUS
                                * h_abs * self.rho + 1.0))
            if s > self.stages_max:
                s = self.stages_max
                h_abs = stability_bound(s) / self.rho
            s = max(s, 2)
            t_new = t + self.direction * h_abs
            if self.direction * (t_new - self.t_bound) > 0.0 or \
                    abs(self.t_bound - t_new) < h_min:
                t_new = self.t_bound
            h = t_new - t
            y_new = self._advance(t, y, f, h, s)
            f_new = self.fun(t_new, y_new)
            est = 0.8 * (y - y_new) + 0.4 * h * (f + f_new)
            unit = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
            err = self._norm(est / unit)
            if np.isfinite(err) and err <= 1.0:
                break
            self.n_rejected += 1
            h_abs *= max(0.1, 0.8 * err ** (-1.0 / 3.0)) if np.isfinite(err) else 0.1
            self._stale = self._measured_at != self.n_accepted
        err = max(err, 1.0e-10)
        if self._last is None:
            fac = 0.8 * err ** (-1.0 / 3.0)
        else:                                   # the controller with memory
            e_old, h_old = self._last
            fac = 0.8 * h_abs * e_old ** (1.0 / 3.0) / (h_old * err ** (2.0 / 3.0))
        self._last = (err, h_abs)
        self.h_abs = h_abs * min(10.0, max(0.1, fac))
        self.y_old, self.f_old = y, f
        self.t, self.y, self.f = t_new, y_new, f_new
        self.n_accepted += 1
        self.stages += s
        if self.n_accepted - self._measured_at >= self.spectral_every:
            self._stale = True
        return True, None

    def _dense_output_impl(self):
        return _Hermite(self.t_old, self.t, self.y_old, self.f_old,
                        self.y, self.f)
