"""1-D edge primitives: the closed-form antiderivatives and the
cancellation-free difference table ``edge_table``.

  (a) symbolic: for m in {-3,-1,1,3,5,7} and k = 0..7, sympy-differentiate
      ``antiderivative(k, m, u, rho2, mod=sympy)`` and require
      simplify(dF/du - u^k (u^2 + rho2)^(-m/2)) == 0;
  (b) numeric: ``edge_table(ua, ub, rho2, spec={5:5, 3:4, 1:3, -1:1})`` vs a
      40-digit mpmath quadrature of u^k (u^2 + rho2)^(-m/2) over [ua, ub],
      the interval split at every point of {-10 rho, -rho, 0, rho, 10 rho}
      inside it (plus +-10^j rho for wide intervals, so tanh-sinh resolves
      the peak and the algebraic tails) -- 13 regimes covering the
      closed-form (mixed / same-sign / vertex), the large-|u| series (far
      along-strike, wide) and the small-|u| series (broadside, symmetric)
      branches, 1e-12 relative per entry.  The spec covers every (m, k) the
      kernels ask for, the force (Kelvin single-layer) element included: its
      ``n = 1`` moment row at nodal order p needs (m, k) = (1, 3) and, through
      the downward vertical seed ``I_{-1}``, (m, k) = (-1, 1) -- neither of
      which the dislocation kernels reach.  Worst measured over all 13
      regimes with the extended spec: 1.5e-15;
  (c) rho = 0 (an observer on an edge's LINE with h_eps = 0, which happens
      bit-exactly at every vertex and edge midpoint of an axis-aligned
      triangle, and which only the force element ever reaches -- eps = 0 on
      the element is permitted for ``want=("G",)`` alone).  ``R`` collapses to
      ``|u|`` and the primitive is the elementary ``int u^k |u|^-m du``.  Every
      convergent entry is checked against mpmath of that integral, split at 0
      (worst 1.7e-16, gated at 1e-12); every entry with ``q = k - m + 1 <= 0``
      on an interval whose closed span contains u = 0 is a genuinely divergent
      principal value and must come back as ``+inf``, never as a finite number
      (the antiderivative difference would silently cancel into a wrong,
      finite value -- that is the failure this row exists to catch);
  (d) batch: all regimes in one ``edge_table`` call equal the per-case
      results bit for bit (regime masks are per element), including a mixed
      batch that carries rho = 0 and rho > 0 rows together.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import mpmath as mp

from _common import Report
from clq.primitives import antiderivative, edge_table
import sys

# (m: kmax).  m = 1 up to k = 3 and m = -1 up to k = 1 are reached only by the
# force element's n = 1 moment row and its I_{-1} seed; see the module docstring.
SPEC = {5: 5, 3: 4, 1: 3, -1: 1}

# rho = 0 regime: (label, ua, ub).  rho2 is exactly 0.0 for all of them.
RHO0_CASES = [
    ("rho=0 positive     (0.3, 1.4)", 0.3, 1.4),
    ("rho=0 negative     (-1.4, -0.3)", -1.4, -0.3),
    ("rho=0 straddling   (-0.7, 0.9)", -0.7, 0.9),
    ("rho=0 foot at ua   (0, 1.1)", 0.0, 1.1),
    ("rho=0 foot at ub   (-1.1, 0)", -1.1, 0.0),
]

# (label, ua, ub, rho2)
CASES = [
    ("mixed near        (-0.7, 0.9, 0.04)", -0.7, 0.9, 0.04),
    ("same-sign near +  (0.3, 1.4, 0.04)", 0.3, 1.4, 0.04),
    ("same-sign near -  (-1.4, -0.3, 0.04)", -1.4, -0.3, 0.04),
    ("vertex +          (0, 1, 0.09)", 0.0, 1.0, 0.09),
    ("vertex -          (-1, 0, 0.09)", -1.0, 0.0, 0.09),
    ("far strike +      (1000, 1001, 1e-4)", 1000.0, 1001.0, 1e-4),
    ("far strike -      (-1001, -1000, 1e-4)", -1001.0, -1000.0, 1e-4),
    ("mixed far         (-1e4, 1e4+1, 1e-4)", -1e4, 1e4 + 1.0, 1e-4),
    ("wide              (0.5, 1e5, 1e-4)", 0.5, 1e5, 1e-4),
    ("broadside small + (0.05, 0.06, 1)", 0.05, 0.06, 1.0),
    ("broadside small m (-0.02, 0.03, 4)", -0.02, 0.03, 4.0),
    ("tiny edge         (1, 1+1e-7, 1e-2)", 1.0, 1.0 + 1e-7, 1e-2),
    ("symmetric         (-0.5, 0.49, 1)", -0.5, 0.49, 1.0),
]


# ---------------------------------------------------------------------------
# (a) symbolic antiderivatives
# ---------------------------------------------------------------------------

def symbolic_check(rep: Report) -> None:
    u, rho2 = sp.symbols("u rho2", positive=True)
    bad = []
    for m in (-3, -1, 1, 3, 5, 7):
        for k in range(8):
            F = antiderivative(k, m, u, rho2, mod=sp)
            resid = sp.simplify(sp.diff(F, u) - u ** k / (u ** 2 + rho2) ** sp.Rational(m, 2))
            if resid != 0:
                bad.append((k, m))
    rep.check_bool("symbolic: dF/du == u^k R^-m for m in {-3..7 odd}, k=0..7 (48)",
                   not bad, f"failures: {bad}" if bad else "(all 48 identities simplify to 0)")


# ---------------------------------------------------------------------------
# (b) mpmath reference
# ---------------------------------------------------------------------------

def split_points(ua: float, ub: float, rho: float) -> list:
    """Integration break points: the endpoints plus every point of
    {-10 rho, -rho, 0, rho, 10 rho} strictly inside, plus +-10^j rho (j >= 2)
    inside so tanh-sinh never sees a piece spanning many decades."""
    a, b = mp.mpf(ua), mp.mpf(ub)
    r = mp.mpf(rho)
    cand = [-10 * r, -r, mp.mpf(0), r, 10 * r]
    j = 2
    while 10 ** j * r < max(abs(a), abs(b)):
        cand += [-(10 ** j) * r, (10 ** j) * r]
        j += 1
    inner = sorted(c for c in cand if a < c < b)
    return [a] + inner + [b]


def reference_table(ua: float, ub: float, rho2: float) -> tuple[dict, float]:
    """{m: [P_k^m, k=0..kmax]} as mpf plus the worst mp.quad error estimate."""
    mp.mp.dps = 40
    r2 = mp.mpf(rho2)
    pts = split_points(ua, ub, mp.sqrt(r2))
    out = {}
    worst_err = mp.mpf(0)
    for m, kmax in SPEC.items():
        row = []
        for k in range(kmax + 1):
            f = lambda t, k=k, m=m: t ** k * (t * t + r2) ** (-mp.mpf(m) / 2)
            val, err = mp.quad(f, pts, error=True)
            row.append(val)
            worst_err = max(worst_err, err / max(abs(val), mp.mpf("1e-300")))
        out[m] = row
    return out, float(worst_err)


def numeric_check(rep: Report) -> list[dict]:
    """Per-case edge_table vs mpmath.  Returns the per-case clq tables."""
    per_case = []
    overall = 0.0
    overall_lbl = ""
    for label, ua, ub, rho2 in CASES:
        tab = edge_table(np.array([ua]), np.array([ub]), np.array([rho2]), SPEC)
        per_case.append(tab)
        ref, qerr = reference_table(ua, ub, rho2)
        worst = 0.0
        worst_km = None
        for m, kmax in SPEC.items():
            for k in range(kmax + 1):
                r = ref[m][k]
                v = mp.mpf(float(tab[m][0, k]))
                rel = float(abs(v - r) / abs(r)) if r != 0 else float(abs(v))
                if rel > worst:
                    worst, worst_km = rel, (k, m)
        # the reference must itself be trustworthy far below the gate
        ok_ref = qerr < 1e-20
        rep.check(f"{label}", worst, 1e-12,
                  f"worst (k,m)={worst_km}, mp.quad rel err {qerr:.1e}" + ("" if ok_ref else " REF?"))
        if not ok_ref:
            rep.check_bool(f"    reference quadrature converged for {label.split()[0]}", False)
        if worst > overall:
            overall, overall_lbl = worst, f"{label.split('(')[0].strip()} (k,m)={worst_km}"
    print(f"  worst over all regimes: {overall:.3e}  [{overall_lbl}]")
    return per_case


# ---------------------------------------------------------------------------
# (c) rho = 0 regime
# ---------------------------------------------------------------------------

def rho0_check(rep: Report) -> list[dict]:
    """``int u^k |u|^-m du`` vs mpmath, plus the +inf contract on the divergent
    principal values.  Returns the per-case clq tables."""
    mp.mp.dps = 40
    per_case = []
    for label, ua, ub in RHO0_CASES:
        tab = edge_table(np.array([ua]), np.array([ub]),
                         np.array([0.0]), SPEC)
        per_case.append(tab)
        straddles = ua <= 0.0 <= ub
        worst, worst_km, n_conv = 0.0, None, 0
        bad_inf = []
        for m, kmax in SPEC.items():
            for k in range(kmax + 1):
                q = k - m + 1
                v = float(tab[m][0, k])
                if straddles and q <= 0:
                    # |u|^(k-m) is not integrable at the foot: the ONLY correct
                    # answer is +inf.  A finite value here means the two
                    # antiderivative limits cancelled into a wrong number.
                    if not (np.isinf(v) and v > 0.0):
                        bad_inf.append((k, m, q, v))
                    continue
                f = lambda t, k=k, m=m: t ** k * abs(t) ** (-m)
                pts = ([mp.mpf(ua), mp.mpf(0), mp.mpf(ub)] if ua < 0 < ub
                       else [mp.mpf(ua), mp.mpf(ub)])
                ref = mp.quad(f, pts)
                rel = (float(abs(mp.mpf(v) - ref) / abs(ref)) if ref != 0
                       else float(abs(v)))
                n_conv += 1
                if rel > worst:
                    worst, worst_km = rel, (k, m)
        rep.check(f"{label}: {n_conv} convergent entries vs mpmath", worst, 1e-12,
                  f"worst (k,m)={worst_km}")
        if straddles:
            n_div = sum(1 for m, kmax in SPEC.items() for k in range(kmax + 1)
                        if k - m + 1 <= 0)
            rep.check_bool(f"{label}: {n_div} divergent (q <= 0) entries are +inf",
                           not bad_inf,
                           "(all +inf)" if not bad_inf else f"finite: {bad_inf}")
    # a non-straddling rho = 0 edge must have NO infinities at all
    tab = edge_table(np.array([0.3]), np.array([1.4]), np.array([0.0]), SPEC)
    rep.check_bool("rho=0 away from the foot: every entry finite",
                   all(bool(np.all(np.isfinite(tab[m]))) for m in SPEC))
    return per_case


# ---------------------------------------------------------------------------
# (d) batch == per-case
# ---------------------------------------------------------------------------

def batch_check(rep: Report, per_case: list[dict]) -> None:
    ua = np.array([c[1] for c in CASES])
    ub = np.array([c[2] for c in CASES])
    rho2 = np.array([c[3] for c in CASES])
    batch = edge_table(ua, ub, rho2, SPEC)
    worst = 0.0
    finite = True
    for m in SPEC:
        stacked = np.vstack([t[m] for t in per_case])
        finite &= bool(np.all(np.isfinite(batch[m])))
        worst = max(worst, float(np.max(np.abs(batch[m] - stacked))))
    rep.check_bool("batch edge_table == per-case (bitwise)", worst == 0.0 and finite,
                   f"(max |diff| = {worst:.1e}, all finite = {finite})")
    # regime masks are per element: a permuted batch must agree too
    perm = np.array([7, 3, 11, 0, 12, 5, 9, 1, 8, 4, 10, 2, 6])
    batch_p = edge_table(ua[perm], ub[perm], rho2[perm], SPEC)
    worst_p = max(float(np.max(np.abs(batch_p[m] - batch[m][perm]))) for m in SPEC)
    rep.check_bool("permuted batch == batch[perm] (bitwise)", worst_p == 0.0,
                   f"(max |diff| = {worst_p:.1e})")


def mixed_batch_check(rep: Report, per_case: list[dict], per_case0: list[dict]) -> None:
    """rho = 0 and rho > 0 rows in ONE call reproduce the per-case tables bit
    for bit: the rho = 0 branch is selected by the exact test ``rho2 == 0.0``,
    per element, and must not disturb its neighbours (or be disturbed by the
    small-|u| / large-|u| series masks, which skip the rho = 0 rows)."""
    ua = np.array([c[1] for c in CASES] + [c[1] for c in RHO0_CASES])
    ub = np.array([c[2] for c in CASES] + [c[2] for c in RHO0_CASES])
    rho2 = np.array([c[3] for c in CASES] + [0.0] * len(RHO0_CASES))
    batch = edge_table(ua, ub, rho2, SPEC)
    stacked = {m: np.vstack([t[m] for t in per_case + per_case0]) for m in SPEC}
    worst, same_inf = 0.0, True
    for m in SPEC:
        b, s = batch[m], stacked[m]
        same_inf &= bool(np.array_equal(np.isinf(b), np.isinf(s)))
        fin = np.isfinite(s)
        worst = max(worst, float(np.max(np.abs(b[fin] - s[fin]))))
    rep.check_bool("mixed rho=0 / rho>0 batch == per-case (bitwise, +inf pattern too)",
                   worst == 0.0 and same_inf,
                   f"(max |diff| = {worst:.1e}, inf pattern matches = {same_inf})")


def main():
    rep = Report("1-D edge primitives: antiderivatives and edge_table")
    symbolic_check(rep)
    per_case = numeric_check(rep)
    per_case0 = rho0_check(rep)
    batch_check(rep, per_case)
    mixed_batch_check(rep, per_case, per_case0)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
