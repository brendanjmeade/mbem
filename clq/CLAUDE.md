# CLAUDE.md

Guidance for Claude Code when working in `clq/`.

## What this is

A self-contained, git-less Python sub-project of `moss-org`: **closed-form
mollified kernels on one flat triangle for constant, linear and quadratic
nodal density** (Lagrange P0/P1/P2), full space, Cortez regularisation
`R = sqrt(r^2 + eps^2)`, for two source types:

* a **dislocation** (slip) source -- `clq.displacement`, `clq.stress`,
  `clq.eigenstress` (`want` keys `U`, `H`, `E`);
* a **force** source, the Kelvin single layer, i.e. a force per unit AREA on
  the triangle -- `clq.force_displacement`, `clq.force_stress` (`want` keys
  `G`, `S`).

It generalises the constant-density results of
`../msd/mollified_kernel/analytical_kernels.py` (frozen oracles there:
`analytical_dd_displacement`, `analytical_stress_kernel`,
`analytical_kelvin_G`, `analytical_kelvin_stress`) and is described in
`docs/derivation.md`.  One public API for all orders and both source types
(`clq.influence`); the order is inferred from the number of nodal rows
(1 / 3 / 6).

## Running

No build, no installer, no pytest.  Run from the clq root with any Python that
has numpy (plus matplotlib for `examples/`, and sympy + mpmath for three gates:
`verify_primitives.py`, `verify_pointwise.py`, `verify_regressions.py`).  The
system `python3` usually lacks numpy; on this machine
`/Users/meade/micromamba/bin/python` has everything.  `verify/run_all.py`
launches each gate with the interpreter that runs it (`sys.executable`), so
nothing in the code assumes a particular path.

```bash
PY=/Users/meade/micromamba/bin/python     # this machine; any numpy (+sympy, mpmath) Python works
$PY verify/run_all.py           # every verify_*.py prints one final PASS/FAIL line
$PY examples/demo_quickstart.py
$PY examples/demo_onfault_stress.py    # figures go to the clq root as fig_*.png/.pdf
```

Scripts insert the clq root on `sys.path` themselves (`verify/_common.py`,
`examples/_fields.py`), so launch them from the root, not from inside the
folder.  `verify/` gates load `../msd` and `../moss` oracle modules by file
path; do not import those packages any other way.

## Conventions (read before editing)

* `tri` is a (3,3) vertex array; the normal is `(v2-v1) x (v3-v1)` normalised.
  There is deliberately NO separate normal argument.
* Slip sign: `Delta u = u(+nhat side) - u(-nhat side)`.
* `slip` is (K,3) nodal Cartesian vectors; nodes from `clq.nodes(tri, order)`
  in the fixed order vertices, then edge midpoints 12, 23, 31.
* `eps` is a scalar >= 0.  `eps = 0` on the element plane is allowed for the
  force displacement kernel ALONE (`want=("G",)`, and the public
  `clq.force_displacement`): the single layer is weakly singular there.  Every
  other kernel -- and any mixed `want`, including `("G","S")` -- still raises
  on every `far_field` path, because their `I_5`, `I_7` genuinely diverge.
  The permission is derived from `want` inside `weighted_tables`; there is no
  user-facing flag, and `MomentTable` refuses `max(deg) >= 5` on such rows.
  No `1e-300` / `1e-60` guards anywhere: every branch is an exact test
  (`rho2 == 0.0`, `h2` snapped to 0) or a structural degree floor
  (`moments.h0_floor`), and edge primitives stay scale-free.  `0.0 * nan` is
  the live hazard in that path: sub-floor table slots are NaN on purpose, and
  `lift` masks them rather than multiplying by `z^c = 0`.
* `clq.stress` returns ELASTIC stress (total minus the exact eigenstress) by
  default -- the tree-wide policy in `../EIGENSTRESS_AUDIT.md`.  Use
  `subtract_eigenstress=False` only for kernel diagnostics and say so in the
  figure.
* The FORCE element has no eigenstress: a mollified body force is a genuine
  body force, not an eigenstrain, so `clq.force_stress` already returns the
  elastic stress and deliberately takes NO `subtract_eigenstress` argument.
  Do not "fix" this to match `clq.stress`.
* `force` is (K,3) nodal force per unit AREA; `u_i = sum_k G[n,k,i,j] f[k,j]`
  and equilibrium is `div sigma + f phi_eps = 0`, so a closed surface around
  the element carries `int sigma.nhat dS = -int f dS`.  (The body-force drafts
  in `../body_forces_bem` use the OPPOSITE sign, `div sigma = f`.)  In a BEM
  the force block is msd's `G`/`U` slot with scale `-sigma(R,p)` and NO `1/2 I`
  free term -- the single layer is continuous; only its traction jumps.  Note
  clq's `want` key `"H"` is the slip -> stress kernel, NOT msd's `BlockTerm`
  `"H"` (which is the slip -> displacement T-kernel).
* The slip -> displacement contraction uses the traction-operator pairing
  `U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]`
  (moss commit f721a6a).  Do not "fix" it back to the msd pre-fix form; the
  gates `verify_hooke_consistency.py` (rejects an inconsistent U/H pairing)
  and `verify_jump.py` plus msd's closed-surface closure gate (absolute
  pairing) exist to catch that.
* Numeric constants live in `clq/defaults.py`, not inline.
* Everything is numpy-vectorised over observation points, one triangle per
  call; the observation axis is the only batch axis.  Keep it that way so a
  numba port stays mechanical.
* Edge primitives must stay cancellation-free (`clq/primitives.py` docstring);
  if you add a primitive, add its regime to `verify_primitives.py`.  The force
  kernel needs `(m,k)` pairs the dislocation kernels never ask for -- the P2
  spec is `{5:5, 3:4, 1:3, -1:1}` -- plus the `rho2 == 0` regime (elementary
  `int u^k |u|^-m du`), which is hit bit-exactly at every vertex and edge
  midpoint of an axis-aligned triangle.
* `S` (force -> stress) reuses the very same `G1` block as `U`; keep that
  expression and the in-place `G1 *= C1` byte-identical so `U` stays bitwise
  unchanged (`verify_regressions.py` gates that `want=("U",)` and
  `("U","S")` agree bitwise -- note this is want-INDEPENDENCE, not
  old-vs-new: the 2026-09-17 force-element commit does move `U` by ~1e-16
  relative, from regrouping `dp ** pw` into `_dp_times` in `moments.py`, and
  no gate covers that), and note `("S",)` closes to
  exactly the degrees of `("U",)`.
* When adding a kernel or a code path, add a `verify/` script that gates it
  against quadrature or an identity and prints PASS/FAIL.
* Figures follow the house matplotlib style (`examples/_paper_style.py`, the
  `matplotlib-figure-style` skill): sparse edge ticks, no grid, `RdBu_r`
  symmetric limits, outer labels only, panel letters top-right.
