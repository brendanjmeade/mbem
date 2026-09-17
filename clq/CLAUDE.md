# CLAUDE.md

Guidance for Claude Code when working in `clq/`.

## What this is

A self-contained, git-less Python sub-project of `moss-org`: **closed-form
mollified dislocation kernels on one flat triangle for constant, linear and
quadratic slip** (Lagrange P0/P1/P2), full space, Cortez regularisation
`R = sqrt(r^2 + eps^2)`.  It generalises the constant-slip result of
`../msd/mollified_kernel/analytical_kernels.py` (frozen oracle there) and is
described in `docs/derivation.md`.  One public API for all orders (`clq.displacement`,
`clq.stress`, `clq.eigenstress`, `clq.influence`); slip order is inferred from the
number of nodal rows (1 / 3 / 6).

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
* `eps` is a scalar >= 0; `eps = 0` only for observers off the plane (raises
  otherwise, on every `far_field` path).  No `1e-300` / `1e-60` guards
  anywhere: `rho^2 >= h^2 >= eps^2 > 0`; edge primitives are written in
  scale-free form so any absolute length scale works.
* `clq.stress` returns ELASTIC stress (total minus the exact eigenstress) by
  default -- the tree-wide policy in `../EIGENSTRESS_AUDIT.md`.  Use
  `subtract_eigenstress=False` only for kernel diagnostics and say so in the
  figure.
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
  if you add a primitive, add its regime to `verify_primitives.py`.
* When adding a kernel or a code path, add a `verify/` script that gates it
  against quadrature or an identity and prints PASS/FAIL.
* Figures follow the house matplotlib style (`examples/_paper_style.py`, the
  `matplotlib-figure-style` skill): sparse edge ticks, no grid, `RdBu_r`
  symmetric limits, outer labels only, panel letters top-right.
