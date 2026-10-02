# clq — rules for the frozen oracle

Conventions for working on `src/clq`. (This file was `clq/CLAUDE.md` before the
tree was packaged; it is kept under its own name because the rules below are
specific to the oracle, while the repo-wide ones live in `CLAUDE.md`.)

**FROZEN.** `clq` is the oracle for the general-order kernels of the trunk,
`src/mbem/kernels/tri_nodal.py`, which
`tests/gates/mbem/verify_nodal_kernels.py` gates against it at 1e-12. Do not
change code here (`tests/gates/clq/verify_baseline_bitwise.py` pins `U`/`H`/`E`
bitwise); new kernel work goes in `src/mbem`. Its own 16 gates
(`python tests/run_all.py --suite clq`) must stay green.

## What this is

**Closed-form mollified kernels on one flat triangle for constant, linear and
quadratic nodal density** (Lagrange P0/P1/P2), full space, Cortez regularisation
`R = sqrt(r^2 + eps^2)`, for two source types:

* a **dislocation** (slip) source -- `clq.displacement`, `clq.stress`,
  `clq.eigenstress` (`want` keys `U`, `H`, `E`);
* a **force** source, the Kelvin single layer, i.e. a force per unit AREA on
  the triangle -- `clq.force_displacement`, `clq.force_stress` (`want` keys
  `G`, `S`).

It generalises the constant-density results of
`src/mollified_kernel/analytical_kernels.py` (frozen oracles there:
`analytical_dd_displacement`, `analytical_stress_kernel`,
`analytical_kelvin_G`, `analytical_kelvin_stress`) and is described in
`docs/clq-derivation.md`.  One public API for all orders and both source types
(`clq.influence`); the order is inferred from the number of nodal rows
(1 / 3 / 6).

## Running

`pip install -e .` once, then `import clq` from anywhere -- the gates and the
demos are run by path and compute no paths of their own.  Three gates need
sympy and mpmath (`verify_primitives.py`, `verify_pointwise.py`,
`verify_regressions.py`), which come with `pip install -e '.[test]'`;
`run_all.py` launches each gate with the interpreter that runs it
(`sys.executable`), so nothing assumes a particular Python.

```bash
python tests/run_all.py --suite clq      # 16 gates, one PASS/FAIL line each
python tests/gates/clq/verify_api.py     # or one at a time
python studies/clq/demo_quickstart.py
python studies/clq/demo_onfault_stress.py   # figures land beside the script
```

Two `moss_kernel` gates still place an oracle directory on `sys.path`, because
two oracle packages ship an identically-named `analytical_kernels` and the gate
has to pin which one it is comparing.  That is the only reason anything here
touches `sys.path`; nothing in `src/` does.

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
  default -- the tree-wide policy in `BACKLOG.md`.  Use
  `subtract_eigenstress=False` only for kernel diagnostics and say so in the
  figure.
* The FORCE element has no eigenstress: a mollified body force is a genuine
  body force, not an eigenstrain, so `clq.force_stress` already returns the
  elastic stress and deliberately takes NO `subtract_eigenstress` argument.
  Do not "fix" this to match `clq.stress`.
* `force` is (K,3) nodal force per unit AREA; `u_i = sum_k G[n,k,i,j] f[k,j]`
  and equilibrium is `div sigma + f phi_eps = 0`, so a closed surface around
  the element carries `int sigma.nhat dS = -int f dS`.  (The equivalent-body-
  force drafts, removed from the tree, used the OPPOSITE sign,
  `div sigma = f`.)  In a BEM the force block is mbem's `G`/`U` slot with
  scale `-sigma(R,p)` and NO `1/2 I` free term -- the single layer is
  continuous; only its traction jumps.  Note
  clq's `want` key `"H"` is the slip -> stress kernel, NOT mbem's `BlockTerm`
  `"H"` (which is the slip -> displacement T-kernel).
* The slip -> displacement contraction uses the traction-operator pairing
  `U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]`
  (moss commit f721a6a).  Do not "fix" it back to the `mollified_kernel`
  pre-fix form; the gates `verify_hooke_consistency.py` (which rejects an
  inconsistent U/H pairing)
  and `verify_jump.py` plus mbem's closed-surface closure gate (absolute
  pairing) exist to catch that.
* Numeric constants live in `src/clq/defaults.py`, not inline.
* Everything is numpy-vectorised over observation points, one triangle per
  call; the observation axis is the only batch axis.  Keep it that way so a
  numba port stays mechanical.
* Edge primitives must stay cancellation-free (`src/clq/primitives.py`
  docstring); if you add a primitive, add its regime to
  `verify_primitives.py`.  The force
  kernel needs `(m,k)` pairs the dislocation kernels never ask for -- the P2
  spec is `{5:5, 3:4, 1:3, -1:1}` -- plus the `rho2 == 0` regime (elementary
  `int u^k |u|^-m du`), which is hit bit-exactly at every vertex and edge
  midpoint of an axis-aligned triangle.
* `S` (force -> stress) reuses the very same `G1` block as `U`; keep that
  expression and the in-place `G1 *= C1` byte-identical so `U` stays bitwise
  unchanged, and note `("S",)` closes to exactly the degrees of `("U",)`.
  TWO gates, which check different things: `verify_regressions.py` gates
  want-INDEPENDENCE (`want=("U",)` vs `("U","S")` inside the current code),
  and `verify_baseline_bitwise.py` gates OLD-VS-NEW, pinning `U`/`H`/`E`
  against 648 byte hashes taken from the pre-force-element tree (72c2840).
  Only the second can see a refactor of the shared moment machinery that
  moves both `want` variants equally -- which has already happened once:
  regrouping `dp ** pw` out of `coeff` in `moments.py` moved 40 of 48 arrays
  by up to 1.9e-15 and nothing caught it.  Keep the factor grouping in
  `MomentTable.__init__` as it is; if you must change the slip kernels
  deliberately, regenerate with `--regenerate` and say so in the commit.
* When adding a kernel or a code path, add a `tests/gates/clq/` script that
  gates it against quadrature or an identity and prints PASS/FAIL.
* Figures follow the house matplotlib style (`studies/clq/_paper_style.py`, the
  `matplotlib-figure-style` skill): sparse edge ticks, no grid, `RdBu_r`
  symmetric limits, outer labels only, panel letters top-right.
