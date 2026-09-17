# clq — constant, linear and quadratic slip on an arbitrary triangle, in closed form

`clq` computes the displacement, the total and the elastic stress of a
**mollified** (Cortez, `R = sqrt(r^2 + eps^2)`) dislocation on one flat
triangle in a 3-D elastic full space, for slip that varies as a polynomial of
degree 0, 1 or 2 over the triangle (Lagrange P0/P1/P2 nodal shape functions),
**analytically**.  It generalises the constant-slip closed form of the
`msd`/`moss` research codes (manuscript appendix "Closed-form mollified
DD-triangle integration") by carrying the same edge recurrence two orders
higher; the derivation is in `docs/derivation.md`.

```python
import numpy as np, clq

tri = clq.equilateral(L=1.0)                 # (3,3) vertices; nhat = (v2-v1)x(v3-v1)/|.|
mu, nu, eps = 1.0, 0.25, 0.05
obs = np.array([[0.1, 0.05, 0.0], [0.3, -0.2, 0.2]])

s0 = np.array([[1.0, 0.0, 0.0]])                       # constant   (1 node)
s1 = np.array([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])     # linear     (3 vertex nodes)
s2 = np.array([[0, 0, 0]] * 3 + [[1.0, 0, 0]] * 3)     # quadratic  (3 vertices + 3 midpoints)
for slip in (s0, s1, s2):                              # the SAME calls for every order
    u   = clq.displacement(obs, tri, slip, mu, nu, eps)         # (N,3)
    sig = clq.stress(obs, tri, slip, mu, nu, eps)               # (N,3,3)  elastic = total - C:eps*
    tot = clq.stress(obs, tri, slip, mu, nu, eps, subtract_eigenstress=False)
inf = clq.influence(obs, tri, mu, nu, eps, order=2)   # nodal tensors U (N,K,3,3), H (N,K,3,3,3), E (N,K)
```

The slip order is inferred from the number of nodal rows (1, 3, 6, 10, ...:
any Lagrange order runs, P0-P2 are the documented and figure-tested ones;
nodes from `clq.nodes(tri, order)`).  A (3,) slip vector means constant slip.  Any consistent units (`mu` sets the stress
unit); the examples use L = mu = s = 1.

## What is inside

| module | content |
|---|---|
| `clq/primitives.py` | edge antiderivatives `J_m, K_m, int u^k/R^m du` for every odd `m` (incl. the new `J_{-1} = int R du`), cancellation-free differences in every regime, solid angle |
| `clq/moments.py` | in-plane moment table `M_n^{(a,b)}` to arbitrary order by the divergence-theorem recurrence; per-node weighted tables; Gauss far-field producer (hybrid beyond `D_STAR * L`) |
| `clq/shape.py` | P0/P1/P2 (any order) Lagrange nodes and shape polynomials, interpolation, on-triangle grids |
| `clq/kernels.py` | lift to tensor moments and the contractions: slip -> displacement, slip -> total stress, eigenstress weight |
| `clq/api.py` | `influence`, `displacement`, `stress`, `eigenstress`, `traction` |
| `clq/pointwise.py`, `clq/quadrature.py` | point kernels and Gauss quadrature used only as oracles |
| `verify/` | PASS/FAIL gates (`python verify/run_all.py`) |
| `examples/` | figures for one equilateral triangle and a quickstart |
| `docs/derivation.md` | the closed-form statement and numerics |

Stress readout policy (tree-wide, see `../EIGENSTRESS_AUDIT.md`): the kernel
returns the TOTAL stress `C:(eps_el + eps*)` of the smeared slip; on the fault
it is dominated by the eigenstress `C:eps*` ~ (3/4) mu s/eps.  `clq.stress`
subtracts the **exact** finite-triangle eigenstress
`E_k = (15 eps^4 / 8 pi) int_T N_k / R^7 dS` by default and returns the
elastic stress, which at interior points of the element stays finite and
converges as `eps -> 0` (on the element boundary, where the nodal slip does
not vanish, the elastic stress has the classical edge/vertex singularity that
emerges as `eps -> 0`).

## Running

Use any Python 3 with numpy (`requirements.txt`): matplotlib for the
examples, and sympy + mpmath for three of the gates (`verify_primitives.py`,
`verify_pointwise.py`, `verify_regressions.py`; they fail on import without
them).  No install step; run everything from the clq root.
`verify/run_all.py` starts each gate with the same interpreter it was
launched with (`sys.executable`).

```bash
PY=/Users/meade/micromamba/bin/python                   # example (this machine); any such Python works
cd /Users/meade/Desktop/moss-org/clq
$PY verify/run_all.py                                   # all gates
$PY examples/demo_quickstart.py
$PY examples/demo_onfault_displacement.py               # fig_onfault_displacement
$PY examples/demo_onfault_stress.py                     # fig_onfault_stress_elastic, fig_onfault_stress_total, fig_nearfault_stress_elastic
$PY examples/demo_eps_finiteness.py                     # fig_eps_finiteness
$PY examples/demo_jump_profiles.py                      # fig_jump_profiles
$PY examples/demo_face_pressure.py                      # fig_slip_contours, fig_face_pressure (mean stress on the fault faces)
```

Figures are written to the clq root as `.png` and `.pdf`.

| gate | what it checks |
|---|---|
| `verify_primitives.py` | primitives vs sympy differentiation; edge differences vs 40-digit mpmath in every regime |
| `verify_pointwise.py` | point kernels vs sympy; regularised Navier residual with the blob; index pairing |
| `verify_moments.py` | full quadratic-slip moment table vs 200x200 Gauss; two-route identity |
| `verify_order0_parity.py` | constant slip == the `msd` (post-fix) and `moss` scalar oracles at nu = 1/4 and 0.3 to 1e-12; pre-fix form rejected |
| `verify_regressions.py` | review regressions: extreme-scale edge primitives, orders 5-6 vs quadrature, eps = 0 guard on every path, translation invariance, eps >> L crossover |
| `verify_nodal_vs_quadrature.py` | P0/P1/P2 nodal tensors vs quadrature, off- and on-plane |
| `verify_subdivision.py` | P1/P2 vs the sum of N^2 constant-slip sub-triangles (independent oracle) |
| `verify_identities.py` | partition of unity, P1 in P2, relabelling, orientation, rigid covariance, scaling |
| `verify_hooke_consistency.py` | FD gradient of the displacement -> Hooke == total stress at nu = 0.3 |
| `verify_jump.py` | displacement jump recovers the slip at nu in {0.25, 0.3, 0.35, 0.45} |
| `verify_eigenstress.py` | exact eigenstress vs quadrature, infinite-plane limit, finiteness as eps -> 0 |
| `verify_far_field.py` | closed form vs quadrature vs distance; hybrid crossover |
| `verify_api.py` | order inference, errors, shapes, BEM block layout |

## Notes

* Correctness finding (2026-09-04): the slip -> displacement contraction in
  `msd` (scalar, batch and numba T-kernel basis) had lambda and mu swapped
  relative to the traction-operator pairing used here (invisible at nu = 1/4).
  `clq` implements the correct form (`docs/derivation.md`, eq. 1.1) and gates
  it by Hooke consistency and the displacement-jump test; `msd` is fixed in
  the same session (`msd/verify/verify_dd_pairing.py`).
* Far field: **use the default `far_field="hybrid"`.** The closed-form
  moments lose digits quickly as the observer's effective distance
  `R = sqrt(D^2 + eps^2)` grows (D from the centroid, L = longest edge).  The
  cause is a closure-sum and shape-coefficient cancellation that is built into
  the divergence theorem.  The loss is worst for quadratic slip and for
  observers in the triangle's plane.  It grows roughly like (R/L)^4-5 and is
  larger for smaller eps.  Measured worst relative error of the U/H tensors
  over P0-P2 and four directions, against Gauss quadrature
  (eps = 0.05 L, nu = 0.3):

  | R/L | 2 | 5 | 10 | 20 | 50 | 100 | 200 | 500 | 1000 |
  |---|---|---|---|---|---|---|---|---|---|
  | `analytic`, equilateral + test triangle | 1e-12 | 2e-10 | 2e-9 | 4e-8 | 7e-6 | 2e-4 | 2e-3 | 6e-1 | 9 |
  | `analytic`, thin (h/L = 0.1) | 3e-10 | 1e-8 | 3e-7 | 8e-6 | 5e-4 | 3e-3 | 1 | 2e2 | 5e3 |
  | `analytic`, sliver (h/L = 0.02) | 3e-8 | 3e-6 | 4e-5 | 2e-3 | 4e-1 | 2e1 | 1e2 | 1e4 | 3e5 |
  | `hybrid` (default), any of these | = analytic | = analytic | = analytic below 10 L | 2e-14 | 2e-14 | 2e-14 | 2e-14 | 2e-14 | 2e-14 |

  (The R/L = 10 column is the worse of 9.5 L and 10.5 L.)  P2 is always
  the worst order and P1 is next.  For P0 alone the well-shaped loss stays
  below 1e-6 even at 1000 L (6e-5 for the sliver).  Smaller eps makes things
  worse: at eps = 0.01 L the well-shaped row is 2-20x larger (1e-8 at 10 L,
  6e-4 at 100 L, 55 at 1000 L).  **`far_field="analytic"` is therefore not
  usable beyond about 20 L**; it is for diagnostics and near-field gates
  only.

  The hybrid default is exact to rounding beyond `R > D_STAR L`
  (`D_STAR = 10`), because there it fills the same tables from a Gauss
  product rule (12x12 points, 8x8 beyond 40 L).  At `R <= D_STAR L` it *is*
  the closed form, so just inside the switch it carries the analytic error
  there.  Measured at R = 9.5 L that is about 1e-9 for well-shaped triangles
  (1e-8 at eps = 0.01 L).  `D_STAR` is set relative to the longest edge and
  ignores aspect ratio, so thin triangles still lose digits under the hybrid:
  3e-7 at h/L = 0.1 and 1e-5 at h/L = 0.02 (1e-6 and 2.5e-4 at
  eps = 0.01 L).  Keep aspect ratios moderate, as BEM meshes do.
  `verify_far_field.py` prints the sweep for the test triangle.
* No numba yet: everything is numpy-vectorised over observation points (one
  triangle at a time); the observation axis is the only batch axis so a
  numba port is mechanical.
