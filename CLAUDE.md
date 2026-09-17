# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`moss-org` is an **umbrella folder, not a single repository**. It holds successive
generations of one research program — a mollified boundary element method (BEM)
and mollified elastic dislocation theory (MEDT) for earthquake mechanics, built on
the Cortez regularization `r → √(r² + ε²)` of the singular Kelvin/Somigliana
kernels — plus the paper and its public reproducibility package. The sub-projects
are largely independent codebases; work happens *inside* one of them, not at this
level.

**`clq/`, `moss/` and `msd/` have their own `CLAUDE.md`, which is authoritative
for work inside them. Read it before editing anything there.** `medt_paper/` is
documented by its `README.md`. This file only maps the territory.

## Sub-project map (as of 2026-09-17)

| dir | git? | what it is |
|---|---|---|
| `clq/` | no | **Newest (Sept 2026).** Closed-form mollified dislocation kernels on one flat triangle for **constant, linear and quadratic slip** (Lagrange P0/P1/P2, one API), exact finite-triangle eigenstress, PASS/FAIL `verify/`, equilateral-triangle figures. Generalises the constant-slip result of `msd`; see `clq/CLAUDE.md` and `clq/docs/derivation.md`. Not yet used by the paper. |
| `medt_paper/` | yes (`github.com/brendanjmeade/medt_paper`, public, Zenodo DOI) | Reproducibility package for the paper: one script per figure, cached heavy results, and copies of the library code it needs (`mollified_kernel/` from `moss`, `mhf/` from the former top-level half-space package, `topo_inclusion/` from `moss2`). The manuscript source is **not** in this repo. Figures are written to `medt_paper/figures/`; see its `README.md`. |
| `msd/` | yes | Clean, self-contained mollified BEM for 3-D full-space elasticity (July 2026): frozen legacy oracles + the rebuilt `mbem/` solver stack, eigenstress subtraction for on-fault stress, `verify/` PASS/FAIL scripts, `examples/`. No half-space, no viscoelasticity, no LaTeX. See `msd/CLAUDE.md`. |
| `moss/` | yes | The original research repo: spherical whole-Earth + local-box BEM, viscoelastic Laplace extension, half-space Mindlin work (`mh/`, `mhf/`, `mh_deploy/`, `MINDLIN_STATUS.md`), and **the paper**: `moss/manuscript/main.tex` (the current MEDT draft; `manuscript.tex` there is the older long draft). See `moss/CLAUDE.md` and `moss/manuscript/README.md`. |

Loose files at this level:

- `EIGENSTRESS_AUDIT.md` — the tree-wide policy (2026-07-16) that every stress
  presented as elastic must have the anelastic eigenstress `C:ε*` of the smeared
  slip subtracted; cited by `clq/`.
- `CLAUDE.md` — this file.

**Archived, not present here.** The full earlier tree is in
`~/Desktop/moss-org.zip` (created 2026-09-17): `moss2/` (region-graph solver
stack, pytest harness, viscoelastic Laplace pipeline, the Figure 10 benchmark
scripts and caches), the former top-level `mhf/` (+ `mhf.pre-eigenstress-backup/`),
`moss_manuscript_wip/`, `moss_figures/`, and the Sept 11 submission packages
`medt_arxiv/` (arXiv) and `medt_gji/` (GJI class, cover letter, checklist) built
from an earlier state of `main.tex`. Paths in older docs that point at these
directories (e.g. `../moss2`, `../mhf`) are dead in this folder.

## Working here

- **Pick the right sub-project first.** Paper text or figures →
  `moss/manuscript/main.tex` + `moss/manuscript/scripts/`, then carry any change
  to the public copy in `medt_paper/`. Higher-order (linear/quadratic) slip on a
  triangle → `clq/`. New full-space solver/kernel work → `msd/`. Sphere/box/
  viscoelastic work → `moss/` (the `moss2` successor is only in the zip).
  Half-space Mindlin → `moss/mhf/` (research copy, has the Route-1 analytic
  correction) or `medt_paper/mhf/` (older fork used for Figure 9).
- **Git works only inside `moss/`, `msd/` and `medt_paper/`** — each is its own
  repo; this top level and `clq/` are not under version control. `medt_paper` is
  public: do not push without being asked.
- **Library code is duplicated across repos** (`mollified_kernel/` in `moss`,
  `msd`, `medt_paper`; `mhf/` in `moss` and `medt_paper`; the `mbem` stack in
  `msd` and `medt_paper/topo_inclusion`). There is no shared package, so a kernel
  fix must be applied to every copy by hand.
- **Slip → displacement kernel pairing.** The correct form is
  `U_ij = -[mu n_m dG_ij/dx_m + lam n_j dG_im/dx_m + mu n_m dG_im/dx_j]`
  (slip and normal in the first index pair of C). A swapped form (λ and μ
  exchanged on the first two terms) survived in several copies until
  2026-09-04 (`msd`) and 2026-09-17 (`moss`, `mh_deploy`, `medt_paper`). It is
  invisible at ν = 1/4 (λ = μ), which almost every example uses, so any new
  kernel code must be checked at ν ≠ 1/4 (gates: `msd/verify/verify_dd_pairing.py`,
  `moss/mollified_kernel/verify_dd_pairing.py`). `moss` viscoelastic results
  made before 2026-09-17 used the swapped batch kernel (λ̃(s) ≠ μ̃(s)).
- **Python.** No installer anywhere: run scripts from the sub-project root with a
  Python that has numpy/scipy/matplotlib (+ numba, jax, sympy, mpmath, triangle
  for some parts). On this machine that is `/Users/meade/micromamba/bin/python`;
  the `/Users/meade/miniforge3/...` path in older notes does not exist here.
  No LaTeX is installed on this machine.
- Shared conventions: units km / GPa / years (slip 0.001 km = 1 m), NumPy
  `(N, 3)` vectors, numba JIT warmup on first call.
- `msd` (and the archived `moss2`) share a core architectural rule: **legacy flat
  modules are frozen validation oracles** — never "improve" them; new code lives
  in `mbem/` and is gated by entrywise parity against the legacy output.
- Correctness gates differ per project: `msd` and `clq` use `verify/*.py`
  scripts that print PASS/FAIL (`clq/verify/run_all.py` runs all of them),
  `moss` has PASS/FAIL gates in `mollified_kernel/verify_*.py` but otherwise
  validation scripts inspected visually, and `mhf` has `python -m mhf.validate`
  (needs `cutde`).
- Publication figures follow a house matplotlib style (`_paper_style.py` in each
  figure-script folder; the `matplotlib-figure-style` skill where available).
