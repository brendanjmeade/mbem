# CLAUDE.md

Guidance for Claude Code in `moss-org`: an umbrella directory, ONE git repo,
holding the mollified boundary element method (BEM) and mollified elastic
dislocation theory (MEDT) program built on the Cortez regularization
`r -> sqrt(r^2 + eps^2)` of the Kelvin/Somigliana kernels, its paper, and the
paper's public reproducibility package. Work happens inside one sub-project;
`msd/CLAUDE.md`, `moss/CLAUDE.md` and `clq/CLAUDE.md` are authoritative there.

## Map

| dir | what it is |
|---|---|
| `msd/` | **The trunk.** Self-contained full-space mollified BEM: frozen legacy oracles plus the live `mbem/` stack (region-graph model, P0/P1/P2 nodal patches and faults, dense and block-compressed backends, elastic-stress readout with the exact eigenstress removed), PASS/FAIL `verify/` gates (`verify/run_all.py`), `examples/`. New solver or kernel work goes here. |
| `clq/` | **Frozen oracle.** Closed-form mollified kernels on one triangle for P0/P1/P2 nodal density (slip and force sources), numpy. `msd/verify/verify_nodal_kernels.py` gates the trunk's numba kernels against it at 1e-12; `clq/verify/run_all.py` must stay green. No development. |
| `moss/` | The paper (`manuscript/main.tex`, its 8 figure scripts), the frozen `mollified_kernel/` oracle (5 modules, 7 gates; loaded by msd and clq gates by path), and `mhf/`, the exact-BC mollified Mindlin half-space package. |
| `medt_paper/` | Public reproducibility package (mirrors `github.com/brendanjmeade/medt_paper`, Zenodo DOI): one script per figure, caches, vendored library copies. Self-contained. Tag `medt_paper-vendored-2026-09-17` pins the tree its code came from; its README has a Provenance section. **Never push it without being asked**; publishing is a manual export. |
| `ddbem/` | Closed. `FINDINGS.md` only: the P0/P1/P2 displacement-discontinuity convergence study and free-term facts the trunk's higher-order patches rest on. Code in git history. |
| `fbem/` | Closed. `FINDINGS.md` only: why the force-element (equivalent body force) BEM was dropped. Do not rebuild it without reading it. |

Loose files: `BACKLOG.md` (the one status document: standing rules, open items,
consolidation), this file. The full earlier tree (`moss2`, the old top-level
`mhf`, submission packages) is in `~/Desktop/moss-org.zip`.

## Rules

- **Git.** One local repo, no remote; commits span sub-projects, so scope them
  by path. The user commits on request; never push `medt_paper`.
- **Python.** No installer. `/Users/meade/micromamba/bin/python` on this
  machine; run scripts from the sub-project root. No LaTeX here.
- **Oracles are frozen.** `msd`'s legacy flat modules, `moss/mollified_kernel`
  and `clq` are never "improved"; new code is gated against them by entrywise
  parity. Every gate prints `PASS:`/`FAIL:` and exits 1 on FAIL.
- **Kernel copies.** `mollified_kernel/` lives in `msd`, `moss` and
  `medt_paper`; `mhf/` in `moss` and `medt_paper`. A kernel fix goes to every
  live copy by hand, checked at nu != 1/4 (the lam/mu pairing swap of the
  slip -> displacement kernel is invisible at nu = 1/4).
- **Conventions.** Units km / GPa / years (slip 0.001 km = 1 m), `(N, 3)`
  arrays. Fault slip is the Burgers vector `b = u(+n) - u(-n)` in `msd`, `clq`
  and `cutde` alike (`msd/mbem/model/core.py` states it once; `mbem.selfcheck`
  pins it). Stress presented as elastic subtracts the eigenstress `C:eps*` of
  every mollified double layer (`BACKLOG.md`).
- **Lean.** Docstrings state the rule and the reason; measurements and history
  go in commit messages; no probe scripts in the tree; extend a gate before
  adding one; no new top-level documents.
- Figures follow the house matplotlib style (`_paper_style.py` beside each
  figure-script folder; the `matplotlib-figure-style` skill where available).
