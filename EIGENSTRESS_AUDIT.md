# Eigenstress (anelastic term) audit — moss-org sweep, 2026-07-16

## Policy (decided 2026-07-16)

A fault slip is an **anelastic (eigen-) strain**.  Stress read off a mollified
slip source is the TOTAL stress `σ_tot = C:(ε_el + ε*)`; on/near the fault
(within ~2ε) it is dominated by the eigenstress `C:ε*` of the smeared slip —
slip-sense shear peaking at `(3/4)·μs/ε`, diverging as ε → 0.  **Every output
presented as elastic / Coulomb / von-Mises stress must subtract it**
(`σ_el = σ_tot − C:ε*`, msd's policy, now adopted tree-wide).  The former
"ε is a physical fault-zone width" interpretation (unsubtracted total as
zone physics) is **deprecated and withdrawn** — including the manuscript's
on-fault positive-CFS-band claim, which was shown quantitatively to be the
eigenstress (see below).  Raw kernel-accuracy visualizations may stay
unsubtracted when clearly presented as kernel demos, not physical stress.

Reference implementation: `msd/anelastic.py` (`eigenstress_at_points`:
nearest-triangle assignment, edge-tapered point–triangle distance, Cortez
marginal `ρ = 0.75ε⁴/(d²+ε²)^2.5`, `λ·tr` term for tensile slip), vendored
into `mhf/anelastic.py`, `moss/mhf/anelastic.py`, `moss/anelastic.py`,
`moss/mh_deploy/anelastic.py`, `moss2/anelastic.py` — **keep in sync
manually**; the repos share no package.

## Sign rules (measured, not assumed)

- **Direct-kernel pipelines** (mhf/mh evaluators, showcase `U_dd@slip`):
  total = elastic + C:ε* → **subtract** `eigenstress_at_points`.
  Verified by validate gates (wiring identity 0 to 1e-12; on-fault raw ×4.4
  growth over ε 1→0.25 vs corrected ×1.01) and by the showcase probe
  (on-trace total +21.2 MPa vs eigenstress +22.3, analytic 22.5, at ε=1).
- **msd's BEM readout** keeps its own convention: the fault term enters as
  `−Sdd@slip`, so `evaluate_stress` *adds* the eigenstress back.  Do not
  copy signs across repos — probe each site.
- **Interpolated-gradient pipelines** (BEM surface displacement → griddata →
  np.gradient; moss fig09/fig12): the ε-scale eigenstress is **mesh-smeared**
  into the trace strip; probed against the mhf exact-BC elastic reference,
  the field beyond ~2 mesh lengths matches the *elastic* value (within ~10%
  interpolation bias) — pointwise subtraction there would over-correct with
  the wrong shape.  Correct treatment: **mask the trace strip** (2 mesh
  lengths) and document (msd `demo_fault_only` precedent).

## Fixes applied (2026-07-16)

| where | what |
|---|---|
| `mhf/` (top level, git-less; pre-edit snapshot at `mhf.pre-eigenstress-backup/`) | `anelastic.py` added; `eval_fields`/`eval_fields_fault_aware` subtract by default (`subtract_anelastic=True`; blend kept, subtraction once after it); scalar `field_hs_mindlin` stays TOTAL (kernel oracle, documented); demos gain `--total`; `demo_sixpanel_noeigen.py` removed, finiteness figure lives on as `demo_eps_sweep.py` → `fig_mhf_epssweep`; README interpretation section replaced by "Stress readout"; **validate group 8** gates it all (8/8 green) |
| `moss/mhf/` | identical retrofit, adapted to the `graded="analytic"` ROUTE-1 recursion (composes unchanged — it improves image-term accuracy, not the eigenstress); 8/8 green |
| `moss/mollified_fault_zone_showcase.py` (= `moss2/…`, byte-identical twins) | subtracts `σ*_xy` from the FD surface stress; keeps totals + eigenstress in v2 caches (`stress_v2_eps_*.npz`; v1 caches ignored); under-resolution mask for ε < 4·dh; stress panels now 4 rows (σ_xy elastic / σ_xx / CFS elastic / CFS total); `_paper_style` import fallback fixes the former moss2 ImportError |
| `moss/manuscript/` | new `fig13_cfs_eps_sweep` (total vs elastic CFS, 2×4); §`sec:eps-fault-zone` rewritten — claim **withdrawn** (Branch A: elastic on-trace band mean ≈ −0.9 MPa, ε-stable, at every ε; the positive band was `C:ε*` = 225/75/22.5/7.5 MPa vs a ±1 MPa scale, its "emergence" at ε≳1 km a rendering effect); "falsifiable prediction" paragraph removed; fig11 caption notes plateaus are `C:ε*` while the edge line singularity is elastic; fig09 mask widened 4→6 km + caption; fig12 caption updated to regenerated numbers (26.1/29.6 MPa, 13%/22% — the published 19.8/4.7/50%/70% did not reproduce) + TOTAL-field caveat; Makefile repaired (real script names, per-figure targets, fig01–fig13, `PY` overridable) |
| `moss/mh/` + `moss/mh_deploy/` | `_demo_field_common.py` evaluators subtract by default; CFS demo gains `--total`; deploy re-synced (+`anelastic.py`), README notes the update |
| `moss2/` | caches restored from git (no re-solve); vendored `anelastic.py`, `_paper_style.py`, fixed showcase; new gate `tests/test_eigenstress_showcase.py` (marginal analytics, sign, boundedness, mask semantics, ε-stability, import smoke) |
| `msd/` | **no changes** — already compliant (subtraction implemented, defaulted, gated); raw-kernel demos deliberately unsubtracted |

Displacement outputs everywhere are unaffected (the eigenstrain is a stress
question); all `*_onfault*` topo-inclusion figures in moss2 are
displacement-only ("onfault" is a mesh-geometry label) — no leak.

## Archive flags (report only — not edited, per scope decision)

- **`moss_manuscript_wip/`** — April–May 2026 manuscript snapshot, superseded
  by `moss/manuscript/`.  Its draft adopts the total-stress fault-zone
  framing ("more physically meaningful stresses", on-fault tractions) and
  its fig scripts do not subtract.  Do not quote from it; the live
  manuscript is authoritative.
- **`moss_figures/new_stress.tex`** (+ `Delta_mathbf_sig.pdf`) — presents
  `Δσ(s, μ) → Δσ(s, μ, ε)` as a feature; that ε-dependence **is** the
  eigenstress contamination signature.  Do not reuse in talks/papers without
  reframing as total-vs-elastic per the new §`sec:eps-fault-zone`.
- **`moss_figures/moment.tex`** — `M₀(ε) = M₀(1 − ε/(4D))` moment accounting:
  displacement-side, independent of the eigenstress question; still valid.
- **`moss_figures/fig_topo_inclusion_contour_onfault_cubic.png`** —
  byte-identical copy of a moss2 figure; displacement-only despite the name.
  Fine, but do not pair it with a stress narrative.
- **`moss_figures/anim_mhf_sixpanel_eps_hires.mp4`** — copy of the OLD
  total-stress animation (deprecated story); the regenerated elastic movie
  lives in `mhf/`.  Replace this copy if it is still being shown.
- **`mhf/moment_accounting.tex`** (both copies) — kept as-is (moment result
  valid), but lines voicing the fault-zone stress reading (~21–24, 117–128,
  134) still reflect the deprecated interpretation; flag for a later pass.
- **`moss/test_validation.py`** — dead code (imports the removed `ls_green`);
  its `eigenstrain` references belong to a removed volume-source solver,
  unrelated to this policy.

## Verification gates

- `JAX_PLATFORMS=cpu python -m mhf.validate` (both copies): 8/8 groups,
  incl. group 8 (off-fault no-op 2.1e-3; on-fault raw ~1/ε vs corrected
  bounded; wiring identities exactly 0; marginal magnitude and sign).
- `python -m pytest tests/ -q` in moss2 (incl. `test_eigenstress_showcase`).
- Showcase/fig13 probes print on-trace total vs eigenstress vs elastic.
- Use `/Users/meade/miniforge3/envs/main/bin/python` — the default `python`
  lacks sympy/jax/pytest.

## Addendum (2026-09-17)

This file was restored from `~/Desktop/moss-org.zip` into the pruned
working folder (which now holds only `clq/`, `medt_paper/`, `moss/`, `msd/`);
the `moss2/`, top-level `mhf/`, `moss_manuscript_wip/` and `moss_figures/`
entries above refer to directories that exist only in that archive.  Since the
sweep above:

- **Exact finite-triangle eigenstress** now exists in closed form, as an
  alternative to the nearest-triangle infinite-plane approximation of
  `anelastic.py`: `moss/mollified_kernel/analytical_kernels.py::analytical_eigenstress_kernel`
  (2026-09-14; Φ_ε = (15ε⁴/8π)·I₇) and its vectorized twin
  `analytical_batch.py::eigenstress_batch` (2026-09-17), both mirrored in
  `medt_paper/mollified_kernel/`; `clq.eigenstress` does the same for P0/P1/P2
  slip.  `anelastic.py` (msd, moss, mhf copies) is unchanged.
- **Paper figures** (`moss/manuscript/main.tex`): figures 2–3 subtract the
  exact kernel (2026-09-14); figure 5 subtracts the blob convolution by an
  angular integral; figure 8 now subtracts the exact eigenstress of the
  piecewise-constant element slips (2026-09-17; it previously subtracted the
  infinite-plane value of the smooth taper, which left a spurious rim ring and
  under-stated the coarse-mesh imprint).
- **Unrelated kernel fix, same date:** the λ/μ pairing of the slip →
  displacement kernel was corrected in `moss`, `mh_deploy` and `medt_paper`
  (see `CLAUDE.md`); it affects displacements, not the eigenstress, and is
  invisible at ν = 1/4.
- The interpreter path in "Verification gates" above does not exist on the
  current machine; use `/Users/meade/micromamba/bin/python`.
