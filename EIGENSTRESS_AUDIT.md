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

## Addendum (2026-09-18) — msd switched to the EXACT eigenstress

The "Fixes applied" table above records `msd/` as **no changes — already
compliant**.  That was true of the *policy* (it subtracted, by default, gated)
but not of the *value*: `msd/mbem/evaluate.py` took `C:ε*` from
`anelastic.py`, whose nearest-triangle assignment and infinite-plane marginal
`ρ = 0.75ε⁴/(d²+ε²)^2.5` are the `d/L → 0` limit of the finite-triangle
integral.  Measured on a planar 10×6 km fault, 1 m strike slip, μ = 30 GPa,
ν = 0.30, at element centroids, at msd's own default ε/h = 1.25: the
approximation is up to **1.93× too large at rim elements** (≈2.00× exactly on
a free patch edge, where the blob only sees a half plane), 33–49 % of peak
`C:ε*`, on 56–98 % of elements depending on refinement — and because
`C:ε* ~ 1/ε`, the ABSOLUTE error grows under refinement.  Deep inside an
element the two agree to ~1e-6 (it *is* the infinite-plane limit there).

- **Fix:** `msd/mbem/kernels/tri_kernels.py::eigenstress_contract` — the exact
  finite-triangle form `Φ_ε = (15ε⁴/8π)·I₇` per element, summed over ALL fault
  elements, each with its OWN ε; numba, parallel over observation points;
  reached via `_stress_from_source(..., kernel="eigen")`.  Machine-identical to
  `moss/mollified_kernel::analytical_eigenstress_kernel` / `eigenstress_batch`
  and to `clq.eigenstress`.
- `msd/anelastic.py` is **unchanged** (frozen oracle, per msd's two-layer
  rule), and remains the reference for the infinite-plane limit.  `msd`'s
  `examples/demo_*` still call it directly and inherit the rim error.
- **Sign preserved:** msd's fault term is `−Sdd@slip`, so `evaluate_stress`
  still *adds* `+C:ε*`.
- The former "eigenstress subtraction needs a near-uniform fault ε" restriction
  (`evaluate_stress` raised on a graded fault ε) is **lifted**: a per-element
  sum has no such requirement.
- **Gate:** `msd/verify/verify_eigenstress_exact.py` (38 checks) — moss and clq
  entrywise parity at ν = 0.25/0.30/0.45, deep-interior agreement with the
  frozen `anelastic.py`, near-edge *disagreement* (so the gate discriminates
  against the old wiring), the sign pinned by an ε-sweep finiteness test with a
  wrong-sign tripwire, and graded-ε parity.  `verify_evaluate_stress.py`
  check 4 was repointed at the exact form.
- Not touched: `moss/`, `medt_paper/`, `clq/`, and the other vendored
  `anelastic.py` copies — the same substitution is still available to them.

## Addendum (2026-09-19) — the policy applies to BOUNDARY double layers too

The policy above speaks of fault slip.  It applies to **every** mollified
double layer.  In the BEM representation formula a boundary patch's `u_p`
enters through the same `Sdd` kernel as a slip: the formula writes the field
as a jump between `u` (inside the region) and zero (outside), and mollifying
that fictitious jump smears an eigenstress `μ u_p ⊗ n Φ_ε(d)` (plus the `λ`
trace term) into the body within ~3 ε of the patch — non-physical there, and
`1/ε`-large.  `msd/mbem/evaluate.py::evaluate_stress` subtracted it for faults
only; measured against the exact Kelvin field on an icosphere it was the whole
"h-independent interior stress error near a boundary" of
`HARDENING_AUDIT.md` item 4 (Neumann, d/h = 0.66, ε/h = 0.3: 2.43e-1 / 2.22e-1
/ 2.10e-1 raw over 80/320/1280 triangles; 2.25e-1 / 1.29e-1 / 7.4e-2 with the
term; Dirichlet at d/h = 0.25: 8–15 raw, 0.2 with the term).

- **Fix:** `evaluate_stress` now removes `+σ_p C:ε*(u_p)` for every boundary
  patch with the patch's own orientation `σ_p`, through the one helper
  `_double_layer_stress` that faults use too.  `subtract_anelastic=True`
  (default) therefore means "the elastic stress of the whole representation".
- `ddbem/ddbem/model.py::_evaluate` had **always** subtracted every patch's
  eigenstress; the 1.5e-3 msd-vs-ddbem interior-stress discrepancy noted in
  `CODE_REVIEW_2026-09-19.md` was exactly this term.
- **Gate:** `msd/verify/verify_boundary_eigenstress.py` (12 checks, 2 s):
  wiring identity at ν = 0.30 with a fault present, accuracy at d/h = 0.5
  (7.0e-2 vs 2.2 raw), the refinement ladder, a rigid-translation Dirichlet
  sphere whose exact stress is zero (elastic/raw 2.4e-3), deep no-op.  The
  wiring checks in `verify_evaluate_stress.py` and `verify_eigenstress_exact.py`
  now restate the boundary terms by hand as well.
- Not touched: the `mhf` / `moss` direct-kernel pipelines have no boundary
  double layers (their "boundary" is the analytic half space), so nothing to
  subtract there.
