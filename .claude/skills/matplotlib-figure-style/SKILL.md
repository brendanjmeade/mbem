---
name: matplotlib-figure-style
description: Use when producing matplotlib figures for scientific publication / presentation. Enforces a clean, sparse house style: minimal ticks at axis edges, no grid, math-notation labels, square panels for cartoons, axis-equal for maps and cross-sections, single-hue fills with black edges, panel letters in top-right insets. Apply for any plt.* / fig.* code, especially geophysical / earthquake / fault / GPS plots, time series, multi-panel grids, and 2-D fields.
---

# Matplotlib figure style

Goal: clean, publication-ready figures that read instantly. Strip everything that does not carry information.

## Core principles

1. **Less ink.** No grid, no extra spines stripped, no tick marks on every increment, no boxed legends if they can sit unframed, no titles when an axis label suffices.
2. **Tick labels sit at the edges of the data range, not interior.** Two to four ticks per axis is typical. Pick round numbers that anchor the limits (e.g. `[-100, 0, 100]`, `[235, 240, 245]`, `[0, 50, 100]`, `[10⁻⁵, 10¹]`). Never accept matplotlib's autolocator output unchanged.
3. **Use `set_aspect('equal')` for anything with shared physical units on x and y** — maps (lon/lat), fault cross-sections (x/depth in km), schematic geometries. Use `ax.set_box_aspect(1)` (square frame) for cartoon / conceptual panels where x and y are not the same quantity but you want a tidy grid of identical boxes. The two are *not* interchangeable: equal preserves data ratio, square forces frame ratio. Never set both at once unless data limits already happen to be square.
4. **Multi-panel grids share inner labels.** Only the leftmost column gets a y-label, only the bottom row gets an x-label. Panel letters (`a`, `b`, …) go in the top-right corner inside the axes, plain text, no parens, no bbox unless background contrast demands it.

## Defaults to set once per figure

```python
import matplotlib.pyplot as plt

# Keep the full box (top + right spines on). Do NOT strip them — these figures use the full rectangle.
# Don't call sns.despine(); don't set spines.top/right to invisible.

# Tick direction: default 'out' is fine. Tick length short. No minor ticks unless on a log axis.
ax.tick_params(direction="out", length=3, width=0.8)
```

Font: default DejaVu Sans is acceptable; if specifying, use a serif/sans that renders math cleanly. Math labels use `$...$` — e.g. `r"$v \propto r^{-2}$"`, `r"$\eta$ (transitions / km / nat)"`, `r"$\dot{m}_d$ (m$^3$ / year)"`.

## Axis labels

- Lowercase, terse, with units in parentheses: `x (km)`, `t (years)`, `velocity`, `latitude`, `longitude`, `r (km)`, `s (m)`, `M`.
- Often no title at all. When a title is used, it is short and either a math expression (`$\dot{m}_d$ (t = 66)`) or a single descriptive phrase.
- Multi-panel column headers replace per-panel titles: put `block bounding (BB)`, `locally driven (LD)`, etc. above the top row only.

## Ticks — the signature move

```python
# Map example
ax.set_xlim(235, 250)
ax.set_ylim(30, 50)
ax.set_xticks([235, 240, 245, 250])
ax.set_yticks([30, 35, 40, 45, 50])

# Cartoon panel example
ax.set_xlim(-100, 100)
ax.set_xticks([-100, 0, 100])
ax.set_yticks([])               # frequently the y-axis has no ticks at all on schematic panels
```

Rules of thumb:
- Endpoints of the tick list usually equal the axis limits — ticks sit *on* the frame corners.
- For schematic / cartoon panels, the y-axis often has **no tick labels at all** (just the word `velocity` as the label). Don't invent y-ticks where none belong.
- Log axes: show only the decade endpoints (`10⁻⁵`, `10¹`). Suppress intermediate decade labels with `ax.yaxis.set_major_locator(LogLocator(numticks=2))` or by setting ticks explicitly.
- Time series spanning 0–1000 yr: ticks at `[0, 200, 400, 600, 800, 1000]` — round, sparse, edge-aligned.

## Colors and fills

- Single-hue regions filled under a curve with `fill_between` and a **black edge** drawn separately as a line on top. The combo of solid fill + thin black contour is a recurring motif.
- Palette is restrained: one warm + one cool per figure is plenty. Recurring choices: matplotlib default orange (`#ff7f0e`-ish), default blue (`#1f77b4`-ish), muted red, green, purple. For the kinematic style cartoons, each column has its own hue (orange / blue / teal / red / green / purple).
- Diverging fields: `RdBu_r` or `coolwarm`, centered on zero, symmetric vmin/vmax. Show the colorbar with sparse ticks (`[-1.78, -1.19, ..., 1.78]` style — endpoints + a few interior values).
- Sequential fields: `viridis`, `plasma`, `inferno`, or `magma`. For density / heatmap overlays on maps, use `inferno`/`plasma` with low alpha so basemap shows through.
- Basemaps: light gray land (`lightgray` or `0.85`), white ocean. Coastlines / faults as thin gray lines. No filled ocean color.
- Reference lines: dashed, soft red or soft orange, with a vertical text label rotated 90° next to the line (e.g. `lower slip rate bound`).

## Aspect and figure sizing

| Plot kind | Aspect choice |
|---|---|
| Map (lat/lon) | `ax.set_aspect('equal')` |
| Fault cross-section (x vs depth) | `ax.set_aspect('equal')` — these come out as wide thin strips, which is correct |
| Cartoon / schematic panel grid | `ax.set_box_aspect(1)` per axes |
| Scatter of two unrelated quantities | neither — let the data set the aspect, but pick a `figsize` that lands near 1:1 |
| Time series stacked panels | shared x, each panel short (e.g. `figsize=(width, 2.0)` per row) |

For multi-panel cross-section stacks (e.g. six 200 km × 25 km strips), build with `gridspec` and give each row the same height; the wide-thin shape is the point.

## Multi-panel layout

```python
fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(7, 7),
    sharex=True, sharey=True,
    gridspec_kw=dict(wspace=0.08, hspace=0.08),
)

for i, ax in enumerate(axes.flat):
    ax.set_box_aspect(1)
    ax.text(0.95, 0.95, "abcdefghi"[i],
            transform=ax.transAxes, ha="right", va="top")

# Only outer labels:
for ax in axes[-1, :]:
    ax.set_xlabel("$x$ (km)")
for ax in axes[:, 0]:
    ax.set_ylabel("velocity")

# Column headers on top row only:
axes[0, 0].set_title("block bounding (BB)")
axes[0, 1].set_title("locally driven (LD)")
axes[0, 2].set_title("embedded strain (ES)")
```

A faint horizontal zero-line (`axhline(0, color='0.5', lw=0.6)`) inside each cartoon panel is common when the curve crosses zero.

## Legends

- Position inside the axes, usually upper right, sometimes upper left.
- Frame on (default) for plots over busy backgrounds; frame off (`frameon=False`) for clean line plots.
- Entries terse: `$\sigma = 0.5$`, `$t = 1.0$ yr`, `coseismic (82%)`, `rupture zone`. Use math mode for symbols.
- Inset metadata boxes (e.g. `n=538, 4 M ≥ 7` or `seed = 42`) sit in a corner with a thin black frame — these are *annotations*, not legends, drawn with `ax.text(..., bbox=dict(boxstyle='square', fc='white', ec='black', lw=0.6))`.

## Colorbars

- Vertical, attached to the right of the axes, narrow.
- Tick labels limited to 5–7 evenly spaced values that include the endpoints, often with two decimal places.
- Label below or beside the bar in math notation with units: `$\dot{m}$ (m$^3$ / year)`.
- For diverging maps, force symmetric limits: `vmin=-vmax`.

## Scatter and "lollipop" markers

- Magnitude / size encoded via `s=` with a perceptually scaled mapping (e.g. `s = 2 ** (M)` or similar) — the result is small purple dots at low M, big yellow circles at high M, with a `viridis`/`plasma`/`inferno` colormap on the same variable. Stems are thin gray vertical lines from baseline to marker.
- Translucent overlapping circles for density (`alpha=0.3–0.5`) — see the entropy scaling figure.

## Things to avoid

- `plt.grid(True)` — never on by default.
- `seaborn-whitegrid` / `ggplot` styles — these break the look.
- Colored axis spines, thick spines (>1 pt), or removed spines.
- Default tick density on tight axes — always override with an explicit short list.
- Titles that duplicate axis labels.
- Legends outside the axes (`bbox_to_anchor=(1.02, 1)`-style) when an inside placement fits.
- Mixing `set_aspect('equal')` and `set_box_aspect(1)` on the same axes.
- 3-D plots, pie charts, stacked bars — none of these appear in the reference set.

## Quick checklist before saving

1. Are tick labels limited to ~2–5 round values per axis, with endpoints on the frame?
2. Is the aspect choice deliberate (`equal` for physical, `box_aspect(1)` for cartoon, neither for free)?
3. Is the grid off? Are there no stray minor ticks?
4. Do multi-panel grids share outer labels and use top-right letter insets?
5. Are math symbols in `$...$` and units in parentheses?
6. Is the palette restrained — one warm + one cool, or a single sequential map?
7. `plt.savefig(..., dpi=500, bbox_inches='tight')` — tight bbox, high dpi.
