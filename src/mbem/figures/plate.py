"""A plate over a relaxing substrate against the antiplane closed form.

A STUDY figure over ``half_x`` (and ``sides``), because the question it answers
is about the box: mbem has no half-space, so "an elastic plate over a
viscoelastic half-space" is a plate over a deep clamped channel with sides,
and the only honest statement of how good that is comes from changing the box.

The reference is the infinitely long fault of Nur & Mavko and Savage &
Prescott (``cases.layered_box``), which the mid-strike profile of a long fault
in a large box approaches. So what the ladder shows is not an error bar on the
method -- the gates pin that against closed forms that ARE the solved problem
-- but how much box and how much fault a result near the fault needs.

Three panels: the surface displacement at three times; the postseismic part
``u(t) - u(0+)``, which is the long-wavelength part and feels the box first;
and that part over the reference at fixed distance, against the box size, for
clamped and free plate sides. Free sides let the plate float as the substrate
relaxes; clamped sides hold it. Neither is a half-space, and they need not
bracket it.
"""

from __future__ import annotations

import numpy as np

from mbem.figures import save_figure
from mbem.figures.style import set_paper_style

TIMES = (0.0, 1.0, 10.0)            # in Maxwell times; 0 is the elastic state
X_MAX = 120.0                       # km: the profile shown
X_BINS = np.geomspace(2.0, X_MAX, 13)
STRIP = 0.06                        # half-width of the mid-strike strip / fault half-length
PROBE = (16.0, 32.0)                # km: where panel c reads the postseismic part
BLUE, ORANGE = "#1f5fa6", "#d9731a"


def _regions(run) -> dict:
    return {r["name"]: r["material"] for r in run.resolved["model"]["regions"]}


def profile(run):
    """``(s, x, u)``: the mid-strike profile of every state of a Laplace run.

    ``u[k]`` is the fault-parallel surface displacement of state ``k`` per unit
    slip, binned in distance from the fault and counted positive in the sense
    the side moves -- both sides are used, the far one mirrored, so the
    profile has twice the points and a net translation of the plate cancels
    out of it instead of hiding in it.
    """
    from mbem import laplace

    s, stack = laplace.run_stack(run.run_dir)
    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    slip = run.resolved["spec"]["model"]["params"].get("slip_mag", 1.0e-3)
    cen = run.meshes["top__v"][run.meshes["top__t"]].mean(axis=1)
    strip = np.abs(cen[:, 1]) < STRIP * gp["fault_half_len"]
    dist = np.abs(cen[:, 0])
    # right-lateral: the +x side moves toward -y, the -x side toward +y
    moved = -np.sign(cen[:, 0]) * stack["u:top"][:, :, 1] / slip
    x, u = [], []
    for a, b in zip(X_BINS[:-1], X_BINS[1:]):
        sel = strip & (dist >= a) & (dist < b)
        if sel.any():
            x.append(dist[sel].mean())
            u.append(moved[:, sel].mean(axis=1))
    return s, np.array(x), np.array(u).T


def reference(run, s, x, t):
    """The antiplane closed form at the times ``t``, ``(n_t, n_x)``.

    Exact for a Maxwell substrate (``maxwell_series``). For any other
    rheology, the image series at ``mu(s)`` put through the SAME fit as the
    solution, so the inversion's own error is common to both and what is left
    is the geometry.
    """
    from mbem import laplace, rheology
    from mbem.cases import layered_box as lb

    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    depth, thick = gp["fault_depth"], -gp["z_interface"]
    mats = _regions(run)
    mu_p, mu_s = mats[lb.PLATE]["mu"], mats[lb.SUBSTRATE]["mu"]
    rheo = rheology.from_spec(
        run.resolved["spec"]["laplace"]["rheology"][lb.SUBSTRATE])
    t = np.atleast_1d(np.asarray(t, float))
    if isinstance(rheo, rheology.Maxwell):
        out = lb.maxwell_series(x, np.maximum(t, 0.0), depth, thick, mu_p,
                                mu_s, rheo.tau)
    else:
        stack = np.array([lb.image_series(x, depth, thick, mu_p,
                                          mu_s * rheo.shear_ratio(sv))
                          for sv in [np.inf, *s]])
        out = laplace.Prony(s).weights(t) @ stack
    out[t <= 0.0] = lb.image_series(x, depth, thick, mu_p, mu_s)
    return out


def in_time(run, t):
    """``(x, u(t), reference(t))`` with ``t = 0`` meaning the elastic state."""
    from mbem import laplace

    s, x, u = profile(run)
    t = np.atleast_1d(np.asarray(t, float))
    ut = laplace.Prony(s).weights(np.maximum(t, 1e-300)) @ u
    ut[t <= 0.0] = u[0]
    return x, ut, reference(run, s, x, t)


def plate_substrate(study):
    import matplotlib.pyplot as plt
    set_paper_style()

    boxes = sorted(study.axis("half_x"), key=float)
    sides = study.axis("sides") if "sides" in study.swept else [None]

    def run_of(half_x, side):
        return study.one(half_x=half_x, **({} if side is None
                                           else {"sides": side}))

    lead = "clamped" if "clamped" in sides else sides[0]
    big = run_of(boxes[-1], lead)
    x, u, ref = in_time(big, TIMES)
    xf = np.linspace(1.0, X_MAX, 240)
    ref_f = reference(big, profile(big)[0], xf, TIMES)
    shades = plt.cm.plasma(np.linspace(0.12, 0.72, len(TIMES)))

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.6, 3.5))
    for k, t in enumerate(TIMES):
        lab = r"$t = 0^+$" if t == 0 else rf"$t = {t:g}\,\tau$"
        a0.plot(xf, ref_f[k], "-", color="k", lw=0.8)
        a0.plot(x, u[k], "o", color=shades[k], ms=3.6, mec="k", mew=0.3,
                label=lab)
        if t > 0:
            a1.plot(xf, ref_f[k] - ref_f[0], "-", color="k", lw=0.8)
            a1.plot(x, u[k] - u[0], "o", color=shades[k], ms=3.6, mec="k",
                    mew=0.3, label=lab)
    for ax, ymax, ylab in ((a0, 0.5, r"$u\,/\,s$"),
                           (a1, 0.15, r"$[u(t) - u(0^+)]\,/\,s$")):
        ax.set_xlim(0.0, X_MAX)
        ax.set_xticks([0, 60, 120])
        ax.set_ylim(0.0, ymax)
        ax.set_yticks([0.0, ymax / 2.0, ymax])
        ax.set_xlabel(r"$x$ (km)")
        ax.set_ylabel(ylab)
        ax.set_box_aspect(1)
        ax.legend(frameon=False, loc="upper right" if ax is a0 else "lower right",
                  bbox_to_anchor=(0.9, 0.9) if ax is a0 else None,
                  handletextpad=0.2)
    name = f"half-width {float(boxes[-1]):g} km" + (
        "" if lead is None else f", {lead} sides")
    a0.text(0.05, 0.05, name, transform=a0.transAxes, ha="left", va="bottom",
            fontsize=8)
    a0.text(0.95, 0.95, "a", transform=a0.transAxes, ha="right", va="top")
    a1.text(0.95, 0.95, "b", transform=a1.transAxes, ha="right", va="top")

    lo = 1.0
    for side, color in zip(sides, (BLUE, ORANGE)):
        for t, ls, mfc in ((1.0, "-", color), (10.0, "--", "white")):
            ratio = []
            for hx in boxes:
                xx, uu, rr = in_time(run_of(hx, side), (0.0, t))
                sel = (xx >= PROBE[0]) & (xx < PROBE[1])
                ratio.append(float((uu[1] - uu[0])[sel].mean()
                                   / (rr[1] - rr[0])[sel].mean()))
            lo = min(lo, min(ratio))
            name = "" if side is None else f"{side}, "
            a2.plot([float(b) for b in boxes], ratio, ls, color=color,
                    marker="o", ms=4.2,
                    mfc=mfc, mec=color, label=rf"{name}$t = {t:g}\,\tau$")
    a2.axhline(1.0, color="0.5", lw=0.6, ls=":")
    a2.set_xscale("log")
    a2.set_xticks([float(b) for b in boxes])
    a2.set_xticklabels([f"{float(b):g}" for b in boxes])
    a2.minorticks_off()
    floor = np.floor(lo * 5.0) / 5.0
    a2.set_ylim(floor, 1.0 + 0.5 * (1.0 - floor) / 4.0)
    a2.set_yticks([floor, 0.5 * (floor + 1.0), 1.0])
    a2.set_xlabel("box half-width (km)")
    a2.set_ylabel("model / closed form")
    a2.set_box_aspect(1)
    a2.legend(frameon=False, loc="lower right")
    a2.text(0.05, 0.95, rf"$u(t) - u(0^+)$, $x$ = {PROBE[0]:g}-{PROBE[1]:g} km",
            transform=a2.transAxes, ha="left", va="top", fontsize=8)
    a2.text(0.95, 0.95, "c", transform=a2.transAxes, ha="right", va="top")

    fig.tight_layout(w_pad=1.6)
    out = save_figure(fig, study.study_dir, "plate_substrate")
    plt.close(fig)
    return out
