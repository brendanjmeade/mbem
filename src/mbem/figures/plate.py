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
X_MAX = 120.0                       # km: the profile shown, at least
X_BINS = np.geomspace(2.0, X_MAX, 13)
STRIP = 0.06                        # half-width of the mid-strike strip / fault half-length
PROBE = (16.0, 32.0)                # km: where panel c reads the postseismic part
CYCLES = 5.0                        # the cycle drawn repeats this often in the fit's window
PHASES = (0.05, 0.95)               # early and late in it, as fractions of the period
X_FAR = 75.0                        # km past the fault's lower edge: where the cycle is followed
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


# ---------------------------------------------------------------------
# a dip-slip fault: the plane-strain reference, both in-plane components
# ---------------------------------------------------------------------

def face_ratio(coarse, fine) -> float:
    """How many times larger ``coarse``'s plate faces are than ``fine``'s,
    read from the two runs: the top, the interface and the interface's eps,
    which must be in ONE ratio (``layered_box.two_mesh_limit``)."""
    def faces(run):
        m = run.resolved["spec"]["model"]
        return np.array([m["geometry"]["params"]["edge_near"],
                         m["geometry"]["params"]["edge_interface"],
                         m["eps"]["interface"]], float)

    r = faces(coarse) / faces(fine)
    if not np.allclose(r, r[0], rtol=1e-9):
        raise ValueError(f"the top, the interface and its eps are in ratios "
                         f"{r}: a two-mesh limit needs one")
    return float(r[0])


def in_plane(run, coarse=None):
    """``(s, x, u)``: as ``profile`` for a dip-slip fault, ``u[k]`` the
    ``(n_x, 2)`` horizontal (fault-normal) and vertical surface displacement
    of state ``k`` per unit slip, at SIGNED distance from the trace.

    With ``coarse``, the same model on larger plate faces, ``u`` is the
    two-mesh limit of the pair at ``run``'s points."""
    from mbem.cases import layered_box as lb

    s, x, u = _in_plane(run)
    if coarse is None:
        return s, x, u
    sc, xc, uc = _in_plane(coarse)
    if sc.shape != s.shape or not np.allclose(sc, s):
        raise ValueError("the two runs of a pair must share their states")
    at = np.empty_like(u)
    for side in (-1.0, 1.0):                # never across the trace: a jump
        m, mc = np.sign(x) == side, np.sign(xc) == side
        for k in range(u.shape[0]):
            for j in range(u.shape[2]):
                at[k, m, j] = np.interp(x[m], xc[mc], uc[k, mc, j])
    return s, x, lb.two_mesh_limit(at, u, face_ratio(coarse, run))


def signed_bins(run) -> np.ndarray:
    """The bin edges of a dip-slip profile, in SIGNED distance from the
    trace, because such a fault is not symmetric about it: 6 km from 60 km
    behind the trace to 24 km past the fault's lower edge (60 km at least),
    where the hanging wall sits over the fault, and about 15 km beyond, out
    to the edge of the near field (``X_MAX`` at least)."""
    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    tip = gp["fault_depth"] / np.tan(np.deg2rad(gp["dip"]))
    fine = max(60.0, 6.0 * np.ceil((tip + 24.0) / 6.0))
    reach = max(X_MAX, gp.get("near_field_radius", X_MAX), fine + 15.0)

    def beyond(start):
        return np.linspace(start, reach, 1 + max(1, round((reach - start) / 15.0)))[1:]

    return np.concatenate([-beyond(60.0)[::-1],
                           np.arange(-60.0, fine + 0.5, 6.0), beyond(fine)])


def _in_plane(run):
    from mbem import laplace

    s, stack = laplace.run_stack(run.run_dir)
    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    slip = run.resolved["spec"]["model"]["params"].get("slip_mag", 1.0e-3)
    cen = run.meshes["top__v"][run.meshes["top__t"]].mean(axis=1)
    strip = np.abs(cen[:, 1]) < STRIP * gp["fault_half_len"]
    moved = stack["u:top"][:, :, [0, 2]] / slip
    x, u = [], []
    edges = signed_bins(run)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = strip & (cen[:, 0] >= a) & (cen[:, 0] < b)
        if sel.any() and a * b > 0.0:               # not the bin on the trace
            x.append(cen[sel, 0].mean())
            u.append(moved[:, sel].mean(axis=1))
    return s, np.array(x), np.array(u).transpose(1, 0, 2)


def weights(run) -> tuple:
    """``rho g`` of the plate and the substrate (GPa/km); zeros without
    gravity."""
    from mbem import defaults
    from mbem.cases import layered_box as lb

    grav = run.resolved["model"].get("gravity")
    if grav is None:
        return 0.0, 0.0
    return tuple(grav["density"][name] * grav["g"] * defaults.WEIGHT_GPA_PER_KM
                 for name in (lb.PLATE, lb.SUBSTRATE))


_REFERENCE: dict = {}       # a figure asks for the same stack many times
_SPACE = False              # every run of the study against the 3-D reference


def _in_space(study) -> bool:
    """A study with a window among its runs is about ruptures as they are:
    all of its runs are then compared with the reference in three
    dimensions, the clamped and the free ones too."""
    global _SPACE
    _SPACE = any(r.resolved["spec"]["model"]["params"].get("sides") == "window"
                 for _p, r in study.rows)
    return _SPACE


def _reference_stack(run, s, x, z_bottom=None):
    """``(1 + len(s), n_x, 2)``: the reference of every state, the elastic
    one first, for the slip the run PRESCRIBES in a plate over a half-space
    (or over a layer clamped at ``z_bottom``).

    A run whose walls are a window is the rupture it is, and its reference
    is that rupture's, in three dimensions, along the mid-strike line
    (``cases.layered_space``). Any other is a long fault in a box standing in
    for an infinitely long one, and its reference is plane strain."""
    from mbem import rheology
    from mbem.cases import layered_box as lb

    key = (str(run.run_dir), np.asarray(x, float).tobytes(), z_bottom, _SPACE)
    if key in _REFERENCE:
        return _REFERENCE[key]
    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    mp = run.resolved["spec"]["model"]["params"]
    mats = _regions(run)
    plate, sub = ((mats[n]["mu"], mats[n]["lam"])
                  for n in (lb.PLATE, lb.SUBSTRATE))
    rheo = rheology.from_spec(
        run.resolved["spec"]["laplace"]["rheology"][lb.SUBSTRATE])
    subs = [rheology.moduli(rheo, *sub, sv) for sv in [np.inf, *s]]
    thick = -gp["z_interface"]
    # a lower reach that creeps is solved, not prescribed
    depth = gp["fault_depth"] if mp.get("lower", "slip") == "slip" \
        else min(gp["fault_depth"], thick)
    if _SPACE or mp.get("sides") == "window":
        from mbem.cases import layered_space

        out = np.asarray(layered_space.surface_displacement(
            np.column_stack([x, np.zeros_like(x)]), gp["fault_half_len"],
            depth, thick, gp["dip"], mp.get("rake", 180.0), plate, subs,
            weights=weights(run), z_bottom=z_bottom))[:, :, [0, 2]]
    else:
        out = np.stack(lb.dip_slip_profile(
            x, depth, thick, gp["dip"], plate, subs, weights(run), z_bottom),
            axis=-1)
    _REFERENCE[key] = out
    return out


CONTRASTS = ("gravity", "sides", "lower", "slab_depth", "creep_tau")
NO_CREEP = 1.0e5            # a creep time no run's window reaches: a held reach


def _contrast(study) -> tuple:
    """``(key, values)``: the two-valued axis a study figure draws in two
    colours, the value drawn first and filled leading -- gravity on, the
    walls a window, the lower reach creeping (the fastest law first), the
    slab there."""
    for key in CONTRASTS:
        if key in study.swept:
            vals = list(study.axis(key))
            if key in ("sides", "lower"):
                vals.sort(key=lambda v: v not in ("window", "creep"))
            else:
                vals.sort(key=float, reverse=key != "creep_tau")
            return key, vals
    return None, [None]


def _named(key, v, tail="") -> str:
    if v is None:
        return ""
    if key == "gravity":
        text = "gravity" if float(v) > 0.0 else "no gravity"
    elif key == "sides":
        text = f"{v} walls"
    elif key == "lower":
        text = "creep below" if v == "creep" else "slip below"
    elif key == "creep_tau":
        text = "no creep below" if float(v) >= NO_CREEP \
            else rf"creep below, $\tau_c = {float(v):g}\,\tau$"
    else:
        text = f"slab to {float(v):g} km" if float(v) > 0.0 else "no slab"
    return text + tail


def in_plane_in_time(run, t, x_ref=None, z_bottom=None, coarse=None):
    """``(x, u(t), x_ref, reference(t))``, ``t = 0`` the elastic state. The
    reference is its profile at ``(mu(s), lam(s))`` put through the SAME fit
    as the solution (``reference``), at the solution's points unless
    ``x_ref`` gives others. ``coarse`` as for ``in_plane``."""
    from mbem import laplace

    s, x, u = in_plane(run, coarse)
    x_ref = x if x_ref is None else x_ref
    t = np.atleast_1d(np.asarray(t, float))
    w = laplace.Prony(s).weights(np.maximum(t, 1e-300))
    w[t <= 0.0] = np.eye(1 + len(s))[0]
    return (x, np.tensordot(w, u, axes=1), x_ref,
            np.tensordot(w, _reference_stack(run, s, x_ref, z_bottom), axes=1))


def misfit(run, t, z_bottom=None, coarse=None):
    """The postseismic displacement ``u(t) - u(0+)`` of a dip-slip run against
    the reference: rms of the difference over rms of the reference, both
    in-plane components, over the profile. ``(n_t,)``."""
    t = np.concatenate([[0.0], np.atleast_1d(np.asarray(t, float))])
    _x, u, _xr, ref = in_plane_in_time(run, t, z_bottom=z_bottom, coarse=coarse)
    post, want = u[1:] - u[0], ref[1:] - ref[0]
    return np.sqrt(((post - want) ** 2).sum(axis=(1, 2))
                   / (want ** 2).sum(axis=(1, 2)))


def plate_thrust(study):
    """A thrust in the plate against the plane-strain reference, with and
    without gravity when the study swept ``gravity``.

    Three panels: the vertical and the horizontal surface displacement of the
    largest box, at the earthquake and at the latest time the fit covers; and
    the misfit of the postseismic part against the box size. Without gravity
    nothing but the plate resists a wavelength the substrate has relaxed
    under, so the late vertical motion is large and belongs to the box as
    much as to the fault; with it the motion is bounded and the box matters
    less.

    A study that swept ``refine`` is a pair on one ray of the plate's faces:
    the first two panels then draw the two-mesh limit, and the third the
    misfit against the size of those faces, the limit at zero. One that swept
    neither the box nor the faces has nothing to put on that axis, and the
    third panel is the misfit through TIME.
    """
    import matplotlib.pyplot as plt
    from mbem import laplace
    set_paper_style()
    _in_space(study)

    boxes = sorted(study.axis("half_x"), key=float) \
        if "half_x" in study.swept else [None]
    contrast, grav = _contrast(study)

    faces = sorted(study.axis("refine"), key=float) \
        if "refine" in study.swept else [None]

    def run_of(half_x, g, refine=faces[-1]):
        sel = {k: v for k, v in (("half_x", half_x), (contrast, g),
                                 ("refine", refine)) if v is not None}
        return study.one(**sel) if sel else study.rows[0][1]

    def coarse_of(half_x, g):
        return run_of(half_x, g, faces[0]) if len(faces) > 1 else None

    def named(g, tail=""):
        return _named(contrast, g, tail)

    late = float(f"{laplace.window(in_plane(run_of(boxes[-1], grav[0]))[0])[1]:.3g}")
    span = signed_bins(run_of(boxes[-1], grav[0]))[-1]
    xf = np.linspace(-span, span, 241)
    xf = xf[np.abs(xf) > 0.5]

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.6, 3.5))
    lo, hi = [0.0, 0.0], [0.0, 0.0]
    for g, color in zip(grav, (BLUE, ORANGE)):
        x, u, _xf, ref = in_plane_in_time(run_of(boxes[-1], g), (0.0, late), xf,
                                          coarse=coarse_of(boxes[-1], g))
        for ax, comp in ((a0, 1), (a1, 0)):
            if g == grav[0]:
                ax.plot(xf, ref[0, :, comp], "-", color="k", lw=0.8)
                ax.plot(x, u[0, :, comp], "o", color="0.8", ms=3.4, mec="k",
                        mew=0.3, label=r"$t = 0^+$")
            ax.plot(xf, ref[1, :, comp], "-", color=color, lw=0.8)
            ax.plot(x, u[1, :, comp], "o", color=color, ms=3.4, mec="k",
                    mew=0.3, label=rf"$t = {late:g}\,\tau$" + named(g, "")
                    .join((", ", "")) if g is not None
                    else rf"$t = {late:g}\,\tau$")
            lo[comp] = min(lo[comp], u[:, :, comp].min(), ref[:, :, comp].min())
            hi[comp] = max(hi[comp], u[:, :, comp].max(), ref[:, :, comp].max())
    for ax, comp, lab, letter in ((a0, 1, r"$u_z\,/\,s$", "a"),
                                  (a1, 0, r"$u_x\,/\,s$", "b")):
        bottom = np.floor(lo[comp] * 5.0) / 5.0
        top = np.ceil(hi[comp] * 5.0) / 5.0
        ax.set_xlim(-span, span)
        ax.set_xticks([-span, 0, span])
        ax.set_ylim(bottom, top)
        ax.set_yticks([bottom, 0.0, top])
        ax.set_xlabel(r"$x$ (km)")
        ax.set_ylabel(lab)
        ax.set_box_aspect(1)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.text(0.95, 0.95, letter, transform=ax.transAxes, ha="right", va="top")
    a0.legend(frameon=False, loc="lower left", handletextpad=0.2, fontsize=8)
    name = "" if boxes[-1] is None else f"half-width {float(boxes[-1]):g} km"
    if len(faces) > 1:
        name = "two-mesh limit" + (", " + name if name else "")
    a1.text(0.05, 0.05, name, transform=a1.transAxes, ha="left", va="bottom",
            fontsize=8)

    worst = 0.0
    in_time_only = len(faces) == 1 and boxes[-1] is None
    if len(faces) > 1:
        # the faces' size over the coarsest's; the limit of the pair at zero
        size = [float(faces[0]) / float(f) for f in faces] + [0.0]
    for g, color in zip(grav, (BLUE, ORANGE)):
        if in_time_only:
            when = np.geomspace(1.0, late, 13)
            m = misfit(run_of(None, g), when)
            worst = max(worst, float(m.max()))
            a2.plot(when, m, "-", color=color, marker="o", ms=4.2, mec=color,
                    label=named(g) or None)
            continue
        if len(faces) > 1:
            across = size
            m = np.array([misfit(run_of(boxes[-1], g, f), (1.0, late))
                          for f in faces]
                         + [misfit(run_of(boxes[-1], g), (1.0, late),
                                   coarse=coarse_of(boxes[-1], g))])
        else:
            across = [float(b or 0.0) for b in boxes]
            m = np.array([misfit(run_of(hx, g), (1.0, late)) for hx in boxes])
        worst = max(worst, float(m.max()))
        for k, (t, ls, mfc) in enumerate(((1.0, "-", color),
                                          (late, "--", "white"))):
            a2.plot(across, m[:, k], ls, color=color,
                    marker="o", ms=4.2, mfc=mfc, mec=color,
                    label=named(g, ", ") + rf"$t = {t:g}\,\tau$")
    if in_time_only:
        a2.set_xscale("log")
        a2.set_xlim(1.0, late)
        a2.set_xticks([1.0, late])
        a2.set_xticklabels(["1", f"{late:g}"])
        a2.minorticks_off()
    elif len(faces) > 1:
        a2.set_xlim(-0.08, 1.08)
        a2.set_xticks([0.0, 0.5, 1.0])
    elif boxes[-1] is not None:
        a2.set_xscale("log")
        a2.set_xticks([float(b) for b in boxes])
        a2.set_xticklabels([f"{float(b):g}" for b in boxes])
        a2.minorticks_off()
    top = np.ceil(worst * 15.0) / 10.0             # the legend's third is clear
    a2.set_ylim(0.0, top)
    a2.set_yticks([0.0, top / 2.0, top])
    a2.set_xlabel(r"$t\,/\,\tau$" if in_time_only
                  else "size of the plate's faces (coarsest = 1)"
                  if len(faces) > 1 else "box half-width (km)")
    a2.set_ylabel("departure from the layered reference"
                  if contrast in ("lower", "slab_depth", "creep_tau")
                  else "misfit to the reference")
    a2.set_box_aspect(1)
    if a2.get_legend_handles_labels()[0]:
        a2.legend(frameon=False, loc="upper right", fontsize=8)
    what = (r"$u(t) - u(0^+)$, both components", f"$|x| <$ {span:g} km")
    if in_time_only:                    # under the letter, clear of curves
        a2.text(0.05, 0.86, "\n".join(what), transform=a2.transAxes,
                ha="left", va="top", fontsize=8)
    elif len(faces) > 1:                # the lines run along the bottom
        a2.text(0.05, 0.62, "\n".join(what), transform=a2.transAxes,
                ha="left", va="top", fontsize=8)
    else:
        a2.text(0.05, 0.05, ", ".join(what), transform=a2.transAxes,
                ha="left", va="bottom", fontsize=8)
    a2.text(0.05, 0.95, "c", transform=a2.transAxes, ha="left", va="top")

    fig.tight_layout(w_pad=1.6)
    out = save_figure(fig, study.study_dir, "plate_thrust")
    plt.close(fig)
    return out


def in_plane_locked(run, phase, period, x_ref=None, coarse=None):
    """``(x, v, x_ref, reference)``: the in-plane surface velocity of the
    fault LOCKED between earthquakes that repeat every ``period``, measured
    from steady slip and per unit slip rate (``Prony.locked_rate_weights``),
    at the fractions ``phase`` of the period after the latest one. The
    reference goes through the same map; ``coarse`` as for ``in_plane``."""
    from mbem import laplace

    s, x, u = in_plane(run, coarse)
    x_ref = x if x_ref is None else x_ref
    w = laplace.Prony(s).locked_rate_weights(
        np.atleast_1d(np.asarray(phase, float)) * period, period)
    return (x, np.tensordot(w, u, axes=1), x_ref,
            np.tensordot(w, _reference_stack(run, s, x_ref), axes=1))


def plate_cycle(study):
    """The earthquake cycle of a dip-slip fault that is LOCKED between
    earthquakes: its surface velocity early and late in the cycle against
    the plane-strain reference, measured from steady slip on the same fault
    and in units of the slip rate.

    The cycle's MEAN is elastic back slip whatever the substrate does
    (``Prony.locked_rate_weights``), so relaxation is the difference between
    the early and the late curve and the black one between them. The third
    panel follows two points through the cycle, with and without gravity
    when the study swept it: the far hanging wall, which just after an
    earthquake moves the way the earthquake moved it, against the mean, and
    the surface over the fault's lower edge, which sinks and then rises.

    The period is the fit's window over ``CYCLES``: a locked fault's velocity
    needs the response for a few periods and no longer.
    """
    import matplotlib.pyplot as plt
    from mbem import laplace
    set_paper_style()
    _in_space(study)

    def axis(key, **kw):
        return sorted(study.axis(key), key=float, **kw) \
            if key in study.swept else [None]

    boxes, faces = axis("half_x"), axis("refine")
    contrast, grav = _contrast(study)

    def pair(g):
        sel = {k: v for k, v in (("half_x", boxes[-1]), (contrast, g))
               if v is not None}
        if faces[-1] is None:
            return (study.one(**sel) if sel else study.rows[0][1]), None
        return (study.one(refine=faces[-1], **sel),
                study.one(refine=faces[0], **sel) if len(faces) > 1 else None)

    run, coarse = pair(grav[0])
    period = float(f"{laplace.window(in_plane(run)[0])[1] / CYCLES:.3g}")
    span = signed_bins(run)[-1]
    xf = np.linspace(-span, span, 241)
    xf = xf[np.abs(xf) > 0.5]

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.6, 3.5))
    x, v, _xf, ref = in_plane_locked(run, PHASES, period, xf, coarse)
    _x, u, _xf, back = in_plane_in_time(run, (0.0,), xf, coarse=coarse)
    for ax, comp, lab, letter in ((a0, 0, r"$v_x\,/\,V$", "a"),
                                  (a1, 1, r"$v_z\,/\,V$", "b")):
        ax.plot(xf, -back[0, :, comp], "-", color="k", lw=0.8)
        ax.plot(x, -u[0, :, comp], "o", color="0.8", ms=3.4, mec="k", mew=0.3,
                label="mean: back slip")
        for k, (color, when) in enumerate(zip((ORANGE, BLUE), ("early", "late"))):
            ax.plot(xf, ref[k, :, comp], "-", color=color, lw=0.8)
            ax.plot(x, v[k, :, comp], "o", color=color, ms=3.4, mec="k",
                    mew=0.3, label=rf"{when}, $t = {PHASES[k]:g}\,T$")
        lo = min(v[:, :, comp].min(), ref[:, :, comp].min(), -back[0, :, comp].max())
        hi = max(v[:, :, comp].max(), ref[:, :, comp].max(), -back[0, :, comp].min())
        bottom, top = np.floor(lo * 5.0) / 5.0, np.ceil(hi * 5.0) / 5.0
        ax.set_xlim(-span, span)
        ax.set_xticks([-span, 0, span])
        ax.set_ylim(bottom, top)
        ax.set_yticks([bottom, 0.0, top])
        ax.set_xlabel(r"$x$ (km)")
        ax.set_ylabel(lab)
        ax.set_box_aspect(1)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.text(0.95, 0.95, letter, transform=ax.transAxes, ha="right", va="top")
    a0.legend(frameon=False, loc="upper left", handletextpad=0.2, fontsize=8)
    name = rf"$T = {period:g}\,\tau$"
    if boxes[-1] is not None:
        name += f", half-width {float(boxes[-1]):g} km"
    a1.text(0.05, 0.95, name + ("\ntwo-mesh limit" if coarse is not None else ""),
            transform=a1.transAxes, ha="left", va="top", fontsize=8)

    # two points through the cycle: the far hanging wall (horizontal) and
    # the surface over the fault's lower edge (vertical)
    gp = run.resolved["spec"]["model"]["geometry"]["params"]
    edge = gp["fault_depth"] / np.tan(np.deg2rad(gp["dip"]))
    sites = ((int(np.argmin(np.abs(x - edge - X_FAR))), 0, "k", r"$v_x$"),
             (int(np.argmin(np.abs(x - edge))), 1, "0.55", r"$v_z$"))
    phase = np.linspace(0.02, 0.98, 49)
    dots = phase[::4]
    lo = hi = 0.0
    for g, ls, mfc in list(zip(grav, ("-", "--"), (None, "white")))[::-1]:
        run, coarse = pair(g)                   # gravity last: drawn on top
        _x, v, _xr, _ref = in_plane_locked(run, dots, period, coarse=coarse)
        for site, comp, color, lab in sites:
            _x, _v, _xr, ref = in_plane_locked(run, phase, period,
                                               x[site:site + 1], coarse)
            a2.plot(phase, ref[:, 0, comp], ls, color=color, lw=0.8)
            a2.plot(dots, v[:, site, comp], "o", color=color, ms=3.4,
                    mfc=color if mfc is None else mfc, mec="k", mew=0.3,
                    label=lab + rf", $x = {x[site]:.0f}$ km"
                    if g == grav[0] else None)
            lo = min(lo, v[:, site, comp].min(), ref[:, 0, comp].min())
            hi = max(hi, v[:, site, comp].max(), ref[:, 0, comp].max())
    bottom, top = np.floor(lo * 5.0) / 5.0, np.ceil(hi * 5.0) / 5.0
    a2.set_xlim(0.0, 1.0)
    a2.set_xticks([0.0, 0.5, 1.0])
    a2.set_ylim(bottom, top)
    a2.set_yticks([bottom, 0.0, top])
    a2.set_xlabel(r"$t\,/\,T$")
    a2.set_ylabel(r"$v\,/\,V$")
    a2.set_box_aspect(1)
    a2.axhline(0.0, color="0.5", lw=0.6)
    a2.legend(frameon=False, loc="lower right", handletextpad=0.2, fontsize=8)
    if len(grav) > 1:
        a2.text(0.05, 0.95, "open, dashed: " + _named(contrast, grav[1]),
                transform=a2.transAxes, ha="left", va="top", fontsize=8)
    a2.text(0.95, 0.95, "c", transform=a2.transAxes, ha="right", va="top")

    fig.tight_layout(w_pad=1.6)
    out = save_figure(fig, study.study_dir, "plate_cycle")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# a lower reach that creeps: afterslip
# ---------------------------------------------------------------------

def creep_on(run):
    """``(s, centroids, areas, slip)``: the SOLVED lower reach of a run with
    ``lower="creep"``, its slip per state ``(n_states, n_elements)`` in the
    sense of the rupture above it and over that rupture's slip."""
    from mbem import laplace
    from mbem.cases import layered_box as lb

    s, stack = laplace.run_stack(run.run_dir)
    mp = run.resolved["spec"]["model"]["params"]
    tri = run.meshes["fault_lower__v"][run.meshes["fault_lower__t"]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    sense = lb.burgers(1.0, mp.get("rake", 180.0), run.meshes["arr__n_hat"],
                       run.meshes["arr__s_hat"])
    return (s, tri.mean(axis=1), area,
            stack["b:fault_lower"] @ sense / mp.get("slip_mag", 1.0e-3))


def plate_creep(study):
    """Afterslip: a megathrust whose lower reach, in the substrate, creeps
    (``lower="creep"``, ``rheology.ViscousFault``), for each ``creep_tau`` the
    study swept.

    Three panels. The slip of the creeping reach against depth at mid-strike,
    at three times, for the fastest law: it is largest against the rupture's
    lower edge, where uniform slip stops, and the mollification width is what
    bounds it there. Its mean against time, one curve a law. And two points
    of the surface through time, each law against the layered reference,
    which knows the rupture and the relaxation and no afterslip: the gap
    between them is what the creep adds.

    A study that also swept ``refine`` is drawn on its finest mesh, not as a
    two-mesh limit: these are dense runs, their pairs are close in size, and
    a limit of meshes in the ratio r magnifies what they do not share by
    (r + 1) / (r - 1).
    """
    import matplotlib.pyplot as plt
    from mbem import laplace
    set_paper_style()
    _in_space(study)

    taus = sorted(study.axis("creep_tau"), key=float) \
        if "creep_tau" in study.swept else [None]
    faces = sorted(study.axis("refine"), key=float) \
        if "refine" in study.swept else [None]

    def run_of(tau, refine=faces[-1]):
        sel = {k: v for k, v in (("creep_tau", tau), ("refine", refine))
               if v is not None}
        return study.one(**sel) if sel else study.rows[0][1]


    first = run_of(taus[0])
    gp = first.resolved["spec"]["model"]["geometry"]["params"]
    lo_t, hi_t = laplace.window(creep_on(first)[0])
    late = float(f"{hi_t:.3g}")
    when = [t for t in (1.0, 10.0) if lo_t <= t < late] + [late]
    colors = (BLUE, ORANGE, "0.45", "k")

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10.6, 3.5))
    s, cen, area, slip = creep_on(first)
    mid = np.abs(cen[:, 1]) < 0.25 * gp["fault_half_len"]
    depth = -cen[:, 2]
    edges = np.linspace(depth.min() - 1e-9, depth.max() + 1e-9, 9)
    w = laplace.Prony(s).weights(np.array(when))
    top = 0.0
    for k, (t, shade) in enumerate(zip(when, ("0.75", "0.45", "k"))):
        z, b = [], []
        for a, c in zip(edges[:-1], edges[1:]):
            sel = mid & (depth >= a) & (depth < c)
            if sel.any():
                z.append(depth[sel].mean())
                b.append(float(w[k] @ slip[:, sel].mean(axis=1)))
        a0.plot(z, b, "-o", color=shade, ms=3.4, mec="k", mew=0.3, lw=0.8,
                label=rf"$t = {t:g}\,\tau$")
        top = max(top, max(b))
    top = np.ceil(top * 10.0) / 10.0
    a0.set_xlim(edges[0], edges[-1])
    a0.set_xticks([round(edges[0]), round(edges[-1])])
    a0.set_ylim(0.0, top)
    a0.set_yticks([0.0, top])
    a0.set_xlabel("depth (km)")
    a0.set_ylabel("afterslip / slip")
    a0.legend(frameon=False, loc="lower left", fontsize=8)

    def law(tau):
        return "held" if float(tau) >= NO_CREEP \
            else rf"$\tau_c = {float(tau):g}\,\tau$"

    t = np.geomspace(lo_t, late, 41)
    top = 0.0
    edge = gp["fault_depth"] / np.tan(np.deg2rad(gp["dip"]))
    lo = hi = 0.0
    for tau, color in zip(taus, colors):
        run = run_of(tau)
        s, cen, area, slip = creep_on(run)
        mean = laplace.Prony(s).weights(t) @ (slip @ area / area.sum())
        a1.plot(t, mean, "-", color=color, lw=1.0,
                label=None if tau is None else law(tau))
        top = max(top, float(mean.max()))
        x = in_plane(run)[1]
        for site, comp, ls in ((int(np.argmin(np.abs(x - edge - X_FAR))), 0, "-"),
                               (int(np.argmin(np.abs(x - edge))), 1, "--")):
            _x, u, _xr, ref = in_plane_in_time(run, np.concatenate([[0.0], t]),
                                               x[site:site + 1])
            post, want = u[1:, site, comp] - u[0, site, comp], \
                ref[1:, 0, comp] - ref[0, 0, comp]
            if tau == taus[0]:
                a2.plot(t, want, ls, color="k", lw=0.8,
                        label=rf"$u_{'xz'[comp]}$, $x = {x[site]:.0f}$ km")
            a2.plot(t, post, ls, color=color, lw=1.0)
            lo, hi = min(lo, post.min(), want.min()), max(hi, post.max(), want.max())
    top = np.ceil(top * 10.0) / 10.0
    a1.set_ylim(0.0, top)
    a1.set_yticks([0.0, top])
    a1.set_ylabel("mean afterslip / slip")
    if taus[0] is not None:
        a1.legend(frameon=False, loc="upper left", fontsize=8)
    bottom, top = np.floor(lo * 10.0) / 10.0, np.ceil(hi * 10.0) / 10.0
    a2.set_ylim(bottom, top)
    a2.set_yticks([bottom, 0.0, top])
    a2.axhline(0.0, color="0.5", lw=0.6)
    a2.set_ylabel(r"$[u(t) - u(0^+)]\,/\,s$")
    a2.legend(frameon=False, loc="upper left", fontsize=8)
    a2.text(0.05, 0.05, "black: the reference, no afterslip",
            transform=a2.transAxes, ha="left", va="bottom", fontsize=8)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlim(lo_t, late)
        ax.set_xticks([lo_t, late])
        ax.set_xticklabels([f"{lo_t:g}", f"{late:g}"])
        ax.minorticks_off()
        ax.set_xlabel(r"$t\,/\,\tau$")
    for ax, letter in ((a0, "a"), (a1, "b"), (a2, "c")):
        ax.set_box_aspect(1)
        ax.text(0.95, 0.95, letter, transform=ax.transAxes, ha="right", va="top")

    fig.tight_layout(w_pad=1.6)
    out = save_figure(fig, study.study_dir, "plate_creep")
    plt.close(fig)
    return out
