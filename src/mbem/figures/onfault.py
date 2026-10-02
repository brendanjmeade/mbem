"""On-fault elastic shear: against the classical TDE, and down the dip.

A TWO-AXIS study, swept over ``eps`` and the top patch's ``order``, because the
figure asks two questions that need different runs:

  (a) does the mollified on-fault shear approach the classical half-space TDE as
      eps falls? -- an eps ladder on one model;
  (b) what does the FIRST ELEMENT ROW below the free surface do? -- the P0 and P1
      tops at one eps, because a P0 staircase is the known failure there and a P1
      top is the cure.

The old demo solved the ladder and the extra P0 model in one process. Same
arithmetic, but the eps ladder and the P0/P1 contrast are different operators, so
as runs they each carry their own resolved.json.

TWO THINGS THIS FIGURE IS CAREFUL ABOUT, both inherited from the demo and both
easy to get wrong. The stress is the ELASTIC part, so the eigenstress of the
mollified double layer is subtracted (``subtract_anelastic=True``, the default) --
a stress presented as elastic that still carries it is simply the wrong quantity.
And the first element row sits inside the near-boundary band by construction, so
the evaluator's warning is switched off deliberately rather than by accident:
that row is exactly what panel (b) is about.
"""

from __future__ import annotations

import numpy as np

from mbem.figures import save_figure
from mbem.figures.style import set_paper_style

GPA_TO_MPA = 1.0e3
HS_C, BEM_C, P0_C = "#444444", "#1f5fa6", "#c1272d"


def resolved_shear(sig, n_hat, s_hat):
    """Shear on the fault plane resolved onto the slip direction."""
    return np.einsum("i,...ij,j->...", n_hat, sig, s_hat)


def _onfault_sets(fault, fault_depth, edge_fault):
    """The centre element and the mid-strike down-dip line.

    The deepest row is dropped: the tip stress concentration there is
    resolution-limited in every method, so it says nothing about mollification.
    The FIRST row is deliberately kept -- it is the question in panel (b).
    """
    c = fault.centroids()
    cz = c[:, 2]
    target = np.array([0.0, 0.0, -0.5 * fault_depth])
    centre = c[int(np.argmin(np.linalg.norm(c - target, axis=1)))][None]
    margin = 1.5 * edge_fault
    band = (np.abs(c[:, 1]) < edge_fault) & (cz > -(fault_depth - margin))
    idx = np.where(band)[0]
    idx = idx[np.argsort(cz[idx])]
    tv = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    first_row = tv[idx][:, :, 2].max(axis=1) > -1e-9      # a vertex on z = 0
    return centre, c[idx], first_row


def onfault_stress(study):
    """(a) centre shear against the TDE as eps falls; (b) the down-dip profile."""
    import matplotlib.pyplot as plt
    from mbem.cases.registry import topo_inclusion_meshes  # noqa: F401 (lazy)
    from mbem.evaluate import evaluate_stress
    from mbem.kernels import basis as kb
    from tde_reference import classical_tde_stress
    set_paper_style()

    eps_values = study.axis("eps")
    orders = study.axis("order_top")
    fine = max(orders)                       # the production top: P1 if present

    # Rebuild the geometry once from the finest-eps production run: the figure
    # needs the fault mesh and the model to evaluate stress at points that are
    # not a solved slot.
    from mbem import config as cfg
    probe = study.one(eps=eps_values[-1], order_top=fine)
    spec = probe.resolved["spec"]
    mesh_fn = cfg.resolve(spec["model"]["geometry"]["builder"], cfg.MESH_BUILDERS)
    bundle = mesh_fn(scale=spec["model"]["geometry"]["scale"],
                     **spec["model"]["geometry"]["params"])
    n_hat = np.asarray(bundle.arrays["n_hat"], float)
    s_hat = np.asarray(bundle.arrays["s_hat"], float)
    fault = bundle.meshes["fault"]
    depth = float(abs(np.asarray(fault.vertices)[:, 2].min()))
    edge = float(spec["model"]["geometry"]["params"].get("edge_fault", 4.0))
    centre, dd, first_row = _onfault_sets(fault, depth, edge)

    mat = probe.resolved["model"]["regions"][0]["material"]
    from mbem import ElasticMaterial
    m = ElasticMaterial(mu=mat["mu"], lam=mat["lam"])

    def ref_shear(obs, halfspace):
        sig = classical_tde_stress(obs, fault, _burgers(probe, bundle),
                                   m.mu, m.nu, halfspace=halfspace)
        return resolved_shear(sig, n_hat, s_hat) * GPA_TO_MPA

    def tau_at(run, obs):
        model_fn = cfg.resolve(run.resolved["spec"]["model"]["builder"],
                              cfg.MODEL_BUILDERS)
        mdl = model_fn(bundle, **run.resolved["spec"]["model"]["params"])
        sol = {k: v for k, v in next(iter(run.fields.values())).items()}
        sig = evaluate_stress(mdl, mdl.regions[0], sol, obs,
                              run.resolved["spec"]["model"]["eps"],
                              subtract_anelastic=True, warn_near=False)
        return resolved_shear(sig, n_hat, s_hat) * GPA_TO_MPA

    hs_centre = ref_shear(centre, True)[0]
    tau_centre = np.array([tau_at(study.one(eps=e, order_top=fine), centre)[0]
                           for e in eps_values])
    eps_axis = [float(e) for e in eps_values]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 4.3))
    ax0.axhline(hs_centre, color=HS_C, lw=1.0, ls="--", label="half-space TDE")
    ax0.plot(eps_axis, tau_centre, "s-", color=BEM_C, lw=1.4, ms=5,
             label="BEM (elastic)")
    ax0.set_xlabel(r"fault $\varepsilon$ (km)")
    ax0.set_ylabel(r"on-fault centre shear $\tau$ (MPa)")
    sp = max(abs(tau_centre.max() - tau_centre.min()), 1e-3)
    ax0.set_ylim(min(tau_centre.min(), hs_centre) - 1.2 * sp,
                 max(tau_centre.max(), hs_centre) + 1.2 * sp)
    ax0.set_box_aspect(1)
    ax0.legend(frameon=False, fontsize=8, loc="best")
    ax0.text(0.96, 0.06,
             f"$\\to$ half-space\n"
             f"({abs(tau_centre[-1] / hs_centre - 1) * 100:.1f}% at "
             f"$\\varepsilon$={eps_axis[-1]:g})",
             transform=ax0.transAxes, ha="right", va="bottom", fontsize=7,
             color="0.3")
    ax0.text(0.05, 0.95, "a", transform=ax0.transAxes, ha="left", va="top")

    z = -dd[:, 2]
    ax1.plot(ref_shear(dd, True), z, "--", color=HS_C, lw=1.0,
             label="half-space TDE")
    ax1.plot(ref_shear(dd, False), z, ":", color="0.6", lw=1.0,
             label="full-space TDE")
    for o in orders:
        run = study.one(eps=eps_values[-1], order_top=o)
        ax1.plot(tau_at(run, dd), z, "-", lw=1.4,
                 color=BEM_C if o == fine else P0_C,
                 label=f"BEM, P{o} top")
    ax1.invert_yaxis()
    ax1.set_xlabel(r"$\tau$ (MPa)")
    ax1.set_ylabel("depth (km)")
    ax1.set_box_aspect(1)
    ax1.legend(frameon=False, fontsize=7, loc="best")
    ax1.text(0.05, 0.95, "b", transform=ax1.transAxes, ha="left", va="top")
    n_first = int(first_row.sum())
    ax1.text(0.95, 0.95, f"first row: {n_first} element(s)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=7,
             color="0.35")

    fig.suptitle("On-fault elastic shear: mollified BEM against the classical "
                 "TDE", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = save_figure(fig, study.study_dir, "onfault_stress")
    plt.close(fig)
    return out


def _burgers(run, bundle):
    """The fault's Burgers vector, from the model the run actually built.

    Rebuilt rather than guessed: the sign convention is stated once, in
    ``model/core.py``, and a figure that wrote its own ``-slip * s_hat`` would be
    the second statement that goes wrong.
    """
    from mbem import config as cfg
    model_fn = cfg.resolve(run.resolved["spec"]["model"]["builder"],
                           cfg.MODEL_BUILDERS)
    mdl = model_fn(bundle, **run.resolved["spec"]["model"]["params"])
    return mdl.regions[0].faults[0].value_array()
