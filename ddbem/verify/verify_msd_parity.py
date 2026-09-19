"""Gate: ddbem's P0 solver reproduces msd's solver EXACTLY on the same problem.

    python verify/verify_msd_parity.py          # from the ddbem root (needs numba)

WHY THIS IS AN ENTRYWISE GATE AND NOT A FIELD COMPARISON
========================================================
msd (``msd/mbem``) solves the direct Somigliana BIE with ``u`` and ``t`` as
boundary unknowns.  ddbem solves a DD representation with one density ``q``.
Those look like different methods -- but on a body whose ENTIRE boundary is
traction-free they are the same equation written twice, and the dictionary is
exact:

* zero boundary traction makes the whole EXTERIOR field of the representation
  vanish, so ``u_exterior = 0`` (ddbem's ``neumann_row="exterior"``) is the same
  statement as msd's ``(1/2 I + sigma H) u = -H_f slip``;
* the densities are ``q_p = -sigma_p u_p`` on the boundary and
  ``q_fault = -slip_msd`` on the fault (msd's fault slip carries the opposite
  sign to clq's ``Delta u = u(+nhat) - u(-nhat)``, which is visible in msd's own
  representation formula ``u = ... - sum_f H_xf slip_f``);
* substituting gives ``A_ddbem == A_msd`` entrywise, with ``b_ddbem == -b_msd``
  (the row is multiplied through by -1 when ``u`` is replaced by ``-q``).

So this gate is machine precision, not discretisation accuracy.  It covers BOTH
jump conventions -- ``half`` (the analytic 1/2) and ``calibrated`` (msd's
``C_q = -sum_p sigma rowsum H_qp``, which the ddbem derivation reproduces as
``F = sigma (R - target)``) -- at nu = 0.25, 0.30 and 0.45, because the lam/mu
pairing of the DD kernel is invisible at nu = 1/4.

WHAT IS NOT COMPARED, AND WHY
-----------------------------
A model with a PRESCRIBED_DISPLACEMENT patch.  msd handles it with a traction
unknown and the Somigliana single layer ``G``; ddbem has one density and states
``u_int(x_c) = u_bar`` instead.  Those are genuinely different formulations, so
there is no entrywise correspondence to gate -- ``verify_solver.py`` gates the
ddbem Dirichlet rows against an exact solution instead.

The rigid-motion constraint differs too: msd's ``deflate=True`` borders with 3
translations, ddbem defaults to all 6 rigid modes.  The gate therefore runs
ddbem with ``constrain="translations"`` for the solution comparison, and
reports what the 6-mode choice changes.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from _common import MOSS_ORG, NU_SWEEP, Report                  # noqa: E402

import ddbem                                                    # noqa: E402

sys.path.insert(0, str(MOSS_ORG / "msd"))

MU = 1.0
SLIP = np.array([0.001, 0.0, 0.0])
OBS = np.array([[0.0, 0.0, 0.8], [0.5, 0.4, -0.6],
                [-0.7, 0.2, 0.3], [0.1, -0.9, 0.4]])


def build(nu, jump, nsub=1):
    """The same free-traction sphere + fault problem, both ways."""
    import mollified_bem as mb
    from mbem.backends.dense import DenseBackend
    from mbem.evaluate import evaluate_displacement
    from mbem.model.core import BCType as MBC
    from mbem.model.core import Patch as MPatch
    from mbem.model.core import Region as MRegion
    from mbem.model.core import RegionModel

    v, t = ddbem.icosphere(nsub, radius=2.0)
    tv = ddbem.tri_verts(v, t)
    fv, ft = ddbem.rectangle([-0.4, -0.3, 0.0], [0.8, 0, 0], [0, 0.6, 0], 2, 2)
    ftv = ddbem.tri_verts(fv, ft)
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    epsf = 0.3 * float(ddbem.element_h(ftv).mean())

    pb = ddbem.Patch("sph", tv, ddbem.BCType.FREE_TRACTION, eps=eps)
    pf = ddbem.Patch("fault", ftv, ddbem.BCType.FAULT, value=-SLIP, eps=epsf)
    m = ddbem.Model([pb, pf], MU, nu, order=0, jump=jump, neumann_row="exterior")
    S = m.assemble()

    lam = 2.0 * MU * nu / (1.0 - 2.0 * nu)
    mat = mb.ElasticMaterial(mu=MU, lam=lam)
    P = MPatch("sph", mb.TriMesh(vertices=v.copy(), triangles=t.copy()),
               MBC.FREE_TRACTION)
    F = MPatch("fault", mb.TriMesh(vertices=fv.copy(), triangles=ft.copy()),
               MBC.FAULT, value=np.broadcast_to(SLIP, (ftv.shape[0], 3)).copy())
    R = MRegion("body", mat, [P], np.zeros(3), faults=[F])
    RM = RegionModel([R])
    RM.validate()
    from mbem.model.equations import generate_system
    back = DenseBackend(mode="direct", jump=jump, deflate=True).assemble(
        generate_system(RM), {"sph": eps, "fault": epsf})
    out = back.solve()
    um = evaluate_displacement(RM, R, out, OBS, {"sph": eps, "fault": epsf},
                               warn_near=False)
    return m, S, back, out, um


def main() -> int:
    rep = Report("ddbem P0 == msd's solver (free-traction sphere + fault)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nu in NU_SWEEP:
            for jump in ("half", "calibrated"):
                m, S, back, out, um = build(nu, jump)
                sA = float(np.max(np.abs(back.A)))
                sb = float(np.max(np.abs(back.b)))
                rep.check(f"nu={nu} jump={jump:11s} matrix A entrywise",
                          float(np.max(np.abs(S.A - back.A))) / sA, 1e-12)
                rep.check(f"nu={nu} jump={jump:11s} rhs b (b_ddbem == -b_msd)",
                          float(np.max(np.abs(S.b + back.b))) / sb, 1e-12)

                sol = m.solve(S, constrain="translations")
                u_msd = out["u:sph"]
                rep.check(f"nu={nu} jump={jump:11s} density (q == -u_msd)",
                          float(np.max(np.abs(sol.q.reshape(-1, 3) + u_msd)))
                          / float(np.max(np.abs(u_msd))), 1e-10,
                          f"cond {sol.cond:.1e} / msd {back.report.cond_estimate:.1e}")
                ud = sol.displacement(OBS)
                rep.check(f"nu={nu} jump={jump:11s} interior displacement field",
                          float(np.max(np.abs(ud - um)) / np.max(np.abs(um))),
                          1e-10)

                sol6 = m.solve(S, constrain="rigid")
                d6 = float(np.max(np.abs(sol6.displacement(OBS) - um))
                           / np.max(np.abs(um)))
                rep.note(f"nu={nu} jump={jump}: constrain='rigid' (6 modes, "
                         f"ddbem's default) moves the field by {d6:.2e} "
                         f"relative to msd's 3-translation deflation -- the "
                         f"rotation msd leaves in")
    return 0 if rep.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
