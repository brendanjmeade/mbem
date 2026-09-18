"""Gate: the checks the seeded-defect sweep proved were MISSING.

    python verify/verify_coverage.py          # from the ddbem root

WHY THIS FILE EXISTS, AND WHY IT IS SEPARATE
============================================
``bench/seeded_defects.py`` broke ddbem and clq on purpose, seventeen ways, and
ran the stage-1/2 gates against each sandbox.  Fourteen defects were caught.
Three were not:

  * the Voigt row order of the public stress API (``yz`` and ``xz`` swapped in
    ``defaults.VOIGT_PAIRS``) -- every gate converts both sides of every stress
    comparison with ddbem's OWN ``tensor_to_voigt``, so the two cancel, and
    ``verify_solver.py`` computes an interior stress error but never gates it;
  * the ``FAR_FIELD`` default (``hybrid`` -> ``analytic``) -- every observation
    set in ``verify/`` sits within ~4 element lengths, where clq's two producers
    agree to 0.0, so nothing notices that they do NOT agree further out
    (2.0e-4 at 1e4 L, 0.86 at 1e5 L, measured);
  * anything about ``Patch(orientation=-1)`` -- the only inward-normal test in
    ``verify_solver.py`` is that a FLIPPED MESH at the default ``+1`` raises, so
    deleting ``orientation`` from the free term changes nothing any gate sees.

Two further gaps were found by reading: a **fault patch is never solved above
P0** by any gate, and a **callable FAULT value** is never exercised at all.

They are gated here rather than by editing the stage-1/2 scripts, so that the
files the seeded-defect sweep was run against stay exactly as they were audited.
Each check below carries a TRIPWIRE that must break when the thing it pins is
broken -- otherwise this file would repeat the mistake it exists to fix.
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

from _common import (MU, Report, gauss_triangle, msd_analytical,   # noqa: E402
                     test_triangles)
from _exact import kelvin_displacement, kelvin_stress             # noqa: E402

import ddbem                                                      # noqa: E402
from ddbem._clq import CLQ as clq                                 # noqa: E402

NU = 0.30
EPS = 0.15
X0 = np.array([3.0, 1.2, -0.7])
FORCE = np.array([1.0, -0.5, 0.8])
OBS = np.array([[0.0, 0.0, 0.0], [0.2, 0.1, -0.1], [-0.15, 0.2, 0.1]])


def _voigt_by_hand(s):
    """(N,3,3) -> (N,6) written out, NOT through ddbem.tensor_to_voigt.

    The whole point: a gate that converts both sides with the library's own
    converter cannot see the library's convention change.
    """
    return np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2],
                     s[:, 1, 2], s[:, 0, 2], s[:, 0, 1]], axis=1)


# ---------------------------------------------------------------------------
# 1.  the Voigt row order of the public stress API
# ---------------------------------------------------------------------------

def gate_voigt_order(rep: Report) -> None:
    """Rows of ``stress_matrix`` are (xx, yy, zz, yz, xz, xy) -- checked against
    a hand-written stack, so ddbem's own converter cannot cancel the error."""
    tri = test_triangles()[:2]
    obs = np.array([[0.5, 0.4, 0.7], [-0.2, 0.3, -0.4], [1.1, 1.3, -0.35]])
    rng = np.random.default_rng(11)
    for order in (0, 1, 2):
        K = ddbem.n_nodes(order)
        slip = rng.normal(size=(tri.shape[0] * K, 3)).reshape(-1)
        S = ddbem.stress_matrix(obs, tri, EPS, MU, NU, order,
                                subtract_eigenstress=False)
        got = (S @ slip).reshape(-1, 6)
        # independent reference: assemble the FULL tensor per source triangle
        # from clq directly and contract, then flatten by hand
        ref = np.zeros((obs.shape[0], 3, 3))
        for s in range(tri.shape[0]):
            inf = clq.influence(obs, tri[s], MU, NU, EPS, order=order,
                                want=("H",))
            ref += np.einsum("nkmlj,kj->nml", inf.H,
                             slip[3 * K * s:3 * K * (s + 1)].reshape(K, 3))
        rep.check(f"P{order}: stress_matrix rows are (xx,yy,zz,yz,xz,xy)",
                  float(np.max(np.abs(got - _voigt_by_hand(ref))))
                  / float(np.max(np.abs(ref))), 1e-13)
    # tripwire: swapping two rows must break it, i.e. the check is not vacuous
    S = ddbem.stress_matrix(obs, tri, EPS, MU, NU, 0, subtract_eigenstress=False)
    got = (S @ np.ones(S.shape[1])).reshape(-1, 6)
    swapped = got[:, [0, 1, 2, 4, 3, 5]]
    rep.check_bool("  TRIPWIRE swapping the yz and xz rows breaks it",
                   float(np.max(np.abs(swapped - got)))
                   / float(np.max(np.abs(got))) > 1e-3,
                   f"(moves it by {float(np.max(np.abs(swapped - got))) / float(np.max(np.abs(got))):.2f} relative)")
    rep.check_bool("voigt_to_tensor . tensor_to_voigt round-trips the HAND form",
                   float(np.max(np.abs(
                       ddbem.voigt_to_tensor(_voigt_by_hand(
                           kelvin_stress(OBS, X0, FORCE, MU, NU)))
                       - kelvin_stress(OBS, X0, FORCE, MU, NU)))) < 1e-15)
    # End to end, and NOT circular: the traction path never touches
    # VOIGT_PAIRS.  t_i = sigma_ij n_j, so the traction on a facet with n = e_x
    # has third component sigma_xz and the one with n = e_y has sigma_yz --
    # two components read out of a completely different contraction.
    tv = ddbem.tri_verts(*ddbem.icosphere(1))
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    p = ddbem.Patch("s", tv, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                    value=lambda x: kelvin_displacement(x, X0, FORCE, MU, NU),
                    eps=eps)
    sol = ddbem.Model([p], MU, NU, order=0).solve()
    sig = sol.stress(OBS)
    tx = sol.traction(OBS, np.array([1.0, 0.0, 0.0]))
    ty = sol.traction(OBS, np.array([0.0, 1.0, 0.0]))
    scale = float(np.max(np.abs(sig)))
    rep.check("Solution.stress row 4 is xz (against the traction path)",
              float(np.max(np.abs(sig[:, 4] - tx[:, 2]))) / scale, 1e-12)
    rep.check("Solution.stress row 3 is yz (against the traction path)",
              float(np.max(np.abs(sig[:, 3] - ty[:, 2]))) / scale, 1e-12)
    swapped = float(np.max(np.abs(sig[:, 3] - tx[:, 2]))) / scale
    rep.check_bool("  TRIPWIRE reading row 3 as xz fails that by orders of "
                   "magnitude", swapped > 1e-3,
                   f"({swapped:.2e} against a 1e-12 tolerance)")


# ---------------------------------------------------------------------------
# 2.  the far field
# ---------------------------------------------------------------------------

def _far_reference(tri, order, obs, mu, nu, eps, n_quad=6):
    """Nodal slip -> displacement by quadrature of msd's frozen POINT kernel.

    Independent of clq's closed form AND of its far-field switch: a Gauss rule
    on the triangle times the traction operator written out here.  At the
    distances used below (1e3-1e4 element lengths) a 6-point rule is exact to
    far beyond the tolerance -- the gate prints the 4-point vs 6-point residual
    so that cannot be taken on trust.
    """
    ak = msd_analytical()
    y, w = gauss_triangle(tri, n_quad)
    Nq = np.ones((y.shape[0], 1)) if order == 0 else None
    if order != 0:
        from _common import shape_independent
        Nq = shape_independent(tri, order, y)
    nhat = ddbem.mesh.unit_normal(tri)
    lam = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    out = np.zeros((obs.shape[0], Nq.shape[1], 3, 3))
    for q in range(y.shape[0]):
        wk = w[q] * Nq[q]
        for n in range(obs.shape[0]):
            DG = ak.kelvin_dG_pointwise(obs[n] - y[q], mu, nu, eps)
            U = np.zeros((3, 3))
            for i in range(3):
                tr = DG[i, 0, 0] + DG[i, 1, 1] + DG[i, 2, 2]
                for j in range(3):
                    U[i, j] = -(mu * (nhat @ DG[i, j]) + lam * nhat[j] * tr
                                + mu * (nhat @ DG[i, :, j]))
            out[n] += wk[:, None, None] * U
    return out


def gate_far_field(rep: Report) -> None:
    """clq's ``hybrid`` producer is right where the closed form loses digits."""
    tri = test_triangles()[0]
    L = float(np.max(np.linalg.norm(tri - tri.mean(axis=0), axis=1)))
    direction = np.array([[0.3, -0.5, 0.81240384], [-0.6, 0.7, 0.39],
                          [0.1, 0.2, 0.97]])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    for scale in (1e3, 1e4):
        obs = tri.mean(axis=0) + scale * L * direction
        ref = _far_reference(tri, 0, obs, MU, NU, EPS, 6)
        ref4 = _far_reference(tri, 0, obs, MU, NU, EPS, 4)
        resid = float(np.max(np.abs(ref4 - ref)) / np.max(np.abs(ref)))
        A = ddbem.displacement_matrix(obs, tri[None], EPS, MU, NU, 0)
        got = A.reshape(-1, 3, 3)
        d_h = float(np.max(np.abs(got - ref[:, 0])) / np.max(np.abs(ref)))
        # tolerance floor 1e-13, not 1e-11: the measured value on correct code
        # is 2e-16 and the quadrature's own residual is 2e-16, so 1e-13 is a
        # 500x margin on a machine-precision statement -- and it is what makes
        # the seeded far_field defect fail by 178x at 1e3 L rather than 1.8x.
        rep.check(f"far field at {scale:.0e} L: default producer vs point-kernel "
                  f"quadrature", d_h, max(30.0 * resid, 1e-13),
                  f"[quad resid {resid:.1e}]")
        B = ddbem.displacement_matrix(obs, tri[None], EPS, MU, NU, 0,
                                      far_field="analytic")
        d_a = float(np.max(np.abs(B.reshape(-1, 3, 3) - ref[:, 0]))
                    / np.max(np.abs(ref)))
        rep.check_bool(f"  TRIPWIRE far_field='analytic' is worse there",
                       d_a > 100.0 * max(d_h, 1e-16),
                       f"hybrid {d_h:.2e} vs analytic {d_a:.2e} "
                       f"({d_a / max(d_h, 1e-300):.0e}x)")
    rep.check_bool("defaults.FAR_FIELD is the producer that survives that",
                   ddbem.defaults.FAR_FIELD == "hybrid",
                   f"(FAR_FIELD = {ddbem.defaults.FAR_FIELD!r})")


# ---------------------------------------------------------------------------
# 3.  orientation = -1
# ---------------------------------------------------------------------------

def gate_orientation(rep: Report) -> None:
    """A patch stored with INWARD normals and ``orientation=-1`` is the same model.

    This is what catches a free term that ignores ``sigma``: with
    ``Fq = (R - target)`` instead of ``sigma (R - target)`` the flipped model
    gets the wrong sign and diverges from the outward one, while every existing
    gate (all of which use ``orientation=+1``) still passes.
    """
    tv = ddbem.tri_verts(*ddbem.icosphere(1))
    eps = 0.3 * float(ddbem.element_h(tv).mean())
    ue = kelvin_displacement(OBS, X0, FORCE, MU, NU)
    for order in (0, 1):
        got = {}
        for tag, TV, orient in (("outward +1", tv, +1),
                                ("inward  -1", ddbem.shapes.flip(tv), -1)):
            p = ddbem.Patch("s", TV, ddbem.BCType.PRESCRIBED_DISPLACEMENT,
                            value=lambda x: kelvin_displacement(x, X0, FORCE,
                                                                MU, NU),
                            eps=eps, orientation=orient)
            m = ddbem.Model([p], MU, NU, order=order, jump="calibrated")
            sol = m.solve()
            got[tag] = (sol.displacement(OBS), sol.q, m.assemble().free_term)
        d = (float(np.max(np.abs(got["outward +1"][0] - got["inward  -1"][0])))
             / float(np.max(np.abs(ue))))
        rep.check(f"P{order}: orientation=-1 gives the same interior field", d,
                  1e-12,
                  f"(error vs exact: "
                  f"{float(np.max(np.abs(got['outward +1'][0] - ue)) / np.max(np.abs(ue))):.2e})")
        # The density must flip sign with the stored normal.  ``shapes.flip``
        # swaps vertices 2 and 3, so at P1 the local NODES are permuted as well
        # (v1, v3, v2) -- comparing index for index without that permutation is
        # a harness bug, not a code bug.
        perm = [0] if order == 0 else [0, 2, 1]
        K = ddbem.n_nodes(order)
        q_out = got["outward +1"][1].reshape(-1, K, 3)
        q_in = got["inward  -1"][1].reshape(-1, K, 3)[:, perm]
        rep.check("  and the density flips sign with the stored normal",
                  float(np.max(np.abs(q_out + q_in)))
                  / float(np.max(np.abs(q_out))), 1e-10)
        f_out, f_in = got["outward +1"][2], got["inward  -1"][2]
        rep.check_bool("  TRIPWIRE the free term is NOT the same for the two "
                       "(sigma is load-bearing)",
                       float(np.max(np.abs(f_out - f_in)))
                       / float(np.max(np.abs(f_out))) > 0.5,
                       f"(differ by {float(np.max(np.abs(f_out - f_in))) / float(np.max(np.abs(f_out))):.2f} relative)")


# ---------------------------------------------------------------------------
# 4.  a FAULT patch above P0, and a callable fault value
# ---------------------------------------------------------------------------

def _fault_mesh(n):
    return ddbem.tri_verts(*ddbem.rectangle([-0.4, -0.3, 0.0], [0.8, 0, 0],
                                            [0, 0.6, 0], n, n))


def gate_fault_higher_order(rep: Report) -> None:
    """The FAULT path at P1/P2, and a callable (spatially varying) slip."""
    def slip_fn(x):
        return np.stack([1e-3 * (1.0 + 0.5 * x[:, 0]),
                         2e-4 * x[:, 1] - 1e-4,
                         np.zeros(x.shape[0])], axis=1)

    ftv = _fault_mesh(2)
    # (a) the nodal density is the callable sampled at clq's OWN nodes
    for order in (0, 1, 2):
        K = ddbem.n_nodes(order)
        p = ddbem.Patch("f", ftv, ddbem.BCType.FAULT, value=slip_fn, eps=0.1)
        dens = p.nodal_density(order)
        man = np.concatenate([slip_fn(clq.nodes(ftv[s], order)).reshape(-1)
                              for s in range(ftv.shape[0])])
        rep.check(f"P{order} fault: nodal_density == the callable at clq.nodes",
                  float(np.max(np.abs(dens - man))), 1e-15,
                  f"({3 * K * ftv.shape[0]} values)")
    # a constant-value fault must agree with the callable that returns it
    pc = ddbem.Patch("f", ftv, ddbem.BCType.FAULT, value=[1e-3, 0.0, 0.0],
                     eps=0.1)
    pk = ddbem.Patch("f", ftv, ddbem.BCType.FAULT,
                     value=lambda x: np.tile([1e-3, 0.0, 0.0], (x.shape[0], 1)),
                     eps=0.1)
    rep.check("P2 fault: constant value == the equivalent callable",
              float(np.max(np.abs(pc.nodal_density(2) - pk.nodal_density(2)))),
              1e-18)

    # (b) end to end: a LINEAR slip field, which P1 represents exactly on each
    #     element, against the same field carried by a 4x refined P0 fault.
    #     They are different discretisations of the SAME source, so they agree
    #     to the P0 mesh's own O(h) error -- the tolerance is set above the
    #     measured value and the value is printed.
    tv = ddbem.tri_verts(*ddbem.icosphere(1, radius=2.0))
    eps_b = 0.3 * float(ddbem.element_h(tv).mean())
    obs = np.array([[0.0, 0.0, 0.9], [0.6, 0.5, -0.7], [-0.8, 0.3, 0.4]])
    fields = {}
    for tag, mesh, order in (("P1, 8 elements", _fault_mesh(2), 1),
                             ("P2, 8 elements", _fault_mesh(2), 2),
                             ("P0, 128 elements", _fault_mesh(8), 0)):
        pb = ddbem.Patch("s", tv, ddbem.BCType.FREE_TRACTION, eps=eps_b)
        pf = ddbem.Patch("f", mesh, ddbem.BCType.FAULT, value=slip_fn,
                         eps=0.3 * float(ddbem.element_h(_fault_mesh(2)).mean()),
                         order=order)
        m = ddbem.Model([pb, pf], MU, NU, order=0, jump="calibrated")
        fields[tag] = m.solve().displacement(obs)
    ref = fields["P0, 128 elements"]
    for tag in ("P1, 8 elements", "P2, 8 elements"):
        d = float(np.max(np.abs(fields[tag] - ref)) / np.max(np.abs(ref)))
        rep.check(f"fault with a varying slip: {tag} vs the refined P0 source",
                  d, 0.05,
                  "(ceiling 5 %; this is a CONSISTENCY check between two "
                  "discretisations of one source, not an accuracy claim)")
    rep.note("the two coarse orders agree with each other to "
             f"{float(np.max(np.abs(fields['P1, 8 elements'] - fields['P2, 8 elements'])) / np.max(np.abs(ref))):.2e}, "
             "which is the part of the difference above that is NOT the P0 "
             "mesh's own error")


def main() -> int:
    rep = Report("ddbem coverage: what the seeded-defect sweep found unpinned")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate_voigt_order(rep)
        gate_far_field(rep)
        gate_orientation(rep)
        gate_fault_higher_order(rep)
    return 0 if rep.finish() else 1


if __name__ == "__main__":
    sys.exit(main())
