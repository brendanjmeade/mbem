"""verify_stepper.py -- a solved fault stepped in time (``mbem.stepper``).

The Laplace route needs a fault's law to be linear. The stepper does not: it
reduces the body to its solved faults at each state
(``AssembledDense.condense``), carries the body's relaxation as the fit's own
exponentials, and integrates the faults (docs/viscoelastic.md section 6). So
it is held to three things the Laplace route is not the judge of -- closed
forms in TIME that no fit enters, a law that is not linear, and its own
stability -- and to the Laplace route wherever both apply.

The anchor is the penny crack of ``verify_viscoelastic`` [o]: faces whose
MOBILITY is graded as sqrt(1 - r^2/a^2) keep Eshelby's elliptical slip under
ANY law of the traction, because that slip's traction is uniform. The peak
slip B then obeys one scalar equation, dB/dt = F(dtau - k_E B), which is in
closed form for a linear F and for an exponential one, and with the
material's two exact poles when the body relaxes. Continuous P2 at a/h = 8,
1,838 slip unknowns; the crack's own bias is 0.6 % in moment (the elastic
gate's), so a ratio sits in ``PENNY_BAND`` and what is asked of it is that it
is FLAT in time.

  [a] the reduction, on the anchor-C box (a fault inside a body with free and
      clamped faces), P0 and continuous P2: ``S b + L`` vanishes at the slip
      whose traction ``evaluate_fault_traction`` read, a different code path
      (7e-15, 2e-14 measured); a fault that drags, and one that does not,
      come out of ``Condensed.dragged`` as they do of the dense backend's own
      solve, slip and free surface alike (1e-14); an operator built with a
      drag is refused.
  [b] elastic body, linear law: one exponential with time eta0 / k_E. No fit
      anywhere, so the ratio is the crack's bias to four figures
      (0.9943 .. 0.9944; the Laplace route's is 0.9932 .. 0.9954).
  [c] elastic body, a law EXPONENTIAL in traction: B = (A / k_E) ln(1 + t /
      t_c), the logarithmic afterslip of rate-strengthening friction, with
      the rate falling a thousandfold from the start to the window's end
      (0.9944 .. 0.9944). ``RateStrengtheningFault`` itself: at a traction
      far below its ``a_sigma`` it is its own ``linearized`` law (8e-5), and
      at five times it, it is not (4.7 times the slip, early: tripwire).
  [d] Maxwell body, linear law: against the contour inversion of the closed
      form (0.9943 .. 0.9946), and against the Laplace route's fit of the
      same reduced operators (5e-4). STABILITY: with the linear law the
      whole system's eigenvalues have no positive real part (5e-14 measured,
      P0 at a/h = 5), and WITHOUT the stepper's correction of the fit's
      relaxed limit they do (2.4e-3 a relaxation time: tripwire).
  [e] Maxwell body, the exponential law: against the scalar equation with
      the material's two exact poles, 1/tau and 1/tau_2, integrated to
      1e-11 (0.9943 .. 0.9949; the body's relaxation multiplies the slip
      by 4.6 by the window's end). It shares nothing with the real-axis
      fit; that its two terms ARE the code's k_E(s) is checked first (2e-16).
  [f] a megathrust's creeping reach in the substrate, under gravity
      (``layered_box`` ``lower="creep"``, 19,266 unknowns, 992 of them the
      reach's slip): the reduction against the dense backend's solve
      (1e-14); every state's stiffness resists (the largest real part of
      its eigenvalues is negative, -0.09); ``ViscousFault`` stepped against
      the Laplace route, the reach's slip (1.4e-3, at the window's end) and
      the free surface (8e-5); the friction law's rake is the rupture's;
      under an earthquake ten thousand times smaller the friction law is
      the linear one along the rake (7e-5), and under this one, 1 m against
      an a_sigma of 1 MPa, it is not (1.75 times the slip at 0.1: tripwire).
  [g] refusals: a time past the fit's window, a solved fault with no law, a
      law on both routes, a law that is not linear on the Laplace route; on
      the compressed kernel a slip ON an interface, a continuous fault, and
      the FMM anywhere (a P0 fault inside a body is TAKEN there: [j]).
      And ``PowerLawFault``: the viscous law at n = 1 (identical), and not
      at n = 3 (0.49: tripwire).
  [h] the fault's own kernel, COMPRESSED (``la.faultkernel``,
      ``stepper.KernelBody``): what lets a fault in the whole space be
      stepped with nothing dense. The same crack at P0, a/h = 12 (1,055
      elements). Two stored matrices give the kernel at any material: the
      assembler's own block at one a quarter of the way along the range
      (4e-12), and the product at three (3e-8, float32 storage); swapped,
      they do not (tripwire). A pool of two processes builds the same
      operator as one (2e-15). ``KernelBody`` against the dense ``Body``:
      the elastic body under the linear law (3e-7); the Maxwell body under
      friction, as fitted (2e-7: the operators are one) and with each one's
      correction of the relaxed limit (1e-3: a matrix's spectrum there, two
      numbers here); the two numbers do not push, and as fitted one does
      (-0.108, of mu = 30: tripwire). Refused: a slip on an interface, a
      continuous fault, a material outside the range. Every run here is
      the embedded pair's, so that what is compared is the operators.
  [i] the STABILISED integrator (``mbem.rkc``), which is what a
      ``KernelBody`` is stepped with: order two at a fixed step (the error
      falls 4.2 times as the step halves); a stiff diffusion with its exact
      answer, read between steps (4e-6), sixteen times the stiffness for
      3.6 times the evaluations where the embedded pair pays 15.9
      (tripwire); and the crack of [h] in its Maxwell body under friction,
      the pair's answer (2e-6) in 1,575 evaluations against 3,020.
  [j] a fault IN A BODY on its compressed kernel: what the boundary adds to
      the faults' stiffness, in low rank (``la.coupling``,
      ``KernelBody.boundary``). The box of [a] with its material relaxing,
      a traction on its top and its base moved (96 slip unknowns, 9,348 of
      the boundary's, eleven states): every state's stiffness is the dense
      reduction's (2e-8), and so are what those data load the fault with
      (3e-8; a third of it is not the fault's own datum: tripwire) and
      every unknown of the boundary at a state the fit weighs (1e-7), in
      ranks of 58 and 60; as fitted the relaxed limit grows a mode (1.5e-3:
      tripwire), and a fluid's, which is none, does not; stepped it is the
      dense body (3e-8). The same body factored in BLOCKS (``la.blr``,
      where a boundary past what a dense matrix holds would be): stiffness
      and loads as before (2e-8, 3e-8), the boundary's unknowns to 4e-5.
      And the megathrust's reach of [f] (992 slip
      unknowns under 17,778, where the boundary sees all of so coarse a
      fault: ranks 982 and 992): stiffness (3e-8), loads (6e-15) and every
      unknown of the box (7e-7) are the dense body's, and stepped under the
      viscous law the two agree as two corrections of one fit can (1.6e-3
      in the slip, 4e-4 in its mean).
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np

import mollified_bem as mb
import verify_solved_bvp as VB                 # sibling: the anchor-C box
import verify_viscoelastic as VV               # sibling: the penny, the megathrust
from local_box_mesh_eq import make_vertical_fault_eq
from mbem import config as cfg, laplace, rheology, runner, stepper
from mbem.backends.dense import AssembledDense
from mbem.cases import layered_box, penny
from mbem.evaluate import evaluate_fault_traction
from mbem.kernels import basis as kb
from mbem.la.faultkernel import FaultKernel, g_of, lam_of
from mbem.model import BCType, Patch, generate_system

MU, LAM = VV.MU, VV.LAM
A, DELTA0 = 1.0, 1.0e-3
WINDOW = VV.WINDOW                             # (0.1, 10): the times read
TOL_READ = 1.0e-10       # S b + L at the slip a readout gave; 7e-15, 2e-14
TOL_SAME = 1.0e-10       # the reduction vs the dense solve; 1e-14
BAND = VV.PENNY_BAND     # the crack's own bias in moment
FLAT_ELASTIC = 3.0e-4    # spread of a ratio over time, no fit in it; 1e-4, 3e-5
FLAT_RELAXING = 1.5e-3   # ... with the body's memory fitted; 3e-4, 6e-4
TOL_ROUTES = 2.0e-3      # stepper vs the Laplace route's fit; 5e-4
TOL_SMALL = 5.0e-4       # friction far below a_sigma vs its linear law; 8e-5
NOT_LINEAR = 3.0         # ... and at five times a_sigma, early; 4.7
TOL_GROWTH = 1.0e-8      # largest real part, corrected; 5e-14
GROWS = 1.0e-3           # ... as fitted; 2.4e-3
TOL_POLES = 1.0e-12      # two exact poles vs the code's k_E(s); 2e-16
TOL_REACH = 3.0e-3       # megathrust: the reach's slip, the two routes; 1.4e-3
TOL_SURFACE = 3.0e-4     # ... the free surface; 8e-5
SMALLER = 1.0e-4         # the earthquake under which friction is linear
TOL_SMALLER = 1.0e-3     # ... to this; 7e-5
EARLY = 1.3              # this one: friction over linear at the first time; 1.75
TOL_TWO = 1.0e-9         # two stored matrices vs the assembler's block; 4e-12
TOL_PRODUCT = 1.0e-6     # compressed product vs dense (float32 store); 3e-8
TOL_POOL = 1.0e-12       # two processes vs one; 2e-15
TOL_KERNEL = 1.0e-5      # KernelBody vs Body, the same operator; 3e-7
TOL_KERNEL_LIMIT = 5.0e-3   # ... each with its own relaxed limit; 1e-3
TOL_ORDER = 0.3          # error ratio at half the step, less 4; 0.18
TOL_HEAT = 2.0e-5        # a stiff diffusion vs its exact answer; 4e-6
TOL_RKC = 1.0e-4         # RKC vs the embedded pair, the crack under friction; 2e-6
TOL_STATE = 1.0e-5       # a state's stiffness and loads, coupling vs dense; 3e-8
TOL_UNKNOWNS = 1.0e-4    # ... every unknown of the boundary; 1e-7, 7e-7
TOL_MEAN = 1.0e-3        # mean afterslip, two corrections of one fit; 4e-4
TOL_BLOCKS = 5.0e-4      # the boundary's unknowns, its body factored in blocks; 4e-5

CHECKS = []


def check(name, val, tol):
    ok = bool(np.isfinite(val)) and val < tol
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3e} < {tol:.0e}")
    return ok


def check_gt(name, val, floor):
    ok = bool(np.isfinite(val)) and val > floor
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {val:9.3e} > {floor:.0e}")
    return ok


def check_band(name, r, flat):
    ok = bool(BAND[0] < r.min() and r.max() < BAND[1] and np.ptp(r) < flat)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {r.min():.4f} .. {r.max():.4f}"
          f"  (flat to {flat:.0e})")
    return ok


def check_bool(name, ok, note=""):
    ok = bool(ok)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:70s} {note}")
    return ok


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def refuses(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


def times():
    return np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 13)


# ---- [a] the reduction ------------------------------------------------

def a_reduction():
    print("\n[a] the reduction onto a solved fault, in a box with free and clamped faces")
    mat = VB.material(0.30)
    box = VB._closed_box(VB.EDGE_C)
    keys = ("top", "sides", "base")
    fc = dict(VB.FAULT_C, target_edge=2.5)
    fmesh, _, _ = make_vertical_fault_eq(**fc)
    zc = 0.5 * sum(fc["depth_range"])
    D, L = fc["depth_range"][1] - fc["depth_range"][0], fc["strike_length"]

    def slip_field(x):
        x = np.asarray(x, float)
        out = np.zeros((x.shape[0], 3))
        out[:, 1] = VB.SLIP * (1.0 + 0.5 * x[:, 1] / (0.5 * L)
                               + 0.3 * (x[:, 2] - zc) / (0.5 * D))
        return out

    eps = {"top": VB.EPS_C, "sides": VB.EPS_C, "base": VB.EPS_C, "fault": "auto"}
    bcs = dict(top="t", sides="t", base="d")
    for order, layout in ((0, "discontinuous"), (2, "continuous")):
        fA = Patch("fault", fmesh, BCType.FAULT, value=slip_field, order=order,
                   layout=layout)
        mA, rA = VB._build_box_model(box, mat, bcs, {k: None for k in keys},
                                     faults=[fA])
        solA = quiet(AssembledDense, generate_system(mA), eps, "direct").solve()
        tA = evaluate_fault_traction(mA, rA, solA, fA, eps)
        nrm = fA.collocation_normals()
        tA_t = tA - np.einsum("ni,ni->n", tA, nrm)[:, None] * nrm
        fB = Patch("fault", fmesh, BCType.FAULT_TRACTION, value=tA_t,
                   order=order, layout=layout, locked=np.zeros(fA.n_dofs, bool))
        mB, _rB = VB._build_box_model(box, mat, bcs, {k: None for k in keys},
                                      faults=[fB])
        asm = quiet(AssembledDense, generate_system(mB), eps, "direct")
        c = asm.condense()
        f = c.faults[0]
        label = f"P{order} {layout}"
        check(f"{label}: S b + L = 0 at the slip whose traction was read",
              float(np.linalg.norm(c.S @ f.coordinates(fA.value_array()) + c.L)
                    / np.linalg.norm(c.L)), TOL_READ)
        for kappa, what in ((float(np.median(np.abs(np.diag(c.S)))), "dragging"),
                            (0.0, "free")):
            full = (asm.rebuild_for_materials({}, fault_drag={"fault": kappa})
                    if kappa else asm).solve()
            beta = c.dragged(kappa)
            check(f"{label}, {what}: the slip is the dense backend's",
                  rel(c.cartesian(beta)["fault"], full["b:fault"]), TOL_SAME)
            check(f"{label}, {what}: ... and so is the free surface",
                  rel(c.slots(c.x0 + c.X @ beta)["u:top"], full["u:top"]),
                  TOL_SAME)
        dragged = asm.rebuild_for_materials({}, fault_drag={"fault": 1.0})
        check_bool(f"{label}: an operator built with a drag is refused",
                   refuses(dragged.condense, ValueError))


# ---- the penny: [b] .. [e] --------------------------------------------

class Penny:
    """The crack of ``verify_viscoelastic`` [o], reduced at every state of a
    Maxwell body, and the pieces of its closed forms."""

    def __init__(self, ah=8.0, order=2):
        self.base = mb.ElasticMaterial(mu=MU, lam=LAM)
        self.drop = penny.stress_drop(MU, LAM, DELTA0, A)
        self.k_e = self.drop / DELTA0                 # the crack's stiffness
        model = penny.build_model(penny.penny_mesh(a=A, h=A / ah), self.base,
                                  drop=self.drop, order=order)
        self.fault = model.regions[0].faults[0]
        asm = quiet(AssembledDense, generate_system(model), "auto", "direct")
        self.s = laplace.s_grid(*WINDOW)
        self.cond = [asm.rebuild_for_materials({"space": VV.elastic(VV.relaxed(
            VV.MAXWELL, self.base, sv))}).condense()
            for sv in (np.inf,) + self.s]
        f = self.cond[0].faults[0]
        x = self.fault.collocation_points()[f.free]
        shape = np.sqrt(np.clip(1.0 - (x[:, 0] ** 2 + x[:, 1] ** 2) / A ** 2,
                                1.0e-12, None))
        self.grade = np.repeat(shape, 2)              # per coordinate
        self.eta0 = MU * VV.CREEP_TAU / A             # ViscousFault(tau, width=a)

    def moment(self, beta) -> float:
        return float(self.fault.integrate(
            self.cond[0].cartesian(beta)["penny"])[0])

    def moments(self, h) -> np.ndarray:
        return np.array([self.moment(b) for b in h.slip])

    def linear(self, tr):
        return self.grade * tr / self.eta0

    def exponential(self, v0, a_s):
        """The graded law that is exponential in the traction along x."""
        def rate(tr):
            v = np.zeros_like(tr)
            v[0::2] = self.grade[0::2] * v0 * np.exp(tr[0::2] / a_s)
            return v
        return rate


def b_elastic_linear(P):
    print("\n[b] elastic body, linear law: one exponential, and no fit anywhere")
    t = times()
    body = stepper.Body([P.cond[0]])
    h = stepper.march(body, P.linear, t)
    ref = penny.eshelby_moment(DELTA0, A) * (1.0 - np.exp(-t * P.k_e / P.eta0))
    check_band("moment over the closed form, in time", P.moments(h) / ref,
               FLAT_ELASTIC)
    check_gt("the slip really grows over the window", float(ref[-1] / ref[0]), 5.0)
    return body


def c_elastic_exponential(P, body):
    print("\n[c] elastic body, a law exponential in traction: logarithmic afterslip")
    t = times()
    a_s = P.drop / 5.0
    v0 = a_s / (P.k_e * np.exp(5.0) * 0.01)           # so that t_c = 0.01
    h = stepper.march(body, P.exponential(v0, a_s), t)
    B = (a_s / P.k_e) * np.log1p(t / 0.01)
    check_band("moment over (A / k) ln(1 + t / t_c)",
               P.moments(h) / penny.eshelby_moment(1.0, A) / B, FLAT_ELASTIC)
    check_gt("the rate falls a thousandfold from the start to the window's end",
             float(1.0 + t[-1] / 0.01), 1.0e3)

    def friction(scale):
        a_sigma = scale * P.drop
        law = rheology.RateStrengtheningFault(v0=a_sigma / P.eta0,
                                              a_sigma=a_sigma, rake=0.0)
        return law, stepper.march(body, stepper.law_rate(
            body, {"penny": law}, {"penny": MU}), t)

    law, small = friction(1.0e4)
    lin = stepper.march(body, stepper.law_rate(
        body, {"penny": law.linearized(MU)}, {"penny": MU}), t)
    check("RateStrengtheningFault far below a_sigma is its linearized law",
          float(np.abs(P.moments(small) / P.moments(lin) - 1.0).max()), TOL_SMALL)
    _law, big = friction(0.2)
    check_gt("... and at five times a_sigma it is not (tripwire)",
             float(P.moments(big)[0] / P.moments(lin)[0]), NOT_LINEAR)


def d_maxwell_linear(P):
    print("\n[d] Maxwell body, linear law: the contour, the Laplace route, stability")
    t = times()
    body = stepper.Body(P.cond, s=P.s)
    h = stepper.march(body, P.linear, t)

    def closed(sv):
        m = VV.relaxed(VV.MAXWELL, P.base, sv)
        k_e = np.pi * m.mu * (1.0 + rheology.g_of(m.mu, m.lam)) / (8.0 * A)
        return penny.eshelby_moment(P.drop / (k_e + P.eta0 * sv), A)

    got = P.moments(h)
    check_band("moment over the contour inversion of the closed form",
               got / VV.step_response(closed, t), FLAT_RELAXING)
    lap = np.array([0.0] + [P.moment(c.dragged(P.eta0 * sv / P.grade))
                            for sv, c in zip(P.s, P.cond[1:])])
    check("... and against the Laplace route's fit of the same operators",
          float(np.abs(got / (laplace.Prony(P.s, secular=True).weights(t) @ lap)
                       - 1.0).max()), TOL_ROUTES)
    print(f"      the fit's relaxed limit: {body.growing} of {body.n} modes "
          f"pushed, {body.removed:.1e} of the elastic stiffness moved")
    small = Penny(ah=5.0, order=0)
    mob = small.grade / small.eta0
    check("no mode grows under the linear law (P0, a/h = 5)",
          stepper.Body(small.cond, s=small.s).growth(mob), TOL_GROWTH)
    check_gt("... and one does with the relaxed limit as fitted (tripwire)",
             stepper.Body(small.cond, s=small.s, passive=False).growth(mob),
             GROWS)
    return body


def e_maxwell_exponential(P, body):
    print("\n[e] Maxwell body, the exponential law: the material's two exact poles")
    from scipy.integrate import solve_ivp

    # k_E(s) / s = c1 / (s + a1) + c2 / (s + a2): shear relaxes at 1/tau and
    # g = 1/(1 - nu) at 1/tau_2 (docs/viscoelastic.md section 4)
    K = LAM + 2.0 * MU / 3.0
    g_e = rheology.g_of(MU, LAM)
    a1 = 1.0 / VV.MAXWELL.tau
    a2 = a1 * 3.0 * K / (3.0 * K + 4.0 * MU)
    lead = np.pi * MU / (8.0 * A)
    c1 = lead * (3.0 + (g_e - 2.0) * a1 / (a1 - a2))
    c2 = -lead * (g_e - 2.0) * a2 / (a1 - a2)
    worst = 0.0
    for sv in (0.03, 1.0, 40.0):
        m = VV.relaxed(VV.MAXWELL, P.base, sv)
        k_s = np.pi * m.mu * (1.0 + rheology.g_of(m.mu, m.lam)) / (8.0 * A)
        worst = max(worst, abs((c1 / (sv + a1) + c2 / (sv + a2)) * sv / k_s - 1.0))
    check("the two poles are the code's k_E(s)", worst, TOL_POLES)

    t = times()
    a_s = P.drop / 5.0
    v0 = a_s / (P.k_e * np.exp(5.0) * 0.01)

    def scalar(_t, y):
        rate = v0 * np.exp((P.drop - c1 * y[1] - c2 * y[2]) / a_s)
        return [rate, rate - a1 * y[1], rate - a2 * y[2]]

    ref = solve_ivp(scalar, (0.0, t[-1]), [0.0, 0.0, 0.0], method="DOP853",
                    t_eval=t, rtol=1.0e-11, atol=1.0e-18).y[0]
    h = stepper.march(body, P.exponential(v0, a_s), t)
    check_band("moment over the scalar equation's",
               P.moments(h) / penny.eshelby_moment(1.0, A) / ref, FLAT_RELAXING)
    elastic = (a_s / P.k_e) * np.log1p(t / 0.01)
    check_gt("the body's relaxation matters by the window's end (tripwire)",
             float(ref[-1] / elastic[-1]), 1.2)


# ---- [f] the megathrust ----------------------------------------------

def f_megathrust():
    print("\n[f] a megathrust's creeping reach, in a substrate that relaxes")
    geo = dict(VV.THRUST, **VV.THROUGH)
    plate = mb.ElasticMaterial(MU, LAM)
    meshes = layered_box.build_layered_box(**geo)
    slip = 1.0e-3
    model = layered_box.build_model(
        meshes, slip, plate, plate, sides="clamped", rake=90.0,
        density=VV.THRUST_DENSITY, g=VV.THRUST_G, lower="creep")
    walls = dict(interface=VV.THRUST_EPS, sides_lower=VV.THRUST_EPS,
                 base=VV.THRUST_EPS)
    eps = dict(top="auto", sides_upper="auto", fault="auto",
               fault_lower="auto", **walls)
    asm = quiet(AssembledDense, generate_system(model), eps, "direct")
    s = laplace.s_grid(*WINDOW)

    def mats(sv):
        return {layered_box.PLATE: plate, layered_box.SUBSTRATE: VV.elastic(
            VV.relaxed(VV.MAXWELL, plate, sv))}

    states = [mats(sv) for sv in (np.inf,) + s]
    cond = [asm.rebuild_for_materials(st).condense() for st in states]
    n = cond[0].n
    print(f"      {asm.A.shape[0]} unknowns, {n} of them the reach's slip, "
          f"{len(cond)} states")
    law = rheology.ViscousFault(tau=VV.CREEP_TAU, width=10.0)
    k = 4
    full = asm.rebuild_for_materials(
        mats(s[k - 1]), fault_drag={"fault_lower": law.drag(MU, s[k - 1])}).solve()
    beta = cond[k].dragged(law.drag(MU, s[k - 1]))
    check("the reduction is the dense backend's solve: the reach's slip",
          rel(cond[k].cartesian(beta)["fault_lower"], full["b:fault_lower"]),
          TOL_SAME)
    check("... and the free surface",
          rel(cond[k].slots(cond[k].x0 + cond[k].X @ beta)["u:top"],
              full["u:top"]), TOL_SAME)
    worst = max(float(np.linalg.eigvals(c.S).real.max()) for c in cond)
    check("every state's stiffness resists (largest real part of its "
          "eigenvalues)", worst, 0.0)

    body = stepper.Body(cond, s=s)
    print(f"      the fit's relaxed limit: {body.growing} of {n} modes pushed, "
          f"{body.removed:.1e} of the elastic stiffness moved")
    t = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 9)
    mu = {"fault_lower": MU}
    h = stepper.march(body, stepper.law_rate(body, {"fault_lower": law}, mu), t)
    W = laplace.Prony(s).weights(t)
    lap = np.array([np.zeros(n)] + [c.dragged(law.drag(MU, sv))
                                    for sv, c in zip(s, cond[1:])])
    slots = cond[0].layout.slot(model.regions[0].patches[0], "u")
    top = slice(slots.offset, slots.stop)
    lap_top = np.array([c.x0[top] + c.X[top] @ b for c, b in zip(cond, lap)])
    got_top = np.array([body.unknowns(tt, y, h.events)[top]
                        for tt, y in zip(h.t, h.y)])
    check("ViscousFault, stepped vs the Laplace route: the reach's slip",
          max(rel(a, b) for a, b in zip(h.slip, W @ lap)), TOL_REACH)
    check("... and the free surface",
          max(rel(a, b) for a, b in zip(got_top, W @ lap_top)), TOL_SURFACE)

    # friction along the rake, against the linear law along the rake
    fr = rheology.RateStrengtheningFault(
        v0=1.0e-3 * 10.0 / (MU * VV.CREEP_TAU), a_sigma=1.0e-3, rake=90.0)
    f = cond[0].faults[0]
    sense = layered_box.burgers(1.0, 90.0, meshes["n_hat"], meshes["s_hat"])
    d = f.coordinates(np.broadcast_to(sense, (f.n_dofs, 3))).reshape(-1, 2)
    check("the friction law's rake is the rupture's",
          float(np.abs(d - fr.direction()).max()), 1.0e-12)
    eta = MU * law.tau / law.width

    def along(tr):
        return ((tr.reshape(-1, 2) @ fr.direction())[:, None]
                * fr.direction() / eta).ravel()

    rate = stepper.law_rate(body, {"fault_lower": fr}, mu)
    lin = stepper.march(body, along, t)
    small = stepper.march(body, rate, t, events=((0.0, SMALLER),))
    check("friction under an earthquake 1e4 times smaller is the linear law",
          max(rel(a / SMALLER, b) for a, b in zip(small.slip, lin.slip)),
          TOL_SMALLER)
    big = stepper.march(body, rate, t)
    check_gt("... and under this one it is not: slip at the first time, over "
             "the linear law's (tripwire)",
             float(big.slip[0].sum() / lin.slip[0].sum()), EARLY)
    return types.SimpleNamespace(body=body, model=model, eps=eps, s=s,
                                 states=states, cond=cond)


# ---- [g] refusals ------------------------------------------------------

def g_refusals(P, body_m, reach):
    print("\n[g] refusals")
    check_bool("a time past the end of the fit's window",
               refuses(lambda: stepper.march(body_m, P.linear,
                                             [2.0 * WINDOW[1]]), ValueError))
    check_bool("a time before the first event",
               refuses(lambda: stepper.march(body_m, P.linear, [0.5],
                                             events=(1.0,)), ValueError))
    check_bool("a solved fault with no law",
               refuses(lambda: stepper.law_rate(body_m, {}, {}), KeyError))
    check_bool("several states and no Laplace parameters",
               refuses(lambda: stepper.Body(P.cond), ValueError))
    visc = rheology.ViscousFault(tau=1.0, width=10.0)
    power = rheology.PowerLawFault(tau=1.0, width=10.0, n=3.0, stress=1.0e-3)
    spec = dict(name="x", model=cfg.Model(
        geometry=cfg.Geometry("layered_box"), builder="layered_box"),
        backend=cfg.Backend(kind="dense"))
    lap = dict(rheology={"substrate": rheology.Maxwell(tau=1.0)},
               s=laplace.s_grid(*WINDOW))
    check_bool("a law on both routes",
               refuses(lambda: cfg.Run(
                   laplace=cfg.Laplace(faults={"fault_lower": visc}, **lap),
                   stepper=cfg.Stepper(faults={"fault_lower": visc}, t=(1.0,)),
                   **spec), ValueError))
    # all these checks ask of a model: its solved faults, their regions,
    # and whether it has a boundary
    def stub(patches, **fault):
        region = types.SimpleNamespace(
            patches=patches, material=types.SimpleNamespace(mu=MU))
        f = types.SimpleNamespace(**dict(dict(
            name="fault_lower", order=0, continuous=False,
            allow_opening=False, bc=BCType.FAULT_TRACTION), **fault))
        return types.SimpleNamespace(regions=[region],
                                     solved_faults=lambda: [(region, f)])

    model = stub(["a wall"])
    run = cfg.Run(laplace=cfg.Laplace(faults={"fault_lower": power}, **lap),
                  **spec)
    check_bool("a law that is not linear, on the Laplace route",
               refuses(lambda: runner._creeping_faults(model, run), ValueError))
    step = dict(laplace=cfg.Laplace(**lap), stepper=cfg.Stepper(
        faults={"fault_lower": power}, t=(1.0,)))
    hmat = cfg.Run(**dict(spec, backend=cfg.Backend(kind="hmat")), **step)
    check_bool("a P0 fault inside a body is taken on the compressed kernel",
               "fault_lower" in runner._stepped_faults(model, hmat))
    space = stub([])
    check_bool("... and one in the whole space",
               "fault_lower" in runner._stepped_faults(space, hmat))
    check_bool("... the FMM is not, anywhere",
               refuses(lambda: runner._stepped_faults(space, cfg.Run(**dict(
                   spec, backend=cfg.Backend(kind="fmm")), **step)),
                   ValueError))
    check_bool("... nor a continuous fault",
               refuses(lambda: runner._stepped_faults(
                   stub([], order=2, continuous=True), hmat), ValueError))
    check_bool("... nor a slip ON an interface, whose stiffness is all the "
               "body's",
               refuses(lambda: runner._stepped_faults(
                   stub(["a wall"], bc=BCType.INTERFACE), hmat), ValueError))
    check_bool("a stepped run that names a fault the model does not solve",
               refuses(lambda: runner._stepped_faults(model, cfg.Run(
                   laplace=cfg.Laplace(**lap), stepper=cfg.Stepper(
                       faults={"elsewhere": power}, t=(1.0,)), **spec)),
                   KeyError))
    # the power law is the viscous one at n = 1, and steps as one
    t = np.array([1.0, 10.0])
    mu = {"fault_lower": MU}
    one = rheology.PowerLawFault(tau=1.0, width=10.0, n=1.0, stress=1.0e-3)
    a = stepper.march(reach, stepper.law_rate(reach, {"fault_lower": one}, mu), t)
    b = stepper.march(reach, stepper.law_rate(reach, {"fault_lower": visc}, mu), t)
    check("PowerLawFault at n = 1 is ViscousFault", rel(a.slip, b.slip), 1.0e-12)
    c = stepper.march(reach, stepper.law_rate(reach, {"fault_lower": power}, mu), t)
    check_gt("... and at n = 3 it is not (tripwire)", rel(c.slip, b.slip), 0.05)


def h_compressed_kernel():
    print("\n[h] the fault's own kernel, compressed: nothing dense")
    P = Penny(ah=12.0, order=0)
    fault = P.fault
    model = penny.build_model(fault.mesh, P.base, drop=P.drop, order=0)
    fault = model.regions[0].faults[0]
    mats = [P.base] + [VV.elastic(VV.relaxed(VV.MAXWELL, P.base, sv))
                       for sv in P.s]
    pairs = [(m.mu, m.lam) for m in mats]
    g = [g_of(*m) for m in pairs]
    K = FaultKernel([fault], "auto", (min(g), max(g)))
    print(f"      {K.n} elements; {K.report['admissible_blocks']} far blocks "
          f"of rank {K.report['rank_median']:.0f}, "
          f"{K.report['dense_blocks']} near; "
          f"{K.report['bytes'] / K.report['dense_bytes']:.2f} of dense")
    # a material at a quarter of the range: not an end, and not the middle,
    # where the two weights are one number and swapping them is nothing
    mid = (12.0, lam_of(12.0, 0.75 * min(g) + 0.25 * max(g)))
    ref = kb.assemble_traction_matrix(
        fault.collocation_points(), fault.collocation_normals(), fault.mesh,
        mb.ElasticMaterial(mu=mid[0], lam=mid[1]),
        kb.resolve_patch_eps("auto", fault), order=0, kind="slip")
    check("two stored matrices are the assembler's block at a third material",
          rel(K.dense(*mid), ref), TOL_TWO)
    x = np.random.default_rng(7).standard_normal(3 * K.n)
    worst = 0.0
    for m in (pairs[0], mid, pairs[-1]):
        want = K.dense(*m) @ x
        got = K.apply(*(w * x for w in K.weights(*m)))
        worst = max(worst, float(np.linalg.norm(got - want)
                                 / np.linalg.norm(want)))
    check("the compressed product at three materials", worst, TOL_PRODUCT)
    w = K.weights(*mid)
    check_gt("... and with the two weights swapped it is not (tripwire)",
             float(np.linalg.norm(K.apply(*(wb * x for wb in w[::-1])) - ref @ x)
                   / np.linalg.norm(ref @ x)), 1.0e-2)
    pool = FaultKernel([fault], "auto", (min(g), max(g)), processes=2)
    check("a pool of two processes builds the same operator",
          float(np.linalg.norm(pool.apply(x, 0.5 * x) - K.apply(x, 0.5 * x))
                / np.linalg.norm(K.apply(x, 0.5 * x))), TOL_POOL)

    t = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 7)
    visc = rheology.ViscousFault(tau=VV.CREEP_TAU, width=A)
    a_sigma = 0.2 * P.drop
    fric = rheology.RateStrengtheningFault(v0=a_sigma / P.eta0,
                                           a_sigma=a_sigma, rake=0.0)

    def run(body, law):                     # the pair: operators compared
        return stepper.march(body, stepper.law_rate(
            body, {"penny": law}, {"penny": MU}), t, method="RK45").slip

    def worst_of(a, b):
        return float(np.abs(a - b).max() / np.abs(b).max())

    check("KernelBody is Body: the elastic body, the linear law",
          worst_of(run(stepper.KernelBody.from_model(model, "auto", pairs[:1]),
                       visc),
                   run(stepper.Body(P.cond[:1]), visc)), TOL_KERNEL)
    fitted = stepper.KernelBody.from_model(model, "auto", pairs, s=P.s,
                                           passive=False)
    check("... the Maxwell body under friction, the fit as it is",
          worst_of(run(fitted, fric),
                   run(stepper.Body(P.cond, s=P.s, passive=False), fric)),
          TOL_KERNEL)
    kern = stepper.KernelBody.from_model(model, "auto", pairs, s=P.s)
    check("... and each with its own correction of the relaxed limit",
          worst_of(run(kern, fric), run(stepper.Body(P.cond, s=P.s), fric)),
          TOL_KERNEL_LIMIT)
    def limit(a):                           # of H_lo and of H_hi - H_lo
        return float(a[:, 0].sum()), float(a[-1, 0])

    check_bool("the corrected limit does not push",
               min(limit(kern._a)) >= 0.0,
               "%.3f, %.3f" % limit(kern._a))
    check_bool("... and as fitted it does (tripwire)",
               min(limit(fitted._a)) < 0.0,
               "%.3f, %.3f" % limit(fitted._a))
    face = types.SimpleNamespace(solved_faults=lambda: [(None, (
        types.SimpleNamespace(name="face", bc=BCType.INTERFACE)))])
    check_bool("refused: a slip on an interface",
               refuses(lambda: stepper.KernelBody.from_model(
                   face, "auto", pairs[:1]), ValueError))
    smooth = penny.build_model(fault.mesh, P.base, drop=P.drop,
                               order=2).regions[0].faults[0]
    check_bool("refused: a continuous fault",
               refuses(lambda: FaultKernel([smooth], "auto", (1.4, 1.6)),
                       NotImplementedError))
    check_bool("refused: a material outside the kernel's range",
               refuses(lambda: K.weights(1.0, lam_of(1.0, 1.05)), ValueError))
    return kern, stepper.law_rate(kern, {"penny": fric}, {"penny": MU}), t


# ---- [i] the stabilised integrator ----------------------------------------

def i_stabilised(kern, rate, t):
    print("\n[i] the stabilised integrator: the square root of the steps")
    from scipy.integrate import solve_ivp
    from mbem.rkc import RKC

    def lag(tt, y):
        return -y + np.sin(tt)

    def lagged(tt):
        return 1.5 * np.exp(-tt) + 0.5 * (np.sin(tt) - np.cos(tt))

    err = [abs(solve_ivp(lag, (0.0, 2.0), [1.0], method=RKC, rtol=1.0e6,
                         atol=1.0e6, first_step=h, max_step=h).y[0, -1]
               - lagged(2.0)) for h in (0.1, 0.05)]
    check("order two: at half the step the error is a quarter",
          abs(err[0] / err[1] - 4.0), TOL_ORDER)

    def heat(n, method):
        x = np.linspace(0.0, 1.0, n + 2)[1:-1]
        dx = x[1] - x[0]

        def f(_t, u):
            d = np.empty_like(u)
            d[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
            d[0], d[-1] = u[1] - 2.0 * u[0], u[-2] - 2.0 * u[-1]
            return d / dx ** 2

        modes = [(1, np.sin(np.pi * x)), (5, 0.3 * np.sin(5.0 * np.pi * x))]
        sol = solve_ivp(f, (0.0, 0.2), modes[0][1] + modes[1][1],
                        method=method, rtol=1.0e-6, atol=1.0e-10,
                        t_eval=[0.05, 0.2])
        want = np.stack([sum(np.exp(-(2.0 - 2.0 * np.cos(k * np.pi * dx))
                                    / dx ** 2 * tt) * m for k, m in modes)
                         for tt in sol.t], 1)
        return float(np.abs(sol.y - want).max()), sol.nfev

    (e1, n1), (e4, n4) = heat(100, RKC), heat(400, RKC)
    check("a stiff diffusion, 100 and 400 points, read between steps: its "
          "exact answer", max(e1, e4), TOL_HEAT)
    check("sixteen times the stiffness, under five times the evaluations",
          n4 / n1, 5.0)
    check_gt("... where the embedded pair pays over ten (tripwire)",
             heat(400, "RK45")[1] / heat(100, "RK45")[1], 10.0)
    pair = stepper.march(kern, rate, t, method="RK45")
    cheb = stepper.march(kern, rate, t)
    check("the crack of [h] under friction: RKC is the embedded pair",
          float(np.abs(cheb.slip - pair.slip).max() / np.abs(pair.slip).max()),
          TOL_RKC)
    check_bool("... and it is what a KernelBody is stepped with, in fewer "
               "evaluations", cheb.nfev < pair.nfev,
               f"{cheb.nfev} against {pair.nfev}")
    explicit = stepper.march(kern, rate, t, method="RKC")
    check_bool("... the default IS that method", explicit.nfev == cheb.nfev)


# ---- [j] a fault in a body, on its compressed kernel ------------------------

def j_boundary(reach):
    print("\n[j] a fault in a body on its compressed kernel: the boundary in "
          "low rank")
    rng = np.random.default_rng(3)

    def worst_state(kb, cond):
        x = rng.standard_normal(kb.n)
        return max(rel(kb.stiffness(x, k), c.S @ x) for k, c in enumerate(cond))

    def worst_unknown(kb, dense):
        m = dense.tau.size
        Y = 1.0e-3 * rng.standard_normal((1 + m, kb.n))
        ev = ((0.0, 1.0),)
        a, b = kb.slots(1.0, Y, ev), dense.slots(1.0, Y, ev)
        return max(rel(a[k], b[k]) for k in b)

    # the box of [a], relaxing: small enough for the eigenvalues
    mat = VB.material(0.30)
    fmesh, _, _ = make_vertical_fault_eq(**dict(VB.FAULT_C, target_edge=2.5))
    value = np.zeros((fmesh.triangles.shape[0], 3))
    value[:, 1] = 1.0e-3                        # along strike, in the plane
    f = Patch("fault", fmesh, BCType.FAULT_TRACTION, value=value, order=0,
              locked=np.zeros(value.shape[0], bool))
    # ... loaded through its faces too, so that prescribed data reach the
    # fault's rows by both kernels: a traction on the top, the base moved
    def press(x):
        out = np.zeros((x.shape[0], 3))
        out[:, 2] = -1.0e-3 * np.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2)
                                     / VB.BOX_H ** 2)
        return out

    def shove(x):
        out = np.zeros((x.shape[0], 3))
        out[:, 1] = 1.0e-3 * np.tanh(x[:, 0] / VB.BOX_H)
        return out

    model, _r = VB._build_box_model(
        VB._closed_box(VB.EDGE_C), mat, dict(top="t", sides="t", base="d"),
        dict(top=press, sides=None, base=shove), faults=[f])
    eps = {"top": VB.EPS_C, "sides": VB.EPS_C, "base": VB.EPS_C,
           "fault": "auto"}
    s = laplace.s_grid(*WINDOW)
    states = [{model.regions[0].name: VV.elastic(VV.relaxed(VV.MAXWELL, mat, sv))}
              for sv in (np.inf,) + s]
    asm = quiet(AssembledDense, generate_system(model), eps, "direct")
    cond = [asm.rebuild_for_materials(st).condense() for st in states]
    dense = stepper.Body(cond, s=s)
    kb = quiet(stepper.KernelBody.from_model, model, eps, states, s=s)
    rep = kb.boundary.report
    print(f"      the box: {kb.n} slip unknowns through "
          f"{rep['boundary_unknowns']} of the boundary's, rank "
          f"{rep['slip_rank']} and {rep['traction_rank']} at {rep['tol']:g}")
    check("every state's stiffness is the dense reduction's",
          worst_state(kb, cond), TOL_STATE)
    check("... what its faces' data load the fault with",
          rel(kb.loads, np.array([c.L for c in cond])), TOL_STATE)
    check_gt("... which is not the fault's own datum alone (tripwire)",
             rel(kb.loads[0], -cond[0].faults[0].coordinates(value)), 0.1)
    check("... and every unknown of the boundary", worst_unknown(kb, dense),
          TOL_UNKNOWNS)
    check_bool("in a rank below the fault's own",
               max(rep["slip_rank"], rep["traction_rank"]) < kb.n,
               f"{rep['slip_rank']}, {rep['traction_rank']} of {kb.n}")
    mob = 1.0 / float(np.median(np.abs(np.diag(cond[0].S))))
    fitted = stepper.KernelBody(kb.kernel, kb.faults, kb.materials, kb.loads,
                                s=s, passive=False, boundary=kb.boundary)
    check_gt("as fitted the relaxed limit grows a mode (tripwire)",
             fitted.growth(mob), GROWS)
    check("... and a fluid's limit, which is none, does not",
          kb.growth(mob), TOL_GROWTH)

    def lin(tr):
        return mob * tr

    t = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 5)
    a = stepper.march(kb, lin, t, method="RK45")
    b = stepper.march(dense, lin, t)
    check("stepped under a linear law, it is the dense body",
          float(np.abs(a.slip - b.slip).max() / np.abs(b.slip).max()),
          TOL_KERNEL)

    # ... and the same body factored in BLOCKS, where a boundary past what
    # a dense matrix holds would be (la.blr)
    kc = quiet(stepper.KernelBody.from_model, model, eps, states, s=s,
               factor="blocks")
    check_bool("the body factored in blocks, the one above densely",
               kc.boundary.report["factor"] == "blocks"
               and rep["factor"] == "dense")
    check("... every state's stiffness is the dense reduction's still",
          worst_state(kc, cond), TOL_STATE)
    check("... what its faces' data load the fault with",
          rel(kc.loads, np.array([c.L for c in cond])), TOL_STATE)
    check("... and every unknown of the boundary", worst_unknown(kc, dense),
          TOL_BLOCKS)

    # the megathrust's reach of [f]
    kb = quiet(stepper.KernelBody.from_model, reach.model, reach.eps,
               reach.states, s=reach.s)
    rep = kb.boundary.report
    print(f"      the reach: {kb.n} slip unknowns through "
          f"{rep['boundary_unknowns']} of the boundary's, rank "
          f"{rep['slip_rank']} and {rep['traction_rank']} at {rep['tol']:g}")
    check("the megathrust's reach: every state's stiffness",
          worst_state(kb, reach.cond), TOL_STATE)
    check("... what the earthquake loads it with",
          rel(kb.loads, np.array([c.L for c in reach.cond])), TOL_STATE)
    check("... and every unknown of the box", worst_unknown(kb, reach.body),
          TOL_UNKNOWNS)
    law = rheology.ViscousFault(tau=VV.CREEP_TAU, width=10.0)
    mu = {"fault_lower": MU}
    t = np.logspace(np.log10(WINDOW[0]), np.log10(WINDOW[1]), 9)
    a = stepper.march(kb, stepper.law_rate(kb, {"fault_lower": law}, mu), t)
    b = stepper.march(reach.body, stepper.law_rate(
        reach.body, {"fault_lower": law}, mu), t)
    check("stepped, each body with its own relaxed limit: the reach's slip",
          float(np.abs(a.slip - b.slip).max() / np.abs(b.slip).max()),
          TOL_KERNEL_LIMIT)
    check("... and its mean",
          float(np.abs(a.slip.mean(1) / b.slip.mean(1) - 1.0).max()), TOL_MEAN)


def main():
    print("=" * 76)
    print("A solved fault stepped in time: the body as the fault's memory")
    print("=" * 76)
    a_reduction()
    P = Penny()
    print(f"\n  the penny: {P.cond[0].n} slip unknowns, {len(P.cond)} states")
    body_e = b_elastic_linear(P)
    c_elastic_exponential(P, body_e)
    body_m = d_maxwell_linear(P)
    e_maxwell_exponential(P, body_m)
    reach = f_megathrust()
    g_refusals(P, body_m, reach.body)
    i_stabilised(*h_compressed_kernel())
    j_boundary(reach)
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: stepper ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: stepper ({sum(1 for c in CHECKS if not c)} of "
              f"{len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
