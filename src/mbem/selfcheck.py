"""Runtime guard on THE fault sign convention.

``mbem.model.core.FAULT_ORIENTATION`` is the single statement of how a
fault's slip enters msd. Every site -- the solve (``equations.py``), the
displacement readout and the stress readout (``evaluate.py``) -- reads it
through ``RegionModel.orientation`` and applies the one coefficient
``-sigma``. That is what makes it centralised; it is NOT what makes it
right. Flip the constant and all three sites flip together: the solve
still agrees with the readouts, every internal consistency identity still
holds, and msd returns fault displacements and fault stresses that are
exactly backwards, silently.

So the constant is pinned here, at runtime, against a PHYSICAL statement
with a hardcoded direction -- the same statement check A3 of
``verify/verify_solved_bvp.py`` makes, shrunk to a model that solves in a
few milliseconds. For

    fault normal      n_hat = +x_hat
    slip              Du    = +SLIP * y_hat   (SLIP > 0)

the displacement discontinuity msd's fault source produces is

    u(x + 0+ n) - u(x - 0+ n) = +Du

(``Patch.value`` is the Burgers vector: the fault enters as ``u = ... -
sigma * H @ slip`` with sigma = FAULT_ORIENTATION = -1, and the density of
H is the jump u(+n) - u(-n)), so the material on the +x side of the fault
must move toward +y and the material on the -x side toward -y. Nothing in
that sentence is read from the code; it is written out below as ``> 0`` /
``< 0`` and the five checks are:

  [1] readout, displacement  ``evaluate_displacement``
        u_y at (+d, 0, 0) > 0 > u_y at (-d, 0, 0), each |u_y| a sizeable
        fraction of SLIP.
  [2] readout, stress (total)  ``evaluate_stress(subtract_anelastic=False)``
        ON the fault the smeared slip carries the anelastic eigenstress
        C:eps_star, and the fault term is -sigma*Sdd@slip, so its
        divergent part is -sigma*C:eps_star = +C:eps_star: sigma_xy must
        be POSITIVE. Only the SIGN is asserted here; the (3/4) mu SLIP /
        eps blob-peak magnitude window is applied to [3], not to [2].
  [3] readout, stress (eigenstress branch)
        subtracting the eigenstress must add +sigma*C:eps_star, i.e.
        REMOVE the positive blob: sigma_xy(elastic) - sigma_xy(total) must
        be NEGATIVE and of the same size -- this is the branch whose sign
        no self-consistency check can see.
  [4] solve  ``generate_system`` + dense LU
        a small box with five clamped faces and one free face, the fault
        the ONLY source of load: the free face must move toward +y where
        x > 0 and toward -y where x < 0. The fault RhsTerm is the entire
        right-hand side, so flipping its sign negates the whole solution
        and this check inverts.
  [5] readout, compressed  ``DisplacementEvaluator``
        statement [1] again, through the block-compressed evaluation
        operator, which carries its own copy of the coefficient.

[1]-[3] and [5] cover the readout paths; [4] covers the solve. After the
fault branch was folded into ``RegionModel.orientation`` all of them
derive from the ONE constant, so any single one of these five would
already pin the lot; they are all here so that RE-INTRODUCING a
hand-written sign at any single site is caught too, which one check
would not do (seeded and measured: each of the five fires on its own
site's flip).

Cost. Run ONCE per process, in three independently cached stages, on the
first call that can reach the convention on a model that actually has
faults:

    "core"        [1] + [4]   generate_system, evaluate_displacement
    "stress"      [2] + [3]   evaluate_stress          (after "core")
    "compressed"  [5]         DisplacementEvaluator    (after "core")

The split exists so the guard never compiles a numba kernel its caller
was not about to use itself: the stress kernels cost ~3.4 s to compile on
a cold numba cache, and a displacement-only user should not pay that.
Measured on this machine:

    warm:  2.0 ms "core", 1.5 ms "stress", 1.7 ms "compressed";
           69 ns for the cached no-op call afterwards.
    cold numba cache, evaluate_displacement + generate_system + solve:
           3733 ms with the guard vs 3626 ms without (+107 ms). Without
           the staging it was +3.8 s, all of it the stress compile.

Every kernel involved is ``@njit(cache=True)``, so even that is a
once-per-machine cost.
"""

from __future__ import annotations

import threading

import numpy as np

# ------------------------------------------------------------------ model --
# A 4x4x4 km box, five faces clamped, the +z face free, and a 2x2 km
# vertical fault through the middle. Small enough to solve in ~1 ms
# (144 unknowns), large enough that the free face genuinely responds.
_BOX_HALF = 2.0            # km, box is [-2, 2]^3
_FAULT_HALF = 1.0          # km, fault is x = 0, |y| <= 1, |z| <= 1
_N = 2                     # quads per side per face
_SLIP = 0.01               # km == 10 m, along +y_hat
_EPS = 0.2                 # km, mollification width (eps/h = 0.1)
_MU = 30.0                 # GPa
# The convention itself is nu-independent, but the guard must not RUN at
# nu = 1/4 alone: lam = mu is exactly where a lam/mu transposition is
# invisible, and that blind spot let a pairing bug ship in this tree for
# months.  An adversarial pass also got a parameter-DEPENDENT sign error
# past a single-point guard (orientation() flipped only for nu > 0.28).
# So every stage runs at BOTH ratios: lam = 30 -> nu = 1/4, lam = 45 ->
# nu = 0.30.  Cost is one extra ~2 ms model per stage.
_LAMS = (30.0, 45.0)
_OBS_D = 0.5               # km, |x| of the straddling observation points

# Hardcoded acceptance windows. Everything below is a SIGN plus a
# factor-of-few magnitude window, never a tight tolerance: the check must
# survive a different BLAS or CPU, and a convention error is a factor of
# -1, not a few percent. Measured values are quoted.
_MIN_JUMP_FRAC = 0.05      # measured 0.237 of SLIP at |x| = 0.5
_MAX_JUMP_FRAC = 1.00      # the jump cannot exceed the slip itself
_EIGEN_LO, _EIGEN_HI = 0.5, 2.0   # x (3/4) mu SLIP / eps; measured -0.993 (removed)

STAGES = ("core", "stress", "compressed")
_REQUIRES = {"stress": "core", "compressed": "core"}
def _nu(lam: float) -> float:
    """Poisson ratio at the guard's fixed mu."""
    return 0.5 * lam / (lam + _MU)


_lock = threading.RLock()
# per stage: "unchecked" | "running" | "ok" | "failed"
_state = {s: "unchecked" for s in STAGES}
_error: dict = {}
_report: dict = {}
_model_cache = None


class FaultConventionError(RuntimeError):
    """The fault sign convention does not match physical reality."""


def _fail(statement: str, measured: str) -> "FaultConventionError":
    from .model.core import FAULT_ORIENTATION
    return FaultConventionError(
        f"mbem refuses to run: the fault sign convention is backwards.\n"
        f"  constant : mbem.model.core.FAULT_ORIENTATION = "
        f"{FAULT_ORIENTATION:+d}\n"
        f"  physical statement that FAILED:\n"
        f"      {statement}\n"
        f"  measured: {measured}\n"
        f"  For a fault with normal n = +x_hat carrying slip "
        f"Du = +{_SLIP} y_hat, msd's fault source gives the jump "
        f"u(+n) - u(-n) = +Du (Patch.value is the Burgers vector), so the "
        f"+x side moves toward +y and the -x side toward -y "
        f"(verify/verify_solved_bvp.py, check A3).\n"
        f"  FAULT_ORIENTATION is the ONE place this convention is stated; "
        f"the solve (mbem/model/equations.py) and the readouts "
        f"(mbem/evaluate.py: evaluate_displacement, evaluate_stress, "
        f"DisplacementEvaluator) all read it through "
        f"RegionModel.orientation. Fix it there, not at a call site.")


# ---------------------------------------------------------------- geometry --

def _panel(p0, e1, e2, n):
    """TriMesh of the parallelogram p0 + s e1 + t e2 (s, t in [0,1]) as
    n x n quads, 2 triangles each, wound so the normal is +cross(e1, e2)."""
    import mollified_bem as mb

    p0, e1, e2 = (np.asarray(v, float) for v in (p0, e1, e2))
    verts = np.array([p0 + (i / n) * e1 + (j / n) * e2
                      for i in range(n + 1) for j in range(n + 1)])
    tris = []
    for i in range(n):
        for j in range(n):
            a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
            tris.append([a, b, b + 1])
            tris.append([a, b + 1, a + 1])
    return mb.TriMesh(vertices=np.ascontiguousarray(verts),
                      triangles=np.ascontiguousarray(np.array(tris, int)))


def _concat(meshes):
    import mollified_bem as mb

    verts, tris, off = [], [], 0
    for m in meshes:
        verts.append(m.vertices)
        tris.append(m.triangles + off)
        off += m.vertices.shape[0]
    return mb.TriMesh(vertices=np.ascontiguousarray(np.vstack(verts)),
                      triangles=np.ascontiguousarray(np.vstack(tris)))


def _build_model(lam: float):
    """Clamped box + free top face + one vertical fault (normal +x_hat)."""
    import mollified_bem as mb
    from .model import BCType, Patch, Region, RegionModel

    L, n = _BOX_HALF, _N
    d = 2.0 * L
    # each face wound so its normal points OUT of [-L, L]^3
    top = _panel([-L, -L, +L], [d, 0, 0], [0, d, 0], n)          # +z
    faces = [_panel([-L, -L, -L], [0, d, 0], [d, 0, 0], n),      # -z
             _panel([+L, -L, -L], [0, d, 0], [0, 0, d], n),      # +x
             _panel([-L, -L, -L], [0, 0, d], [0, d, 0], n),      # -x
             _panel([-L, +L, -L], [0, 0, d], [d, 0, 0], n),      # +y
             _panel([-L, -L, -L], [d, 0, 0], [0, 0, d], n)]      # -y
    rest = _concat(faces)
    # fault in the x = 0 plane, wound so its normal is +x_hat
    f = _FAULT_HALF
    fault = _panel([0.0, -f, -f], [0, 2 * f, 0], [0, 0, 2 * f], n)

    # geometry sanity: the whole check is about these normals
    nt, _ = top.normals_and_areas()
    nf, _ = fault.normals_and_areas()
    assert np.allclose(nt, [0.0, 0.0, 1.0]), "self-check top face mis-wound"
    assert np.allclose(nf, [1.0, 0.0, 0.0]), "self-check fault mis-wound"

    slip = np.broadcast_to(np.array([0.0, _SLIP, 0.0]), (fault.n_triangles, 3))
    p_top = Patch("selfcheck_top", top, BCType.FREE_TRACTION)
    p_rest = Patch("selfcheck_rest", rest, BCType.PRESCRIBED_DISPLACEMENT)
    p_fault = Patch("selfcheck_fault", fault, BCType.FAULT, value=slip)
    region = Region("selfcheck_box",
                    mb.ElasticMaterial(mu=_MU, lam=lam),
                    [p_top, p_rest],
                    probe_point=np.array([0.11, 0.07, -0.13]),
                    faults=[p_fault])
    return RegionModel([region]), region, top, fault


def _model(lam: float):
    """The self-check model at one lam, built once per lam and shared."""
    global _model_cache
    if _model_cache is None:
        _model_cache = {}
    if lam not in _model_cache:
        model, region, top, fault = _build_model(lam)
        # No boundary data: with a zero solution the representation
        # formula reduces to the fault term alone, which is exactly what
        # is on trial in the readout checks.
        zero = {"u:selfcheck_top": np.zeros((top.n_triangles, 3)),
                "t:selfcheck_rest": np.zeros((region.patches[1].n_triangles,
                                              3))}
        _model_cache[lam] = (model, region, top, fault, zero)
    return _model_cache[lam]


# ------------------------------------------------------------------- check --

def _assert_jump(who: str, u: np.ndarray):
    """Statement [1]: u[0] is at (+d,0,0), u[1] at (-d,0,0)."""
    u_plus, u_minus = float(u[0, 1]), float(u[1, 1])
    if not (u_minus < 0.0 < u_plus):
        raise _fail(
            f"{who}: with n = +x_hat and Du = +y_hat, u_y must be > 0 on "
            f"the +x side and < 0 on the -x side",
            f"u_y(+x) = {u_plus:+.4e}, u_y(-x) = {u_minus:+.4e}")
    frac = 0.5 * (abs(u_plus) + abs(u_minus)) / _SLIP
    if not (_MIN_JUMP_FRAC < frac < _MAX_JUMP_FRAC):
        raise _fail(
            f"{who}: the jump across the fault must be a fraction "
            f"{_MIN_JUMP_FRAC}-{_MAX_JUMP_FRAC} of the slip",
            f"|u_y| / |Du| = {frac:.4f}")
    return u_plus, u_minus, frac


def _check_core() -> dict:
    """[1] displacement readout and [4] the solve.

    Swept over _LAMS: a defect that only appears away from nu = 1/4
    must not be able to hide behind a single-material guard.
    """
    return {f'nu={_nu(l):.2f}': _check_core_at(l) for l in _LAMS}


def _check_core_at(_lam: float) -> dict:
    """One material point of _check_core."""
    from .backends.dense import AssembledDense
    from .evaluate import evaluate_displacement
    from .model import generate_system

    model, region, top, fault, zero = _model(_lam)

    # --- [1] displacement readout -----------------------------------
    obs = np.array([[+_OBS_D, 0.0, 0.0], [-_OBS_D, 0.0, 0.0]])
    u = evaluate_displacement(model, region, zero, obs, _EPS, warn_near=False)
    u_plus, u_minus, frac = _assert_jump("evaluate_displacement", u)

    # --- [4] solve ---------------------------------------------------
    # The fault is the only load (every prescribed value is zero), so its
    # RhsTerm IS the right-hand side: a wrong sign negates the solution.
    sol = AssembledDense(generate_system(model), _EPS, "direct",
                         jump="half").solve()
    xc = top.centroids()[:, 0]
    uy = sol["u:selfcheck_top"][:, 1]
    s_plus = float(uy[xc > 0.0].mean())
    s_minus = float(uy[xc < 0.0].mean())
    if not (s_minus < 0.0 < s_plus):
        raise _fail(
            "generate_system + solve: on the free face the solved u_y must "
            "be > 0 where x > 0 and < 0 where x < 0",
            f"u_y(free, x>0) = {s_plus:+.4e}, "
            f"u_y(free, x<0) = {s_minus:+.4e}")

    return {"u_y_plus": u_plus, "u_y_minus": u_minus, "jump_frac": frac,
            "solved_u_y_plus": s_plus, "solved_u_y_minus": s_minus}


def _check_stress() -> dict:
    """[2] the total on-fault stress and [3] the eigenstress branch.

    Swept over _LAMS: a defect that only appears away from nu = 1/4
    must not be able to hide behind a single-material guard.
    """
    return {f'nu={_nu(l):.2f}': _check_stress_at(l) for l in _LAMS}


def _check_stress_at(_lam: float) -> dict:
    """One material point of _check_stress."""
    from .evaluate import evaluate_stress

    model, region, top, fault, zero = _model(_lam)
    fc = fault.centroids()
    s_tot = evaluate_stress(model, region, zero, fc, _EPS,
                            subtract_anelastic=False, warn_near=False)
    s_el = evaluate_stress(model, region, zero, fc, _EPS,
                           subtract_anelastic=True, warn_near=False)
    blob = 0.75 * _MU * _SLIP / _EPS          # Cortez on-fault peak, > 0
    xy_tot = float(np.mean(s_tot[:, 0, 1]))
    xy_eig = float(np.mean(s_el[:, 0, 1] - s_tot[:, 0, 1]))
    if not (xy_tot > 0.0):
        raise _fail(
            "evaluate_stress(subtract_anelastic=False): the TOTAL on-fault "
            "sigma_xy carries +C:eps_star and must be POSITIVE",
            f"sigma_xy = {xy_tot:+.4e} GPa "
            f"(expected about {blob:+.4e})")
    if not (-_EIGEN_HI * blob < xy_eig < -_EIGEN_LO * blob):
        raise _fail(
            "evaluate_stress: subtracting the anelastic term must REMOVE "
            "+C:eps_star from the on-fault sigma_xy, i.e. a NEGATIVE change "
            "of order (3/4) mu |Du| / eps",
            f"sigma_xy(elastic) - sigma_xy(total) = {xy_eig:+.4e} GPa, "
            f"(3/4) mu |Du| / eps = {blob:.4e} GPa")

    return {"sigma_xy_total": xy_tot, "sigma_xy_eigen": xy_eig,
            "blob_peak": blob}


def _check_compressed() -> dict:
    """[5] statement [1] again, through the compressed evaluation operator.

    Swept over _LAMS: a defect that only appears away from nu = 1/4
    must not be able to hide behind a single-material guard.
    """
    return {f'nu={_nu(l):.2f}': _check_compressed_at(l) for l in _LAMS}


def _check_compressed_at(_lam: float) -> dict:
    """One material point of _check_compressed."""
    from .evaluate import DisplacementEvaluator

    model, region, top, fault, zero = _model(_lam)
    obs = np.array([[+_OBS_D, 0.0, 0.0], [-_OBS_D, 0.0, 0.0]])
    u = DisplacementEvaluator(model, region, obs, _EPS)(zero)
    u_plus, u_minus, frac = _assert_jump("DisplacementEvaluator", u)
    return {"u_y_plus": u_plus, "u_y_minus": u_minus, "jump_frac": frac}


_RUNNERS = {"core": _check_core, "stress": _check_stress,
            "compressed": _check_compressed}


def ensure_fault_convention(stage: str = "core") -> None:
    """Pin the fault sign convention, once per process; raise if it moved.

    Called by every entry point that can reach the convention: ``stage``
    selects which of the independently cached stages must have passed
    (``"stress"`` and ``"compressed"`` imply ``"core"``). Cheap after the
    first call (one dict lookup) and re-entrant, because the stages run
    the real solve and readout paths, which call back in here.
    """
    required = _REQUIRES.get(stage)
    if required is not None:
        ensure_fault_convention(required)
    st = _state[stage]
    if st == "ok":
        return
    if st == "failed":
        raise _error[stage]
    with _lock:
        st = _state[stage]
        if st in ("ok", "running"):
            return
        if st == "failed":
            raise _error[stage]
        _state[stage] = "running"
        try:
            _report[stage] = _RUNNERS[stage]()
        except FaultConventionError as exc:
            _state[stage], _error[stage] = "failed", exc
            raise
        except BaseException:
            _state[stage] = "unchecked"    # transient (interrupt, OOM): retry
            raise
        _state[stage] = "ok"


def fault_convention_report() -> dict:
    """Measurements from the stages that have run so far."""
    return {k: dict(v) for k, v in _report.items()}
