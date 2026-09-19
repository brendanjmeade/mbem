"""verify_eigenstress_exact.py -- the EXACT finite-triangle eigenstress.

On-fault ELASTIC stress is ``sigma_el = sigma_tot - C:eps_star`` (the tree-wide
policy, ``BACKLOG.md``).  The frozen ``anelastic.py`` computes
``C:eps_star`` with NEAREST-TRIANGLE
assignment and the INFINITE-PLANE Cortez marginal
``rho = 0.75 eps^4 / (d^2 + eps^2)^2.5``.  That is the ``d/L -> 0`` limit of the
true finite-triangle integral: right deep inside a large element, and up to ~2x
too large near element EDGES -- i.e. over the whole fault RIM, which is exactly
where crack-tip stress and stress-drop diagnostics live.

``mbem/kernels/tri_kernels.py::eigenstress_contract`` replaces it with the exact
form: the Cortez blob integrated over each actual triangle,

    Phi_eps = (15 eps^4 / 8 pi) * I7,
    sigma*_mn = Phi_eps [ lam d_mn (n . Du) + mu (Du_m n_n + Du_n n_m) ],

summed over ALL fault elements, each with its OWN eps.  ``anelastic.py`` is
UNCHANGED: it stays a frozen oracle, and check [c] below is what certifies it is
still an oracle where it is valid.

Gates (each prints [ok]/[XX]):

  [a] entrywise parity with moss's exact scalar and batch oracles
      (``analytical_eigenstress_kernel`` / ``eigenstress_batch``, loaded by file
      path) at nu = 0.25, 0.30, 0.45 -- single element and summed over a mesh;
  [b] independent parity with ``clq.eigenstress`` (a separate closed-form
      derivation, ``clq/docs/derivation.md``);
  [c] DEEP-INTERIOR agreement with the frozen ``anelastic.py`` inside one large
      element (40 eps across): they MUST agree there -- it is the infinite-plane
      limit;
  [d] NEAR-EDGE DISAGREEMENT with ``anelastic.py``: ratio -> 2 exactly on a
      free patch edge, and > 1.7 at the worst rim element centroid at msd's own
      default eps/h = 1.25.  Without this the gate would pass with the OLD code
      still wired in;
  [e] the SIGN, pinned by finiteness: as eps -> 0 the raw total on-fault shear
      diverges like 1/eps while the corrected one stays bounded, and the
      opposite sign (a DOUBLED eigenstress) diverges too -- plus the
      ``evaluate_stress`` wiring identity (elastic - total) == +C:eps_star;
  [f] PER-ELEMENT / GRADED eps: the sum is element by element, so the
      near-uniform-fault-eps restriction ``evaluate_stress`` used to raise is
      gone; gated against the per-element moss oracle and end to end.

Run from the repo root:  python verify/verify_eigenstress_exact.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mollified_kernel"))
sys.path.insert(0, str(ROOT / "examples"))

import mollified_bem as mb                                          # noqa: E402
from anelastic import eigenstress_at_points                         # noqa: E402
from local_box_mesh_eq import make_vertical_fault_eq                # noqa: E402
from mbem.evaluate import _stress_from_source, evaluate_stress      # noqa: E402
from mbem import defaults                                           # noqa: E402
from mbem.kernels import basis as kb                                # noqa: E402
from mbem.kernels.tri_kernels import eigenstress_contract           # noqa: E402

MU = 30.0                      # GPa
SLIP = 0.001                   # km = 1 m
MOSS = ROOT.parent / "moss" / "mollified_kernel"
CLQ = ROOT.parent / "clq"

CHECKS = []


def check(name, val, tol):
    ok = bool(val < tol)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:58s} {val:9.2e} (tol {tol:.0e})")
    return ok


def check_band(name, val, lo, hi):
    ok = bool(lo < val < hi)
    CHECKS.append(ok)
    print(f"  [{'ok' if ok else 'XX'}] {name:58s} {val:9.4f} "
          f"(want {lo:g} .. {hi:g})")
    return ok


def _load_moss():
    """Load moss's exact eigenstress oracles BY FILE PATH (cross-tree)."""
    if not (MOSS / "analytical_kernels.py").exists():
        return None, None
    sys.path.insert(0, str(MOSS))
    out = []
    for name, fn in (("_moss_eig_ak", "analytical_kernels.py"),
                     ("_moss_eig_ab", "analytical_batch.py")):
        spec = importlib.util.spec_from_file_location(name, str(MOSS / fn))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        out.append(m)
    return out[0], out[1]


def _eig(obs, verts, normals, eps_arr, density, mu, nu):
    """eigenstress_contract with contiguous arrays (N,3,3)."""
    return eigenstress_contract(
        np.ascontiguousarray(np.asarray(obs, float)),
        np.ascontiguousarray(np.asarray(verts, float)),
        np.ascontiguousarray(np.asarray(normals, float)),
        np.ascontiguousarray(np.asarray(eps_arr, float)),
        np.ascontiguousarray(np.asarray(density, float)), mu, nu)


def _boundary_eigen(model, region, sol, obs, eps_spec, mat):
    """+sum_p sigma_p * C:eps_star(u_p) over the region's boundary patches
    -- the part of evaluate_stress's subtraction that is NOT the fault's
    (every double layer is subtracted). Restated by hand
    so the wiring checks below stay independent of evaluate_stress."""
    from mbem.model import BCType
    out = np.zeros((obs.shape[0], 3, 3))
    for p in region.patches:
        u_p = (p.value_array() if p.bc is BCType.PRESCRIBED_DISPLACEMENT
               else sol[f"u:{p.name}"])
        if not np.any(u_p):
            continue
        e = eps_spec[p.name] if isinstance(eps_spec, dict) else eps_spec
        out += float(model.orientation(region, p)) * _stress_from_source(
            obs, p.mesh, u_p, "eigen", mat.mu, mat.nu,
            kb.resolve_eps(e, p.mesh))
    return out


def _eig_tensor(obs, v, n, eps, mu, nu):
    """Full H*[obs,m,n,k] for ONE element, by probing the three slip axes."""
    out = np.zeros((len(obs), 3, 3, 3))
    for k in range(3):
        d = np.zeros((1, 3)); d[0, k] = 1.0
        out[:, :, :, k] = _eig(obs, v[None], n[None], np.full(1, eps), d, mu, nu)
    return out


# ---------------------------------------------------------------------------
# geometry fixtures
# ---------------------------------------------------------------------------
V1 = np.array([0.1, -0.2, 0.0])
V2 = np.array([1.3, 0.0, 0.0])
V3 = np.array([0.6, 1.1, 0.05])
NRM = np.cross(V2 - V1, V3 - V1)
NRM = NRM / np.linalg.norm(NRM)
OBS = np.array([
    [0.5, 0.4, 0.02],       # just off the element plane, inside the footprint
    [0.6, 0.3, 0.0],        # in-plane, inside
    [0.9, -0.1, 0.0],       # in-plane, just OUTSIDE an edge (the failure zone)
    [0.1, -0.2, 0.0],       # exactly on a vertex
    [0.5, 0.4, 0.7],        # off the plane
    [-0.3, 0.9, -0.6],      # off the footprint
    [2.0, 1.0, 1.5],        # far field
])


def a_moss_parity():
    print("\n[a] entrywise parity with moss's exact eigenstress oracles")
    ak, ab = _load_moss()
    if ak is None:
        print(f"  (moss oracle not found at {MOSS} -- SKIPPED)")
        CHECKS.append(False)
        return
    eps = 0.3
    for nu in (0.25, 0.30, 0.45):
        ref = np.array([ak.analytical_eigenstress_kernel(
            o, V1, V2, V3, NRM, MU, nu, eps) for o in OBS])
        bat = ab.eigenstress_batch(V1, V2, V3, NRM, OBS, MU, nu, eps)
        got = _eig_tensor(OBS, np.array([V1, V2, V3]), NRM, eps, MU, nu)
        scale = np.abs(ref).max()
        check(f"nu={nu:.2f} vs moss analytical_eigenstress_kernel",
              np.abs(got - ref).max() / scale, 1e-11)
        check(f"nu={nu:.2f} vs moss eigenstress_batch",
              np.abs(got - bat).max() / scale, 1e-11)

    # summed over a whole mesh: the exact form sums over ALL elements
    fault, _n, s_hat = make_vertical_fault_eq(6.0, (-4.0, 0.0), 2.0)
    verts = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    normals, _ = fault.normals_and_areas()
    nt = fault.n_triangles
    rng = np.random.default_rng(7)
    dens = rng.normal(size=(nt, 3)) * SLIP        # non-uniform slip
    obs = fault.centroids()
    eps = 0.9
    for nu in (0.25, 0.30, 0.45):
        ref = np.zeros((len(obs), 3, 3))
        for s in range(nt):
            H = ab.eigenstress_batch(verts[s, 0], verts[s, 1], verts[s, 2],
                                     normals[s], obs, MU, nu, eps)
            ref += np.einsum("nabk,k->nab", H, dens[s])
        got = _eig(obs, verts, normals, np.full(nt, eps), dens, MU, nu)
        check(f"nu={nu:.2f} {nt}-element sum vs moss batch sum",
              np.abs(got - ref).max() / np.abs(ref).max(), 1e-11)


def b_clq_parity():
    print("\n[b] independent parity with clq.eigenstress (separate derivation)")
    if not (CLQ / "clq" / "api.py").exists():
        print(f"  (clq not found at {CLQ} -- SKIPPED)")
        CHECKS.append(False)
        return
    sys.path.insert(0, str(CLQ))
    import clq                                                  # noqa: E402

    tri = np.array([V1, V2, V3])
    slip = SLIP * np.array([0.7, -0.2, 0.5])
    for nu in (0.25, 0.30, 0.45):
        for eps in (0.3, 0.05):
            ref = clq.eigenstress(OBS, tri, slip, MU, nu, eps)
            got = _eig(OBS, tri[None], NRM[None], np.full(1, eps),
                       slip[None, :], MU, nu)
            check(f"nu={nu:.2f} eps={eps:g} vs clq.eigenstress",
                  np.abs(got - ref).max() / np.abs(ref).max(), 1e-11)


def c_deep_interior():
    print("\n[c] DEEP INTERIOR: exact == frozen anelastic.py (infinite-plane limit)")
    eps = 1.0
    L = 40.0 * eps                      # one element 40 eps across
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([L, 0.0, 0.0])
    c = np.array([L / 2, L * np.sqrt(3) / 2, 0.0])
    mesh = mb.TriMesh(vertices=np.array([a, b, c]), triangles=np.array([[0, 1, 2]]))
    inc = (a + b + c) / 3.0
    slip = np.array([[SLIP, 0.0, 0.0]])           # in-plane (normal = +z)
    for nu in (0.25, 0.30):
        for zoff in (0.0, 0.5, 1.0, 2.0):
            obs = np.ascontiguousarray((inc + np.array([0.0, 0.0, zoff]))[None, :])
            ex = _stress_from_source(obs, mesh, slip, "eigen", MU, nu,
                                     np.full(1, eps))
            ap = eigenstress_at_points(obs, mesh, slip, MU, nu, eps)
            check(f"nu={nu:.2f} incenter + {zoff:.1f} eps: exact vs anelastic.py",
                  np.abs(ex - ap).max() / np.abs(ap).max(), 1e-3)


def d_near_edge_disagreement():
    print("\n[d] NEAR EDGE: exact must DISAGREE with anelastic.py (else the old "
          "code would still pass)")
    # (d1) on a free patch edge the blob is integrated over a HALF plane, so the
    # infinite-plane marginal is exactly 2x too large.
    fault, _n, s_hat = make_vertical_fault_eq(20.0, (-12.0, 0.0), 1.0)
    nt = fault.n_triangles
    slip = np.broadcast_to(SLIP * np.asarray(s_hat, float), (nt, 3)).copy()
    eps = 0.3                                    # << h, so the patch is locally a half plane
    edge = np.array([[0.0, 0.0, -12.0]])         # midpoint of the bottom patch edge
    ex = _stress_from_source(edge, fault, slip, "eigen", MU, 0.30, np.full(nt, eps))
    ap = eigenstress_at_points(edge, fault, slip, MU, 0.30, eps)
    check_band("patch-edge midpoint: anelastic.py / exact",
               float(np.abs(ap).max() / np.abs(ex).max()), 1.95, 2.05)

    # (d2) production geometry: rim element centroids at msd's own eps/h = 1.25.
    fault, _n, s_hat = make_vertical_fault_eq(10.0, (-6.0, 0.0), 0.5)
    nt = fault.n_triangles
    slip = np.broadcast_to(SLIP * np.asarray(s_hat, float), (nt, 3)).copy()
    h = kb.element_sizes(fault)
    # The disagreement bands below were MEASURED at eps/h = 1.25 (the former
    # eps='auto' policy) and are pinned to that literal, not to
    # defaults.EPS_OVER_H: at eps/h = 0.3 the ratio drops to 1.27 and the
    # check would fail for a non-bug.
    eps_arr = 1.25 * h
    e_scalar = float(eps_arr.mean())
    cen = np.ascontiguousarray(fault.centroids())
    ex = _stress_from_source(cen, fault, slip, "eigen", MU, 0.30, eps_arr)
    ap = eigenstress_at_points(cen, fault, slip, MU, 0.30, e_scalar)
    y, z = cen[:, 1], cen[:, 2]
    dist = np.minimum.reduce([5.0 - np.abs(y), z + 6.0, -z])
    rim = dist < 0.5 * h.mean()
    ratio = (np.abs(ap[rim]).max(axis=(1, 2))
             / np.abs(ex[rim]).max(axis=(1, 2)))
    print(f"      {nt} tri, h = {h.mean():.3f}, eps = {e_scalar:.3f} (eps/h = 1.25), "
          f"{int(rim.sum())} rim elements")
    check_band("worst rim element: anelastic.py / exact", float(ratio.max()),
               1.70, 2.30)
    check_band("median rim element: anelastic.py / exact",
               float(np.median(ratio)), 1.25, 2.05)
    peak = np.abs(ex).max()
    frac = float(np.mean(np.abs(ap - ex).max(axis=(1, 2)) > 0.05 * peak))
    print(f"      max |anelastic.py - exact| = {np.abs(ap - ex).max() * 1e3:.2f} MPa "
          f"= {100 * np.abs(ap - ex).max() / peak:.1f}% of peak "
          f"({peak * 1e3:.2f} MPa); {100 * frac:.1f}% of elements > 5%")
    CHECKS.append(bool(frac > 0.05))
    print(f"  [{'ok' if frac > 0.05 else 'XX'}] "
          f"{'fraction of elements differing by > 5% of peak':58s} "
          f"{frac:9.4f} (want > 0.05)")


def e_sign():
    print("\n[e] the SIGN, pinned by finiteness as eps -> 0")
    fault, n_hat, s_hat = make_vertical_fault_eq(20.0, (-12.0, 0.0), 2.0)
    nt = fault.n_triangles
    slip = np.broadcast_to(SLIP * np.asarray(s_hat, float), (nt, 3)).copy()
    n_hat = np.asarray(n_hat, float)
    s_hat = np.asarray(s_hat, float)
    cen = fault.centroids()
    # a deep-interior fault point (far from every rim edge)
    i = int(np.argmin(np.linalg.norm(cen - np.array([0.0, 0.0, -6.0]), axis=1)))
    obs = np.ascontiguousarray(cen[i:i + 1])
    nu = 0.30

    def tau(a):
        return float(np.einsum("nij,i,j->n", a, n_hat, s_hat)[0]) * 1e3   # MPa

    eps_list = (1.0, 0.5, 0.25, 0.125)
    raw, cor, wrong = [], [], []
    for eps in eps_list:
        ea = np.full(nt, eps)
        tot = -_stress_from_source(obs, fault, slip, "dd", MU, nu, ea)
        star = _stress_from_source(obs, fault, slip, "eigen", MU, nu, ea)
        raw.append(tau(tot))                 # uncorrected TOTAL
        cor.append(tau(tot + star))          # SHIPPED sign: the fault term is
        wrong.append(tau(tot - star))        #   -Sdd@slip, so removing C:eps* ADDS it
    for eps, r, c, w in zip(eps_list, raw, cor, wrong):
        print(f"      eps={eps:6.3f}  total {r:10.3f}  corrected {c:8.3f}  "
              f"wrong-sign {w:10.3f}   MPa")
    # raw ~ 1/eps over the 8x sweep; corrected flat; wrong sign ~ -1/eps
    check_band("raw total growth over eps 1 -> 0.125 (want ~8x)",
               abs(raw[-1]) / abs(raw[0]), 5.0, 11.0)
    check_band("CORRECTED growth over eps 1 -> 0.125 (want ~1x)",
               abs(cor[-1]) / abs(cor[0]), 0.80, 1.25)
    check_band("wrong-sign (doubled) growth -- must NOT be bounded",
               abs(wrong[-1]) / abs(wrong[0]), 5.0, 11.0)
    check_band("|wrong-sign| / |corrected| at the smallest eps",
               abs(wrong[-1]) / abs(cor[-1]), 3.0, 1e9)

    # the evaluate_stress WIRING: (elastic - total) must be +C:eps_star exactly
    from _fault_box import build_fault_box, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system

    mat = mb.ElasticMaterial(mu=30.0, lam=30.0)
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0, fault_half_len=20.0,
                             fault_depth=18.0, edge_fault=3.0, edge_near=24.0,
                             edge_far=50.0, edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mat)
    system = generate_system(model)
    region = model.regions[0]
    fmesh = meshes["fault"]
    fslip = np.broadcast_to(0.01 * np.asarray(meshes["s_hat"], float),
                            (fmesh.n_triangles, 3)).copy()
    call = fmesh.centroids()
    # three deep-interior centroids and three at the BOTTOM RIM, where the old
    # approximation is ~2x off -- the rim points are what make the "no longer
    # anelastic.py" tripwire below bite.
    sel = np.concatenate([
        np.argsort(np.linalg.norm(call - [0, 0, -9.0], axis=1))[:3],
        np.argsort(np.linalg.norm(call - [0, 0, -18.0], axis=1))[:3]])
    obs_b = np.ascontiguousarray(call[sel])
    eps = 3.0
    asm = AssembledDense(system, eps, "direct", jump="calibrated")
    sol = asm.solve()
    tot = evaluate_stress(model, region, sol, obs_b, eps,
                          subtract_anelastic=False, warn_near=False)
    ela = evaluate_stress(model, region, sol, obs_b, eps,
                          subtract_anelastic=True, warn_near=False)
    star = _stress_from_source(obs_b, fmesh, fslip, "eigen", mat.mu, mat.nu,
                               np.full(fmesh.n_triangles, eps))
    star_f = star.copy()                      # the fault's own eigenstress
    star = star + _boundary_eigen(model, region, sol, obs_b, eps, mat)
    check("evaluate_stress: (elastic - total) == +C:eps_star (fault + patches)",
          np.abs((ela - tot) - star).max() / np.abs(star).max(), 1e-12)
    # the boundary part must be a genuine contribution at these points (the
    # top row of the fault is 3 eps below host_top), else the check above
    # could not tell the patch subtraction from its absence
    d_bdy = np.abs(star - star_f).max() / np.abs(star_f).max()
    CHECKS.append(bool(d_bdy > 1e-6))
    print(f"  [{'ok' if d_bdy > 1e-6 else 'XX'}] "
          f"{'boundary-patch eigenstress is non-negligible here':58s} "
          f"{d_bdy:9.2e} (want > 1e-6)")
    star = star_f
    # and it must NOT be the old approximate one any more
    old = eigenstress_at_points(obs_b, fmesh, fslip, mat.mu, mat.nu, eps)
    d_old = np.abs((ela - tot) - old).max() / np.abs(star).max()
    CHECKS.append(bool(d_old > 0.1))
    print(f"  [{'ok' if d_old > 0.1 else 'XX'}] "
          f"{'evaluate_stress no longer subtracts anelastic.py':58s} "
          f"{d_old:9.2e} (want > 1e-1)")


def f_graded_eps():
    print("\n[f] PER-ELEMENT / GRADED eps (the near-uniform restriction is lifted)")
    ak, ab = _load_moss()
    fault, _n, s_hat = make_vertical_fault_eq(6.0, (-4.0, 0.0), 1.0)
    nt = fault.n_triangles
    verts = np.asarray(fault.vertices, float)[np.asarray(fault.triangles)]
    normals, _ = fault.normals_and_areas()
    slip = np.broadcast_to(SLIP * np.asarray(s_hat, float), (nt, 3)).copy()
    obs = np.ascontiguousarray(fault.centroids())
    cz = fault.centroids()[:, 2]
    graded = 0.4 + 1.2 * (cz - cz.min()) / (cz.max() - cz.min())   # 0.4 -> 1.6
    nu = 0.30

    # additivity: the per-element eps really is routed per element -- splitting
    # the mesh in two and summing the halves must reproduce the single call
    half = nt // 2
    both = _eig(obs, verts, normals, graded, slip, MU, nu)
    lo = _eig(obs, verts[:half], normals[:half], graded[:half], slip[:half],
              MU, nu)
    hi = _eig(obs, verts[half:], normals[half:], graded[half:], slip[half:],
              MU, nu)
    check("per-element eps routing: split-mesh sum == single call",
          np.abs(both - (lo + hi)).max() / np.abs(both).max(), 1e-14)

    # the graded sum, element by element, against the moss oracle at each eps
    if ab is not None:
        ref = np.zeros((len(obs), 3, 3))
        for s in range(nt):
            H = ab.eigenstress_batch(verts[s, 0], verts[s, 1], verts[s, 2],
                                     normals[s], obs, MU, nu, float(graded[s]))
            ref += np.einsum("nabk,k->nab", H, slip[s])
        got = _eig(obs, verts, normals, graded, slip, MU, nu)
        check(f"graded eps {graded.min():.2f}..{graded.max():.2f} vs per-element "
              f"moss oracle", np.abs(got - ref).max() / np.abs(ref).max(), 1e-11)
    else:
        print("  (moss oracle not found -- per-element parity SKIPPED)")
        CHECKS.append(False)

    # a graded eps must CHANGE the answer (otherwise the test is vacuous)
    flat = _eig(obs, verts, normals, np.full(nt, float(graded.mean())), slip,
                MU, nu)
    got = _eig(obs, verts, normals, graded, slip, MU, nu)
    rel = float(np.abs(got - flat).max() / np.abs(flat).max())
    CHECKS.append(bool(rel > 0.05))
    print(f"  [{'ok' if rel > 0.05 else 'XX'}] "
          f"{'graded eps differs from its own mean':58s} {rel:9.2e} (want > 5e-2)")

    # end to end: evaluate_stress used to RAISE on a graded fault eps
    from _fault_box import build_fault_box, build_model
    from mbem.backends.dense import AssembledDense
    from mbem.model import generate_system

    mat = mb.ElasticMaterial(mu=30.0, lam=30.0)
    meshes = build_fault_box(half_x=100.0, z_bottom=-60.0, fault_half_len=20.0,
                             fault_depth=18.0, edge_fault=3.0, edge_near=24.0,
                             edge_far=50.0, edge_side=50.0, near_field_radius=50.0)
    model = build_model(meshes, 0.01, mat)
    system = generate_system(model)
    region = model.regions[0]
    fmesh = meshes["fault"]
    fz = fmesh.centroids()[:, 2]
    fgrad = 1.0 + 2.0 * (fz - fz.min()) / (fz.max() - fz.min())    # 1 -> 3, 3x spread
    spec = {"top": 3.0, "sides": 3.0, "base": 3.0, "fault": fgrad}
    asm = AssembledDense(system, spec, "direct", jump="calibrated")
    sol = asm.solve()
    call = fmesh.centroids()
    sel = np.argsort(np.linalg.norm(call - [0, 0, -9.0], axis=1))[:6]
    obs_b = np.ascontiguousarray(call[sel])
    try:
        tot = evaluate_stress(model, region, sol, obs_b, spec,
                              subtract_anelastic=False, warn_near=False)
        ela = evaluate_stress(model, region, sol, obs_b, spec,
                              subtract_anelastic=True, warn_near=False)
        ran = True
    except ValueError as exc:
        print(f"      evaluate_stress raised on a graded fault eps: {exc}")
        ran = False
    CHECKS.append(ran)
    print(f"  [{'ok' if ran else 'XX'}] "
          f"{'evaluate_stress accepts a graded fault eps (3x spread)':58s}")
    if ran:
        fslip = np.broadcast_to(0.01 * np.asarray(meshes["s_hat"], float),
                                (fmesh.n_triangles, 3)).copy()
        star = _stress_from_source(obs_b, fmesh, fslip, "eigen", mat.mu, mat.nu,
                                   fgrad)
        star = star + _boundary_eigen(model, region, sol, obs_b, spec, mat)
        check("graded end to end: (elastic - total) == +C:eps_star",
              np.abs((ela - tot) - star).max() / np.abs(star).max(), 1e-12)


def main():
    print("=" * 76)
    print("Exact finite-triangle eigenstress (mbem eigenstress_contract)")
    print("=" * 76)
    a_moss_parity()
    b_clq_parity()
    c_deep_interior()
    d_near_edge_disagreement()
    e_sign()
    f_graded_eps()
    print("-" * 76)
    if all(CHECKS):
        print(f"PASS: exact finite-triangle eigenstress ({len(CHECKS)} checks)")
    else:
        print(f"FAIL: exact finite-triangle eigenstress "
              f"({sum(1 for c in CHECKS if not c)} of {len(CHECKS)} checks failed)")
    return all(CHECKS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
