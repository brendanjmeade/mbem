"""A FINITE fault in a plate over a substrate: the reference in three
dimensions, and the far field a box's walls can carry.

``layered_box.dip_slip_profile`` is an infinitely long fault. This is the same
layer system for a RECTANGULAR one -- trace on x = 0 for |y| <= ``half_len``,
from the surface down to ``depth``, dipping toward +x, slip at any rake -- at
any point of the plate or of the substrate. Two uses: the surface of a finite
rupture is compared with it, and a box whose sides and base are given this
displacement is a window on the layered half-space instead of a truncation of
it (the walls are what a relaxed plate slides to; docs/viscoelastic.md,
section 8).

The method is ``layered_box``'s, turned about the vertical. In the frame of a
horizontal wavevector (xi along it, t across it, exp(i k xi) dependence)
elasticity splits into the P-SV system of ``_layer_system`` at |k|, unchanged,
and an SH one for the transverse displacement, which buoyancy does not enter.
Slip is an eigenstrain E = sym(b (x) n), and at a source depth it is a jump

    P-SV   [u_xi] = 2 E_xiz
           [u_z] = E_zz + lam (E_xixi + E_tt) / (lam + 2 mu)
           [tau_xiz] = i k (4 mu (lam + mu) E_xixi + 2 mu lam E_tt)
                       / (lam + 2 mu)
    SH     [u_t] = 2 E_tz,   [tau_tz] = 2 i k mu E_xit

of which ``_slip_field`` is the E_tt = 0 case. The fault is integrated along
strike in closed form and down dip by Gauss points; the transform back is a
quadrature in polar wavenumbers, |k| on ``_wavenumbers``' panels (geometric
from tiny k: a relaxed plate's far field decays like 1/r and lives there) and
the azimuth by the trapezoid rule, which is spectrally accurate on a circle.
``half_len = inf`` is the plane-strain / antiplane limit: one azimuth.

Shares nothing with the boundary elements. Real moduli only.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .layered_box import _layer_system, _modes, _wavenumbers, burgers

# What the transforms are resolved to. REACH and TAPER_ORDER are the far-field
# filter of ``displacement`` (its docstring: 1e-4 of the field at 22, fourth
# order in 1 / REACH); SURFACE_DECAYS is how many e-folds of
# exp(-k thickness) ``surface_displacement`` integrates through.
REACH = 22.0
TAPER_ORDER = 4
TAPER_FLOOR = 1.0e-12
SURFACE_DECAYS = 16.0
LONG = 1.0e8                    # km: an infinitely long fault, in closed form
EXACT_LEVELS = 48               # more distinct heights are interpolated
_SPEC_BYTES = 4.0e8             # the spectra of one batch of wavevectors,
_BATCH = 2.0e5                  # and how many wavevectors it may hold


# ---------------------------------------------------------------------
# the uniform half-space, in closed form (Okada 1985)
# ---------------------------------------------------------------------

def half_space_rectangle(xy, half_len, depth, dip, rake, nu):
    """``(n, 3)`` displacement at the surface points ``xy`` of a uniform
    half-space without gravity, per unit slip on the rectangle of this
    module (Okada 1985, eqs. 25-26): the closed form ``surface_displacement``
    is built on. ``rake`` as ``layered_box.burgers``: Okada's U1 is its
    cosine and U2 its sine."""
    xy = np.atleast_2d(np.asarray(xy, float))
    d = np.deg2rad(dip)
    sd, cd = np.sin(d), np.cos(d)
    vertical = abs(cd) < 1.0e-12
    if vertical:
        cd = 0.0
    if np.isinf(half_len):
        half_len, xy = LONG, np.column_stack([xy[:, 0], np.zeros(len(xy))])
    length, width = 2.0 * half_len, depth / sd
    # Okada's axes: x along strike (+y here), y up dip in plan (-x here),
    # origin over the lower corner at y = -half_len
    xo, yo = xy[:, 1] + half_len, depth * cd / sd - xy[:, 0]
    p, q = yo * cd + depth * sd, yo * sd - depth * cd
    alpha = 1.0 - 2.0 * nu                       # mu / (lam + mu)

    def corner(xi, eta):
        yt, dt = eta * cd + q * sd, eta * sd - q * cd
        r = np.sqrt(xi ** 2 + eta ** 2 + q ** 2)
        x_ = np.sqrt(xi ** 2 + q ** 2)
        tiny = 1.0e-300
        # R + eta and R + xi without the cancellation a long fault brings
        r_eta = np.where(eta > 0.0, r + eta,
                         (xi ** 2 + q ** 2) / np.maximum(r - eta, tiny))
        r_xi = np.where(xi > 0.0, r + xi,
                        (eta ** 2 + q ** 2) / np.maximum(r - xi, tiny))
        r_d = np.maximum(r + dt, tiny)
        on = np.abs(q) < 1.0e-14 * (1.0 + r)
        atn = np.where(on, 0.0, np.arctan(xi * eta / np.where(on, 1.0, q * r)))
        log_eta = np.log(np.maximum(r_eta, tiny))
        inv_eta = 1.0 / np.maximum(r * r_eta, tiny)
        inv_xi = 1.0 / np.maximum(r * r_xi, tiny)
        if vertical:
            i5 = -alpha * xi * sd / r_d
            i4 = -alpha * q / r_d
            i3 = 0.5 * alpha * (eta / r_d + yt * q / r_d ** 2 - log_eta)
            i1 = -0.5 * alpha * xi * q / r_d ** 2
        else:
            flat = np.abs(xi) < 1.0e-14 * (1.0 + r)
            i5 = np.where(flat, 0.0, alpha * 2.0 / cd * np.arctan(
                (eta * (x_ + q * cd) + x_ * (r + x_) * sd)
                / np.where(flat, 1.0, xi * (r + x_) * cd)))
            i4 = alpha / cd * (np.log(r_d) - sd * log_eta)
            i3 = alpha * (yt / (cd * r_d) - log_eta) + sd / cd * i4
            i1 = -alpha / cd * xi / r_d - sd / cd * i5
        i2 = -alpha * log_eta - i3
        strike = np.stack([xi * q * inv_eta + atn + i1 * sd,
                           yt * q * inv_eta + q * cd / r_eta + i2 * sd,
                           dt * q * inv_eta + q * sd / r_eta + i4 * sd])
        thrust = np.stack([q / r - i3 * sd * cd,
                           yt * q * inv_xi + cd * atn - i1 * sd * cd,
                           dt * q * inv_xi + sd * atn - i5 * sd * cd])
        return strike, thrust

    total = [0.0, 0.0]
    for xi, eta, sign in ((xo, p, 1.0), (xo, p - width, -1.0),
                          (xo - length, p, -1.0),
                          (xo - length, p - width, 1.0)):
        for i, part in enumerate(corner(xi, eta)):
            total[i] = total[i] + sign * part
    r = np.deg2rad(rake)
    u = -(round(float(np.cos(r)), 15) * total[0]
          + round(float(np.sin(r)), 15) * total[1]) / (2.0 * np.pi)
    return np.column_stack([-u[1], u[0], u[2]])


# ---------------------------------------------------------------------
# the source, in the frame of a wavevector
# ---------------------------------------------------------------------

def _strain(dip, rake, phi):
    """The eigenstrain sym(b (x) n) of unit slip in the frames turned to the
    azimuths ``phi``: its six components, ``(n_phi,)`` each."""
    d = np.deg2rad(dip)
    n = np.array([np.sin(d), 0.0, np.cos(d)])
    b = burgers(1.0, rake, n, np.array([0.0, 1.0, 0.0]))
    c, s = np.cos(phi), np.sin(phi)
    bx, bt, bz = b[0] * c + b[1] * s, b[1] * c - b[0] * s, b[2] + 0.0 * c
    nx, nt, nz = n[0] * c, -n[0] * s, n[2] + 0.0 * c
    return dict(xx=bx * nx, tt=bt * nt, zz=bz * nz,
                xz=0.5 * (bx * nz + bz * nx), tz=0.5 * (bt * nz + bz * nt),
                xt=0.5 * (bx * nt + bt * nx))


def _amplitudes(strain, material):
    """What a source point radiates in an unbounded ``material``: the P-SV
    amplitudes ``(n_phi, 4)``, the pair that decays upward then the pair that
    decays downward, and the SH ones ``(n_phi, 2)``, up then down."""
    mu, lam = material
    e = strain
    jump = np.stack([2.0 * e["xz"] + 0.0j,
                     -1j * (e["zz"] + lam * (e["xx"] + e["tt"])
                            / (lam + 2.0 * mu)),
                     1j * (4.0 * mu * (lam + mu) * e["xx"]
                           + 2.0 * mu * lam * e["tt"]) / (lam + 2.0 * mu),
                     0.0j * e["xx"]], axis=-1)
    here = np.concatenate([_modes(-1, 0.0, mu, lam),
                           -_modes(+1, 0.0, mu, lam)], axis=-1)
    return (np.linalg.solve(here, jump.T).T,
            np.stack([e["tz"] - 1j * e["xt"], -e["tz"] - 1j * e["xt"]],
                     axis=-1))


def _span_basis(k, kx, strike, dip, span, levels, n_gauss):
    """What the stretch ``span`` of the fault (distance down dip) sends to
    the heights ``levels``, before any material: the sums over its source
    points of ``phase e`` and ``phase kz e``, ``e = exp(-k |z - z_s|)``, for
    the points below a level (they reach it decaying UP) and for those above
    it. Four ``(nk, n_phi, n_levels)``. Every mode of a uniform material is a
    combination of these two, so a stack of substrates shares them.

    A level inside the stretch has sources on both sides and the field jumps
    there by the slip: the stretch is cut at that level, each part with its
    own Gauss points, never integrated across."""
    d = np.deg2rad(dip)
    sd, cd = np.sin(d), np.cos(d)
    nk, nf = kx.shape
    basis = np.zeros((4, nk, nf, levels.size), complex)
    node, weight = np.polynomial.legendre.leggauss(n_gauss)

    def add(which, lo, hi, sign):
        if hi - lo <= 0.0 or not which.size:
            return
        n = max(8, int(np.ceil(n_gauss * (hi - lo) / (span[1] - span[0]))) + 4)
        nd, wt = (node, weight) if n >= n_gauss else \
            np.polynomial.legendre.leggauss(n)
        half = 0.5 * (hi - lo)
        xi = lo + half * (1.0 + nd)
        phase = (half * wt) * strike[..., None] * np.exp(
            -1j * cd * kx[..., None] * xi)
        kz = k[:, None, None] * (levels[None, which, None]
                                 + (xi * sd)[None, None, :])
        e = np.exp(sign * kz)
        g = phase @ np.concatenate([e, kz * e], axis=1).transpose(0, 2, 1)
        at = 0 if sign < 0 else 2
        basis[at][:, :, which] += g[..., :which.size]
        basis[at + 1][:, :, which] += g[..., which.size:]

    tol = 1.0e-9 * max(span[1] * sd, 1.0)
    over = levels >= -span[0] * sd - tol
    under = levels <= -span[1] * sd + tol
    add(np.flatnonzero(over), span[0], span[1], -1)
    add(np.flatnonzero(under & ~over), span[0], span[1], +1)
    for i in np.flatnonzero(~(over | under)):
        cut = -levels[i] / sd
        add(np.array([i]), span[0], cut, +1)        # the sources above it
        add(np.array([i]), cut, span[1], -1)        # and below
    return basis


def _radiate(basis, material, amplitudes, rows=4):
    """The field of a stretch in an UNBOUNDED ``material`` from its
    ``_span_basis``: ``(nk, n_phi, n_levels, rows)`` of (U, W, T/k, S/k), and
    V, tau_tz/k. The rows of ``_modes`` written out in ``e`` and ``kz e``."""
    mu, lam = material
    kappa = (lam + 3.0 * mu) / (lam + mu)
    psv, sh = amplitudes
    y = 0.0
    for sign, at, pair in ((-1.0, 0, psv[:, :2]), (1.0, 2, psv[:, 2:])):
        plain = np.array([
            [1.0, 0.0], [-sign, kappa], [2.0 * mu * sign, mu * (1.0 - kappa)],
            [-2.0 * mu, (lam + 2.0 * mu) * sign * (kappa - 1.0)]])[:rows]
        ramp = np.array([[0.0, 1.0], [0.0, -sign], [0.0, 2.0 * mu * sign],
                         [0.0, -2.0 * mu]])[:rows]
        y = (y + basis[at][..., None] * (pair @ plain.T)[None, :, None]
             + basis[at + 1][..., None] * (pair @ ramp.T)[None, :, None])
    up, down = (basis[0] * sh[None, :, 0, None],
                basis[2] * sh[None, :, 1, None])
    return y, up + down, mu * (down - up)


# ---------------------------------------------------------------------
# the layer system at a batch of wavevectors
# ---------------------------------------------------------------------

def _spectra(k, phi, weight, fault, plate, cases, lev_plate, lev_sub,
             n_gauss):
    """The Cartesian displacement spectra ``(nk n_phi, n_levels, 3 n_cases)``
    at radial nodes ``k`` and azimuths ``phi``, each node times its
    ``weight``. A case is ``(substrate, weights, z_bottom)``; what the fault
    sends to each level is computed once for all of them
    (``_span_basis``)."""
    half_len, depth, thick, dip, rake = fault
    nk, nf = k.size, phi.size
    sd = np.sin(np.deg2rad(dip))
    c, s = np.cos(phi), np.sin(phi)
    kx = k[:, None] * c[None, :]
    strike = weight[:, None] * (
        np.ones((nk, nf)) if np.isinf(half_len) else 2.0 * half_len
        * np.sinc(k[:, None] * s[None, :] * half_len / np.pi))
    strain = _strain(dip, rake, phi)
    mu1, lam1 = plate
    nlp, nls = lev_plate.size, lev_sub.size
    span = (0.0, min(depth, thick) / sd)
    edge = _span_basis(k, kx, strike, dip, span, np.array([0.0, -thick]),
                       n_gauss[0])
    amp = _amplitudes(strain, plate)
    yp, vp, tp = _radiate(edge, plate, amp)
    if nlp:
        yl, vl, _ = _radiate(_span_basis(k, kx, strike, dip, span, lev_plate,
                                         n_gauss[0]), plate, amp, rows=2)
    deep = depth > thick
    if deep:                            # the slip below the interface
        span = (thick / sd, depth / sd)
        floors = sorted({cs[2] for cs in cases if cs[2] is not None})
        d_edge = _span_basis(k, kx, strike, dip, span,
                             np.array([-thick] + floors), n_gauss[1])
        d_lev = _span_basis(k, kx, strike, dip, span, lev_sub,
                            n_gauss[1]) if nls else None
    e_h = np.exp(-k * thick)
    # the plate's own modes at its levels: P-SV as _layer_system orders them
    zp = k[:, None] * lev_plate[None, :]
    mp = np.concatenate([_modes(+1, zp, mu1, lam1),
                         _modes(-1, zp + (k * thick)[:, None], mu1, lam1)],
                        axis=-1)
    cf, sf = c[None, :, None], s[None, :, None]
    out = np.empty((nk, nf, nlp + nls, len(cases), 3), complex)
    for i, ((mu2, lam2), (w1, w2), z_bottom) in enumerate(cases):
        a, _top, _bot = _layer_system(k, thick, plate, (mu2, lam2), (w1, w2),
                                      z_bottom)
        n = a.shape[-1]
        g1, g2 = (w1 / (mu1 * k))[:, None], ((w2 - w1) / (mu1 * k))[:, None]
        y_top, y_bot = yp[:, :, 0], yp[:, :, 1]
        rhs = np.zeros((nk, n, nf), complex)
        rhs[:, 0] = -y_top[..., 2] / mu1
        rhs[:, 1] = -(y_top[..., 3] / mu1 + g1 * y_top[..., 1])
        rhs[:, 2], rhs[:, 3] = -y_bot[..., 0], -y_bot[..., 1]
        rhs[:, 4] = -y_bot[..., 2] / mu1
        rhs[:, 5] = y_bot[..., 3] / mu1 - g2 * y_bot[..., 1]
        base = z_bottom is not None
        m = 4 if base else 3
        sh = np.zeros((nk, m, m))
        sh[:, 0, 0], sh[:, 0, 1] = 1.0, -e_h
        sh[:, 1, 0], sh[:, 1, 1], sh[:, 1, 2] = e_h, 1.0, -1.0
        sh[:, 2, 0], sh[:, 2, 1], sh[:, 2, 2] = e_h, -1.0, -mu2 / mu1
        sh_rhs = np.zeros((nk, m, nf), complex)
        sh_rhs[:, 0] = -tp[:, :, 0] / mu1
        sh_rhs[:, 1] = -vp[:, :, 1]
        sh_rhs[:, 2] = -tp[:, :, 1] / mu1
        if base:
            e_d = np.exp(k * (thick + z_bottom))
            sh[:, 1, 3], sh[:, 2, 3] = -e_d, mu2 / mu1 * e_d
            sh[:, 3, 2], sh[:, 3, 3] = e_d, 1.0
        if deep:
            d_amp = _amplitudes(strain, (mu2, lam2))
            yd, vd, td = _radiate(d_edge, (mu2, lam2), d_amp)
            y_int = yd[:, :, 0]
            rhs[:, 2] += y_int[..., 0]
            rhs[:, 3] += y_int[..., 1]
            rhs[:, 4] += y_int[..., 2] / mu1
            rhs[:, 5] -= y_int[..., 3] / mu1
            sh_rhs[:, 1] += vd[:, :, 0]
            sh_rhs[:, 2] += td[:, :, 0] / mu1
            if base:
                at = 1 + floors.index(z_bottom)
                rhs[:, 6], rhs[:, 7] = -yd[:, :, at, 0], -yd[:, :, at, 1]
                sh_rhs[:, 3] = -vd[:, :, at]
        coef = np.linalg.solve(a, rhs)
        sc = np.linalg.solve(sh, sh_rhs)
        u = np.empty((nk, nf, nlp + nls), complex)
        w = np.empty_like(u)
        v = np.empty_like(u)
        if nlp:
            u[..., :nlp] = np.einsum("klj,kjf->kfl", mp[:, :, 0],
                                     coef[:, :4]) + yl[..., 0]
            w[..., :nlp] = np.einsum("klj,kjf->kfl", mp[:, :, 1],
                                     coef[:, :4]) + yl[..., 1]
            v[..., :nlp] = (np.einsum("kl,kf->kfl", np.exp(zp), sc[:, 0])
                            + np.einsum("kl,kf->kfl", np.exp(
                                -zp - (k * thick)[:, None]), sc[:, 1]) + vl)
        if nls:
            zs = k[:, None] * (lev_sub[None, :] + thick)
            ms = _modes(+1, zs, mu2, lam2)[:, :, :2]
            v_s = np.einsum("kl,kf->kfl", np.exp(zs), sc[:, 2])
            if base:
                zb = k[:, None] * (lev_sub[None, :] - z_bottom)
                ms = np.concatenate(
                    [ms, _modes(-1, zb, mu2, lam2)[:, :, :2]], axis=-1)
                v_s = v_s + np.einsum("kl,kf->kfl", np.exp(-zb), sc[:, 3])
            u_s = np.einsum("klj,kjf->kfl", ms[:, :, 0], coef[:, 4:])
            w_s = np.einsum("klj,kjf->kfl", ms[:, :, 1], coef[:, 4:])
            if deep:
                ys, vs, _ = _radiate(d_lev, (mu2, lam2), d_amp, rows=2)
                u_s, w_s, v_s = u_s + ys[..., 0], w_s + ys[..., 1], v_s + vs
            u[..., nlp:], w[..., nlp:], v[..., nlp:] = u_s, w_s, v_s
        out[..., i, 0] = u * cf - v * sf
        out[..., i, 1] = u * sf + v * cf
        out[..., i, 2] = 1j * w
    return out.reshape(nk * nf, nlp + nls, 3 * len(cases))


@njit(cache=True, parallel=True)
def _synthesize(out, x, y, kx, ky, spec, level, share):
    """``out[p] += Re sum_q spec[q, level of p] exp(i k_q . x_p)``, a point's
    levels weighted by ``share``. Parallel over points: call it from the main
    thread only."""
    n_col, n_q, m = out.shape[1], kx.size, level.shape[1]
    for p in prange(x.size):
        acc = np.zeros(n_col)
        for q in range(n_q):
            a = kx[q] * x[p] + ky[q] * y[p]
            ca, sa = np.cos(a), np.sin(a)
            for j in range(m):
                wj = share[p, j]
                if wj != 0.0:
                    lv = level[p, j]
                    for i in range(n_col):
                        f = spec[q, lv, i]
                        acc[i] += wj * (f.real * ca - f.imag * sa)
        for i in range(n_col):
            out[p, i] += acc[i]


def _levels(z, clearance):
    """The heights the spectra are formed at, and each point's
    ``(index, share)`` among them. Few distinct heights are taken as they
    are. More (the centroids of an unstructured wall have hundreds) are
    interpolated on Chebyshev points, enough of them for the nearest
    singularity, which is the fault, ``clearance`` away: 1e-9 of the field
    on a box's walls, for a fifth of the work."""
    if not z.size:
        return np.zeros(0), np.zeros((0, 1), np.int64), np.zeros((0, 1))
    key = np.round(z, 9)
    uniq, idx = np.unique(key, return_inverse=True)
    if uniq.size <= EXACT_LEVELS:
        return uniq, idx[:, None].astype(np.int64), np.ones((z.size, 1))
    lo, hi = float(z.min()), float(z.max())
    b = clearance / (0.5 * (hi - lo))
    n = int(np.clip(np.ceil(16.0 / np.log(b + np.hypot(1.0, b))), 6, 96))
    j = np.arange(n + 1)
    node = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * j / n)
    bary = (-1.0) ** j
    bary[[0, -1]] *= 0.5
    gap = z[:, None] - node[None, :]
    hit = np.abs(gap) < 1.0e-12 * max(hi - lo, 1.0)
    share = bary / np.where(hit, 1.0, gap)
    share /= share.sum(axis=1, keepdims=True)
    on = hit.any(axis=1)
    share[on] = hit[on].astype(float)
    return node, np.broadcast_to(j, share.shape).astype(np.int64), share


def _transform(points, fault, plate, cases, k_max, taper, n_gauss=None,
               oversample: float = 1.0):
    """``(n_cases, n, 3)`` at ``points``: the wavenumber quadrature, in
    batches of radial panels that share an azimuth count.

    Both are set by the fastest oscillation the integrand has, ``k`` times
    the farthest a point lies from the far end of the fault: radial panels a
    fraction of that period wide (``_wavenumbers``) and as many azimuths as
    the circle has cycles. ``oversample`` scales that distance, to show the
    quadrature is converged (it is: the answer does not move)."""
    half_len, depth, thick, dip, _rake = fault
    pts = np.atleast_2d(np.asarray(points, float))
    x, y, z = (np.ascontiguousarray(pts[:, i]) for i in range(3))
    floor = min(cs[2] for cs in cases if cs[2] is not None) \
        if any(cs[2] is not None for cs in cases) else -np.inf
    if np.any(z > 1.0e-9) or np.any(z < floor - 1.0e-9):
        raise ValueError("points must lie in the body: z <= 0, above a base")
    long = np.isinf(half_len)
    d = np.deg2rad(dip)
    reach_x = depth / np.tan(d) if dip != 90.0 else 0.0
    clearance = _clearance(pts, half_len, depth, dip)
    up = z >= -thick - 1.0e-9
    lev_p, idx_p, sh_p = _levels(np.minimum(z[up], 0.0), clearance)
    lev_s, idx_s, sh_s = _levels(z[~up], clearance)
    m = max(idx_p.shape[1], idx_s.shape[1])
    level = np.zeros((z.size, m), np.int64)
    share = np.zeros((z.size, m))
    level[up, :idx_p.shape[1]], share[up, :idx_p.shape[1]] = idx_p, sh_p
    level[~up, :idx_s.shape[1]] = idx_s + lev_p.size
    share[~up, :idx_s.shape[1]] = sh_s

    r_max = float(np.abs(x).max() if long else np.hypot(x, y).max())
    x_max = oversample * max(
        r_max + reach_x + (0.0 if long else half_len), thick)
    k, wk = _wavenumbers(x_max, k_max)
    k, wk = k.reshape(-1, 8), wk.reshape(-1, 8)
    if n_gauss is None:
        # e^{i k xi} on a stretch is resolved by half its phase in points
        sd = np.sin(d)
        spans = (min(depth, thick) / sd, max(depth - thick, 0.0) / sd)
        n_gauss = tuple(int(np.ceil(0.25 * k_max * span)) + 16
                        for span in spans)
    if long:
        count = np.ones(k.shape[0], int)
    else:
        band = k[:, -1] * x_max
        count = np.ceil(0.5 * (band + 6.0 * np.cbrt(band) + 24.0))
        count = (8 * np.ceil(count / 8.0)).astype(int)
    n_lev, n_col = lev_p.size + lev_s.size, 3 * len(cases)
    cap = int(min(_SPEC_BYTES / (16.0 * n_lev * n_col), _BATCH))
    out = np.zeros((z.size, n_col))
    i = 0
    while i < k.shape[0]:
        j = i + 1
        while (j < k.shape[0] and count[j] == count[i]
               and 8 * (j + 1 - i) * count[i] <= cap):
            j += 1
        kb, nf = k[i:j].ravel(), int(count[i])
        scale = wk[i:j].ravel() * (taper(kb) if taper is not None else 1.0)
        if long:                        # one line of wavenumbers: dk / pi
            phi, scale = np.zeros(1), scale / np.pi
        else:                           # half the circle: k dk dphi / 2 pi^2
            phi = np.pi * np.arange(nf) / nf
            scale = scale * kb / (2.0 * np.pi * nf)
        spec = _spectra(kb, phi, scale, fault, plate, cases, lev_p, lev_s,
                        n_gauss)
        _synthesize(out, x, y, np.outer(kb, np.cos(phi)).ravel(),
                    np.outer(kb, np.sin(phi)).ravel(), spec, level, share)
        i = j
    return out.reshape(z.size, len(cases), 3).transpose(1, 0, 2)


def _clearance(points, half_len, depth, dip, n: int = 24):
    """The distance from the nearest of ``points`` to the fault."""
    d = np.deg2rad(dip)
    xi = np.linspace(0.0, depth / np.sin(d), n)
    if np.isinf(half_len):
        gap = np.hypot(points[:, None, 0] - xi * np.cos(d),
                       points[:, None, 2] + xi * np.sin(d))
        return float(gap.min())
    ys = np.linspace(-half_len, half_len, 4 * n)
    best = np.inf
    for a in xi:                        # by rows: n_points x 4n at a time
        gap = np.sqrt((points[:, None, 0] - a * np.cos(d)) ** 2
                      + (points[:, None, 1] - ys) ** 2
                      + (points[:, None, 2] + a * np.sin(d)) ** 2)
        best = min(best, float(gap.min()))
    return best


def _cases(substrate, weights, z_bottom):
    many = np.ndim(substrate[0]) > 0 or isinstance(substrate[0], (tuple, list))
    subs = list(substrate) if many else [substrate]
    for sub in subs:
        if np.iscomplexobj(np.asarray(sub)):
            raise ValueError("real moduli only: a state, not a contour")
    return many, [((float(m), float(l)), tuple(weights), z_bottom)
                  for m, l in subs]


# ---------------------------------------------------------------------
# what the module is for
# ---------------------------------------------------------------------

def displacement(points, half_len, depth, thickness, dip, rake, plate,
                 substrate, weights=(0.0, 0.0), z_bottom=None,
                 reach: float = REACH, order: int = TAPER_ORDER):
    """``(n, 3)`` displacement at ``points`` (x, y, z <= 0: in the plate, in
    the substrate, on a base), per unit slip on a rectangular fault with its
    trace on x = 0 for |y| <= ``half_len``, from the surface down to
    ``depth``, dipping toward +x at ``dip`` degrees, slip at ``rake``
    (``layered_box.burgers``: 90 a thrust, 180 right-lateral). The fault cuts
    an elastic plate ``thickness`` thick over a substrate: a half-space, or a
    layer clamped at ``z_bottom``. Deeper than the plate is thick, it goes on
    through the interface with the same slip, in the substrate's moduli.

    ``plate`` and ``substrate`` are ``(mu, lam)``, ``weights`` their
    ``rho g`` (GPa/km), as in ``dip_slip_profile``. A LIST of substrates
    returns the stack ``(n_sub, n, 3)`` for little more than the price of
    one. ``half_len = inf`` is the infinitely long fault; y is then ignored.

    THE FAR FIELD ONLY. The spectrum is tapered by
    ``exp(-(k clearance / reach)^order)``, ``clearance`` the distance from
    the nearest point to the fault: a smoothing over about
    ``clearance / reach``, which costs a field that varies on the scale of
    its distance from the fault ``23 / reach^4`` of itself -- 1e-4 at the
    default, 1e-5 at 40 for about three times the work. So the points should
    all be far from the fault, a box's walls, and one call should not mix
    near points with far ones: the nearest sets the taper, and the cost, for
    all. Next to the fault use ``surface_displacement``.
    """
    pts = np.atleast_2d(np.asarray(points, float))
    many, cases = _cases(substrate, weights, z_bottom)
    fault = (half_len, depth, thickness, dip, rake)
    kc = reach / _clearance(pts, half_len, depth, dip)
    got = _transform(pts, fault, plate, cases,
                     kc * (-np.log(TAPER_FLOOR)) ** (1.0 / order),
                     lambda k: np.exp(-(k / kc) ** order))
    return got if many else got[0]


def surface_displacement(xy, half_len, depth, thickness, dip, rake, plate,
                         substrate, weights=(0.0, 0.0), z_bottom=None,
                         k_max: float = None):
    """``displacement`` at the SURFACE points ``xy``, good next to the fault
    as well: the half-space closed form (``half_space_rectangle``) plus the
    transform of the DIFFERENCE from the uniform half-space, which is smooth
    because the slip at the trace is common to both -- ``dip_slip_profile``'s
    device. The difference is made at the interface, a plate thickness down,
    so its spectrum falls like exp(-k thickness) and ``k_max`` defaults to
    ``SURFACE_DECAYS`` e-folds of that. The uniform half-space is computed
    with the same fault points in the same two stretches, so a fault through
    the interface needs nothing more.

    One thing does not fall that fast: buoyancy on the free surface, which
    the closed form lacks, leaves a tail ``rho g / (mu k)`` of the trace's
    own spectrum. Cut at the default it is 1e-4 of the slip within a few km
    of the trace and nothing beyond; ``k_max = 4`` takes it to 1e-8 at the
    cost of the wavenumbers."""
    xy = np.atleast_2d(np.asarray(xy, float))
    many, cases = _cases(substrate, weights, z_bottom)
    fault = (half_len, depth, thickness, dip, rake)
    mu, lam = plate
    pts = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy))])
    got = _transform(pts, fault, plate,
                     cases + [((float(mu), float(lam)), (0.0, 0.0), None)],
                     SURFACE_DECAYS / thickness if k_max is None else k_max,
                     None)
    closed = half_space_rectangle(xy, half_len, depth, dip, rake,
                                  0.5 * lam / (lam + mu))
    got = closed[None] + got[:-1] - got[-1][None]
    return got if many else got[0]
