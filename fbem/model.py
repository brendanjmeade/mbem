"""Force-element (equivalent body force) BEM for a piecewise-uniform solid.

Unknown: one constant force density q (force per unit AREA) per boundary
triangle, collocated at centroids, in a UNIFORM reference medium (the host).
The inclusion enters as a polarization density on its interface; the outer
boundary conditions enter as truncation densities.  There is no (u, t) pair
and no rigid-body nullspace on the inclusion: 3 unknowns per triangle, not 6.

Representation
--------------
    u(x) = u_F(x) + int_S G^eps(x, y) q(y) dS(y)

with u_F the fault's displacement field in the UNIFORM host.  The stress of
this representation is the HOST stress everywhere; the TRUE stress inside the
inclusion is ``a`` times it, with ``a = mu_inc/mu_host``.  This is legitimate
only because both materials have the same Poisson ratio here, so
``C_inc = a C_host`` exactly and the interface polarization has no volume
term (``div(C0:(eps - eps*))`` inside the inclusion is proportional to
``lam0 mu1 - lam1 mu0 = 0``).  With unequal Poisson ratios a volume source
appears; see README.

Rows
----
With ``n`` the field normal, ``t_on = B q + t_F`` the on-surface (average)
traction, and the layer's jump ``t(+n) - t(-n) = -q``:

  R1 free traction, solid on the -n side:    (1/2) q + B q = -t_F
  R2 interface, n out of the inclusion:      ((1+a)/2) q - (1-a) B q = (1-a) t_F
  R3 prescribed displacement u = 0:          [G - (eps/4) M] q = -u_F

R2 is traction continuity of the TRUE traction, ``t(+n) = a t(-n)``:
``t_on - q/2 = a (t_on + q/2)``.  It degenerates correctly at both ends --
``a = 1`` gives ``q = 0`` (no interface), ``a = 0`` gives ``t(+n) = 0`` (a
cavity).  The inclusion's OUTCROP (inclusion_top) is a free surface of the
soft material, ``a sigma . n = 0``, and since ``a != 0`` that is exactly R1:
the contrast cancels and no special row is needed there.

Displacement is continuous across a single layer, so the interface needs no
displacement equation at all -- that is the whole economy of the method.

Sign convention: clq's, ``div sigma + f phi_eps = 0``, so a closed surface
around an element carries ``int sigma.n dS = -int f dS``.  The
``body_forces_bem`` drafts use the opposite sign; every density here is minus
theirs.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
from scipy.linalg import get_lapack_funcs, lu_factor, lu_solve

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "clq"))
import clq                                                       # noqa: E402

from assembly import (traction_matrix, displacement_matrix,       # noqa: E402
                      apply_displacement, eps_bias_operator)

FREE = "free"            # R1
INTERFACE = "interface"  # R2
FIXED = "fixed"          # R3
ROWS = (FREE, INTERFACE, FIXED)


# --------------------------------------------------------------------------
# geometry bookkeeping
# --------------------------------------------------------------------------
class Patch:
    """A triangulated surface with a row type and an enforced normal sense.

    The winding of the incoming mesh is NOT trusted: ``orient`` states the
    convention geometrically and every normal is flipped to satisfy it.  For
    an outer free surface and for a Dirichlet surface the normal points OUT
    OF THE SOLID; for an interface it points OUT OF THE INCLUSION.
    """

    def __init__(self, name, verts, tris, row, orient):
        if row not in ROWS:
            raise ValueError(f"unknown row type {row!r}")
        self.name, self.row, self.orient = name, row, orient
        self.tv = np.ascontiguousarray(np.asarray(verts, float)[np.asarray(tris)])
        self.centroids = self.tv.mean(axis=1)
        cr = np.cross(self.tv[:, 1] - self.tv[:, 0], self.tv[:, 2] - self.tv[:, 0])
        self.area = 0.5 * np.linalg.norm(cr, axis=1)
        if not (self.area > 0).all():
            raise ValueError(f"{name}: degenerate triangle")
        self.normals = cr / (2.0 * self.area)[:, None]
        self._orient()

    @property
    def n(self):
        return len(self.tv)

    def _orient(self):
        c, n = self.centroids, self.normals
        kind = self.orient[0]
        if kind == "up":
            want = n[:, 2] > 0
        elif kind == "down":
            want = n[:, 2] < 0
        elif kind == "radial":                    # outward from a vertical axis
            r = c[:, :2] - np.asarray(self.orient[1], float)[None, :]
            want = np.einsum("mi,mi->m", n[:, :2], r) > 0
        else:
            raise ValueError(kind)
        self.flipped = int((~want).sum())
        self.normals = np.where(want[:, None], n, -n)


class ForceElementModel:
    def __init__(self, patches, mu, nu, eps, alpha):
        self.patches = list(patches)
        self.mu, self.nu, self.eps, self.alpha = mu, nu, float(eps), alpha
        cat = lambda a: np.ascontiguousarray(np.concatenate(a))
        self.tv = cat([p.tv for p in self.patches])
        self.centroids = cat([p.centroids for p in self.patches])
        self.normals = cat([p.normals for p in self.patches])
        self.area = np.concatenate([p.area for p in self.patches])
        self.N = len(self.tv)
        self.eps_arr = np.full(self.N, self.eps)
        off, self.slices = 0, {}
        for p in self.patches:
            self.slices[p.name] = slice(off, off + p.n)
            off += p.n
        self.row_of = np.concatenate([[p.row] * p.n for p in self.patches])
        self.q = None

    def rows(self, kind):
        return np.flatnonzero(self.row_of == kind)

    @staticmethod
    def _dof(elems):
        return (3 * np.asarray(elems)[:, None] + np.arange(3)[None, :]).ravel()

    def sel(self, *names):
        return np.concatenate([np.arange(self.N)[self.slices[n]] for n in names])

    # ---------------- assembly ----------------
    def traction_block(self, verbose=True):
        """B: the adjoint-double-layer block.  Independent of alpha, so the
        het and hom solves of a decomposition share one assembly."""
        t0 = time.time()
        if verbose:
            print(f"  traction block {3*self.N}^2 = "
                  f"{(3*self.N)**2*8/1e9:.2f} GB ...", flush=True)
        B = traction_matrix(self.centroids, self.normals, self.tv,
                            self.eps_arr, self.mu, self.nu)      # F-ordered
        if verbose:
            print(f"  B assembled in {time.time()-t0:.1f}s", flush=True)
        return B

    def assemble(self, t_F, u_F, verbose=True, row_block=2048, B=None):
        """Build the (3N, 3N) Fortran-ordered system matrix and RHS.

        ``B`` may be a previously assembled traction block; it is CONSUMED
        (modified in place and then owned by the model), so pass a copy to
        reuse it -- and that copy must be ``B.copy(order="F")``.  Plain
        ``B.copy()`` returns C order even from an F-ordered array, which
        silently costs the in-place LU (and trips the assert below).
        """
        N, a, t0 = self.N, self.alpha, time.time()
        A = self.traction_block(verbose=verbose) if B is None else B
        assert A.flags.f_contiguous
        b = np.zeros(3 * N)
        free, inter, fixed = (self.rows(k) for k in ROWS)
        if verbose:
            print(f"  rows: {len(free)} free, {len(inter)} interface, "
                  f"{len(fixed)} fixed;  alpha = {a}", flush=True)

        # R2 first: it rescales whole rows, so it must not see R1's free term.
        if len(inter):
            di = self._dof(inter)
            for s in range(0, len(di), row_block):          # blocked: no 1.5 GB temp
                sl = di[s:s + row_block]
                A[sl, :] *= -(1.0 - a)
            A[di, di] += 0.5 * (1.0 + a)
            b[di] = (1.0 - a) * np.asarray(t_F)[inter].ravel()

        # R1
        d = self._dof(free)
        A[d, d] += 0.5
        b[d] = -np.asarray(t_F)[free].ravel()

        # R3, row-equilibrated (G and B have different units)
        self.fixed_scale = None
        if len(fixed):
            df = self._dof(fixed)
            G = displacement_matrix(self.centroids[fixed], self.tv,
                                    self.eps_arr, self.mu, self.nu)
            M = eps_bias_operator(self.normals[fixed], self.mu, self.nu)
            for k, e in enumerate(fixed):                  # remove the O(eps) self bias
                G[3 * k:3 * k + 3, 3 * e:3 * e + 3] -= 0.25 * self.eps * M[k]
            s = 1.0 / np.abs(G).max(axis=1)
            A[df, :] = G * s[:, None]
            b[df] = -np.asarray(u_F)[fixed].ravel() * s
            self.fixed_scale = s
            del G
        self.A, self.b = A, b
        if verbose:
            print(f"  system built in {time.time()-t0:.1f}s", flush=True)
        return A, b

    def solve(self, verbose=True, keep_lu=False):
        t0 = time.time()
        if verbose:
            print(f"  LU {self.A.shape[0]}^2 in place ...", flush=True)
        anorm = float(np.linalg.norm(self.A, 1))
        lu, piv = lu_factor(self.A, overwrite_a=True)
        self.A = None
        gecon = get_lapack_funcs(("gecon",), (lu,))[0]      # free from the LU
        rcond, info = gecon(lu, anorm, norm="1")
        self.cond_estimate = (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
        self.q = lu_solve((lu, piv), self.b).reshape(self.N, 3)
        self._lu = (lu, piv) if keep_lu else None
        if not keep_lu:
            del lu
        if verbose:
            print(f"  solved in {time.time()-t0:.1f}s; "
                  f"cond {self.cond_estimate:.3e}; "
                  f"|q|max {np.abs(self.q).max():.4e}", flush=True)
        return self.q

    # ---------------- evaluation ----------------
    def displacement(self, obs, q=None, self_elems=None, correct_eps=True,
                     chunk=512, verbose=False):
        """u at obs from the densities (the fault field is added by the caller).

        ``self_elems``: for an observation point that IS the centroid of source
        element m, the mollified layer carries a removable O(eps) bias,
        u_on = u_sharp + (eps/4) g; pass the element index per observation
        (or -1 for none) to subtract it.
        """
        q = self.q if q is None else q
        u = apply_displacement(obs, self.tv, self.eps_arr, self.mu, self.nu, q,
                               chunk=chunk, verbose=verbose)
        if correct_eps and self_elems is not None:
            e = np.asarray(self_elems)
            ok = e >= 0
            M = eps_bias_operator(self.normals[e[ok]], self.mu, self.nu)
            u[ok] -= 0.25 * self.eps * np.einsum("mij,mj->mi", M, q[e[ok]])
        return u


# --------------------------------------------------------------------------
# source field
# --------------------------------------------------------------------------
def fault_source(obs, normals, fault_tv, slip, mu, nu, eps, verbose=True,
                 report=200):
    """Displacement and traction at obs from a slip patch in the UNIFORM host.

    SIGN: mbem assembles the fault with scale -1 on the right-hand side, so the
    `value` stored on a FAULT patch is -Delta_u in clq's convention
    (Delta u = u(+nhat) - u(-nhat)).  The caller must negate it; the driver
    does, and gates the choice against the cache.
    """
    obs = np.ascontiguousarray(np.asarray(obs, float))
    slip = np.asarray(slip, float)
    u = np.zeros((len(obs), 3))
    sig = np.zeros((len(obs), 3, 3))
    t0 = time.time()
    for k, tri in enumerate(fault_tv):
        s = slip if slip.ndim == 1 else slip[k]
        u += clq.displacement(obs, tri, s, mu, nu, eps)
        sig += clq.stress(obs, tri, s, mu, nu, eps)
        if verbose and (k + 1) % report == 0:
            print(f"    fault {k+1}/{len(fault_tv)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return u, np.einsum("nij,nj->ni", sig, normals)
