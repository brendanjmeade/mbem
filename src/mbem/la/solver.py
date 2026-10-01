"""Flexible right-preconditioned GMRES with honest reporting.

Replaces the bare ``scipy.sparse.linalg.gmres(rtol=1e-10, ...)`` calls of
the legacy stack. Differences that matter:

* RIGHT preconditioning — the iteration's residual is the residual of
  the actual system, and convergence is confirmed against the TRUE
  relative residual ||b - A x|| / ||b|| before reporting success.
* Flexible (FGMRES) — the preconditioner may change between iterations
  (inner iterative solves, rebuilt ladder rungs).
* Stagnation detection — if the residual fails to improve by
  ``stagnation_factor`` over ``stagnation_window`` iterations, the solve
  stops and says so instead of burning maxiter.
* Every solve returns a ``SolveReport``; nothing is silently swallowed.
* Krylov subspace RECYCLING across a sequence of solves (GCRO-DR), for
  the material sweeps: pass a ``RecycleSpace``.

Real systems only (both backends assemble real operators).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .. import defaults


@dataclass
class SolveReport:
    converged: bool
    iterations: int
    true_relres: float
    arnoldi_relres: float
    stagnated: bool = False
    restarts: int = 0
    residual_history: list = field(default_factory=list)
    # ``BlockGaussSeidel.summary()`` of the preconditioner the solve used
    # (rung per super-block, sizes, build time); None for an
    # unpreconditioned solve. Attached by the backend, not by ``fgmres``.
    precond_summary: dict | None = None
    # Recycling (GCRO-DR): dimension of the subspace carried OUT of this
    # solve, and the matvecs spent on the subspace carried IN (k of them,
    # with k preconditioner applications, re-deriving U = M Y and
    # C = A U). Those are not iterations and are not counted as such, so
    # an iteration count from a recycled solve is only half the cost
    # story.
    recycle_dim: int = 0
    setup_matvecs: int = 0

    def __str__(self):
        status = "converged" if self.converged else (
            "STAGNATED" if self.stagnated else "NOT CONVERGED")
        s = (f"FGMRES {status}: {self.iterations} iters, "
             f"true relres {self.true_relres:.3e}")
        if self.recycle_dim or self.setup_matvecs:
            s += (f"; recycle dim {self.recycle_dim} "
                  f"(+{self.setup_matvecs} setup matvecs)")
        if self.precond_summary:
            counts: dict = {}
            for sb in self.precond_summary["super_blocks"]:
                counts[sb["rung"]] = counts.get(sb["rung"], 0) + 1
            rungs = ", ".join(f"{n} {r}" for r, n in counts.items())
            s += (f"; preconditioner {rungs}, built in "
                  f"{self.precond_summary['build_s']:.1f} s")
        return s


@dataclass
class RecycleSpace:
    """The subspace GCRO-DR carries from one solve of a sequence to the next.

    ``vectors`` (n x k') live in the domain of the PRECONDITIONED
    operator A M -- that is the spectrum FGMRES sees, and its outliers
    are what recycling deflates. (Recycling solution-space vectors
    instead, the natural choice for a flexible GCRO-DR, makes the
    harmonic Ritz values those of A rather than of A M: it deflates the
    wrong spectrum and costs iterations.) Nothing about the operator or
    the preconditioner is stored with them: each solve re-derives
    U = M Y and C = A U for its own A and M, so both may change between
    solves. Within a solve the same space is what the deflated restart
    carries between cycles.

    Memory: k n floats between solves; 3 k n while a solve runs (Y, U
    and C). Setup cost per solve: k preconditioner applications and k
    matvecs, which is why recycling is opt-in.

    Reset (``reset()``, i.e. the next solve starts from nothing) when the
    vectors no longer describe that subspace: a changed unknown layout (a
    length mismatch is caught here, a re-meshed model by the caller) and
    a material step outside the preconditioner-reuse window
    (``backends/hmat.py``; beyond it the outliers have moved and the
    setup work would be spent on a stale space).
    """

    dim: int = defaults.GCRO_RECYCLE_DIM
    vectors: np.ndarray | None = None

    def reset(self):
        self.vectors = None

    def store(self, U: np.ndarray | None):
        """Keep the span, with unit columns (only the span matters: the
        next solve re-orthonormalizes against its own operator)."""
        if U is None or U.size == 0:
            self.vectors = None
            return
        norms = np.linalg.norm(U, axis=0)
        keep = norms > 0.0
        self.vectors = np.ascontiguousarray(U[:, keep] / norms[keep])


def fgmres(A, b, M=None, x0=None,
           rtol: float = defaults.GMRES_RTOL,
           restart: int = defaults.GMRES_RESTART,
           maxiter: int = defaults.GMRES_MAXITER,
           stagnation_window: int = defaults.GMRES_STAGNATION_WINDOW,
           stagnation_factor: float = defaults.GMRES_STAGNATION_FACTOR,
           callback=None, recycle: RecycleSpace | None = None):
    """Solve A x = b. Returns (x, SolveReport).

    ``A`` and ``M`` are callables v -> A@v / M@v (or objects with a
    ``matvec``/``__matmul__``); ``M`` approximates A^{-1} (right
    preconditioner).

    ``recycle`` switches the iteration to GCRO-DR (``_gcrodr``), which
    augments the Krylov space with that subspace and refreshes it in
    place; the same ``RecycleSpace`` passed to a sequence of solves is
    what recycling means.
    """
    A_mv = _as_matvec(A)
    M_mv = _as_matvec(M) if M is not None else (lambda v: v)

    b = np.asarray(b)
    if np.iscomplexobj(b):
        raise TypeError("fgmres is real-only")
    n = b.shape[0]
    dtype = np.float64
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return np.zeros(n, dtype=dtype), SolveReport(True, 0, 0.0, 0.0)

    x = np.zeros(n, dtype=dtype) if x0 is None else np.array(x0, dtype=dtype)

    if recycle is not None:
        return _gcrodr(A_mv, M_mv, b, x, recycle, rtol, restart, maxiter,
                       stagnation_window, stagnation_factor, callback)

    history: list[float] = []
    total_iters = 0
    restarts = 0
    best_res = np.inf
    best_res_at = 0
    stagnated = False

    while total_iters < maxiter and not stagnated:
        r = b - A_mv(x)
        beta = np.linalg.norm(r)
        arnoldi_res = beta / b_norm
        if arnoldi_res < rtol:
            break

        m = min(restart, maxiter - total_iters)
        V = np.empty((m + 1, n), dtype=dtype)
        Z = np.empty((m, n), dtype=dtype)
        H = np.zeros((m + 1, m), dtype=dtype)
        cs = np.zeros(m, dtype=dtype)
        sn = np.zeros(m, dtype=dtype)
        g = np.zeros(m + 1, dtype=dtype)
        V[0] = r / beta
        g[0] = beta

        j_done = 0
        for j in range(m):
            Z[j] = M_mv(V[j])
            w = A_mv(Z[j])
            # Modified Gram-Schmidt
            for i in range(j + 1):
                H[i, j] = np.vdot(V[i], w)
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-300:
                V[j + 1] = w / H[j + 1, j]

            # Apply accumulated Givens rotations, then form a new one
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            denom = np.sqrt(np.abs(H[j, j]) ** 2 + np.abs(H[j + 1, j]) ** 2)
            if denom == 0.0:
                j_done = j + 1
                break
            cs[j] = np.abs(H[j, j]) / denom if np.abs(H[j, j]) > 0 else 0.0
            if np.abs(H[j, j]) > 0:
                phase = H[j, j] / np.abs(H[j, j])
                sn[j] = phase * H[j + 1, j] / denom
            else:
                sn[j] = 1.0
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            total_iters += 1
            j_done = j + 1
            arnoldi_res = float(np.abs(g[j + 1])) / b_norm
            history.append(arnoldi_res)
            if callback is not None:
                callback(total_iters, arnoldi_res)

            # Stagnation bookkeeping
            if arnoldi_res < best_res / stagnation_factor:
                best_res = arnoldi_res
                best_res_at = total_iters
            elif total_iters - best_res_at >= stagnation_window:
                stagnated = True
                break

            if arnoldi_res < rtol or total_iters >= maxiter:
                break

        # Form the restart/final iterate
        if j_done > 0:
            y = _solve_upper(H[:j_done, :j_done], g[:j_done])
            x = x + Z[:j_done].T @ y
        restarts += 1

        if stagnated:
            break

    true_relres = float(np.linalg.norm(b - A_mv(x)) / b_norm)
    arnoldi_final = history[-1] if history else true_relres
    converged = true_relres < rtol
    return x, SolveReport(
        converged=converged,
        iterations=total_iters,
        true_relres=true_relres,
        arnoldi_relres=float(arnoldi_final),
        stagnated=stagnated and not converged,
        restarts=restarts - 1,
        residual_history=history,
    )


def _gcrodr(A_mv, M_mv, b, x, rs, rtol, restart, maxiter,
            stagnation_window, stagnation_factor, callback):
    """GCRO with deflated restarting, right-preconditioned and flexible.

    Parks, de Sturler, Mackey, Johnson & Maiti, "Recycling Krylov
    subspaces for sequences of linear systems", SIAM J. Sci. Comput.
    28(5), 1651-1674 (2006); the appendix pseudocode, equation numbers
    below are that paper's, and the algorithm is theirs applied to the
    right-preconditioned operator A M.

    Right preconditioning is carried explicitly rather than by composing
    a single operator, so that M may differ between SOLVES (the material
    sweeps rebuild it): the recycled vectors Y are (A M)-domain, and
    every solve re-derives U = M Y and C = A U for its own A and M. Each
    cycle then keeps M Y = U and A U = C, with both representations of
    the same k vectors, and the solution update runs through U -- the
    Arnoldi relation applied is

        A [U_k  Z_{m-k}] = [C_k  V_{m-k+1}] G,   Z_j = M v_j.

    Per solve: U = M Y, C = A U and a QR of it (2.7), the optimal
    correction over range(U), then cycles of m - k Arnoldi steps with
    (I - C C^T) A M (2.9), the augmented least squares (2.13), and a new
    recycle space from the k smallest-magnitude harmonic Ritz vectors of
    the pencil (2.16). The cycle's least-squares residual is the
    iteration's stop; the TRUE residual is what converged is reported on.
    """
    n = b.shape[0]
    b_norm = float(np.linalg.norm(b))

    # -- the subspace carried in: M Y = U, A U = C orthonormal (2.7) --
    C = U = Y = None
    setup_matvecs = 0
    Y_in = rs.vectors
    if Y_in is not None and (Y_in.ndim != 2 or Y_in.shape[0] != n
                             or Y_in.shape[1] == 0):
        rs.reset()
        Y_in = None
    if Y_in is not None:
        U_in = np.empty_like(Y_in)
        AU = np.empty_like(Y_in)
        for i in range(Y_in.shape[1]):
            U_in[:, i] = M_mv(Y_in[:, i])
            AU[:, i] = A_mv(U_in[:, i])
        setup_matvecs = Y_in.shape[1]
        C, (U, Y) = _orthonormalize(AU, (U_in, Y_in))

    r = b - A_mv(x)
    if C is not None:
        y0 = C.T @ r
        x = x + U @ y0
        r = r - C @ y0

    history: list[float] = []
    total_iters = 0
    restarts = 0
    best_res = np.inf
    best_res_at = 0
    stagnated = False
    res = float(np.linalg.norm(r)) / b_norm

    while res >= rtol and total_iters < maxiter and not stagnated:
        k = 0 if C is None else C.shape[1]
        m = min(restart - k, maxiter - total_iters)
        if m < 1:
            break
        beta = float(np.linalg.norm(r))
        V = np.empty((m + 1, n))
        Z = np.empty((m, n))
        H = np.zeros((m + 1, m))
        Bk = np.zeros((k, m))
        V[0] = r / beta
        # D_k of (2.11), one scaling for both representations (A U D =
        # C D = A M Y D): unit columns in the (A M)-domain, where the
        # Arnoldi vectors it sits next to in the pencil are unit too.
        d = 1.0 / np.linalg.norm(Y, axis=0) if k else None
        Yh = Y * d if k else None
        Uh = U * d if k else None
        rhs_head = C.T @ r if k else np.zeros(0)

        j_done = 0
        for j in range(m):
            Z[j] = M_mv(V[j])
            w = A_mv(Z[j])
            if k:                       # Arnoldi with (I - C C^T) A M
                Bk[:, j] = C.T @ w
                w = w - C @ Bk[:, j]
            for i in range(j + 1):      # modified Gram-Schmidt
                H[i, j] = np.vdot(V[i], w)
                w -= H[i, j] * V[i]
            H[j + 1, j] = np.linalg.norm(w)
            # A lucky breakdown ends the cycle with an exact solution;
            # the zero column keeps the harvest below well defined.
            V[j + 1] = w / H[j + 1, j] if H[j + 1, j] > 1e-300 else 0.0

            total_iters += 1
            j_done = j + 1
            G = _augmented_G(d, Bk[:, :j_done], H[:j_done + 1, :j_done])
            rhs = np.concatenate([rhs_head, [beta], np.zeros(j_done)])
            _, res_abs = _small_lstsq(G, rhs)
            res = res_abs / b_norm
            history.append(res)
            if callback is not None:
                callback(total_iters, res)

            if res < best_res / stagnation_factor:
                best_res = res
                best_res_at = total_iters
            elif total_iters - best_res_at >= stagnation_window:
                stagnated = True
                break

            if res < rtol or total_iters >= maxiter:
                break

        if j_done == 0:
            break
        # -- augmented least squares, solution and residual (2.13-2.15)
        G = _augmented_G(d, Bk[:, :j_done], H[:j_done + 1, :j_done])
        rhs = np.concatenate([rhs_head, [beta], np.zeros(j_done)])
        y, _ = _small_lstsq(G, rhs)
        x = x + Z[:j_done].T @ y[k:]
        if k:
            x = x + Uh @ y[:k]
        g = rhs - G @ y
        r = V[:j_done + 1].T @ g[k:]
        if k:
            r = r + C @ g[:k]
        restarts += 1
        if stagnated:
            break
        # -- deflated restart / the space carried out (2.16, A.30-A.34)
        C, U, Y = _recycle_update(C, Uh, Yh, V[:j_done + 1], Z[:j_done],
                                  G, rs.dim)

    rs.store(Y)
    true_relres = float(np.linalg.norm(b - A_mv(x)) / b_norm)
    return x, SolveReport(
        converged=true_relres < rtol,
        iterations=total_iters,
        true_relres=true_relres,
        arnoldi_relres=float(history[-1] if history else true_relres),
        stagnated=stagnated and not true_relres < rtol,
        restarts=max(restarts - 1, 0),
        residual_history=history,
        recycle_dim=0 if Y is None else Y.shape[1],
        setup_matvecs=setup_matvecs,
    )


def _augmented_G(d, B, Hbar):
    """G of (2.11): [[D_k, B_k], [0, H_{m-k}]] (empty D/B when k = 0)."""
    k = 0 if d is None else d.size
    nr, nc = Hbar.shape
    G = np.zeros((k + nr, k + nc))
    if k:
        G[:k, :k] = np.diag(d)
        G[:k, k:] = B
    G[k:, k:] = Hbar
    return G


def _small_lstsq(G, rhs):
    """Least squares over the (k + m + 1) x (k + m) augmented system;
    m is the restart length, so this is small enough to redo per
    iteration for the residual estimate."""
    y, *_ = np.linalg.lstsq(G, rhs, rcond=None)
    return y, float(np.linalg.norm(rhs - G @ y))


def _orthonormalize(AU, mats):
    """(2.7): C = Q of the reduced QR of A U, and every representation of
    the same vectors scaled by the same R^{-1}, so A U = C with
    orthonormal C. Returns (None, Nones) if A U is rank deficient -- a
    recycled space that has collapsed onto the operator's near-null
    directions is dropped rather than inverted."""
    Q, R = np.linalg.qr(AU)
    diag = np.abs(np.diag(R))
    if diag.size == 0 or diag.min() <= 1e-12 * max(diag.max(), 1e-300):
        return None, (None,) * len(mats)
    return Q, tuple(sla.solve_triangular(R, X.T, trans="T").T for X in mats)


def _recycle_update(C, Uh, Yh, V, Z, G, k_keep):
    """Harmonic Ritz vectors of the cycle's augmented space, and the
    (C, U, Y) triple for the next cycle (A.30-A.34).

    The pencil is (2.16), G^T G z = theta G^T W^T Vhat z, over the
    (A M)-domain search space Vhat = [Y_k, V_{m-k}] with W =
    [C_k, V_{m-k+1}]: V is orthonormal and orthogonal to C by
    construction, so only the two Y blocks are inner products.
    Smallest |theta| is the paper's choice. The solution-space
    representation U of the selected vectors rides along through the
    same coefficients, which is what keeps M Y = U exact.
    """
    k = 0 if C is None else C.shape[1]
    j = Z.shape[0]
    nv = V.shape[0]                       # m - k + 1
    WtV = np.zeros((k + nv, k + j))
    WtV[k:k + j, k:] = np.eye(j)          # V_{m-k+1}^T V_{m-k}
    if k:
        WtV[:k, :k] = C.T @ Yh
        WtV[k:, :k] = V @ Yh

    P = _harmonic_basis(G, WtV, min(k_keep, k + j))
    if P is None:
        return None, None, None
    Y_new = V[:j].T @ P[k:]
    U_new = Z.T @ P[k:]
    if k:
        Y_new = Y_new + Yh @ P[:k]
        U_new = U_new + Uh @ P[:k]
    Q, R = np.linalg.qr(G @ P)
    diag = np.abs(np.diag(R))
    if diag.size == 0 or diag.min() <= 1e-12 * max(diag.max(), 1e-300):
        return None, None, None
    C_new = V.T @ Q[k:]
    if k:
        C_new = C_new + C @ Q[:k]
    scaled = tuple(sla.solve_triangular(R, X.T, trans="T").T
                   for X in (U_new, Y_new))
    return (C_new,) + scaled


def _harmonic_basis(G, WtV, k_keep):
    """Real orthonormal basis of the span of the k_keep harmonic Ritz
    vectors of smallest |theta|, in the coordinates of the search space.

    A complex conjugate pair spans one real plane, so the selected
    eigenvectors are split into real and imaginary parts and an SVD
    returns an orthonormal basis of what they span (only the SPAN enters
    the next cycle). One extra dimension is allowed, which is the
    paper's "sometimes k + 1 vectors" for a pair whose partner would
    otherwise be cut. Returns None if the pencil is degenerate.
    """
    try:
        theta, vec = sla.eig(G.T @ G, G.T @ WtV)
    except (sla.LinAlgError, ValueError):
        return None
    mag = np.abs(theta)
    mag[~np.isfinite(mag)] = np.inf
    order = np.argsort(mag)[:k_keep]
    if order.size == 0 or not np.isfinite(mag[order[0]]):
        return None
    sel = vec[:, order]
    cols = [sel.real]
    if np.any(np.abs(sel.imag) > 0):
        cols.append(sel.imag)
    Q, s, _ = np.linalg.svd(np.column_stack(cols), full_matrices=False)
    rank = int(np.sum(s > s[0] * 1e-12)) if s.size else 0
    if rank == 0:
        return None
    return Q[:, :min(rank, k_keep + 1, G.shape[1])]


def _solve_upper(R, g):
    """Back-substitution for the small upper-triangular least-squares system."""
    m = R.shape[0]
    y = np.zeros(m, dtype=R.dtype)
    for i in range(m - 1, -1, -1):
        s = g[i] - R[i, i + 1:] @ y[i + 1:]
        y[i] = s / R[i, i]
    return y


def _as_matvec(A):
    if A is None:
        return None
    if callable(A) and not hasattr(A, "matvec"):
        return A
    if hasattr(A, "matvec"):
        return A.matvec
    return lambda v: A @ v
