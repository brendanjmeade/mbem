"""A dense operator factored once: in double precision, or in single and refined.

``factor(A)`` is LU with partial pivoting and ``Factor.solve`` as many
right-hand sides as it is given: what the dense backend solves a state with
(``AssembledDense.solve``, ``condense``) and what a stepped run factors its
body with (``mbem.stepper``). ``how`` is one of three:

* ``"double"`` -- LAPACK on the host. The reference, and the default on
  every machine: the other two are ASKED FOR (``backend.lu``) and a card is
  never looked for, so a result does not depend on where it was made. A
  card asked for where none answers is refused, not replaced by the host.
* ``"gpu"`` -- the factors in SINGLE precision on a CUDA card (through
  ``cupy``), every solve refined against the double-precision matrix, which
  stays on the host. A workstation card has no fast double precision (this
  one's is slower than the host's), but in single it factors at twenty times
  the host's rate, and the bodies here are conditioned well enough for it:
  a solve from single factors alone is off by 4e-6, one correction leaves
  7e-12 and a second rounding, at the megathrust body's softest state as at
  its elastic one. What a state then costs is its assembly and one product
  with the matrix per correction.
* ``"single"`` -- the same on the host: half again LAPACK's double rate, and
  the way to the refinement on a machine with no card.

Refinement is the classical one -- solve, residual in double, correct --
stopped on the SIZE OF THE CORRECTION rather than on a residual, which would
take one more product with the matrix to see. A correction ``d`` that
follows one of size ``d'`` contracted at ``rho = |d| / |d'|`` (the first
against the solution itself), and what is left after it is
``rho / (1 - rho) |d|`` (Demmel et al. 2006, the bound of LAPACK's
extra-precise refinement). Where the corrections do not contract, a matrix
single precision does not hold, the matrix is factored in double on the host
after all, with a warning.

The factors are of the TRANSPOSE. A C-ordered matrix read in LAPACK's order
is its transpose, so nothing is copied on a card the matrix nearly fills,
and a solve asks for the transposed system.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from numba import get_num_threads, njit, prange

from .. import defaults

HOW = ("double", "single", "gpu")


@njit(parallel=True, cache=True)
def _column_sums(A, out, blocks):
    """The column sums of ``|A|``; with ``out`` of ``A``'s shape, ``A`` is
    also written to it (in ``out``'s precision), without a temporary. Row
    blocks, one to a thread, each with its own sums."""
    n, m = A.shape
    cast = out.shape[0] == n
    sums = np.zeros((blocks, m))
    for b in prange(blocks):
        for i in range(b * n // blocks, (b + 1) * n // blocks):
            for j in range(m):
                v = A[i, j]
                sums[b, j] += abs(v)
                if cast:
                    out[i, j] = v
    return sums.sum(axis=0)


def _cupy():
    """``cupy``, with a device that answers; a ``RuntimeError`` that says
    what to do otherwise."""
    try:
        import cupy
        import cupyx.scipy.linalg                       # noqa: F401
        cupy.cuda.Device().mem_info
    except Exception as exc:                            # no package, no card
        raise RuntimeError(
            f"how='gpu' needs a CUDA device through cupy "
            f"({type(exc).__name__}: {exc}); factor with how='single' or "
            f"'double'") from exc
    return cupy


def gpu_available() -> bool:
    try:
        _cupy()
    except RuntimeError:
        return False
    return True


def gpu_capacity() -> int:
    """Unknowns of the largest matrix the card holds NOW in single precision
    (its free memory, and what this process's pool holds unused)."""
    cp = _cupy()
    free = cp.cuda.Device().mem_info[0] \
        + cp.get_default_memory_pool().free_bytes()
    return int(np.sqrt(defaults.LU_GPU_FILL * free / 4.0))


def dense_limit(how=None) -> int:
    """Unknowns up to which a stepped run factors its body densely
    (``stepper``): on the host what is faster than blocks, on a card what
    it holds."""
    how = defaults.DENSE_LU if how is None else how
    return gpu_capacity() if how == "gpu" else defaults.STEPPER_DENSE_BOUNDARY


class Factor:
    """``A`` factored (module docstring). ``solve``, ``cond``, and after
    them ``report``: ``how``, ``factor_s``, and of the refinement the most
    ``passes`` a solve took, the size of its last ``correction`` and what
    was ``left`` after it (relative to the solution, the worst column), and
    ``fallback`` if the matrix was factored in double after all.

    A matrix factored in single precision is KEPT (the refinement reads
    it) and must not change while the factors are used; ``overwrite_a`` is
    for ``"double"`` alone. C-ordered, or it is copied.
    """

    def __init__(self, A, how: str | None = None, overwrite_a: bool = False):
        how = defaults.DENSE_LU if how is None else how
        if how not in HOW:
            raise ValueError(f"a dense matrix is factored {HOW}, not {how!r}")
        A = np.ascontiguousarray(A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"factor: a square matrix, not {A.shape}")
        self.how, self.n = how, A.shape[0]
        self.report = {"how": how}
        self._A = self._lu = self._double = None
        t0 = time.perf_counter()
        if how == "gpu":
            cp, room = _cupy(), gpu_capacity()
            if self.n > room:
                raise ValueError(
                    f"how='gpu': {self.n} unknowns in single precision are "
                    f"{4 * self.n ** 2 / 2 ** 30:.1f} GiB and the card has "
                    f"room for {room} now; factor the body in blocks "
                    f"(stepper factor='blocks') or on the host")
        single = np.empty(A.shape if how != "double" else (0, 0), np.float32)
        sums = _column_sums(A, single, get_num_threads())
        if not np.isfinite(sums).all():
            raise ValueError("factor: the matrix must not contain infs or NaNs")
        self.anorm = float(sums.max(initial=0.0))
        if how == "double":
            self._factor_double(A, overwrite_a)
        else:
            self._A = A
            if how == "gpu":
                import cupyx.scipy.linalg as csl
                on_card = cp.asarray(single)
                del single
                self._lu = csl.lu_factor(on_card.T, overwrite_a=True,
                                         check_finite=False)
                cp.cuda.Stream.null.synchronize()
            else:
                from scipy.linalg import lu_factor
                self._lu = lu_factor(single.T, overwrite_a=True,
                                     check_finite=False)
        self.report["factor_s"] = time.perf_counter() - t0

    def _factor_double(self, A, overwrite_a=False):
        from scipy.linalg import lu_factor
        self._double = lu_factor(A, overwrite_a=overwrite_a,
                                 check_finite=False)

    @property
    def nbytes(self) -> int:
        """What the factors hold (on the card, for ``"gpu"``)."""
        if self._double is not None:
            return int(self._double[0].nbytes)
        return 4 * self.n ** 2

    # -- solves ----------------------------------------------------------

    def _single(self, R, trans: int = 1):
        """``A^-1 R`` from the single factors, as double (``trans=0`` the
        transposed system's): the factors are of ``A'``."""
        R = np.asarray(R, dtype=np.float32, order="F")
        if self.how == "single":
            from scipy.linalg import lu_solve
            return lu_solve(self._lu, R, trans=trans, overwrite_b=True,
                            check_finite=False).astype(float)
        import cupy as cp
        import cupyx.scipy.linalg as csl
        out = np.empty(R.shape)
        # as many columns at a time as a part of the card's room holds
        room = cp.cuda.Device().mem_info[0] \
            + cp.get_default_memory_pool().free_bytes()
        step = max(1, int(defaults.LU_GPU_RHS * room / (4 * self.n)))
        for a in range(0, R.shape[1], step):
            out[:, a:a + step] = cp.asnumpy(csl.lu_solve(
                self._lu, cp.asarray(R[:, a:a + step]), trans=trans,
                overwrite_b=True, check_finite=False))
        return out

    def solve(self, B):
        """``A^-1 B`` for ``B`` ``(n,)`` or ``(n, k)``, in double."""
        from scipy.linalg import lu_solve
        B = np.asarray(B, dtype=float)
        if self._double is not None:
            return lu_solve(self._double, B, check_finite=False)
        B2 = B.reshape(self.n, -1)
        X = self._single(B2)
        live = np.linalg.norm(X, axis=0) > 0.0
        tol, before = defaults.LU_REFINE_TOL, None
        done, rate, size, left, k = not live.any(), 0.0, 0.0, 0.0, 0
        while not done and k < defaults.LU_REFINE_PASSES:
            D = self._single(B2 - self._A @ X)
            X += D
            k += 1
            now = np.linalg.norm(D[:, live], axis=0) \
                / np.linalg.norm(X[:, live], axis=0)
            # the first correction is measured against the solution: a
            # solve's own error and its contraction are one number
            rate = float((now if before is None else np.divide(
                now, before, out=np.zeros_like(now),
                where=before > 0.0)).max())
            size, before = float(now.max()), now
            left = rate / (1.0 - rate) * size if rate < 1.0 else np.inf
            done = left <= tol or size <= tol
            if rate > defaults.LU_REFINE_RATE:
                break
        rep = self.report
        if not done:
            warnings.warn(
                f"dense LU in single precision ({self.how}): the refinement "
                f"did not contract (rate {rate:.1e}, {left:.1e} left after "
                f"{k} corrections); factoring in double on the host. The "
                f"matrix is too ill-conditioned for single-precision factors")
            rep["fallback"] = True
            self._lu = None
            self._factor_double(self._A)
            return lu_solve(self._double, B, check_finite=False)
        rep["passes"] = max(k, rep.get("passes", 0))
        if left >= rep.get("left", 0.0):
            rep["correction"], rep["left"] = size, left
        return X.reshape(B.shape)

    def cond(self) -> float:
        """1-norm condition estimate: LAPACK's ``gecon`` on double factors,
        the same estimator (Higham's) on the solves of single ones."""
        if self._double is not None:
            from scipy.linalg import get_lapack_funcs
            gecon = get_lapack_funcs(("gecon",), (self._double[0],))[0]
            rcond, info = gecon(self._double[0], self.anorm, norm="1")
            return (1.0 / rcond) if (info == 0 and rcond > 0) else np.inf
        from scipy.sparse.linalg import LinearOperator, onenormest

        def inv(t):
            return lambda x: self._single(
                np.asarray(x, float).reshape(self.n, -1), t).reshape(x.shape)
        est = onenormest(LinearOperator(
            (self.n, self.n), dtype=float, matvec=inv(1), rmatvec=inv(0),
            matmat=inv(1), rmatmat=inv(0)))
        est = self.anorm * float(est)
        return est if np.isfinite(est) else np.inf


def factor(A, how: str | None = None, overwrite_a: bool = False) -> Factor:
    return Factor(A, how, overwrite_a)
