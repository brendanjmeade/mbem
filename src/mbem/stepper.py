"""A solved fault through time, under a law that need not be linear.

The Laplace route (``mbem.laplace``) needs every part of a model to be
linear. A fault whose slip rate is not proportional to its traction is not,
but it is the only part that is not, so the body is REDUCED TO THE FAULT and
the fault alone is stepped (docs/viscoelastic.md section 6).

At each state of a run the dense backend eliminates every unknown but the
solved faults' slip (``AssembledDense.condense``), leaving on the faults

    traction = S_k @ slip + L_k .

In time ``S`` is a constant plus decaying exponentials, the fit the Laplace
route makes (``Prony.coefficients``), so a body that relaxes is one decaying
vector ``p_j`` per relaxation time of the fit:

    traction(t) = sum_k S_k c_k(t) + L(t),    c = C.T @ [slip; p_1 .. p_m]
    d slip / dt = law(traction)
    d p_j  / dt = d slip / dt - p_j / tau_j

an ordinary differential equation, integrated by an embedded Runge-Kutta
pair (``scipy.integrate.solve_ivp``, Dormand-Prince by default) or, where
the faults are fine enough to be stiff, by the stabilised Chebyshev method
of ``mbem.rkc``. A body that does not relax is one state and no ``p``.
Nothing is assumed of the law but a rate for a traction; the body's
linearity is what is used, and it is exact.

Two things the fit owes a consumer that FEEDS BACK through it:

* **Its relaxed limit must not push.** The constant of the fit is what the
  faults' stiffness has relaxed to at the end of the window. It misses by
  the fit's error (1e-3 of the kernel), and where the exact limit is zero or
  nearly -- a fault in a region that relaxes completely -- it misses with
  the wrong sign for some modes, which then GROW (2-5 % by the window's
  end, measured). ``Body`` moves the growing part of that constant's
  spectrum onto the slowest pole: the elastic instant stays exact, the
  kernel changes by less than the fit's own error, and no mode grows.
* **The window is the body's, the step is the fault's.** The fit must reach
  below the body's fastest relaxation and beyond the last time asked for
  (``march`` refuses a later one); how fast the FAULT moves is the
  integrator's business, and the embedded pair's step is bounded by the
  fit's fastest pole (about 3.3 of its relaxation times) and by the fault's
  shortest wavelength, not by the law.

A body with NO FASTEST RELAXATION (``rheology.Andrade``) is fitted with a
tail, every pole beyond the window's short end in one term
(``laplace.Prony(tails=...)``). Here the tail is memory like the rest: its
poles within a decade of the fastest fitted one are carried, one decaying
vector each, and what lies beyond has gone by any lag greater than zero
(``Prony.memory``). So a step's LOAD is the elastic one at its own instant
and has lost that part an instant later (``load`` at ``t == te``, which is
what a history records there and what ``march`` sizes its first step by),
and the stiffness a fault slips against is the elastic one LESS that part
throughout: a tenth to a fifth of the modulus, wrong only at lags the
window does not hold. A decade of tail is ten times the fastest rate, so
such a body is stepped with the stabilised method whatever its faults.

The reduction is the dense backend's: a compressed operator solves no slip,
and ``condense`` is one right-hand side per slip unknown. ``KernelBody`` is
the way past both for P0 faults inside a region: their own kernel
compressed, and what a boundary adds to it in low rank.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import defaults, laplace


def _growth(G, tau, mobility) -> float:
    """The largest real part among the eigenvalues of ``d slip/dt =
    mobility * traction``, ``traction = sum_i G[i] y_i`` over the state
    ``[slip; p_1 .. p_m]``, ``G`` dense ``(1 + m, n, n)``."""
    m, n = tau.size, G.shape[1]
    mob = np.broadcast_to(np.asarray(mobility, float), (n,))
    J = np.zeros(((1 + m) * n, (1 + m) * n))
    for i in range(1 + m):
        for j in range(1 + m):
            J[i * n:(i + 1) * n, j * n:(j + 1) * n] = mob[:, None] * G[j]
        if i:
            J[i * n:(i + 1) * n, i * n:(i + 1) * n] -= np.eye(n) / tau[i - 1]
    return float(np.linalg.eigvals(J).real.max())


class Body:
    """What a body's solved faults feel through time.

    ``states`` are the ``Condensed`` operators of a run's states in the order
    ``Prony`` takes them, the elastic one first, and ``s`` their finite
    Laplace parameters; ``s=None`` is a body that does not relax (one
    state). ``passive=False`` keeps the fit's relaxed limit as fitted: for a
    gate that shows what that costs. ``tails`` are the fit's
    (``rheology.tails`` of what relaxes).
    """

    def __init__(self, states, s=None, passive: bool = True, tails=()):
        states = list(states)
        self.states = states
        self.faults = states[0].faults
        self.layout = states[0].layout
        self.n = states[0].n
        if s is None:
            if len(states) != 1:
                raise ValueError(f"a body that does not relax is ONE state; "
                                 f"got {len(states)} and no s")
            self.tau, self.C, self.window = np.zeros(0), np.ones((1, 1)), None
            self._gone, tails = np.zeros(1), ()
        else:
            s = np.asarray(s, float)
            if len(states) != 1 + s.size:
                raise ValueError(f"{s.size} Laplace parameters want "
                                 f"{1 + s.size} states (the elastic one "
                                 f"first), got {len(states)}")
            fit = laplace.Prony(s, tails=tails)
            self.tau, self.C, self._gone = fit.memory()
            tails = fit.tails          # the fit's: a band may have none here
            self.window = laplace.window(s)
        self.tails = tuple(tails)
        K, n = len(states), self.n
        S = np.stack([c.S for c in states])
        # traction = _S @ c.ravel(), c the (K, n) slip each state is asked for
        self._S = np.ascontiguousarray(S.transpose(1, 0, 2).reshape(n, K * n))
        self._aL = self.C @ np.stack([c.L for c in states])
        self._goneL = self._gone @ np.stack([c.L for c in states])
        self._slow = int(np.argmax(self.tau)) if self.tau.size else -1
        self._D, self.growing, self.removed = None, 0, 0.0
        if passive and self.tau.size:
            self._make_passive(np.tensordot(self.C[0], S, axes=1), S[0])

    def _make_passive(self, relaxed, elastic) -> None:
        """Take the growing part of the fit's relaxed limit out of it and
        give it to the slowest pole (module docstring): the part of
        ``relaxed`` on its invariant subspace of eigenvalues with a positive
        real part, by an ordered Schur form, which stays accurate where the
        eigenvectors of a matrix that is not symmetric do not."""
        from scipy.linalg import schur, solve_sylvester

        T, Z, k = schur(relaxed, output="real", sort="rhp")
        if k == 0:
            return
        n = relaxed.shape[0]
        top = np.zeros((k, n))
        top[:, :k] = T[:k, :k]
        if k < n:
            # the spectral projector onto that subspace is Z [[I, Y], [0, 0]] Z'
            Y = solve_sylvester(T[:k, :k], -T[k:, k:], -T[:k, k:])
            top[:, k:] = T[:k, :k] @ Y
        self._D = Z[:, :k] @ top @ Z.T
        self.growing = int(k)
        self.removed = float(np.linalg.norm(self._D, 2)
                             / np.linalg.norm(elastic, 2))

    # -- the pieces of the equation ------------------------------------

    def _basis(self, lag) -> np.ndarray:
        """``[1, exp(-lag / tau_j)]``: a unit step's fit, ``lag`` after it."""
        return np.concatenate([[1.0], np.exp(-lag / self.tau)])

    def load(self, t, events) -> np.ndarray:
        """(n,): what the prescribed data load the held faults with at ``t``,
        each of ``events = [(time, amplitude)]`` a step of them."""
        out = np.zeros(self.n)
        for te, amp in events:
            if t >= te:
                out += amp * (self._basis(t - te) @ self._aL)
            if t == te:                 # a tail's far end, not yet gone
                out += amp * self._goneL
        return out

    def traction(self, Y, load) -> np.ndarray:
        """(n,): the traction on the faults for the state ``Y = [slip; p]``,
        ``(1 + m, n)``."""
        out = self._S @ (self.C.T @ Y).ravel()
        if self._D is not None:
            out -= self._D @ (Y[0] - Y[1 + self._slow])
        return out + load

    def unknowns(self, t, Y, events) -> np.ndarray:
        """(N,): every unknown of the model at ``t``, in the layout's order
        (``Condensed.slots`` names them): the prescribed data's part, each
        state at the fit's weight, and the faults' slip through each state's
        own response."""
        w = np.zeros(len(self.states))
        for te, amp in events:
            if t >= te:
                w += amp * (self._basis(t - te) @ self.C)
            if t == te:
                w += amp * self._gone
        c = self.C.T @ Y
        x = np.zeros(self.states[0].x0.shape[0])
        for k, st in enumerate(self.states):
            x += w[k] * st.x0 + st.X @ c[k]
        return x

    def slots(self, t, Y, events) -> dict:
        """Every unknown at ``t``, as a solve returns them."""
        return self.states[0].slots(self.unknowns(t, Y, events))

    def slip_scale(self) -> float:
        """The largest slip the elastic body's faults would take to shed the
        load entirely: what an absolute tolerance on slip is a fraction of."""
        el = self.states[0]
        if not np.any(el.L):
            return 0.0
        return float(np.abs(np.linalg.solve(el.S, el.L)).max())

    def growth(self, mobility) -> float:
        """The largest real part among the eigenvalues of this body with the
        LINEAR law ``d slip/dt = mobility * traction`` (a number, or one per
        coordinate): positive is a mode that grows. A diagnostic, of the size
        of the whole state -- for a gate, not a run."""
        n = self.n
        K = len(self.states)
        G = np.tensordot(self.C, self._S.reshape(n, K, n).transpose(1, 0, 2),
                         axes=1)
        if self._D is not None:
            G[0] -= self._D
            G[1 + self._slow] += self._D
        return _growth(G, self.tau, mobility)


@dataclass
class Boundary:
    """What a body's boundary is to the faults of a ``KernelBody``, state by
    state: ``P M[k] W'`` is its part of their stiffness, and the boundary's
    own unknowns are ``x0[k] - Z[k] W' slip`` in ``layout``'s order."""
    P: np.ndarray               # (n, r_p), orthonormal
    W: np.ndarray               # (n, r_w), orthonormal
    M: np.ndarray               # (K, r_p, r_w)
    Z: np.ndarray               # (K, N, r_w)
    x0: np.ndarray              # (K, N)
    layout: object
    report: dict


class KernelBody:
    """A body with no boundary, as its faults feel it: no dense matrix.

    One region that is the whole space, its solved faults (P0) the only
    unknowns. Nothing is then eliminated, and the reduced stiffness of
    ``Body`` is the faults' own traction kernel, which is a kernel sum:
    ``la.faultkernel.FaultKernel`` holds it compressed, as TWO matrices that
    serve every material. So

        S(t) = a_lo(t) H_lo + a_hi(t) H_hi

    with ``a`` SCALAR functions of time, the fit of each state's two
    weights, and a stage of the integrator is two compressed products
    however many states the body has. What ``Body`` does to the fit's
    relaxed limit with an ordered Schur form is here two numbers: a relaxed
    weight below zero is one that pushes, and it is moved onto the slowest
    pole.

    ``materials`` are the region's ``(mu, lam)`` state by state, the elastic
    one first, ``s`` their finite Laplace parameters (None: one state).
    ``loads`` is ``(n_states, n)``: what the prescribed data load the held
    faults with at each state, in coordinates.

    IN A BODY WITH A BOUNDARY the faults' stiffness has a second part, what
    the boundary returns of a slip, and ``boundary`` holds it
    (``Boundary``): ``P M_k W'`` at state ``k``, of the rank the BOUNDARY'S
    mesh gives it and not the faults' (``la.coupling``). A stage is then
    two thin products more. Its relaxed limit cannot be judged apart from
    the kernel's where the faults' region still resists, since a boundary
    may rightly soften a fault, and it is then left as fitted. Where that
    region has relaxed to a FLUID within the fit's error
    (``defaults.STEPPER_FLUID_LIMIT``) there is nothing to judge: a fluid
    carries no shear, the faults' relaxed stiffness is none, and the whole
    of the fitted limit is moved to the slowest pole.
    """

    def __init__(self, kernel, faults, materials, loads, s=None,
                 passive: bool = True, boundary=None, tails=()):
        self.kernel, self.faults = kernel, tuple(faults)
        self.boundary = boundary
        self.n = int(self.faults[-1].stop)
        materials = [(float(m), float(lam)) for m, lam in materials]
        if s is None:
            if len(materials) != 1:
                raise ValueError(f"a body that does not relax is ONE state; "
                                 f"got {len(materials)} and no s")
            self.tau, self.C, self.window = np.zeros(0), np.ones((1, 1)), None
            self._gone, tails = np.zeros(1), ()
        else:
            s = np.asarray(s, float)
            if len(materials) != 1 + s.size:
                raise ValueError(f"{s.size} Laplace parameters want "
                                 f"{1 + s.size} states (the elastic one "
                                 f"first), got {len(materials)}")
            fit = laplace.Prony(s, tails=tails)
            self.tau, self.C, self._gone = fit.memory()
            tails = fit.tails          # the fit's: a band may have none here
            self.window = laplace.window(s)
        self.tails = tuple(tails)
        self.materials = materials
        W = np.array([kernel.weights(m, lam) for m, lam in materials])  # (K, B)
        self._a = (self.C @ W).T.copy()                  # (B, 1 + m)
        self.loads = np.asarray(loads, float)            # (K, n)
        self._aL = self.C @ self.loads
        self._goneL = self._gone @ self.loads
        self._L0 = self.loads[0]
        self.growing, self.removed = 0, 0.0
        if passive and self.tau.size:
            # Not weight by weight: the two fitted limits miss in opposite
            # senses and all but cancel, so clipping one alone would add ten
            # times what it removes. In S = c1 H_lo + c2 (H_hi - H_lo) both
            # operators resist (the stiffer material is the stiffer fault)
            # and each limit is its own fit: c1 = a_lo + a_hi, c2 = a_hi.
            slow = 1 + int(np.argmax(self.tau))
            a = self._a
            c = [a[:, 0].sum(), a[-1, 0]] if a.shape[0] == 2 else [a[0, 0]]
            cut = [min(v, 0.0) for v in c]              # what pushes
            self.growing = int(sum(v < 0.0 for v in cut))
            self.removed = float(max(-v for v in cut) / np.abs(W[0]).sum())
            if a.shape[0] == 2:
                move = np.array([cut[0] - cut[1], cut[1]])
            else:
                move = np.array([cut[0]])
            a[:, 0] -= move
            a[:, slow] += move
        # element offset of each fault in the kernel's order
        self._at, at = [], 0
        for f in self.faults:
            self._at.append(at)
            at += f.n_dofs
        self._N = None
        if boundary is not None:
            # (1 + m, r_p, r_w): the fit of the small matrices, pole by pole
            self._N = np.tensordot(self.C, np.asarray(boundary.M), axes=1)
            if passive and self.tau.size:
                self._passive_boundary(1 + int(np.argmax(self.tau)),
                                       np.abs(W[0]).sum())

    def _cart(self, v) -> np.ndarray:
        """Coordinates ``(n,)`` -> the kernel's Cartesian ``(3 n_el,)``."""
        return np.concatenate([f.cartesian(v[f.start:f.stop]).ravel()
                               for f in self.faults])

    def _coords(self, x) -> np.ndarray:
        x = x.reshape(-1, 3)
        return np.concatenate([f.coordinates(x[at:at + f.n_dofs])
                               for f, at in zip(self.faults, self._at)])

    _basis = Body._basis
    load = Body.load

    def _passive_boundary(self, slow: int, elastic: float) -> None:
        """The relaxed limit of faults in a region that has become a FLUID
        (class docstring) is none: a fluid carries no shear, whatever its
        boundary does. What the fit left there -- of the kernel's two
        weights and of the boundary's small matrices alike, pushing or not
        -- is the fit's error, and all of it goes to the slowest pole."""
        a, b = self._a, self.boundary
        left = float(np.abs(a[:, 0]).sum() / elastic)
        if left > defaults.STEPPER_FLUID_LIMIT:
            import warnings
            warnings.warn(
                f"KernelBody: the faults' region keeps {left:.2g} of its "
                f"shear modulus at the end of the fit's window, so the "
                f"boundary's part of their relaxed stiffness is left as "
                f"fitted: nothing here checks it for a mode that grows "
                f"(stepper.Body does, densely)", stacklevel=3)
            return
        a[:, slow] += a[:, 0]
        a[:, 0] = 0.0
        N0 = self._N[0].copy()
        lam = np.linalg.eigvals((b.W.T @ b.P) @ N0)
        self.growing += int((lam.real > float(b.report["tol"])
                             * np.abs(lam).max()).sum())
        self._N[slow] += N0
        self._N[0] = 0.0
        self.removed = max(self.removed, float(
            np.linalg.norm(N0, 2) / self._elastic_norm()))

    def stiffness(self, v, k: int = 0) -> np.ndarray:
        """State ``k``'s stiffness on a slip, in coordinates (the elastic
        body's by default): ``Condensed.S @ v`` of the dense reduction."""
        w = self.kernel.weights(*self.materials[k])
        out = self._coords(self.kernel.apply(
            *(wb * self._cart(v) for wb in w)))
        if self.boundary is not None:
            b = self.boundary
            out += b.P @ (b.M[k] @ (b.W.T @ v))
        return out

    def _elastic_norm(self) -> float:
        """The elastic stiffness's size, by a few steps of the power
        iteration: one figure."""
        v = np.random.default_rng(0).standard_normal(self.n)
        norm = 1.0
        for _ in range(8):
            v = self.stiffness(v / np.linalg.norm(v))
            norm = float(np.linalg.norm(v))
        return norm

    def growth(self, mobility) -> float:
        """As ``Body.growth``, with the kernel made dense: for a gate."""
        every = np.arange(self.kernel.n)
        eye = np.eye(self.n)
        H = [np.stack([self._coords(h @ self._cart(e)) for e in eye], 1)
             for h in self.kernel.stack(every, every)]
        G = np.tensordot(self._a.T, np.array(H), axes=1)
        if self._N is not None:
            b = self.boundary
            G = G + np.einsum("np,ipw,kw->ink", b.P, self._N, b.W)
        return _growth(G, self.tau, mobility)

    def traction(self, Y, load) -> np.ndarray:
        out = self._coords(self.kernel.apply(
            *(self._cart(a @ Y) for a in self._a))) + load
        if self._N is not None:
            b = self.boundary
            out += b.P @ np.einsum("ipw,iw->p", self._N, Y @ b.W)
        return out

    def slots(self, t, Y, events) -> dict:
        """Every unknown at ``t``: the faults' slip, and with a boundary
        its unknowns, each state's at the fit's weight as in ``Body``."""
        out = {f"b:{f.name}": f.cartesian(Y[0][f.start:f.stop])
               for f in self.faults}
        b = self.boundary
        if b is not None:
            w = np.zeros(len(self.materials))
            for te, amp in events:
                if t >= te:
                    w += amp * (self._basis(t - te) @ self.C)
                if t == te:
                    w += amp * self._gone
            c = (self.C.T @ Y) @ b.W                    # (K, r_w)
            x = w @ b.x0
            for k in range(len(self.materials)):
                x -= b.Z[k] @ c[k]
            out.update({s.name: x[s.offset:s.stop].reshape(-1, 3)
                        for s in b.layout.slots})
        return out

    def slip_scale(self) -> float:
        """As ``Body.slip_scale``, by a few steps of GMRES on the compressed
        kernel: the scale of a tolerance needs one figure, not a solve."""
        from scipy.sparse.linalg import LinearOperator, gmres

        if not np.any(self._L0):
            return 0.0
        op = LinearOperator((self.n, self.n),
                            matvec=lambda v: -self.stiffness(v))
        x, _info = gmres(op, self._L0, rtol=1.0e-2, restart=40, maxiter=3)
        return float(np.abs(x).max())

    @staticmethod
    def refusal(model) -> str | None:
        """Why ``model`` cannot be a ``KernelBody``, or ``None``: the one
        statement of what the compressed kernel is of, for whoever must
        refuse before building anything."""
        from .model.core import BCType

        every = model.solved_faults()
        if not every:
            return ("the model has no FAULT_TRACTION fault, so there is "
                    "nothing to step")
        on_face = sorted(q.name for _r, q in every
                         if q.bc is not BCType.FAULT_TRACTION)
        if on_face:
            return (f"{on_face} slip ON an interface, which has no kernel of "
                    f"its own -- all of such a slip's stiffness is the "
                    f"body's (AssembledDense.condense)")
        if any(r is not every[0][0] for r, _f in every):
            return ("the solved faults are in more than one region, and one "
                    "kernel is of one region's")
        rough = sorted(f.name for _r, f in every if f.order != 0 or f.continuous)
        if rough:
            return (f"the compressed kernel is of P0 faults (a cluster tree "
                    f"of elements), and {rough} are not")
        opens = sorted(f.name for _r, f in every if f.allow_opening)
        if opens:
            return f"{opens} open, and the stepper is of faults that slide"
        return None

    @classmethod
    def from_model(cls, model, eps, materials, s=None, tol: float = None,
                   passive: bool = True, log=None, coupling_tol: float = None,
                   rank: int = None, factor: str = None,
                   factor_tol: float = None, tails=(), lu: str = None):
        """Build one for ``model``: the FAULT_TRACTION faults of ONE region,
        P0, and whatever boundary the model has.

        ``materials`` are the states, the elastic one first: each the
        ``(mu, lam)`` of a model that is one region, or ``{region name:
        material}`` for every region. Faults with PRESCRIBED slip, and every
        other datum of the model, load the solved ones state by state.

        With no boundary nothing but the kernel is built. With one, the
        body WITHOUT the solved faults is factored at every state and solved
        for one right-hand side per column of the coupling's slip basis
        (``la.coupling.Coupling``; ``coupling_tol``, ``rank``) where
        ``condense`` solves one per slip unknown. ``factor`` says how:
        ``"dense"``, as ``AssembledDense.condense`` factors it, or
        ``"blocks"``, the compressed operator in block low-rank LU
        (``la.blr``) at ``factor_tol``; by default dense up to
        ``defaults.STEPPER_DENSE_BOUNDARY`` unknowns. ``lu`` is how a dense
        one is factored (``la.lu``): on a card (``"gpu"``) dense is the
        default up to what the card holds.
        """
        from types import SimpleNamespace

        from .backends.dense import FaultCoordinates
        from .la.faultkernel import FaultKernel, g_of
        from .model.equations import traction_row_frame

        why = cls.refusal(model)
        if why:
            raise ValueError(f"KernelBody: {why}")
        every = model.solved_faults()
        region = every[0][0]
        solved = [f for _r, f in every]
        states = [m if isinstance(m, dict) else
                  {region.name: SimpleNamespace(mu=float(m[0]), lam=float(m[1]))}
                  for m in materials]
        pairs = [(float(st[region.name].mu), float(st[region.name].lam))
                 for st in states]
        coords, start = [], 0
        for f in solved:
            free = np.flatnonzero(~f.locked_mask(region.patches))
            coords.append(FaultCoordinates(
                f.name, free, traction_row_frame(f)[free, :2, :], start,
                start + 2 * free.size, f.n_dofs))
            start += 2 * free.size
        g = [g_of(m, lam) for m, lam in pairs]
        kernel = FaultKernel(solved, eps, (min(g), max(g)), tol=tol)
        if log is not None:
            r = kernel.report
            log(f"fault kernel: {r['elements']} elements, "
                f"{r['bytes'] / 1e9:.2f} GB against {r['dense_bytes'] / 1e9:.1f} "
                f"dense, built in {r['build_s']:.0f} s")
        if len(model.regions) == 1 and not region.patches:
            loads = cls._whole_space_loads(region, solved, coords, eps, pairs)
            return cls(kernel, coords, pairs, loads, s=s, passive=passive,
                       tails=tails)
        boundary, loads = cls._boundary(model, eps, states, pairs, coords,
                                        (min(g), max(g)), coupling_tol, rank,
                                        log, factor, factor_tol, lu)
        return cls(kernel, coords, pairs, loads, s=s, passive=passive,
                   boundary=boundary, tails=tails)

    @staticmethod
    def _whole_space_loads(region, solved, coords, eps, pairs) -> np.ndarray:
        """``(K, n)``: minus the faults' own datum, plus the traction of
        every prescribed slip (b[b_f] of model/equations.py, with the sign
        of L), a state at a time."""
        from .kernels import basis as kb
        from .kernels import tri_nodal as tn
        from .model.core import BCType

        given = [f for f in region.faults if f.bc is BCType.FAULT]
        datum = np.concatenate([c.coordinates(f.value_array())
                                for c, f in zip(coords, solved)])
        loads = np.tile(-datum, (len(pairs), 1))
        if given:
            x = np.concatenate([f.collocation_points() for f in solved])
            n_c = np.concatenate([f.collocation_normals() for f in solved])
            body_at = np.cumsum([0] + [f.n_dofs for f in solved])
            arrays = kb.MeshArrays()
            for gf in given:
                tv, nrm = arrays.source_arrays(gf.mesh)
                ge = kb.resolve_patch_eps(eps, gf)
                ids = np.arange(tv.shape[0], dtype=np.int64).reshape(-1, 1)
                slip = gf.value_array().ravel()
                step = max(1, int(defaults.ASSEMBLY_SOURCE_CHUNK))
                for k, (m, lam) in enumerate(pairs):
                    t_c = np.concatenate([tn.traction_matrix(
                        np.ascontiguousarray(x[i:i + step]),
                        np.ascontiguousarray(n_c[i:i + step]), tv, nrm, ge,
                        ids, tv.shape[0], m, lam, 0, "slip") @ slip
                        for i in range(0, x.shape[0], step)]).reshape(-1, 3)
                    loads[k] += np.concatenate([
                        c.coordinates(t_c[a:b]) for c, a, b in
                        zip(coords, body_at[:-1], body_at[1:])])
        return loads

    @staticmethod
    def _boundary(model, eps, states, pairs, coords, g_range, tol, rank, log,
                  factor=None, factor_tol=None, lu=None):
        """``(Boundary, loads)``: the body without its solved faults, state
        by state, through the coupling's two bases."""
        from .la import lu as dense_lu
        from .la.coupling import Coupling
        from .model import generate_system
        from .model.equations import rhs_terms_at

        system = generate_system(model)
        body = generate_system(model.without_solved_faults())
        cp = Coupling(system, body.layout, coords, eps, g_range, pairs[0][0],
                      [rhs_terms_at(system, st) for st in states], tol=tol,
                      rank=rank, log=log)
        K, N = len(states), body.layout.n_unknowns
        if factor is None:
            factor = "dense" if N <= dense_lu.dense_limit(lu) else "blocks"
        M = np.empty((K, cp.P.shape[1], cp.W.shape[1]))
        Z = np.empty((K, N, cp.W.shape[1]))
        x0 = np.empty((K, N))
        loads = np.empty((K, cp.n))
        t0, nbytes, notes = time.time(), 0, []
        for k, (b, solve, held_bytes, note) in enumerate(
                _factored(body, eps, states, factor, factor_tol, lu)):
            B, E, held = cp.state(*pairs[k], k)
            Z[k] = solve(B)
            x0[k] = solve(b)
            # a state's factors go before the next state's are made
            del solve
            M[k] = -E @ Z[k]
            loads[k] = -held + cp.P @ (E @ x0[k])
            nbytes = max(nbytes, held_bytes)
            notes.append(note)
            if log is not None:
                log(f"state {k + 1} of {K}: the body factored ({factor}) and "
                    f"solved for {cp.W.shape[1]} slips")
        report = dict(cp.report, factor=factor, factor_s=time.time() - t0,
                      factor_bytes=int(nbytes))
        if factor == "dense":
            # the worst of the states, where the factors were single
            report["lu"] = dict(
                how=notes[0]["how"],
                **{key: max(n.get(key, 0) for n in notes)
                   for key in ("passes", "left") if key in notes[0]},
                **({"fallback": True}
                   if any(n.get("fallback") for n in notes) else {}))
        return Boundary(P=cp.P, W=cp.W, M=M, Z=Z, x0=x0, layout=body.layout,
                        report=report), loads


def _factored(body, eps, states, how: str, tol, lu=None):
    """A body's states one after another, each as ``(b, solve, bytes,
    note)``: its right-hand side, ``B -> A^-1 B`` for as many columns as are
    given, what the factors hold, and their own report. ``how="dense"`` is
    LU of the dense operator (``la.lu``, in the way ``lu`` says), what
    ``AssembledDense.condense`` factors; ``"blocks"`` is the compressed
    operator factored in blocks (``la.blr``), for a boundary a dense matrix
    does not hold. A state's factors live as long as its ``solve`` does."""
    if how == "dense":
        from .backends.dense import AssembledDense
        from .la import lu as dense_lu

        asm = AssembledDense(body, eps, "direct")
        for st in states:
            op = asm.rebuild_for_materials(dict(st))
            fac = dense_lu.factor(op.A, lu, overwrite_a=True)
            op.A = None
            yield op.b, fac.solve, fac.nbytes, fac.report
            del fac
        return
    if how != "blocks":
        raise ValueError(f"a body is factored 'dense' or in 'blocks', not "
                         f"{how!r}")
    from .backends.hmat import HBackend
    from .la import blr

    tol = defaults.BLR_TOL if tol is None else float(tol)
    asm = HBackend(tol=tol, storage="basis", sweep=True).assemble(body, eps)
    clusters = None
    for st in states:
        op = asm.rebuild_for_materials(dict(st))
        lu, clusters = blr.from_assembled(op, tol=tol, clusters=clusters)
        lu.factor()
        yield (op.b, lambda B, lu=lu, d=clusters.dofs: lu.solve(B, d),
               lu.nbytes(), {})
        del lu


def law_rate(body: Body, laws, mu):
    """``traction -> slip rate`` on a body's coordinates, from a law per
    fault: ``laws`` maps EVERY solved fault to its law (``rheology``), ``mu``
    to the unrelaxed shear modulus of the region holding it. A solved fault
    with no law would be an algebraic constraint, which nothing here steps."""
    from . import rheology

    missing = [f.name for f in body.faults if f.name not in laws]
    if missing:
        raise KeyError(f"no law for the solved fault(s) {missing}: every "
                       f"FAULT_TRACTION fault of a stepped run needs one")
    parts = []
    for f in body.faults:
        law = rheology.from_spec(laws[f.name])
        if not hasattr(law, "rate"):
            raise TypeError(f"fault {f.name!r}: {law!r} is not a fault law "
                            f"(it has no rate)")
        if f.frame.shape[1] != 2:
            raise ValueError(f"fault {f.name!r} opens (allow_opening): the "
                             f"laws are of a fault that slides")
        parts.append((f.start, f.stop, law, float(mu[f.name])))

    def rate(traction):
        out = np.empty_like(traction)
        for a, b, law, m in parts:
            out[a:b] = law.rate(traction[a:b].reshape(-1, 2), m).ravel()
        return out
    return rate


@dataclass
class History:
    """What ``march`` returns: the faults at the times asked for.

    ``y[i]`` is the whole state ``[slip; p]`` at ``t[i]`` and ``traction[i]``
    the traction the law saw there, both in the body's coordinates.
    ``nfev`` counts the law's evaluations (six to a Dormand-Prince step, a
    stage each of a Chebyshev one)."""
    t: np.ndarray
    y: np.ndarray               # (n_t, 1 + m, n)
    traction: np.ndarray        # (n_t, n)
    body: Body
    events: tuple
    nfev: int
    method: str = ""

    @property
    def slip(self) -> np.ndarray:
        return self.y[:, 0]

    def fault(self, name: str, what: str = "slip") -> np.ndarray:
        """``(n_t, n_dofs, 3)``: a fault's Burgers vector, or the traction on
        it (``what="traction"``), as Cartesian vectors at its dof nodes."""
        f = next(f for f in self.body.faults if f.name == name)
        src = self.slip if what == "slip" else self.traction
        return np.stack([f.cartesian(v[f.start:f.stop]) for v in src])

    def slots(self, i: int) -> dict:
        """Every unknown at ``t[i]``, as a solve returns them."""
        return self.body.slots(self.t[i], self.y[i], self.events)


def _events(events) -> tuple:
    out = sorted((float(e[0]), float(e[1])) if np.ndim(e) else (float(e), 1.0)
                 for e in events)
    if not out:
        raise ValueError("march: no event, so nothing ever loads the faults")
    return tuple(out)


def march(body: Body, rate, t, events=(0.0,), rtol: float = None,
          atol: float = None, method: str = None) -> History:
    """Integrate a body's faults under ``rate`` and return them at ``t``.

    ``rate`` maps the traction on the faults to their slip rate, both in the
    body's coordinates (``law_rate`` makes one from laws). ``events`` are the
    times at which the model's prescribed data are applied as a step -- a
    time, or ``(time, amplitude)`` -- the first of them the start: the faults
    are unslipped before it, and they do not slip AT a step, whatever the
    law, because a rate is finite. ``rtol`` is the integrator's, and ``atol``
    its absolute tolerance on slip as a fraction of ``body.slip_scale()``.

    ``method`` is ``scipy.integrate.solve_ivp``'s, or ``"RKC"``, the
    stabilised method of ``mbem.rkc``; left ``None`` it is the one that
    suits the body, the embedded pair for a dense ``Body`` and RKC for a
    ``KernelBody``, whose faults are fine enough to be stiff, and for any
    body whose fit has a tail, whose memory is stiff whatever the fault
    (``defaults.STEPPER_TAIL_DECADES``).

    A relaxing body is refused past the end of its fit's window, counted
    from the first event: beyond it nothing constrains the kernel.
    """
    from scipy.integrate import solve_ivp

    from .rkc import RKC

    rtol = defaults.STEPPER_RTOL if rtol is None else rtol
    atol = defaults.STEPPER_ATOL if atol is None else atol
    if method is None:
        method = (defaults.STEPPER_METHOD_KERNEL
                  if isinstance(body, KernelBody) or body.tails
                  else defaults.STEPPER_METHOD)
    solver = {"RKC": RKC}.get(method, method)
    events = _events(events)
    t = np.atleast_1d(np.asarray(t, float))
    if t.ndim != 1 or t.size == 0 or np.any(np.diff(t) < 0.0):
        raise ValueError("march: t must be one or more non-decreasing times")
    t0 = events[0][0]
    if t[0] < t0:
        raise ValueError(f"march: t = {t[0]:g} is before the first event "
                         f"({t0:g})")
    if body.window is not None and t[-1] - t0 > body.window[1] * (1 + 1e-9):
        raise ValueError(
            f"march: t = {t[-1]:g} is {t[-1] - t0:g} after the first event, "
            f"past the end of the fit's window ({body.window[1]:g}): beyond "
            f"it nothing constrains the body's memory. Lay the Laplace "
            f"points out for a longer window")
    n, m = body.n, body.tau.size
    scale = body.slip_scale()
    atol_abs = atol * scale if scale > 0.0 else np.finfo(float).tiny
    count = [0]

    def fun_for(active):
        def fun(tt, y):
            count[0] += 1
            Y = y.reshape(1 + m, n)
            with np.errstate(over="ignore", invalid="ignore"):
                v = rate(body.traction(Y, body.load(tt, active)))
            # A TRIAL stage may overshoot far enough for a law exponential in
            # traction to overflow. That is a step too long, not a result:
            # hand the integrator a rate so large that it rejects the step.
            v = np.nan_to_num(v, nan=defaults.STEPPER_REJECT,
                              posinf=defaults.STEPPER_REJECT,
                              neginf=-defaults.STEPPER_REJECT)
            out = np.empty_like(Y)
            out[0] = v
            if m:
                out[1:] = v[None, :] - Y[1:] / body.tau[:, None]
            return out.ravel()
        return fun

    edges = [e[0] for e in events if e[0] < t[-1]] + [float(t[-1])]
    y = np.zeros((1 + m) * n)
    got = np.empty((t.size, (1 + m) * n))
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        sel = np.flatnonzero((t >= a) & ((t <= b) if last else (t < b)))
        active = tuple(e for e in events if e[0] <= a)
        te = np.unique(np.concatenate([t[sel], [b]]))
        fun = fun_for(active)
        v0 = np.abs(fun(a, y)[:n])
        if v0.max() >= defaults.STEPPER_REJECT:
            raise FloatingPointError(
                f"march: at t = {a:g} the law starts at a rate beyond "
                f"{defaults.STEPPER_REJECT:g}, or at none: a law exponential "
                f"in traction does where the traction is over a hundred "
                f"times its scale, and no step is short enough")
        # the integrator's own first step probes with one that a stiff law
        # overshoots; start from a step that moves the slip by a millionth
        # of its scale, and let it grow
        first = (min(b - a, defaults.STEPPER_FIRST_STEP * scale / v0.max())
                 if scale > 0.0 and v0.max() > 0.0 else None)
        with np.errstate(over="ignore", invalid="ignore"):
            sol = solve_ivp(fun, (a, b), y, method=solver, t_eval=te,
                            rtol=rtol, atol=atol_abs, first_step=first)
        if sol.status != 0:
            raise RuntimeError(f"march: the integrator stopped at "
                               f"t = {sol.t[-1] if sol.t.size else a:g}: "
                               f"{sol.message}")
        got[sel] = sol.y[:, np.searchsorted(te, t[sel])].T
        y = sol.y[:, -1]
    if len(edges) == 1:                          # asked only AT the start
        got[:] = 0.0
    Y = got.reshape(t.size, 1 + m, n)
    trac = np.stack([body.traction(Y[i], body.load(t[i], events))
                     for i in range(t.size)])
    return History(t=t, y=Y, traction=trac, body=body, events=events,
                   nfev=count[0], method=str(method))
