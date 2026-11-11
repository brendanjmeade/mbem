# Linear viscoelasticity on the boundary: derivation

Quasistatic, time-dependent, linear viscoelasticity for a body made of
homogeneous regions, with unknowns on the region boundaries only. No volume
mesh, no stored history.

Sections 2 and 3 are properties of the kernels, and section 5, the Laplace
route, is built on them (`mbem.rheology`, `mbem.laplace`, `Run.laplace`); all
three are gated by `tests/gates/mbem/verify_viscoelastic.py`. Sections 1 and 4
are the algebra around them. Section 6, time-stepping, is a design and is NOT
built. Section 7 is what is open.

Notation as in `clq-derivation.md`: $\mathbf d=\mathbf x-\mathbf y$,
$R_\varepsilon=\sqrt{|\mathbf d|^2+\varepsilon^2}$. One new symbol carries the
whole note:

$$ g = \frac{1}{1-\nu} . $$

## 1. Correspondence

Each region is isotropic and linear viscoelastic. With the Laplace transform
$\hat f(s)=\int_0^\infty f(t)e^{-st}dt$ and the operational moduli
$\tilde\mu(s)$, $\tilde K(s)$ ($s$ times the transform of the relaxation
modulus), the constitutive law is

$$ \hat\sigma = 2\tilde\mu(s)\,\operatorname{dev}\hat\epsilon
   + \tilde K(s)\operatorname{tr}(\hat\epsilon)\,\mathbf I . $$

So the transformed quasistatic problem is an elastic problem on the same
geometry, each region at $(\tilde\mu,\ \tilde\lambda=\tilde K-\tfrac23\tilde\mu)$,
provided the boundary-condition TYPE of every patch is fixed in time.

**Bulk is elastic** ($\tilde K=K$) and only shear relaxes. This is the
PSGRN/Pollitz convention, and it is the general case: Poisson's ratio moves in
time. Moduli are always formed as $(\tilde\mu,\tilde\lambda)$, never through
$1/(1-2\nu)$ (CLAUDE.md rule 8); a relaxing region goes to $\nu\to1/2$.

| rheology | $\tilde\mu(s)$ | $s\to\infty$ | $s\to0$ |
|---|---|---|---|
| Maxwell, $\tau_M=\eta/\mu$ | $\mu\,s/(s+1/\tau_M)$ | $\mu$ | $0$ |
| Burgers | $1/\tilde\mu = 1/\mu_M + 1/(\eta_M s) + 1/(\mu_K+\eta_K s)$ | $\mu_M$ | $0$ |

**Step data.** If every prescribed quantity is $d_0H(t)$, then
$\tilde u(s)\equiv s\,\hat u(s)$ is the ELASTIC solution at moduli$(s)$ with
data $d_0$. Nothing but the material changes between values of $s$, and

$$ u(t)=\mathcal L^{-1}\!\left[\tilde u(s)/s\right],\qquad
   u(0^+)=\tilde u(\infty),\qquad u(\infty)=\tilde u(0) . $$

The last is the equivalence theorem of Fukahata & Matsu'ura (2006). This holds
only for data that do not themselves contain a modulus: a prescribed traction
written as $C{:}A\,\mathbf n$, or a displacement written as $\tau z/\mu$, is not
step data.

**Mollification commutes with all of this.** $\varepsilon$ is geometry. The
mollified kernel at moduli$(s)$ is the transform of the response to the same
blob force, and a fault's smeared eigenstrain $\epsilon^*$ is fixed in space.

## 2. Every operator is two scalar functions times geometry

### 2.1 Force kernel

$C_1(3-4\nu)=(4-g)/(16\pi\mu)$, $C_1=g/(16\pi\mu)$, and the blob term's
$2(1-\nu)C_1=1/(8\pi\mu)$ has no $\nu$ in it. So

$$ G^\varepsilon=\frac1\mu\Big[\Gamma_A+g\,\Gamma_B\Big],\qquad
 \Gamma_A=\frac{\delta_{ij}}{4\pi R_\varepsilon}
          +\frac{\varepsilon^2\delta_{ij}}{8\pi R_\varepsilon^3},\qquad
 \Gamma_B=\frac1{16\pi}\Big(\frac{d_id_j}{R_\varepsilon^3}
          -\frac{\delta_{ij}}{R_\varepsilon}\Big). $$

The whole mollification correction sits in the material-free part. In the
code's U basis `[B1, B2, B3]` (`kernels/tri_kernels.py`) the coefficients are
$\big((4-g),\ 2,\ g\big)/(16\pi\mu)$.

### 2.2 Slip kernel

The pairing (1.1) of `clq-derivation.md` has a $\lambda$ term,
$\lambda\,n_j\,\partial_mG_{im}$. Its divergence is

$$ \partial_m\big(\Gamma_A+g\Gamma_B\big)_{im}
   = \frac{g-2}{16\pi}\Big[\frac{2d_i}{R_\varepsilon^3}
     +\frac{3\varepsilon^2d_i}{R_\varepsilon^5}\Big],\qquad
   g-2=-\frac{1-2\nu}{1-\nu}, $$

so $\lambda/\mu=2\nu/(1-2\nu)$ meets a factor $(1-2\nu)$ and the product is
$-2(g-1)$: finite at $\nu=1/2$, mollified terms included. In the code's T basis
`[N1, N2, N3, R1, R2, R3]` this cancellation is a linear dependency of the
basis itself,

$$ N_2 = N_1 + 3N_3 , $$

because $\sum_m P^{(2)}_{imm}=V^3_i+3\varepsilon^2V^5_i$. The coefficients
`t_coeffs` $c_1..c_3$ each grow like $\lambda/\mu$; the combinations that
multiply independent matrices do not:

$$ T = \frac{g-1}{4\pi}N_1+\frac{3(g-1)}{8\pi}N_3
     +\frac{4-g}{16\pi}R_1-\frac{g}{16\pi}R_2+\frac{3}{8\pi}R_3
   \;=\; T_A+g\,T_B . $$

$T$ contains no $\mu$ at all.

### 2.3 Elastic stress

The stress of a mollified double layer with the eigenstress removed,
$\sigma=C{:}(\nabla u-\epsilon^*)$, is

$$ \sigma = \mu\,\big(S_A+g\,S_B\big) . $$

Reason: it is the stress of a smooth eigenstrain field in a whole space, whose
pressure $-K\operatorname{tr}\epsilon^{el}$ stays finite as $K\to\infty$, so the
elastic dilatation carries a factor $(1-2\nu)$ that cancels $\lambda$'s pole.
The RAW total stress does not have this form. It differs by
$C{:}\epsilon^*=2\mu\epsilon^*+\lambda\operatorname{tr}(\epsilon^*)\mathbf I$,
and $\lambda/\mu=2(g-1)/(2-g)$ is not affine in $g$. Removing the eigenstress
(CLAUDE.md rule 4) is therefore also what makes the viscoelastic problem cheap.

### 2.4 Summary

| operator | form |
|---|---|
| slip $\to$ displacement (`KERNEL_T`) | $T_A+g\,T_B$ |
| traction $\to$ displacement (`KERNEL_U`) | $\mu^{-1}(U_A+g\,U_B)$ |
| slip $\to$ elastic stress, traction (`KERNEL_HT`, readout) | $\mu\,(S_A+g\,S_B)$ |

Every $*_A$, $*_B$ depends on the mesh and on $\varepsilon$ only. Per region
the time dependence enters through $g(s)$, $\tilde\mu(s)$ and $1/\tilde\mu(s)$,
with

$$ g(s)=\frac{2\,(3K+\tilde\mu)}{3K+4\tilde\mu}\ \in\ [\,g_e,\ 2\,] . $$

## 3. One $\varepsilon$ on each closed surface

With ONE blob the kernel is a single exact elastic field, and two identities
hold on a closed surface $\partial V$. They hold on the faceted mesh itself,
to rounding, because each triangle's integral is closed-form and the
divergence theorem applies to the polyhedron.

**(i) Rigid translation.** The traction of the blob-force solution integrates
to minus the force enclosed:

$$ \oint_{\partial V}T(\mathbf x,\mathbf y)\,dS_y
   = -\,\mathbf I\int_V\phi_\varepsilon(\mathbf x-\mathbf y)\,dV_y , $$

which has no material in it. The calibrated diagonal (`equations.py`) is this
row sum, so with one $\varepsilon$ it is the same for every $s$.

**(ii) Pressure.** At $g=2$, $\mu\,G^\varepsilon$ is Cortez's regularized
Stokeslet, which is divergence-free, so

$$ \oint_{\partial V}\big(\Gamma_A+2\Gamma_B\big)\,\mathbf n\,dS_y = 0 : $$

a uniform pressure on a closed surface moves nothing in an incompressible
body. This null mode is what lets the small compressibility $K$ alone set the
pressure of a nearly relaxed region.

A per-element $\varepsilon$ (`eps="auto"`) uses a different blob on each
triangle. The sum is then not one field and both identities fail, at the level
of the variation of $\varepsilon$ over the surface. The defect in (ii) is
amplified by $K/\tilde\mu(s)$.

**Rule.** The closed boundary of a region that relaxes carries one scalar
$\varepsilon$.

Measured on the Eshelby sphere of `verify_solved_bvp` B9 with the HOST
relaxing at fixed $K$ (hydrostatic strain concentration against the closed
form; the deviatoric part does not care):

| $\tilde\mu(s)/K$ | `eps="auto"` | one scalar $\varepsilon$ |
|---|---|---|
| 4.6e-3 | -0.2 % | -0.1 % |
| 4.6e-4 | -9.4 % | -0.2 % |
| 4.6e-5 | -54 % | -0.2 % |
| 4.6e-6 | -95 % | -0.2 % |

With one $\varepsilon$ the solve is unchanged down to $\nu=0.499998$; the
condition number grows as $K/\tilde\mu$ for an enclosed region and about as
its square for a region with outer faces.

## 4. Poles

Each scalar function is rational in $s$ with real, non-positive poles, so each
is a constant plus a few terms $s/(s+\beta)$, plus $1/s$ for a fluid.

Maxwell, $a=1/\tau_M$:

$$ \tilde\mu=\mu\frac{s}{s+a},\qquad
   \frac1{\tilde\mu}=\frac1\mu\Big(1+\frac as\Big),\qquad
   g(s)=2+(g_e-2)\frac{s}{s+1/\tau_2},\qquad
   \tau_2=\tau_M\frac{3K+4\mu}{3K} . $$

Step responses: $\mu e^{-t/\tau_M}$; $(1+t/\tau_M)/\mu$, the steady creep of a
fluid; $2+(g_e-2)e^{-t/\tau_2}$.

Burgers: $\tilde\mu$ is a ratio of quadratics, so it has two poles, and
$3K+4\tilde\mu=0$ gives two more for $g$.

A problem with prescribed slip and displacement relaxes to a finite state. A
problem with a prescribed traction in a Maxwell or Burgers region creeps
forever (the $1/s$ term): a crack under a fixed stress drop never stops
slipping.

## 5. The Laplace route

A run declares which regions relax and how (`Run.laplace`, `mbem.rheology`);
its states are DERIVED from the model the builder returns: the elastic one,
then one elastic solve per real $s_k$, all on one assembly. Then fit, with
$\tau_j$ fixed at the reciprocals of the $s_k$ (`mbem.laplace.Prony`),

$$ \tilde u(\mathbf x,s)=a_0(\mathbf x)+\sum_ja_j(\mathbf x)\frac{s}{s+1/\tau_j}
   \ \Big(+\,\frac{c(\mathbf x)}{s}\ \text{if traction-driven}\Big)
   \quad\Longleftrightarrow\quad
   u(\mathbf x,t)=a_0+\sum_ja_je^{-t/\tau_j}\ (+\,c\,t) . $$

The fit is linear, and it is the same linear map for every slot and every
observation point: `weights(t)` times the stack of state solutions. The elastic
state is always one of its rows, which pins $u(0^+)$ exactly. Repeating
earthquakes of period $T$ sum in closed form,
$\sum_{n\ge0}e^{-(t+nT)/\tau_j}=e^{-t/\tau_j}/(1-e^{-T/\tau_j})$. The result is
trusted for $t$ inside the span of the $\tau_j$, not beyond it.

Measured on exact transforms (one pole, two poles, an $n$-fold pole, a
creeping response), worst case over three decades of time:

| points per decade | solves | $u$ | $t\,\dot u$ |
|---|---|---|---|
| 2 | 9 | 4.4e-3 | 6.7e-2 |
| 3 | 13 | 9.5e-4 | 1.2e-2 |
| 4 | 17 | 4.2e-5 | 1.9e-3 |

Three is the default: the inversion then sits well under the spatial error. A
rate wants four.

**Time has no unit here.** A rheology is stated in relaxation times and
modulus ratios, never in viscosities, so a run is in units of whatever its
$\tau$ is in.

**Closed forms, each checked per $s$ and in time** (the gate inverts the same
closed form on a complex contour at 30 digits, which shares nothing with the
real-axis fit):

| case | what it exercises | spatial bias, flat in $s$ |
|---|---|---|
| dislocation in a relaxing full space | the fault INSIDE a relaxing region; $u$ moves with $g$, $\sigma$ relaxes to zero | none (kernel parity with `clq` at $\nu(s)$ to 1e-11) |
| Eshelby sphere, host relaxing | two regions, $\lambda$ live, two relaxation times | 1.6 % hydrostatic, 0.8 % deviatoric |
| torsion of a two-layer cylinder | a relaxing region with an interface, a free wall and a clamped base | $-5$ % (847 P0 triangles) |
| penny crack under a sustained stress drop | a solved (traction) fault that creeps for ever | $-0.5$ % in moment |

In every solved case the time-domain answer carries the elastic gate's own
spatial bias and nothing more.

**The plate over a substrate** (`mbem.cases.layered_box`,
`configs/plate_substrate.py`) is the Nur-Mavko / Savage-Prescott model on a
finite box. Its reference is the antiplane image series
(`layered_box.image_series`), which is plain arithmetic in the substrate's
modulus and so is its own Laplace-domain reference at $\tilde\mu(s)$; for a
Maxwell substrate $G(s)$ has one pole and the time response is a series of
incomplete gamma functions (`maxwell_series`).

## 6. Design: time-stepping with memory on the boundary

Multiplying by $s/(s+\beta)$ is, in time, the variable
$m(t)=\int_0^te^{-\beta(t-\tau)}\dot x(\tau)\,d\tau$, which obeys
$\dot m=-\beta m+\dot x$ and updates in one line per step:

$$ m^n=e^{-\beta\Delta t}m^{n-1}+\psi\,(x^n-x^{n-1}),\qquad
   \psi=\frac{1-e^{-\beta\Delta t}}{\beta\Delta t} . $$

Because every operator is geometry times a few such functions, the history of
a region is a few vectors of the size of its boundary unknowns. The implicit
step is an elastic solve at effective moduli, for example
$g_{\rm eff}=2+(g_e-2)\psi$, which lies between the unrelaxed and relaxed
values and so is an admissible material; one factorization serves every step
of equal length. History enters the right-hand side through the same geometric
operators applied to the stored vectors. This is the method of Mesquita & Coda
and of Panagiotopoulos, Mantic & Roubicek without their restriction that bulk
and shear relax alike.

In a relaxing region the eigenstress is hereditary,
$\sigma^*(t)=\int_0^tC(t-\tau)\,d\epsilon^*(\tau)$: after a step of slip it
decays with the shear modulus.

## 7. Open

- **A plate with free sides floats.** As $\tilde\mu(s)\to0$ only the shear of
  the substrate holds the plate against a rigid horizontal translation. A
  dislocation is self-equilibrated, but its discrete net force is small rather
  than zero, and the translation it causes is that force over $\tilde\mu(s)$.
  On the split fault box of B6 with a relaxing lower region, the net
  translation of the free surface (zero by symmetry) is 1 % of the deformation
  at $\tilde\mu/\mu=10^{-2}$, 11 % at $10^{-3}$ and as large as the
  deformation at $10^{-4}$, and the condition number grows as
  $\tilde\mu^{-2}$. It is the same with one $\varepsilon$ or per-element
  $\varepsilon$. Clamping the plate's sides removes it (0.3 % at $10^{-4}$,
  condition number as $\tilde\mu^{-1}$); `deflate` does not, because the mode
  is not a translation of every unknown. The force-over-$\tilde\mu$ reading
  fits the $1/\tilde\mu$ growth but has not been isolated.
- A region that extends to infinity while touching an interface (a true
  half-space under a plate). Regions must close today.
- Gravity. A buoyancy term on the free surface and on density interfaces is a
  boundary term, but a new condition type.
- A solved (traction) fault that crosses an interface.
- Nonlinear rheology is out of scope: it needs the viscous strain in the
  volume.

## References

Rizzo & Shippy (1971) SIAM J. Appl. Math. Kusama & Mitsui (1982) Appl.
Math. Model. Sim & Kwak (1988) Comput. Struct. Schanz & Antes (1997)
Comput. Mech. Gaul & Schanz (1999) Comput. Methods Appl. Mech. Eng. 179.
Pan, Sassolas, Amadei & Pfeffer (1997) Comput. Mech. 19. Syngellakis & Wu
(2004) Eng. Anal. Bound. Elem. 28. Mesquita & Coda (2007) Appl. Math. Model.,
"A boundary element methodology for viscoelastic analysis: Part II without
cells". Panagiotopoulos, Mantic & Roubicek (2014) arXiv:1402.6631. Hetland &
Hager (2005) J. Geophys. Res. 110. Fukahata & Matsu'ura (2006) Geophys. J.
Int. 166. Wang, Lorenzo-Martin & Roth (2006) Comput. Geosci. 32. Barbot, Moore
& Lambert (2017) Bull. Seismol. Soc. Am. Miyake & Noda (2019) Earth Planets
Space 71. Mallick, Lambert & Meade (2022) J. Geophys. Res. 127. Loveless &
Meade (2025) Earth Space Sci.
