# Linear viscoelasticity on the boundary: derivation

Quasistatic, time-dependent, linear viscoelasticity for a body made of
homogeneous regions, with unknowns on the region boundaries only. No volume
mesh, no stored history.

Sections 2 and 3 are properties of the kernels, and section 5, the Laplace
route, is built on them (`mbem.rheology`, `mbem.laplace`, `Run.laplace`); all
three are gated by `tests/gates/mbem/verify_viscoelastic.py`. Sections 1 and 4
are the algebra around them. Section 6, time-stepping, is a design and is NOT
built. Section 7, gravity, is built: buoyancy on the free surface and on
density interfaces, with nothing in the volume; under it a thrust in the
plate held to an independent reference, the earthquake cycle of a locked
fault, and a megathrust through the interface. Section 8 gives the box
walls that are a window on a layered half-space, and the reference in three
dimensions they are made of. Section 9 is a fault that creeps, section 10 a
plate that subducts. Section 11 is what is open.

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

That reference is an infinitely long fault over a half-space, which a box can
only approach, so the study is a ladder: the box half-width doubled twice with
the fault kept half as long as the box is wide (plate 30 km, fault to 15 km,
equal moduli). Model over closed form at mid-strike, clamped plate sides (free
sides are 1-3 points closer at the smaller boxes and the same at the largest):

| half-width | $u(0^+)$, 16-32 km | $u(10\tau)$, 16-32 km | postseismic at $\tau$ | postseismic at $10\tau$ |
|---|---|---|---|---|
| 400 km | 0.980 | 0.912 | 0.862 | 0.768 |
| 800 km | 0.995 | 0.970 | 0.960 | 0.912 |
| 1600 km | 0.997 | 0.989 | 0.988 | 0.969 |

A Burgers substrate (`--set tau_K=0.1 --set t_min=0.02`: transient time
$0.1\tau$, Kelvin spring equal to the unrelaxed modulus, the window reaching
below the transient) converges on the same ladder at the same rate:
postseismic 0.84 / 0.95 / 0.98 at $\tau$ and 0.76 / 0.91 / 0.97 at $10\tau$,
and 0.87 / 0.96 / 0.99 at $0.03\tau$, inside the transient. It has no series
in closed form, so its reference is the image series at $\tilde\mu(s_k)$ put
through the same fit as the solution, which the gate holds to the contour
inversion (2e-4).

The postseismic part $u(t)-u(0^+)$ is the long-wavelength part and feels the
box first. Within 8 km of the fault the total displacement is within 1.1 % at
every size and time. So a result near the fault at a few Maxwell times wants a
box some fifty plate thicknesses across; `mbem run configs/plate_substrate.py
--sweep half_x=400,800,1600 --sweep sides=clamped,free` draws the figure.

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

## 7. Gravity

Built (`mbem.model.Gravity`) and gated against a body that has an exact
solution under gravity. In this section $w=\rho\,\mathrm g$ is the weight
density (GPa/km; 0.0294 for 3000 kg/m$^3$), so that $g$ keeps its meaning,
$1/(1-\nu)$.

**The equations.** Before it is deformed the body is at rest under gravity,
its prestress hydrostatic: $\sigma_0=-p_0\mathbf I$,
$\nabla p_0=-w\,\hat{\mathbf e}$, $\hat{\mathbf e}$ pointing up. A displacement
carries prestressed material with it, so there are two stress increments: the
MATERIAL one $\tau^L$, the change in the stress of a particle, and the LOCAL
one, the change at a fixed point of space,

$$ \tau=\tau^L-w\,(\mathbf u\cdot\hat{\mathbf e})\,\mathbf I . $$

(A column lifted by $u_z$ puts at each point the material from $u_z$ deeper:
more pressure at the point, none in any particle.) With uniform density in a
region and no self-gravitation, equilibrium of the perturbation is

$$ \nabla\cdot\tau+w\,(\nabla\cdot\mathbf u)\,\hat{\mathbf e}=0 , $$

the second term being the buoyancy of material whose density the deformation
has changed.

**What is kept.** Two things are done, and both are exact for an
incompressible material: the internal buoyancy
$w(\nabla\cdot\mathbf u)\hat{\mathbf e}$ is dropped, and Hooke's law is applied
to $\tau$ (the pressure of an incompressible body is not constitutive, so it
cannot tell $\tau$ from $\tau^L$). Then

$$ \nabla\cdot\big(C{:}\epsilon(\mathbf u)\big)=0 $$

inside every region, which is the equation the kernels already solve: gravity
has left the volume. For a compressible material the error is of relative
order $wL/K$, $L$ the scale of the deformation: 6 % per 100 km at $K=50$ GPa.
The compressible treatments in use differ from one another at that same order.
Keeping the internal buoyancy in uniform compressible layers has no solution
at long wavelength (Wang 2005); keeping only the advection of prestress, in
the material frame, leaves a body force $w\nabla u_z$, which a boundary method
would need the volume for.

**Where gravity went.** What is continuous across a material surface is the
material traction $\tau^L\mathbf n$. So, for the local traction
$\mathbf t=\tau\mathbf n$ and any normal $\mathbf n$:

| surface | condition |
|---|---|
| the free surface: air above, applied load $\bar{\mathbf t}$ | $\mathbf t=\bar{\mathbf t}-w\,(\mathbf u\cdot\hat{\mathbf e})\,\mathbf n$ |
| a welded interface between regions 1 and 2 | $\mathbf t_2=\mathbf t_1-(w_2-w_1)\,(\mathbf u\cdot\hat{\mathbf e})\,\mathbf n$ |
| a fault, inside one region | $\mathbf t$ continuous: no term |
| an outer face that is a cut and not a surface (the sides and base of a box) | $\mathbf t=\bar{\mathbf t}$: no term |

*The fault.* Its two faces at one point of space are different particles, but
$\tau$ is the stress AT THE POINT, and that is continuous. Prescribed slip and
the rows of a traction fault are untouched, and a fault needs no density.

*The cut.* The body continues beyond the side of a box. If what lies beyond is
hydrostatic at the same density it loads the face with exactly the term that
would otherwise appear, and nothing is left. That is also a reservoir: a
relaxed substrate with such sides holds no pressure, as under a half-space it
would not, while clamped sides trap it.

One rule covers the table (`RegionModel.buoyancy`, its only statement):
**region $R$ sees on patch $p$ the traction $\mathbf t_p+B\,\mathbf u_p$,**

$$ B=-(w_R-w_{\rm ref})\ \mathbf n\otimes\hat{\mathbf e}, $$

with $w_{\rm ref}$ zero on a patch named as the free surface, the first
region's weight on an interface, and $w_R$ everywhere else. In the equations
that is one more block on the displacement columns,

$$ A[\,\cdot\,,\mathbf u_p]\ \mathrel{+}=\ -\sigma(R,p)\,G^{m(R)}B_p , $$

a single layer whose density is proportional to the displacement:
$\tilde\mu^{-1}(U_A+g\,U_B)$ times $w$ times geometry. No kernel work, and
nothing outside the structure of section 2. $w$ is not a modulus, so section 1
stands: a state under gravity is an elastic solve.

**It contains the classical results.** On a half-space a normal load of
wavenumber $k$ meets the stiffness $w+\tilde\mu\,g\,k$, whose ratio
$w/(\mu g)=(1-\nu)\rho\,\mathrm g/\mu$ is the buoyancy wavenumber of Barbot &
Fialko (2010). For an incompressible Maxwell half-space ($g=2$) its one pole
is at $\tau=\tau_M+2\eta k/w$: the relaxation time of a viscous half-space
under gravity (Haskell 1935) plus the Maxwell time. And relaxed, a substrate
is hydrostatic, so a plate on it feels $w_p$ per unit deflection at its top
and $w_s-w_p$ at its base, $w_s$ in all: flexural isostasy.

**Stable stratification only, on the Laplace route.** Where the heavier side
is above, a body that relaxes has a growing mode, and the fit of section 5 has
only decaying exponentials in it; the runner refuses
(`rheology.require_stable`). An elastic run may keep such a contrast. With
stable contrasts this treatment has no instability of its own: the stiffness
above is positive for every $k$ and every $s>0$.

**What a readout means.** On the free surface `Patch.value` is the applied
load. A traction slot is the local traction (on an interface, the first
region's). `evaluate_stress` returns $\tau$, the stress at a fixed point; the
material stress is $\tau+w(\mathbf u\cdot\hat{\mathbf e})\mathbf I$. The shear
components are the same; the normal stress on a fault differs by
$w\,(\mathbf u\cdot\hat{\mathbf e})$.

**Anchors.** A flat earth under gravity has no exact solution on a finite
body, so the reference is a sphere: elastic shells welded on a rigid core,
loaded on the outer surface by $\sigma P_2(\cos\theta)$, with RADIAL buoyancy
(`Gravity(center=...)`) on the surface and on the density interface. It is a
4-by-4 system per shell in Love's spheroidal solutions, itself pinned by
Kelvin's stiffness of the incompressible sphere. It is physical, not
manufactured, so it fixes the sign and the size of the term.

| case | measured |
|---|---|
| one elastic shell, $wa/\mu=2$: surface displacement over exact | 0.984 without gravity, 0.989 with; gravity stiffens it by 28 %, and the solved ratio is the exact one to 0.4 % |
| its interior displacement | 0.6 % (36 % if the readout leaves the buoyancy out) |
| two shells, weights 150 and 60 | surface 0.984, interface 0.992; the contrast moves the interface by 20 % |
| the shell relaxing, per $s$ and in time, Maxwell and Burgers | 0.974 .. 0.998 per $s$, 0.976 .. 0.995 in time, with no secular term |
| relaxed to $\tilde\mu/\mu=10^{-4}$ | $U\,w/\sigma=1$ to 4e-4, where without gravity the same load has sunk 2500 times as far |
| a relaxed flat box under a smooth load, `up` vertical | $u_z=(P-\text{load})/w$ to 3e-3: $P=0$ with free sides, the mean load over $1+wH/K$ with clamped ones |

The P0 bias of the elastic state vanishes as the shell relaxes: the isostatic
answer $\sigma/w$ has no elasticity in it. The compressed backends carry the
term exactly as the dense one does (ACA: operator, preconditioner blocks and
its Gauss-Seidel sweep to 5e-8 of dense; FMM: the same error with and without
gravity).

**On the plate of section 5** (`--set rho_plate=2700 --set rho_substrate=3300`,
clamped sides, half-widths 800 and 1600 km) gravity does what it should to a
long vertical strike-slip fault, which is very little. The fault-parallel
displacement at mid-strike moves by at most 6e-4 of its peak, and the
comparison with the antiplane closed form is unchanged to the third digit.
The vertical motion near the fault, 5 % of the slip at its ends, is smaller
in rms by 4 % at $0^+$, 6 % at $\tau$, 9 % at $3\tau$ and 16 % at $10\tau$,
the same on both boxes. A state costs one more single-layer product on the
surface and the interface (solves 52 s against 42, 159 against 129) and
converges in fewer iterations (at most 27 against 38).

**A thrust in the plate** (`--set dip=30 --set rake=90`: `layered_box` takes
a dip, the model a rake) is where gravity is first order. Dip slip bends the
plate, and under a wavelength the substrate has relaxed beneath, only the
plate's stiffness and gravity are left to resist. No closed form covers it,
so the reference is a second calculation that shares nothing with the
boundary elements (`layered_box.dip_slip_profile`): plane strain, one
wavenumber at a time, the slip an eigenstrain that enters only as a jump at
its depth, the conditions of the table above at the surface and the
interface, a half-space or a rigid base below; inverted as the half-space
closed form (Freund & Barnett 1976) plus the transform of the difference,
and put through the same fit in time as the solution. A reference is only as
good as its own anchors:

| the reference against | measured |
|---|---|
| cutde's half-space, a very long thrust, three dips | 4e-6 of the slip |
| a buried fault's transform, inverted with no closed form's help | 3e-7 |
| a load on a half-space, $w+\mu g k$; on a plate over a fluid, $w_s+Dk^4$ | rounding; 4e-4 at $kH=0.09$ |
| a dislocated FREE plate, its halves turned by $6\,b_x d\,(H-d)/H^3$ and closed by $b_x d/H$ | 7e-5 |
| its fit in time against a contour inversion (Talbot), to $100\tau$ | 3e-4 |
| a locked fault's velocity through the cycle against its Fourier series on the imaginary axis, $T=20\tau$ | 4e-4 |
| a fault THROUGH the interface: between like materials, which is no interface; under a plate 10 m thick, where the closed form holds whatever the substrate | 7e-15; 5e-4 |

The free plate is the one exact result in which the source meets the plate's
lower face. Far from the fault each half is rigid, and reciprocity with pure
bending and with a pull gives the rotation and the closure of the mid-planes
for a fault of depth $d$ and horizontal slip $b_x$, whatever its dip.

*What gravity does.* A 30-degree thrust to 15 km in a 30 km plate, 2700 over
3300 kg/m$^3$, a Maxwell substrate; displacements over the slip, in the
reference. At the earthquake the hanging wall is up 0.41 at the trace and a
trough of 0.11 lies over the fault's lower edge. Relaxation deepens the
trough and raises a bulge either side of it:

| depth of the trough | $0^+$ | $\tau$ | $10\tau$ | $100\tau$ | relaxed |
|---|---|---|---|---|---|
| under gravity | 0.11 | 0.14 | 0.30 | 0.52 | 0.52 |
| without | 0.11 | 0.14 | 0.33 | 0.97 | none: a free plate keeps a kink |

Gravity is little for ten Maxwell times and a factor of two by a hundred,
when the plate is afloat (flexural parameter 69 km). Without it the answer
also belongs to whatever lies below: a rigid base 400 km down takes the
trough at $100\tau$ from 0.97 to 0.81, and under gravity from 0.52 to 0.49.

*The model against it* (clamped sides, at mid-strike, $|x|<120$ km, the fault
half as long as the box is wide). The elastic state is within 3 % in $u_x$
and 4 % in $u_z$ of the reference with a rigid base where the model's is, 2 %
and 3 % on the finer plate. The misfit of the postseismic displacement from
the reference over a half-space, both components, rms over rms:

| half-width (km) | top / interface / $\varepsilon$ (km) | gravity: $\tau$ | $10\tau$ | $100\tau$ | none: $\tau$ | $10\tau$ | $100\tau$ |
|---|---|---|---|---|---|---|---|
| 400 | 10 / 15 / 1.5 | 0.065 | 0.104 | 0.178 | 0.121 | 0.116 | 0.199 |
| 800 | | 0.037 | 0.074 | 0.142 | 0.068 | 0.074 | 0.145 |
| 1600 | | 0.039 | 0.074 | 0.136 | 0.046 | 0.071 | 0.136 |
| 400 | 5 / 7.5 / 0.75 | 0.065 | 0.083 | 0.138 | 0.126 | 0.103 | 0.139 |
| 800 | | 0.026 | 0.041 | 0.092 | 0.074 | 0.056 | 0.085 |

The box matters at 400 km and no longer by 800. Two other things are in what
is left, and the gravity term is neither: the gate holds the same thrust
under ten times gravity to 5 % and 10 % where the reference without gravity
misses by 64 %.

*A plate that bends is only as good as its two faces.* On the first mesh the
trough at $100\tau$ under gravity is 0.87-0.89 of the reference's on all
three boxes. Followed down in $\tilde\mu$, the model's trough stops deepening where
the reference's goes on, so it is the plate afloat that differs, and it is
discretization. Afloat at $\tilde\mu/\mu=10^{-4}$, dense, a 300 km box,
$\nu=0.31$, trough over the reference's:

| top / interface / $\varepsilon$ / fault (km) | trough |
|---|---|
| 10 / 15 / 1.5 / 5 | 0.885 |
| 5 / 15 / 1.5 / 5 | 0.911 |
| 10 / 7.5 / 1.5 / 5 | 0.890 |
| 10 / 15 / 0.75 / 5 | 0.898 |
| 5 / 7.5 / 0.75 / 5 | 0.943 |
| 5 / 5 / 0.5 / 2.5 | 0.958 |
| P1 top at 10, the rest as the first row | 0.915 |

First order in the elements of each face and in the interface's
$\varepsilon$, each modest and all of one sign; a P1 top is worth a P0 top of
half the size. It is not the box, the walls, the fault's length, the mesh
beyond the near field or the compressed backend, each of which was varied.
The strike-slip plate never showed it because antiplane shear does not bend
the plate, and while the substrate still carries load, at
$\tilde\mu/\mu=0.03$, the first mesh is within 3 %. On the finer plate the
trough at $100\tau$ is 0.93-0.94 of the reference's.

*Two meshes cancel it.* An error that is first order in three lengths is
gone from $(r\,u_{\rm fine}-u_{\rm coarse})/(r-1)$ when the three are in one
ratio $r$ between the meshes (`layered_box.two_mesh_limit`). `refine` divides
exactly those three, so `--sweep refine=1,2` is such a pair, and the thrust
figure draws its limit. The 800 km box under gravity, misfit against the
reference:

| | 10 / 15 / 1.5 km | half of each | the limit of the two |
|---|---|---|---|
| vertical, postseismic: $\tau$ | 0.043 | 0.032 | 0.040 |
| $10\tau$ | 0.087 | 0.045 | 0.011 |
| $100\tau$ | 0.140 | 0.079 | 0.029 |
| horizontal, postseismic: $100\tau$ | 0.147 | 0.116 | 0.093 |
| trough at $100\tau$ over the reference's | 0.879 | 0.931 | 0.991 |
| at the earthquake, $u_x$ and $u_z$ (reference with the model's base) | 0.030, 0.039 | 0.016, 0.027 | 0.003, 0.013 |

It removes that error and no other. At $\tau$ it gains nothing, and the
horizontal at $100\tau$, which is the walls and not the faces, is what is
left: after the limit it is the largest error in the model. On the gate's
dense box, afloat, the trough is 17.6 % short on faces 1.5 times the gate's
and 11.8 % on the gate's, and the limit of the two is off by 4e-4 (1.4 % at
$\nu=1/4$). So higher order on the faces, which buys what halving them does,
is no longer the remedy: a second P0 mesh buys more.

*A relaxed plate slides, and the walls hold it.* The horizontal postseismic
motion is short by 8-12 % at $100\tau$ on the largest boxes. Over a relaxed
substrate the two halves of the plate approach from as far as the
substrate's remaining shear reaches, which late on is farther than any box
here, and clamped sides stop them. Free sides give the amplitude back (0.98
of the reference against 0.93 at $\tilde\mu/\mu=0.003$ on the 300 km box)
and float the plate (section 11). Section 8 gives the walls what they lack.

*The fault is 800 km long, not infinitely.* Every number above is against
plane strain. Against the reference in three dimensions (section 8), for the
fault as long as it is, the limit of the two meshes is off by 4.0 % / 2.5 % /
3.7 % at $\tau$ / $10\tau$ / $100\tau$ (plane strain: 2.8 / 2.0 / 5.7), the
vertical by 1.8 % at $100\tau$ and the horizontal by 6.2 %, not 9.3: within
120 km of an 800 km fault the two references differ by a few percent, and
a third of what read as the walls was that.

The 800 km box on the finer plate is 205,026 unknowns, 14 states in 19
minutes, 42-48 iterations a state under gravity and up to 98 without; the
coarser one of the pair is 62,622 unknowns and 7 minutes.

*The earthquake cycle.* A fault that is locked between earthquakes repeating
every $T$ is the same fault slipping steadily at $V=b/T$, less the
difference, and the difference is periodic. With $u(t)$ the response to a
unit step of slip, the velocity measured from steady slip, per unit $V$, is

$$v(t)=T\sum_{n\ge0}\dot u(t+nT)-u(\infty),\qquad 0<t<T$$

(`Prony.locked_rate_weights`; the sum is geometric for each exponential of
the fit, so it costs nothing). Three things follow. Its mean over the cycle
is $-u(0^+)$: elastic back slip (Savage 1983) is the cycle's AVERAGE
whatever relaxes, and relaxation only moves velocity between early and late.
It does not need the body to have settled, because what the response does
long after a period enters the sum and the limit alike and all but cancels;
the window must hold a few periods, not the relaxed state. And it has a
reference that needs no fit at all: the train is periodic, so its response
is a Fourier series whose coefficient at $s_n=2\pi i n/T$ is the solution at
the moduli of $s_n$,

$$v(t)=-u(0^+)+2\,\mathrm{Re}\sum_{n\ge1}\left[\tilde u(s_n)-u(0^+)\right]e^{s_nt}.$$

The fitted velocity of the reference agrees with that series to 4e-4 of its
peak at $T=20\tau$ under gravity (window 0.1 to $100\tau$; 1.5e-4 at
$T=5\tau$, 7e-4 at $50\tau$), and to 1 % without gravity on the same
window, 0.2 % with the window to $1000\tau$.

For the thrust above at $T=20\tau$ (`plate_cycle`, drawn by any dip-slip
run), 97 km into the hanging wall the horizontal velocity is $-0.14\,V$ at
$t=0.05\,T$ and $+0.21\,V$ at $0.95\,T$, about a mean of $+0.11\,V$: just
after an earthquake the far hanging wall still moves the way the earthquake
moved it, and it turns a sixth of the way through the cycle. Over the
fault's lower edge the surface sinks at $0.38\,V$ early and rises at
$0.27\,V$ late. The model's velocity (the two-mesh limit, both components,
$|x|<120$ km) is within 2.3 % of the reference early and 2.1 % late, against
4.6 % and 4.5 % on the coarser mesh alone (4.0 % and 2.0 % against the
fault as long as it is, section 8). Gravity changes none of this by
more than 0.02 $V$ at this period, because a cycle of $20\tau$ reads the
response to a few tens of $\tau$, where gravity has done little (the table
above); it is the long cycles it bounds.

**A megathrust: the fault through the interface** (`--set dip=20 --set
depth=45`). A fault deeper than the plate is thick goes through the
interface, and what lies below is a second fault, `fault_lower`, on the same
plane and with the same slip, in the SUBSTRATE: slip inside the region that
relaxes. The interface is aligned on the line where the fault cuts it, as the
top is on the trace, because the interface's displacement jumps there by the
slip. In the reference it is two sources, each in the moduli of its own
layer, the lower one entering on the substrate's side of the interface
conditions, and three limits that are exact hold that lower source: an
interface between like materials is none (7e-15); under a plate 10 m thick
the fault lies in a half-space of the substrate, whose closed form knows no
moduli (5e-4, first order in the plate's thickness); and in a fluid
substrate it moves nothing at the surface (1e-6).

The last is also the physics. Take a megathrust dipping 20 degrees to 45 km
through the 30 km plate: 132 km down dip, the lowest 44 of them in the
substrate. At the earthquake that lower reach is two thirds of the vertical
motion at the surface (rms) and a quarter of the horizontal; by $100\tau$ it
is under 1 % of either. Slip inside a body that relaxes is forgotten. What
is left is a plate cut through, and once the substrate has let go its two
halves have simply converged, by $\cos 20^\circ=0.94$ of the slip. So the
far field catches up with the trench, in the reference (plane strain: a
megathrust that is infinitely long):

| over the slip | $0^+$ | $\tau$ | $10\tau$ | $100\tau$ | $250\tau$ |
|---|---|---|---|---|---|
| $u_x$, 200 km into the hanging wall | -0.26 | -0.30 | -0.41 | -0.46 | -0.47 |
| $u_x$, 400 km | -0.09 | -0.13 | -0.29 | -0.45 | -0.46 |
| uplift at the trench | 0.30 | 0.29 | 0.27 | 0.20 | 0.20 |

Its cycle is the one subduction zones show (Thatcher & Rundle 1984; Wang,
Hu & He 2012). Repeating every $50\tau$, the
mean velocity 200 km into the hanging wall is $+0.26\,V$, landward: back
slip. At $t=0.05\,T$ it is $-0.56\,V$, at 300 km $-0.87\,V$: everything
past 151 km moves toward the trench, as the earthquake moved it, and at 300
km twice as fast as it will later move back. At $0.95\,T$ it is $+0.45\,V$
at 200 km and $+0.39\,V$ at 400 km, the whole hanging wall moving landward
together. Gravity leaves those horizontal rates alone (0.02 $V$) and changes
the vertical ones by a factor of two and more: over the interface cut the
surface sinks late in the cycle at $0.07\,V$ with gravity and $0.13\,V$
without.

*The model against it* (800 km box, near field 200 km, a two-mesh pair with
clamped sides and the coarser mesh again with free ones; $|x|<200$ km at
mid-strike). The model's fault is 800 km long, and for a megathrust that is
not long enough to be read against plane strain: in a half-space its
mid-strike uplift differs from the infinitely long fault's by 0.015 of the
slip, 11 % of itself, a difference that falls only as one over the length,
and relaxed the two differ by 5-10 % in $u_x$ (section 8). So the reference here is the one
in three dimensions, for the fault as long as it is.

At the earthquake the model is within 0.2 % in $u_x$ and 0.8 % in $u_z$ of
it with a rigid base where the model's is. Against a half-space the vertical
is off by 4.9 %, and that is one number: the surface stands 0.006 of the
slip higher everywhere, for a base 800 km down. A source 132 km wide feels
the bottom of this box where the thrust did not.

Relaxation is where the box takes over. Misfit of the postseismic
displacement against the half-space reference, and how far the two sides of
the trench have converged over how far the reference's have:

| | $\tau$ | $10\tau$ | $100\tau$ | $250\tau$ |
|---|---|---|---|---|
| clamped sides, 10 / 15 / 1.5 km | 0.118 | 0.112 | 0.142 | 0.142 |
| half of each | 0.096 | 0.101 | 0.129 | 0.130 |
| the limit of the two | 0.078 | 0.093 | 0.124 | 0.124 |
| free sides, 10 / 15 / 1.5 km | 0.149 | 0.119 | 0.151 | 0.156 |
| convergence, clamped | 0.96 | 0.90 | 0.81 | 0.81 |
| convergence, free | 1.04 | 1.07 | 1.11 | 1.11 |

The two-mesh limit mends the vertical (8 % at $10\tau$ and after, from 19 %
and 13 % on the two meshes) and leaves the horizontal at 13 %: the plate is
cut through, its halves converge as rigid blocks once the substrate lets go,
and clamped walls 800 km away hold them at 0.81 of the way. Free sides let
them go too far, 1.11, and float them. So the two side conditions bracket
the reference from either side at every time, and neither is what the plate
beyond a wall does. (Read against plane strain these runs gave 11-22 % and
a convergence of 0.69 clamped, 0.95 free: a third of that was the
reference, an infinitely long fault standing for one 800 km long.)

The cycle at $T=50\tau$ inherits it: velocity from steady slip within 7 %
of the reference early and 6 % late with clamped sides, 10 % and 8 % with
free ones. At 191 km the reference moves at $-0.48\,V$ early and
$+0.42\,V$ late; clamped gives $-0.37$ and $+0.37$, free $-0.56$ and
$+0.48$. Section 8 takes the walls out of all of it.

The pair is 105,246 and 373,224 unknowns, 16 states each, in 12 and 58
minutes on a shared machine, 38-51 iterations a state (37-82 with free
sides, which condition worse as the plate floats).

## 8. The walls: a box as a window

A region must close, so the plate over a half-space of section 5 is a plate
over a deep box, and the box has walls. Section 7 ended on them: a relaxed
plate slides from farther than any box, clamped walls hold it, free ones let
it go too far, and the two bracket the reference. Two remedies were measured
where the problem is smallest and has an exact answer, the elastic fault box
against the half-space (`cutde`): surface displacement beyond 128 km of a
strike-slip fault, in a box 200 km to its walls and 100 km deep.

| walls | misfit |
|---|---|
| free sides over a clamped base (the box as it was) | 0.22 |
| clamped sides and base | 0.49 |
| springs, $\mathbf t=-k\mathbf u$ with $k=\alpha\mu/L$, at the best $\alpha$ | 0.15 |
| the half-space's own displacement of the same fault | 0.010 |

A spring at its best stiffness halves the error of the middle distances
(8 % to 4 % between 64 and 128 km) and leaves the rim; the best stiffness is
fitted (between $\alpha=2$ and 4) and differs with the source (an opening
fault's best leaves 0.25); and over a substrate that relaxes, what a plate
beyond a wall does is not one stiffness but stiff early and slack late. The
third row needs no fit. Tell the walls what the model's own sources do in
the medium the box was cut from.

**The window** (`model.FarField`). Named walls are prescribed-displacement
patches whose datum is $\mathbf u_{\rm ref}(\mathbf x)$, the displacement of
the model's prescribed slip in an unbounded reference medium that has a
solution of its own: here a plate over a half-space. The box then solves
only what the reference does not know, its interior, whatever is in it, and
the reference carries the far field. The datum depends on the materials, so
every state has its own, and three things in the code follow from that:

- the model holds it, not the patch: `RegionModel.value_of` is the one place
  a datum is read, by the equations and by the evaluators alike, so a state
  is solved and read back with the same walls;
- `with_materials` carries it to the state it builds, so the rule of section
  5 stands as it was, a state is the base model with its own materials;
- the operator does not see it. Walls are a right-hand side
  (`equations.rhs_terms_at`), so `rebuild_for_materials` solves each state
  with its own on the assembly it already has, on both backends. The runner
  asks the reference for every state's walls in one call.

It is exact where the reference is: for the sources it solves, in the medium
it solves. What the walls miss is the far field of anything else, a slip
that is solved and not prescribed (section 9), a body that is not layered
(section 10). That part falls off with distance, so the size of the box
comes back as a parameter, for it alone.

**The reference in three dimensions** (`cases.layered_space`) is the
plane-strain reference of section 7 turned about the vertical: a RECTANGULAR
fault, from the surface down, in the plate over the substrate, under the
same buoyancy, at any point of either layer. In the frame of a horizontal
wavevector ($\xi$ along it, $t$ across it) elasticity splits into the P-SV
system of section 7 at $|k|$, unchanged, and an SH one for the transverse
displacement, which buoyancy does not enter. Slip is an eigenstrain
$\mathbf E={\rm sym}(\mathbf b\otimes\mathbf n)$, and at a source depth it
is a jump

$$[u_\xi]=2E_{\xi z},\quad
[u_z]=E_{zz}+\frac{\lambda\,(E_{\xi\xi}+E_{tt})}{\lambda+2\mu},\quad
[\tau_{\xi z}]=ik\,\frac{4\mu(\lambda+\mu)E_{\xi\xi}+2\mu\lambda E_{tt}}
{\lambda+2\mu},\qquad
[u_t]=2E_{tz},\quad[\tau_{tz}]=2ik\mu E_{\xi t}$$

of which the plane-strain source is the case $E_{tt}=0$. The fault is
integrated along strike in closed form and down dip by Gauss points, cut at
any level it straddles; the transform back is a quadrature in polar
wavenumbers, geometric from tiny $k$, because a relaxed plate's far field
decays like $1/r$ and lives there. It has two forms. At the SURFACE, near
the fault as well as far, it is Okada's closed form for the half-space plus
the transform of the smooth difference from it, as in section 7. At DEPTH,
which is what a wall asks for, the spectrum is tapered, which is a smoothing
over a length the caller's distance from the fault sets; its error is fourth
order in that ratio, 1e-4 of the field at the default, and it is not for
points near the fault.

| anchor | error |
|---|---|
| Okada's rectangle against `cutde`, every dip and rake | 7e-11 |
| the transform on the sides and base of an 800 km box, like materials, against `cutde` (to 800 km depth; the megathrust through an interface between like materials included) | 7e-5 |
| an infinitely long thrust and megathrust against the plane-strain reference, gravity, $\mu_s/\mu$ to 0.003, with and without a base | 4e-5 far away, 1.1e-4 of the slip next to the fault |
| an infinitely long strike-slip fault against the antiplane series | 6e-5 |
| a finite megathrust at mid-strike as it lengthens, against plane strain: half-length 2000 / 4000 / 8000 / 16000 km | 2.6e-2 / 5e-3 / 2.4e-4 / 7e-6 |
| a finite layered fault: its two forms against each other far away | 1e-5 |
| continuity at the interface; zero on a clamped base; a stack against one at a time | 3e-11; rounding; exact |

The walls and base of the 800 km megathrust box are 1,686 points, and
sixteen states of them cost 7 s.

**A fault is as long as it is.** The fifth row is the one that changed what
was known. A megathrust 800 km long is not infinitely long where sections 5
to 7 took it to be: in a half-space, by the two closed forms, its
mid-strike uplift within 200 km differs from the infinitely long fault's by
0.015 of the slip, 11 % of itself (at most 0.029 / 0.016 / 0.008 / 0.004 of
the slip at half-lengths of 200 / 400 / 800 / 1600 km: one over the
length), and over a relaxed substrate the two postseismic displacements
differ by 5-10 % in $u_x$ and more in $u_z$. For
the thrust, read within 120 km, it is a few percent. So section 7 gives its
megathrust numbers against this reference, and a study with a window among
its runs is compared with it throughout (`figures/plate.py`).

**The megathrust in a window** (`--set sides=window`).

The megathrust of section 7 again, the same box and the same mesh, every
wall a window. Against the reference in three dimensions:

| 800 km box, fault 800 km long, faces 10 / 15 / 1.5 km | clamped | free | window |
|---|---|---|---|
| at the earthquake, $u_x$ and $u_z$ | 0.009, 0.039 | 0.022, 0.035 | 0.008, 0.017 |
| postseismic $u_x$ at $\tau$ / $10\tau$ / $100\tau$ / $250\tau$ | 0.065 / 0.102 / 0.140 / 0.141 | 0.065 / 0.088 / 0.154 / 0.160 | 0.021 / 0.017 / 0.023 / 0.023 |
| postseismic $u_z$ | 0.41 / 0.19 / 0.15 / 0.15 | 0.55 / 0.29 / 0.13 / 0.13 | 0.22 / 0.16 / 0.15 / 0.15 |
| the two sides' convergence over the reference's | 0.94 / 0.90 / 0.81 / 0.81 | 1.04 / 1.07 / 1.11 / 1.11 | 0.98 / 0.99 / 0.99 / 0.99 |
| the cycle at $T=50\tau$, early / late | 0.086 / 0.064 | 0.098 / 0.083 | 0.039 / 0.025 |

The horizontal was the walls, and it is mended: 2 % where clamped walls left
14 % and free ones 16 %, the plate's two halves converging as the
reference's do. In the cycle a point 191 km into the hanging wall moves at
$-0.45\,V$ early and $+0.42\,V$ late, against the reference's $-0.48$ and
$+0.42$ and the clamped box's $-0.36$ and $+0.37$. The base no longer lifts
the surface (1.7 % in $u_z$ at the earthquake, from 3.9 %). The walls of all
sixteen states are 14 s of the run.

The vertical was never the walls: late on it is 15 % of the postseismic
part with all three. It is the plate's faces on this one mesh (section 7).
A second mesh takes it out, as it did for the thrust. A rupture 300 km long
in a window 400 km to its walls, as a pair (`--sweep refine=1,2`):

| against the reference in three dimensions | 10 / 15 / 1.5 km | half of each | the limit of the two |
|---|---|---|---|
| at the earthquake, $u_x$ and $u_z$ | 0.007, 0.019 | 0.004, 0.009 | 0.001, 0.004 |
| postseismic, both components, at $\tau$ / $10\tau$ / $100\tau$ | 0.055 / 0.051 / 0.086 | 0.026 / 0.026 / 0.043 | 0.007 / 0.010 / 0.016 |
| ... $u_x$ | 0.020 / 0.014 / 0.026 | 0.010 / 0.008 / 0.014 | 0.003 / 0.004 / 0.005 |
| ... $u_z$ | 0.199 / 0.163 / 0.172 | 0.095 / 0.083 / 0.085 | 0.026 / 0.031 / 0.032 |
| the cycle at $T=20\tau$, early / late | 0.036 / 0.018 | 0.017 / 0.010 | 0.005 / 0.004 |

Every row halves with the faces, so the limit is what a first-order error
leaves: a megathrust through the interface within 1.6 % of the reference at
every time and its cycle within 0.5 %, where section 7's, between clamped
walls, was at 12 % and 7 %. The pair is 62,544 and 219,315 unknowns, fourteen states each, in
13 and 39 minutes with three studies sharing the machine; the walls are 7 s
and 41 s of that.

*A window can be small.* A rupture 300 km long in a box 400 km to its
walls, the same faces: clamped, the postseismic $u_x$ is off by 20 % / 23 %
/ 23 % at $\tau$ / $10\tau$ / $100\tau$, the two sides converge 0.78 / 0.67
/ 0.61 of the reference's, and in a cycle of $20\tau$ the point 191 km into
the hanging wall moves the wrong way early ($+0.11\,V$ for $-0.06$). As a
window: 2.0 % / 1.4 % / 2.6 %, 0.98 / 0.99 / 0.99, and $-0.05\,V$.
With clamped walls twice as far away the same rupture is off by 3.4 % /
5.0 % / 7.9 % and its sides converge 0.96 / 0.93 / 0.88: a rupture of finite
length does let go of its walls as they recede, slowly. In a window the box
is not in the answer: 1.9 % / 1.6 % / 2.8 % at 800 km for 2.0 % / 1.4 % /
2.6 % at 400, and with the vertical in, 5.3 / 5.1 / 8.5 % for 5.5 / 5.1 /
8.6 %.

**The elastic box** has the same option (`fault_box.build_model(walls=
"window")`, `configs/fault_box.py --set walls=window`): its sides and base
on a uniform half-space, which is `layered_space` with nothing layered. The
case's own window agrees with the one made of `cutde` to 8e-4 in the
solution, and on the gate's box (120 km deep) takes the surface beyond 128
km from 30 % off to 1.5 %. The default is unchanged, because every
committed baseline was measured on the box.

## 9. A fault that creeps

Until here slip was prescribed, or solved from a traction it must produce
(`BCType.FAULT_TRACTION`). Let a solved fault resist its slip RATE instead,
the side the normal points into dragging the other with a shear traction
$\eta_f\,\dot{\mathbf b}$. With nothing slipped before the step,
$\tilde{\boldsymbol\tau}=\eta_f s\,\tilde{\mathbf b}$, and for step data
(section 1) the state at $s$ has, on that fault, the rows

$$\mathbf n\cdot\boldsymbol\sigma(\mathbf x_c)-\kappa(s)\,\mathbf b_c
=\Delta\mathbf t,\qquad\kappa(s)=\eta_f s:$$

the elastic traction of every source, as before, less a stiffness times the
fault's own slip. That is an elastic problem like every other state, so a
linear viscous fault needs no time-stepper, and neither does friction that
strengthens with rate, linearized about a steady rate $V$, which is the same
law with $\eta_f=(a-b)\sigma_n/V$. At $s=\infty$ the drag is infinite and
the fault is locked: it does not slip at the instant of loading. As $s\to0$
it vanishes and the fault ends where it carries no traction. In between,
its slip is whatever the rest of the model loads it with: afterslip.

`rheology.ViscousFault(tau, width)` states it as the shear zone it stands
for, `width` thick, of a material whose Maxwell time is `tau` at the
unrelaxed modulus of the region holding the fault: $\eta_f=\mu\tau/w$, a
time and a ratio like every rheology here. A patch of size $L$ unloads in
about $\tau L/w$. `Laplace.faults` gives a solved fault its law; every state
then carries the fault's $\kappa$ (`State.drag`), which the dense backend
hands to `equations.apply_fault_rows`, where each enforced row is divided by
$1+\kappa/\kappa_0$, $\kappa_0$ the fault's own stiffness, so that it stays
bounded and becomes $\mathbf b=0$ at infinity.

**The anchor is exact.** A penny crack under a shear drop $\Delta\tau$ whose
faces drag with $\kappa_0/\sqrt{1-r^2/a^2}$ keeps Eshelby's elliptical slip,
because that slip's traction is uniform: the peak is
$\Delta\tau/(k_E+\kappa_0)$, $k_E$ the crack's own stiffness. In an elastic
body $\kappa_0=\eta_0 s$ makes it one exponential with time $\eta_0/k_E$;
in a Maxwell body $k_E$ relaxes too and the fault's creep and the body's
are one transform. Moment over the closed form, continuous P2 at $a/h=8$:
0.9943-0.9944 per $s$ and 0.9932-0.9954 in time in the elastic body,
0.9943-0.9949 and 0.9941-0.9949 in the Maxwell one, where the reference in
time is a contour inversion. The 0.6 % is the crack's own (section 5);
against the same crack without drag the drag is right to 2e-4.

**Afterslip below a megathrust** (`--set lower=creep`). The megathrust of
section 7 with the rupture in the plate and its lower reach, in the
substrate, a fault that creeps: locked at the earthquake, then loaded by the
rupture above it and by the substrate as it relaxes.

*The study* (`--sweep creep_tau=1,1e6 --sweep refine=1,1.5`): a rupture 200
km long in a window 300 km to its walls and 200 km deep. The rupture is the
plate's part of the fault, to 30 km; its lower reach, to 45 km, creeps with
$\tau_c=\tau$ over a zone 10 km wide, or is held ($\tau_c=10^6\tau$, a reach
no window sees slip). Dense: 19,338 unknowns on faces of 15 / 22.5 / 2.25 km
and 31,179 on 10 / 15 / 1.5, fourteen states in 5 and 16 minutes on a shared
machine.

Held, the model is the reference's: within 1.1 % and 1.4 % ($u_x$, $u_z$) at
the earthquake and, in $u_x$, within 1.4 % / 1.0 % / 1.4 % of the
postseismic part at $\tau$ / $10\tau$ / $100\tau$, in a box 300 km to its
walls. Its vertical is the plate's faces, 14 % / 17 % / 23 % of the
postseismic part on the finer mesh and 22-30 % on the coarser.

Let go, the reach slips by 0.22 of the rupture's slip on average once it has
unloaded: 0.04 at $0.3\tau$, 0.10 at $\tau$, 0.17 at $3\tau$, 0.21 at
$10\tau$, and no more after. It is 0.45 just under the rupture and 0.08 at
its lower edge, and its moment is 11 % of the rupture's. On the coarser mesh
the mean is 0.20 and the peak 0.38.

At the surface it is an EARLY signal. At $\tau$ the postseismic displacement
departs from the reference, which knows no afterslip, by 34 % of it (18 %
in $u_x$), where the held model departs by 4 %; at $10\tau$ by 11 % against
5 %; and at $100\tau$ the two are one. In the cycle (every $20\tau$) a point
148 km into the hanging wall moves toward the trench at $0.41\,V$ just after
an earthquake with the creep and $0.29\,V$ without it, and away from it at
$0.31\,V$ late, either way. Slip in a body that relaxes is forgotten
(section 7), and a reach that creeps in the substrate is such slip: it
reshapes the first Maxwell times and leaves the rest alone.

Three limits. A solved fault is on the dense backend, so these models are
tens of thousands of unknowns, their plates' faces are coarse (section 7),
and a pair of them is too close in size for a two-mesh limit: meshes in the
ratio $r$ magnify what they do not share by $(r+1)/(r-1)$, five here. The
window's walls carry the far field of the prescribed slip and not of the
creep, whose moment is a tenth of the rupture's. And uniform slip ends in a
line where its traction is unbounded: the creeping reach begins on that
line, so its slip against it is set by the mollification width (0.38 and
0.45 on the two meshes), and it is the integral of the afterslip that
settles (0.20, 0.22), not its peak.

## 10. A plate that subducts

Layers are what the reference of section 8 knows, and a boundary method is
for what it does not. `layered_box` with `slab_depth` lets the footwall
plate go on down the dip as an elastic SLAB, in the substrate that relaxes
around it. Plate and slab are one region, so the megathrust is the plate's
own fault, down to the plate's base at the deepest; below that the plane is
the slab's upper face, an interface and not a fault. The interface wraps the
slab: flat, less the footprint under the megathrust, plus the slab's top,
its bottom (the parallel plane through the point under the trench at the
plate's base), a horizontal tip and two vertical caps. The pieces do not
share nodes and need not: elements are discontinuous and a surface closes
by its geometry (its area is exact to rounding).

Two things it rests on. A slab of the substrate's own material is no slab,
and the solved surface says so to 1.1 % (`verify_viscoelastic` [p]), where
over a soft substrate the slab changes it by 21 %. And a slab of a plate
lighter than the mantle is a light body under a heavy one, which the
Laplace route refuses once anything relaxes (section 7): the study gives
plate and mantle one density, as the base of the lithosphere has nearly
none to offer, and keeps the free surface's.

**The study** (`--sweep slab_depth=0,150`): a megathrust dipping 20 degrees
to the base of a 30 km plate, 200 km long, with and without a slab that
reaches 150 km of depth, 412 km from the trench, in a window 500 km to its
walls; 38,130 and 63,240 unknowns, fourteen states. At the earthquake the
slab is invisible, plate and mantle being one elastic material until
something relaxes, and the surface says so at production scale: within 0.8 %
and 1.4 % of the layered reference in $u_x$ and $u_z$ with the slab, 1.0 %
and 1.3 % without. After it, the slab is the largest thing in the model:

| over the slip | $0^+$ | $\tau$ | $10\tau$ | $100\tau$ |
|---|---|---|---|---|
| uplift 27 km behind the trench, no slab | 0.29 | 0.28 | 0.21 | 0.14 |
| ... with the slab | 0.29 | 0.29 | 0.30 | 0.25 |
| over the megathrust's lower edge (99 km), no slab | -0.13 | -0.10 | -0.02 | -0.01 |
| ... with the slab | -0.12 | -0.15 | -0.20 | -0.08 |
| $u_x$ 28 km seaward of the trench, no slab | 0.15 | 0.20 | 0.36 | 0.40 |
| ... with the slab | 0.15 | 0.18 | 0.27 | 0.34 |

Without a slab the plate lets go of its bend as the mantle relaxes: the
uplift halves and the trough fills. With one, the uplift stays and the
trough DEEPENS for ten Maxwell times before it recovers: the slab is an
elastic strut through the mantle, it goes on carrying what the mantle
sheds, and the mantle wedge above it relaxes against a face that does not.
The postseismic displacement of the two models differs by 61 % of the
slabless one at $\tau$, 70 % at $10\tau$ and 43 % at $100\tau$ (the
horizontal by 36 / 43 / 26 %, the vertical by more than itself until late),
where the slabless model is within 3-10 % of the reference. In the cycle
(every $20\tau$) the surface over the megathrust's lower edge, 82 km from
the trench, rises at $0.41\,V$ just after an earthquake without the slab
(the reference: 0.39) and sinks at $0.18\,V$ with it. A layered reference
is not a stand-in for a subduction zone.

*The walls do not know the slab.* They carry the slabless far field, and
the slab's tip is 88 km from one of them. The same model in a window 800 km
to its walls (67,182 unknowns) differs from this one by 0.1 % at the
earthquake and, in the postseismic part, by 1.5 % at $\tau$, 2.7 % at
$10\tau$ and 7 % at $100\tau$: the horizontal by 2 % or less at every time,
the vertical by 3 %, 5 % and 31 %. So late on, a third of the slab model's
vertical change, which is small, belongs to the box (12 % of the slab's
effect there). What is said above stands; a number for the vertical at
$100\tau$ does not, until the walls know the slab or the box is laddered
further. (The two boxes also differ in their far mesh, 83 and 133 km.)

## 11. Open

- **The box is a window only on what the reference solves.** Its walls
  carry the far field of prescribed slip in a plate over a half-space
  (section 8). A slip that is solved (section 9) and a body that is not
  layered (section 10) radiate a part the walls do not know; the size of
  the box is the parameter for that part, and it has not been laddered.
  Folding a solved slip's far field into the solve is a rank-few coupling of
  the walls to the fault's mean slip; not built.
- **A plate that bends costs two meshes** (section 7), and on the dense
  backend, where a creeping fault must be, the second mesh is small. The
  compressed backend solves no slip.
- **Time-stepping with memory on the boundary** (section 6): what nonlinear
  fault behaviour needs, friction that is not linearized. Not built.
- **A slab is elastic and welded.** Slip on the slab's face below the plate
  is slip ON AN INTERFACE, a jump of displacement between two regions; the
  rows are a prescribed displacement's and are not written. Nor is a slab
  denser than the mantle around it, which wants the stratification rule of
  section 7 read face by face.
- **Uniform slip on a plane.** Slip that tapers, a fault that curves, a
  rupture that stops short of the trench (a difference of two faults, by
  linearity): none attempted. The reference of section 8 is one rectangle
  from the surface.
- **A plate with free sides floats** (below), and a region cannot run to
  infinity. Neither matters to a window, and both remain for a box that is
  not one.
- Gravity beyond section 7: the coupling of gravity with COMPRESSION, of
  order $wL/K$, is a term in the volume; so is self-gravitation.
- The cycle of section 7 is a locked patch measured from steady slip on the
  same patch. With section 9 the reach below it may creep, at a rate that
  varies through the cycle; what steady slip itself builds ($u(\infty)/T$ a
  year of relief) is still the term it is measured from.
- Nonlinear rheology is out of scope: it needs the viscous strain in the
  volume.

*The floating plate.* As $\tilde\mu(s)\to0$ only the shear of the substrate
holds a plate with free sides against a rigid horizontal translation. A
dislocation is self-equilibrated, but its discrete net force is small rather
than zero, and the translation it causes is that force over $\tilde\mu(s)$.
On the split fault box of B6 with a relaxing lower region, the net
translation of the free surface (zero by symmetry) is 1 % of the deformation
at $\tilde\mu/\mu=10^{-2}$, 11 % at $10^{-3}$ and as large as the
deformation at $10^{-4}$, and the condition number grows as
$\tilde\mu^{-2}$. It is the same with one $\varepsilon$ or per-element
$\varepsilon$. Clamping the plate's sides removes it (0.3 % at $10^{-4}$,
condition number as $\tilde\mu^{-1}$); `deflate` does not, because the mode
is not a translation of every unknown. A window removes it too: its walls
are prescribed.

## References

Haskell (1935) Physics 6. Rizzo & Shippy (1971) SIAM J. Appl. Math. Freund &
Barnett (1976) Bull. Seismol. Soc. Am. 66, 667 (dip slip in a half-space,
plane strain). Okada (1985) Bull. Seismol. Soc. Am. 75, 1135 (a rectangular
fault in a half-space, at the surface). Marone, Scholz & Bilham (1991) J.
Geophys. Res. 96, 8441 and Perfettini & Avouac (2004) J. Geophys. Res. 109,
B02304 (afterslip as rate-strengthening creep). Kusama & Mitsui (1982) Appl.
Math. Model. Sim & Kwak (1988) Comput. Struct. Schanz & Antes (1997)
Comput. Mech. Gaul & Schanz (1999) Comput. Methods Appl. Mech. Eng. 179.
Pan, Sassolas, Amadei & Pfeffer (1997) Comput. Mech. 19. Syngellakis & Wu
(2004) Eng. Anal. Bound. Elem. 28. Abate & Valko (2004) Int. J. Numer.
Methods Eng. 60, 979 (the fixed Talbot contour). Mesquita & Coda (2007) Appl. Math. Model.,
"A boundary element methodology for viscoelastic analysis: Part II without
cells". Panagiotopoulos, Mantic & Roubicek (2014) arXiv:1402.6631. Hetland &
Hager (2005) J. Geophys. Res. 110. Fukahata & Matsu'ura (2006) Geophys. J.
Int. 166. Savage (1983) J. Geophys. Res. 88, 4984 (back slip). Thatcher &
Rundle (1984) J. Geophys. Res. 89, 7631 (the cycle of repeated earthquakes
at a subduction zone, a plate over a viscoelastic half-space). Wang, Hu & He
(2012) Nature 484, 327. Wang (2005) Geophys. J. Int. 161, 191. Wang, Lorenzo-Martin & Roth (2006)
Comput. Geosci. 32. Barbot & Fialko (2010) Geophys. J. Int. 182, 568 (the
half-space under surface buoyancy) and 1124. Love (1927) A Treatise on the
Mathematical Theory of Elasticity (the sphere). Barbot, Moore
& Lambert (2017) Bull. Seismol. Soc. Am. Miyake & Noda (2019) Earth Planets
Space 71. Mallick, Lambert & Meade (2022) J. Geophys. Res. 127. Loveless &
Meade (2025) Earth Space Sci.
