# mbem — mollified full-space boundary element method

A clean, self-contained implementation of a **mollified boundary element method
(BEM)** for 3-D linear elasticity, in the full space.  Following Cortez, the
regularization is applied to the SOURCE, not to the kernel: the point force is
replaced by a smooth blob `phi_eps` of unit integral and the elastostatic
equations are then solved exactly, giving

```
G_ij = C1 [ (3-4nu) delta_ij / R_eps  +  d_i d_j / R_eps^3
                                      +  2(1-nu) eps^2 delta_ij / R_eps^3 ]

  R_eps = sqrt(r^2 + eps^2),  C1 = 1/(16 pi mu (1-nu)),
  L_ik G_kj = -delta_ij phi_eps,  phi_eps = 15 eps^4 / (8 pi R_eps^7).
```

The first two terms are the classical Kelvin solution carrying `R_eps` in place
of `r`.  **The third has no singular counterpart** and is what the blob
convolution contributes: without it the regularized Cauchy-Navier residual is
~1e-2, with it ~1e-16 at every Poisson ratio
(`tests/gates/moss_kernel/verify_pde_residual.py`, symbolically).  Simply
substituting `r -> R_eps` gives a smooth function that solves nothing, and it is
the PDE, not the smoothness, that licenses integrating over the element the
observation point lies on.

This removes the singular surface integrals and lets the mollification width
`eps` and the mesh size `h` be chosen independently.  Each triangle's kernel
contribution is integrated **analytically**, so (following Ferranti & Cortez)
`eps` is decoupled from `h` — there is no quadrature error tying them together.

This package contains only the elastic full-space machinery: the point-source
kernel, the triangular displacement-discontinuity solution, the `mbem` BEM
solver (region model + calibrated dense backend), and worked examples.  There is
**no half-space / Mindlin code and no LaTeX** here.  Linear viscoelasticity
(Maxwell, Burgers) is available by the Laplace route, and gravity as buoyancy
on the boundary, both with no volume mesh; see "Viscoelastic relaxation" and
"Gravity" below.  A box can be a WINDOW on a layered half-space, its walls
carrying the far field ("The box as a window" below).

Boundary patches and faults carry a piecewise-constant (P0) density by default
or a Lagrange P1/P2 nodal density (`Patch(..., order=1|2)`): one code path,
collocation at the shrunk element nodes, the free term a shape-function matrix,
kernels integrated in closed form for every order (`mbem/kernels/tri_nodal.py`,
gated against the frozen `src/clq` oracle).  Higher order pays on traction
(Neumann) rows and is the cure for the first element row of on-fault stress
below a free surface (beyond ~3-5 eps_top; `tests/gates/mbem/verify_solved_bvp.py` A4);
the compressed backend is P0-only.

## The anelastic (eigenstrain) term

A fault's `Patch.value` is the Burgers vector `b = u(+n face) - u(-n face)` for
stored normal `n`, the same sign clq and cutde use (`mbem/selfcheck.py` pins it
at runtime).

A fault slip is an **anelastic** (inelastic / eigen-) strain.  Stress read off a
mollified slip source is therefore the *total* stress `C:eps_total =
C:eps_elastic + C:eps_star` inside the ~`eps` fault zone; on the fault it is
dominated by the eigenstress, peaking at `(3/4) mu s / eps` and diverging as
`eps -> 0`.  To recover the genuine **elastic** stress you must subtract the
anelastic term:

```
sigma_elastic = sigma_total - C:eps_star,
```

The live implementation is the **exact finite-triangle** form in
`mbem/kernels/tri_kernels.py::eigenstress_contract`, applied by
`mbem.evaluate_stress(subtract_anelastic=True)` (the default); `anelastic.py`
(`eigenstress_at_points`) is the frozen infinite-plane / nearest-triangle
*approximation*, right deep inside a large element and up to 2x too large at
element edges; the demos use the exact form through
`mbem.evaluate._stress_from_source(..., "eigen", ...)`.
`python -m mbem figure anelastic_subtraction` shows that the
corrected on-fault stress stays finite (bounded, `eps`-independent) while the
raw value blows up like `1/eps`.

The same eigenstress lives in **every** mollified double layer, not only faults:
a boundary patch's `u_p` is a jump between the field inside the region and zero
outside, and its smeared eigenstress is non-physical inside the body.  Since
2026-09-19 `evaluate_stress` subtracts it for boundary patches too — that term,
not a mesh limit, was the interior stress error near boundaries that did not
improve under refinement (`tests/gates/mbem/verify_boundary_eigenstress.py`).

The subtracted on-fault elastic stress not only stays finite — at the fault
interior it **converges to a constant** as `eps -> 0` (observed order `eps^2`),
and to the *physically correct* value: `python -m mbem figure onfault_convergence`
matches it to the finite part of the classical triangular-dislocation stress
(an independent `cutde` full-space reference) to ~1e-7.  (The interior converges
to a constant; the genuine elastic field still concentrates at the fault tips,
so the interior, not the peak, is the convergence metric.)

Because the BEM is solved in **displacement** (a prescribed fault enters only
as a smooth slip -> displacement source on the RHS), the eigenstress never
enters a displacement row — subtracting it there would be wrong.  It belongs in
the **stress readout**: `mbem.evaluate_stress` sums the boundary and fault
stress kernels and removes the fault eigenstress, producing the genuine
on-fault elastic (Coulomb) stress the BEM previously could not.
`configs/onfault_stress.py` reads it off a free-surface box solve and recovers
the half-space dislocation stress.  The one place it enters a *solve* is the
traction row of a fault whose slip is unknown (next section): that row IS a
stress readout, and with the eigenstress left in a rigid translation of the
body would load the fault.

## Faults with prescribed traction, and continuous slip

A `BCType.FAULT_TRACTION` fault solves its slip instead of prescribing it: the
unknown is the Burgers vector `b` at the fault's nodes, and the rows are the
elastic traction there, every source of the region included, equal to the
prescribed traction change `Patch.value` (a drop `dtau` along `s` is
`value = -dtau s`).  The traction of a dislocation is continuous across its own
surface, so the row has no free term; mollified, the hypersingular integral is
an ordinary one, computed by the same closed forms as the readout
(`kernels/tri_nodal.py::traction_matrix`).  Three Cartesian components per
node; by default the two tangential components are enforced and `b . n = 0`
constrains the third (`allow_opening=True` enforces all three, for a
pressurised crack); the crack front is locked at zero slip unless it lies on a
boundary patch, where a trace stays free.  It couples to topography,
interfaces and the box like any other unknown, and a crack alone in an
infinite body is a region with no boundary patch.

Such a fault can carry the **continuous P2 layout** (`layout="continuous"`):
one value per shared node (vertices and edge midpoints), collocated at the
nodes.  A piecewise-constant slip radiates a `1/d` field from every
inter-element jump, which floors the on-fault stress at a few per cent of the
drop at the centroids, with tails to tens of per cent on an irregular mesh,
and no refinement moves it; continuous slip leaves only the kink in the
gradient, which the mollified kernel bounds, so collocating at the nodes is
legal and the readout converges in eps.  On the penny-shaped crack
(`mbem.cases.penny`, `tests/gates/mbem/verify_penny_crack.py`): the readout of
Eshelby's slip is 1.6-1.8 eps/a on a quality mesh, a jittered one and a
lattice alike, the same at the kinks as between them; the stress-drop solve
at eps = 0.1 h has a slip error of 1.6 % / 1.0 % of the peak at 8 / 16
elements per radius (P0: 4.8 % / 2.7 %), 0.7 % / 0.4 % at the fault's "auto"
eps, moments within 1.5 %, and a coupled round trip through a box recovers a
prescribed solution to 1e-14.  Two unknowns per element instead of six for
the discontinuous P2 layout.  Continuous P1 is too stiff and is refused; on a
faceted fault the nodal unknown is a vector, never a scalar times a per-facet
direction (the gate's clause [g]); the symmetric part of the fault operator
is indefinite at eps/h <= 0.1, which the dense LU does not mind.  The
`configs/penny.py` study runs the ladder.

## Viscoelastic relaxation

A region may relax: its shear modulus becomes a function of the Laplace
parameter `s` (`mbem.rheology.Maxwell`, `Burgers`) while its bulk modulus stays
elastic, so Poisson's ratio moves in time.  By the correspondence principle the
problem at each real `s` is an ELASTIC one at `(mu(s), lam(s))`, on the same
mesh, with the same boundary unknowns -- there is no volume mesh and no stored
history.  It costs no kernel work either, because with `g = 1/(1 - nu)` every
operator of a region is geometry times `{1, g} x {1, mu, 1/mu}`
(`docs/viscoelastic.md`), which holds for the elastic stress and NOT for the
raw total: the eigenstress removal above is also what makes this work.

A run declares the rheology and the code derives the states:

```
python -m mbem run configs/plate_substrate.py --sweep half_x=400,800 --sweep sides=free,clamped
```

```python
from mbem import laplace
s, stack = laplace.run_stack(run_dir)            # elastic state, then one per s
u_top = np.tensordot(laplace.Prony(s).weights(t), stack["u:top"], axes=1)
```

`Prony` fits a constant plus exponentials whose relaxation times are the
reciprocals of the collocation points, so the map from the stack to any time
is one matrix: value, rate, the closed-form sum over an infinite train of
repeating earthquakes, and the velocity of a fault locked between them.  At the default three points per decade the inversion
error is under 1e-3 on exact transforms, well below the spatial error.  Time
has the unit the rheology's `tau` has.

Three things to know.  A region that relaxes needs ONE scalar `eps` on its
closed boundary: with `eps="auto"` a nearly relaxed region loses its pressure
response (9 % at mu(s)/K = 5e-4, 54 % at 5e-5 on the Eshelby sphere) and with
one scalar it does not (0.2 % throughout), so the runner refuses the former.
The box is part of the model: there is no half-space, and a plate with free
sides floats as its substrate relaxes, which clamped sides prevent (a box
whose walls are a window, further down, is the way out of both).  And a
fault in a fluid under a sustained traction creeps for ever, which the fit
represents only with `Prony(secular=True)`.

`configs/plate_substrate.py` is the Nur-Mavko / Savage-Prescott model, an
elastic plate over a Maxwell (or Burgers) substrate with a strike-slip fault in
the plate, and its study is a box-size ladder against the antiplane closed
form.  The postseismic displacement 16-32 km from the fault is 0.86 / 0.96 /
0.99 of the closed form at one Maxwell time and 0.77 / 0.91 / 0.97 at ten, for
box half-widths of 400 / 800 / 1600 km: it converges, and it tells you how much
box a result needs.

A Burgers substrate (`--set tau_K=0.1 --set t_min=0.02`) converges on the same
ladder at the same rate.

The time-stepper with memory on the boundary, which would couple this to
stress-driven fault slip, is designed in `docs/viscoelastic.md` and not built.

## Gravity

A body that relaxes with no gravity sinks under a load for ever; with it, the
surface settles where the relief it has built weighs what the load does.
`mbem.model.Gravity` puts that in as a condition on the boundary:

```python
from mbem.model import Gravity, RegionModel
model = RegionModel(regions, gravity=Gravity(
    density={"plate": 2700.0, "substrate": 3300.0},   # kg/m^3, every region
    surface=("top",)))                                # the patches open to the air
```

```
python -m mbem run configs/plate_substrate.py --set rho_plate=2700 --set rho_substrate=3300
```

The body is prestressed hydrostatically and a displacement carries that
prestress with it.  Treating the material as incompressible IN THE GRAVITY
TERMS ONLY, the equations inside a region stay the homogeneous elastic ones
and all that is left is a traction proportional to the displacement:
`rho g (u . up)` along the normal on the free surface, the same with the
density CONTRAST on an interface between regions, and nothing on a fault, or
on the sides and base of a box, which are cuts through material that
continues.  So there is no kernel work and no volume term, the buoyancy
coefficient is not a modulus (a Laplace state under gravity is still an elastic
solve), and both compressed backends carry it.  The price is an error of
relative order `rho g L / K` in a compressible body, 6 % per 100 km of
deformation scale at K = 50 GPa; `docs/viscoelastic.md` section 7 has the
derivation.

It is gated against a body that HAS an exact solution under gravity, which a
flat earth does not: shells welded on a rigid core with radial buoyancy
(`Gravity(center=...)`).  The solved surface displacement is within 2 % of
exact with and without gravity while gravity itself stiffens the shell by
28 %; relaxing, it follows the exact response in time to the same 2 % and
ends isostatic to 4e-4, where without gravity the load has sunk 2500 times as
far.  A relaxed flat box under a smooth load is hydrostatic to 3e-3.

Three things to know.  The stress `evaluate_stress` returns is the stress at a
fixed point of space; the stress of the material is that plus
`rho g (u . up) I`, the same in shear.  A Laplace run refuses a stratification
with the heavier side up, because a body that relaxes then has a growing mode
and the fit has only decaying ones.  And a long strike-slip fault barely feels
any of this (6e-4 of the fault-parallel displacement on the plate above);
gravity acts on vertical motion.

A thrust is where it is first order, and `configs/plate_substrate.py` runs one:

```
python -m mbem run configs/plate_substrate.py \
    --set dip=30 --set rake=90 --set rho_plate=2700 --set rho_substrate=3300 \
    --set edge_near=10 --set t_max=100 --set half_x=800 \
    --sweep refine=1,2 --sweep gravity=0,1
```

The fault breaks the surface along x = 0 and dips toward +x; `rake` is Aki &
Richards' (180 right-lateral, the default; 90 a thrust).  Dip slip bends the
plate.  In the reference a 30-degree thrust to 15 km in a 30 km plate leaves
a trough over its lower edge 0.11 of the slip deep at the earthquake, 0.30
after ten Maxwell times and 0.52 after a hundred, where it stops, the plate
afloat; without gravity it is 0.97 at a hundred and has no limit.

That reference is a second calculation, since no closed form covers a plate
over a relaxing substrate under gravity: plane strain, one wavenumber at a
time (`mbem.cases.layered_box.dip_slip_profile`), itself held to cutde's
half-space (4e-6), to the classical stiffness of a loaded half-space and of a
plate on a fluid, to the exact rotation of a dislocated free plate (7e-5) and
to a contour inversion in time (3e-4).  Against it, on the 800 km box above,
the postseismic displacement within 120 km of the trace is off by 2.8 % at
one Maxwell time, 2.0 % at ten and 5.7 % at a hundred, and what is left at a
hundred is horizontal (9 %, the walls of the box; the vertical is 2.9 %).
The box's fault is 800 km long, not infinitely: against the same reference
in three dimensions (below) those are 4.0 %, 2.5 % and 3.7 %, and the
horizontal 6 %.

One thing to know that the strike-slip plate never showed.  A plate that
bends is only as good as its two faces: with piecewise-constant elements the
relaxed vertical motion is short by an amount first order in the elements on
the top and on the interface and in the interface's eps -- 12 % at 10 km,
15 km and 1.5 km on a 30 km plate, 6 % at half of each.  Being first order,
it cancels between two meshes that have those three in one ratio, which is
what `--sweep refine=1,2` runs and the figure draws
(`mbem.cases.layered_box.two_mesh_limit`): 14 % and 8 % in the vertical at a
hundred Maxwell times become the 2.9 % above.

**The earthquake cycle.**  A fault locked between earthquakes that repeat
every `period` is the same fault slipping steadily, less a difference that is
periodic, and the fit sums it in closed form:

```python
s, stack = laplace.run_stack(run_dir)
v = np.tensordot(laplace.Prony(s).locked_rate_weights(t, period),
                 stack["u:top"], axes=1)       # per unit slip RATE
```

is the surface velocity at `t` after the latest earthquake, measured from
steady slip.  Its mean over the cycle is minus the elastic response: back
slip is the cycle's average whatever relaxes, and relaxation only moves
velocity between early and late.  For the thrust above, repeating every
twenty Maxwell times, a point 97 km into the hanging wall moves at -0.14 of
the slip rate just after an earthquake and +0.21 just before the next, about
a mean of +0.11: it goes on the way the earthquake moved it and turns a sixth
of the way through the cycle.  The reference for this needs no fit (a
periodic response is a Fourier series, its coefficients the solution at
imaginary `s`), and the model is within 2.3 % of it early in the cycle and
2.1 % late (4.0 % and 2.0 % against the fault as long as it is).  Any dip-slip run of `configs/plate_substrate.py` draws it
(`plate_cycle`).

**A megathrust.**  A fault deeper than the plate is thick goes through the
interface, and its lower reach is a second fault, in the substrate: slip
inside the region that relaxes.

```
python -m mbem run configs/plate_substrate.py \
    --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 \
    --set rho_substrate=3300 --set edge_near=10 --set near_field_radius=200 \
    --set half_x=800 --set t_max=250 --sweep refine=1,2
```

The reference takes the same fault (two sources, each in the moduli of its
own layer), and its lower source is held by three limits that are exact: an
interface between like materials is none (7e-15), under a plate 10 m thick
the half-space closed form holds whatever the substrate (5e-4), and slip in
a fluid moves nothing at the surface (1e-6).  That last is the physics too.
For a megathrust dipping 20 degrees to 45 km through a 30 km plate, the part
below the plate is two thirds of the vertical motion at the earthquake and
under 1 % by a hundred Maxwell times: slip in a body that relaxes is
forgotten, and what is left is a plate cut through, whose two halves
converge as the substrate lets go.  So the far field catches up with the
trench (400 km into the hanging wall, 0.09 of the slip at the earthquake and
0.45 at a hundred Maxwell times), and the cycle is the one subduction zones
show: repeating every fifty Maxwell times, a point 200 km inland moves
toward the trench at 0.56 of the slip rate just after an earthquake and away
from it at 0.45 just before the next.

That reference is an infinitely long megathrust, and the run's is 800 km
long, which is not long enough: at mid-strike the two differ by 11 % in the
vertical at the earthquake and by 5-10 % in the horizontal once the
substrate relaxes.  So the model is read against the same layer system in
three dimensions (`mbem.cases.layered_space`, a finite rectangular fault).
With clamped walls 800 km away it is within 0.2 % (horizontal) and 0.8 %
(vertical) at the earthquake, with a rigid base where the model's is, and
within 8 % at one Maxwell time and 12 % at 250 in relaxation: the plate's
two halves are held at 0.81 of the reference's convergence.  Free sides
(`--set sides=free`) let them go too far, 1.11.  The two bracket the
reference at every time, and neither is what a plate beyond a wall does.

**The box as a window.**  Clamped or free, the walls of a box are wrong
about a plate over a substrate that relaxes, at any size the box can have.
`--set sides=window` gives every wall of the box the displacement the slip
produces in a plate over a half-space instead, state by state
(`mbem.model.FarField`), from that same reference in three dimensions.  The
box then solves the near field, and its size stops being the model.

```
python -m mbem run configs/plate_substrate.py \
    --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 \
    --set rho_substrate=3300 --set edge_near=10 --set near_field_radius=200 \
    --set fault_half_len=150 --set half_x=400 --set sides=window \
    --set t_max=100 --sweep refine=1,2
```

It was chosen by measurement, on the elastic fault box against the
half-space: 22 % off beyond 128 km with the walls it had, 15-25 % with
spring walls at their best fitted stiffness, 1 % as a window
(`configs/fault_box.py --set walls=window`).  On the megathrust above, same
box and same mesh, the horizontal postseismic misfit goes from 14 % clamped
and 16 % free to 2 %, and the plate's halves converge 0.99 of the
reference's.  The box can then be small: a rupture 300 km long in a box 400
km to its walls is 23 % off in the horizontal between clamped walls and
2.6 % in a window.  What a window does not mend is the vertical, which was
never the walls but the plate's two faces, and that is the two-mesh limit's:
the pair this command runs is within 1.6 % of the reference at one, ten and
a hundred Maxwell times (0.5 % horizontal, 3 % vertical) and within 0.5 %
through the cycle.

A window knows what its reference solves, prescribed slip in a plate over a
half-space, and is blind to the far field of anything else: the next two
sections each have something else in them.


**Afterslip.**  A fault that resists its slip RATE is, at each `s`, a fault
with a stiffness, so linear creep on a fault needs no time-stepper either
(`mbem.rheology.ViscousFault`: a shear zone's Maxwell time and width;
linearized rate-strengthening friction is the same law).  It is exact
against a penny crack whose faces drag so that its slip stays Eshelby's
(0.6 %, the crack's own, per `s` and in time).  `--set lower=creep` makes
the megathrust's lower reach such a fault: locked at the earthquake, then
loaded by the rupture above it and by the substrate as it relaxes.

```
python -m mbem run configs/plate_substrate.py \
    --set dip=20 --set rake=90 --set depth=45 --set rho_plate=2700 \
    --set rho_substrate=3300 --set edge_near=15 --set edge_interface=22.5 \
    --set eps_substrate=2.25 --set near_field_radius=120 --set sides=window \
    --set half_x=300 --set depth_box=200 --set fault_half_len=100 \
    --set t_max=100 --set backend=dense --set lower=creep \
    --sweep creep_tau=1,1e6 --sweep refine=1,1.5
```

The reach slips by 0.22 of the rupture's slip on average (0.45 just under
the rupture, 0.08 at its lower edge), all of it within ten Maxwell times,
and at the surface it is an early signal: one Maxwell time after the
earthquake the postseismic displacement is 34 % away from the reference
without afterslip, at ten 11 %, and at a hundred the creeping and the held
model are one.  A solved fault is on the dense backend, so these runs are
31,179 unknowns at most and their vertical carries the plate's faces (14-23
% of the postseismic part; the horizontal is within 1.4 %).


**A slab.**  Layers are what that reference knows; a boundary method is for
what it does not.  `slab_depth` lets the footwall plate go on down the dip
as an elastic slab, in the mantle that relaxes around it:

```
python -m mbem run configs/plate_substrate.py \
    --set dip=20 --set rake=90 --set depth=30 --set rho_plate=3300 \
    --set rho_substrate=3300 --set edge_near=10 --set near_field_radius=150 \
    --set sides=window --set half_x=500 --set depth_box=300 \
    --set fault_half_len=100 --set t_max=100 --sweep slab_depth=0,150
```

Plate and slab are one region and the interface wraps the slab; a slab of
the mantle's own material is no slab, to 1 %, which is how the geometry is
gated.  At the earthquake it is invisible (the surface within 0.8 % and
1.4 % of the layered reference).  Afterwards it is the largest thing in the
model: the uplift behind the trench, which halves in a hundred Maxwell times
without a slab, stays (0.29 of the slip at the earthquake, 0.25 at a
hundred), and the trough over the megathrust's lower edge deepens for ten
Maxwell times where without a slab it fills.  The slab is an elastic strut
through the mantle and goes on carrying what the mantle sheds.  The
postseismic displacement with and without it differs by 61 % at one Maxwell
time, 70 % at ten and 43 % at a hundred.  Plate and mantle have one density
here: a slab of a lighter plate hanging in a heavier mantle is unstable once
anything relaxes, and a Laplace run refuses it.

## Layout

```
src/mbem/                 region-model BEM solver: dense and block-compressed
                          backends (far="aca" flat H + ACA, far="fmm" bbFMM),
                          numba kernels, topography, preconditioned FGMRES
  backends/               dense and block-compressed assembly
  model/                  region graph, patches, faults, the free term, the
                          shared-node (continuous P2) layout
  kernels/                numba kernels, P0/P1/P2 closed forms
  la/                     ACA, flat H-matrix, bbFMM, FGMRES, preconditioners
  cases/                  reference models the gates and studies share (the
                          fault box, the inclusion, topography, the penny
                          crack, the plate over a substrate, and that plate
                          in three dimensions: layered_space, a finite fault
                          at the surface and at depth)
  rheology.py, laplace.py linear viscoelasticity by correspondence: Maxwell
                          and Burgers shear moduli, a fault that creeps, and
                          the fit that turns per-s elastic solves into a
                          function of time
  figures/                the figure makers, in a lazy registry
src/mollified_kernel/     point-source kernel + analytic triangle integration
  mollified_elastic_kernels.py   regularized Kelvin point kernels
  analytical_kernels.py          analytic per-triangle integration (scalar)
  analytical_batch.py            analytic per-triangle integration (vectorized)
src/moss_kernel/          a second, INDEPENDENT copy of the analytic kernels;
                          the gates compare the two entrywise
src/clq/                  frozen oracle: separate closed-form derivation
src/mollified_bem.py      TriMesh, ElasticMaterial, full-space Kelvin kernels
src/local_box_mesh.py     box / fault mesh builders
src/local_box_mesh_eq.py  equilateral Delaunay box / fault mesh builders
src/inclusion_mesh.py     host + cylindrical-inclusion mesh builder
src/anelastic.py          anelastic (eigenstrain) eigenstress -> elastic stress
src/tde_reference.py      classical triangular-dislocation reference
configs/                  nine studies, each a Python module declaring a run
tests/                    run_all.py + test_gates.py; 47 gates (print PASS/FAIL)
studies/                  bench_scaling.py and the printing quickstarts
docs/                     clq derivation, the two closed-line findings
```

`docs/figures/` is the tracked figure gallery and ships **empty by design** —
`python -m mbem publish <run-dir>` is what puts a figure there, with the
provenance of the run that made it. Nothing writes to it as a side effect.

## Install

Python >= 3.11. MIT licensed.

```
pip install -e .                 # numpy scipy matplotlib numba triangle threadpoolctl
pip install -e '.[tde,test,viz]'  # + cutde (two gates FAIL without it), pytest,
                                  #   and vtk (verify_volume's independent reader)
# plus an OpenMP runtime numba can load (conda-forge llvm-openmp, or brew libomp)
```

One editable install and everything imports as a package from any working
directory. Nothing in the solver's import path manipulates `sys.path`, and
adding some there would be a bug. Three places legitimately do: the `__main__`
self-test blocks of the two frozen `analytical_kernels.py` copies, and the gate
harness (`tests/conftest.py`, plus two `moss_kernel` gates), which has to place
a specific oracle directory ahead of an identically-named sibling.

numba must resolve its OpenMP threading layer (`numba.threading_layer() ==
"omp"`) and threadpoolctl must be importable: the compressed backend runs
nogil kernels on a Python thread pool while parallel kernels run beside them,
which the OpenMP layer tolerates and the default workqueue layer does not,
and threadpoolctl is what pins BLAS to one thread inside that pool.

## Run a study

A study is a Python config module; each run writes a new self-describing folder
under `runs/` (gitignored).

```
python -m mbem run configs/fault_box.py
python -m mbem run configs/topo_inclusion.py --set surface=flat --set backend=fmm
python -m mbem run configs/penny.py --sweep a_over_h=8,16 --sweep order=0,2   # the crack ladder
python -m mbem run configs/fault_box.py --dry-run   # validate only, build nothing
python -m mbem list
python -m mbem show <run-dir> --section effective
python -m mbem verify [-k fmm] [--fast]             # the gates, through the CLI
```

A run folder holds `resolved.json` (the spec, the keyword arguments actually
passed, eps resolved to numbers per patch, every uppercase `defaults` constant, and the
environment), `report.json` (iterations, residuals, phase timings, peak RSS),
the solution fields as `.npz`, and a `MANIFEST` of sha256 hashes.

A config *names* a mesh and model builder rather than describing patches and
boundary conditions, so it cannot restate the fault sign convention or the eps
rule. `tests/gates/mbem/verify_config.py` proves the config path builds the same
model the gates do, fault Burgers vector bitwise included.

## Verify the kernels

Every gate prints one final `PASS: <title>` / `FAIL: <title>` line and exits 1
on FAIL; `python tests/run_all.py` runs all 47 sequentially and tabulates
verdict and wall time; `pytest` is the same set behind a second front end.
`--fast` / `-m "not slow"` skips the ~900 s FMM gate.

```
python tests/gates/mbem/verify_analytical_vs_quadrature.py   # analytic == high-order quadrature
python tests/gates/mbem/verify_arbitrary_triangle.py         # arbitrary-triangle / rigid / scaling
python tests/gates/mbem/verify_batch_vs_scalar.py            # vectorized == scalar
python tests/gates/mbem/verify_dd_pairing.py                 # lambda/mu pairing of the DD displacement kernel (closure, nu sweep); point kernel satisfies the regularized Navier equation
python tests/gates/mbem/verify_evaluate_stress.py            # mbem stress == classical TDE (cutde); on-fault elastic stays finite
python tests/gates/mbem/verify_stress_assembler.py           # numba batched stress assemblers == scalar oracle (machine precision)
python tests/gates/mbem/verify_eigenstress_exact.py           # EXACT finite-triangle eigenstress == moss/clq oracles; sign; rim disagreement with anelastic.py
python tests/gates/mbem/verify_boundary_eigenstress.py       # boundary double layers subtracted too: near-boundary interior stress converges (icosphere vs exact Kelvin)
python tests/gates/mbem/verify_solved_bvp.py                 # assembly -> BCs -> solve -> displacement vs cutde half-space (strike slip, and a 30-degree thrust) and manufactured solutions (end to end)
python tests/gates/mbem/verify_disp_contract.py              # matrix-free displacement evaluation == dense matrices == legacy oracle
python tests/gates/mbem/verify_dense_backend.py              # dense assembly invariants (calibration cache bit-identity, rebuilds)
python tests/gates/mbem/verify_hbackend.py                   # H backend == dense (ACA, calibrated jump, combined storage, rung ladder)
python tests/gates/mbem/verify_eps_auto.py                   # eps="auto" (0.1 h): kernel order-2 convergence, solved-BVP accuracy vs Kelvin, half-jump guard
python tests/gates/mbem/verify_deflation_estimate.py         # all-Neumann rigid-body deflation; memory estimator
python tests/gates/mbem/verify_nodal_kernels.py              # P0/P1/P2 numba kernels == clq oracle (1e-12); edge primitives; order 0 == the P0 path
python tests/gates/mbem/verify_nodal_solve.py                # P1/P2 solves: P0 bitwise, patch test, rigid covariance, h-convergence, P1 faults, refusals, the shared-node layout
python tests/gates/mbem/verify_penny_crack.py                # traction-BC fault on the penny crack vs Eshelby/Sneddon; continuous P2 readout; coupled round trip; rigid translation
python tests/gates/mbem/verify_viscoelastic.py               # every operator is {1, g} x {1, mu, 1/mu} times geometry, g = 1/(1-nu); one eps per closed surface; the Laplace route against closed forms per s and in time (relaxing full space, Eshelby sphere, two-layer torsion, creeping penny crack); gravity on a relaxing shell; a dipping fault's plane-strain reference and a thrust under gravity against it, its two-mesh limit, a locked fault's cycle against a Fourier series, and the fault taken through the interface
```

## The field in the volume

The solve reports its answer on the boundary. `mbem sample` re-evaluates a
stored run's solution on a 3-D grid and writes VTK ImageData, which opens by
dragging into ParaView Glance or with desktop ParaView:

```
python -m mbem sample <run-dir>                       # 4 km default
python -m mbem sample <study-dir> --spacing 8 --difference
python -m mbem sample <run-dir> --state het --no-eigenstress
```

Post-processing, not a re-solve: the model is rebuilt from the run's own spec
(the mesher is re-run, ~3 ms, because `meshes.npz` does not carry
`bundle.scalars` and so cannot reproduce a fault), the per-patch `mesh_sha256`
is checked against the stored fingerprint, and the assembly and solve are
skipped. The run must have saved every slot — `outputs=Output(slots=())` — since
an interior point needs the density on all of them.

Each `.vti` carries displacement, elastic stress and strain as six named
components each, the subtracted eigenstress `C:eps*`, and `u_mag`,
`von_mises`, `mean_stress`, `max_shear`, `dilatation`. A study samples every
child on **one shared grid**, which is what makes a cross-run difference well
defined when the two runs have different meshes; a difference is written only
where both bodies contain the point, and its scalars are recomputed from the
differenced tensors rather than differenced (the von Mises of a perturbation is
not a difference of von Mises values).

Two arrays decide whether a value is trustworthy. `region` is 0 outside the
body — points above the topography and outside the box are NaN, because 0 is a
value. `clearance_h` and `clearance_eps` give each point's distance to the
nearest element in units of that element's size and of its own eps. **Both are
flagged, never blanked**, and it matters: on the showcase model the topography
effect reads 1287 mm at a point lying on the surface and 96 mm once
`clearance_h > 0.5`.

### What the eigenstress subtraction does and does not buy

Subtracting `C:eps*` is what makes a stress *finite and convergent* where the
singular formulation diverges — on a fault, which is why `_warn_near_boundary`
exempts faults. It does **not** make the stress exact, and the two limits are
separable only if eps is given absolutely (`eps="auto"` ties it to 0.1 h).
Measured on the manufactured `u = A x` box, whose interior stress is exactly
`C:A`, over h in (20, 10, 5) km x eps in (0.6, 1.5, 3.6) km:

| | eps = 0.6 | eps = 1.5 | eps = 3.6 |
|---|---|---|---|
| deep interior (`clearance_h > 2`), h = 10 km | 7.3e-3 | 1.66e-2 | 4.21e-2 |
| deep interior, h = 5 km | 6.8e-3 | 1.71e-2 | 4.13e-2 |
| degradation near a boundary (`clearance_h` 0.15-0.3) | 12.9x | 4.7x | 2.9x |

So there is a floor everywhere, and it is **eps/L for the DOMAIN size L** —
not eps/h. So 1 % needs eps <~ 0.01 L, and refining the mesh does not help.
Measured, each factor isolated:

| held fixed | varied | result |
|---|---|---|
| L, eps | h: 10 -> 5 km | moves under 8 % |
| L, h | eps x8 | `eps^1.01` at P1 and P2 |
| eps, eps/h = 0.145 | L: 40 -> 120 km | `err*L/eps` = 1.060, 1.091, 1.101 (`err*h/eps` moves 3x) |
| L, h, eps | order P0 -> P1 -> P2 | P1 and P2 agree to 0.5 % |
| L, h, eps | jump calibrated -> half | agree to 9 % |
| — | a CONSTANT field | 3e-15 at every eps and every order |

It is not the discretization (flat in h), not the density order (P1 == P2), and
not the calibration (half == calibrated). What is left is the smeared boundary
itself: in the interior a symmetric unit-integral blob has zero first moment and
reproduces a linear field exactly, but AT A BOUNDARY the body is on one side
only, the convolution is truncated, and the effective surface sits O(eps) off —
which costs a field with a gradient a relative `eps/L`. That last step is an
inference from the scalings rather than a derivation, but it is what survives
after order and free term were ruled out, and it predicts the asymmetry below:
a fault is interior, its blob is two-sided, and this error does not arise there.

Proximity to a *solved* boundary degrades it a further 3-13x on top. Neither
variable governs alone: binning on `clearance_h` leaves a 4.2x spread across
(h, eps) and on `clearance_eps` a 3.8x spread, which is why both arrays ship.

**A fault is exempt**, and for a stronger reason than its slip being prescribed
data. Because the regularization is applied to the SOURCE, a fault's smearing
over eps IS the finite-width fault zone the method exists to represent, so a
point 2 eps from the fault is reading the model. A boundary patch's smearing
has no such warrant — the free surface is not physically smeared — so within
~eps of it you are inside a layer the numerics invented. The same quantity,
distance in units of eps, is physics at one and an artefact at the other, which
is why `clearance_h`/`clearance_eps` are measured to BOUNDARIES only and
`fault_eps` is reported separately and is not a defect measure.

One practical consequence, against what the warning text advises: at fixed eps,
refining h by 4x improved the near-boundary residual only 1.27x, while standing
off from `clearance_h` 0.2 to 2 improved it 13x. Evaluate deeper; refining the
patch alone barely helps.

Cost is linear in the interior point count and dominated by the stress kernel:
measured 2.3 ms per point on the 11k-triangle showcase model (0.22 for
displacement, 2.09 for stress with the eigenstress in the same pass). So 8 km
spacing is ~1.6 min a state and 4 km is ~12 min.

## Scaling

The solver stack scales far beyond the dense backend's ~30k-element cap:
`HBackend(jump="calibrated", storage="combined")` assembles the operator
block-compressed (matrix-free ACA, parallel across blocks, 1x material-combined
storage), FGMRES is preconditioned by a block-Gauss-Seidel ladder whose rung
past the dense-LU cap (cluster block-Jacobi) has O(N) build cost at any patch
size (a HODLR rung is reachable explicitly, for memory over speed), and field
evaluation is matrix-free (O(N_obs) memory; a 250k-point map over a model that
would need a 49 GB dense operator runs in ~9 s and 0.4 GB).
`mbem.estimate.estimate_memory` predicts the footprint of each backend/mode
before assembling; `studies/mbem/bench_scaling.py` is the regression harness:
`--model fault_box|topo_inclusion --scale s` mesh ladders with per-rung JSON
records (phase times, iterations, ranks, bytes, RSS, operator error against
the dense operator or exact matrix-free rows), `--gate REV` against the
`bench-json:` line of a commit message, and `--panel N` for the large-block
compression primitive alone.

## Examples

Every figure is produced by a named maker, and the output lands in a new
directory under `runs/` together with the provenance of whatever produced it.
Three kinds, because the figures are genuinely of three shapes:

**Model-free** — the kernels alone, no solve:

```
python -m mbem figure point_kernel            # singular vs mollified point source
python -m mbem figure triangle_field          # triangle stress field
python -m mbem figure triangle_eps_sweep      # the field over an eps ladder
python -m mbem figure eps_h_convergence       # eps-h decoupling (analytic)
python -m mbem figure anelastic_subtraction   # the anelastic term is finite
python -m mbem figure onfault_convergence     # on-fault stress -> const, vs TDE
python studies/mbem/demo_triangle_quickstart.py     # prints numbers, no figure
```

**One solve:**

```
python -m mbem run configs/fault_only.py      # displacement + elastic surface stress
```

**Several runs**, because the quantity drawn is a difference between operators —
a different mesh, eps or backend is a different operator, so it is a different
run:

```
# fault + inclusion + topography: the four-state decomposition
python -m mbem run configs/topo_inclusion.py --sweep surface=topo,flat

# the fast operator against the reference, same model and eps
python -m mbem run configs/backend_agreement.py --sweep backend=hmat,dense

# surface field convergence in eps, at a fixed mesh
python -m mbem run configs/eps_convergence.py --sweep eps=12,8,6,4,3,2

# on-fault shear vs the classical TDE, and the first-row P0/P1 contrast
python -m mbem run configs/onfault_stress.py --sweep eps=4,2,1 --sweep order_top=0,1

# what the coarse box sides cost the top surface, top mesh held fixed
python -m mbem run configs/side_grading.py --sweep edge_side=80,40,20,10
```

`python -m mbem publish <run-dir>` copies chosen figures into the tracked
gallery at `docs/figures/` and appends a provenance line naming the run, the
commit and the file hash. Nothing writes there as a side effect: before this,
every demo overwrote one tracked image in place, so the committed figure was
whatever ran last and nothing recorded which code produced it.

The BEM demos run at paper resolution (tens of seconds to a few minutes; the
four-state topography+inclusion solve is the longest).  The free surfaces are
fault-ALIGNED (the surface-breaking fault trace is embedded as exact mesh edges
via `make_top_patch_with_fault`), so no triangle straddles the slip
discontinuity.  The inclusion is the soft `mu/10` body in the showcase; any
model keyword is reachable from the command line as `--set mu_inc=3.0`.
The `topo_inclusion_showcase` figure draws the three rows of the
decomposition (raw field / inclusion effect / topography effect) and
`topo_inclusion_contour` the topography effect alone.

The topography effect differences two fields that agree to 43-72x their
difference, so it inherits ~2e-4 relative error from fields accurate to ~2e-6.
That is a property of the quantity, not of the backend: the ACA and FMM far
fields land in the same place.

The H-matrix demo solves the `mu/10` inclusion with the block-compressed backend
(`mbem.backends.HBackend(eta=0.8)`, ACA + preconditioned FGMRES) and reproduces
the dense LU reference to ~1e-9.  (Use `eta=0.8`: the looser default
admissibility can accept inaccurate low-rank factors for a near-field block.)

## References

Viscoelasticity by correspondence on the boundary goes back to Rizzo & Shippy
(1971); the layered-earth problems it is compared with are Nur & Mavko (1974),
Savage & Prescott (1978) and Rybicki (1971), and the cycle of a locked fault
is Savage (1983) and Thatcher & Rundle (1984). `docs/viscoelastic.md` has the
full list.

The mollification follows the method of regularized Stokeslets and its surface
(triangulated-BIE) extension:

- Cortez, R. (2001). *The Method of Regularized Stokeslets.* SIAM Journal on
  Scientific Computing, 23(4), 1204–1225. doi:10.1137/S106482750038146X
- Cortez, R., Fauci, L., & Medovikov, A. (2005). *The Method of Regularized
  Stokeslets in Three Dimensions: Analysis, Validation, and Application to
  Helical Swimming.* Physics of Fluids, 17(3), 031504. doi:10.1063/1.1830486
- Ferranti, D., & Cortez, R. (2024). *Regularized Stokeslet Surfaces.*
  arXiv:2310.14470 [math.NA]. — establishes the analytic per-triangle
  integration that decouples `eps` from `h`, the property exploited here.
