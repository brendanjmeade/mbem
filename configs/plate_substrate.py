"""An elastic plate over a relaxing substrate, a fault in the plate.

The Nur-Mavko / Savage-Prescott earthquake-cycle model, as a LAPLACE run: the
elastic state plus one elastic solve per real s, all on one assembly, which
``mbem.laplace`` turns into the response in time (docs/viscoelastic.md).
A vertical strike-slip fault unless ``dip`` and ``rake`` say otherwise.

    python -m mbem run configs/plate_substrate.py
    python -m mbem run configs/plate_substrate.py \
        --sweep half_x=400,800 --sweep sides=free,clamped
    python -m mbem run configs/plate_substrate.py \
        --set tau_K=0.1 --set t_min=0.02 --sweep half_x=400,800   # Burgers
    python -m mbem run configs/plate_substrate.py \
        --set dip=30 --set rake=90 --set rho_plate=2700 --set rho_substrate=3300 \
        --set edge_near=10 --set t_max=100 \
        --sweep refine=1,2 --sweep gravity=0,1                    # a thrust

Lengths are km. TIME IS IN UNITS OF THE SUBSTRATE'S MAXWELL TIME (``tau = 1``),
which is all a linear problem needs: the rheology is stated in relaxation
times, and a year enters only when someone multiplies by one.

This is a box, not a half-space, and two of its parameters exist because of
that. ``half_x`` is the box: relaxation spreads deformation outward with time,
so how far the answer can be trusted depends on it and it is laddered. The
fault is half as long as the box is wide unless ``fault_half_len`` says
otherwise, so a ladder in ``half_x`` lengthens the fault with it and the
mid-strike profile approaches the infinitely long fault of the closed form.
``sides`` is the condition on the plate's sides: free sides let the plate float
as the substrate relaxes (a spurious net translation that grows as 1/mu(s)),
clamped sides do not, and the truth lies between them. The box is as deep as it
is half-wide.

The substrate's closed boundary carries ONE scalar eps, ``eps_substrate``. A
region that relaxes needs that (docs/viscoelastic.md, section 3) and the
runner refuses a Laplace run without it; the plate's own faces and the fault
keep ``"auto"``.

``tau_K > 0`` makes the substrate a Burgers body with that transient time and
the Kelvin spring ``kelvin`` times the unrelaxed modulus; ``tau`` is then its
steady (Maxwell) time, still 1. Lower ``t_min`` with it: the fit is trusted
only inside its window, and the transient is over by a few ``tau_K``.

``rho_plate`` / ``rho_substrate`` (kg/m^3) turn gravity on: buoyancy on the
free surface and on the interface (docs/viscoelastic.md, section 7). It leaves
the fault-parallel motion of a long strike-slip fault alone and holds back the
vertical motion, which without it grows for as long as the substrate relaxes.
A Laplace run refuses a substrate lighter than the plate. ``gravity`` scales
g, so ``--sweep gravity=0,1`` is the same body with and without it.

``dip`` (degrees, toward +x) and ``rake`` (Aki & Richards: 180 right-lateral,
90 a thrust) turn the fault; ``depth`` stays its depth extent. A dip-slip
fault is where gravity is first order: the plate bends, and without gravity
nothing but the plate's own stiffness resists a wavelength the substrate has
relaxed under. It also asks more of the mesh than strike slip does, because
a plate that bends is only as good as its two faces: with piecewise-constant
elements the relaxed vertical motion is short by an amount first order in
``edge_near`` (the top within ``near_field_radius`` of the trace),
``edge_interface`` (the interface beneath it) and ``eps_substrate`` -- 12 %
at 10 km, 15 km and 1.5 km on a 30 km plate, 6 % at half of each.

``refine`` divides exactly those three, so ``--sweep refine=1,2`` is a pair
on one ray and the error cancels between them: the thrust figure draws the
two-mesh limit (``layered_box.two_mesh_limit``) and the misfit of each.
"""
from mbem import defaults, laplace, rheology
from mbem.config import Backend, Geometry, Laplace, Model, Output, Run, Solve


def run_spec(half_x: float = 600.0, sides: str = "clamped",
             backend: str = "hmat", scale: float = 1.0,
             thickness: float = 30.0, depth: float = 15.0,
             fault_half_len: float = 0.0, eps_substrate: float = 1.5,
             mu_substrate: float = 30.0, tau_K: float = 0.0,
             kelvin: float = 1.0, t_min: float = 0.1, t_max: float = 10.0,
             per_decade: int = 3, rho_plate: float = 0.0,
             rho_substrate: float = 0.0, gravity: float = 1.0,
             dip: float = 90.0, rake: float = 180.0, edge_near: float = 20.0,
             near_field_radius: float = 90.0,
             edge_interface: float = 15.0, refine: float = 1.0) -> Run:
    if fault_half_len <= 0.0:
        fault_half_len = 0.5 * half_x
    # the plate's two faces and the interface's eps: what a plate that bends
    # is first order in, kept in one ratio
    edge_near, edge_interface, eps_substrate = (
        v / refine for v in (edge_near, edge_interface, eps_substrate))
    substrate = (rheology.Burgers(tau_M=1.0, tau_K=tau_K, kelvin=kelvin)
                 if tau_K > 0.0 else rheology.Maxwell(tau=1.0))
    eps = dict(top="auto", sides_upper="auto", fault="auto",
               interface=eps_substrate, sides_lower=eps_substrate,
               base=eps_substrate)
    return Run(
        name=f"plate_substrate_{sides}_{backend}",
        model=Model(
            geometry=Geometry("layered_box", scale=scale,
                              params=dict(half_x=half_x,
                                          z_interface=-thickness,
                                          z_bottom=-half_x,
                                          fault_half_len=fault_half_len,
                                          fault_depth=depth, dip=dip,
                                          edge_near=edge_near,
                                          near_field_radius=near_field_radius,
                                          edge_interface=edge_interface,
                                          # the far field coarsens WITH the
                                          # box, or a ladder in half_x is a
                                          # ladder in cost and nothing else
                                          edge_far=half_x / 6.0,
                                          edge_side=half_x / 6.0)),
            builder="layered_box",
            params=dict(sides=sides, mu_substrate=mu_substrate,
                        lam_substrate=mu_substrate, rho_plate=rho_plate,
                        rho_substrate=rho_substrate,
                        g=gravity * defaults.GRAVITY, rake=rake),
            eps=eps),
        backend=Backend(kind=backend),
        solve=Solve(),
        laplace=Laplace(rheology={"substrate": substrate},
                        s=laplace.s_grid(t_min, t_max, per_decade=per_decade)),
        outputs=Output(slots=(), save_meshes=True,
                       figures=(("plate_substrate",) if rake % 180.0 == 0.0
                                else ("plate_thrust", "plate_cycle"))),
        notes="plate over a relaxing substrate; times in Maxwell times",
        tags=("viscoelastic", "laplace"))
