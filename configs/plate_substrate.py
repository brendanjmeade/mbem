"""An elastic plate over a relaxing substrate, a strike-slip fault in the plate.

The Nur-Mavko / Savage-Prescott earthquake-cycle model, as a LAPLACE run: the
elastic state plus one elastic solve per real s, all on one assembly, which
``mbem.laplace`` turns into the response in time (docs/viscoelastic.md).

    python -m mbem run configs/plate_substrate.py
    python -m mbem run configs/plate_substrate.py \
        --sweep half_x=400,800 --sweep sides=free,clamped

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
steady (Maxwell) time, still 1.
"""
from mbem import laplace, rheology
from mbem.config import Backend, Geometry, Laplace, Model, Output, Run, Solve


def run_spec(half_x: float = 600.0, sides: str = "clamped",
             backend: str = "hmat", scale: float = 1.0,
             thickness: float = 30.0, depth: float = 15.0,
             fault_half_len: float = 0.0, eps_substrate: float = 1.5,
             mu_substrate: float = 30.0, tau_K: float = 0.0,
             kelvin: float = 1.0, t_min: float = 0.1, t_max: float = 10.0,
             per_decade: int = 3) -> Run:
    if fault_half_len <= 0.0:
        fault_half_len = 0.5 * half_x
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
                                          fault_depth=depth,
                                          # the far field coarsens WITH the
                                          # box, or a ladder in half_x is a
                                          # ladder in cost and nothing else
                                          edge_far=half_x / 6.0,
                                          edge_side=half_x / 6.0)),
            builder="layered_box",
            params=dict(sides=sides, mu_substrate=mu_substrate,
                        lam_substrate=mu_substrate),
            eps=eps),
        backend=Backend(kind=backend),
        solve=Solve(),
        laplace=Laplace(rheology={"substrate": substrate},
                        s=laplace.s_grid(t_min, t_max, per_decade=per_decade)),
        outputs=Output(slots=(), save_meshes=True,
                       figures=("plate_substrate",)),
        notes="plate over a relaxing substrate; times in Maxwell times",
        tags=("viscoelastic", "laplace"))
