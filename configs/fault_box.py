"""Homogeneous box + buried strike-slip fault: the smallest real solve.

The geometry is bench_scaling's scale-1 rung, so this config and the historical
benchmark build the same model. ``scale`` divides every mesh edge.

    python -m mbem run configs/fault_box.py
    python -m mbem run configs/fault_box.py --set scale=1.6 --set backend=fmm
"""
from mbem.config import Backend, Geometry, Model, Output, Run, Solve


def run_spec(scale: float = 1.0, backend: str = "hmat",
             slip_mag: float = 0.01, rtol: float = 1e-8) -> Run:
    return Run(
        name=f"fault_box_{backend}",
        model=Model(
            geometry=Geometry("fault_box", scale=scale),
            builder="fault_box",
            params=dict(slip_mag=slip_mag),
            eps="auto"),
        backend=Backend(kind=backend),
        solve=Solve(rtol=rtol),
        outputs=Output(save_fields=True),
        notes="smallest end-to-end solve; the config/bench parity reference",
        tags=("smoke", "fault_box"))
