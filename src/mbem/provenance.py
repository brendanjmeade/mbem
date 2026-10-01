"""What was true when a run ran: commit, environment, and every default.

DELIBERATE DUPLICATION. ``studies/mbem/bench_scaling.py`` has its own
``git_hash``/``defaults_snapshot``/``peak_rss_gb``/``numba_threads`` and keeps
them. Sharing would couple the benchmark's ``bench-json:`` record -- whose
schema is already committed in git history and which ``--gate REV`` compares
against -- to this module's evolution. The benchmark's baseline comparison with
every past commit is worth more than four small functions being written once.
Recorded here so it reads as a decision rather than an oversight.

``json_safe`` is the only part that has grown: a run dump must never fail to
serialise, so arrays are summarised rather than inlined (the data belongs in the
npz) and non-finite floats are spelled out, because ``material_step`` returns
``inf`` for a region that newly appeared.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import platform
import resource
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]


def git_hash(cwd: pathlib.Path | None = None) -> str:
    """``<short sha>`` plus ``+dirty`` when the working tree has changes."""
    cwd = cwd or REPO
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                           capture_output=True, text=True, check=True
                           ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                               capture_output=True, text=True,
                               check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def peak_rss_gb() -> float:
    """Peak resident set of THIS process, monotone and process-wide.

    Monotone is why a run is one process: a parent that already built a mesh
    cannot attribute its peak to the solve that followed.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e9 if sys.platform == "darwin" else ru * 1e3 / 1e9


def numba_threads() -> dict:
    import numba
    try:
        layer = numba.threading_layer()     # resolved once a parallel kernel ran
    except ValueError:
        layer = None
    return {"threads": int(numba.get_num_threads()), "layer": layer}


def packages() -> dict:
    out = {}
    for name in ("numpy", "scipy", "numba", "matplotlib", "triangle", "cutde"):
        try:
            out[name] = getattr(__import__(name), "__version__", "unknown")
        except ImportError:
            out[name] = None
        except Exception:                   # noqa: BLE001  triangle is odd
            out[name] = "unknown"
    return out


def environment() -> dict:
    """Everything outside the repo that can change a number."""
    from mbem.estimate import total_ram_bytes
    thread_env = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    return {
        "git": git_hash(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": packages(),
        "numba": numba_threads(),
        "thread_env": thread_env,
        "cpu_count": os.cpu_count(),
        "ram_gb": (total_ram_bytes() or 0) / 1e9,
        "load_avg": os.getloadavg()[0],
    }


def defaults_snapshot() -> dict:
    """Every uppercase name in ``mbem.defaults``, as it was during the run."""
    from mbem import defaults
    return {k: json_safe(v) for k, v in vars(defaults).items() if k.isupper()}


def json_safe(v):
    """A JSON-serialisable view. Arrays are SUMMARISED, never inlined.

    A dump that silently failed to serialise would be worse than none, so this
    handles everything the specs and reports actually contain -- including
    non-finite floats, which `json.dumps(allow_nan=False)` rejects and which
    `material_step` legitimately produces.
    """
    if isinstance(v, np.ndarray):
        return {"__array__": {"shape": list(v.shape), "dtype": str(v.dtype),
                              "min": json_safe(v.min()) if v.size else None,
                              "max": json_safe(v.max()) if v.size else None}}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return json_safe(float(v))
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(v, pathlib.Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: json_safe(getattr(v, f.name))
                for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [json_safe(x) for x in v]
    if callable(v):
        return f"{getattr(v, '__module__', '?')}:{getattr(v, '__qualname__', v)}"
    return v
