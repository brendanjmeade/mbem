"""Locate and import the ``clq`` closed-form kernel package.

``clq`` is a sibling sub-project of ``moss-org`` with no installer (the tree has
none anywhere), so it is put on ``sys.path`` by file location: ``../clq`` next
to this repository's root.  ``DDBEM_CLQ_ROOT`` overrides the location.  Nothing
in ``ddbem`` writes to ``clq``; it is read-only and frozen (see
``clq/verify/verify_baseline_bitwise.py``, which pins U/H/E against byte
hashes -- if a ddbem change ever makes that gate fail, something has reached
into clq and the manifest must NOT be regenerated).
"""
from __future__ import annotations

import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_DEFAULT_CLQ_ROOT = _HERE.parents[2] / "clq"


def _clq_root() -> pathlib.Path:
    env = os.environ.get("DDBEM_CLQ_ROOT")
    return pathlib.Path(env).expanduser().resolve() if env else _DEFAULT_CLQ_ROOT


try:                                        # already importable?
    import clq                              # noqa: F401
except ImportError:                         # pragma: no cover - path bootstrap
    root = _clq_root()
    if not (root / "clq" / "__init__.py").exists():
        raise ImportError(
            f"cannot find the clq package at {root}; set DDBEM_CLQ_ROOT") from None
    sys.path.insert(0, str(root))
    import clq                              # noqa: F401

CLQ = clq
