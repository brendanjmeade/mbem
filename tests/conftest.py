"""pytest configuration for the gate suites.

``tests/`` is on ``sys.path`` for the duration of collection so
``test_gates.py`` can ``import run_all`` -- the authoritative runner -- rather
than restate the gate set. One definition, two front ends.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
