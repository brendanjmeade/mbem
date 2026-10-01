"""mbem — rebuilt mollified-BEM solver stack.

Fast, robust, geometry-general successor to the legacy flat modules
(``mollified_bem.py``, ``local_box_bem.py``, ``hmatrix.py``, ...), which
remain frozen as validation oracles. See the plan referenced in CLAUDE.md
for the phase layout and binding design decisions.
"""

# The live geometry/material types. DEFINED in the frozen oracle
# mollified_bem.py and re-exported here, so new code has one stable import
# while the oracle stays the single definition. Not moved into the package: the
# oracle must remain loadable as a standalone file, which is how the clq gates
# and verify_eigenstress_exact reach it, and editing it is forbidden anyway.
from mollified_bem import ElasticMaterial, TriMesh  # noqa: E402

__all__ = ["ElasticMaterial", "TriMesh"]
