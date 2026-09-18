"""ddbem -- displacement-discontinuity collocation BEM on mollified triangles.

The foundation this research program builds on: a DIRECT / DD formulation whose
unknown is a displacement discontinuity (bounded) rather than the indirect
single-layer density of the dropped force-element BEM (which carries a
rho^(-1/3) edge singularity on a polyhedron and stalls at O(h^0.31) -- see
``fbem/FINDINGS.md``), with one code path for constant, linear and quadratic
nodal slip.

Stage 1 is the kernel/assembly layer only:

    displacement_matrix(x_field, tri_verts, eps, mu, nu, order)
    traction_matrix(x_field, n_field, tri_verts, eps, mu, nu, order)
    stress_matrix(x_field, tri_verts, eps, mu, nu, order)        # elastic, Voigt
    eigenstress_matrix(x_field, tri_verts, eps, mu, nu, order)   # C:eps*, Voigt

built on ``clq.influence`` (closed form on one flat triangle, P0/P1/P2, Cortez
blob).  Read :mod:`ddbem.assemble` for the three decisions those matrices
encode (elastic vs total stress, nodal layout, Fortran ordering),
:mod:`ddbem.layout` for the discontinuous/continuous seam, and
:mod:`ddbem.mesh` for the eps/h budget that governs whether p-refinement pays.

Units follow the tree: km / GPa / years (slip 0.001 km = 1 m), NumPy (N, 3)
vectors.  ``clq``, ``msd``, ``moss``, ``medt_paper`` and ``fbem`` are read-only
from here.

Gates: ``python verify/run_all.py`` from the ddbem root.
"""
from __future__ import annotations

from . import defaults, layout, mesh, shapes                           # noqa: F401
from .assemble import (                                                # noqa: F401
    displacement_matrix,
    traction_matrix,
    stress_matrix,
    eigenstress_matrix,
    eigen_column_tensor,
    voigt_to_tensor,
    tensor_to_voigt,
)
from .layout import NodalLayout, discontinuous, continuous, n_nodes    # noqa: F401
from .mesh import (                                                    # noqa: F401
    element_nodes,
    element_normals,
    element_areas,
    element_h,
    collocation_points,
    collocation_shape_matrix,
    shape_at,
    node_clearance,
    eps_report,
    eps_auto,
    shrink_for_clearance,
)
from . import model as _model                                          # noqa: F401
from .model import (                                                   # noqa: F401
    BCType,
    RowType,
    Patch,
    Model,
    System,
    Solution,
)
from .shapes import icosphere, box, rectangle, tri_verts               # noqa: F401

__all__ = [
    "defaults", "layout", "mesh", "shapes",
    "displacement_matrix", "traction_matrix", "stress_matrix",
    "eigenstress_matrix", "eigen_column_tensor",
    "voigt_to_tensor", "tensor_to_voigt",
    "NodalLayout", "discontinuous", "continuous", "n_nodes",
    "element_nodes", "element_normals", "element_areas", "element_h",
    "collocation_points", "collocation_shape_matrix", "shape_at",
    "node_clearance", "eps_report", "eps_auto", "shrink_for_clearance",
    "BCType", "RowType", "Patch", "Model", "System", "Solution",
    "icosphere", "box", "rectangle", "tri_verts",
]
