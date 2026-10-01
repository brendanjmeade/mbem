"""Cartesian mesh generators for the local-scale 3-layer box model.

Provides planar analogues of the spherical helpers in ``mollified_bem.py``:
- ``make_rectangular_patch``     — flat tensor-grid rectangle at fixed z
- ``make_vertical_panel``        — vertical wall of a box side
- ``make_layered_box``           — full closed box split into per-layer sides
                                    plus two interior horizontal interfaces
- ``make_vertical_fault_xy``     — rectangular vertical strike-slip fault

Convention: all meshes are stored as ``TriMesh`` (from ``mollified_bem``) with
outward-pointing normals set consistently:
  * ``top`` (z = 0):            normal = +ẑ
  * ``base`` (z = z_bottom):    normal = −ẑ
  * ``sides`` (per layer):       normal = outward from box
  * ``interfaces`` (z = z_i):   normal = +ẑ (pointing from the deeper
                                 layer up into the shallower one; this
                                 mirrors the sphere convention of
                                 "outward from Earth center")
  * fault: normal stored in accompanying vector; for a N–S (+y) striking
           vertical fault, ``n_hat = +x̂``, ``slip_hat = +ŷ``
"""

from __future__ import annotations

import numpy as np

from mollified_bem import TriMesh


__all__ = [
    "make_rectangular_patch",
    "make_vertical_panel",
    "make_layered_box",
    "make_vertical_fault_xy",
]


def _rect_grid_triangles(
    nx: int, ny: int, flip: bool, symmetric: bool = True,
) -> np.ndarray:
    """Return triangle index array for an (nx+1)×(ny+1) tensor grid.

    The grid is laid out row-major with the j-index (second axis) varying
    slowest, so vertex (i, j) has global index ``j*(nx+1) + i``.

    ``flip=False`` gives triangles whose computed normal (right-hand rule
    on the first edge pair) points in the +(third-axis) direction when
    vertices are laid in the natural (+x, +y) order. ``flip=True`` reverses
    the winding so the normal points in the opposite direction.

    ``symmetric=True`` (default) alternates the quad diagonal so the
    triangulation is invariant under both x→−x and y→−y mirrors. This is
    essential for BEM problems with fault geometry that has the same
    mirror symmetries: with a single fixed diagonal, the BEM solution
    picks up per-element outliers aligned along that diagonal (visible
    as "speckle" in u_z and u_x for N-S strike-slip). The symmetric
    pattern uses the NW-SE diagonal in the (+x, +y) and (−x, −y)
    quadrants, and the NE-SW diagonal in the (+x, −y) and (−x, +y)
    quadrants — giving 4-fold mirror symmetry across the grid center.
    """
    faces = []
    row = nx + 1
    for j in range(ny):
        for i in range(nx):
            v00 = j * row + i            # bottom-left in grid index order
            v10 = j * row + i + 1        # bottom-right
            v01 = (j + 1) * row + i      # top-left
            v11 = (j + 1) * row + i + 1  # top-right
            if symmetric:
                # Quadrants (++) and (−−) use NW-SE diagonal (v00–v11);
                # quadrants (+−) and (−+) use NE-SW diagonal (v10–v01).
                # This is the minimal 4-fold-symmetric triangulation.
                in_left = (2 * i + 1) < nx
                in_bot = (2 * j + 1) < ny
                use_nw_se = (in_left == in_bot)
            else:
                use_nw_se = True
            if use_nw_se:
                # NW-SE diagonal: triangles (v00, v10, v11) and (v00, v11, v01)
                tri_a = [v00, v10, v11]
                tri_b = [v00, v11, v01]
            else:
                # NE-SW diagonal: triangles (v00, v10, v01) and (v10, v11, v01)
                tri_a = [v00, v10, v01]
                tri_b = [v10, v11, v01]
            if flip:
                tri_a = [tri_a[0], tri_a[2], tri_a[1]]
                tri_b = [tri_b[0], tri_b[2], tri_b[1]]
            faces.append(tri_a)
            faces.append(tri_b)
    return np.asarray(faces, dtype=int)


def make_rectangular_patch(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_level: float,
    nx: int,
    ny: int,
    normal_up: bool = True,
) -> TriMesh:
    """Flat rectangle in the xy-plane at fixed z, triangulated on a regular grid."""
    xs = np.linspace(x_range[0], x_range[1], nx + 1)
    ys = np.linspace(y_range[0], y_range[1], ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    verts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_level)])
    tris = _rect_grid_triangles(nx, ny, flip=not normal_up)
    return TriMesh(vertices=verts, triangles=tris)


def make_vertical_panel(
    axis: str,
    fixed_coord: float,
    tangent_range: tuple[float, float],
    z_range: tuple[float, float],
    nt: int,
    nz: int,
    outward_sign: int,
) -> TriMesh:
    """One vertical wall of a box.

    ``axis`` selects the normal axis: ``"x"`` means the panel lies in a plane
    of constant x (so its in-plane tangents are y and z); likewise for
    ``"y"``. ``outward_sign`` is ±1 and determines whether the triangle
    winding should place the normal along +axis or −axis.
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    ts = np.linspace(tangent_range[0], tangent_range[1], nt + 1)
    zs = np.linspace(z_range[0], z_range[1], nz + 1)
    T, Z = np.meshgrid(ts, zs, indexing="xy")

    if axis == "x":
        xs = np.full(T.size, fixed_coord)
        ys = T.ravel()
    else:  # axis == "y"
        xs = T.ravel()
        ys = np.full(T.size, fixed_coord)

    verts = np.column_stack([xs, ys, Z.ravel()])
    # Natural winding gives normal along +axis × (+z) = ± direction depending
    # on axis; simplest path is to build, compute, and flip to match outward_sign.
    tris = _rect_grid_triangles(nt, nz, flip=False)
    mesh = TriMesh(vertices=verts, triangles=tris)

    normals, _ = mesh.normals_and_areas()
    want = np.zeros(3)
    want[0 if axis == "x" else 1] = float(outward_sign)
    # flip globally if the first triangle points the wrong way
    if np.dot(normals[0], want) < 0:
        mesh.triangles = mesh.triangles[:, [0, 2, 1]]
    return mesh


def make_layered_box(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_interfaces: tuple[float, float],
    z_bottom: float,
    nx_top: int,
    ny_top: int,
    n_side_tangent: int,
    n_side_per_km: float,
    nx_interface: int,
    ny_interface: int,
) -> dict:
    """Build the complete closed-box mesh with two internal horizontal interfaces.

    Returns a dict with:
      ``"top"``        : TriMesh at z = 0, outward normal +ẑ
      ``"base"``       : TriMesh at z = z_bottom, outward normal −ẑ
      ``"sides"``      : dict {1, 2, 3} → TriMesh for each layer's side strips
                         (four-sided closed loop around the layer); outward
                         normals point away from the box interior
      ``"interfaces"`` : dict {1, 2} → TriMesh at z = z_interfaces[k],
                         normal +ẑ (into the shallower layer)

    ``z_interfaces`` is ``(z1, z2)`` with ``0 > z1 > z2 > z_bottom``.

    ``n_side_tangent`` is the horizontal discretisation of each side-wall
    strip. ``n_side_per_km`` controls the vertical discretisation — each
    layer's side strip uses ``ceil(thickness * n_side_per_km)`` divisions.
    """
    z1, z2 = z_interfaces
    if not (0 > z1 > z2 > z_bottom):
        raise ValueError("Expected 0 > z_interfaces[0] > z_interfaces[1] > z_bottom")

    top = make_rectangular_patch(x_range, y_range, 0.0, nx_top, ny_top, normal_up=True)
    base = make_rectangular_patch(
        x_range, y_range, z_bottom, nx_top, ny_top, normal_up=False
    )
    interface1 = make_rectangular_patch(
        x_range, y_range, z1, nx_interface, ny_interface, normal_up=True
    )
    interface2 = make_rectangular_patch(
        x_range, y_range, z2, nx_interface, ny_interface, normal_up=True
    )

    layer_bounds = {1: (z1, 0.0), 2: (z2, z1), 3: (z_bottom, z2)}
    sides: dict[int, TriMesh] = {}

    for layer, (z_lo, z_hi) in layer_bounds.items():
        thickness = z_hi - z_lo
        nz = max(1, int(np.ceil(thickness * n_side_per_km)))

        panels = [
            make_vertical_panel("x", x_range[1], y_range, (z_lo, z_hi), n_side_tangent, nz, +1),
            make_vertical_panel("x", x_range[0], y_range, (z_lo, z_hi), n_side_tangent, nz, -1),
            make_vertical_panel("y", y_range[1], x_range, (z_lo, z_hi), n_side_tangent, nz, +1),
            make_vertical_panel("y", y_range[0], x_range, (z_lo, z_hi), n_side_tangent, nz, -1),
        ]
        sides[layer] = _concatenate_meshes(panels)

    return {
        "top": top,
        "base": base,
        "sides": sides,
        "interfaces": {1: interface1, 2: interface2},
    }


def _concatenate_meshes(meshes: list[TriMesh]) -> TriMesh:
    """Concatenate a list of TriMeshes into one mesh (no vertex deduplication).

    Vertex deduplication is unnecessary for piecewise-constant BEM: each
    triangle contributes independently via its centroid, normal, and area.
    """
    verts_list = []
    tris_list = []
    offset = 0
    for m in meshes:
        verts_list.append(m.vertices)
        tris_list.append(m.triangles + offset)
        offset += m.n_vertices
    return TriMesh(
        vertices=np.vstack(verts_list),
        triangles=np.vstack(tris_list),
    )


def make_vertical_fault_xy(
    strike_length: float,
    depth_range: tuple[float, float],
    n_along: int,
    n_down: int,
    center_xy: tuple[float, float] = (0.0, 0.0),
    strike_azimuth_deg: float = 0.0,
) -> tuple[TriMesh, np.ndarray, np.ndarray]:
    """Vertical rectangular fault for a strike-slip earthquake.

    ``strike_azimuth_deg = 0`` means the fault strikes along +y (N–S).
    The returned normal ``n_hat`` is the unit vector perpendicular to the
    fault plane (x̂ for strike=0); ``slip_hat`` is the along-strike unit
    vector pointing in +strike direction (+ŷ for strike=0).

    ``depth_range = (z_bot, z_top)`` with ``z_bot < z_top ≤ 0``.
    """
    z_bot, z_top = depth_range
    if not (z_bot < z_top <= 0):
        raise ValueError("Expected z_bot < z_top ≤ 0 in depth_range")

    theta = np.radians(strike_azimuth_deg)
    s_hat = np.array([np.sin(theta), np.cos(theta), 0.0])
    n_hat = np.array([np.cos(theta), -np.sin(theta), 0.0])

    ss = np.linspace(-strike_length / 2, strike_length / 2, n_along + 1)
    zs = np.linspace(z_bot, z_top, n_down + 1)

    verts = np.empty(((n_along + 1) * (n_down + 1), 3))
    idx = 0
    for z in zs:
        for s in ss:
            verts[idx] = [
                center_xy[0] + s * s_hat[0],
                center_xy[1] + s * s_hat[1],
                z,
            ]
            idx += 1

    tris = _rect_grid_triangles(n_along, n_down, flip=False)
    mesh = TriMesh(vertices=verts, triangles=tris)

    # Ensure the fault's stored normal matches ``n_hat``: the BEM slip RHS
    # does not depend on it (T-kernel uses the source-element normal from
    # ``normals_and_areas``), but keeping them consistent avoids surprises.
    normals, _ = mesh.normals_and_areas()
    if np.dot(normals[0], n_hat) < 0:
        mesh.triangles = mesh.triangles[:, [0, 2, 1]]

    return mesh, n_hat, s_hat
