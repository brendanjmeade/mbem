"""VTK ImageData (``.vti``) writer and reader: a volume a viewer can open.

Why hand-rolled. The tree's only output format is ``.npz``, which no viewer
reads, and a regular grid wants ImageData specifically -- it is the one VTK
dataset vtk.js volume-renders directly, and it stores no coordinates, so a
500k-point volume with nine fields is ~20 MB instead of the ~8x that a
hexahedral unstructured mesh would spend on connectivity. ``meshio`` is
installed but writes ``vtk``/``vtu`` only, never ImageData, and ``vtk`` itself
is a ~100 MB wheel to add for what is an XML header plus a length-prefixed
binary block.

The format: a little-endian ``VTKFile`` of type ``ImageData`` with
``appended``/``raw`` arrays. Each array in the appended section is a uint64
byte count followed by that many bytes, at the offset its ``DataArray`` header
declares. Point data is x-fastest, matching C order on ``(nz, ny, nx)``.

Written float32 (viewers render single precision; the solve stays float64) and
read back float32, so ``read`` of a ``write`` is bitwise -- which is what lets
the gate prove the round trip without depending on VTK.
"""
from __future__ import annotations

import pathlib
import re
import struct

import numpy as np

_HEADER_T = "<Q"            # appended-data length prefix: little-endian uint64


def write(path, origin, spacing, dims, arrays: dict) -> pathlib.Path:
    """Write ``arrays`` on a regular grid to ``path`` as ImageData.

    ``dims`` is ``(nx, ny, nz)`` POINT counts; ``arrays`` maps a name to an
    array whose leading axes are ``(nz, ny, nx)`` -- so ``(nz, ny, nx)`` is a
    scalar field and ``(nz, ny, nx, 3)`` a vector. Everything is stored
    float32.
    """
    path = pathlib.Path(path)
    nx, ny, nz = (int(d) for d in dims)
    flat, headers, offset = [], [], 0
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.shape[:3] != (nz, ny, nx):
            raise ValueError(f"{name!r} has leading shape {a.shape[:3]}, "
                             f"expected (nz, ny, nx) = {(nz, ny, nx)}")
        ncomp = 1 if a.ndim == 3 else int(np.prod(a.shape[3:]))
        buf = np.ascontiguousarray(a, dtype="<f4").reshape(-1)
        headers.append((name, ncomp, offset))
        flat.append(struct.pack(_HEADER_T, buf.nbytes))
        flat.append(buf.tobytes())
        offset += struct.calcsize(_HEADER_T) + buf.nbytes

    point_data = "\n".join(
        f'        <DataArray type="Float32" Name="{n}" '
        f'NumberOfComponents="{c}" format="appended" offset="{o}"/>'
        for n, c, o in headers)
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64">\n'
        f'  <ImageData WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}" '
        f'Origin="{origin[0]!r} {origin[1]!r} {origin[2]!r}" '
        f'Spacing="{spacing[0]!r} {spacing[1]!r} {spacing[2]!r}">\n'
        f'    <Piece Extent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}">\n'
        '      <PointData>\n'
        f'{point_data}\n'
        '      </PointData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '  <AppendedData encoding="raw">\n   _')
    with open(path, "wb") as fh:
        fh.write(xml.encode())
        for chunk in flat:
            fh.write(chunk)
        fh.write(b"\n  </AppendedData>\n</VTKFile>\n")
    return path


def read(path) -> dict:
    """Inverse of :func:`write`: ``{origin, spacing, dims, arrays}``.

    Only rich enough to read what ``write`` produced -- it exists so the gate
    can prove the round trip without a VTK dependency, not to read arbitrary
    ImageData.
    """
    raw = pathlib.Path(path).read_bytes()
    head, _, rest = raw.partition(b'<AppendedData encoding="raw">\n   _')
    text = head.decode()
    ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', text)
           .group(1).split()]
    dims = (ext[1] + 1, ext[3] + 1, ext[5] + 1)
    origin = tuple(float(v) for v in
                   re.search(r'Origin="([^"]+)"', text).group(1).split())
    spacing = tuple(float(v) for v in
                    re.search(r'Spacing="([^"]+)"', text).group(1).split())
    nx, ny, nz = dims
    arrays = {}
    for m in re.finditer(r'<DataArray type="Float32" Name="([^"]+)" '
                         r'NumberOfComponents="(\d+)" format="appended" '
                         r'offset="(\d+)"/>', text):
        name, ncomp, off = m.group(1), int(m.group(2)), int(m.group(3))
        (n_bytes,) = struct.unpack_from(_HEADER_T, rest, off)
        start = off + struct.calcsize(_HEADER_T)
        a = np.frombuffer(rest, dtype="<f4", count=n_bytes // 4, offset=start)
        arrays[name] = a.reshape((nz, ny, nx) if ncomp == 1
                                 else (nz, ny, nx, ncomp))
    return {"origin": origin, "spacing": spacing, "dims": dims,
            "arrays": arrays}
