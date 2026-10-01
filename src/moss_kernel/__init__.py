"""The INDEPENDENT copy of the mollified kernels, installed as ``moss_kernel``.

This directory and ``src/mollified_kernel/`` are two different
implementations of the same analytic kernels, and the difference is the
point: the gates compare them entrywise, so neither may be "fixed" to agree
with the other. ``analytical_kernels.py`` here is 51,562 B against msd's
49,069 B, and it alone defines ``analytical_eigenstress_kernel`` and
``eigenstress_batch`` -- the only eigenstress oracle in the tree, which
``verify_eigenstress_exact`` clauses [a] and [f] require.

It is packaged under a DIFFERENT name (``moss_kernel``, see the repo
``pyproject.toml``) rather than as a second ``mollified_kernel``, because two
packages sharing one name cannot both be imported: one would shadow the
other, and the parity check would silently compare a copy against itself --
a gate that still prints PASS while measuring nothing. The modules here
therefore spell their intra-package imports ``moss_kernel.``, with the flat
fallback kept for running a module directly from this directory.
"""
