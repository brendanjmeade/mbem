# Kernel identifiers, defined once. Every BlockTerm, PairCompressed and
# evaluation helper names its kernel with one of these; a dispatch on any
# other string raises instead of silently taking a branch.
KERNEL_T = "H"     # T-kernel: displacement jump (slip / u_p) -> displacement
KERNEL_U = "G"     # U-kernel: traction -> displacement
KERNEL_HT = "Ht"   # displacement jump -> ELASTIC traction at a point (eigenstress
                   # removed): the rows of a traction-BC fault
KERNEL_GT = "Gt"   # traction -> traction at a point (Kelvin single layer)
KERNELS = (KERNEL_T, KERNEL_U, KERNEL_HT, KERNEL_GT)
# The two displacement kernels have material-independent basis stacks; the
# traction kernels are assembled directly (dense "direct" mode only).
DISPLACEMENT_KERNELS = (KERNEL_T, KERNEL_U)

from .basis import (  # noqa: E402
    UBasis,
    TBasis,
    assemble_u_basis,
    assemble_t_basis,
    assemble_u_matrix,
    assemble_t_matrix,
    assemble_traction_matrix,
    condense_columns,
    identity_ids,
    u_coeffs,
    t_coeffs,
    n_nodes,
)


def kernel_n_basis(kernel: str) -> int:
    """Basis count of a kernel (6 for T, 3 for U); ValueError otherwise --
    the traction kernels have no basis stack, which is what keeps them out of
    basis mode and every compressed path."""
    if kernel == KERNEL_T:
        return 6
    if kernel == KERNEL_U:
        return 3
    if kernel in (KERNEL_HT, KERNEL_GT):
        raise ValueError(f"kernel {kernel!r} (a traction kernel) has no basis "
                         f"stack: dense direct assembly only")
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")


def kernel_coeffs(kernel: str, material):
    """Material coefficient vector of a kernel; ValueError otherwise."""
    if kernel == KERNEL_T:
        return t_coeffs(material.mu, material.lam)
    if kernel == KERNEL_U:
        return u_coeffs(material.mu, material.lam)
    if kernel in (KERNEL_HT, KERNEL_GT):
        raise ValueError(f"kernel {kernel!r} (a traction kernel) has no "
                         f"coefficient vector: material applied in-loop")
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {KERNELS}")


__all__ = [
    "KERNEL_T", "KERNEL_U", "KERNEL_HT", "KERNEL_GT", "KERNELS",
    "DISPLACEMENT_KERNELS", "kernel_n_basis", "kernel_coeffs",
    "UBasis", "TBasis",
    "assemble_u_basis", "assemble_t_basis",
    "assemble_u_matrix", "assemble_t_matrix", "assemble_traction_matrix",
    "condense_columns", "identity_ids",
    "u_coeffs", "t_coeffs", "n_nodes",
]
