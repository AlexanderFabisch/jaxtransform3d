"""3D rotations."""

from ._axis_angle import (
    matrix_from_compact_axis_angle,
    quaternion_from_compact_axis_angle,
)
from ._matrix import (
    apply_matrix,
    compact_axis_angle_from_matrix,
    compose_matrices,
    matrix_inverse,
)
from ._polar_decomp import norm_matrix, robust_polar_decomposition
from ._quaternion import (
    apply_quaternion,
    compact_axis_angle_from_quaternion,
    compose_quaternions,
    quaternion_conjugate,
)

__all__ = [
    "matrix_from_compact_axis_angle",
    "quaternion_from_compact_axis_angle",
    "matrix_inverse",
    "apply_matrix",
    "compose_matrices",
    "compact_axis_angle_from_matrix",
    "compose_quaternions",
    "quaternion_conjugate",
    "apply_quaternion",
    "compact_axis_angle_from_quaternion",
    "norm_matrix",
    "robust_polar_decomposition",
]
