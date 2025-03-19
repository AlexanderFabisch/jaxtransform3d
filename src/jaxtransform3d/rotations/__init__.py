r"""
Rotations in 3D
---------------

The group of all rotations in the 3D Cartesian space is called :math:`SO(3)`
(SO: special orthogonal group). It is typically represented by 3D rotation
matrices. The minimum number of components that are required to describe
any rotation from :math:`SO(3)` is 3. However, there is no representation that
is non-redundant, continuous, and free of singularities.
"""

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
    norm_quaternion,
    quaternion_conjugate,
)
from ._jacobians import (
    left_jacobian_SO3,
    left_jacobian_SO3_series,
    left_jacobian_SO3_inv,
    left_jacobian_SO3_inv_series,
)

__all__ = [
    "matrix_from_compact_axis_angle",
    "quaternion_from_compact_axis_angle",
    "matrix_inverse",
    "apply_matrix",
    "compose_matrices",
    "compact_axis_angle_from_matrix",
    "norm_quaternion",
    "compose_quaternions",
    "quaternion_conjugate",
    "apply_quaternion",
    "compact_axis_angle_from_quaternion",
    "norm_matrix",
    "robust_polar_decomposition",
    "left_jacobian_SO3",
    "left_jacobian_SO3_series",
    "left_jacobian_SO3_inv",
    "left_jacobian_SO3_inv_series",
]
