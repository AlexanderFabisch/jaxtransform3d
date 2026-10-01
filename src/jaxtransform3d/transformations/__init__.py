r"""
Proper Rigid Transformations in 3D
----------------------------------

The group of all proper rigid transformations (rototranslations) in
3D Cartesian space is :math:`SE(3)` (SE: special Euclidean group).
Transformations consist of a rotation and a translation. Those can be
represented in different ways just like rotations can be expressed
in different ways. The minimum number of components that are required to
describe any transformation from :math:`SE(3)` is 6.

We support transformation matrices, unit dual quaternions, and exponential
coordinates of transformation.
"""

from ._dual_quaternion import (
    apply_dual_quaternion,
    compose_dual_quaternions,
    dual_quaternion_quaternion_conjugate,
    dual_quaternion_squared_norm,
    exponential_coordinates_from_dual_quaternion,
    norm_dual_quaternion,
)
from ._exp_coords import (
    dual_quaternion_from_exponential_coordinates,
    transform_from_exponential_coordinates,
)
from ._transform import (
    adjoint_from_transform,
    apply_transform,
    compose_transforms,
    create_transform,
    exponential_coordinates_from_transform,
    transform_inverse,
)

__all__ = [
    "transform_inverse",
    "adjoint_from_transform",
    "apply_transform",
    "compose_transforms",
    "create_transform",
    "exponential_coordinates_from_transform",
    "transform_from_exponential_coordinates",
    "dual_quaternion_from_exponential_coordinates",
    "norm_dual_quaternion",
    "dual_quaternion_squared_norm",
    "compose_dual_quaternions",
    "dual_quaternion_quaternion_conjugate",
    "apply_dual_quaternion",
    "exponential_coordinates_from_dual_quaternion",
]
