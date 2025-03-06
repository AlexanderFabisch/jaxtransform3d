"""Proper rigid transformations in 3D."""

from ._dual_quaternion import (
    apply_dual_quaternion,
    compose_dual_quaternions,
    dual_quaternion_norm,
    dual_quaternion_quaternion_conjugate,
    exponential_coordinates_from_dual_quaternion,
    norm_dual_quaternion,
)
from ._exp_coords import (
    dual_quaternion_from_exponential_coordinates,
    transform_from_exponential_coordinates,
)
from ._transform import (
    apply_transform,
    compose_transforms,
    create_transform,
    exponential_coordinates_from_transform,
    transform_inverse,
)

__all__ = [
    "transform_inverse",
    "apply_transform",
    "compose_transforms",
    "create_transform",
    "exponential_coordinates_from_transform",
    "transform_from_exponential_coordinates",
    "dual_quaternion_from_exponential_coordinates",
    "norm_dual_quaternion",
    "dual_quaternion_norm",
    "compose_dual_quaternions",
    "dual_quaternion_quaternion_conjugate",
    "apply_dual_quaternion",
    "exponential_coordinates_from_dual_quaternion",
]
