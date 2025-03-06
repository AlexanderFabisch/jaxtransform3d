import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import (
    compact_axis_angle_from_quaternion,
    compose_quaternions,
    quaternion_conjugate,
)
from ..utils import norm_vector
from ._transform import _v


def exponential_coordinates_from_dual_quaternion(dual_quat: ArrayLike) -> jax.Array:
    """Compute dual quaternion from exponential coordinates.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Dual quaternion.

    Returns
    -------
    exp_coords : array, shape (..., 6)
        Exponential coordinates.
    """
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    real = dual_quat[..., :4]
    dual = dual_quat[..., 4:]

    t = 2.0 * compose_quaternions(dual, quaternion_conjugate(real))[..., 1:]
    axis_angle = compact_axis_angle_from_quaternion(real)

    angle = jnp.linalg.norm(axis_angle, axis=-1)
    axis = norm_vector(axis_angle, norm=angle)
    v = _v(axis, angle, t)

    return jnp.concatenate((axis_angle, v), axis=-1)
