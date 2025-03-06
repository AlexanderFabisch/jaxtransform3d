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


def compose_dual_quaternions(dual_quat1: ArrayLike, dual_quat2: ArrayLike) -> jax.Array:
    """Concatenate dual quaternions.

    Suppose we want to apply two extrinsic transforms given by dual
    quaternions dq1 and dq2 to a vector v. We can either apply dq2 to v and
    then dq1 to the result or we can concatenate dq1 and dq2 and apply the
    result to v.

    Parameters
    ----------
    dual_quat1 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    dual_quat2 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dual_quat3 : array, shape (8,)
        Products of the two batches of dual quaternions:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    dual_quat1 = jnp.asarray(dual_quat1)
    dual_quat2 = jnp.asarray(dual_quat2)

    chex.assert_equal_shape([dual_quat1, dual_quat2])

    real1 = dual_quat1[..., :4]
    dual1 = dual_quat1[..., 4:]
    real2 = dual_quat2[..., :4]
    dual2 = dual_quat2[..., 4:]

    real = compose_quaternions(real1, real2)
    dual = compose_quaternions(real1, dual2) + compose_quaternions(dual1, real2)

    return jnp.concatenate((real, dual), axis=-1)


def dual_quaternion_quaternion_conjugate(dual_quat: ArrayLike) -> jax.Array:
    """Quaternion conjugate of dual quaternions.

    For unit dual quaternions that represent transformations,
    this function is equivalent to the inverse of the
    corresponding transformation matrix.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dual_quat2 : array, shape (..., 8)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)
    """
    real = quaternion_conjugate(dual_quat[..., :4])
    dual = quaternion_conjugate(dual_quat[..., 4:])
    return jnp.concatenate((real, dual), axis=-1)


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
