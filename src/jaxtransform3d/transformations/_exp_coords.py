import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import (
    compose_quaternions,
    left_jacobian_SO3,
    matrix_from_compact_axis_angle,
    quaternion_from_compact_axis_angle,
)
from ._transform import create_transform


def transform_from_exponential_coordinates(exp_coords: ArrayLike) -> jax.Array:
    r"""Compute transformation matrix from exponential coordinates.

    This is the exponential map.

    .. math::

        Exp: \mathcal{S} \theta \in \mathbb{R}^6
        \rightarrow \boldsymbol{T} \in SE(3)

    .. math::

        Exp(\mathcal{S}\theta) =
        Exp\left(\left(\begin{array}{c}
        \hat{\boldsymbol{\omega}}\\
        \boldsymbol{v}
        \end{array}\right)\theta\right)
        =
        \exp(\left[\mathcal{S}\right] \theta)
        =
        \left(\begin{array}{cc}
        Exp(\hat{\boldsymbol{\omega}} \theta) &
        \boldsymbol{J}(\theta)\boldsymbol{v}\theta\\
        \boldsymbol{0} & 1
        \end{array}\right),

    where :math:`\boldsymbol{J}(\theta)` is the left Jacobian of :math:`SO(3)`.

    Parameters
    ----------
    exp_coords : array-like, shape (..., 6)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    Returns
    -------
    T : array, shape (..., 4, 4)
        Transformation matrix.
    """
    exp_coords = jnp.asarray(exp_coords)

    chex.assert_axis_dimension(exp_coords, axis=-1, expected=6)

    axis_angle = exp_coords[..., :3]
    v_theta = exp_coords[..., 3:]

    R = matrix_from_compact_axis_angle(axis_angle=axis_angle)

    J = left_jacobian_SO3(axis_angle)
    t = (J @ v_theta[..., jnp.newaxis])[..., 0]
    angle = jnp.linalg.norm(exp_coords[..., :3], axis=-1)[..., jnp.newaxis]
    t = jnp.where(angle < jnp.finfo(angle.dtype).eps, v_theta, t)

    return create_transform(R, t)


def dual_quaternion_from_exponential_coordinates(exp_coords: ArrayLike) -> jax.Array:
    """Compute dual quaternion from exponential coordinates.

    Parameters
    ----------
    exp_coords : array-like, shape (..., 6)
        Exponential coordinates.

    Returns
    -------
    dual_quat : array, shape (..., 8)
        Dual quaternion
    """
    exp_coords = jnp.asarray(exp_coords)

    chex.assert_axis_dimension(exp_coords, axis=-1, expected=6)

    axis_angle = exp_coords[..., :3]
    v_theta = exp_coords[..., 3:]

    real = quaternion_from_compact_axis_angle(axis_angle=axis_angle)

    J = left_jacobian_SO3(axis_angle)
    t = (J @ v_theta[..., jnp.newaxis])[..., 0]
    angle = jnp.linalg.norm(exp_coords[..., :3], axis=-1)[..., jnp.newaxis]
    t = jnp.where(angle < jnp.finfo(angle.dtype).eps, v_theta, t)

    t_quat = jnp.concatenate((jnp.zeros_like(t[..., :1]), t), axis=-1)
    dual = 0.5 * compose_quaternions(t_quat, real)

    return jnp.concatenate((real, dual), axis=-1)
