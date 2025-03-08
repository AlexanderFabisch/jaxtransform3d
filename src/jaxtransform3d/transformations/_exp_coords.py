import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import (
    compose_quaternions,
    matrix_from_compact_axis_angle,
    quaternion_from_compact_axis_angle,
)
from ..utils import norm_vector
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

    angle = jnp.linalg.norm(exp_coords[..., :3], axis=-1)
    screw_axis = norm_vector(exp_coords, norm=angle)

    R = matrix_from_compact_axis_angle(
        axis_angle=exp_coords[..., :3], axis=screw_axis[..., :3], angle=angle
    )

    t = _translation_from_exp_coords(exp_coords, screw_axis, angle)

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

    angle = jnp.linalg.norm(exp_coords[..., :3], axis=-1)
    screw_axes = norm_vector(exp_coords, norm=angle)

    real = quaternion_from_compact_axis_angle(axis_angle=exp_coords[..., :3])

    t = _translation_from_exp_coords(exp_coords, screw_axes, angle)
    t = jnp.concatenate((jnp.zeros_like(t[..., :1]), t), axis=-1)
    dual = 0.5 * compose_quaternions(t, real)

    return jnp.concatenate((real, dual), axis=-1)


def _translation_from_exp_coords(
    exp_coords: jax.Array, screw_axis: jax.Array, angle: jax.Array
) -> jax.Array:
    tms = angle - jnp.sin(angle)
    cm1 = jnp.cos(angle) - 1.0
    o0 = screw_axis[..., 0]
    o1 = screw_axis[..., 1]
    o2 = screw_axis[..., 2]
    v0 = screw_axis[..., 3]
    v1 = screw_axis[..., 4]
    v2 = screw_axis[..., 5]
    o01tms = o0 * o1 * tms
    o12tms = o1 * o2 * tms
    o02tms = o0 * o2 * tms
    o0cm1 = o0 * cm1
    o1cm1 = o1 * cm1
    o2cm1 = o2 * cm1
    o00tms = o0 * o0 * tms
    o11tms = o1 * o1 * tms
    o22tms = o2 * o2 * tms
    t = jnp.stack(
        (
            -v0 * (o11tms + o22tms - angle)
            + v1 * (o01tms + o2cm1)
            + v2 * (o02tms - o1cm1),
            v0 * (o01tms - o2cm1)
            - v1 * (o00tms + o22tms - angle)
            + v2 * (o0cm1 + o12tms),
            v0 * (o02tms + o1cm1)
            - v1 * (o0cm1 - o12tms)
            - v2 * (o00tms + o11tms - angle),
        ),
        axis=-1,
    )
    t = jnp.where(angle[..., jnp.newaxis] != 0.0, t, exp_coords[..., 3:])
    return t
