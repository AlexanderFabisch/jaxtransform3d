import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import (
    apply_matrix,
    compact_axis_angle_from_matrix,
    left_jacobian_SO3_inv,
    matrix_inverse,
)


def transform_inverse(T: ArrayLike) -> jax.Array:
    r"""Invert transformation matrix.

    .. math::

        \boldsymbol{T}^{-1}
        =
        \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{t}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)^{-1}
        =
        \left(
        \begin{array}{cc}
        \boldsymbol{R}^T & -\boldsymbol{R}^T \boldsymbol{t}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)

    Parameters
    ----------
    T : array-like, shape (..., 4, 4)
        Transformation matrix.

    Returns
    -------
    T_inv : array, shape (..., 4, 4)
        Inverted transformation matrix.
    """
    T = jnp.asarray(T)
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    R_inv = matrix_inverse(R)
    t_inv = -apply_matrix(R_inv, t)

    return create_transform(R_inv, t_inv)


def apply_transform(T: ArrayLike, v: ArrayLike) -> jax.Array:
    r"""Apply transformation matrix to vector.

    .. math::

        \boldsymbol{w} = \boldsymbol{R} \boldsymbol{v} + \boldsymbol{t}

    Parameters
    ----------
    T : array-like, shape (..., 4, 4) or (4, 4)
        Transformation matrix.

    v : array-like, shape (..., 3) or (3,)
        3d vector.

    Returns
    -------
    w : array, shape (..., 3) or (3,)
        3d vector.
    """
    T = jnp.asarray(T)
    v = jnp.asarray(v)
    if not jnp.issubdtype(T.dtype, jnp.floating):
        T = T.astype(jnp.float64)
    if not jnp.issubdtype(v.dtype, jnp.floating):
        v = v.astype(jnp.float64)

    chex.assert_axis_dimension(T, axis=-2, expected=4)
    chex.assert_axis_dimension(T, axis=-1, expected=4)
    chex.assert_axis_dimension(v, axis=-1, expected=3)

    return apply_matrix(T[..., :3, :3], v) + T[..., :3, 3]


def compose_transforms(T1: ArrayLike, T2: ArrayLike) -> jax.Array:
    """Compose transformation matrices.

    Parameters
    ----------
    T1 : array-like, shape (..., 4, 4) or (4, 4)
        Transformation matrix.

    T2 : array-like, shape (..., 4, 4) or (4, 4)
        Transformation matrix.

    Returns
    -------
    T1_T2 : array, shape (..., 4, 4) or (4, 4)
        Composed transformation matrix.
    """
    T1 = jnp.asarray(T1)
    T2 = jnp.asarray(T2)
    bigger_shape = T1.shape if T1.size > T2.size else T2.shape
    return (T1.reshape(-1, 4, 4) @ T2.reshape(-1, 4, 4)).reshape(bigger_shape)


def create_transform(R: ArrayLike, t: ArrayLike) -> jax.Array:
    r"""Make transformation from rotation matrix and translation.

    .. math::

        \boldsymbol{T} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{t}\\
        \boldsymbol{0} & 1
        \end{array}
        \right) \in SE(3)

    Parameters
    ----------
    R : array-like, shape (..., 3, 3)
        Rotation matrix.

    t : array-like, shape (..., 3)
        Translation.

    Returns
    -------
    T : array, shape (..., 4, 4)
        Transformation matrix.
    """
    R = jnp.asarray(R)
    t = jnp.asarray(t)

    chex.assert_equal_shape_prefix((R, t), prefix_len=R.ndim - 1)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    T = jnp.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype)
    T = T.at[..., :3, :3].set(R)
    T = T.at[..., :3, 3].set(t)
    T = T.at[..., 3, 3].set(1)
    return T


def exponential_coordinates_from_transform(T: ArrayLike) -> jax.Array:
    r"""Compute exponential coordinates from transformation matrix.

    This is the logarithm map.

    .. math::

        Log: \boldsymbol{T} \in SE(3)
        \rightarrow \mathcal{S} \theta \in \mathbb{R}^6

    .. math::

        Log(\boldsymbol{T}) =
        Log\left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        Log(\boldsymbol{R})\\
        \boldsymbol{J}^{-1}(\theta) \boldsymbol{p}
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        \hat{\boldsymbol{\omega}}\\
        \boldsymbol{v}
        \end{array}
        \right)
        \theta
        =
        \mathcal{S}\theta,

    where :math:`\boldsymbol{J}^{-1}(\theta)` is the inverse left Jacobian of
    :math:`SO(3)`.

    Parameters
    ----------
    T : array-like, shape (..., 4, 4)
        Transformation matrix.

    Returns
    -------
    exp_coords : array, shape (..., 6)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    See also
    --------
    transform_from_exponential_coordinates : Exponential map.
    exponential_coordinates_from_dual_quaternion
        Logarithmic map for dual quaternions.
    """
    T = jnp.asarray(T)

    chex.assert_axis_dimension(T, axis=-2, expected=4)
    chex.assert_axis_dimension(T, axis=-1, expected=4)

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    axis_angle = compact_axis_angle_from_matrix(R)
    v_theta = (left_jacobian_SO3_inv(axis_angle) @ t[..., jnp.newaxis])[..., 0]

    return jnp.concatenate((axis_angle, v_theta), axis=-1)
