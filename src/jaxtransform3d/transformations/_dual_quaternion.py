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


def norm_dual_quaternion(dual_quat):
    """Normalize unit dual quaternion.

    A unit dual quaternion has a real quaternion with unit norm and an
    orthogonal real part. Both properties are enforced by multiplying a
    normalization factor [1]_. This is not always necessary. It is often
    sufficient to only enforce the unit norm property of the real quaternion.
    This can also be done with :func:`check_dual_quaternion`.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dual_quat_norm : array, shape (..., 8)
        Unit dual quaternion to represent transform with orthogonal real and
        dual quaternion.

    References
    ----------
    .. [1] enki (2023). Properly normalizing a dual quaternion.
       https://stackoverflow.com/a/76313524
    """
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    real = dual_quat[..., :4]
    dual = dual_quat[..., 4:]

    # 1. ensure unit norm of real quaternion
    real_norm = jnp.linalg.norm(real, axis=-1)[..., jnp.newaxis]
    real = real / real_norm
    dual = dual / real_norm

    # 2. ensure orthogonality of real and dual quaternion
    dual = dual - jnp.sum(real * dual, axis=-1)[..., jnp.newaxis] * real

    return jnp.concatenate((real, dual), axis=-1)


def dual_quaternion_squared_norm(dual_quat: ArrayLike) -> jax.Array:
    """Compute squared norm of dual quaternion.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    squared_norm : array, shape (..., 2)
        Squared norm of dual quaternion, which is a dual number with a real and
        a dual part.
    """
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    prod = compose_dual_quaternions(
        dual_quat, dual_quaternion_quaternion_conjugate(dual_quat)
    )

    real = prod[..., 0, jnp.newaxis]
    dual = prod[..., 4, jnp.newaxis]

    return jnp.concatenate((real, dual), axis=-1)


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
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    real = quaternion_conjugate(dual_quat[..., :4])
    dual = quaternion_conjugate(dual_quat[..., 4:])
    return jnp.concatenate((real, dual), axis=-1)


def apply_dual_quaternion(dual_quat: ArrayLike, v: ArrayLike) -> jax.Array:
    r"""Apply transform represented by a dual quaternion to a vector.

    To apply the transformation defined by a unit dual quaternion
    :math:`\boldsymbol{q}` to a point :math:`\boldsymbol{v} \in \mathbb{R}^3`,
    we first represent the vector as a dual quaternion: we set the real part to
    (1, 0, 0, 0) and the dual part is a pure quaternion with the scalar part
    0 and the vector as its vector part
    :math:`\left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right) \in
    \mathbb{R}^4`. Then we left-multiply the dual quaternion and right-multiply
    its dual quaternion conjugate

    .. math::

        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{w}\end{array}\right)
        =
        \boldsymbol{q}
        \cdot
        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{v}\end{array}\right)
        \cdot
        \boldsymbol{q}^*.

    The vector part of the dual part :math:`\boldsymbol{w}` of the resulting
    quaternion is the rotated point.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    v : array-like, shape (..., 3)
        3d vector

    Returns
    -------
    w : array, shape (..., 3)
        3d vector
    """
    dual_quat = jnp.asarray(dual_quat)
    v = jnp.asarray(v)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)
    chex.assert_axis_dimension(v, axis=-1, expected=3)
    chex.assert_equal_shape_prefix((dual_quat, v), prefix_len=v.ndim - 1)

    pure_dual_quat = jnp.concatenate(
        (
            jnp.ones(v.shape[:-1] + (1,)),
            jnp.zeros(v.shape[:-1] + (4,)),
            v,
        ),
        axis=-1,
    )
    dual_quat_conj = jnp.concatenate(
        (
            dual_quat[..., 0, jnp.newaxis],
            -dual_quat[..., 1:5],
            dual_quat[..., 5:],
        ),
        axis=-1,
    )
    transformed_pure_dual_quat = compose_dual_quaternions(
        dual_quat, compose_dual_quaternions(pure_dual_quat, dual_quat_conj)
    )
    return transformed_pure_dual_quat[..., 5:]


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
