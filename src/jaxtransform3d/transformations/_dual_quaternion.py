import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import (
    compact_axis_angle_from_quaternion,
    compose_quaternions,
    left_jacobian_SO3_inv,
    quaternion_conjugate,
)


def norm_dual_quaternion(dual_quat):
    r"""Normalize unit dual quaternion.

    A unit dual quaternion has a real quaternion with unit norm and an
    orthogonal real part. Both properties are enforced by this function.

    We use a two-step normalization procedure (see [2]_ for details):

    1. Multiply the dual quaternion :math:`\sigma = p + q \epsilon` with the
       normalization factor :math:`\frac{1}{\sqrt{s}} = \frac{1}{\sqrt{p^T p}}`
       for the real quaternion to obtain :math:`\sigma' = \frac{\sigma}{\sqrt{s}}`,
       which ensures that the norm is :math:`1 + t' \epsilon`.
    2. Multiply :math:`\sigma' = p' + q' \epsilon` with the normalization factor
       :math:`(1 - \frac{t'}{2} \epsilon)` for the dual quaternion (since s' = 1)
       to obtain :math:`\sigma'' = p' + q' \epsilon - p'^T q' p' \epsilon`.

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

    See Also
    --------
    dual_quaternion_squared_norm
        Computes the squared norm of a dual quaternion.

    References
    ----------
    .. [1] enki (2023). Properly normalizing a dual quaternion.
       https://stackoverflow.com/a/76313524
    .. [2] Alexander Fabisch (2025). Normalizing dual quaternions.
       https://alexanderfabisch.github.io/normalizing-dual-quaternions.html
    """
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    real = dual_quat[..., :4]
    dual = dual_quat[..., 4:]

    # 1. ensure valid real quaternion
    real_norm = jnp.linalg.norm(real, axis=-1)[..., jnp.newaxis]
    invalid_real = real_norm == 0.0
    identity = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=dual_quat.dtype)
    real = jnp.where(invalid_real, identity, real)
    real_norm = jnp.where(invalid_real, 1.0, real_norm)

    # 2. ensure unit norm of real quaternion
    real = real / real_norm
    dual = dual / real_norm

    # 3. ensure orthogonality of real and dual quaternion
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
    r"""Concatenate dual quaternions.

    Computes the dual quaternion product

    .. math::

        \begin{eqnarray}
        \sigma_1 \sigma_2 &=& (p_1 + q_1 \epsilon) (p_2 + q_2 \epsilon)\\
        &=& p_1 p_2 + (p_1 q_2 + q_1 p_2) \epsilon.
        \end{eqnarray}

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
    :math:`\sigma` to a point :math:`\boldsymbol{v} \in \mathbb{R}^3`,
    we first represent the vector as a dual quaternion: we set the real part to
    (1, 0, 0, 0) and the dual part is a pure quaternion with the scalar part
    0 and the vector as its vector part
    :math:`\left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right) \in
    \mathbb{R}^4`. Then we left-multiply the dual quaternion and right-multiply
    its dual quaternion conjugate

    .. math::

        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{w}\end{array}\right)
        =
        \sigma
        \cdot
        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{v}\end{array}\right)
        \cdot
        \sigma^*.

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

    This operation is called logarithmic map.

    Parameters
    ----------
    dual_quat : array-like, shape (..., 8)
        Dual quaternion.

    Returns
    -------
    exp_coords : array, shape (..., 6)
        Exponential coordinates.

    See also
    --------
    dual_quaternion_from_exponential_coordinates : Exponential map.
    exponential_coordinates_from_transform
        Logarithmic map for transformation matrices.
    """
    dual_quat = jnp.asarray(dual_quat)

    chex.assert_axis_dimension(dual_quat, axis=-1, expected=8)

    real = dual_quat[..., :4]
    dual = dual_quat[..., 4:]

    axis_angle = compact_axis_angle_from_quaternion(real)

    t = 2.0 * compose_quaternions(dual, quaternion_conjugate(real))[..., 1:]
    v_theta = (left_jacobian_SO3_inv(axis_angle) @ t[..., jnp.newaxis])[..., 0]

    return jnp.concatenate((axis_angle, v_theta), axis=-1)
