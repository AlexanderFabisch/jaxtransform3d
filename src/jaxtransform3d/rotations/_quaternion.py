import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import norm_angle, norm_vector


def norm_quaternion(q: ArrayLike) -> jax.Array:
    r"""Normalize quaternion to unit norm.

    Parameters
    ----------
    q : array-like, shape (..., 4)
        Quaternion to normalize.

    Returns
    -------
    q_norm : array-like, shape (..., 4)
        Normalized quaternion.
    """
    q = jnp.asarray(q)
    if not jnp.issubdtype(q.dtype, jnp.floating):
        q = q.astype(jnp.float64)

    chex.assert_axis_dimension(q, axis=-1, expected=4)

    return norm_vector(q)


def compose_quaternions(q1: ArrayLike, q2: ArrayLike) -> jax.Array:
    r"""Compose two quaternions.

    We concatenate two quaternions by quaternion multiplication
    :math:`\boldsymbol{q}_1\boldsymbol{q}_2`.

    We use Hamilton's quaternion multiplication.

    If the two quaternions are divided up into scalar part and vector part
    each, i.e.,
    :math:`\boldsymbol{q} = (w, \boldsymbol{v}), w \in \mathbb{R},
    \boldsymbol{v} \in \mathbb{R}^3`, then the quaternion product is

    .. math::

        \boldsymbol{q}_{12} =
        (w_1 w_2 - \boldsymbol{v}_1 \cdot \boldsymbol{v}_2,
        w_1 \boldsymbol{v}_2 + w_2 \boldsymbol{v}_1
        + \boldsymbol{v}_1 \times \boldsymbol{v}_2)

    with the scalar product :math:`\cdot` and the cross product :math:`\times`.

    Parameters
    ----------
    q1 : array-like, shape (..., 4)
        First quaternion

    q2 : array-like, shape (..., 4)
        Second quaternion

    Returns
    -------
    q12 : array, shape (..., 4)
        Quaternion that represents the concatenated rotation q1 * q2
    """
    q1 = jnp.asarray(q1)
    if not jnp.issubdtype(q1.dtype, jnp.floating):
        q1 = q1.astype(jnp.float64)
    q2 = jnp.asarray(q2)
    if not jnp.issubdtype(q2.dtype, jnp.floating):
        q2 = q2.astype(jnp.float64)

    chex.assert_equal_shape((q1, q2))
    chex.assert_axis_dimension(q1, axis=-1, expected=4)
    chex.assert_axis_dimension(q2, axis=-1, expected=4)

    vec1 = q1[..., 1:]
    vec2 = q2[..., 1:]
    real = q1[..., 0] * q2[..., 0] - jnp.sum(vec1 * vec2, axis=-1)
    vec = (
        q1[..., 0, jnp.newaxis] * vec2
        + q2[..., 0, jnp.newaxis] * vec1
        + jnp.cross(vec1.reshape(-1, 3), vec2.reshape(-1, 3)).reshape(vec1.shape)
    )
    return jnp.concatenate((real[..., jnp.newaxis], vec), axis=-1)


def quaternion_conjugate(q: ArrayLike) -> jax.Array:
    r"""Conjugate of quaternion.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion.

    The conjugate of a quaternion :math:`\boldsymbol{q}` is often denoted as
    :math:`\boldsymbol{q}^*`. For a quaternion :math:`\boldsymbol{q} = w
    + x \boldsymbol{i} + y \boldsymbol{j} + z \boldsymbol{k}` it is defined as

    .. math::

        \boldsymbol{q}^* = w - x \boldsymbol{i} - y \boldsymbol{j}
        - z \boldsymbol{k}.

    Parameters
    ----------
    q : array-like, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z).

    Returns
    -------
    q_c : array-like, shape (..., 4)
        Conjugate (w, -x, -y, -z).
    """
    q = jnp.asarray(q)
    if not jnp.issubdtype(q.dtype, jnp.floating):
        q = q.astype(jnp.float64)

    chex.assert_axis_dimension(q, axis=-1, expected=4)

    return jnp.concatenate((q[..., 0, jnp.newaxis], -q[..., 1:]), axis=-1)


def apply_quaternion(q: ArrayLike, v: ArrayLike) -> jax.Array:
    r"""Apply rotation represented by a quaternion to a vector.

    We use Hamilton's quaternion multiplication.

    To apply the rotation defined by a unit quaternion :math:`\boldsymbol{q}
    \in S^3` to a vector :math:`\boldsymbol{v} \in \mathbb{R}^3`, we
    first represent the vector as a quaternion: we set the scalar part to 0 and
    the vector part is exactly the original vector
    :math:`\left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right) \in
    \mathbb{R}^4`. Then we left-multiply the quaternion and right-multiply
    its conjugate

    .. math::

        \left(\begin{array}{c}0\\\boldsymbol{w}\end{array}\right)
        =
        \boldsymbol{q}
        \cdot
        \left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right)
        \cdot
        \boldsymbol{q}^*.

    The vector part :math:`\boldsymbol{w}` of the resulting quaternion is
    the rotated vector.

    Parameters
    ----------
    q : array-like, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z)

    v : array-like, shape (..., 3)
        3d vector

    Returns
    -------
    w : array, shape (..., 3)
        3d vector
    """
    q = jnp.asarray(q)
    if not jnp.issubdtype(q.dtype, jnp.floating):
        q = q.astype(jnp.float64)

    chex.assert_equal_shape_prefix((q, v), prefix_len=q.ndim - 1)
    chex.assert_axis_dimension(q, axis=-1, expected=4)
    chex.assert_axis_dimension(v, axis=-1, expected=3)

    q = q.reshape(-1, 4)
    real = q[:, 0]
    vec = q[:, 1:]
    t = 2.0 * jnp.cross(vec, v.reshape(-1, 3))
    return v + (real[:, jnp.newaxis] * t + jnp.cross(vec, t)).reshape(v.shape)


def compact_axis_angle_from_quaternion(q: ArrayLike) -> jax.Array:
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    Parameters
    ----------
    q : array-like, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array, shape (..., 3)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    q = jnp.asarray(q)
    if not jnp.issubdtype(q.dtype, jnp.floating):
        q = q.astype(jnp.float64)

    chex.assert_axis_dimension(q, axis=-1, expected=4)

    real = q[..., 0]
    vec = q[..., 1:]
    vec_norm = jnp.linalg.norm(vec, axis=-1)

    axis = norm_vector(vec, norm=vec_norm)
    angle = norm_angle(2.0 * jnp.arccos(jnp.clip(real, -1.0, 1.0)))

    angle_nonzero = (vec_norm >= jnp.finfo(q.dtype).eps)[..., jnp.newaxis]

    return jnp.where(angle_nonzero, axis * angle[..., jnp.newaxis], 0.0)
