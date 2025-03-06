import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import norm_vector


def matrix_from_compact_axis_angle(
    axis_angle: ArrayLike | None = None,
    axis: ArrayLike | None = None,
    angle: ArrayLike | None = None,
) -> jax.Array:
    """Compute rotation matrices from compact axis-angle representations.

    This is called exponential map or Rodrigues' formula.

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Axes of rotation and rotation angles in compact representation:
        angle * (x, y, z), also known as rotation vector.

    axis : array, shape (..., 3)
        If the unit axes of rotation have been precomputed, you can pass them
        here.

    angle : array, shape (...)
        If the angles have been precomputed, you can pass them here.

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    axis_angle = jnp.asarray(axis_angle)
    if not jnp.issubdtype(axis_angle.dtype, jnp.floating):
        axis_angle = axis_angle.astype(jnp.float64)

    if angle is None:
        angle = jnp.linalg.norm(axis_angle, axis=-1)
    else:
        angle = jnp.asarray(angle)

    if axis is None:
        axis = norm_vector(axis_angle, angle)
    else:
        axis = jnp.asarray(axis)

    chex.assert_axis_dimension(axis_angle, axis=-1, expected=3)
    chex.assert_equal_shape((axis_angle, axis))
    chex.assert_equal_shape_prefix((axis_angle, angle), prefix_len=axis_angle.ndim - 1)

    c = jnp.cos(angle)
    s = jnp.sin(angle)
    ci = 1.0 - c
    ux = axis[..., 0]
    uy = axis[..., 1]
    uz = axis[..., 2]

    uxs = ux * s
    uys = uy * s
    uzs = uz * s
    ciux = ci * ux
    ciuy = ci * uy
    ciuxuy = ciux * uy
    ciuxuz = ciux * uz
    ciuyuz = ciuy * uz

    col1 = jnp.stack((ciux * ux + c, ciuxuy + uzs, ciuxuz - uys), axis=-1)
    col2 = jnp.stack((ciuxuy - uzs, ciuy * uy + c, ciuyuz + uxs), axis=-1)
    col3 = jnp.stack((ciuxuz + uys, ciuyuz - uxs, ci * uz * uz + c), axis=-1)

    return jnp.stack((col1, col2, col3), axis=-1)


def quaternion_from_compact_axis_angle(axis_angle: ArrayLike) -> jax.Array:
    """Compute quaternion from axis-angle.

    This operation is called exponential map.

    Parameters
    ----------
    axis_angle : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    axis_angle = jnp.asarray(axis_angle)

    chex.assert_axis_dimension(axis_angle, axis=-1, expected=3)

    angle = jnp.linalg.norm(axis_angle, axis=-1)
    axis = norm_vector(axis_angle, norm=angle)

    half_angle = 0.5 * angle
    real = jnp.cos(half_angle)[..., jnp.newaxis]
    vec = jnp.sin(half_angle)[..., jnp.newaxis] * axis

    return jnp.concatenate((real, vec), axis=-1)
