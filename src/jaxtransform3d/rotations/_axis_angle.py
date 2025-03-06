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
    if angle is None:
        angle = jnp.linalg.norm(axis_angle, axis=-1)
    else:
        angle = jnp.asarray(angle)

    if axis is None:
        axis = norm_vector(axis_angle, angle)
    else:
        axis = jnp.asarray(axis)

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

    row1 = jnp.stack((ciux * ux + c, ciuxuy - uzs, ciuxuz + uys), axis=-1)
    row2 = jnp.stack((ciuxuy + uzs, ciuy * uy + c, ciuyuz - uxs), axis=-1)
    row3 = jnp.stack((ciuxuz - uys, ciuyuz + uxs, ci * uz * uz + c), axis=-1)
    return jnp.stack((row1, row2, row3), axis=-2)
