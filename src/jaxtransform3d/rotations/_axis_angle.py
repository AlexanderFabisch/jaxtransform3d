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
    r"""Compute rotation matrices from compact axis-angle representations.

    This is called exponential map or Rodrigues' formula.

    Given a compact axis-angle representation (rotation vector)
    :math:`\hat{\boldsymbol{\omega}} \theta \in \mathbb{R}^3`, we compute
    the rotation matrix :math:`\boldsymbol{R} \in SO(3)` as

    .. math::
        :nowrap:

        \begin{eqnarray}
        \boldsymbol{R}(\hat{\boldsymbol{\omega}} \theta)
        &=&
        Exp(\hat{\boldsymbol{\omega}} \theta)\\
        &=&
        \cos{\theta} \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta})
        \hat{\boldsymbol{\omega}}\hat{\boldsymbol{\omega}}^T\\
        &=&
        \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta}) \left[\hat{\boldsymbol{\omega}}\right]^2.
        \end{eqnarray}

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
    R : array, shape (..., 3, 3)
        Rotation matrices

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import matrix_from_compact_axis_angle
    >>> matrix_from_compact_axis_angle(jnp.zeros(3))
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=...)
    >>> import jax
    >>> a = jax.random.normal(jax.random.PRNGKey(42), shape=(2, 3))
    >>> a
    Array([[-0.02830462,  0.46713185,  0.29570296],
           [ 0.15354592, -0.12403282,  0.21692315]], dtype=...)
    >>> matrix_from_compact_axis_angle(a)
    Array([[[ 0.8510371 , -0.2872734 ,  0.43955666],
            [ 0.27438563,  0.95699465,  0.09420117],
            [-0.44771487,  0.0404393 ,  0.89326155]],
           [[ 0.96900326, -0.22328097, -0.10572752],
            [ 0.20437238,  0.96493644, -0.16471078],
            [ 0.1387971 ,  0.1379975 ,  0.980659  ]]], dtype=...)
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

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import quaternion_from_compact_axis_angle
    >>> quaternion_from_compact_axis_angle(jnp.zeros(3))
    Array([1., 0., 0., 0.], dtype=...)
    >>> import jax
    >>> a = jax.random.normal(jax.random.PRNGKey(42), shape=(2, 3))
    >>> a
    Array([[-0.02830462,  0.46713185,  0.29570296],
           [ 0.15354592, -0.12403282,  0.21692315]], dtype=...)
    >>> quaternion_from_compact_axis_angle(a)
    Array([[ 0.96193725, -0.01397229,  0.23059493,  0.14597079],
           [ 0.98926723,  0.0764981 , -0.06179438,  0.10807326]], ...)
    """
    axis_angle = jnp.asarray(axis_angle)

    chex.assert_axis_dimension(axis_angle, axis=-1, expected=3)

    angle = jnp.linalg.norm(axis_angle, axis=-1)
    axis = norm_vector(axis_angle, norm=angle)

    half_angle = 0.5 * angle
    real = jnp.cos(half_angle)[..., jnp.newaxis]
    vec = jnp.sin(half_angle)[..., jnp.newaxis] * axis

    return jnp.concatenate((real, vec), axis=-1)
