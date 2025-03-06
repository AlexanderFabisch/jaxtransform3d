import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..rotations import compact_axis_angle_from_matrix
from ..utils import norm_vector


def create_transform(R: ArrayLike, t: ArrayLike) -> jax.Array:
    r"""Make transformation from rotation matrix and translation.

    .. math::

        \boldsymbol{T}_{BA} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
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
    """Compute exponential coordinates from transformation matrix.

    This is the logarithm map.

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
    """
    T = jnp.asarray(T)

    chex.assert_axis_dimension(T, axis=-2, expected=4)
    chex.assert_axis_dimension(T, axis=-1, expected=4)

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    axis_angle = compact_axis_angle_from_matrix(R)
    angle = jnp.linalg.norm(axis_angle, axis=-1)
    axis = norm_vector(axis_angle, norm=angle)

    ti = jnp.where(angle != 0.0, 1.0 / angle, 0.0)
    tan_term = -0.5 / jnp.tan(angle / 2.0) + ti
    o0 = axis[..., 0]
    o1 = axis[..., 1]
    o2 = axis[..., 2]
    t0 = t[..., 0]
    t1 = t[..., 1]
    t2 = t[..., 2]
    o00 = o0 * o0
    o01 = o0 * o1
    o02 = o0 * o2
    o11 = o1 * o1
    o12 = o1 * o2
    o22 = o2 * o2

    v = (
        jnp.stack(
            (
                t0 * ((-o11 - o22) * tan_term + ti)
                + t1 * (o01 * tan_term + 0.5 * o2)
                + t2 * (o02 * tan_term - 0.5 * o1),
                t0 * (o01 * tan_term - 0.5 * o2)
                + t1 * ((-o00 - o22) * tan_term + ti)
                + t2 * (0.5 * o0 + o12 * tan_term),
                t0 * (o02 * tan_term + 0.5 * o1)
                + t1 * (-0.5 * o0 + o12 * tan_term)
                + t2 * ((-o00 - o11) * tan_term + ti),
            ),
            axis=-1,
        )
        * angle[..., jnp.newaxis]
    )
    v = jnp.where((angle != 0.0)[..., jnp.newaxis], v, t)

    exp_coords = jnp.concatenate((axis_angle, v), axis=-1)

    return exp_coords
