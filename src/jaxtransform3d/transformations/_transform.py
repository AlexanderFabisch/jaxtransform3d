import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


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
