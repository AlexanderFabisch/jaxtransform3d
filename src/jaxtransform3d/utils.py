"""Utility functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

two_pi = 2.0 * jnp.pi


def norm_angle(a: ArrayLike) -> jax.Array:
    """Normalize angle to (-pi, pi].

    It is worth noting that using `numpy.ceil` to normalize angles will lose
    more digits of precision as angles going larger but can keep more digits
    of precision when angles are around zero. In common use cases, for example,
    -10.0*pi to 10.0*pi, it performs well.

    For more discussions on numerical precision:
    https://github.com/dfki-ric/pytransform3d/pull/263

    Parameters
    ----------
    a : float or array-like, shape (n,)
        Angle(s) in radians.

    Returns
    -------
    a_norm : float or array, shape (n,)
        Normalized angle(s) in radians.
    """
    a = jnp.asarray(a)
    return a - (jnp.ceil((a + jnp.pi) / two_pi) - 1.0) * two_pi


def norm_vector(vec: ArrayLike, norm: ArrayLike | None = None) -> jax.Array:
    """Normalize vector.

    Parameters
    ----------
    vec : array-like, shape (..., n)
        nd vectors.

    norm : array-like, shape (...)
        Precomputed norms of vectors if available.

    Returns
    -------
    vec_unit : array, shape (..., n)
        nd unit vectors with norm 1 or zero vectors.
    """
    vec = jnp.asarray(vec)
    if norm is None:
        norm = jnp.linalg.norm(vec, axis=-1)
    norm = jnp.where(norm != 0.0, norm, 1.0)
    norm = norm[..., jnp.newaxis]
    return jnp.where(norm != 0.0, vec / norm, 0.0)


def cross_product_matrix(v: ArrayLike) -> jnp.ndarray:
    r"""Cross-product matrix of a vector.

    The cross-product matrix :math:`\boldsymbol{V}` satisfies the equation

    .. math::

        \boldsymbol{V} \boldsymbol{w} = \boldsymbol{v} \times \boldsymbol{w}.

    It is a skew-symmetric (antisymmetric) matrix, i.e.,
    :math:`-\boldsymbol{V} = \boldsymbol{V}^T`. Its elements are

    .. math::

        \left[\boldsymbol{v}\right]
        =
        \left[\begin{array}{c}
        v_1\\ v_2\\ v_3
        \end{array}\right]
        =
        \boldsymbol{V}
        =
        \left(\begin{array}{ccc}
        0 & -v_3 & v_2\\
        v_3 & 0 & -v_1\\
        -v_2 & v_1 & 0
        \end{array}\right).

    Parameters
    ----------
    v : array, shape (..., 3)
        3d vector.

    Returns
    -------
    V : array, shape (..., 3, 3)
        Cross-product matrix.
    """
    v = jnp.asarray(v)

    v1 = v[..., 0]
    v2 = v[..., 1]
    v3 = v[..., 2]
    z = jnp.zeros_like(v1)

    col1 = jnp.stack((z, v3, -v2), axis=-1)
    col2 = jnp.stack((-v3, z, v1), axis=-1)
    col3 = jnp.stack((v2, -v1, z), axis=-1)

    return jnp.stack((col1, col2, col3), axis=-1)
