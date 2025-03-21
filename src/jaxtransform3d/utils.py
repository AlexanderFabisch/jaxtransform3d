"""Utility functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

two_pi = 2.0 * jnp.pi


def differentiable_norm(x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
    """Differentiable norm.

    The derivative of sqrt(x) is 1 / (2 * sqrt(x)), so it is close to inf
    at x == 0 and might exceed the representable range of numbers by the
    floating point type when x is close to 0 as the derivative might become
    very large. So we ensure that the norm does not become too small.

    Parameters
    ----------
    x : array, any shape
        Array of which we want to compute the norm.

    axis
        Axis over which we compute the norm.

    Returns
    -------
    x_norm : jnp.ndarray
        Norm of x along given axis.
    """
    squared_norm = (x * x).sum(axis=axis)
    return jnp.sqrt(jnp.maximum(squared_norm, jnp.finfo(x.dtype).eps))


def differentiable_arccos(x: jnp.ndarray) -> jnp.ndarray:
    """Differentiable arccos.

    The derivative of arccos(x) is -1 / sqrt(1 - x**2) and it will become
    infinity at -1 and 1, so we ensure that we do not come to close.

    Parameters
    ----------
    x : array, any shape
        Array of which we want to compute arccos.
    """
    x = jnp.where(
        jnp.abs(1.0 - x) < jnp.finfo(x.dtype).eps, 1.0 - jnp.finfo(x.dtype).eps, x
    )
    x = jnp.where(
        jnp.abs(-1.0 - x) < jnp.finfo(x.dtype).eps, -1.0 + jnp.finfo(x.dtype).eps, x
    )
    return jnp.arccos(x)


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
    return vec / norm[..., jnp.newaxis]


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
