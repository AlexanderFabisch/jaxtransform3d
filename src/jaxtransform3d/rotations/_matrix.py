import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import differentiable_arccos, norm_vector


def matrix_inverse(R: ArrayLike) -> jax.Array:
    r"""Invert rotation matrix.

    The inverse of a rotation matrix :math:`\boldsymbol{R} \in SO(3)` is its
    transpose :math:`\boldsymbol{R}^{-1} = \boldsymbol{R}^T` because of the
    orthonormality constraint
    :math:`\boldsymbol{R}\boldsymbol{R}^T = \boldsymbol{I}` (see
    :func:`~norm_matrix`).

    Parameters
    ----------
    R : array-like, shape (..., 3, 3)
        Rotation matrix.

    Returns
    -------
    R_inv : array, shape (..., 3, 3)
        Inverted rotation matrix.

    See also
    --------
    quaternion_conjugate : Inverts the rotation represented by a unit quaternion.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import matrix_inverse

    Inverting a single rotation matrix:

    >>> matrix_inverse(jnp.eye(3))
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

    Inversion is inhenrently vectorized. You can easily apply it to any number
    of dimensions, e.g., a 1D list of rotation matrices:

    >>> import jax
    >>> from jaxtransform3d.rotations import matrix_from_compact_axis_angle
    >>> key = jax.random.PRNGKey(42)
    >>> a = jax.random.normal(key, shape=(20, 3))
    >>> R = matrix_from_compact_axis_angle(a)
    >>> R_inv = matrix_inverse(R)
    >>> R_inv
    Array([[[...]]], dtype=float32)
    >>> R_inv.shape
    (20, 3, 3)
    >>> from jaxtransform3d.rotations import compose_matrices
    >>> I = compose_matrices(R, R_inv)
    >>> I[0].round(5)
    Array([[...1., ...0., ...0.],
           [...0., ...1., ...0.],
           [...0., ...0., ...1.]], ...)

    Or a 2D list of rotation matrices:

    >>> R = R.reshape(5, 4, 3, 3)
    >>> R_inv = matrix_inverse(R)
    >>> R_inv
    Array([[[[...]]]], dtype=float32)
    >>> R_inv.shape
    (5, 4, 3, 3)
    >>> I = compose_matrices(R, R_inv)
    >>> I[0, 0].round(5)
    Array([[...1., ...0., ...0.],
           [...0., ...1., ...0.],
           [...0., ...0., ...1.]], ...)
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)

    chex.assert_axis_dimension(R, axis=-2, expected=3)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    return jnp.swapaxes(R, -1, -2)


def apply_matrix(R: ArrayLike, v: ArrayLike) -> jax.Array:
    r"""Apply rotation matrix to vector.

    Computes the matrix-vector product

    .. math::

        \boldsymbol{w} = \boldsymbol{R} \boldsymbol{v}.

    Parameters
    ----------
    R : array-like, shape (..., 3, 3) or (3, 3)
        Rotation matrix.

    v : array-like, shape (..., 3) or (3,)
        3d vector.

    Returns
    -------
    w : array, shape (..., 3) or (3,)
        3d vector.

    See also
    --------
    apply_quaternion : Apply rotation represented by a unit quaternion.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import (
    ...    apply_matrix, matrix_from_compact_axis_angle)
    >>> a = jnp.array([[0.5 * jnp.pi, 0.0, 0.0],
    ...                [0.0, 0.5 * jnp.pi, 0.0]])
    >>> R = matrix_from_compact_axis_angle(a)
    >>> v = jnp.array([[0.5, 1.0, 2.5], [1, 2, 3]])
    >>> apply_matrix(R[0], v[0]).round(7)
    Array([ 0.5, -2.5,  1. ], ...)
    >>> apply_matrix(R, v)
    Array([[ 0.5, -2.5,  1. ],
           [ 3. ,  2. , -1. ]], dtype=float32)
    """
    R = jnp.asarray(R)
    v = jnp.asarray(v)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)
    if not jnp.issubdtype(v.dtype, jnp.floating):
        v = v.astype(jnp.float64)

    chex.assert_axis_dimension(v, axis=-1, expected=3)
    chex.assert_axis_dimension(R, axis=-2, expected=3)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    return (R.reshape(-1, 3, 3) @ v.reshape(-1, 3, 1)).reshape(*v.shape)


def compose_matrices(R1: ArrayLike, R2: ArrayLike) -> jax.Array:
    r"""Compose rotation matrices.

    Computes the matrix-matrix product

    .. math::

        \boldsymbol{R}_1 \cdot \boldsymbol{R}_2.

    Parameters
    ----------
    R1 : array-like, shape (..., 3, 3) or (3, 3)
        Rotation matrix.

    R2 : array-like, shape (..., 3, 3) or (3, 3)
        Rotation matrix.

    Returns
    -------
    R1_R2 : array, shape (..., 3, 3) or (3, 3)
        Composed rotation matrix.

    See also
    --------
    compose_quaternions : Compose two quaternions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import (
    ...    compose_matrices, matrix_from_compact_axis_angle)
    >>> a1 = jnp.array([0.5 * jnp.pi, 0.0, 0.0])
    >>> R1 = matrix_from_compact_axis_angle(a1)
    >>> a2 = jnp.array([0.0, 0.5 * jnp.pi, 0.0])
    >>> R2 = matrix_from_compact_axis_angle(a2)
    >>> compose_matrices(R1, R2).round(6)
    Array([[...0., ...0., ...1.],
           [...1., ...0., ...0.],
           [...0., ...1., ...0.]], ...)
    """
    R1 = jnp.asarray(R1)
    R2 = jnp.asarray(R2)
    bigger_shape = R1.shape if R1.size > R2.size else R2.shape
    return (R1.reshape(-1, 3, 3) @ R2.reshape(-1, 3, 3)).reshape(bigger_shape)


def compact_axis_angle_from_matrix(R: ArrayLike) -> jax.Array:
    r"""Compute axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    Parameters
    ----------
    R : array-like, shape (..., 3, 3)
        Rotation matrix.

    Returns
    -------
    a : array, shape (..., 3)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].

    See also
    --------
    matrix_from_compact_axis_angle : Exponential map.
    compact_axis_angle_from_quaternion : Logarithmic map for quaternions.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import compact_axis_angle_from_matrix
    >>> compact_axis_angle_from_matrix(jnp.eye(3))
    Array([0., 0., 0.], dtype=...)
    >>> compact_axis_angle_from_matrix(
    ...     jnp.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]))
    Array([ 0..., -1.57...,  0...], dtype=...)

    References
    ----------
    .. [1] Williams, A. (n.d.). Computing the exponential map on SO(3).
       https://arwilliams.github.io/so3-exp.pdf
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)

    chex.assert_axis_dimension(R, axis=-2, expected=3)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    # determine angle from traces
    traces = jnp.einsum("...ii", R)
    cos_angle = 0.5 * (traces - 1.0)
    angle = differentiable_arccos(cos_angle)

    # same as:
    # RT = R.transpose(tuple(range(R.ndim - 2)) + (R.ndim - 1, R.ndim - 2))
    # matrix_unnormalized = R - RT
    # axis_unnormalized = cross_product_vector(matrix_unnormalized)
    axis_unnormalized = jnp.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        axis=-1,
    )

    # Direct solution with correction for small angles with Taylor series.
    # We do not use it because normalizing the axis seems to be more accurate.
    # s = jnp.sin(angle)
    # factor = 0.5 * angle / jnp.where(s == 0.0, 1.0, s)
    # factor_taylor = 0.5 + angle**2 / 12.0 + 7.0 * angle**4 / 720.0  # + O(theta**6)
    # factor = jnp.where(angle < 1e-4, factor_taylor, factor)
    # axis_angle = axis_unnormalized * factor[..., jnp.newaxis]

    # special case: angle close to pi
    R_diag = jnp.clip(jnp.einsum("...ii->...i", R), -1.0, 1.0)
    signs = 2.0 * (axis_unnormalized >= 0.0).astype(R.dtype) - 1.0
    axis_close_to_pi = jnp.sqrt(0.5 * (R_diag + 1.0)) * signs
    angle_close_to_pi = jnp.abs(angle - jnp.pi) < 1e-4
    axis_unnormalized = jnp.where(
        angle_close_to_pi[..., jnp.newaxis], axis_close_to_pi, axis_unnormalized
    )
    axis = norm_vector(axis_unnormalized)

    return axis * angle[..., jnp.newaxis]
