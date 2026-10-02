import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import matmul, norm_vector
from ._quaternion import compact_axis_angle_from_quaternion


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
           [0., 0., 1.]], dtype=...)

    Inversion is inhenrently vectorized. You can easily apply it to any number
    of dimensions, e.g., a 1D list of rotation matrices:

    >>> import jax
    >>> from jaxtransform3d.rotations import matrix_from_compact_axis_angle
    >>> key = jax.random.PRNGKey(42)
    >>> a = jax.random.normal(key, shape=(20, 3))
    >>> R = matrix_from_compact_axis_angle(a)
    >>> R_inv = matrix_inverse(R)
    >>> R_inv
    Array([[[...]]], dtype=...)
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
    Array([[[[...]]]], dtype=...)
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
           [ 3. ,  2. , -1. ]], dtype=...)
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

    return matmul(R.reshape(-1, 3, 3), v.reshape(-1, 3, 1)).reshape(*v.shape)


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
    return matmul(R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3)).reshape(bigger_shape)


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

    # The axis cannot be read off the skew-symmetric part R - R^T accurately
    # close to pi. The conversion to a quaternion avoids this.
    return compact_axis_angle_from_quaternion(quaternion_from_matrix(R))


def quaternion_from_matrix(R: ArrayLike) -> jax.Array:
    r"""Compute quaternion from rotation matrix.

    We use the method of Markley (2008), which is a variant of Shepperd's
    method (1978): each of the four columns of the symmetric matrix

    .. math::

        \boldsymbol{K} = \left( \begin{array}{cccc}
        1 + tr(\boldsymbol{R}) & r_{32} - r_{23} & r_{13} - r_{31}
        & r_{21} - r_{12}\\
        r_{32} - r_{23} & 1 + 2 r_{11} - tr(\boldsymbol{R})
        & r_{12} + r_{21} & r_{13} + r_{31}\\
        r_{13} - r_{31} & r_{12} + r_{21} & 1 + 2 r_{22} - tr(\boldsymbol{R})
        & r_{23} + r_{32}\\
        r_{21} - r_{12} & r_{13} + r_{31} & r_{23} + r_{32}
        & 1 + 2 r_{33} - tr(\boldsymbol{R})
        \end{array} \right) = 4 \boldsymbol{q} \boldsymbol{q}^T

    is a multiple of the quaternion. We normalize the column with the largest
    diagonal element, which is at least 1, so the result is accurate and
    differentiable for all rotations.

    Parameters
    ----------
    R : array-like, shape (..., 3, 3)
        Rotation matrix.

    Returns
    -------
    q : array, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z).

    See also
    --------
    compact_axis_angle_from_matrix : Logarithmic map for rotation matrices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import quaternion_from_matrix
    >>> quaternion_from_matrix(jnp.eye(3))
    Array([1., 0., 0., 0.], dtype=...)
    >>> quaternion_from_matrix(
    ...     jnp.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
    Array([0., 1., 0., 0.], dtype=...)

    References
    ----------
    .. [1] Markley, F. L. (2008). Unit Quaternion from Rotation Matrix.
       Journal of Guidance, Control, and Dynamics, 31(2), pp. 440-442,
       doi: 10.2514/1.31730.
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)

    chex.assert_axis_dimension(R, axis=-2, expected=3)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    trace = jnp.einsum("...ii", R)[..., jnp.newaxis, jnp.newaxis]
    skew = jnp.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        axis=-1,
    )[..., jnp.newaxis, :]
    sym = R + jnp.swapaxes(R, -1, -2) - (trace - 1.0) * jnp.eye(3, dtype=R.dtype)
    K = jnp.concatenate(
        (
            jnp.concatenate((1.0 + trace, skew), axis=-1),
            jnp.concatenate((jnp.swapaxes(skew, -1, -2), sym), axis=-1),
        ),
        axis=-2,
    )
    column = jnp.argmax(jnp.einsum("...ii->...i", K), axis=-1)
    q = jnp.take_along_axis(K, column[..., jnp.newaxis, jnp.newaxis], axis=-2)
    return norm_vector(q[..., 0, :])
