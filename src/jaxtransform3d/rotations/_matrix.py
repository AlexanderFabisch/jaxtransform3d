import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import differentiable_norm, norm_vector


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

    # precision="highest" avoids reduced-precision (TF32) matmul, which XLA
    # would otherwise select for the batched product and which makes the result
    # depend on the batch size (see compose_matrices).
    return jnp.matmul(
        R.reshape(-1, 3, 3), v.reshape(-1, 3, 1), precision="highest"
    ).reshape(*v.shape)


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
    # precision="highest" avoids reduced-precision (TF32) matmul, which XLA
    # would otherwise select for the batched product. Without it the result of
    # composing a single matrix with a batch differs from the element-wise
    # composition by ~1e-4 in float32.
    return jnp.matmul(
        R1.reshape(-1, 3, 3), R2.reshape(-1, 3, 3), precision="highest"
    ).reshape(bigger_shape)


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

    # same as:
    # RT = R.transpose(tuple(range(R.ndim - 2)) + (R.ndim - 1, R.ndim - 2))
    # matrix_unnormalized = R - RT
    # axis_unnormalized = cross_product_vector(matrix_unnormalized)
    # From Rodrigues' formula, this is 2 * sin(angle) * axis.
    axis_unnormalized = jnp.stack(
        (
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ),
        axis=-1,
    )

    # Determine the angle with atan2 from sin(angle) (skew part) and
    # cos(angle) (trace). arccos of the trace alone loses precision near 0
    # and pi: for small angles the trace rounds to 3, while the skew part
    # still carries the angle with full relative precision.
    traces = jnp.einsum("...ii", R)
    # clip to [-1, 1]: floating-point error can push the cosine slightly
    # outside the valid range (e.g. trace just below -1 for a rotation by pi)
    cos_angle = jnp.clip(0.5 * (traces - 1.0), -1.0, 1.0)
    sin_angle = 0.5 * differentiable_norm(axis_unnormalized, axis=-1)
    angle = jnp.arctan2(sin_angle, cos_angle)

    # Special case: angle close to pi. Here R is (numerically) symmetric, so
    # the skew part R - R^T is zero and its sign cannot recover a general axis.
    # From Rodrigues' formula, R = 2 ee^T - I at pi, i.e. ee^T = 0.5 * (R + I),
    # whose diagonal holds the squared axis components e_i**2. We read the
    # magnitudes |e_i| off that diagonal and the relative signs off the
    # dominant row k = argmax(e_i**2) of the symmetric part, where
    # sign(R_sym[k, j]) = sign(e_k) * sign(e_j). Using the symmetric part (not
    # R) keeps this accurate just below pi, where the skew part then fixes the
    # overall sign of the axis.
    R_sym = 0.5 * (R + jnp.swapaxes(R, -1, -2))
    eeT_diag = jnp.clip(0.5 * (jnp.einsum("...ii->...i", R_sym) + 1.0), 0.0, 1.0)
    dominant = jax.nn.one_hot(jnp.argmax(eeT_diag, axis=-1), 3, dtype=R.dtype)
    dominant_row = jnp.einsum("...i,...ij->...j", dominant, R_sym)
    # fix the dominant component positive (its diagonal sign is unreliable)
    signs = jnp.where(dominant > 0.0, 1.0, jnp.sign(dominant_row))
    axis_close_to_pi = jnp.sqrt(eeT_diag) * signs
    # just below pi the skew part still carries the correct overall sign
    flip = (angle < jnp.pi) & (
        jnp.sum(axis_close_to_pi * axis_unnormalized, axis=-1) < 0.0
    )
    axis_close_to_pi = jnp.where(
        flip[..., jnp.newaxis], -axis_close_to_pi, axis_close_to_pi
    )
    # Near pi the skew part is only 2 * sin(angle), so the axis read off it
    # is dominated by rounding errors; use the symmetric solution instead.
    pi_threshold = 1e-4
    angle_close_to_pi = jnp.abs(angle - jnp.pi) < pi_threshold
    axis_unnormalized = jnp.where(
        angle_close_to_pi[..., jnp.newaxis], axis_close_to_pi, axis_unnormalized
    )
    axis = norm_vector(axis_unnormalized)

    return axis * angle[..., jnp.newaxis]
