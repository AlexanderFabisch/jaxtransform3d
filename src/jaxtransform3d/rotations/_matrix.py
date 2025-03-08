import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


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
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)
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
    """
    R1 = jnp.asarray(R1)
    R2 = jnp.asarray(R2)
    bigger_shape = R1.shape if R1.size > R2.size else R2.shape
    return (R1.reshape(-1, 3, 3) @ R2.reshape(-1, 3, 3)).reshape(bigger_shape)


def compact_axis_angle_from_matrix(R: ArrayLike) -> jax.Array:
    """Compute axis-angle from rotation matrix.

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
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)

    chex.assert_axis_dimension(R, axis=-2, expected=3)
    chex.assert_axis_dimension(R, axis=-1, expected=3)

    instances_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    traces = jnp.einsum("nii", R)
    cos_angle = jnp.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    axis_unnormalized = jnp.column_stack(
        (R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1])
    )

    R_diag = jnp.clip(jnp.einsum("nii->ni", R), -1.0, 1.0)
    eeT_diag = 0.5 * (R_diag + 1.0)
    signs = 2.0 * (axis_unnormalized >= 0.0).astype(R.dtype) - 1.0
    axis_close_to_pi = jnp.sqrt(eeT_diag) * signs

    angle_close_to_pi = abs(angle - jnp.pi) < 1e-4
    axis = jnp.where(
        angle_close_to_pi[:, jnp.newaxis], axis_close_to_pi, axis_unnormalized
    )
    axis = axis / jnp.linalg.norm(axis, axis=-1)[..., jnp.newaxis]

    angle_nonzero = angle != 0.0
    axis = jnp.where(angle_nonzero[:, jnp.newaxis], axis, 0.0)

    axis_angle = axis * angle[:, jnp.newaxis]

    if instances_shape:
        axis_angle = axis_angle.reshape(*instances_shape + (3,))
    else:
        axis_angle = axis_angle[0]

    return axis_angle
