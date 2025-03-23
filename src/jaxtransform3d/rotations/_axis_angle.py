import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import cross_product_matrix, differentiable_norm, min_diff_norm


def matrix_from_compact_axis_angle(axis_angle: ArrayLike | None = None) -> jax.Array:
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

    For small angles we use a Taylor series approximation.

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Axis of rotation and rotation angle in compact form (also known as rotation
        vector): angle * (x, y, z) or :math:`\hat{\boldsymbol{\omega}} \theta`.

    Returns
    -------
    R : array, shape (..., 3, 3)
        Rotation matrices

    See also
    --------
    compact_axis_angle_from_matrix : Logarithmic map.
    quaternion_from_compact_axis_angle : Exponential map for quaternions.

    References
    ----------
    .. [1] Williams, A. (n.d.). Computing the exponential map on SO(3).
       https://arwilliams.github.io/so3-exp.pdf

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
    Array([[[ 0.85103..., -0.28727...,  0.43955...],
            [ 0.27438...,  0.95699...,  0.09420...],
            [-0.44771...,  0.04043...,  0.89326...]],
           [[ 0.96900..., -0.22328..., -0.10572...],
            [ 0.20437...,  0.96493..., -0.16471...],
            [ 0.13879...,  0.13799...,  0.98065...]]], dtype=...)
    """
    axis_angle = jnp.asarray(axis_angle)
    if not jnp.issubdtype(axis_angle.dtype, jnp.floating):
        axis_angle = axis_angle.astype(jnp.float64)

    angle = differentiable_norm(axis_angle, axis=-1)

    chex.assert_axis_dimension(axis_angle, axis=-1, expected=3)
    chex.assert_equal_shape_prefix((axis_angle, angle), prefix_len=axis_angle.ndim - 1)

    valid_angle = angle > min_diff_norm(angle)
    angle_safe = jnp.where(valid_angle, angle, 1.0)
    factor1 = jnp.sin(angle) / angle_safe
    angle_p2 = angle * angle
    angle_p2_safe = jnp.where(valid_angle, angle_p2, 1.0)
    factor2 = (1.0 - jnp.cos(angle)) / angle_p2_safe

    angle_p4 = angle_p2 * angle_p2
    factor1_taylor = 1.0 - angle_p2 / 6.0 + angle_p4 / 120.0  # + O(angle**6)
    factor2_taylor = 0.5 - angle_p2 / 24.0 + angle_p4 / 720.0  # + O(angle**6)

    factor1 = jnp.where(angle < 1e-3, factor1_taylor, factor1)
    factor2 = jnp.where(angle < 1e-3, factor2_taylor, factor2)

    omega_matrix = cross_product_matrix(axis_angle)
    eye = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)

    return (
        eye
        + factor1[..., jnp.newaxis, jnp.newaxis] * omega_matrix
        + factor2[..., jnp.newaxis, jnp.newaxis] * omega_matrix @ omega_matrix
    )


def quaternion_from_compact_axis_angle(axis_angle: ArrayLike) -> jax.Array:
    r"""Compute quaternion from axis-angle.

    This operation is called exponential map.

    Given a compact axis-angle representation (rotation vector)
    :math:`\hat{\boldsymbol{\omega}} \theta \in \mathbb{R}^3`, we compute
    the unit quaternion :math:`\boldsymbol{q} \in S^3` as

    .. math::

        \boldsymbol{q}(\hat{\boldsymbol{\omega}} \theta)
        =
        Exp(\hat{\boldsymbol{\omega}} \theta)
        =
        \left(
        \begin{array}{c}
        \cos{\frac{\theta}{2}}\\
        \hat{\boldsymbol{\omega}} \sin{\frac{\theta}{2}}
        \end{array}
        \right).


    For small angles we use a Taylor series approximation.

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Axis of rotation and rotation angle in compact form (also known as rotation
        vector): angle * (x, y, z) or :math:`\hat{\boldsymbol{\omega}} \theta`.

    Returns
    -------
    q : array, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z)

    See also
    --------
    compact_axis_angle_from_quaternion : Logarithmic map.
    matrix_from_compact_axis_angle : Exponential map for rotation matrices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import quaternion_from_compact_axis_angle
    >>> quaternion_from_compact_axis_angle(jnp.zeros(3))
    Array([1., 0., 0., 0.], dtype=...)
    >>> import jax
    >>> a = jax.random.normal(jax.random.PRNGKey(42), shape=(2, 3))
    >>> a
    Array([[-0.0283...,  0.4671...,  0.2957...],
           [ 0.1535..., -0.1240...,  0.2169...]], dtype=...)
    >>> quaternion_from_compact_axis_angle(a)
    Array([[ 0.9619..., -0.0139...,  0.2305...,  0.1459...],
           [ 0.9892...,  0.0764..., -0.0617...,  0.1080...]], ...)
    """
    axis_angle = jnp.asarray(axis_angle)

    chex.assert_axis_dimension(axis_angle, axis=-1, expected=3)

    angle = jnp.linalg.norm(axis_angle, axis=-1)
    angle_safe = jnp.where(angle == 0, 1.0, angle)
    half_angle = 0.5 * angle

    axis_scale = jnp.sin(half_angle) / angle_safe
    # small angle Taylor series expansion based on
    # https://github.com/scipy/scipy/blob/ae25ba2385e62d5372a47ed59f9cfddc5ab3dc6a/scipy/spatial/transform/_rotation.pyx#L1300
    angle_p2 = angle * angle
    axis_scale_taylor = 0.5 - angle_p2 / 48.0 * angle_p2 * angle_p2 / 3840.0
    axis_scale = jnp.where(angle < 1e-3, axis_scale_taylor, axis_scale)

    real = jnp.cos(half_angle)[..., jnp.newaxis]
    vec = axis_scale[..., jnp.newaxis] * axis_angle

    return jnp.concatenate((real, vec), axis=-1)


def assert_compact_axis_angle_equal(a1, a2, *args, **kwargs):
    from numpy.testing import assert_array_almost_equal

    angle1 = jnp.linalg.norm(a1)
    angle2 = jnp.linalg.norm(a2)
    # required despite normalization in case of 180 degree rotation
    if (
        abs(angle1) - jnp.pi < 1e-2
        and abs(angle2) - jnp.pi < 1e-2
        and jnp.any(jnp.sign(a1) != jnp.sign(a2))
    ):
        a1 = -a1
    assert_array_almost_equal(a1, a2, *args, **kwargs)
