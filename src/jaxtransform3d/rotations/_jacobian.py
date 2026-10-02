import jax
import jax.numpy as jnp

from ..utils import (
    COSC_SERIES,
    COTC_SERIES,
    SINC3_SERIES,
    cross_product_matrix,
    matmul,
    series_or_closed_form,
)


def left_jacobian_SO3(axis_angle: jnp.ndarray) -> jnp.ndarray:
    r"""Left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \frac{\partial Exp(\hat{\boldsymbol{\omega}}\theta)}
        {\partial\hat{\boldsymbol{\omega}}\theta}
        =
        \boldsymbol{J}(\hat{\boldsymbol{\omega}}\theta)
        =
        \frac{\sin{\theta}}{\theta} \boldsymbol{I}
        + \left(\frac{1 - \cos{\theta}}{\theta}\right)
        \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\sin{\theta}}{\theta} \right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    axis_angle : array, shape (..., 3)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (..., 3, 3)
        Left Jacobian of SO(3).

    See also
    --------
    left_jacobian_SO3_series :
        Left Jacobian of SO(3) at theta from Taylor series.

    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    axis_angle = jnp.asarray(axis_angle)
    # (1 - cos(theta)) / theta ** 2
    factor1 = series_or_closed_form(
        axis_angle, lambda t: (1.0 - jnp.cos(t)) / t**2, COSC_SERIES
    )
    # (theta - sin(theta)) / theta ** 3
    factor2 = series_or_closed_form(
        axis_angle, lambda t: (t - jnp.sin(t)) / t**3, SINC3_SERIES
    )

    omega_matrix = cross_product_matrix(axis_angle)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=omega_matrix.dtype), omega_matrix.shape)
    return (
        eye
        + factor1[..., jnp.newaxis, jnp.newaxis] * omega_matrix
        + factor2[..., jnp.newaxis, jnp.newaxis] * matmul(omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_series(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """Left Jacobian of SO(3) at theta from Taylor series with 10 terms.

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (..., 3, 3)
        Left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).
    """
    eye = jnp.broadcast_to(jnp.eye(3), axis_angle.shape + (3,))
    px = cross_product_matrix(axis_angle)
    pxn = eye
    J = eye
    for n in range(10):
        pxn = matmul(pxn, px) / (n + 2)
        J = J + pxn
    return J


def left_jacobian_SO3_inv(axis_angle: jnp.ndarray) -> jnp.ndarray:
    r"""Inverse left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}^{-1}(\theta)
        =
        \frac{\theta}{2 \tan{\frac{\theta}{2}}} \boldsymbol{I}
        - \frac{\theta}{2} \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\theta}{2 \tan{\frac{\theta}{2}}}\right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    axis_angle : array-like, shape (..., 3)
        Compact axis-angle representation.

    Returns
    -------
    J_inv : array, shape (..., 3, 3)
        Inverse left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).

    left_jacobian_SO3_inv_series :
        Inverse left Jacobian of SO(3) at theta from Taylor series.
    """
    axis_angle = jnp.asarray(axis_angle)
    # (1 - theta / (2 * tan(theta / 2))) / theta ** 2
    factor = series_or_closed_form(
        axis_angle, lambda t: (1.0 - 0.5 * t / jnp.tan(0.5 * t)) / t**2, COTC_SERIES
    )

    omega_matrix = cross_product_matrix(axis_angle)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=omega_matrix.dtype), omega_matrix.shape)
    return (
        eye
        - 0.5 * omega_matrix
        + factor[..., jnp.newaxis, jnp.newaxis] * matmul(omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_inv_series(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """Inverse left Jacobian of SO(3) at theta from Taylor series with 10 terms.

    Parameters
    ----------
    axis_angle : array, shape (..., 3)
        Compact axis-angle representation.

    Returns
    -------
    J_inv : array, shape (..., 3, 3)
        Inverse left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    eye = jnp.broadcast_to(jnp.eye(3), axis_angle.shape + (3,))
    px = cross_product_matrix(axis_angle)

    J_inv = eye
    pxn = eye
    px = cross_product_matrix(axis_angle)
    b = jax.scipy.special.bernoulli(11)
    for n in range(10):
        pxn = matmul(pxn, px) / (n + 1)
        J_inv = J_inv + b[n + 1] * pxn
    return J_inv
