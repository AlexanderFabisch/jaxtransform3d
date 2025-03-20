import jax
import jax.numpy as jnp

from ..utils import cross_product_matrix, norm_vector


def left_jacobian_SO3(axis_angle: jnp.ndarray) -> jnp.ndarray:
    r"""Left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}(\theta)
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
    theta = jnp.linalg.norm(axis_angle, axis=-1)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)  # avoid division by 0
    omega_unit = norm_vector(axis_angle, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    eye = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    factor1 = (1.0 - jnp.cos(theta_safe)) / theta_safe
    factor2 = 1.0 - jnp.sin(theta_safe) / theta_safe
    J = (
        eye
        + factor1[..., jnp.newaxis, jnp.newaxis] * omega_matrix
        + factor2[..., jnp.newaxis, jnp.newaxis] * omega_matrix @ omega_matrix
    )
    J_taylor = left_jacobian_SO3_series(axis_angle)

    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_taylor, J)


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
        pxn = pxn @ px / (n + 2)
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
    theta = jnp.linalg.norm(axis_angle, axis=-1)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)  # avoid division by 0
    omega_unit = norm_vector(axis_angle, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    eye = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    factor1 = 0.5 * theta
    factor2 = 1.0 - 0.5 * theta / jnp.tan(theta_safe / 2.0)
    J_inv = (
        eye
        - factor1[..., jnp.newaxis, jnp.newaxis] * omega_matrix
        + factor2[..., jnp.newaxis, jnp.newaxis] * omega_matrix @ omega_matrix
    )
    J_inv_taylor = left_jacobian_SO3_inv_series(axis_angle)
    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_inv_taylor, J_inv)


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
        pxn = pxn @ px / (n + 1)
        J_inv = J_inv + b[n + 1] * pxn
    return J_inv
