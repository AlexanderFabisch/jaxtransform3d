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
    theta = jnp.linalg.norm(axis_angle)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)
    omega_unit = norm_vector(axis_angle, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    eye = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    J = (
        eye
        + (1.0 - jnp.cos(theta_safe)) / theta_safe * omega_matrix
        + (1.0 - jnp.sin(theta_safe) / theta_safe) * omega_matrix @ omega_matrix
    )
    return J
    J_taylor = left_jacobian_SO3_series(axis_angle)
    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_taylor, J)


def left_jacobian_SO3_series(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """Left Jacobian of SO(3) at theta from Taylor series with 3 terms.

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
    # n-th term (recursive): pxn = pxn @ px / (n + 2)
    px0 = px * 0.5
    px1 = px0 @ px / 3.0
    px2 = px1 @ px * 0.25
    return eye + px0 + px1 + px2


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
    theta = jnp.linalg.norm(axis_angle)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)
    omega_unit = norm_vector(axis_angle, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    eye = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    J_inv = (
        eye
        - 0.5 * omega_matrix * theta
        + (1.0 - 0.5 * theta / jnp.tan(theta_safe / 2.0)) * omega_matrix @ omega_matrix
    )
    return J_inv
    J_inv_taylor = left_jacobian_SO3_inv_series(axis_angle)
    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_inv_taylor, J_inv)


def left_jacobian_SO3_inv_series(axis_angle: jnp.ndarray) -> jnp.ndarray:
    """Inverse left Jacobian of SO(3) at theta from Taylor series with 3 terms.

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
    # n-th term (recursive): pxn = pxn @ px / (n + 1)
    # multiplied with the Beroulli number b[n + 1]:
    # from scipy.special import bernoulli
    # b = bernoulli(n_terms + 1)
    # -> array([1.0, -0.5, 0.16666667, 0.0])
    #                  ^         ^      ^
    #                  0         1      2
    return eye - px * 0.5 + px @ px / 12.0
