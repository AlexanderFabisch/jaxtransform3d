import jax.numpy as jnp
import numpy as np

from ..utils import norm_vector


def cross_product_matrix(v: jnp.ndarray) -> jnp.ndarray:
    r"""Cross-product matrix of a vector.

    The cross-product matrix :math:`\boldsymbol{V}` satisfies the equation

    .. math::

        \boldsymbol{V} \boldsymbol{w} = \boldsymbol{v} \times
        \boldsymbol{w}.

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
    v0 = v[..., 0]
    v1 = v[..., 1]
    v2 = v[..., 2]
    z = jnp.zeros_like(v0)

    col1 = jnp.stack((z, v2, -v1), axis=-1)
    col2 = jnp.stack((-v2, z, v0), axis=-1)
    col3 = jnp.stack((v1, -v0, z), axis=-1)

    return jnp.stack((col1, col2, col3), axis=-1)


def left_jacobian_SO3(omega: jnp.ndarray) -> jnp.ndarray:
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
    omega : array, shape (..., 3)
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
    theta = jnp.linalg.norm(omega)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)
    omega_unit = norm_vector(omega, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    I = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    J = (
        I
        + (1.0 - jnp.cos(theta)) / theta_safe * omega_matrix
        + (1.0 - jnp.sin(theta) / theta_safe) * omega_matrix @ omega_matrix
    )
    J_taylor = left_jacobian_SO3_series(omega)
    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_taylor, J)


def left_jacobian_SO3_series(omega: jnp.ndarray):
    """Left Jacobian of SO(3) at theta from Taylor series with 3 terms.

    Parameters
    ----------
    omega : array-like, shape (..., 3)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (..., 3, 3)
        Left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).
    """
    I = jnp.broadcast_to(jnp.eye(3), omega.shape + (3,))
    px = cross_product_matrix(omega)
    # n-th term (recursive): pxn = pxn @ px / (n + 2)
    px0 = px * 0.5
    px1 = px0 @ px / 3.0
    px2 = px1 @ px * 0.25
    return I + px0 + px1 + px2


def left_jacobian_SO3_inv(omega: jnp.ndarray) -> jnp.ndarray:
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
    omega : array-like, shape (..., 3)
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
    theta = jnp.linalg.norm(omega)
    theta_safe = jnp.where(theta != 0.0, theta, 1.0)
    omega_unit = norm_vector(omega, norm=theta)
    omega_matrix = cross_product_matrix(omega_unit)

    I = jnp.broadcast_to(jnp.eye(3), omega_matrix.shape)
    J_inv = (
        I
        - 0.5 * omega_matrix * theta
        + (1.0 - 0.5 * theta / jnp.tan(theta_safe / 2.0))
        * omega_matrix @ omega_matrix
    )
    J_inv_taylor = left_jacobian_SO3_inv_series(omega)
    return jnp.where(theta[..., jnp.newaxis, jnp.newaxis] < 1e-3, J_inv_taylor, J_inv)


def left_jacobian_SO3_inv_series(omega: jnp.ndarray) -> jnp.ndarray:
    """Inverse left Jacobian of SO(3) at theta from Taylor series with 3 terms.

    Parameters
    ----------
    omega : array-like, shape (..., 3)
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
    I = jnp.broadcast_to(jnp.eye(3), omega.shape + (3,))
    px = cross_product_matrix(omega)
    # n-th term (recursive): pxn = pxn @ px / (n + 1)
    # multiplied with the Beroulli number b[n + 1]:
    # from scipy.special import bernoulli
    # b = bernoulli(n_terms + 1)
    # -> array([1.0, -0.5, 0.16666667, 0.0])
    #                  ^         ^      ^
    #                  0         1      2
    return I - px * 0.5 + px @ px / 12.0
