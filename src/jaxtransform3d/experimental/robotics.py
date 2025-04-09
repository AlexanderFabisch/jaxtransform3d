import jax.numpy as jnp

from ..transformations import (
    adjoint_from_transform,
    transform_from_exponential_coordinates,
)


def jacobian_space(screw_axes: jnp.ndarray, thetas: jnp.ndarray) -> jnp.ndarray:
    """Computes the space Jacobian.

    Parameters
    ----------
    screw_axes : array, shape (6, n_joints)
        The joint screw axes in the space frame when the manipulator is at the
        home position, in the format of a matrix with axes as the columns.

    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    Js : array, shape (6, n_joints)
        The space Jacobian corresponding to the inputs.
    """
    # https://github.com/NxRLab/ModernRobotics/blob/36f0f1b47118f026ac76f406e1881edaba9389f2/packages/Python/modern_robotics/core.py#L663
    exp_coords = screw_axes * thetas[:, jnp.newaxis]
    Js = jnp.copy(screw_axes)
    T = jnp.eye(4)
    for i in range(1, len(thetas)):
        T = T @ transform_from_exponential_coordinates(exp_coords[i - 1])
        Js = Js.at[:, i].set(adjoint_from_transform(T) @ screw_axes[i])
    return Js
