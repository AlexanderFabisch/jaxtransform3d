import chex
import jax.numpy as jnp

from ..transformations import (
    adjoint_from_transform,
    transform_from_exponential_coordinates,
)


def product_of_exponentials(ee2base_home, screw_axes_home, joint_limits, thetas):
    """Compute forward kinematics based on the product of exponentials.

    Parameters
    ----------
    ee2base_home : array, shape (4, 4)
        The home configuration (position and orientation) of the
        end-effector.

    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame when the manipulator is at
        the home position.

    joint_limits : array, shape (n_joints, 2)
        Joint limits: joint_limits[:, 0] contains the minimum values and
        joint_limits[:, 1] contains the maximum values.

    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    ee2base : array, shape (4, 4)
        Transformation from end-effector to base.
    """
    chex.assert_equal_shape_prefix((screw_axes_home, thetas), prefix_len=1)

    thetas = jnp.clip(thetas, joint_limits[:, 0], joint_limits[:, 1])
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    for joint_displacement in joint_displacements:
        T = T @ joint_displacement
    return T @ ee2base_home


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
