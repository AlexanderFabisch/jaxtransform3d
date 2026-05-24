import chex
import jax.numpy as jnp

from ..transformations import (
    adjoint_from_transform,
    transform_from_exponential_coordinates,
    transform_inverse,
)


def product_of_exponentials(
    ee2base_home: jnp.ndarray,
    screw_axes_home: jnp.ndarray,
    joint_limits: jnp.ndarray,
    thetas: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute forward kinematics based on the product of exponentials.

    .. math::

        f: \mathbb{R}^n \rightarrow SE(3),\quad
        f(\boldsymbol{\theta}) = \boldsymbol{T},

    with the joint angle vector :math:`\boldsymbol{\theta} \in \mathbb{R}^n`.

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

    See also
    --------
    jacobian_space
        Derivative of forward kinematics.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.experimental.robotics import product_of_exponentials
    >>> ee2base_home = jnp.array([[-1, 0,  0, 0],
    ...                           [ 0, 1,  0, 6],
    ...                           [ 0, 0, -1, 2],
    ...                           [ 0, 0,  0, 1]], dtype=jnp.float32)
    >>> screw_axes_home = jnp.array([[0, 0,  1,  4, 0,    0],
    ...                              [0, 0,  0,  0, 1,    0],
    ...                              [0, 0, -1, -6, 0, -0.1]])
    >>> thetas = jnp.array([jnp.pi / 2.0, 3, jnp.pi])
    >>> joint_limits = jnp.vstack(([-jnp.inf] * 3, [jnp.inf] * 3)).T
    >>> product_of_exponentials(
    ...     ee2base_home, screw_axes_home, joint_limits, thetas).round(6)
    Array([[ 0...,  1...,  0..., -5...],
           [ 1..., ...0...,  0...,  4...],
           [ 0...,  0..., -1...,  1.685841],
           [ 0...,  0...,  0...,  1...]]...)
    """
    # https://github.com/NxRLab/ModernRobotics/blob/36f0f1b47118f026ac76f406e1881edaba9389f2/packages/Python/modern_robotics/core.py#L593
    chex.assert_equal_shape_prefix((screw_axes_home, thetas), prefix_len=1)

    thetas = jnp.clip(thetas, joint_limits[:, 0], joint_limits[:, 1])
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    for joint_displacement in joint_displacements:
        T = T @ joint_displacement
    return T @ ee2base_home


def jacobian_space(screw_axes_home: jnp.ndarray, thetas: jnp.ndarray) -> jnp.ndarray:
    r"""Computes the space Jacobian.

    .. math::

        \boldsymbol{J}(\boldsymbol{\theta})
        = \frac{\partial f(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}},

    with the joint angle vector :math:`\boldsymbol{\theta} \in \mathbb{R}^n`
    and the forward kinematics function

    .. math::

        f: \mathbb{R}^n \rightarrow \mathbb{R}^6,\quad
        f(\boldsymbol{\theta}) = \boldsymbol{\xi}

    that maps the joint angle vector :math:`\boldsymbol{\theta} \in \mathbb{R}^n`
    to the exponential coordinates :math:`\boldsymbol{\xi}` of the end-effector
    pose with respect to the base frame.

    The Jacobian can be used to map instantaneous velocities in joint space
    to instantaneous spatial velocities in Cartesian space with

    .. math::

        \dot{\boldsymbol{\xi}}
        =
        \boldsymbol{J}(\boldsymbol{\theta}) \dot{\boldsymbol{\theta}}.

    Parameters
    ----------
    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame when the manipulator is at
        the home position.

    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    Js : array, shape (6, n_joints)
        The space Jacobian corresponding to the inputs.

    See also
    --------
    product_of_exponentials
        Forward kinematics.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.experimental.robotics import jacobian_space
    >>> screw_axes = jnp.array([[0, 0, 1,   0, 0.2, 0.2],
    ...                         [1, 0, 0,   2,   0,   3],
    ...                         [0, 1, 0,   0,   2,   1],
    ...                         [1, 0, 0, 0.2, 0.3, 0.4]])
    >>> thetas = jnp.array([0.2, 1.1, 0.1, 1.2])
    >>> jacobian_space(screw_axes, thetas)
    Array([[ 0.        ,  0.980..., -0.090...,  0.957...],
           [ 0.        ,  0.198...,  0.444...,  0.284...],
           [ 1.        ,  0.   ...,  0.891..., -0.045...],
           [ 0.        ,  1.952..., -2.216..., -0.511...],
           [ 0.2       ,  0.436..., -2.437...,  2.775...],
           [ 0.2       ,  2.960...,  3.235...,  2.225...]], ...)
    """
    # https://github.com/NxRLab/ModernRobotics/blob/36f0f1b47118f026ac76f406e1881edaba9389f2/packages/Python/modern_robotics/core.py#L663
    chex.assert_shape(screw_axes_home, (len(thetas), 6))
    chex.assert_shape(thetas, (len(thetas),))

    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    Js = jnp.copy(screw_axes_home.T)
    T = jnp.eye(4)
    for i in range(1, len(thetas)):
        T = T @ transform_from_exponential_coordinates(exp_coords[i - 1])
        Js = Js.at[:, i].set(adjoint_from_transform(T) @ screw_axes_home[i])
    return Js


def jacobian_body(
    ee2base_home: jnp.ndarray,
    screw_axes_home: jnp.ndarray,
    joint_limits: jnp.ndarray,
    thetas: jnp.ndarray,
) -> jnp.ndarray:
    r"""Computes the body Jacobian.

    The body Jacobian :math:`\boldsymbol{J}_b` relates joint velocities to the
    end-effector twist expressed in the body frame:

    .. math::

        \mathcal{V}_b
        =
        \boldsymbol{J}_b(\boldsymbol{\theta}) \dot{\boldsymbol{\theta}}.

    It is related to the space Jacobian
    :math:`\boldsymbol{J}_s` through the adjoint of
    :math:`\boldsymbol{T}_{bs} = \boldsymbol{T}_{sb}^{-1}`:

    .. math::

        \boldsymbol{J}_b
        =
        \left[Ad_{\boldsymbol{T}_{bs}}\right] \boldsymbol{J}_s.

    Unlike :func:`jacobian_space`, the body Jacobian is the right choice when
    the corresponding quantity (e.g., a manipulability ellipsoid) should be
    interpreted at the end-effector.

    Parameters
    ----------
    ee2base_home : array, shape (4, 4)
        The home configuration of the end-effector.

    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame at the home position.

    joint_limits : array, shape (n_joints, 2)
        Joint limits: minimum in column 0, maximum in column 1.

    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    Jb : array, shape (6, n_joints)
        The body Jacobian corresponding to the inputs.

    See also
    --------
    jacobian_space
        The space Jacobian.

    product_of_exponentials
        Forward kinematics.
    """
    Js = jacobian_space(screw_axes_home, thetas)
    T_sb = product_of_exponentials(ee2base_home, screw_axes_home, joint_limits, thetas)
    return adjoint_from_transform(transform_inverse(T_sb)) @ Js
