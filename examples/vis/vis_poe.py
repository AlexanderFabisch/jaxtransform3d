"""
=======================
Product of Exponentials
=======================

We compute the inverse kinematics of a robot and visualize it.
"""

import os
from functools import partial

import chex
import jax.numpy as jnp
import jax.random
import numpy as np

import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
import jaxtransform3d.transformations as jt


# %%
# Robot Kinematics
# ----------------
#
# The end-effector's pose distribution is computed based on the Product of
# Exponentials POE.
#
# The complicated part of this example is the conversion of kinematics
# parameters from URDF data to screw axes that are needed for the product
# of exponentials formulation of forward kinematics.
#
# Once we have this information, the implementation of the product of
# exponentials is straightforward:
#
# 1. We multiply the screw axis of each joint with the corresponding joint
#    angle to obtain the exponential coordinates of each relative joint
#    displacement.
# 2. We concatenate the relative joint displacements and the base pose to
#    obtain the end-effector's pose.


def get_screw_axes(
    robot_urdf,
    ee_frame,
    base_frame,
    joint_names,
    mesh_path=None,
    package_dir=None,
):
    """Get screw axes of joints in space frame at robot's home position.

    Parameters
    ----------
    robot_urdf : str
        URDF description of robot

    ee_frame : str
        Name of the end-effector frame

    base_frame : str
        Name of the base frame

    joint_names : list
        Names of joints in order from base to end effector

    mesh_path : str, optional (default: None)
        Path in which we search for meshes that are defined in the URDF.
        Meshes will be ignored if it is set to None and no 'package_dir'
        is given.

    package_dir : str, optional (default: None)
        Some URDFs start file names with 'package://' to refer to the ROS
        package in which these files (textures, meshes) are located. This
        variable defines to which path this prefix will be resolved.

    Returns
    -------
    tm : UrdfTransformManager
        Robot graph.

    ee2base_home : array, shape (4, 4)
        The home configuration (position and orientation) of the
        end-effector.

    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame when the manipulator is at
        the home position.

    joint_limits : array, shape (n_joints, 2)
        Joint limits: joint_limits[:, 0] contains the minimum values and
        joint_limits[:, 1] contains the maximum values.
    """
    tm = UrdfTransformManager()
    tm.load_urdf(robot_urdf, mesh_path=mesh_path, package_dir=package_dir)

    ee2base_home = tm.get_transform(ee_frame, base_frame)
    screw_axes_home = []
    for jn in joint_names:
        ln, _, _, s_axis, limits, joint_type = tm._joints[jn]
        link2base = tm.get_transform(ln, base_frame)
        s_axis = np.dot(link2base[:3, :3], s_axis)
        q = link2base[:3, 3]

        if joint_type == "revolute":
            h = 0.0
        elif joint_type == "prismatic":
            h = np.inf
        else:
            raise NotImplementedError(
                "Joint type %s not supported." % joint_type
            )

        screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
        screw_axes_home.append(screw_axis)
    screw_axes_home = np.vstack(screw_axes_home)

    joint_limits = jnp.array(
        [tm.get_joint_limits(jn) for jn in joint_names]
    )

    return tm, jnp.asarray(ee2base_home), jnp.asarray(screw_axes_home), joint_limits


def product_of_exponentials(ee2base_home, screw_axes_home, joint_limits, thetas):
    """Compute probabilistic forward kinematics.

    This is based on the probabilistic product of exponentials.

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
    ee2base : array, shape (6,)
        Exponential coordinates of transformation from end-effector to base.
    """
    chex.assert_equal_shape_prefix((screw_axes_home, thetas), prefix_len=1)

    thetas = jnp.clip(thetas, joint_limits[:, 0], joint_limits[:, 1])
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = jt.transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    for joint_displacement in joint_displacements:
        T = T @ joint_displacement
    T = T @ ee2base_home

    return jt.exponential_coordinates_from_transform(T)


# %%
# Then we define a callback to animate the visualization.
def animation_callback(
    step, tm, graph, target_frame, joint_names
):
    global key, target, thetas

    if step == 0:
        key, sampling_key = jax.random.split(key, 2)
        current_exp_coords = forward(thetas)
        exp_coords = 0.9 * current_exp_coords + 0.1 * jax.random.normal(sampling_key, shape=(6,))
        target = jt.transform_from_exponential_coordinates(exp_coords)
        target_frame.set_data(A2B=target)

    J = jac(thetas)
    error = jt.exponential_coordinates_from_transform(target) - forward(thetas)
    new_thetas = thetas + 0.2 * jnp.linalg.pinv(J) @ error
    if not jnp.any(jnp.isnan(new_thetas)):
        thetas = new_thetas
    print(error)

    for joint_name, value in zip(joint_names, thetas):
        tm.set_joint(joint_name, value)
    graph.set_data()

    return graph, target_frame


# %%
# Setup
# -----
# We load the URDF file,
BASE_DIR = "data/"
data_dir = BASE_DIR
search_path = "."
while (
    not os.path.exists(data_dir)
    and os.path.dirname(search_path) != "jaxtransform3d"
):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()

# %%
# define the kinematic chain that we are interested in,
joint_names = ["joint%d" % i for i in range(1, 7)]
tm, ee2base_home, screw_axes_home, joint_limits = get_screw_axes(
    robot_urdf, "tcp", "linkmount", joint_names
)

# %%
# define the Jacobian,
forward = jax.jit(
    partial(
        product_of_exponentials,
        ee2base_home,
        screw_axes_home,
        joint_limits,
    )
)
jac = jax.jit(jax.jacobian(forward))

# %%
# and define the joint angles.
thetas = -0.5 * jnp.ones(6)
for joint_name, theta in zip(joint_names, thetas):
    tm.set_joint(joint_name, theta)
key = jax.random.PRNGKey(42)

# %%
# The following code visualizes the result.
fig = pv.figure()
graph = fig.plot_graph(tm, "robot_arm", show_visuals=True)
target = jnp.zeros(6)
target_frame = fig.plot_transform(np.eye(4), s=0.3)
fig.view_init(elev=5, azim=50)
n_frames = 100
if "__file__" in globals():
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(tm, graph, target_frame, joint_names),
    )
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
