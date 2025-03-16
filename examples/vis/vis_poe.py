"""
=======================
Product of Exponentials
=======================

We compute the forward kinematics of a robot and visualize it.
"""

import os

import chex
import jax.numpy as jnp
import jax.random
import numpy as np
import open3d as o3d

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
    tm = UrdfTransformManager(check=False)
    tm.load_urdf(robot_urdf, mesh_path=mesh_path, package_dir=package_dir)

    ee2base_home = jnp.asarray(tm.get_transform(ee_frame, base_frame))
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
    screw_axes_home = jnp.stack(screw_axes_home, axis=1)

    joint_limits = jnp.array(
        [tm.get_joint_limits(jn) for jn in joint_names]
    )

    return tm, ee2base_home, screw_axes_home, joint_limits


def product_of_exponentials(thetas, ee2base_home, screw_axes_home, joint_limits):
    """Compute probabilistic forward kinematics.

    This is based on the probabilistic product of exponentials.

    Parameters
    ----------
    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    ee2base : array, shape (4, 4)
        A homogeneous transformation matrix representing the end-effector
        frame when the joints are at the specified coordinates.

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
    chex.assert_equal_shape_prefix((screw_axes_home, thetas), prefix_len=1)

    thetas = jnp.clip(thetas, joint_limits[:, 0], joint_limits[:, 1])
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = jt.transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    for joint_displacement in joint_displacements:
        T = T @ joint_displacement
    T = T @ ee2base_home

    return T


def pose_error(thetas, desired):
    actual = product_of_exponentials(thetas, ee2base_home, screw_axes_home, joint_limits)
    return jnp.linalg.norm(
        jt.exponential_coordinates_from_transform(
            jt.transform_inverse(desired) @ actual
        )
    )

pose_grad = jax.jit(jax.grad(pose_error, argnums=(0,)))


# %%
# Then we define a callback to animate the visualization.
def animation_callback(
    step, n_frames, tm, graph, joint_names, thetas, key
):
    angle = 0.5 * np.cos(2.0 * np.pi * (0.5 + step / n_frames))
    thetas_t = angle * thetas
    for joint_name, value in zip(joint_names, thetas_t):
        tm.set_joint(joint_name, value)
    graph.set_data()

    key, sampling_key = jax.random.split(key, 2)
    exp_coords = jax.random.normal(sampling_key, shape=(6,))
    desired = jt.transform_from_exponential_coordinates(exp_coords)
    T = product_of_exponentials(thetas_t, ee2base_home, screw_axes_home, joint_limits)

    return graph


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
# define the joint angles,
thetas = np.array([1, 1, 1, 0, 1, 0])
current_thetas = -0.5 * thetas
for joint_name, theta in zip(joint_names, current_thetas):
    tm.set_joint(joint_name, theta)
key = jax.random.PRNGKey(42)

# %%
# PPOE and Visualization
# ----------------------
#
# Then we can finally use PPOE to compute the end-effector pose and its
# covariance.
T = product_of_exponentials(current_thetas, ee2base_home, screw_axes_home, joint_limits)

# %%
# The following code visualizes the result.
fig = pv.figure()
graph = fig.plot_graph(tm, "robot_arm", show_visuals=True)
fig.plot_transform(np.eye(4), s=0.3)
fig.view_init(elev=5, azim=50)
n_frames = 200
if "__file__" in globals():
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(n_frames, tm, graph, joint_names, thetas, key),
    )
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
