"""
=========================
Manipulability of a Robot
=========================


"""

import os
from functools import partial

import chex
import jax.numpy as jnp
import jax.random
import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
import pytransform3d.visualizer as pv
from matplotlib import cbook
from pytransform3d.urdf import UrdfTransformManager

import jaxtransform3d.transformations as jt


# %%
# Forward Kinematics
# ------------------
# The end-effector's pose is computed based on the Product of Exponentials.
#
# The complicated part of this example is the conversion of kinematics
# parameters from URDF data to screw axes that are needed for the product
# of exponentials formulation of forward kinematics.
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
            raise NotImplementedError(f"Joint type {joint_type} not supported.")

        screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
        screw_axes_home.append(screw_axis)
    screw_axes_home = np.vstack(screw_axes_home)

    joint_limits = jnp.array([tm.get_joint_limits(jn) for jn in joint_names])

    return tm, jnp.asarray(ee2base_home), jnp.asarray(screw_axes_home), joint_limits


# %%
# Once we have this information, the implementation of the product of
# exponentials is straightforward:
#
# 1. We multiply the screw axis of each joint with the corresponding joint
#    angle to obtain the exponential coordinates of each relative joint
#    displacement.
# 2. We concatenate the relative joint displacements and the base pose to
#    obtain the end-effector's pose.
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
    J : array, shape (6, n_joints)
        The space Jacobian corresponding to the inputs.
    """
    # https://github.com/NxRLab/ModernRobotics/blob/36f0f1b47118f026ac76f406e1881edaba9389f2/packages/Python/modern_robotics/core.py#L663
    exp_coords = screw_axes * thetas[:, jnp.newaxis]
    JsT = jnp.copy(screw_axes)
    T = jnp.eye(4)
    for i in range(1, len(thetas)):
        T = T @ jt.transform_from_exponential_coordinates(exp_coords[i - 1])
        JsT = JsT.at[i].set(jt.adjoint_from_transform(T) @ screw_axes[i])
    return JsT.T


# %%
# Setup
# -----
# We load the URDF file,
BASE_DIR = "data/"
data_dir = BASE_DIR
search_path = "."
while not os.path.exists(data_dir) and os.path.dirname(search_path) != "jaxtransform3d":
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename) as f:
    robot_urdf = f.read()

# %%
# define the kinematic chain that we are interested in,
joint_names = [f"joint{i}" for i in range(1, 7)]
tm, ee2base_home, screw_axes_home, joint_limits = get_screw_axes(
    robot_urdf, "tcp", "linkmount", joint_names
)

# %%
# define the Jacobian with JAX,
forward = jax.jit(
    partial(
        product_of_exponentials,
        ee2base_home,
        screw_axes_home,
        joint_limits,
    )
)
jacobian = jax.jit(jax.jacfwd(forward))
jacobian_ana = jax.jit(partial(jacobian_space, screw_axes_home))

# %%
# and define the joint angles.
thetas = jnp.array([0, 0, 0.5, 0.1, 0.5, 0], dtype=np.float32)
for joint_name, theta in zip(joint_names, thetas, strict=True):
    tm.set_joint(joint_name, theta)

print((jacobian(thetas) - jacobian_ana(thetas)).round(3))


def jacobian_ellipsoids(
    thetas: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes manipulability and force ellipsoid based on Jacobian.

    Parameters
    ----------

    thetas : array, shape (n_joints,)
        A list of joint coordinates.

    Returns
    -------
    manipulability : array, shape (6, 6)
        Manipulability matrix.

    force : array, shape (6, 6)
        Force matrix.

    manipulability_radii : array, shape (6,)
        Radii of manipulability ellipsoid.

    force_radii : array, shape (6,)
        Radii of force ellipsoid.
    """
    J = jacobian_ana(thetas)
    A = J @ J.T
    try:
        A_eigenvalues = jnp.linalg.eigvals(A)
        manipulability_radii = jnp.sqrt(jnp.maximum(A_eigenvalues, 0.0))
    except np.linalg.LinAlgError:
        manipulability_radii = jnp.linalg.svd(A)[1]
    Ainv = jnp.linalg.pinv(A)
    Ainv_eigenvalues = jnp.linalg.eigvals(Ainv)
    force_radii = jnp.sqrt(Ainv_eigenvalues)
    return A, Ainv, manipulability_radii, force_radii


ee2base = jt.transform_from_exponential_coordinates(forward(thetas))
manipulability, force, _, _ = jacobian_ellipsoids(thetas)

x, y, z = pu.to_projected_ellipsoid(
    np.asarray(ee2base), np.asarray(manipulability), factor=1, n_steps=50
)
polys = np.stack([cbook._array_patch_perimeters(a, 1, 1) for a in (x, y, z)], axis=-1)
vertices = polys.reshape(-1, 3)
triangles = (
    [[4 * i + 0, 4 * i + 1, 4 * i + 2] for i in range(len(polys))]
    + [[4 * i + 2, 4 * i + 3, 4 * i + 0] for i in range(len(polys))]
    + [[4 * i + 0, 4 * i + 3, 4 * i + 2] for i in range(len(polys))]
    + [[4 * i + 2, 4 * i + 1, 4 * i + 0] for i in range(len(polys))]
)
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
)
mesh.paint_uniform_color((0, 0.7, 0))

# %%
# The following code visualizes the result.
fig = pv.figure()
fig.plot_graph(tm, "robot_arm", show_visuals=True)
fig.add_geometry(mesh)
fig.view_init(elev=5, azim=50)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
