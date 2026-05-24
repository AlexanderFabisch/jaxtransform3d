"""
=========================
Manipulability of a Robot
=========================


"""

import os
from functools import partial

import jax.numpy as jnp
import jax.random
import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager

import jaxtransform3d.experimental.robotics as jrob
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
    ee2base : array, shape (6,)
        Exponential coordinates of transformation from end-effector to base.
    """
    return jt.exponential_coordinates_from_transform(
        jrob.product_of_exponentials(
            ee2base_home, screw_axes_home, joint_limits, thetas
        )
    )


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
jacobian_space = jax.jit(partial(jrob.jacobian_space, screw_axes_home))
jacobian_body = jax.jit(
    partial(jrob.jacobian_body, ee2base_home, screw_axes_home, joint_limits)
)

# %%
# and define the joint angles.
thetas = jnp.array([0, 0, 0.5, 0.1, 0.5, 0], dtype=np.float32)
for joint_name, theta in zip(joint_names, thetas, strict=True):
    tm.set_joint(joint_name, theta)

# %%
# Sanity check: the body and space Jacobians are related by the adjoint of
# the inverse end-effector pose (Lynch & Park, *Modern Robotics*, eq. 5.22):
#
# .. math::
#
#     \boldsymbol{J}_b(\boldsymbol{\theta})
#     = \left[Ad_{\boldsymbol{T}_{sb}^{-1}}\right]
#       \boldsymbol{J}_s(\boldsymbol{\theta}).
#
# This is an exact algebraic identity and should hold to machine precision.
T_sb = jrob.product_of_exponentials(
    ee2base_home, screw_axes_home, joint_limits, thetas
)
Jb_via_adjoint = jt.adjoint_from_transform(jt.transform_inverse(T_sb)) @ jacobian_space(
    thetas
)
print(
    "|J_body - Ad(T_sb^-1) J_space|_max =",
    float(jnp.max(jnp.abs(jacobian_body(thetas) - Jb_via_adjoint))),
)


# %%
# Manipulability ellipsoid
# ------------------------
# Following Yoshikawa (1985) and Lynch & Park's *Modern Robotics* (§5.4), the
# translational manipulability ellipsoid at the end-effector is the image of
# the unit ball in joint-velocity space under the linear part of the
# **body** Jacobian:
#
# .. math::
#
#     \mathcal{E}_v = \{\boldsymbol{J}_v \dot{\boldsymbol{\theta}} \;:\;
#     \|\dot{\boldsymbol{\theta}}\| \le 1\},
#
# whose principal axes are the eigenvectors of
# :math:`\boldsymbol{A}_v = \boldsymbol{J}_v \boldsymbol{J}_v^T` and whose
# semi-axis lengths are :math:`\sqrt{\lambda_i}`. We use the *body* Jacobian
# (rather than the space Jacobian) so the ellipsoid is naturally anchored at
# the end-effector. We treat the angular and translational parts separately
# because mixing units in a single 6D ellipsoid is not meaningful.
#
# The dual **force ellipsoid** shares the same principal axes but with
# inverse semi-axis lengths :math:`1/\sqrt{\lambda_i}` -- directions in which
# the manipulator moves easily are precisely those in which it can resist
# external forces least.
def manipulability_ellipsoid(thetas: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Compute the translational manipulability ellipsoid at the end-effector.

    Parameters
    ----------
    thetas : array, shape (n_joints,)
        Joint coordinates.

    Returns
    -------
    eigvals : array, shape (3,)
        Eigenvalues of :math:`\boldsymbol{A}_v = \boldsymbol{J}_v \boldsymbol{J}_v^T`
        in ascending order. Semi-axis lengths are :math:`\sqrt{\lambda_i}`.

    eigvecs : array, shape (3, 3)
        Eigenvectors as columns, expressed in the end-effector (body) frame.
    """
    # Body Jacobian: (omega; v) convention, so rows 3:6 are the linear part.
    Jb = jacobian_body(thetas)
    Jv = Jb[3:, :]
    A_v = Jv @ Jv.T
    eigvals, eigvecs = jnp.linalg.eigh(A_v)
    return jnp.maximum(eigvals, 0.0), eigvecs


def ellipsoid_mesh(
    center: np.ndarray,
    rotation: np.ndarray,
    radii: np.ndarray,
    color: tuple,
    n_steps: int = 30,
) -> o3d.geometry.TriangleMesh:
    """Build a closed o3d triangle mesh for an ellipsoid."""
    u = np.linspace(0.0, 2.0 * np.pi, n_steps)
    v = np.linspace(0.0, np.pi, n_steps)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    sphere = np.stack(
        [np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)], axis=-1
    ).reshape(-1, 3)
    points = sphere * radii  # scale to ellipsoid in the eigenframe
    points = points @ rotation.T + center  # rotate to world frame, translate

    triangles = []
    for i in range(n_steps - 1):
        for j in range(n_steps - 1):
            a = i * n_steps + j
            b = a + n_steps
            triangles.append([a, b, a + 1])
            triangles.append([a + 1, b, b + 1])

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


ee2base = jt.transform_from_exponential_coordinates(forward(thetas))
eigvals, eigvecs_body = manipulability_ellipsoid(thetas)

# Express eigenvectors in the base frame (the body Jacobian is in body coords).
R_sb = np.asarray(ee2base[:3, :3])
eigvecs_base = R_sb @ np.asarray(eigvecs_body)
center = np.asarray(ee2base[:3, 3])

manip_radii = np.sqrt(np.asarray(eigvals))
print("Translational manipulability semi-axes:", manip_radii)
print("Yoshikawa manipulability index sqrt(det(A_v)):", float(np.prod(manip_radii)))

# Manipulability ellipsoid (green): velocity ellipsoid at the end-effector.
manip_mesh = ellipsoid_mesh(center, eigvecs_base, manip_radii, (0.0, 0.7, 0.0))

# Force ellipsoid (red): dual ellipsoid with inverse radii. Skip directions
# where the manipulator is singular (eigvals == 0).
force_radii = np.where(manip_radii > 1e-6, 1.0 / np.maximum(manip_radii, 1e-6), 0.0)
# Normalize so the two ellipsoids are visually comparable.
scale = manip_radii.max() / max(force_radii.max(), 1e-6) * 0.5
force_mesh = ellipsoid_mesh(
    center, eigvecs_base, force_radii * scale, (0.7, 0.0, 0.0)
)

# %%
# The following code visualizes the result.
fig = pv.figure()
fig.plot_graph(tm, "robot_arm", show_visuals=True)
fig.add_geometry(manip_mesh)
fig.add_geometry(force_mesh)
fig.view_init(elev=5, azim=50)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
