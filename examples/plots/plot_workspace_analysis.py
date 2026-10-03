"""
=====================================================
Workspace Analysis with Batched Inverse Kinematics
=====================================================

Which poses can a robot arm reach, and how well can it move there? We answer
this question numerically: we sample half a million target poses on a
3D grid, solve the inverse kinematics (IK) for all of them at once, and
compute the manipulability of each solution.

Because all functions of jaxtransform3d are batched, differentiable, and can
be compiled with ``jax.jit``, we can write the IK solver for a single target
and use ``jax.vmap`` to solve the IK for all targets and several random
initial configurations in parallel on a GPU.
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytransform3d.transformations as pt
from pytransform3d.urdf import UrdfTransformManager

import jaxtransform3d.transformations as jt


# %%
# Kinematic Model
# ---------------
# We use the same conversion from a URDF to screw axes as in the inverse
# kinematics example.
def get_screw_axes(robot_urdf, ee_frame, base_frame, joint_names):
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
    tm.load_urdf(robot_urdf)

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
# The forward kinematics is the product of exponentials
#
# .. math::
#
#     \boldsymbol{T}(\boldsymbol{\theta}) =
#     \exp([\mathcal{S}_1] \theta_1) \cdots \exp([\mathcal{S}_6] \theta_6)
#     \boldsymbol{M},
#
# where :math:`\mathcal{S}_i` are the screw axes in the base frame and
# :math:`\boldsymbol{M}` is the end-effector's pose in the home configuration.
# In contrast to the inverse kinematics example, we return the transformation
# matrix and not its exponential coordinates.
def forward_kinematics(ee2base_home, screw_axes_home, thetas):
    """Compute end-effector pose with the product of exponentials.

    Parameters
    ----------
    ee2base_home : array, shape (4, 4)
        The home configuration of the end-effector.

    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame at the home position.

    thetas : array, shape (n_joints,)
        Joint angles.

    Returns
    -------
    ee2base : array, shape (4, 4)
        Transformation from end-effector to base.
    """
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = jt.transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    for joint_displacement in joint_displacements:
        T = jt.compose_transforms(T, joint_displacement)
    return jt.compose_transforms(T, ee2base_home)


# %%
# Body Jacobian
# -------------
# We could compute the Jacobian with ``jax.jacfwd`` of the forward kinematics,
# but the product of exponentials also gives us an analytic expression (see
# Lynch and Park, Modern Robotics, chapter 5). The columns of the space
# Jacobian are the screw axes transformed by the partial products of the
# forward kinematics,
#
# .. math::
#
#     \mathcal{J}_{s,i} = \left[\mathrm{Ad}_{\boldsymbol{T}_{i-1}}\right]
#     \mathcal{S}_i,
#     \quad
#     \boldsymbol{T}_{i-1} =
#     \exp([\mathcal{S}_1] \theta_1) \cdots
#     \exp([\mathcal{S}_{i-1}] \theta_{i-1}),
#
# and the body Jacobian is
# :math:`\boldsymbol{J}_b = [\mathrm{Ad}_{\boldsymbol{T}^{-1}}] \boldsymbol{J}_s`.
# For twists :math:`\mathcal{V} = (\boldsymbol{\omega}, \boldsymbol{v})`
# ordered as jaxtransform3d's exponential coordinates, the adjoint
# representation of a transformation with rotation :math:`\boldsymbol{R}` and
# translation :math:`\boldsymbol{t}` is
#
# .. math::
#
#     [\mathrm{Ad}_{\boldsymbol{T}}] =
#     \left(
#     \begin{array}{cc}
#     \boldsymbol{R} & \boldsymbol{0}\\
#     [\boldsymbol{t}] \boldsymbol{R} & \boldsymbol{R}
#     \end{array}
#     \right).
#
# We do not build this :math:`6 \times 6` matrix, but apply it directly to
# the twists.
def adjoint_transform(T, twists):
    """Apply adjoint representation of a transformation to twists.

    Parameters
    ----------
    T : array, shape (4, 4)
        Transformation matrix.

    twists : array, shape (..., 6)
        Twists (omega, v).

    Returns
    -------
    twists : array, shape (..., 6)
        Transformed twists.
    """
    R = T[:3, :3]
    omega = jnp.matmul(twists[..., :3], R.T, precision="highest")
    v = jnp.cross(T[:3, 3], omega) + jnp.matmul(
        twists[..., 3:], R.T, precision="highest"
    )
    return jnp.concatenate((omega, v), axis=-1)


# %%
# We compute the forward kinematics and the Jacobian in the same loop over
# the joints so that we can reuse the partial products
# :math:`\boldsymbol{T}_{i-1}`. We prefer the body Jacobian over the space
# Jacobian, because its translational part :math:`\boldsymbol{J}_v` maps joint
# velocities to the linear velocity of the end-effector (expressed in the
# end-effector's frame), while the translational part of a space twist is the
# velocity of a point that coincides with the base frame's origin. Hence, the
# body Jacobian is the right choice for a manipulability measure of the
# end-effector's position.
#
# ``jt.compose_transforms`` uses the highest precision for matrix
# multiplication and so do we. On GPUs, JAX would otherwise use TF32 for
# float32 matrix products, which introduces errors of about :math:`10^{-4}`.
def forward_kinematics_and_body_jacobian(ee2base_home, screw_axes_home, thetas):
    """Compute end-effector pose and body Jacobian.

    Parameters
    ----------
    ee2base_home : array, shape (4, 4)
        The home configuration of the end-effector.

    screw_axes_home : array, shape (n_joints, 6)
        The joint screw axes in the space frame at the home position.

    thetas : array, shape (n_joints,)
        Joint angles.

    Returns
    -------
    ee2base : array, shape (4, 4)
        Transformation from end-effector to base.

    J : array, shape (6, n_joints)
        Body Jacobian. The first three rows are related to rotation, the last
        three rows to translation.
    """
    exp_coords = screw_axes_home * thetas[:, jnp.newaxis]
    joint_displacements = jt.transform_from_exponential_coordinates(exp_coords)

    T = jnp.eye(4)
    space_jacobian = []
    for screw_axis, joint_displacement in zip(
        screw_axes_home, joint_displacements, strict=True
    ):
        space_jacobian.append(adjoint_transform(T, screw_axis))
        T = jt.compose_transforms(T, joint_displacement)
    T = jt.compose_transforms(T, ee2base_home)

    J = adjoint_transform(jt.transform_inverse(T), jnp.stack(space_jacobian)).T
    return T, J


# %%
# Inverse Kinematics Solver
# -------------------------
# We use a variant of the pseudo-inverse update from the inverse kinematics
# example,
#
# .. math::
#
#     \boldsymbol{\theta} \leftarrow \boldsymbol{\theta}
#     + \alpha \boldsymbol{J}_b^T
#     \left(\boldsymbol{J}_b \boldsymbol{J}_b^T + \lambda^2 \boldsymbol{I}
#     \right)^{-1} \mathcal{V}_b,
#     \quad
#     \mathcal{V}_b = \log\left(\boldsymbol{T}(\boldsymbol{\theta})^{-1}
#     \boldsymbol{T}_{target}\right).
#
# The error :math:`\mathcal{V}_b` is expressed in exponential coordinates
# relative to the current end-effector pose so that it matches the body
# Jacobian. Instead of the pseudo-inverse :math:`\boldsymbol{J}_b^+`, we use
# damped least squares with a small damping factor :math:`\lambda`. For
# :math:`\lambda = 0` and a regular Jacobian, both are the same. Damping keeps
# the steps small close to singularities. In addition, solving a small
# linear system is much faster on a GPU than the singular value decomposition
# that ``jnp.linalg.pinv`` requires (more than 100 times in a batch of
# :math:`3 \cdot 10^5` matrices on our machine). The joint angles are clipped
# to the joint limits after each step.
def ik_step(ee2base_home, screw_axes_home, joint_limits, target, thetas):
    """Perform one step of the damped least squares IK solver."""
    T, J = forward_kinematics_and_body_jacobian(ee2base_home, screw_axes_home, thetas)
    error = jt.exponential_coordinates_from_transform(
        jt.compose_transforms(jt.transform_inverse(T), target)
    )
    JJT = jnp.matmul(J, J.T, precision="highest")
    damped = JJT + 1e-4 * jnp.eye(len(JJT))
    delta = jnp.matmul(J.T, jnp.linalg.solve(damped, error), precision="highest")
    thetas = jnp.clip(thetas + 0.5 * delta, joint_limits[:, 0], joint_limits[:, 1])
    return thetas


def pose_errors(ee2base_home, screw_axes_home, target, thetas):
    """Position error (m) and orientation error (rad) to the target."""
    T = forward_kinematics(ee2base_home, screw_axes_home, thetas)
    position_error = jnp.linalg.norm(T[:3, 3] - target[:3, 3])
    rotation_error = jnp.linalg.norm(
        jt.exponential_coordinates_from_transform(
            jt.compose_transforms(jt.transform_inverse(T), target)
        )[:3]
    )
    return position_error, rotation_error


# %%
# Batched Solver
# --------------
# The functions above handle a single target and a single configuration. We
# use ``jax.vmap`` twice: once over random restarts (initial joint angles)
# and once over targets. ``jax.lax.scan`` runs the iterations. In each
# iteration we also record the fraction of targets that is already solved
# so that we can see how fast the solver converges. ``jax.jit`` compiles
# everything into one GPU program.
def solve_batch(
    ee2base_home,
    screw_axes_home,
    joint_limits,
    targets,
    initial_thetas,
    n_iter,
    position_tolerance,
    rotation_tolerance,
):
    """Solve IK for many targets with several restarts each.

    Parameters
    ----------
    targets : array, shape (n_targets, 4, 4)
        Target poses of the end-effector.

    initial_thetas : array, shape (n_targets, n_restarts, n_joints)
        Initial joint angles.

    Returns
    -------
    thetas : array, shape (n_targets, n_joints)
        Best solution per target.

    position_error : array, shape (n_targets,)
        Position error of best solution.

    rotation_error : array, shape (n_targets,)
        Rotation error of best solution.

    success_rate : array, shape (n_iter,)
        Fraction of targets solved by at least one restart after each
        iteration.
    """
    step = partial(ik_step, ee2base_home, screw_axes_home, joint_limits)
    errors = partial(pose_errors, ee2base_home, screw_axes_home)
    # inner vmap: restarts share one target, outer vmap: targets
    step_all = jax.vmap(jax.vmap(step, in_axes=(None, 0)))
    errors_all = jax.vmap(jax.vmap(errors, in_axes=(None, 0)))

    def is_solved(thetas):
        position_error, rotation_error = errors_all(targets, thetas)
        return (position_error < position_tolerance) & (
            rotation_error < rotation_tolerance
        )

    def body(thetas, _):
        thetas = step_all(targets, thetas)
        return thetas, jnp.mean(jnp.any(is_solved(thetas), axis=1))

    thetas, success_rate = jax.lax.scan(body, initial_thetas, length=n_iter)

    position_error, rotation_error = errors_all(targets, thetas)
    # select the restart with the smallest (scaled) pose error
    best = jnp.argmin(position_error + 0.1 * rotation_error, axis=1)
    take = partial(jnp.take_along_axis, indices=best[:, jnp.newaxis], axis=1)
    thetas = jnp.take_along_axis(thetas, best[:, jnp.newaxis, jnp.newaxis], axis=1)
    return (
        thetas[:, 0],
        take(position_error)[:, 0],
        take(rotation_error)[:, 0],
        success_rate,
    )


# %%
# Manipulability
# --------------
# Yoshikawa's manipulability index measures how well the end-effector can
# move in all directions:
#
# .. math::
#
#     w = \sqrt{\det\left(\boldsymbol{J}_v \boldsymbol{J}_v^T\right)},
#
# where :math:`\boldsymbol{J}_v` are the three translational rows of the body
# Jacobian. It is the volume of the velocity ellipsoid (up to a constant
# factor) and it is zero at singular configurations.
def manipulability(ee2base_home, screw_axes_home, thetas):
    """Yoshikawa manipulability of the end-effector's position."""
    _, J = forward_kinematics_and_body_jacobian(ee2base_home, screw_axes_home, thetas)
    J_v = J[3:]
    JJT = jnp.matmul(J_v, J_v.T, precision="highest")
    return jnp.sqrt(jnp.maximum(jnp.linalg.det(JJT), 0.0))


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
# and extract the kinematic model.
joint_names = [f"joint{i}" for i in range(1, 7)]
tm, ee2base_home, screw_axes_home, joint_limits = get_screw_axes(
    robot_urdf, "tcp", "linkmount", joint_names
)

# %%
# Automatic differentiation is a good way to check the analytic Jacobian.
# ``jax.jacfwd`` gives us :math:`\partial \boldsymbol{T} / \partial \theta_i`,
# and :math:`\boldsymbol{T}^{-1} \partial \boldsymbol{T} / \partial \theta_i`
# is the matrix representation :math:`[\mathcal{J}_{b,i}]` of the i-th column
# of the body Jacobian. The analytic Jacobian is much faster though: for
# :math:`2.5 \cdot 10^5` configurations, ``jax.jacfwd`` takes about 150 ms on
# our GPU, while forward kinematics and analytic Jacobian together take about
# 5 ms.
fk = partial(forward_kinematics, ee2base_home, screw_axes_home)
fk_and_jac = partial(
    forward_kinematics_and_body_jacobian, ee2base_home, screw_axes_home
)
thetas = jax.random.uniform(jax.random.PRNGKey(1), (len(joint_names),))
T, J = fk_and_jac(thetas)
dT = jnp.moveaxis(jax.jacfwd(fk)(thetas), -1, 0)
twists = jt.compose_transforms(jnp.broadcast_to(jt.transform_inverse(T), dT.shape), dT)
J_autodiff = jnp.concatenate(
    (twists[:, (2, 0, 1), (1, 2, 0)], twists[:, :3, 3]), axis=-1
).T
print(f"Max. deviation from autodiff: {jnp.max(jnp.abs(J - J_autodiff)):.1e}")

# %%
# Targets
# -------
# The targets are positions on a regular 3D grid around the robot. The tool
# should point downwards, i.e., its z-axis should be aligned with the negative
# z-axis of the base, which is a rotation by :math:`\pi` about the x-axis.
# The grid contains the vertical plane :math:`y = 0` through the base, which
# we will look at in more detail later.
n_x, n_y, n_z = 81, 81, 81
xs = np.linspace(-0.8, 0.8, n_x)
ys = np.linspace(-0.8, 0.8, n_y)
zs = np.linspace(-0.9, 0.7, n_z)
grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
tool_down = np.diag([1.0, -1.0, -1.0])
targets = np.tile(np.eye(4), (len(grid), 1, 1))
targets[:, :3, :3] = tool_down
targets[:, :3, 3] = grid
targets = jnp.asarray(targets)

# %%
# For each target, we sample initial joint angles uniformly within the joint
# limits, or within :math:`[-\pi, \pi]` for unbounded joints.
n_restarts = 6
lower = jnp.maximum(joint_limits[:, 0], -jnp.pi)
upper = jnp.minimum(joint_limits[:, 1], jnp.pi)
key = jax.random.PRNGKey(0)
initial_thetas = jax.random.uniform(
    key, (len(targets), n_restarts, len(joint_names)), minval=lower, maxval=upper
)

# %%
# Solve Inverse Kinematics
# ------------------------
# We compile the solver first and then measure the runtime of the compiled
# function. A target is reachable if the position error is below 1 mm and
# the orientation error is below 0.01 rad (about 0.6 degrees).
n_iter = 40
solve = jax.jit(
    partial(
        solve_batch,
        ee2base_home,
        screw_axes_home,
        joint_limits,
        n_iter=n_iter,
        position_tolerance=1e-3,
        rotation_tolerance=1e-2,
    )
)
solve = solve.lower(targets, initial_thetas).compile()

start = time.perf_counter()
thetas, position_error, rotation_error, success_rate = jax.block_until_ready(
    solve(targets, initial_thetas)
)
runtime = time.perf_counter() - start

reachable = np.asarray((position_error < 1e-3) & (rotation_error < 1e-2))
n_solves = len(targets) * n_restarts
print(f"Device: {jax.devices()[0]}")
print(f"Targets: {len(targets)}, restarts per target: {n_restarts}")
print(f"IK runtime ({n_iter} iterations): {runtime:.2f} s")
print(f"Throughput: {n_solves / runtime:.0f} IK solves per second")
print(f"Reachable: {reachable.sum()} ({100 * reachable.mean():.1f} %)")

# %%
# Compute Manipulability
# ----------------------
manipulability_all = jax.jit(
    jax.vmap(partial(manipulability, ee2base_home, screw_axes_home))
)
w = np.asarray(manipulability_all(thetas))
w_reachable = w[reachable]
print(f"Manipulability: min {w_reachable.min():.4f}, max {w_reachable.max():.4f}")

# %%
# Plotting
# --------
# The interactive 3D view shows the reachable region with the tool pointing
# downwards as a translucent hull. Move the sliders to cut through it with
# vertical or horizontal slices that are colored by manipulability. Cells
# without color are not reachable. The robot is shown in the solution with
# the highest manipulability in the vertical slice through the base. Note that
# we ignore self-collisions and collisions with the environment, so the
# workspace extends below the mounting plane of the robot.
#
# The workspace is roughly a sphere around the second joint. Manipulability is
# low close to the vertical axis through the base and highest at the sides of
# the workspace. A few isolated unreachable cells inside the workspace are
# failures of the local IK solver, which could be fixed with more restarts.
W = np.where(reachable, w, np.nan).reshape(n_x, n_y, n_z)
G = grid.reshape(n_x, n_y, n_z, 3)
slice_index = n_y // 2
best = np.unravel_index(np.nanargmax(W[:, slice_index]), (n_x, n_z))
robot_thetas = np.asarray(thetas).reshape(n_x, n_y, n_z, -1)[
    best[0], slice_index, best[1]
]
for joint_name, theta in zip(joint_names, robot_thetas, strict=True):
    tm.set_joint(joint_name, theta)


def cylinder_mesh(radius, length, A2B, n_segments=24):
    """Vertices and faces of a cylinder along the z-axis of frame A."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    circle = radius * np.c_[np.cos(angles), np.sin(angles)]
    vertices = np.vstack(
        (
            np.c_[circle, np.full(n_segments, -0.5 * length)],
            np.c_[circle, np.full(n_segments, 0.5 * length)],
            [[0.0, 0.0, -0.5 * length], [0.0, 0.0, 0.5 * length]],
        )
    )
    i = np.arange(n_segments)
    j = (i + 1) % n_segments
    bottom = np.full(n_segments, 2 * n_segments)
    faces = np.vstack(
        (
            np.c_[i, j, n_segments + i],
            np.c_[j, n_segments + j, n_segments + i],
            np.c_[bottom, j, i],
            np.c_[bottom + 1, n_segments + i, n_segments + j],
        )
    )
    return vertices @ A2B[:3, :3].T + A2B[:3, 3], faces


def slice_mesh(centers, normal_axis, half_size):
    """Squares around cell centers, perpendicular to an axis."""
    in_plane = [axis for axis in range(3) if axis != normal_axis]
    corners = half_size * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    vertices = np.repeat(centers, 4, axis=0)
    vertices[:, in_plane] += np.tile(corners, (len(centers), 1))
    first = 4 * np.arange(len(centers))
    faces = np.vstack(
        (np.c_[first, first + 1, first + 2], np.c_[first, first + 2, first + 3])
    )
    return vertices, faces


# The interactive figure would be very large with all grid points, so we only
# use every second grid point in each direction and single precision.
step = 2
W_coarse = W[::step, ::step, ::step]
G_coarse = G[::step, ::step, ::step]
reachable_coarse = ~np.isnan(W_coarse)
xs_coarse, ys_coarse, zs_coarse = xs[::step], ys[::step], zs[::step]
slice_index_coarse = slice_index // step
G32 = G_coarse.astype(np.float32)
fig = go.Figure()
fig.add_trace(
    go.Surface(
        x=[[-0.9, 0.9], [-0.9, 0.9]],
        y=[[-0.9, -0.9], [0.9, 0.9]],
        z=np.zeros((2, 2)),
        colorscale=[[0.0, "tan"], [1.0, "tan"]],
        opacity=0.25,
        showscale=False,
        hoverinfo="skip",
        name="Mounting plane",
        showlegend=True,
    )
)
fig.add_trace(
    go.Isosurface(
        x=G32[..., 0].ravel(),
        y=G32[..., 1].ravel(),
        z=G32[..., 2].ravel(),
        value=reachable_coarse.astype(np.float32).ravel(),
        isomin=0.5,
        isomax=1.0,
        surface_count=1,
        opacity=0.08,
        colorscale=[[0.0, "gray"], [1.0, "gray"]],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        hoverinfo="skip",
        name="Reachable region",
        showlegend=True,
    )
)
for k, visual in enumerate(tm.visuals):
    vertices, faces = cylinder_mesh(
        visual.radius, visual.length, tm.get_transform(visual.frame, "linkmount")
    )
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="rgb(70, 70, 90)",
            flatshading=True,
            hoverinfo="skip",
            name="Robot",
            legendgroup="robot",
            showlegend=k == 0,
        )
    )

# one trace per slice, the sliders switch their visibility
n_static_traces = len(fig.data)
half_size = 0.5 * (xs_coarse[1] - xs_coarse[0])
for normal_axis, n_slices in [(1, len(ys_coarse)), (2, len(zs_coarse))]:
    for index in range(n_slices):
        centers = np.take(G_coarse, index, axis=normal_axis).reshape(-1, 3)
        values = np.take(W_coarse, index, axis=normal_axis).ravel()
        in_workspace = ~np.isnan(values)
        vertices, faces = slice_mesh(centers[in_workspace], normal_axis, half_size)
        vertices, faces = vertices.astype(np.float32), faces.astype(np.int32)
        visible = normal_axis == 1 and index == slice_index_coarse
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=np.tile(values[in_workspace], 2).astype(np.float32),
                intensitymode="cell",
                colorscale="Viridis",
                cmin=0.0,
                cmax=w_reachable.max(),
                showscale=visible,
                colorbar=dict(title="Manipulability w", len=0.6),
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                hovertemplate="w = %{intensity:.3f}<extra></extra>",
                visible=visible,
                name="Slice",
            )
        )


def slice_slider(label, positions, first_trace, active, y):
    traces = list(range(first_trace, first_trace + len(positions)))
    steps = [
        dict(
            method="restyle",
            label="off",
            args=[{"visible": [False] * len(positions)}, traces],
        )
    ] + [
        dict(
            method="restyle",
            label=f"{position:.2f}",
            args=[{"visible": [i == k for i in range(len(positions))]}, traces],
        )
        for k, position in enumerate(positions)
    ]
    return dict(
        active=active,
        steps=steps,
        currentvalue=dict(prefix=label),
        x=0.05,
        len=0.9,
        y=y,
        pad=dict(t=0),
    )


fig.update_layout(
    title=(
        f"Workspace with tool pointing downwards: {reachable.sum()} of "
        f"{len(reachable)} targets reachable"
    ),
    sliders=[
        slice_slider(
            "Vertical slice y = ",
            ys_coarse,
            n_static_traces,
            slice_index_coarse + 1,
            0.12,
        ),
        slice_slider(
            "Horizontal slice z = ",
            zs_coarse,
            n_static_traces + len(ys_coarse),
            0,
            0.0,
        ),
    ],
    scene=dict(
        aspectmode="data",
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        zaxis_title="z [m]",
        camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
    ),
    legend=dict(x=0.01, y=0.95),
    margin=dict(l=0, r=0, t=40, b=60),
    height=800,
)
fig.show()

# %%
# The heatmap shows the vertical slice through the base (:math:`y = 0`).
# White cells are not reachable with the tool pointing downwards. The second
# plot shows how many targets are solved after each iteration of the solver.
fig, (ax_slice, ax_convergence) = plt.subplots(
    1, 2, figsize=(12, 5), width_ratios=(1.2, 1.0), layout="constrained"
)
dx, dz = xs[1] - xs[0], zs[1] - zs[0]
im = ax_slice.imshow(
    W[:, slice_index].T,
    origin="lower",
    extent=(xs[0] - dx / 2, xs[-1] + dx / 2, zs[0] - dz / 2, zs[-1] + dz / 2),
    cmap="viridis",
    vmin=0.0,
    vmax=w_reachable.max(),
)
ax_slice.plot([0.0], [0.0], marker="^", color="k", markersize=10, label="Robot base")
ax_slice.set_aspect("equal")
ax_slice.set_xlabel("x [m]")
ax_slice.set_ylabel("z [m]")
ax_slice.set_title(f"Slice y = {ys[slice_index]:.1f} m")
ax_slice.legend(loc="upper left")
fig.colorbar(im, ax=ax_slice, shrink=0.8, label="Manipulability $w$")

ax_convergence.plot(np.arange(1, n_iter + 1), 100 * np.asarray(success_rate), lw=2)
ax_convergence.set_xlabel("Iteration")
ax_convergence.set_ylabel("Targets solved [%]")
ax_convergence.set_title("Convergence of batched IK")
ax_convergence.set_xlim((1, n_iter))
ax_convergence.set_ylim(bottom=0)
ax_convergence.grid(alpha=0.3)
plt.show()
