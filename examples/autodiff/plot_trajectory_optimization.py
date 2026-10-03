"""
================================
Trajectory Optimization on SE(3)
================================

We optimize a smooth and collision-free trajectory of a rigid body for the
"cubicles" problem, a standard benchmark for rigid body motion planning from
the `Open Motion Planning Library (OMPL) <https://ompl.kavrakilab.org>`_. An
L-shaped robot has to move through an office building with two floors. The
direct path is blocked by walls, so the robot has to go downstairs, cross the
building on the lower floor, and come back upstairs. Trajectory optimization
is a local method and needs a reasonable initial guess. We take a few
keyframes of a path found by a sampling-based planner and connect them with
screw linear interpolation (ScLERP). This initial trajectory cuts corners and
collides with the building. We formulate a nonlinear least squares problem in
the tangent space of SE(3) and solve it with the Levenberg-Marquardt
algorithm. All derivatives are computed with JAX.
"""

import configparser
import os
import urllib.request
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.plot_utils as ppu
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import jaxtransform3d.rotations as jr
import jaxtransform3d.transformations as jt

# %%
# Benchmark Data
# --------------
# The benchmark is part of OMPL.app. We download the configuration file,
# the meshes of the environment and the robot, and a solution path found by a
# sampling-based planner, and store them in the data directory.
OMPL_URL = (
    "https://raw.githubusercontent.com/ompl/omplapp/"
    "958884d6593809ee6a564907ff4e796691d48ffc/resources/3D/"
)
BASE_DIR = "data/"
data_dir = BASE_DIR
search_path = "."
while not os.path.exists(data_dir) and os.path.dirname(search_path) != "jaxtransform3d":
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)


def fetch(filename):
    path = os.path.join(data_dir, "ompl", filename)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(OMPL_URL + filename, path)
    return path


config = configparser.ConfigParser()
config.read(fetch("cubicles.cfg"))
problem = config["problem"]


# %%
# The meshes are stored in COLLADA files, which are XML files. We extract all
# triangles and transform them with the transformations of the scene graph.
# OMPL.app loads meshes with Assimp, which converts the z-up convention of
# these files to a y-up convention. The start and goal poses of the benchmark
# are defined in this frame, so we apply the same conversion.
def load_collada(filename):
    ns = "{http://www.collada.org/2005/11/COLLADASchema}"
    root = ET.parse(filename).getroot()
    geometries = {}
    for geometry in root.iter(f"{ns}geometry"):
        mesh = geometry.find(f"{ns}mesh")
        sources = {
            source.get("id"): np.array(
                source.find(f"{ns}float_array").text.split(), dtype=float
            ).reshape(-1, 3)
            for source in mesh.findall(f"{ns}source")
        }
        for vertices in mesh.findall(f"{ns}vertices"):
            source_id = vertices.find(f"{ns}input").get("source")[1:]
            sources[vertices.get("id")] = sources[source_id]
        triangles = []
        for element in mesh.findall(f"{ns}triangles"):
            inputs = element.findall(f"{ns}input")
            stride = max(int(i.get("offset")) for i in inputs) + 1
            vertex_input = next(i for i in inputs if i.get("semantic") == "VERTEX")
            indices = np.array(element.find(f"{ns}p").text.split(), dtype=int)
            indices = indices.reshape(-1, stride)[:, int(vertex_input.get("offset"))]
            positions = sources[vertex_input.get("source")[1:]]
            triangles.append(positions[indices].reshape(-1, 3, 3))
        geometries[geometry.get("id")] = np.concatenate(triangles)

    triangles = []

    def visit(node, node2world):
        matrix = node.find(f"{ns}matrix")
        if matrix is not None:
            node2world = node2world @ np.array(matrix.text.split(), float).reshape(4, 4)
        for instance in node.findall(f"{ns}instance_geometry"):
            vertices = geometries[instance.get("url")[1:]]
            triangles.append(vertices @ node2world[:3, :3].T + node2world[:3, 3])
        for child in node.findall(f"{ns}node"):
            visit(child, node2world)

    for scene in root.iter(f"{ns}visual_scene"):
        for node in scene.findall(f"{ns}node"):
            visit(node, np.eye(4))
    z_up_to_y_up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    triangles = np.concatenate(triangles) @ z_up_to_y_up.T

    # remove degenerate triangles and back faces of double-sided faces
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    triangles = triangles[areas > 1e-6]
    rounded = triangles.round(3)
    order = np.lexsort(rounded[..., ::-1].transpose(2, 0, 1), axis=-1)
    keys = np.take_along_axis(rounded, order[..., None], axis=1)
    _, first = np.unique(keys.reshape(len(triangles), 9), axis=0, return_index=True)
    return triangles[np.sort(first)]


env_triangles = load_collada(fetch(problem["world"]))
robot_triangles = load_collada(fetch(problem["robot"]))
# OMPL.app moves the origin of the robot to the mean of its vertices
robot_triangles -= np.unique(robot_triangles.reshape(-1, 3), axis=0).mean(axis=0)


# %%
# Start and goal pose are given by a position and a rotation axis and angle.
def pose_from_config(name):
    axis = np.array([float(problem[f"{name}.axis.{d}"]) for d in "xyz"])
    angle = float(problem[f"{name}.theta"])
    position = np.array([float(problem[f"{name}.{d}"]) for d in "xyz"])
    R = jr.matrix_from_compact_axis_angle(jnp.asarray(angle * axis))
    return jt.create_transform(R, jnp.asarray(position))


T_start = pose_from_config("start")
T_goal = pose_from_config("goal")
volume_min = np.array([float(problem[f"volume.min.{d}"]) for d in "xyz"])
volume_max = np.array([float(problem[f"volume.max.{d}"]) for d in "xyz"])


# %%
# Signed Distance Field
# ---------------------
# We need the distance of the robot to the environment and its gradient. We
# precompute the signed distance to the environment on a grid with a
# resolution of 4 units. The distance of a point to the environment is the
# minimum distance to all triangles. We determine whether a point is inside of
# an obstacle with the generalized winding number, i.e., the sum of solid
# angles of all triangles as seen from the point divided by :math:`4 \pi`.
# These are many point-triangle pairs, but they can be processed in parallel
# with :func:`jax.vmap` on the GPU.
def point_triangle_distance(point, triangle):
    a, b, c = triangle
    normal = jnp.cross(b - a, c - a)
    normal /= jnp.linalg.norm(normal)
    inside = (
        (jnp.dot(jnp.cross(b - a, point - a), normal) >= 0.0)
        & (jnp.dot(jnp.cross(c - b, point - b), normal) >= 0.0)
        & (jnp.dot(jnp.cross(a - c, point - c), normal) >= 0.0)
    )

    def segment_distance(start, end):
        t = jnp.dot(point - start, end - start) / jnp.dot(end - start, end - start)
        return jnp.linalg.norm(point - start - jnp.clip(t, 0.0, 1.0) * (end - start))

    edge_distance = jnp.minimum(
        jnp.minimum(segment_distance(a, b), segment_distance(b, c)),
        segment_distance(c, a),
    )
    return jnp.where(inside, jnp.abs(jnp.dot(point - a, normal)), edge_distance)


def solid_angle(point, triangle):
    a, b, c = triangle - point
    la, lb, lc = jnp.linalg.norm(a), jnp.linalg.norm(b), jnp.linalg.norm(c)
    numerator = jnp.dot(a, jnp.cross(b, c))
    denominator = (
        la * lb * lc + jnp.dot(a, b) * lc + jnp.dot(b, c) * la + jnp.dot(c, a) * lb
    )
    return 2.0 * jnp.arctan2(numerator, denominator)


env = jnp.asarray(env_triangles)


@jax.jit
def exact_signed_distance(points):
    def signed_distance(point):
        distance = jax.vmap(point_triangle_distance, (None, 0))(point, env).min()
        winding_number = jax.vmap(solid_angle, (None, 0))(point, env).sum()
        inside = jnp.abs(winding_number) > 2.0 * jnp.pi
        return jnp.where(inside, -distance, distance)

    return jax.lax.map(signed_distance, points, batch_size=8192)


resolution = 4.0
grid_shape = tuple(np.ceil((volume_max - volume_min) / resolution).astype(int) + 1)
grid = np.stack(
    np.meshgrid(
        *[volume_min[i] + resolution * np.arange(grid_shape[i]) for i in range(3)],
        indexing="ij",
    ),
    axis=-1,
)
sdf_grid = exact_signed_distance(jnp.asarray(grid.reshape(-1, 3))).reshape(grid_shape)


# %%
# Between grid points we interpolate trilinearly, which is differentiable.
def signed_distance(points):
    coordinates = ((points - volume_min) / resolution).T
    return jax.scipy.ndimage.map_coordinates(
        sdf_grid, list(coordinates), order=1, mode="nearest"
    )


# %%
# We represent the robot by points on its surface. We sample many points
# uniformly and select a subset that covers the surface evenly with farthest
# point sampling. A second, denser set of points is used to evaluate the
# clearance of the final trajectories.
def sample_surface(triangles, n_samples, rng):
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    indices = rng.choice(len(triangles), n_samples, p=areas / areas.sum())
    u, v = rng.random((2, n_samples, 1))
    flip = u + v > 1.0
    u, v = np.where(flip, 1.0 - u, u), np.where(flip, 1.0 - v, v)
    a, b, c = triangles[indices].transpose(1, 0, 2)
    return a + u * (b - a) + v * (c - a)


def farthest_point_sampling(points, n_points):
    selected = [0]
    distances = np.linalg.norm(points - points[0], axis=-1)
    for _ in range(n_points - 1):
        selected.append(np.argmax(distances))
        distances = np.minimum(
            distances, np.linalg.norm(points - points[selected[-1]], axis=-1)
        )
    return points[selected]


rng = np.random.default_rng(0)
dense_robot_points = jnp.asarray(sample_surface(robot_triangles, 5_000, rng))
robot_points = jnp.asarray(farthest_point_sampling(np.asarray(dense_robot_points), 400))


# %%
# Initial Trajectory
# ------------------
# The benchmark comes with a solution path computed by a sampling-based
# planner. Each line contains a position and a quaternion (x, y, z, w). We
# select every 20th pose as a keyframe. The trajectory consists of
# :math:`N = 200` poses with a duration of 30 seconds. Between two consecutive
# keyframes :math:`\boldsymbol{T}_a` and :math:`\boldsymbol{T}_b` we
# interpolate along the screw axis
#
# .. math::
#
#     \boldsymbol{T}(s) = \boldsymbol{T}_a
#     Exp\left(s \cdot Log\left(\boldsymbol{T}_a^{-1} \boldsymbol{T}_b\right)\right),
#     \quad s \in [0, 1].
#
# The time assigned to each segment is proportional to its length, where we
# convert rotation angles to lengths with a characteristic length :math:`l`.
# Each segment has a constant velocity, hence the velocity jumps at the
# keyframes. Note that trajectory optimization only finds a local optimum
# close to the initial trajectory. If we select too few keyframes, the
# initial trajectory might, for instance, cut through the floor between both
# levels and the optimizer cannot resolve this collision.
n_steps = 200
duration = 30.0
dt = duration / (n_steps - 1)
times = np.linspace(0.0, duration, n_steps)
characteristic_length = 50.0

planner_path = np.loadtxt(fetch("cubicles.path"))
keyframes = planner_path[np.r_[0 : len(planner_path) - 1 : 20, -1]]
R_keyframes = jr.matrix_from_compact_axis_angle(
    jr.compact_axis_angle_from_quaternion(jnp.asarray(keyframes[:, [6, 3, 4, 5]]))
)
T_keyframes = jt.create_transform(R_keyframes, jnp.asarray(keyframes[:, :3]))
segments = jt.exponential_coordinates_from_transform(
    jt.compose_transforms(jt.transform_inverse(T_keyframes[:-1]), T_keyframes[1:])
)
segment_lengths = jnp.linalg.norm(
    segments[:, 3:], axis=-1
) + characteristic_length * jnp.linalg.norm(segments[:, :3], axis=-1)
keyframe_times = jnp.concatenate((jnp.zeros(1), jnp.cumsum(segment_lengths)))
keyframe_times *= duration / keyframe_times[-1]
segment_indices = jnp.clip(
    jnp.searchsorted(keyframe_times, times, side="right") - 1, 0, len(segments) - 1
)
s = (times - keyframe_times[segment_indices]) / (
    keyframe_times[segment_indices + 1] - keyframe_times[segment_indices]
)
T_baseline = jt.compose_transforms(
    T_keyframes[segment_indices],
    jt.transform_from_exponential_coordinates(s[:, None] * segments[segment_indices]),
)


# %%
# Cost Function
# -------------
# The body velocity between two consecutive poses is
#
# .. math::
#
#     \mathcal{V}_k = \frac{1}{\Delta t}
#     Log\left(\boldsymbol{T}_k^{-1} \boldsymbol{T}_{k+1}\right)
#     \in \mathbb{R}^6
#
# and we approximate accelerations with finite differences
# :math:`\dot{\mathcal{V}}_k = (\mathcal{V}_{k+1} - \mathcal{V}_k) / \Delta t`.
# We formulate the cost as a sum of squared residuals
#
# .. math::
#
#     c = \frac{1}{2} \sum_k \Delta t \|\boldsymbol{W} \dot{\mathcal{V}}_k\|^2
#     + \frac{w_c^2}{2} \sum_k \sum_i
#     \max\left(0, m - d\left(\boldsymbol{T}_k \boldsymbol{p}_i\right)\right)^2
#     + \frac{w_c^2}{2} \sum_k \left\|
#     \max(\boldsymbol{0}, \boldsymbol{t}_k - \boldsymbol{t}_{max})
#     + \max(\boldsymbol{0}, \boldsymbol{t}_{min} - \boldsymbol{t}_k)
#     \right\|^2,
#
# where the first term measures smoothness (:math:`\boldsymbol{W}` weights
# angular and linear accelerations), the second term penalizes points
# :math:`\boldsymbol{p}_i` on the robot's surface with a signed distance
# :math:`d` to the environment that is smaller than the safety margin
# :math:`m`, and the third term keeps the robot's position
# :math:`\boldsymbol{t}_k` within the bounds of the benchmark. The building
# has no roof, so without bounds the robot would fly over the walls.
#
# We do not optimize poses directly. Instead, we perturb each pose in its
# local tangent space, :math:`\boldsymbol{T}_k Exp(\boldsymbol{\delta}_k)`,
# with :math:`\boldsymbol{\delta}_k \in \mathbb{R}^6`. Start and goal are hard
# constraints: their perturbations are masked out.
acceleration_weights = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
acceleration_weights = acceleration_weights.at[3:].divide(characteristic_length)
collision_weight = 1.0
safety_margin = 10.0
free = jnp.ones(n_steps).at[jnp.array([0, n_steps - 1])].set(0.0)


def retract(T, delta):
    return jt.compose_transforms(
        T, jt.transform_from_exponential_coordinates(free[:, None] * delta)
    )


def body_velocities(T):
    T_rel = jt.compose_transforms(jt.transform_inverse(T[:-1]), T[1:])
    return jt.exponential_coordinates_from_transform(T_rel) / dt


def smoothness_residuals(delta, T):
    accelerations = jnp.diff(body_velocities(retract(T, delta)), axis=0) / dt
    return (jnp.sqrt(dt) * acceleration_weights * accelerations).ravel()


def collision_residuals(delta, T):
    T = jt.compose_transforms(T, jt.transform_from_exponential_coordinates(delta))
    distances = signed_distance(jt.apply_transform(T, robot_points))
    r_collision = jnp.maximum(0.0, safety_margin - distances)
    r_bounds = jnp.maximum(0.0, volume_min - T[:3, 3]) + jnp.maximum(
        0.0, T[:3, 3] - volume_max
    )
    return collision_weight * jnp.concatenate((r_collision, r_bounds))


@jax.jit
def cost(T):
    delta = jnp.zeros((n_steps, 6))
    return 0.5 * (
        jnp.sum(smoothness_residuals(delta, T) ** 2)
        + jnp.sum(jax.vmap(collision_residuals)(delta, T) ** 2)
    )


# %%
# Optimization
# ------------
# We use the Levenberg-Marquardt algorithm. JAX computes the Jacobian of the
# residuals with respect to the perturbations :math:`\boldsymbol{\delta}` at
# :math:`\boldsymbol{\delta} = \boldsymbol{0}` with forward-mode automatic
# differentiation. The collision residuals of a pose only depend on its own
# perturbation, hence we compute the corresponding blocks of the Jacobian
# separately for each pose with :func:`jax.vmap`.
# After each step, we apply the perturbation to the poses (retraction), so
# that we always linearize around the current trajectory. The damping is
# adapted depending on whether a step decreases the cost. We stop when the
# cost does not decrease significantly anymore.
@jax.jit
def levenberg_marquardt_step(T, damping):
    delta0 = jnp.zeros((n_steps, 6))
    r_smooth = smoothness_residuals(delta0, T)
    J_smooth = jax.jacfwd(smoothness_residuals)(delta0, T).reshape(-1, n_steps * 6)
    r_collision = jax.vmap(collision_residuals)(delta0, T)
    J_collision = jax.vmap(jax.jacfwd(collision_residuals))(delta0, T)
    J_collision *= free[:, None, None]
    H = J_smooth.T @ J_smooth + damping * jnp.eye(n_steps * 6)
    H_collision = jnp.einsum("kpi,kpj->kij", J_collision, J_collision)
    H = H.reshape(n_steps, 6, n_steps, 6)
    H = H.at[jnp.arange(n_steps), :, jnp.arange(n_steps), :].add(H_collision)
    g = (
        J_smooth.T @ r_smooth
        + jnp.einsum("kpi,kp->ki", J_collision, r_collision).ravel()
    )
    delta = -jnp.linalg.solve(H.reshape(n_steps * 6, n_steps * 6), g)
    T_new = retract(T, delta.reshape(n_steps, 6))
    return T_new, cost(T_new)


T = T_baseline
damping = 1e-2
costs = [float(cost(T))]
for iteration in range(500):
    T_candidate, candidate_cost = levenberg_marquardt_step(T, damping)
    if candidate_cost < costs[-1]:  # accept step
        T = T_candidate
        costs.append(float(candidate_cost))
        damping /= 3.0
        print(f"{iteration=}, cost={costs[-1]:.4f}, {damping=:.1e}")
        if costs[-2] - costs[-1] < 1e-5 * costs[-2]:
            break
    else:  # reject step
        damping *= 4.0
        if damping > 1e8:
            break
T_optimized = T


# %%
# Evaluation
# ----------
# We compare angular and linear speed, accelerations, and the clearance, i.e.,
# the minimum signed distance between the robot and the environment, of both
# trajectories. We compute the clearance with the exact signed distance and
# the dense set of points on the robot's surface. In addition, we check the
# motion between consecutive poses at 10 intermediate time steps.
def speeds(T):
    V = body_velocities(T)
    angular_speed = jnp.linalg.norm(V[:, :3], axis=-1)
    linear_speed = jnp.linalg.norm(jnp.diff(T[:, :3, 3], axis=0), axis=-1) / dt
    return np.asarray(angular_speed), np.asarray(linear_speed)


def max_acceleration(T):
    accelerations = jnp.diff(body_velocities(T), axis=0) / dt
    return (
        float(jnp.linalg.norm(accelerations[:, :3], axis=-1).max()),
        float(jnp.linalg.norm(accelerations[:, 3:], axis=-1).max()),
    )


def clearance(T, n_substeps=1):
    s = jnp.linspace(0.0, 1.0, n_substeps, endpoint=False)
    exp_coords = s[None, :, None] * body_velocities(T)[:, None] * dt
    T_fine = jt.compose_transforms(
        jnp.broadcast_to(T[:-1, None], exp_coords.shape[:2] + (4, 4)),
        jt.transform_from_exponential_coordinates(exp_coords),
    )
    T_fine = jnp.concatenate((T_fine.reshape(-1, 4, 4), T[-1:]))
    points = jax.vmap(jt.apply_transform, (0, None))(T_fine, dense_robot_points)
    distances = exact_signed_distance(points.reshape(-1, 3))
    return np.asarray(distances.reshape(len(T_fine), -1).min(axis=1))


clearance_baseline = clearance(T_baseline)
clearance_optimized = clearance(T_optimized)
for name, T, clearance_profile in [
    ("Baseline", T_baseline, clearance_baseline),
    ("Optimized", T_optimized, clearance_optimized),
]:
    angular_speed, linear_speed = speeds(T)
    max_angular_acc, max_linear_acc = max_acceleration(T)
    print(
        f"{name}: cost={float(cost(T)):.3f}, "
        f"max. speed={angular_speed.max():.2f} rad/s, {linear_speed.max():.1f} u/s, "
        f"max. acc.={max_angular_acc:.2f} rad/s^2, {max_linear_acc:.1f} u/s^2, "
        f"clearance={clearance_profile.min():.2f}, "
        f"clearance between poses={clearance(T, 10).min():.2f}"
    )

# %%
# Plotting
# --------
# The 3D plot shows the building, both trajectories, and the robot at a few
# poses along the optimized trajectory. Colors indicate time, from purple
# (start) to yellow (goal). The top views show the walls of the upper and the
# lower floor, the keyframes (gray dots), and poses of the initial trajectory
# that collide with the building (red crosses). The optimized trajectory goes
# around the corners and passes through the doors. On the right side, we see
# that the speed profiles of the optimized trajectory do not have
# discontinuities and that it keeps a positive clearance. Note that we do not
# minimize speeds but accelerations, so the speed is not constant.
P_baseline = np.asarray(T_baseline[:, :3, 3])
P_optimized = np.asarray(T_optimized[:, :3, 3])
snapshot_indices = np.linspace(0, n_steps - 1, 14).astype(int)
snapshot_colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(snapshot_indices)))
collisions = clearance_baseline < 0.0


def robot_mesh(T):
    vertices = jax.vmap(jt.apply_transform, (None, 0))(T, jnp.asarray(robot_triangles))
    return np.asarray(vertices)


fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(4, 3, width_ratios=(1.0, 1.0, 0.9), wspace=0.25, hspace=0.4)

ax = ppu.make_3d_axis(ax_s=1.0, pos=gs[:2, :2], n_ticks=5)
ax.add_collection3d(
    Poly3DCollection(
        env_triangles, facecolor="0.85", edgecolor="0.4", linewidths=0.2, alpha=0.1
    )
)
ax.plot(*P_baseline.T, c="gray", ls="--", lw=1.5, label="Keyframes + ScLERP")
ax.plot(*P_optimized.T, c="k", lw=2, label="Optimized")
for i, color in zip(snapshot_indices, snapshot_colors, strict=True):
    ax.add_collection3d(
        Poly3DCollection(robot_mesh(T_optimized[i]), facecolor=color, alpha=0.9)
    )
ax.set_xlim((volume_min[0], volume_max[0]))
ax.set_ylim((volume_min[1], volume_max[1]))
ax.set_zlim((volume_min[2], volume_max[2]))
ax.set_box_aspect(volume_max - volume_min, zoom=1.15)
ax.view_init(elev=40, azim=-60)
ax.set_title("Cubicles benchmark (OMPL)")
ax.legend(loc="upper left")

# top views: walls are vertical triangles, the upper floor is above z = 0
normals = np.cross(
    env_triangles[:, 1] - env_triangles[:, 0], env_triangles[:, 2] - env_triangles[:, 0]
)
walls = env_triangles[np.abs(normals[:, 2]) < 1e-3 * np.linalg.norm(normals, axis=1)]
lower_floor_x_min = walls[walls[:, :, 2].mean(axis=1) < 0.0][:, :, 0].min() - 10.0
gs_top = gs[2:, :2].subgridspec(
    1,
    2,
    width_ratios=(volume_max[0] - volume_min[0], volume_max[0] - lower_floor_x_min),
)
for gs_pos, title, upper_floor, x_limits in [
    (gs_top[0], "Upper floor (top view)", True, (volume_min[0], volume_max[0])),
    (gs_top[1], "Lower floor", False, (lower_floor_x_min, volume_max[0])),
]:
    ax_top = fig.add_subplot(gs_pos)
    floor_walls = walls[(walls[:, :, 2].mean(axis=1) > 0.0) == upper_floor]
    ax_top.add_collection(
        PolyCollection(floor_walls[:, :, :2], edgecolor="0.3", facecolor="none")
    )
    on_floor = (P_optimized[:, 2] > 0.0) == upper_floor
    for i, color in zip(snapshot_indices, snapshot_colors, strict=True):
        if on_floor[i]:
            robot_footprint = robot_mesh(T_optimized[i])[:, :, :2]
            ax_top.add_collection(PolyCollection(robot_footprint, color=color))
    baseline_on_floor = (P_baseline[:, 2] > 0.0) == upper_floor
    P = np.where(baseline_on_floor[:, None], P_baseline, np.nan)
    ax_top.plot(P[:, 0], P[:, 1], c="gray", ls="--")
    P = np.where(on_floor[:, None], P_optimized, np.nan)
    ax_top.plot(P[:, 0], P[:, 1], c="k", lw=2)
    keyframes_on_floor = (keyframes[:, 2] > 0.0) == upper_floor
    ax_top.scatter(*keyframes[keyframes_on_floor, :2].T, c="gray", s=20, zorder=3)
    P = P_baseline[collisions & baseline_on_floor]
    ax_top.scatter(P[:, 0], P[:, 1], c="tab:red", marker="x", s=25, zorder=3)
    ax_top.set_xlim(x_limits)
    ax_top.set_ylim((volume_min[1], volume_max[1]))
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("x")
    ax_top.set_title(title)
    if upper_floor:
        ax_top.set_ylabel("y")

ax_angular = fig.add_subplot(gs[0, 2])
ax_linear = fig.add_subplot(gs[1, 2], sharex=ax_angular)
ax_clearance = fig.add_subplot(gs[2, 2], sharex=ax_angular)
for name, T, clearance_profile, style in [
    ("Keyframes + ScLERP", T_baseline, clearance_baseline, dict(c="gray", ls="--")),
    ("Optimized", T_optimized, clearance_optimized, dict(c="k")),
]:
    angular_speed, linear_speed = speeds(T)
    ax_angular.plot(times[:-1], angular_speed, label=name, **style)
    ax_linear.plot(times[:-1], linear_speed, label=name, **style)
    ax_clearance.plot(times, clearance_profile, label=name, **style)
ax_clearance.axhspan(clearance_baseline.min() - 5.0, 0.0, color="tab:red", alpha=0.15)
ax_clearance.axhline(safety_margin, c="tab:red", ls=":", lw=1)
for ax_profile in (ax_angular, ax_linear, ax_clearance):
    ax_profile.grid(alpha=0.3)
ax_angular.set_title("Profiles")
ax_angular.set_ylabel("Angular speed [rad/s]")
ax_angular.legend(loc="upper left")
ax_linear.set_ylabel("Linear speed [units/s]")
ax_clearance.set_ylabel("Clearance [units]")
ax_clearance.set_xlabel("Time [s]")
ax_clearance.set_title("Clearance (dotted: safety margin)")

ax_cost = fig.add_subplot(gs[3, 2])
ax_cost.semilogy(costs, c="k")
ax_cost.set_title("Levenberg-Marquardt")
ax_cost.set_xlabel("Accepted step")
ax_cost.set_ylabel("Cost")
ax_cost.grid(alpha=0.3)

fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.05)
plt.show()
