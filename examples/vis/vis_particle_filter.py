r"""
=======================================================
Global Localization of a Drone with a Particle Filter
=======================================================

A drone flies through a room that is equipped with ultra-wideband (UWB)
anchors. It measures the distances to all anchors and its own velocity, but
it does not know where it starts. We estimate its 6D pose with a particle
filter on :math:`SE(3)` that uses one million particles. In the beginning,
the particles are spread over the whole room and the belief has many modes.
Particle filters can represent such beliefs, but they need many particles to
cover the space. With JAX, we propagate, weight, and resample all particles
in parallel on the GPU in real time.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv

import jaxtransform3d.rotations as jr
import jaxtransform3d.transformations as jt

# %%
# Simulation
# ----------
# The drone flies with a time-varying body twist
# :math:`\mathcal{V}(t) = (\boldsymbol{\omega}(t), \boldsymbol{v}(t))`
# (angular and linear velocity in the body frame), which we integrate with
# the exponential map :math:`\boldsymbol{T}_{k+1} = \boldsymbol{T}_k
# \exp(\Delta t \mathcal{V}_k)`.
dt = 0.02
n_steps = 3000
measurement_every = 5  # one range every 0.1 s, velocities at 50 Hz
room_min = jnp.array([0.0, 0.0, 0.0])
room_max = jnp.array([8.0, 6.0, 3.0])
anchors = jnp.array(
    [
        [0.2, 0.2, 2.8],
        [7.8, 0.2, 0.3],
        [7.8, 5.8, 2.8],
        [0.2, 5.8, 0.3],
    ]
)

# The drone circles with a radius of 2 m with varying speed and altitude.
t = dt * jnp.arange(n_steps)
speed = 1.0 + 0.3 * jnp.sin(0.4 * t)
true_twists = jnp.stack(
    [
        jnp.zeros_like(t),
        jnp.zeros_like(t),
        speed / 2.0,
        speed,
        jnp.zeros_like(t),
        0.4 * jnp.cos(0.5 * t),
    ],
    axis=-1,
)


def integrate(T_init, twists):
    """Integrate a sequence of body twists, return poses after each step."""

    def step(T, twist):
        T = jt.compose_transforms(
            T, jt.transform_from_exponential_coordinates(dt * twist)
        )
        return T, T

    return jax.lax.scan(step, T_init, twists)[1]


T_start = jt.create_transform(jnp.eye(3), jnp.array([4.0, 1.4, 1.2]))
true_poses = integrate(T_start, true_twists)

# %%
# Sensors
# -------
# The drone measures its twist (e.g., with an IMU and optical flow) with white
# noise and a small bias. Every 0.1 s, it measures the distance to one of the
# anchors (in turn) with a standard deviation of 10 cm. A single range only
# tells us that the drone is on a sphere around the anchor. Roll and pitch are
# observable from the direction of gravity, but the heading and the position
# are unknown in the beginning.
key = jax.random.PRNGKey(0)
key, key_twist, key_range = jax.random.split(key, 3)
twist_bias = jnp.array([0.005, -0.005, 0.02, 0.05, -0.03, 0.0])
twist_std = jnp.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
measured_twists = (
    true_twists + twist_bias + twist_std * jax.random.normal(key_twist, (n_steps, 6))
)

range_std = 0.1
true_ranges = jnp.linalg.norm(
    true_poses[:, jnp.newaxis, :3, 3] - anchors[jnp.newaxis], axis=-1
)
measured_ranges = true_ranges + range_std * jax.random.normal(
    key_range, true_ranges.shape
)
has_measurement = (jnp.arange(n_steps) + 1) % measurement_every == 0
measured_anchor = (
    jax.nn.one_hot(
        (jnp.arange(n_steps) // measurement_every) % len(anchors), len(anchors)
    )
    * has_measurement[:, jnp.newaxis]
)

# %%
# Particle Filter
# ---------------
# Each particle is a full pose. We initialize the positions uniformly in the
# room and the heading uniformly in :math:`[-\pi, \pi)`. The filter has three
# steps:
#
# * **Prediction**: propagate each particle with the measured twist and a
#   sampled perturbation :math:`\boldsymbol{\epsilon}_i` in the tangent space,
#   :math:`\boldsymbol{T}_i \leftarrow \boldsymbol{T}_i
#   \exp(\Delta t \tilde{\mathcal{V}} + \boldsymbol{\epsilon}_i)`.
# * **Update**: multiply the weights by the likelihood of the measured range
#   :math:`\mathcal{N}(r | \|\boldsymbol{p}_i - \boldsymbol{a}\|,
#   \sigma^2)` to anchor :math:`\boldsymbol{a}`. Particles outside of the
#   room get the weight 0.
# * **Resampling**: when the effective sample size drops below half of the
#   number of particles, draw a new set of particles with systematic
#   resampling.
#
# The estimate is the weighted mean on :math:`SE(3)`, computed iteratively
# with logarithmic and exponential maps in the tangent space. It is only
# meaningful once the belief has a single mode.
n_particles = 1_000_000
process_std = jnp.array([0.02, 0.02, 0.05, 0.1, 0.1, 0.1])


def initialize_particles(key):
    key_position, key_heading, key_tilt = jax.random.split(key, 3)
    positions = jax.random.uniform(
        key_position, (n_particles, 3), minval=room_min, maxval=room_max
    )
    axis_angles = jnp.concatenate(
        (
            0.03 * jax.random.normal(key_tilt, (n_particles, 2)),
            jax.random.uniform(
                key_heading, (n_particles, 1), minval=-jnp.pi, maxval=jnp.pi
            ),
        ),
        axis=-1,
    )
    R = jr.matrix_from_compact_axis_angle(axis_angles)
    particles = jt.create_transform(R, positions)
    log_weights = jnp.full(n_particles, -jnp.log(n_particles))
    return particles, log_weights


def systematic_resampling(key, weights):
    positions = (jax.random.uniform(key) + jnp.arange(n_particles)) / n_particles
    indices = jnp.searchsorted(jnp.cumsum(weights), positions)
    return jnp.minimum(indices, n_particles - 1)


def mean_pose(poses, weights, n_iter=3):
    T_mean = poses[jnp.argmax(weights)]

    def update(_, T_mean):
        xi = jt.exponential_coordinates_from_transform(
            jt.compose_transforms(jt.transform_inverse(T_mean), poses)
        )
        return jt.compose_transforms(
            T_mean, jt.transform_from_exponential_coordinates(weights @ xi)
        )

    return jax.lax.fori_loop(0, n_iter, update, T_mean)


def filter_step(carry, inputs):
    particles, log_weights, key = carry
    twist, ranges, anchor_mask = inputs
    measurement_available = jnp.any(anchor_mask > 0.0)
    key, key_noise, key_resample = jax.random.split(key, 3)

    # prediction
    noise = jnp.sqrt(dt) * process_std * jax.random.normal(key_noise, (n_particles, 6))
    particles = jt.compose_transforms(
        particles, jt.transform_from_exponential_coordinates(dt * twist + noise)
    )

    # update
    positions = particles[:, :3, 3]
    predicted_ranges = jnp.linalg.norm(
        positions[:, jnp.newaxis] - anchors[jnp.newaxis], axis=-1
    )
    log_likelihood = -0.5 * (predicted_ranges - ranges) ** 2 @ anchor_mask
    log_likelihood /= range_std**2
    inside = jnp.all((positions >= room_min) & (positions <= room_max), axis=-1)
    log_likelihood = jnp.where(inside, log_likelihood, -jnp.inf)
    log_weights = jnp.where(
        measurement_available, log_weights + log_likelihood, log_weights
    )
    log_weights -= jax.nn.logsumexp(log_weights)
    weights = jnp.exp(log_weights)

    # resampling
    ess = 1.0 / jnp.sum(weights**2)
    resample = measurement_available & (ess < n_particles / 2)
    indices = systematic_resampling(key_resample, weights)
    particles = jnp.where(resample, particles[indices], particles)
    log_weights = jnp.where(resample, -jnp.log(n_particles), log_weights)

    return (particles, log_weights, key), None


# %%
# In each frame of the animation, we perform several steps of the filter in a
# single call of a compiled function.
steps_per_frame = 1


@jax.jit
def filter_steps(particles, log_weights, key, twists, ranges, anchor_masks):
    (particles, log_weights, key), _ = jax.lax.scan(
        filter_step, (particles, log_weights, key), (twists, ranges, anchor_masks)
    )
    T_mean = mean_pose(particles, jnp.exp(log_weights))
    return particles, log_weights, key, T_mean


# %%
# Visualization
# -------------
# We display a random subset of the particles, the true pose of the drone, the
# estimated pose, and the anchors with the measured distances.
n_displayed_particles = 100_000


class PointCloud(pv.Artist):
    """Point cloud that can be updated in an animation."""

    def __init__(self, points, color):
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.pcd.paint_uniform_color(color)

    def set_data(self, points):
        self.pcd.points = o3d.utility.Vector3dVector(points)

    @property
    def geometries(self):
        return [self.pcd]


def animation_callback(
    frame, particle_cloud, true_frame, estimate_frame, range_lines, trajectory
):
    global particles, log_weights, filter_key

    if frame == 0:  # new global localization whenever the animation restarts
        filter_key, key_init = jax.random.split(filter_key)
        particles, log_weights = initialize_particles(key_init)

    steps = slice(frame * steps_per_frame, (frame + 1) * steps_per_frame)
    start = time.perf_counter()
    particles, log_weights, filter_key, T_mean = filter_steps(
        particles,
        log_weights,
        filter_key,
        measured_twists[steps],
        measured_ranges[steps],
        measured_anchor[steps],
    )
    positions = np.asarray(particles[displayed, :3, 3])
    duration = time.perf_counter() - start
    if frame % 25 == 0:
        print(
            f"t = {frame * steps_per_frame * dt:4.1f} s: {steps_per_frame} steps "
            f"with {n_particles} particles in {1000 * duration:.1f} ms"
        )

    T_true = np.asarray(true_poses[steps.stop - 1])
    particle_cloud.set_data(positions)
    true_frame.set_data(T_true)
    estimate_frame.set_data(np.asarray(T_mean))
    range_lines.set_data(
        np.vstack([np.vstack((a, T_true[:3, 3])) for a in np.asarray(anchors)]),
        c=(0.6, 0.6, 1.0),
    )
    trajectory.set_data(np.asarray(true_poses[: steps.stop, :3, 3]), c=(0, 0, 0))
    return particle_cloud, true_frame, estimate_frame, range_lines, trajectory


filter_key = jax.random.PRNGKey(42)
displayed = jax.random.choice(
    jax.random.PRNGKey(1), n_particles, (n_displayed_particles,), replace=False
)
particles, log_weights = initialize_particles(filter_key)

fig = pv.figure()
room = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
    o3d.geometry.AxisAlignedBoundingBox(np.asarray(room_min), np.asarray(room_max))
)
room.paint_uniform_color((0.3, 0.3, 0.3))
fig.add_geometry(room)
for anchor in np.asarray(anchors):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    sphere.translate(anchor)
    sphere.paint_uniform_color((0.1, 0.1, 0.8))
    sphere.compute_vertex_normals()
    fig.add_geometry(sphere)
particle_cloud = PointCloud(np.asarray(particles[displayed, :3, 3]), (1.0, 0.5, 0.0))
particle_cloud.add_artist(fig)
true_frame = fig.plot_transform(np.asarray(T_start), s=0.5, strict_check=False)
estimate_frame = fig.plot_transform(np.eye(4), s=0.8, strict_check=False)
range_lines = pv.Line3D(np.zeros((2 * len(anchors), 3)), c=(0.6, 0.6, 1.0))
range_lines.add_artist(fig)
trajectory = pv.Line3D(np.zeros((2, 3)), c=(0, 0, 0))
trajectory.add_artist(fig)
fig.view_init(elev=35, azim=-60)

n_frames = n_steps // steps_per_frame
fargs = (particle_cloud, true_frame, estimate_frame, range_lines, trajectory)
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True, fargs=fargs)
    fig.show()
else:
    for frame in range(150):
        animation_callback(frame, *fargs)
    fig.save_image("__open3d_rendered_image.jpg")
