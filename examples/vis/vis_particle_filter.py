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
    log_weights = jnp.full(n_particles, -np.log(n_particles), dtype=particles.dtype)
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
# We display a random subset of the particles. Their color shows their heading,
# so we can see that the heading is unknown in the beginning. The drone model
# shows the true pose and the coordinate frame the estimated pose. The
# wireframe shows the positions in the room that agree with the last measured
# distance to the highlighted anchor.
n_displayed_particles = 100_000


def colors_from_headings(R):
    """Map heading (yaw angle) to a color on the hue circle."""
    hue = (np.arctan2(R[:, 1, 0], R[:, 0, 0]) / (2.0 * np.pi)) % 1.0
    channels = np.abs(
        (6.0 * hue[:, np.newaxis] + np.array([0.0, 4.0, 2.0])) % 6.0 - 3.0
    )
    return np.clip(channels - 1.0, 0.0, 1.0)


class Particles(pv.Artist):
    """Particle positions colored by heading."""

    def __init__(self):
        self.pcd = o3d.geometry.PointCloud()

    def set_data(self, poses):
        self.pcd.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
        self.pcd.colors = o3d.utility.Vector3dVector(
            colors_from_headings(poses[:, :3, :3])
        )

    @property
    def geometries(self):
        return [self.pcd]


class RangeSphere(pv.Artist):
    """Positions in the room with a given distance to an anchor."""

    def __init__(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        self.unit_sphere = np.asarray(sphere.vertices)
        self.all_lines = np.asarray(lines.lines)
        self.lines = o3d.geometry.LineSet()

    def set_data(self, center, radius):
        points = center + radius * self.unit_sphere
        inside = np.all(
            (points >= np.asarray(room_min)) & (points <= np.asarray(room_max)),
            axis=1,
        )
        lines = self.all_lines[np.all(inside[self.all_lines], axis=1)]
        self.lines.points = o3d.utility.Vector3dVector(points)
        self.lines.lines = o3d.utility.Vector2iVector(lines)
        self.lines.paint_uniform_color((0.2, 0.4, 0.9))

    @property
    def geometries(self):
        return [self.lines]


class Drone(pv.Artist):
    """Simple model of a quadrotor."""

    def __init__(self, arm_length=0.35):
        self.mesh = o3d.geometry.TriangleMesh()
        for angle in np.deg2rad([45.0, 135.0]):
            arm = o3d.geometry.TriangleMesh.create_box(2.0 * arm_length, 0.03, 0.02)
            arm.translate((-arm_length, -0.015, -0.01))
            arm.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0, 0, angle)))
            self.mesh += arm
        for angle in np.deg2rad([45.0, 135.0, 225.0, 315.0]):
            rotor = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.1, height=0.02, resolution=24
            )
            rotor.translate(
                (arm_length * np.cos(angle), arm_length * np.sin(angle), 0.02)
            )
            self.mesh += rotor
        body = o3d.geometry.TriangleMesh.create_box(0.12, 0.08, 0.05)
        self.mesh += body.translate((-0.06, -0.04, -0.025))
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color((0.25, 0.25, 0.25))
        self.vertices = np.asarray(self.mesh.vertices).copy()

    def set_data(self, T):
        vertices = self.vertices @ T[:3, :3].T + T[:3, 3]
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.compute_vertex_normals()

    @property
    def geometries(self):
        return [self.mesh]


class Anchors(pv.Artist):
    """UWB anchors, the measured one is highlighted."""

    def __init__(self, positions):
        self.spheres = []
        for position in positions:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(position)
            sphere.compute_vertex_normals()
            self.spheres.append(sphere)
        self.set_data(None)

    def set_data(self, active):
        for i, sphere in enumerate(self.spheres):
            color = (0.2, 0.4, 0.9) if i == active else (0.6, 0.6, 0.6)
            sphere.paint_uniform_color(color)

    @property
    def geometries(self):
        return self.spheres


def room_geometries():
    floor = o3d.geometry.TriangleMesh.create_box(*np.asarray(room_max[:2]), 0.01)
    floor.translate((0.0, 0.0, -0.01))
    floor.paint_uniform_color((0.85, 0.85, 0.85))
    floor.compute_vertex_normals()
    walls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(np.asarray(room_min), np.asarray(room_max))
    )
    walls.paint_uniform_color((0.4, 0.4, 0.4))
    return [floor, walls]


def animation_callback(
    frame, particle_artist, drone, estimate, anchor_artist, range_sphere, trajectory
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
    displayed_particles = np.asarray(particles[displayed])
    duration = time.perf_counter() - start
    if frame % 50 == 0:
        print(
            f"t = {frame * steps_per_frame * dt:4.1f} s: {steps_per_frame} steps "
            f"with {n_particles} particles in {1000 * duration:.1f} ms"
        )

    step = steps.stop - 1
    particle_artist.set_data(displayed_particles)
    drone.set_data(np.asarray(true_poses[step]))
    estimate.set_data(np.asarray(T_mean))
    last_measurement = step - (step + 1) % measurement_every
    if last_measurement >= 0:
        active = int(jnp.argmax(measured_anchor[last_measurement]))
        anchor_artist.set_data(active)
        range_sphere.set_data(
            np.asarray(anchors[active]),
            float(measured_ranges[last_measurement, active]),
        )
    recent = np.asarray(true_poses[max(0, step - int(10.0 / dt)) : step + 1, :3, 3])
    trajectory.set_data(recent, c=(0.0, 0.0, 0.0))
    return particle_artist, drone, estimate, anchor_artist, range_sphere, trajectory


filter_key = jax.random.PRNGKey(42)
displayed = jax.random.choice(
    jax.random.PRNGKey(1), n_particles, (n_displayed_particles,), replace=False
)
particles, log_weights = initialize_particles(filter_key)

fig = pv.figure()
for geometry in room_geometries():
    fig.add_geometry(geometry)
particle_artist = Particles()
particle_artist.set_data(np.asarray(particles[displayed]))
drone = Drone()
drone.set_data(np.asarray(T_start))
estimate = pv.Frame(np.eye(4), s=0.5)
anchor_artist = Anchors(np.asarray(anchors))
range_sphere = RangeSphere()
range_sphere.set_data(np.zeros(3), 0.0)
trajectory = pv.Line3D(np.asarray(true_poses[:2, :3, 3]), c=(0.0, 0.0, 0.0))
artists = (particle_artist, drone, estimate, anchor_artist, range_sphere, trajectory)
for artist in artists:
    artist.add_artist(fig)
fig.visualizer.get_render_option().point_size = 2.0
fig.view_init(elev=35, azim=-60)

n_frames = n_steps // steps_per_frame
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True, fargs=artists)
    fig.show()
else:
    for frame in range(150):
        animation_callback(frame, *artists)
    fig.save_image("__open3d_rendered_image.jpg")
