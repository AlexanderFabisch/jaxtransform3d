r"""
============================================================
Differentiable Rigid Body Simulation: The Perfect Phone Flip
============================================================

Throw your phone so that it flips three times about its short axis and catch
it: it will likely land screen-down. This is the tennis racket theorem (or
Dzhanibekov effect): rotation of a rigid body about its intermediate
principal axis of inertia is unstable. Small perturbations grow
exponentially and the phone makes an additional half twist about its long
axis.

We simulate the rotation of the phone with Euler's equations and the
exponential map of :math:`SO(3)`, differentiate through the whole simulation
with JAX, and search for a throw that lands screen-up even if we cannot throw
precisely. The problem has many local minima, so we optimize thousands of
throws in parallel, each of which is evaluated with dozens of noisy
simulations. This requires about 20 million simulations with gradients and
runs on a GPU in less than a minute.
"""

import base64
import io
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import jaxtransform3d.rotations as jr

# %%
# Rigid Body Dynamics
# -------------------
# The phone is a box with a length of 15 cm, a width of 7.5 cm, a thickness of
# 8 mm, and a mass of 200 g. In its body frame, the x-axis is the long axis,
# the y-axis the short axis, and the z-axis points out of the screen. The
# principal moments of inertia are :math:`I_x < I_y < I_z`, so the short axis
# is the intermediate axis.
mass = 0.2
size = np.array([0.15, 0.075, 0.008])
inertia = jnp.asarray(mass / 12.0 * (np.sum(size**2) - size**2))
print(f"principal moments of inertia: {np.asarray(inertia)} kg m^2")

# %%
# Without external torques (gravity acts on the center of mass), the angular
# velocity :math:`\boldsymbol{\omega}` in the body frame follows Euler's
# equations
#
# .. math::
#
#     \boldsymbol{I} \dot{\boldsymbol{\omega}} = -\boldsymbol{\omega} \times
#     (\boldsymbol{I} \boldsymbol{\omega}).
#
# We integrate them with the Runge-Kutta method of fourth order. We update
# the orientation with the exponential map,
# :math:`\boldsymbol{R}_{k+1} = \boldsymbol{R}_k
# \exp(\Delta t \bar{\boldsymbol{\omega}}_k)`, with the angular velocity
# :math:`\bar{\boldsymbol{\omega}}_k` at the middle of the time step. Hence,
# the orientation is always a valid rotation matrix.
flight_time = 1.0
n_steps = 250
dt = flight_time / n_steps


def angular_acceleration(omega):
    return -jnp.cross(omega, inertia * omega) / inertia


def time_step(state, _):
    R, omega = state
    k1 = angular_acceleration(omega)
    k2 = angular_acceleration(omega + 0.5 * dt * k1)
    k3 = angular_acceleration(omega + 0.5 * dt * k2)
    k4 = angular_acceleration(omega + dt * k3)
    omega_mid = omega + 0.5 * dt * k2
    R = jr.compose_matrices(R, jr.matrix_from_compact_axis_angle(dt * omega_mid))
    omega = omega + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return (R, omega), R


def simulate(omega0):
    """Orientations of the phone during the flight."""
    _, orientations = jax.lax.scan(
        time_step, (jnp.eye(3), omega0), None, length=n_steps
    )
    return orientations


# %%
# To differentiate through the simulation, JAX stores intermediate results of
# all time steps. With ``jax.checkpoint``, it only stores the state and
# recomputes the rest during the backward pass. This allows us to
# differentiate through hundreds of thousands of simulations in parallel.
def landing_orientation(omega0):
    (R, _), _ = jax.lax.scan(
        jax.checkpoint(time_step), (jnp.eye(3), omega0), None, length=n_steps
    )
    return R


# %%
# The phone should land screen-up and with the same heading as in the
# beginning, i.e., with the identity as orientation. The landing error is the
# squared rotation angle between the final orientation and the identity.
def landing_error(omega0):
    return jnp.sum(jr.compact_axis_angle_from_matrix(landing_orientation(omega0)) ** 2)


# %%
# Tennis Racket Theorem
# ---------------------
# The phone should flip three times about its short axis during one second of
# flight. A perfect throw would only rotate about this axis. We compare it with
# throws that have a tiny additional rotation about the long axis.
n_flips = 3
omega_flip = 2.0 * np.pi * n_flips / flight_time
for omega_x in [0.0, 0.01, 0.1, 1.0]:
    error = jnp.sqrt(landing_error(jnp.array([omega_x, omega_flip, 0.0])))
    print(f"omega_x = {omega_x:4.2f} rad/s: landing error {np.degrees(error):5.1f} deg")

# %%
# Already with :math:`0.1` rad/s, the phone lands almost upside down. We
# compute the landing error for a grid of :math:`1000 \times 1000` additional
# angular velocities :math:`(\omega_x, \omega_z)` about the long axis and the
# axis normal to the screen. These are one million simulations, which we run
# in parallel with ``jax.vmap``.
limit = 15.0
n_grid = 1000
grid = np.linspace(-limit, limit, n_grid)
grid_points = jnp.asarray(np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2))


def with_flip(omega_xz):
    return jnp.stack(
        (
            omega_xz[..., 0],
            jnp.full_like(omega_xz[..., 0], omega_flip),
            omega_xz[..., 1],
        ),
        axis=-1,
    )


nominal_errors = jax.jit(jax.vmap(landing_error))
nominal_errors(with_flip(grid_points[:10]))  # compile
start = time.perf_counter()
nominal_landscape = np.asarray(nominal_errors(with_flip(grid_points)))
print(f"{len(grid_points)} simulations: {time.perf_counter() - start:.2f} s")
nominal_landscape = np.sqrt(nominal_landscape).reshape(n_grid, n_grid)

# %%
# Robust Throws
# -------------
# A pure rotation about the short axis would be perfect, but nobody can throw
# that precisely. We model throws with independent Gaussian noise with a
# standard deviation of :math:`\sigma = 0.3` rad/s on all components of the
# angular velocity and minimize the expected landing error, which we estimate
# with 32 noisy throws,
#
# .. math::
#
#     J(\omega_x, \omega_z) = \frac{1}{32} \sum_{s=1}^{32}
#     e\left((\omega_x, \omega_{flip}, \omega_z) + \boldsymbol{\epsilon}_s
#     \right),
#     \quad \boldsymbol{\epsilon}_s \sim \mathcal{N}(\boldsymbol{0},
#     \sigma^2 \boldsymbol{I}).
#
# We sample new noise in every iteration of the optimization. Small
# perturbations of the perfect throw result in large errors, so the robust
# solutions have to be somewhere else.
sigma = 0.3
n_noisy_throws = 32


def expected_landing_error(omega_xz, noise):
    return jnp.mean(jax.vmap(landing_error)(with_flip(omega_xz) + noise))


# %%
# For a smooth background in the plots, we also evaluate the expected error on
# a coarser grid of :math:`250 \times 250` points with 32 noisy throws each.
n_coarse = 250
coarse = np.linspace(-limit, limit, n_coarse)
coarse_points = jnp.asarray(
    np.stack(np.meshgrid(coarse, coarse), axis=-1).reshape(-1, 2)
)
landscape_noise = sigma * jax.random.normal(jax.random.PRNGKey(0), (n_noisy_throws, 3))
robust_landscape = np.asarray(
    jax.jit(jax.vmap(expected_landing_error, in_axes=(0, None)))(
        coarse_points, landscape_noise
    )
).reshape(n_coarse, n_coarse)

# %%
# Massively Parallel Optimization
# -------------------------------
# We start 4096 optimizations from random throws and update all of them with
# the Adam optimizer. In each iteration, we compute
# :math:`4096 \cdot 32 = 131072` simulations with gradients in one call of a
# compiled function.
n_candidates = 4096
n_iterations = 150
learning_rate = 0.2
beta1, beta2 = 0.9, 0.999

value_and_gradient = jax.jit(
    jax.vmap(jax.value_and_grad(expected_landing_error), in_axes=(0, None))
)


@jax.jit
def adam_step(x, m, v, gradient, t):
    m = beta1 * m + (1.0 - beta1) * gradient
    v = beta2 * v + (1.0 - beta2) * gradient**2
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    return x - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1e-8), m, v


key = jax.random.PRNGKey(42)
key, key_init = jax.random.split(key)
x = jax.random.uniform(key_init, (n_candidates, 2), minval=-10.0, maxval=10.0)
m = jnp.zeros_like(x)
v = jnp.zeros_like(x)
candidates = [np.asarray(x)]
losses = []
start = time.perf_counter()
for t in range(1, n_iterations + 1):
    key, key_noise = jax.random.split(key)
    noise = sigma * jax.random.normal(key_noise, (n_noisy_throws, 3))
    loss, gradient = value_and_gradient(x, noise)
    x, m, v = adam_step(x, m, v, gradient, t)
    candidates.append(np.asarray(x))
    losses.append(np.asarray(loss))
runtime = time.perf_counter() - start
candidates = np.array(candidates)
losses = np.array(losses)
n_simulations = n_candidates * n_noisy_throws * n_iterations
print(
    f"{n_simulations} simulations with gradients in {runtime:.1f} s (incl. compilation)"
)

# %%
# The losses of the last iteration are estimated with different noise
# samples for each iteration. We evaluate the final candidates with 1024 new
# noisy throws to select the best one.
evaluation_noise = sigma * jax.random.normal(jax.random.PRNGKey(7), (1024, 3))
final_errors = np.concatenate(
    [
        np.asarray(
            jax.jit(jax.vmap(expected_landing_error, in_axes=(0, None)))(
                jnp.asarray(batch), evaluation_noise
            )
        )
        for batch in np.array_split(candidates[-1], 16)
    ]
)
best = np.argmin(final_errors)
omega_best = np.r_[candidates[-1, best, 0], omega_flip, candidates[-1, best, 1]]
print(f"best throw: omega = {omega_best.round(2)} rad/s")
print(
    f"RMS landing error with noise: {np.degrees(np.sqrt(final_errors[best])):.1f} "
    f"deg (perfect throw with noise: "
    f"{np.degrees(np.sqrt(expected_landing_error(jnp.zeros(2), evaluation_noise))):.1f}"
    f" deg)"
)

# %%
# Optimization Process
# --------------------
# The left plot shows the landing error of throws without noise. The perfect
# throw is in the center. Along the dark lines, the phone lands screen-down
# (error close to 180 degrees). The middle plot shows the expected error with
# noise and the paths of 300 of the 4096 optimizations. Each path ends in one
# of several local minima. The right plot shows the distribution of the
# expected landing error over all candidates during the optimization.
fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
extent = (-limit, limit, -limit, limit)
im = axes[0].imshow(
    np.degrees(nominal_landscape), origin="lower", extent=extent, cmap="magma_r"
)
fig.colorbar(im, ax=axes[0], label="Landing error [deg]")
axes[0].set_title("Landing error without noise")

im = axes[1].imshow(
    np.degrees(np.sqrt(robust_landscape)),
    origin="lower",
    extent=extent,
    cmap="magma_r",
)
fig.colorbar(im, ax=axes[1], label="RMS landing error [deg]")
shown = np.linspace(0, n_candidates - 1, 300).astype(int)
for i in shown:
    axes[1].plot(*candidates[:, i].T, c="w", lw=0.5, alpha=0.5)
axes[1].scatter(*candidates[-1, shown].T, c="tab:cyan", s=6, zorder=3)
axes[1].scatter(*candidates[0, shown].T, c="w", s=4, zorder=3)
axes[1].scatter(
    *candidates[-1, best], marker="*", s=300, c="tab:green", ec="k", zorder=4
)
axes[1].set_title("Expected landing error and optimization paths")
for ax in axes[:2]:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r"$\omega_x$ (long axis) [rad/s]")
    ax.set_ylabel(r"$\omega_z$ (screen normal) [rad/s]")

percentiles = np.percentile(np.degrees(np.sqrt(losses)), [10, 50, 90], axis=1)
iterations = np.arange(1, n_iterations + 1)
axes[2].fill_between(
    iterations, percentiles[0], percentiles[2], alpha=0.3, label="10% - 90%"
)
axes[2].plot(iterations, percentiles[1], label="median")
axes[2].plot(
    iterations,
    np.degrees(np.sqrt(losses.min(axis=1))),
    c="tab:green",
    label="best",
)
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("RMS landing error [deg]")
axes[2].set_title(f"Convergence of {n_candidates} optimizations")
axes[2].legend()
axes[2].grid(alpha=0.3)
plt.show()

# %%
# The animation shows all 4096 candidates during the optimization on top of
# the expected landing error. Plotly removes heatmaps during animations, so we
# add the landscape as a background image.
frame_iterations = np.r_[0 : n_iterations + 1 : 5]
robust_degrees = np.degrees(np.sqrt(robust_landscape))
color_range = (robust_degrees.min(), robust_degrees.max())
background = io.BytesIO()
plt.imsave(
    background,
    robust_degrees,
    cmap="magma_r",
    vmin=color_range[0],
    vmax=color_range[1],
    origin="lower",
    format="png",
)
background_image = dict(
    source="data:image/png;base64," + base64.b64encode(background.getvalue()).decode(),
    xref="x",
    yref="y",
    x=-limit,
    y=limit,
    sizex=2 * limit,
    sizey=2 * limit,
    sizing="stretch",
    layer="below",
)
colorbar = go.Scatter(
    x=[None],
    y=[None],
    mode="markers",
    marker=dict(
        colorscale="Magma",
        reversescale=True,
        cmin=color_range[0],
        cmax=color_range[1],
        color=[color_range[0]],
        showscale=True,
        colorbar=dict(title="RMS error [deg]"),
    ),
    hoverinfo="skip",
    showlegend=False,
)


def candidate_trace(iteration):
    return go.Scatter(
        x=candidates[iteration, :, 0],
        y=candidates[iteration, :, 1],
        mode="markers",
        marker=dict(color="cyan", size=3, line=dict(width=0)),
        hoverinfo="skip",
        name="Candidates",
    )


animation = go.Figure(
    data=[colorbar, candidate_trace(0)],
    frames=[
        go.Frame(data=[candidate_trace(i)], traces=[1], name=str(i))
        for i in frame_iterations
    ],
)
animation.update_layout(
    title="4096 parallel optimizations",
    images=[background_image],
    xaxis=dict(
        title="omega_x [rad/s]", range=[-limit, limit], showgrid=False, zeroline=False
    ),
    yaxis=dict(
        title="omega_z [rad/s]",
        range=[-limit, limit],
        scaleanchor="x",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
    ),
    showlegend=False,
    height=650,
    width=700,
    updatemenus=[
        dict(
            type="buttons",
            x=0.0,
            y=-0.12,
            xanchor="left",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=150), fromcurrent=True)],
                )
            ],
        )
    ],
    sliders=[
        dict(
            x=0.12,
            y=-0.08,
            len=0.88,
            currentvalue=dict(prefix="Iteration "),
            steps=[
                dict(
                    method="animate",
                    label=str(i),
                    args=[[str(i)], dict(mode="immediate", frame=dict(duration=0))],
                )
                for i in frame_iterations
            ],
        )
    ],
)
animation.show()

# %%
# Throws in 3D
# ------------
# We compare a nearly perfect flip with a small error of :math:`0.1` rad/s
# about the long axis and the optimized throw. The phone is thrown 1.5 m
# forward and caught at the same height after one second. The dark side is
# the screen. The nearly perfect flip lands screen-down, the optimized throw
# twists about the long axis during the flight and lands screen-up.
gravity = np.array([0.0, 0.0, -9.81])
times = dt * np.arange(1, n_steps + 1)
velocity = np.array([1.5, 0.0, -0.5 * gravity[2] * flight_time])
positions = velocity * times[:, np.newaxis] + 0.5 * gravity * times[:, np.newaxis] ** 2

corners = (
    0.5
    * size
    * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
)
faces = np.array(
    [
        [1, 3, 7], [1, 7, 5],  # screen (z = +1)
        [0, 4, 6], [0, 6, 2],  # back (z = -1)
        [0, 1, 5], [0, 5, 4], [2, 6, 7], [2, 7, 3],  # sides
        [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
    ]
)  # fmt: skip
face_colors = ["rgb(20, 20, 30)"] * 2 + ["rgb(190, 190, 200)"] * 10

throws = [
    ("Nearly perfect flip: lands screen-down", np.array([0.1, omega_flip, 0.0])),
    ("Optimized throw: lands screen-up", omega_best),
]
throw_figure = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=[name for name, _ in throws],
    horizontal_spacing=0.02,
)
for column, (_, omega0) in enumerate(throws, start=1):
    orientations = np.asarray(simulate(jnp.asarray(omega0)))
    throw_figure.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines",
            line=dict(color="gray", width=2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=column,
    )
    for i in np.linspace(0, n_steps - 1, 13).astype(int):
        vertices = 2.0 * corners @ orientations[i].T + positions[i]  # enlarged
        throw_figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                facecolor=face_colors,
                flatshading=True,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
scene = dict(
    aspectmode="data",
    xaxis_title="x [m]",
    yaxis_title="y [m]",
    zaxis_title="z [m]",
    camera=dict(eye=dict(x=0.4, y=-2.8, z=0.7)),
)
throw_figure.update_layout(
    title="Throws (phone enlarged, dark side: screen)",
    scene=scene,
    scene2=scene,
    margin=dict(l=0, r=0, t=60, b=0),
    height=500,
)
throw_figure.show()
