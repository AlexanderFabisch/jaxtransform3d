r"""
=======================================
Pose Graph Optimization (SLAM Back-End)
=======================================

Pose graph optimization is the back-end of many SLAM (simultaneous
localization and mapping) systems. Poses of a robot are the nodes of a graph
and relative pose measurements (from odometry or loop closures) are its
edges. We search for the poses that agree best with all measurements. We
solve this nonlinear least squares problem on the Lie group :math:`SE(3)` with
the Levenberg-Marquardt algorithm.

We use the parking garage dataset, a standard benchmark for 3D pose graph
optimization. It was recorded with a robot that drove through a multi-storey
parking garage and has 1661 poses and 6275 edges [1]_.

.. [1] Carlone, L., Tron, R., Daniilidis, K., Dellaert, F. (2015).
   Initialization techniques for 3D SLAM: a survey on rotation estimation and
   its use in pose graph optimization. IEEE International Conference on
   Robotics and Automation (ICRA), pp. 4597-4604.
"""

import os
import time
import urllib.request

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxtransform3d.rotations as jr
import jaxtransform3d.transformations as jt

# %%
# Dataset
# -------
# The dataset is stored in the g2o format. Each vertex is a pose
# (position and quaternion) and each edge is a relative pose measurement
# :math:`\boldsymbol{Z}_{ij}` between the poses :math:`i` and :math:`j` with an
# information matrix :math:`\boldsymbol{\Omega}_{ij}` (inverse covariance).
# The poses of the vertices are the initial guess, which was obtained from
# odometry. We download the dataset once.
URL = "https://www.dropbox.com/s/zu23p8d522qccor/parking-garage.g2o?dl=1"
BASE_DIR = "data/"
data_dir = BASE_DIR
search_path = "."
while not os.path.exists(data_dir) and os.path.dirname(search_path) != "jaxtransform3d":
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "parking-garage.g2o")
if not os.path.exists(filename):
    urllib.request.urlretrieve(URL, filename)


def transforms_from_g2o(values):
    """Convert positions and quaternions (x, y, z, qx, qy, qz, qw)."""
    values = jnp.asarray(values, dtype=jnp.float32)
    q = jnp.concatenate((values[:, 6:], values[:, 3:6]), axis=-1)  # w, x, y, z
    R = jr.matrix_from_compact_axis_angle(jr.compact_axis_angle_from_quaternion(q))
    return jt.create_transform(R, values[:, :3])


def sqrt_information_from_g2o(values):
    """Square root of information matrix from its upper triangle.

    The rotational error of g2o is the vector part of a quaternion, which is
    half of the rotation vector, and g2o orders translation before rotation.
    We convert the information matrix to rotation vectors and the order
    (omega, v) of exponential coordinates.
    """
    information = np.zeros((6, 6))
    information[np.triu_indices(6)] = values
    information += np.triu(information, 1).T
    scale = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    reorder = np.roll(np.eye(6), 3, axis=0)
    information = reorder.T @ scale @ information @ scale @ reorder
    return np.linalg.cholesky(information).T


def read_g2o(filename):
    vertices, edges, measurements, sqrt_information = [], [], [], []
    with open(filename) as f:
        for line in f:
            values = line.split()
            if not values:
                continue
            if values[0] == "VERTEX_SE3:QUAT":
                vertices.append(np.asarray(values[2:9], dtype=float))
            elif values[0] == "EDGE_SE3:QUAT":
                edges.append((int(values[1]), int(values[2])))
                measurements.append(np.asarray(values[3:10], dtype=float))
                sqrt_information.append(
                    sqrt_information_from_g2o(np.asarray(values[10:31], dtype=float))
                )
    return (
        transforms_from_g2o(np.stack(vertices)),
        jnp.asarray(edges),
        transforms_from_g2o(np.stack(measurements)),
        jnp.asarray(np.stack(sqrt_information), dtype=jnp.float32),
    )


poses_init, edges, measurements, sqrt_information = read_g2o(filename)
n_poses = len(poses_init)
odometry = edges[:, 1] == edges[:, 0] + 1
print(
    f"{n_poses} poses, {int(odometry.sum())} odometry edges, "
    f"{int((~odometry).sum())} loop closures"
)

# %%
# Error of an Edge
# ----------------
# An edge measures the pose of :math:`j` relative to :math:`i`. The residual
# is the difference between the measured and the estimated relative pose in
# the tangent space, computed with the logarithmic map
#
# .. math::
#
#     \boldsymbol{r}_{ij} = \log\left(\boldsymbol{Z}_{ij}^{-1}
#     \boldsymbol{T}_i^{-1} \boldsymbol{T}_j\right) \in \mathbb{R}^6.
#
# We minimize :math:`E = \frac{1}{2} \sum_{(i, j)} \boldsymbol{r}_{ij}^T
# \boldsymbol{\Omega}_{ij} \boldsymbol{r}_{ij}`. Poses are elements of
# :math:`SE(3)`, so we cannot simply add an update to them. Instead, we
# parametrize a change of each pose :math:`\boldsymbol{T}_k` with a vector
# :math:`\boldsymbol{\delta}_k \in \mathbb{R}^6` in the tangent space,
# :math:`\boldsymbol{T}_k \leftarrow \boldsymbol{T}_k
# \exp(\boldsymbol{\delta}_k)`, and linearize around
# :math:`\boldsymbol{\delta} = \boldsymbol{0}`.


def edge_residual(delta_i, delta_j, T_i, T_j, Z_ij, sqrt_info):
    T_i = jt.compose_transforms(T_i, jt.transform_from_exponential_coordinates(delta_i))
    T_j = jt.compose_transforms(T_j, jt.transform_from_exponential_coordinates(delta_j))
    T_ij = jt.compose_transforms(jt.transform_inverse(T_i), T_j)
    error = jt.compose_transforms(jt.transform_inverse(Z_ij), T_ij)
    xi = jt.exponential_coordinates_from_transform(error)
    return jnp.matmul(sqrt_info, xi, precision="highest")


no_perturbation = jnp.zeros(6)


def residuals(poses):
    return jax.vmap(edge_residual, in_axes=(None, None, 0, 0, 0, 0))(
        no_perturbation,
        no_perturbation,
        poses[edges[:, 0]],
        poses[edges[:, 1]],
        measurements,
        sqrt_information,
    )


@jax.jit
def cost(poses):
    return 0.5 * jnp.sum(residuals(poses) ** 2)


# %%
# Levenberg-Marquardt
# -------------------
# The Levenberg-Marquardt step solves the damped normal equations
#
# .. math::
#
#     \left(\boldsymbol{H} + \lambda \operatorname{diag}(\boldsymbol{H})
#     \right) \boldsymbol{\delta} = -\boldsymbol{g},\quad
#     \boldsymbol{H} = \boldsymbol{J}^T \boldsymbol{J},\quad
#     \boldsymbol{g} = \boldsymbol{J}^T \boldsymbol{r}.
#
# The Jacobian :math:`\boldsymbol{J}` of all residuals would be a
# :math:`37650 \times 9966` matrix, but each residual only depends on two
# poses. Hence, we compute the :math:`6 \times 12` Jacobians of all edges
# with ``jax.jacfwd`` and ``jax.vmap`` and add their contributions to the
# blocks of :math:`\boldsymbol{H}` and :math:`\boldsymbol{g}`. The pose graph
# only contains relative measurements, so we could move all poses together
# without changing the error. We fix the first pose by removing its
# perturbation from the normal equations.
edge_jacobians = jax.vmap(
    jax.jacfwd(edge_residual, argnums=(0, 1)), in_axes=(None, None, 0, 0, 0, 0)
)


def transposed_products(A, B):
    """Compute A_e^T B_e for each edge e."""
    return jnp.einsum("eki,ekj->eij", A, B, precision="highest")


@jax.jit
def levenberg_marquardt_step(poses, damping):
    i, j = edges[:, 0], edges[:, 1]
    r = residuals(poses)
    J_i, J_j = edge_jacobians(
        no_perturbation,
        no_perturbation,
        poses[i],
        poses[j],
        measurements,
        sqrt_information,
    )

    # Each edge contributes to four 6x6 blocks of H and two blocks of g.
    H = jnp.zeros((n_poses, 6, n_poses, 6))
    H = H.at[i, :, i, :].add(transposed_products(J_i, J_i))
    H = H.at[j, :, j, :].add(transposed_products(J_j, J_j))
    H = H.at[i, :, j, :].add(transposed_products(J_i, J_j))
    H = H.at[j, :, i, :].add(transposed_products(J_j, J_i))
    g = jnp.zeros((n_poses, 6))
    g = g.at[i].add(transposed_products(J_i, r[..., jnp.newaxis])[..., 0])
    g = g.at[j].add(transposed_products(J_j, r[..., jnp.newaxis])[..., 0])
    H = H.reshape(6 * n_poses, 6 * n_poses)[6:, 6:]
    g = g.reshape(-1)[6:]

    H = H + damping * jnp.diag(jnp.diag(H))
    delta = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(H), -g)
    delta = jnp.vstack((jnp.zeros((1, 6)), delta.reshape(-1, 6)))
    poses = jt.compose_transforms(
        poses, jt.transform_from_exponential_coordinates(delta)
    )
    # Repeated products of float32 matrices slowly lose orthonormality.
    R = jax.vmap(jr.norm_matrix)(poses[:, :3, :3])
    return jt.create_transform(R, poses[:, :3, 3])


# %%
# We decrease the damping :math:`\lambda` after a successful step and increase
# it otherwise. We stop when the cost does not decrease significantly anymore.
# The first step takes longer, because JAX compiles the function.
poses = poses_init
damping = 1e-4
costs = [float(cost(poses))]
print(f"initial cost: {costs[0]:.3f}")
for iteration in range(1, 21):
    start = time.perf_counter()
    poses_candidate = levenberg_marquardt_step(poses, damping)
    cost_candidate = float(cost(poses_candidate))
    duration = time.perf_counter() - start
    accepted = cost_candidate < costs[-1]
    if accepted:
        converged = costs[-1] - cost_candidate < 1e-3 * costs[-1]
        poses = poses_candidate
        costs.append(cost_candidate)
        damping /= 10.0
    else:
        converged = False
        damping *= 10.0
    print(
        f"iteration {iteration:2d}: cost {cost_candidate:9.3f} "
        f"({'accepted' if accepted else 'rejected'}) in {1000 * duration:.0f} ms"
    )
    if converged:
        break

# %%
# Results
# -------
# The garage has four parking decks. In the initial guess, the trajectory
# drifts, so the decks appear as wide bands in the side view. After the
# optimization, the decks are planar and parallel. The 3D plots only show the
# section of the trajectory in the garage with exaggerated height.
P_init = np.asarray(poses_init[:, :3, 3])
P_opt = np.asarray(poses[:, :3, 3])

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(
    2, 3, height_ratios=(1.5, 1.0), width_ratios=(1.0, 1.0, 1.0), hspace=0.25
)
garage = P_opt[:, 1] > 120.0  # the section with the parking decks
for k, (P, title, color) in enumerate(
    [(P_init, "Initial guess", "tab:red"), (P_opt, "Optimized", "tab:blue")]
):
    ax = fig.add_subplot(gs[0, k], projection="3d")
    P_garage = np.where(garage[:, np.newaxis], P, np.nan)
    ax.plot(*P_garage.T, lw=0.7, c=color)
    ax.set_box_aspect((1.0, 1.3, 0.8), zoom=1.15)  # height exaggerated
    ax.set_zlim(-7.0, 8.0)
    ax.set_zticks([-5, 0, 5])
    ax.view_init(elev=12, azim=-110)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)

ax = fig.add_subplot(gs[0, 2])
ax.semilogy(costs, marker="o", c="k")
ax.set_xticks(range(len(costs)))
ax.set_xlabel("Accepted iteration")
ax.set_ylabel("Cost $E$")
ax.set_title("Convergence")
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(gs[1, :])
ax.plot(P_init[:, 1], P_init[:, 2], lw=0.7, c="tab:red", label="Initial guess")
ax.plot(P_opt[:, 1], P_opt[:, 2], lw=0.7, c="tab:blue", label="Optimized")
ax.set_xlim(120.0, 260.0)
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
ax.set_title("Side view of the parking decks")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.suptitle("Parking garage: pose graph optimization", fontsize=14)
fig.subplots_adjust(left=0.06, right=0.97, top=0.92, bottom=0.06)
plt.show()
