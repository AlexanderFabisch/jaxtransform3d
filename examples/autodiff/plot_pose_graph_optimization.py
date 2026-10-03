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


# The rotational error of g2o is the vector part of a quaternion, which is
# half of the rotation vector, and g2o orders translation before rotation. We
# convert the information matrices to rotation vectors and the order
# (omega, v) of exponential coordinates.
scale = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
reorder = np.block([[np.zeros((3, 3)), np.eye(3)], [np.eye(3), np.zeros((3, 3))]])

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
            information = np.zeros((6, 6))
            information[np.triu_indices(6)] = np.asarray(values[10:31], dtype=float)
            information += np.triu(information, 1).T
            information = reorder.T @ scale @ information @ scale @ reorder
            sqrt_information.append(np.linalg.cholesky(information).T)
poses_init = transforms_from_g2o(np.stack(vertices))
edges = jnp.asarray(edges)
measurements = transforms_from_g2o(np.stack(measurements))
sqrt_information = jnp.asarray(np.stack(sqrt_information), dtype=jnp.float32)
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


zero = jnp.zeros(6)


def residuals(poses):
    return jax.vmap(edge_residual, in_axes=(None, None, 0, 0, 0, 0))(
        zero,
        zero,
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


def JtJ(A, B):
    return jnp.einsum("eki,ekj->eij", A, B, precision="highest")


@jax.jit
def levenberg_marquardt_step(poses, damping):
    i, j = edges[:, 0], edges[:, 1]
    r = residuals(poses)
    J_i, J_j = edge_jacobians(
        zero, zero, poses[i], poses[j], measurements, sqrt_information
    )

    H = jnp.zeros((n_poses, 6, n_poses, 6))
    H = H.at[i, :, i, :].add(JtJ(J_i, J_i)).at[j, :, j, :].add(JtJ(J_j, J_j))
    H = H.at[i, :, j, :].add(JtJ(J_i, J_j)).at[j, :, i, :].add(JtJ(J_j, J_i))
    g = jnp.zeros((n_poses, 6))
    g = g.at[i].add(jnp.einsum("eki,ek->ei", J_i, r, precision="highest"))
    g = g.at[j].add(jnp.einsum("eki,ek->ei", J_j, r, precision="highest"))
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
# it otherwise.
poses = poses_init
damping = 1e-4
costs = [float(cost(poses))]
print(f"initial cost: {costs[0]:.3f}")
start = time.perf_counter()
for it in range(1, 11):
    poses_candidate = levenberg_marquardt_step(poses, damping)
    cost_candidate = float(cost(poses_candidate))
    if cost_candidate < costs[-1]:
        poses = poses_candidate
        costs.append(cost_candidate)
        damping /= 10.0
    else:
        damping *= 10.0
    print(f"iteration {it:2d}: cost {costs[-1]:.3f}")
print(f"optimization took {time.perf_counter() - start:.1f} s (incl. compilation)")

# %%
# Results
# -------
# The garage has four parking decks. In the initial guess, the trajectory
# drifts, so the decks appear as wide bands in the side view. After the
# optimization, the decks are planar and parallel. The 3D plots only show the
# section of the trajectory in the garage.
P_init = np.asarray(poses_init[:, :3, 3])
P_opt = np.asarray(poses[:, :3, 3])

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=(1.3, 1.0), wspace=0.3, hspace=0.35)
garage = P_opt[:, 1] > 120.0  # the section with the parking decks
for k, (P, title) in enumerate([(P_init, "Initial guess"), (P_opt, "Optimized")]):
    ax = fig.add_subplot(gs[0, k], projection="3d")
    P_garage = np.where(garage[:, np.newaxis], P, np.nan)
    ax.plot(*P_garage.T, lw=0.6, c="tab:red" if k == 0 else "tab:blue")
    ax.set_box_aspect((1.0, 1.3, 0.6))  # height exaggerated
    ax.set_zlim(-7.0, 8.0)
    ax.set_zticks([-5, 0, 5])
    ax.view_init(elev=12, azim=-110)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{title} (parking decks)")

ax = fig.add_subplot(gs[0, 2])
ax.semilogy(costs, marker="o", c="k")
ax.set_xlabel("Accepted iteration")
ax.set_ylabel("Cost $E$")
ax.set_title("Convergence of Levenberg-Marquardt")
ax.grid(alpha=0.3, which="both")

ax = fig.add_subplot(gs[1, :])
ax.plot(P_init[:, 1], P_init[:, 2], lw=0.6, c="tab:red", label="Initial guess")
ax.plot(P_opt[:, 1], P_opt[:, 2], lw=0.6, c="tab:blue", label="Optimized")
ax.set_xlim(120.0, 260.0)
ax.set_xlabel("y [m]")
ax.set_ylabel("z [m]")
ax.set_title("Side view of the parking decks")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.show()
