"""
========================
Plot Polar Decomposition
========================

Robust polar decomposition orthonormalizes basis vectors (i.e., rotation
matrices). It is more expensive than standard Gram-Schmidt orthonormalization,
but it spreads the error more evenly over all basis vectors. The top row of
these plots shows the unnormalized bases that were obtained by randomly
rotating one of the columns of the identity matrix. The middle row shows
Gram-Schmidt orthonormalization and the bottom row shows orthonormalization
through robust polar decomposition. For comparison, we show the unnormalized
basis with dashed lines in the last two rows.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr

import jaxtransform3d.rotations as jr

n_cases = 8
fig, axes = plt.subplots(3, n_cases, subplot_kw={"projection": "3d"}, figsize=(14, 8))
ax_s = 1.0
plot_center = np.array([-0.2, -0.2, -0.2])
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-ax_s, ax_s)
    ax.set_ylim(-ax_s, ax_s)
    ax.set_zlim(-ax_s, ax_s)

titles = ["Unnormalized Bases", "Gram-Schmidt", "Polar Decomposition"]
for ax, title in zip(axes[:, 0], titles):
    ax.set_title(title)

rng = np.random.default_rng(46)
R_unnormalized = np.array([np.eye(3) for _ in range(n_cases)])
for i in range(n_cases):
    random_axis = rng.integers(0, 3)
    R_unnormalized[i, :, random_axis] = np.dot(
        pr.random_matrix(rng, cov=0.1 * np.eye(3)),
        R_unnormalized[i, :, random_axis],
    )
robust_polar_decomposition = jax.jit(jr.robust_polar_decomposition)
for i in range(n_cases):
    pr.plot_basis(axes[0, i], R_unnormalized[i], p=plot_center, strict_check=False)

    R_gs = pr.norm_matrix(R_unnormalized[i])
    pr.plot_basis(
        axes[1, i], R_unnormalized[i], p=plot_center, strict_check=False, ls="--"
    )
    pr.plot_basis(axes[1, i], R_gs, p=plot_center)

    R_pd = robust_polar_decomposition(R_unnormalized[i])
    pr.plot_basis(
        axes[2, i], R_unnormalized[i], p=plot_center, strict_check=False, ls="--"
    )
    pr.plot_basis(axes[2, i], R_pd, p=plot_center)

plt.tight_layout()
plt.show()
