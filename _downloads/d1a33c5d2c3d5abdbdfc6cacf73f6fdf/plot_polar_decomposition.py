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

import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.rotations as pr

import jaxtransform3d.rotations as jr

gram_schmidt = jax.jit(jax.vmap(jr.norm_matrix, in_axes=0, out_axes=0))
robust_polar_decomposition = jax.jit(
    jax.vmap(partial(jr.robust_polar_decomposition, n_iter=5), in_axes=0, out_axes=0)
)

start = time.time()
gram_schmidt(jnp.eye(3)[jnp.newaxis]).block_until_ready()
gs_jit_time = time.time() - start
start = time.time()
robust_polar_decomposition(jnp.eye(3)[jnp.newaxis]).block_until_ready()
rpd_jit_time = time.time() - start

n_cases = 8
fig, axes = plt.subplots(3, n_cases, subplot_kw={"projection": "3d"}, figsize=(14, 8))
ax_s = 1.0
plot_center = jnp.array([-0.2, -0.2, -0.2])
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-ax_s, ax_s)
    ax.set_ylim(-ax_s, ax_s)
    ax.set_zlim(-ax_s, ax_s)

titles = ["Unnormalized Bases", "Gram-Schmidt", "Polar Decomposition"]
for ax, title in zip(axes[:, 0], titles, strict=False):
    ax.set_title(title)

rng = np.random.default_rng(46)
R_unnormalized = jnp.array([jnp.eye(3) for _ in range(n_cases)])
for i in range(n_cases):
    random_axis = rng.integers(0, 3)
    R_unnormalized = R_unnormalized.at[i, :, random_axis].set(
        jnp.dot(
            pr.random_matrix(rng, cov=0.1 * jnp.eye(3)),
            R_unnormalized[i, :, random_axis],
        )
    )

start = time.time()
R_gs = gram_schmidt(R_unnormalized)
gs_time = time.time() - start

start = time.time()
R_rpd = robust_polar_decomposition(R_unnormalized)
rpd_time = time.time() - start

print(f"JIT-compiled Gram-Schmidt orthogonalization: {gs_jit_time:.5f} s")
print(f"JIT-compiled robust polar decomposition: {rpd_jit_time:.5f} s")
print(f"Gram-Schmidt orthogonalization: {gs_time:.5f} s")
print(f"Robost polar decomposition: {rpd_time:.5f} s")

for i in range(n_cases):
    pr.plot_basis(axes[0, i], R_unnormalized[i], p=plot_center, strict_check=False)

    pr.plot_basis(
        axes[1, i], R_unnormalized[i], p=plot_center, strict_check=False, ls="--"
    )
    pr.plot_basis(axes[1, i], R_gs[i], p=plot_center)

    pr.plot_basis(
        axes[2, i], R_unnormalized[i], p=plot_center, strict_check=False, ls="--"
    )
    pr.plot_basis(axes[2, i], R_rpd[i], p=plot_center)

plt.tight_layout()
plt.show()
