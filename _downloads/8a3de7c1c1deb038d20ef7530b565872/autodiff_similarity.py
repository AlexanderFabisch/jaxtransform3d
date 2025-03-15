"""
=====================
Find Similar Rotation
=====================

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.rotations as pr

import jaxtransform3d.rotations as jr

rng = np.random.default_rng(20)
q_true = jnp.asarray(pr.random_quaternion(rng))


def error(a):
    q = jr.quaternion_from_compact_axis_angle(a)
    q_diff = jr.compose_quaternions(jr.quaternion_conjugate(q), q_true)
    return jnp.linalg.norm(jr.compact_axis_angle_from_quaternion(q_diff))


error_with_grad = jax.jit(jax.value_and_grad(error))

a = jnp.asarray(pr.random_compact_axis_angle(rng))

for it in range(1000):
    e, grad = error_with_grad(a)
    print(f"{it=}, {e=:.3f}, {jnp.linalg.norm(grad)=:.3f}")
    if e == 0:
        break
    a = a - 0.01 * grad
