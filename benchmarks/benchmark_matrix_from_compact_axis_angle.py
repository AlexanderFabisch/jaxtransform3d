"""
Because matrix_from_compact_axis_angle uses multiple jnp.stack calls, we
compare the version of the library to a vmapped non-vectorized version.
The difference in speed is negligible. Both versions are insanely fast...
"""

import timeit
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import jaxtransform3d.rotations as jr
from jaxtransform3d.utils import norm_vector


def matrix_from_compact_axis_angle(
    axis_angle: ArrayLike | None = None,
    axis: ArrayLike | None = None,
    angle: ArrayLike | None = None,
) -> jax.Array:
    axis_angle = jnp.asarray(axis_angle)
    if not jnp.issubdtype(axis_angle.dtype, jnp.floating):
        axis_angle = axis_angle.astype(jnp.float64)

    if angle is None:
        angle = jnp.linalg.norm(axis_angle)
    else:
        angle = jnp.asarray(angle)

    if axis is None:
        axis = norm_vector(axis_angle, angle)
    else:
        axis = jnp.asarray(axis)

    chex.assert_axis_dimension(axis_angle, axis=0, expected=3)
    chex.assert_equal_shape((axis_angle, axis))

    c = jnp.cos(angle)
    s = jnp.sin(angle)
    ci = 1.0 - c
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    uxs = ux * s
    uys = uy * s
    uzs = uz * s
    ciux = ci * ux
    ciuy = ci * uy
    ciuxuy = ciux * uy
    ciuxuz = ciux * uz
    ciuyuz = ciuy * uz

    return jnp.array(
        [
            [ciux * ux + c, ciuxuy - uzs, ciuxuz + uys],
            [ciuxuy + uzs, ciuy * uy + c, ciuyuz - uxs],
            [ciuxuz - uys, ciuyuz + uxs, ci * uz * uz + c],
        ]
    )


key = jax.random.PRNGKey(42)
a = jax.random.normal(key, (1_000_000, 3))

matrix_from_compact_axis_angle1 = jax.jit(jr.matrix_from_compact_axis_angle)
matrix_from_compact_axis_angle2 = jax.jit(jax.vmap(matrix_from_compact_axis_angle))

# JIT compile
matrix_from_compact_axis_angle1(a)
matrix_from_compact_axis_angle2(a)

times = timeit.repeat(
    partial(matrix_from_compact_axis_angle1, axis_angle=a), repeat=10, number=100
)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
# Mean: 0.07380; Std. dev.: 0.08942

times = timeit.repeat(
    partial(matrix_from_compact_axis_angle2, axis_angle=a), repeat=10, number=100
)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
# Mean: 0.07288; Std. dev.: 0.09512
