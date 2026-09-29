import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.batch_rotations as pbr
import pytransform3d.rotations as pr
from jax.experimental import enable_x64
from numpy.testing import assert_allclose, assert_array_almost_equal

import jaxtransform3d.rotations as jr

matrix_from_compact_axis_angle = jax.jit(jr.matrix_from_compact_axis_angle)


def test_matrix_from_compact_axis_angle_0dim():
    a = jnp.zeros(3)
    assert_array_almost_equal(
        matrix_from_compact_axis_angle(a), pr.matrix_from_compact_axis_angle(a)
    )

    a = jnp.array([jnp.sqrt(0.5) * jnp.pi, jnp.sqrt(0.5) * jnp.pi, 0.0])
    assert_array_almost_equal(
        matrix_from_compact_axis_angle(a), pr.matrix_from_compact_axis_angle(a)
    )

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = jnp.asarray(pr.random_compact_axis_angle(rng))
        assert_array_almost_equal(
            matrix_from_compact_axis_angle(a), pr.matrix_from_compact_axis_angle(a)
        )


def test_matrix_from_compact_axis_angle_1dim():
    rng = np.random.default_rng(1)
    a = jnp.asarray(rng.normal(size=(50, 3)))
    assert_array_almost_equal(
        matrix_from_compact_axis_angle(a), pbr.matrices_from_compact_axis_angles(a)
    )


def test_matrix_from_compact_axis_angle_2dims():
    rng = np.random.default_rng(2)
    a = jnp.asarray(rng.normal(size=(5, 10, 3)))
    assert_array_almost_equal(
        matrix_from_compact_axis_angle(a), pbr.matrices_from_compact_axis_angles(a)
    )


def test_quaternion_from_compact_axis_angle_taylor_branch():
    """The small-angle Taylor branch matches the closed form.

    The error of a wrong series is far below float32 precision, so this
    test runs in float64.
    """
    with enable_x64():
        axis = jnp.array([1.0, 2.0, -2.0]) / 3.0
        for angle in [1e-3 * (1.0 - 1e-6), 1e-4]:
            q = jr.quaternion_from_compact_axis_angle(axis * angle)
            assert_allclose(q[1:], np.sin(0.5 * angle) * axis, rtol=1e-14)
