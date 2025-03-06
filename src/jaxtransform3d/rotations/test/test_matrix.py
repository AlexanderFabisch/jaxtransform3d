import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.batch_rotations as pbr
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.rotations as jr

matrix_from_compact_axis_angle = jax.jit(jr.matrix_from_compact_axis_angle)
compact_axis_angle_from_matrix = jax.jit(jr.compact_axis_angle_from_matrix)


def test_compact_axis_angle_from_matrix_0dim():
    R = jnp.eye(3)
    a = compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, jnp.zeros(3))

    R = pr.active_matrix_from_intrinsic_euler_xyz(jnp.array([-jnp.pi, -jnp.pi, 0.0]))
    a = compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, jnp.array([0, 0, jnp.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(jnp.array([-jnp.pi, 0.0, -jnp.pi]))
    a = compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, jnp.array([0, jnp.pi, 0]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(jnp.array([0.0, -jnp.pi, -jnp.pi]))
    a = compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, jnp.array([jnp.pi, 0, 0]))

    a = jnp.pi * jnp.array([jnp.sqrt(0.5), jnp.sqrt(0.5), 0.0])
    R = pr.matrix_from_compact_axis_angle(a)
    a1 = pr.compact_axis_angle_from_matrix(R)
    a2 = compact_axis_angle_from_matrix(R)
    assert_array_almost_equal(a1, a2)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.normal(size=3)
        R = matrix_from_compact_axis_angle(a)
        pr.assert_rotation_matrix(R)

        a2 = compact_axis_angle_from_matrix(R)
        pr.assert_compact_axis_angle_equal(a, a2)

        R2 = matrix_from_compact_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_compact_axis_angle_from_matrix_1dim():
    rng = np.random.default_rng(84)
    a = rng.standard_normal(size=(20, 3))
    a /= np.linalg.norm(a, axis=-1)[..., np.newaxis]
    a *= rng.random(size=(20, 1)) * np.pi
    a[0, :] = 0.0

    R = pbr.matrices_from_compact_axis_angles(a)
    a1 = pbr.axis_angles_from_matrices(R)
    a1_compact = a1[..., :3] * a1[..., 3, np.newaxis]
    a2 = compact_axis_angle_from_matrix(R)
    assert_array_almost_equal(a1_compact, a2)


def test_compact_axis_angle_from_matrix_2dims():
    rng = np.random.default_rng(84)
    a = rng.standard_normal(size=(2, 5, 3))
    a /= np.linalg.norm(a, axis=-1)[..., np.newaxis]
    a *= rng.random(size=(2, 5, 1)) * np.pi
    a[0, 0, :] = 0.0

    R = pbr.matrices_from_compact_axis_angles(a)
    a1 = pbr.axis_angles_from_matrices(R)
    a1_compact = a1[..., :3] * a1[..., 3, np.newaxis]
    a2 = compact_axis_angle_from_matrix(R)
    assert_array_almost_equal(a1_compact, a2, decimal=5)
