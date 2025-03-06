import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.batch_rotations as pbr
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.rotations as jr

compose_quaternions = jax.jit(jr.compose_quaternions)
quaternion_conjugate = jax.jit(jr.quaternion_conjugate)
quaternion_from_compact_axis_angle = jax.jit(jr.quaternion_from_compact_axis_angle)
compact_axis_angle_from_quaternion = jax.jit(jr.compact_axis_angle_from_quaternion)


def test_batch_concatenate_quaternions_0dim():
    rng = np.random.default_rng(230)
    for _ in range(5):
        q1 = pr.random_quaternion(rng)
        q2 = pr.random_quaternion(rng)
        q12 = pbr.batch_concatenate_quaternions(q1, q2)
        assert_array_almost_equal(q12, compose_quaternions(q1, q2))


def test_batch_concatenate_q_conj():
    rng = np.random.default_rng(231)
    Q = np.array([pr.random_quaternion(rng) for _ in range(10)]).reshape(2, 5, 4)

    Q_conj = quaternion_conjugate(Q)
    Q_Q_conj = compose_quaternions(Q, Q_conj)

    assert_array_almost_equal(Q_Q_conj.reshape(-1, 4), np.array([[1, 0, 0, 0]] * 10))


def test_quaternion_from_compact_axis_angle_0dim():
    q = jnp.array([1, 0, 0, 0])
    a = compact_axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, jnp.zeros(3))
    q2 = quaternion_from_compact_axis_angle(a)
    assert_array_almost_equal(q2, q)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=3)
        q = quaternion_from_compact_axis_angle(a)

        a2 = compact_axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_compact_axis_angle(a2)
        pr.assert_quaternion_equal(q, q2)


def test_compact_axis_angle_from_quaternion_ndims():
    rng = np.random.default_rng(48322)
    n_rotations = 20
    q = pbr.norm_vectors(rng.standard_normal(size=(n_rotations, 4)))

    # 1D
    a = pbr.axis_angles_from_quaternions(q[0])
    pr.assert_quaternion_equal(a[:3] * a[3], compact_axis_angle_from_quaternion(q[0]))

    # 2D
    a = pbr.axis_angles_from_quaternions(q)
    a = a[:, :3] * a[:, 3, np.newaxis]
    assert_array_almost_equal(a, compact_axis_angle_from_quaternion(q))

    # 3D
    q_3d = q.reshape(n_rotations // 4, 4, 4)
    a_3d = pbr.axis_angles_from_quaternions(q_3d)
    a_3d = a_3d[..., :3] * a_3d[..., 3, np.newaxis]
    assert_array_almost_equal(a_3d, compact_axis_angle_from_quaternion(q_3d))
