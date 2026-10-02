import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytransform3d.batch_rotations as pbr
import pytransform3d.rotations as pr
from jax.experimental import enable_x64
from numpy.testing import assert_allclose, assert_array_almost_equal

import jaxtransform3d.rotations as jr

compose_quaternions = jax.jit(jr.compose_quaternions)
quaternion_conjugate = jax.jit(jr.quaternion_conjugate)
apply_quaternion = jax.jit(jr.apply_quaternion)
quaternion_from_compact_axis_angle = jax.jit(jr.quaternion_from_compact_axis_angle)
compact_axis_angle_from_quaternion = jax.jit(jr.compact_axis_angle_from_quaternion)


def test_norm_quaternion():
    rng = np.random.default_rng(0)
    q = rng.normal(size=4)
    q_norm = jr.norm_quaternion(q)
    assert pytest.approx(np.linalg.norm(q_norm)) == 1.0

    q = rng.normal(size=(20, 4))
    q_norm = jr.norm_quaternion(q)
    assert_array_almost_equal(jnp.linalg.norm(q_norm, axis=-1), jnp.ones(20))

    q = rng.normal(size=(5, 4, 4))
    q_norm = jr.norm_quaternion(q)
    assert_array_almost_equal(jnp.linalg.norm(q_norm, axis=-1), jnp.ones((5, 4)))


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


def test_apply_quaternion_0dim():
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=3)
        v = rng.normal(size=3)
        q = quaternion_from_compact_axis_angle(a)
        R = jr.matrix_from_compact_axis_angle(a)
        vR = jr.apply_matrix(R, v)
        vq = jr.apply_quaternion(q, v)
        assert_array_almost_equal(vR, vq)


def test_apply_quaternion_1dim():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(10, 3))
    v = rng.normal(size=(10, 3))
    q = quaternion_from_compact_axis_angle(a)
    R = jr.matrix_from_compact_axis_angle(a)
    vR = jr.apply_matrix(R, v)
    vq = jr.apply_quaternion(q, v)
    assert_array_almost_equal(vR, vq)


def test_apply_quaternion_2dims():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(2, 5, 3))
    v = rng.normal(size=(2, 5, 3))
    q = quaternion_from_compact_axis_angle(a)
    R = jr.matrix_from_compact_axis_angle(a)
    vR = jr.apply_matrix(R, v)
    vq = jr.apply_quaternion(q, v)
    assert_array_almost_equal(vR, vq)


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


def test_compact_axis_angle_from_quaternion_small_angle():
    with enable_x64():
        axis = jnp.array([1.0, 2.0, -2.0]) / 3.0
        for angle in [1e-4, 1e-6, 1e-8, 1e-10, 1e-16, 1e-100]:
            a = axis * angle
            q = jr.quaternion_from_compact_axis_angle(a)
            a2 = jr.compact_axis_angle_from_quaternion(q)
            assert_allclose(a2, a, rtol=1e-12)


def test_compact_axis_angle_from_quaternion_negative_real():
    rng = np.random.default_rng(87)
    q = pbr.norm_vectors(rng.standard_normal(size=(20, 4)))
    q[:, 0] = -np.abs(q[:, 0])
    a = compact_axis_angle_from_quaternion(q)
    assert (np.linalg.norm(a, axis=-1) <= np.pi + 1e-6).all()
    q2 = quaternion_from_compact_axis_angle(a)
    for i in range(len(q)):
        pr.assert_quaternion_equal(q[i], q2[i], decimal=5)


@pytest.mark.parametrize("scale", [0.0, 1e-40, 1e-30, 1e-22, 1e-12])
def test_compact_axis_angle_from_quaternion_gradient_at_identity(scale):
    # The derivative of the log map at the identity is 2 * I with respect to
    # the vector part and 0 with respect to the real part.
    q = jnp.array([1.0, 4.0 * scale, -2.0 * scale, scale], dtype=jnp.float32)
    jac_fwd = jax.jacfwd(jr.compact_axis_angle_from_quaternion)(q)
    jac_rev = jax.jacrev(jr.compact_axis_angle_from_quaternion)(q)
    assert_allclose(jac_fwd[:, 1:], 2.0 * np.eye(3), rtol=1e-6, atol=1e-6)
    assert_allclose(jac_fwd[:, 0], np.zeros(3), atol=1e-6)
    assert_allclose(jac_rev, jac_fwd, rtol=1e-6)


@pytest.mark.parametrize(
    "angle", [1e-6, 2.4e-4, 2.5e-4, 1e-3, 0.5, 2.0, 3.0, np.pi - 1e-4]
)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_compact_axis_angle_from_quaternion_jacobian(angle, sign):
    # angles around 2.44e-4 are on both sides of the series threshold
    axis = np.array([1.0, 2.0, -2.0]) / 3.0
    q = sign * np.r_[np.cos(0.5 * angle), np.sin(0.5 * angle) * axis]
    with enable_x64():
        jac = jax.jacfwd(jr.compact_axis_angle_from_quaternion)(jnp.asarray(q))
        h = 1e-7
        jac_num = np.column_stack(
            [
                (
                    jr.compact_axis_angle_from_quaternion(q + h * e)
                    - jr.compact_axis_angle_from_quaternion(q - h * e)
                )
                / (2.0 * h)
                for e in np.eye(4)
            ]
        )
    assert_allclose(jac, jac_num, rtol=1e-6, atol=1e-6)
