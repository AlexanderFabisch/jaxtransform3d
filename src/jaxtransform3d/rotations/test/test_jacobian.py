import jax
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.rotations as jr

left_jacobian_SO3 = jax.jit(jr.left_jacobian_SO3)
left_jacobian_SO3_series = jax.jit(jr.left_jacobian_SO3_series)
left_jacobian_SO3_inv = jax.jit(jr.left_jacobian_SO3_inv)
left_jacobian_SO3_inv_series = jax.jit(jr.left_jacobian_SO3_inv_series)


def test_left_jacobian_SO3():
    key = jax.random.key(41)
    axis_angle = jax.random.normal(key, shape=(10, 3))

    jac = left_jacobian_SO3(axis_angle)
    for a, j in zip(axis_angle, jac, strict=False):
        jac_gt = pr.left_jacobian_SO3(a)
        assert_array_almost_equal(j, jac_gt)


def test_left_jacobian_SO3_series():
    key = jax.random.key(42)
    axis_angle = jax.random.normal(key, shape=(10, 3))

    jac_series = left_jacobian_SO3_series(axis_angle)
    for a, j in zip(axis_angle, jac_series, strict=False):
        jac_gt = pr.left_jacobian_SO3_series(a, 10)
        assert_array_almost_equal(j, jac_gt)


def test_left_jacobian_SO3_inv():
    key = jax.random.key(42)
    axis_angle = jax.random.normal(key, shape=(10, 3))

    jac_inv = left_jacobian_SO3_inv(axis_angle)
    for a, j in zip(axis_angle, jac_inv, strict=False):
        jac_inv_gt = pr.left_jacobian_SO3_inv(a)
        assert_array_almost_equal(j, jac_inv_gt)


def test_left_jacobian_SO3_inv_series():
    key = jax.random.key(42)
    axis_angle = jax.random.normal(key, shape=(10, 3))

    jac_inv_series = left_jacobian_SO3_inv_series(axis_angle)
    for a, j in zip(axis_angle, jac_inv_series, strict=False):
        jac_inv_gt = pr.left_jacobian_SO3_inv_series(a, 10)
        assert_array_almost_equal(j, jac_inv_gt)
