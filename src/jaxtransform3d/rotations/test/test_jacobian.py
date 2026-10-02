import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytransform3d.rotations as pr
from jax.experimental import enable_x64
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.linalg import expm

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


@pytest.mark.parametrize(
    "angle", [1e-6, 9.99e-4, 1.001e-3, 2e-3, 5e-3, 1e-2, 0.1, 1.0, np.pi]
)
@pytest.mark.parametrize(
    "dtype, rtol, atol", [("float32", 1e-6, 2e-7), ("float64", 1e-14, 2e-15)]
)
def test_left_jacobian_SO3_against_matrix_exponential(angle, dtype, rtol, atol):
    # exp([[[omega], I], [0, 0]]) contains the integral of exp(t * [omega])
    # from 0 to 1 in its upper right block, which is the left Jacobian.
    # This is independent of the closed form and the Taylor series.
    omega = angle * np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    generator = np.zeros((6, 6))
    generator[:3, :3] = pr.cross_product_matrix(omega)
    generator[:3, 3:] = np.eye(3)
    expected = expm(generator)[:3, 3:]

    with enable_x64(dtype == "float64"):
        omega = jnp.asarray(omega, dtype=dtype)
        J = left_jacobian_SO3(omega)
        J_inv = left_jacobian_SO3_inv(omega)

    assert_allclose(J, expected, rtol=rtol, atol=atol)
    assert_allclose(J_inv, np.linalg.inv(expected), rtol=rtol, atol=atol)
