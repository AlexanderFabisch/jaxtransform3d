"""Numerical accuracy and differentiability of exponential and logarithmic maps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytransform3d.rotations as pr
from jax.experimental import enable_x64
from numpy.testing import assert_allclose

import jaxtransform3d.rotations as jr
import jaxtransform3d.transformations as jt
import jaxtransform3d.utils as ju

AXIS = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
TRANSLATION = np.array([0.3, -1.2, 0.5])
# critical angles: 0, tiny, around the series cutoff, close to pi
ANGLES = [0.0, 1e-30, 1e-12, 1e-6, 1e-3, 0.49, 0.51, 1.0, 3.0, np.pi - 1e-5]


def _input(angle, n):
    return angle * AXIS if n == 3 else np.r_[angle * AXIS, TRANSLATION]


EXP_MAPS = {
    "matrix_from_compact_axis_angle": (jr.matrix_from_compact_axis_angle, 3),
    "quaternion_from_compact_axis_angle": (jr.quaternion_from_compact_axis_angle, 3),
    "left_jacobian_SO3": (jr.left_jacobian_SO3, 3),
    "left_jacobian_SO3_inv": (jr.left_jacobian_SO3_inv, 3),
    "transform_from_exponential_coordinates": (
        jt.transform_from_exponential_coordinates,
        6,
    ),
    "dual_quaternion_from_exponential_coordinates": (
        jt.dual_quaternion_from_exponential_coordinates,
        6,
    ),
}

ROUND_TRIPS = {
    "matrix": (
        lambda a: jr.compact_axis_angle_from_matrix(
            jr.matrix_from_compact_axis_angle(a)
        ),
        3,
    ),
    "quaternion": (
        lambda a: jr.compact_axis_angle_from_quaternion(
            jr.quaternion_from_compact_axis_angle(a)
        ),
        3,
    ),
    "transform": (
        lambda x: jt.exponential_coordinates_from_transform(
            jt.transform_from_exponential_coordinates(x)
        ),
        6,
    ),
    "dual_quaternion": (
        lambda x: jt.exponential_coordinates_from_dual_quaternion(
            jt.dual_quaternion_from_exponential_coordinates(x)
        ),
        6,
    ),
}


@pytest.mark.parametrize("name", EXP_MAPS)
@pytest.mark.parametrize("angle", ANGLES)
def test_exp_maps_float32_jacobians_match_float64(name, angle):
    """Values and Jacobians in float32 are finite and close to float64."""
    f, n = EXP_MAPS[name]
    x = _input(angle, n)
    with enable_x64():
        x64 = jnp.asarray(x, dtype=jnp.float64)
        value64 = f(x64)
        jac64 = jax.jacfwd(f)(x64)
    x32 = jnp.asarray(x, dtype=jnp.float32)
    for jac in (jax.jacfwd, jax.jacrev):
        jac32 = jac(f)(x32)
        assert np.isfinite(np.asarray(jac32)).all()
        assert_allclose(jac32, jac64, rtol=0, atol=2e-6)
    assert_allclose(f(x32), value64, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", ROUND_TRIPS)
@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("dtype, atol", [("float32", 2e-6), ("float64", 1e-12)])
def test_log_of_exp_is_identity(name, angle, dtype, atol):
    """Log(Exp(x)) = x, so its Jacobian is the identity for angles < pi."""
    f, n = ROUND_TRIPS[name]
    with enable_x64(dtype == "float64"):
        x = jnp.asarray(_input(angle, n), dtype=dtype)
        assert_allclose(f(x), x, rtol=0, atol=atol)
        for jac in (jax.jacfwd, jax.jacrev):
            assert_allclose(jac(f)(x), np.eye(n), rtol=0, atol=atol)


def test_exp_maps_translation_gradient_at_zero_rotation():
    """The translation depends on the rotation also for tiny angles.

    t = J(omega) v, so dt / domega = -[v] / 2 at omega = 0.
    """
    expected = -0.5 * pr.cross_product_matrix(TRANSLATION)

    def t_from_transform(x):
        return jt.transform_from_exponential_coordinates(x)[:3, 3]

    def t_from_dual_quaternion(x):
        dq = jt.dual_quaternion_from_exponential_coordinates(x)
        return 2.0 * jr.compose_quaternions(dq[4:], jr.quaternion_conjugate(dq[:4]))[1:]

    for angle in [0.0, 1e-30, 1e-8]:
        x = jnp.asarray(_input(angle, 6), dtype=jnp.float32)
        for f in (t_from_transform, t_from_dual_quaternion):
            for jac in (jax.jacfwd, jax.jacrev):
                assert_allclose(jac(f)(x)[:, :3], expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "angle", [0.0, 1e-8, 0.5, 2.0, np.pi - 1e-6, np.pi - 1e-8, np.pi]
)
def test_quaternion_from_matrix(angle):
    rng = np.random.default_rng(int(angle * 1e3))
    for _ in range(10):
        axis = pr.norm_vector(rng.standard_normal(3))
        with enable_x64():
            R = jr.matrix_from_compact_axis_angle(jnp.asarray(angle * axis))
            q = jr.quaternion_from_matrix(R)
            q_expected = pr.quaternion_from_compact_axis_angle(angle * axis)
            pr.assert_quaternion_equal(q, q_expected)
            jac = jax.jacrev(jr.quaternion_from_matrix)(R)
        assert np.isfinite(np.asarray(jac)).all()


def test_quaternion_from_matrix_batch():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 5, 3))
    R = jr.matrix_from_compact_axis_angle(a)
    q = jr.quaternion_from_matrix(R)
    assert q.shape == (2, 5, 4)
    for i in range(2):
        for j in range(5):
            pr.assert_quaternion_equal(
                q[i, j], pr.quaternion_from_compact_axis_angle(a[i, j]), decimal=5
            )


@pytest.mark.parametrize("angle", [0.0, 1e-15, 1e-4, 0.49, 0.51])
def test_series_match_closed_forms(angle):
    closed_forms = {
        ju.SINC_SERIES: lambda t: np.sin(t) / t,
        ju.COSC_SERIES: lambda t: (1.0 - np.cos(t)) / t**2,
        ju.SINC3_SERIES: lambda t: (t - np.sin(t)) / t**3,
        ju.COTC_SERIES: lambda t: (1.0 - 0.5 * t / np.tan(0.5 * t)) / t**2,
    }
    with enable_x64():
        for coefficients, closed_form in closed_forms.items():
            value = ju.series_or_closed_form(
                jnp.asarray(angle * AXIS), closed_form, coefficients
            )
            # the closed forms lose precision for small angles
            if angle > 0.1:
                assert_allclose(value, closed_form(angle), rtol=1e-12)
            else:
                assert_allclose(value, coefficients[0], rtol=1e-6)


def test_norm_dual_quaternion_gradients():
    for real in [jnp.zeros(4), 1e25 * jnp.array([1.0, 2.0, 3.0, 4.0])]:
        dq = jnp.concatenate((real, jnp.ones(4)))
        real_norm = jnp.linalg.norm(jt.norm_dual_quaternion(dq)[:4])
        assert_allclose(real_norm, 1.0, rtol=1e-6)
        for jac in (jax.jacfwd, jax.jacrev):
            assert np.isfinite(np.asarray(jac(jt.norm_dual_quaternion)(dq))).all()


def test_norm_vector_gradient_of_tiny_vector():
    for vec in [jnp.zeros(4), 1e-30 * jnp.ones(4), 1e25 * jnp.ones(4)]:
        assert np.isfinite(np.asarray(jax.jacrev(ju.norm_vector)(vec))).all()


def test_assert_compact_axis_angle_equal_at_pi():
    a = np.pi * np.array([0.0, 0.0, 1.0])
    jr.assert_compact_axis_angle_equal(-a + 1e-12, a)
    jr.assert_compact_axis_angle_equal(a + np.array([1e-12, -1e-12, 0.0]), a)
    with pytest.raises(AssertionError):
        jr.assert_compact_axis_angle_equal(-0.5 * a, 0.5 * a)
