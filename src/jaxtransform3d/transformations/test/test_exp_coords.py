import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
from jax.experimental import enable_x64
from numpy.testing import assert_allclose, assert_array_almost_equal

import jaxtransform3d.transformations as jt

transform_from_exponential_coordinates = jax.jit(
    jt.transform_from_exponential_coordinates
)
dual_quaternion_from_exponential_coordinates = jax.jit(
    jt.dual_quaternion_from_exponential_coordinates
)


def test_transform_from_exponential_coordinates_0dim():
    T = transform_from_exponential_coordinates(jnp.zeros(6))
    assert_array_almost_equal(T, jnp.eye(4))

    T = transform_from_exponential_coordinates(
        jnp.array([0, 0, 0, 2, 3, 4], dtype=float)
    )
    assert_array_almost_equal(T, jt.create_transform(R=jnp.eye(3), t=jnp.arange(2, 5)))

    T = transform_from_exponential_coordinates(jnp.array([jnp.pi, 0, 0, 0, 0, 0]))
    assert_array_almost_equal(
        T,
        jt.create_transform(
            R=jnp.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), t=jnp.zeros(3)
        ),
    )

    rng = np.random.default_rng(22)
    for _ in range(50):
        exp_coords = rng.normal(size=6)
        T = transform_from_exponential_coordinates(exp_coords)
        assert_array_almost_equal(
            T,
            ptr.transforms_from_exponential_coordinates(exp_coords),
        )


def test_transform_from_exponential_coordinates_1dim():
    rng = np.random.default_rng(84)
    exp_coords = rng.standard_normal(size=(20, 6))
    exp_coords[0] = 0.0
    exp_coords[1, :3] = 0.0
    exp_coords[2, 3:] = 0.0

    T_actual = transform_from_exponential_coordinates(exp_coords)
    T_expected = ptr.transforms_from_exponential_coordinates(exp_coords)
    assert_array_almost_equal(T_actual, T_expected)


def test_transform_from_exponential_coordinates_2dims():
    rng = np.random.default_rng(84)
    exp_coords = rng.standard_normal(size=(5, 4, 6))

    T_actual = transform_from_exponential_coordinates(exp_coords)
    T_expected = ptr.transforms_from_exponential_coordinates(exp_coords)
    assert_array_almost_equal(T_actual, T_expected)


def test_dual_quaternion_from_exponential_coordinates_0dim():
    dual_quat = dual_quaternion_from_exponential_coordinates(jnp.zeros(6))
    assert_array_almost_equal(dual_quat, jnp.array([1, 0, 0, 0, 0, 0, 0, 0]))

    dual_quat = dual_quaternion_from_exponential_coordinates(
        jnp.array([0, 0, 0, 2, 3, 4], dtype=float)
    )
    assert_array_almost_equal(dual_quat, jnp.array([1, 0, 0, 0, 0, 1, 1.5, 2]))

    dual_quat = dual_quaternion_from_exponential_coordinates(
        jnp.array([0.5 * np.pi, 0, 0, 0, 0, 0], dtype=float)
    )
    assert_array_almost_equal(
        dual_quat, jnp.array([0.707107, 0.707107, 0, 0, 0, 0, 0, 0])
    )

    rng = np.random.default_rng(25)
    for _ in range(20):
        exp_coords = rng.normal(size=6)
        dual_quat = dual_quaternion_from_exponential_coordinates(exp_coords)
        pt.assert_unit_dual_quaternion_equal(
            dual_quat,
            pt.dual_quaternion_from_transform(
                pt.transform_from_exponential_coordinates(exp_coords)
            ),
        )


def test_dual_quaternion_from_exponential_coordinates_1dim():
    rng = np.random.default_rng(84)
    exp_coords = rng.standard_normal(size=(20, 6))
    exp_coords[0] = 0.0
    exp_coords[1, :3] = 0.0
    exp_coords[2, 3:] = 0.0

    dual_quat_actual = dual_quaternion_from_exponential_coordinates(exp_coords)
    dual_quat_expected = ptr.dual_quaternions_from_transforms(
        ptr.transforms_from_exponential_coordinates(exp_coords)
    )
    flip = np.sign(dual_quat_actual[:, 0]) != np.sign(dual_quat_expected[:, 0])
    dual_quat_actual = dual_quat_actual.at[flip].set(-dual_quat_actual[flip])
    assert_array_almost_equal(dual_quat_actual, dual_quat_expected)


def test_dual_quaternion_from_exponential_coordinates_2dims():
    rng = np.random.default_rng(84)
    exp_coords = rng.standard_normal(size=(5, 4, 6))

    dual_quat_actual = dual_quaternion_from_exponential_coordinates(exp_coords)
    dual_quat_expected = ptr.dual_quaternions_from_transforms(
        ptr.transforms_from_exponential_coordinates(exp_coords)
    )
    flip = np.sign(dual_quat_actual[..., 0]) != np.sign(dual_quat_expected[..., 0])
    dual_quat_actual = dual_quat_actual.at[flip].set(-dual_quat_actual[flip])
    assert_array_almost_equal(dual_quat_actual, dual_quat_expected)


def test_logarithmic_maps_small_angles():
    """The translation survives when the rotation is tiny or zero."""
    with enable_x64():
        axis = jnp.array([1.0, 2.0, -2.0]) / 3.0
        t = jnp.array([0.3, -0.2, 0.5])
        for angle in [1e-4, 1e-6, 1e-8, 1e-10, 1e-16, 1e-100, 0.0]:
            exp_coords = jnp.concatenate((axis * angle, t))
            T = jt.transform_from_exponential_coordinates(exp_coords)
            assert_allclose(
                jt.exponential_coordinates_from_transform(T), exp_coords, rtol=1e-12
            )
            dq = jt.dual_quaternion_from_exponential_coordinates(exp_coords)
            assert_allclose(
                jt.exponential_coordinates_from_dual_quaternion(dq),
                exp_coords,
                rtol=1e-12,
            )


def test_logarithmic_maps_gradient_identity_and_pure_translation():
    for t in [jnp.zeros(3), jnp.array([0.3, -0.2, 0.5])]:
        exp_coords = jnp.concatenate((jnp.zeros(3), t))
        T = jt.transform_from_exponential_coordinates(exp_coords)
        dq = jt.dual_quaternion_from_exponential_coordinates(exp_coords)
        for jac in (jax.jacfwd, jax.jacrev):
            J_T = jac(jt.exponential_coordinates_from_transform)(T)
            assert np.isfinite(np.asarray(J_T)).all()
            J_dq = jac(jt.exponential_coordinates_from_dual_quaternion)(dq)
            assert np.isfinite(np.asarray(J_dq)).all()
