import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.transformations as jt

create_transform = jax.jit(jt.create_transform)
exponential_coordinates_from_dual_quaternion = jax.jit(
    jt.exponential_coordinates_from_dual_quaternion
)
apply_dual_quaternion = jax.jit(jt.apply_dual_quaternion)


def test_apply_dual_quaternion():
    rng = np.random.default_rng(83)
    for _ in range(20):
        exp_coords = rng.normal(size=6)
        dual_quat = jt.dual_quaternion_from_exponential_coordinates(exp_coords)
        v = rng.normal(size=3)
        w1 = apply_dual_quaternion(dual_quat, v)
        T = jt.transform_from_exponential_coordinates(exp_coords)
        w2 = jt.apply_transform(T, v)
        assert_array_almost_equal(w1, w2)

    exp_coords = rng.normal(size=(20, 6))
    dual_quat = jt.dual_quaternion_from_exponential_coordinates(exp_coords)
    v = rng.normal(size=(20, 3))
    w1 = apply_dual_quaternion(dual_quat, v)
    T = jt.transform_from_exponential_coordinates(exp_coords)
    w2 = jt.apply_transform(T, v)
    assert_array_almost_equal(w1, w2)

    exp_coords = rng.normal(size=(5, 4, 6))
    dual_quat = jt.dual_quaternion_from_exponential_coordinates(exp_coords)
    v = rng.normal(size=(5, 4, 3))
    w1 = apply_dual_quaternion(dual_quat, v)
    T = jt.transform_from_exponential_coordinates(exp_coords)
    w2 = jt.apply_transform(T, v)
    assert_array_almost_equal(w1, w2)


def test_exponential_coordinates_from_dual_quaternion_0dim():
    dual_quat1 = jnp.array([1, 0, 0, 0, 0, 0, 0, 0])
    exp_coords1 = exponential_coordinates_from_dual_quaternion(dual_quat1)
    assert_array_almost_equal(exp_coords1, jnp.zeros(6))

    dual_quat2 = jnp.array([0, 1, 0, 0, 0, 0, 0, 0])
    exp_coords2 = exponential_coordinates_from_dual_quaternion(dual_quat2)
    assert_array_almost_equal(exp_coords2, jnp.array([jnp.pi, 0, 0, 0, 0, 0]))

    dual_quat3 = jnp.array([1, 0, 0, 0, 0, 0.5, 1, 1.5])
    exp_coords3 = exponential_coordinates_from_dual_quaternion(dual_quat3)
    assert_array_almost_equal(exp_coords3, jnp.array([0, 0, 0, 1, 2, 3]))

    rng = np.random.default_rng(83)
    for _ in range(20):
        T = pt.random_transform(rng)
        dual_quat = pt.dual_quaternion_from_transform(T)
        exp_coords = exponential_coordinates_from_dual_quaternion(dual_quat)
        assert_array_almost_equal(
            exp_coords, ptr.exponential_coordinates_from_transforms(T)
        )


def test_exponential_coordinates_from_dual_quaternion_2dims():
    rng = np.random.default_rng(85)
    T = ptr.random_trajectories(rng, n_trajectories=3, n_steps=5)
    dual_quat = ptr.dual_quaternions_from_transforms(T)
    exp_coords = exponential_coordinates_from_dual_quaternion(dual_quat)
    assert_array_almost_equal(
        exp_coords, ptr.exponential_coordinates_from_transforms(T), decimal=5
    )
