import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.trajectories as ptr
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.transformations as jt

transform_from_exponential_coordinates = jax.jit(
    jt.transform_from_exponential_coordinates
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
            ptr.transform_from_exponential_coordinates(exp_coords),
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
