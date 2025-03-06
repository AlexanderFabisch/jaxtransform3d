import jax
import jax.numpy as jnp
import numpy as np
import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.transformations as jt

create_transform = jax.jit(jt.create_transform)
exponential_coordinates_from_transform = jax.jit(
    jt.exponential_coordinates_from_transform
)
compose_transforms = jax.jit(jt.compose_transforms)


def test_create_transform():
    R1 = jnp.eye(3)
    t1 = jnp.zeros(3)
    T1 = create_transform(R1, t1)
    assert T1.shape == (4, 4)
    assert_array_almost_equal(T1, jnp.eye(4))

    R2 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t2 = jnp.array([1, 2, 3])
    T2 = create_transform(R2, t2)
    assert_array_almost_equal(T2[:3, 3], t2)
    assert_array_almost_equal(T2[:3, :3], R2)
    assert jnp.all(T2[3, :3] == 0)
    assert T2[3, 3] == 1

    T = create_transform(jnp.array([R1, R2]), jnp.array([t1, t2]))
    assert_array_almost_equal(T, jnp.stack([T1, T2]))


def test_apply_transform():
    rng = np.random.default_rng(843)
    exp_coords = rng.normal(size=(20, 6))
    T = jt.transform_from_exponential_coordinates(exp_coords)
    v = jnp.asarray(rng.normal(size=(20, 3)))

    w = jt.apply_transform(T, v)
    w00 = jt.apply_transform(T[0], v[0])
    w11 = jt.apply_transform(T[1], v[1])
    assert_array_almost_equal(w[0], w00)
    assert_array_almost_equal(w[1], w11)

    w01 = jt.apply_transform(T[0], v[1])
    w0 = jt.apply_transform(T[0], v)
    assert_array_almost_equal(w0[0], w00)
    assert_array_almost_equal(w0[1], w01)


def test_compose_transform():
    rng = np.random.default_rng(832)
    T1 = jt.transform_from_exponential_coordinates(rng.normal(size=(20, 6)))
    T2 = jt.transform_from_exponential_coordinates(rng.normal(size=(20, 6)))

    T = compose_transforms(T1, T2)
    T00 = compose_transforms(T1[0], T2[0])
    T11 = compose_transforms(T1[1], T2[1])
    assert_array_almost_equal(T[0], T00)
    assert_array_almost_equal(T[1], T11)

    T0 = compose_transforms(T1[0], T2)
    T01 = compose_transforms(T1[0], T2[1])
    assert_array_almost_equal(T0[0], T[0])
    assert_array_almost_equal(T0[1], T01)


def test_exponential_coordinates_from_transform_0dim():
    T1 = jnp.eye(4)
    exp_coords1 = exponential_coordinates_from_transform(T1)
    assert_array_almost_equal(exp_coords1, jnp.zeros(6))

    R2 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    exp_coords2 = exponential_coordinates_from_transform(
        create_transform(R2, jnp.zeros(3))
    )
    assert_array_almost_equal(exp_coords2, jnp.array([0, 0, 0.5 * jnp.pi, 0, 0, 0]))

    exp_coords3 = exponential_coordinates_from_transform(
        create_transform(jnp.eye(3), jnp.array([1, 2, 3]))
    )
    assert_array_almost_equal(exp_coords3, jnp.array([0, 0, 0, 1, 2, 3]))

    rng = np.random.default_rng(83)
    for _ in range(20):
        T = pt.random_transform(rng)
        exp_coords = exponential_coordinates_from_transform(T)
        assert_array_almost_equal(
            exp_coords, ptr.exponential_coordinates_from_transforms(T)
        )


def test_exponential_coordinates_from_transform_2dim():
    rng = np.random.default_rng(84)
    T = ptr.random_trajectories(rng, n_trajectories=3, n_steps=5)
    exp_coords = exponential_coordinates_from_transform(T)
    assert_array_almost_equal(
        exp_coords, ptr.exponential_coordinates_from_transforms(T), decimal=5
    )
