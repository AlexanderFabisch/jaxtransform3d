import jax
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.transformations as jt

create_transform = jax.jit(jt.create_transform)


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
