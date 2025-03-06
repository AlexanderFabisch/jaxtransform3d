import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.utils as ju


def test_norm_vectors_0dim():
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        vec = rng.normal(size=n)
        vec_unit = ju.norm_vector(vec)
        assert pytest.approx(jnp.linalg.norm(vec_unit)) == 1


def test_norm_vectors_1dim():
    rng = np.random.default_rng(8381)
    vec = rng.standard_normal(size=(100, 3))
    vec_unit = ju.norm_vector(vec)
    assert_array_almost_equal(np.linalg.norm(vec_unit, axis=1), np.ones(len(vec)))


def test_norm_vectors_3dims():
    rng = np.random.default_rng(8382)
    vec = rng.standard_normal(size=(8, 2, 8, 3))
    vec_unit = ju.norm_vector(vec)
    assert_array_almost_equal(
        jnp.linalg.norm(vec_unit, axis=-1), jnp.ones(vec_unit.shape[:-1])
    )


def test_norm_zero_vector():
    normalized = ju.norm_vector(jnp.zeros(3))
    assert jnp.isfinite(jnp.linalg.norm(normalized))
