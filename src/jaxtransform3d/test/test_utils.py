import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.utils as ju

norm_vector = jax.jit(ju.norm_vector)
cross_product_matrix = jax.jit(ju.cross_product_matrix)


def test_norm_vectors_0dim():
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        vec = rng.normal(size=n)
        vec_unit = norm_vector(vec)
        assert pytest.approx(jnp.linalg.norm(vec_unit)) == 1


def test_norm_vectors_1dim():
    rng = np.random.default_rng(8381)
    vec = rng.standard_normal(size=(100, 3))
    vec_unit = norm_vector(vec)
    assert_array_almost_equal(np.linalg.norm(vec_unit, axis=1), np.ones(len(vec)))


def test_norm_vectors_3dims():
    rng = np.random.default_rng(8382)
    vec = rng.standard_normal(size=(8, 2, 8, 3))
    vec_unit = norm_vector(vec)
    assert_array_almost_equal(
        jnp.linalg.norm(vec_unit, axis=-1), jnp.ones(vec_unit.shape[:-1])
    )


def test_norm_zero_vector():
    normalized = norm_vector(jnp.zeros(3))
    assert jnp.isfinite(jnp.linalg.norm(normalized))


def test_cross_product_matrix_0dim():
    v = jnp.array([1.0, 2.0, 3.0])
    w = jnp.array([1.5, 2.5, 3.5])
    V = cross_product_matrix(v)
    assert_array_almost_equal(V @ w, jnp.cross(v, w))
    assert_array_almost_equal(-V, V.T)


def test_cross_product_matrix_1dim():
    rng = np.random.default_rng(8383)
    v = rng.standard_normal(size=(100, 3))
    V = cross_product_matrix(v)
    assert_array_almost_equal(-V, V.transpose([0, 2, 1]))


def test_cross_product_matrix_2dim():
    rng = np.random.default_rng(8384)
    v = rng.standard_normal(size=(5, 4, 3))
    V = cross_product_matrix(v)
    assert_array_almost_equal(-V, V.transpose([0, 1, 3, 2]))
