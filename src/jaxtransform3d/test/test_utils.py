import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.utils as ju

norm_vector = jax.jit(ju.norm_vector)
cross_product_matrix = jax.jit(ju.cross_product_matrix)


def test_differentiable_arccos():
    x = jnp.logspace(-20, -2, 100)
    x = jnp.hstack((-1.0 + x, 1.0 - x[::-1]))
    y = ju.differentiable_arccos(x)
    assert_array_almost_equal(jnp.arccos(x), y)

    x = jnp.array([-1.0, 1.0])
    jac_fwd = jax.jacfwd(ju.differentiable_arccos)(x)
    assert jnp.isfinite(jac_fwd).all()

    jax_rev = jax.jacrev(ju.differentiable_arccos)(x)
    assert jnp.isfinite(jax_rev).all()

    grad = jax.grad(lambda x: ju.differentiable_arccos(x).sum())(x)
    assert jnp.isfinite(grad).all()


def test_differentiable_norm():
    rng = np.random.default_rng(2323)
    vec = jnp.asarray(rng.standard_normal(size=(100, 3)))
    norm = ju.differentiable_norm(vec, axis=-1)
    assert_array_almost_equal(norm, jnp.linalg.norm(vec, axis=-1))

    norm_0 = ju.differentiable_norm(jnp.zeros(3), axis=0)
    assert norm_0 == 0.0

    norm_grad = jax.grad(ju.differentiable_norm, argnums=0)
    assert jnp.isfinite(norm_grad(jnp.zeros(3), axis=0)).all()
    assert jnp.isfinite(norm_grad(1e-25 * jnp.ones(3), axis=0)).all()


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
