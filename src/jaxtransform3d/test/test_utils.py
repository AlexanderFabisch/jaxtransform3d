import jax.numpy as jnp
import numpy as np
import pytest

import jaxtransform3d.utils as ju


def test_norm_vector():
    """Test normalization of vectors."""
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        vec = rng.normal(size=n)
        vec_unit = ju.norm_vector(vec)
        assert pytest.approx(jnp.linalg.norm(vec_unit)) == 1


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = ju.norm_vector(jnp.zeros(3))
    assert jnp.isfinite(jnp.linalg.norm(normalized))
