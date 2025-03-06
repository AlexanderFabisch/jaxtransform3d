"""Utility functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def norm_vector(vec: ArrayLike, norm: ArrayLike | None = None) -> jax.Array:
    """Normalize vector.

    Parameters
    ----------
    vec : array-like, shape (..., n)
        nd vectors.

    norm : array-like, shape (...)
        Precomputed norms of vectors if available.

    Returns
    -------
    vec_unit : array, shape (..., n)
        nd unit vectors with norm 1 or zero vectors.
    """
    vec = jnp.asarray(vec)
    if norm is None:
        norm = jnp.linalg.norm(vec, axis=-1)
    # Avoid division by zero with np.maximum(..., smallest positive float).
    # The norm is zero only when the vector is zero so this case does not
    # require further processing.
    return vec / jnp.maximum(norm[..., jnp.newaxis], jnp.finfo(vec.dtype).tiny)
