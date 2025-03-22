import chex
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..utils import norm_vector
from ._axis_angle import matrix_from_compact_axis_angle


def norm_matrix(R: ArrayLike) -> jax.Array:
    r"""Orthonormalize rotation matrix.

    A rotation matrix is defined as

    .. math::

        \boldsymbol R =
        \left( \begin{array}{ccc}
            r_{11} & r_{12} & r_{13}\\
            r_{21} & r_{22} & r_{23}\\
            r_{31} & r_{32} & r_{33}\\
        \end{array} \right)
        \in SO(3)

    and must be orthonormal, which results in 6 constraints:

    * column vectors must have unit norm (3 constraints)
    * and must be orthogonal to each other (3 constraints)

    A more compact representation of these constraints is
    :math:`\boldsymbol R^T \boldsymbol R = \boldsymbol I`. In addition,
    to ensure right-handedness of the basis :math:`det(R) = 1`.

    Because of numerical problems, a rotation matrix might not satisfy the
    constraints anymore. This function will enforce them with Gram-Schmidt
    orthonormalization optimized for 3 dimensions.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix with small numerical errors.

    Returns
    -------
    R : array, shape (3, 3)
        Orthonormalized rotation matrix.

    See Also
    --------
    robust_polar_decomposition
        A more expensive orthonormalization method that spreads the error more
        evenly between the basis vectors.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import norm_matrix
    >>> norm_matrix(jnp.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]]))
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    """
    R = jnp.asarray(R)

    chex.assert_shape(R, (3, 3))

    c2 = R[:, 1]
    c3 = norm_vector(R[:, 2])
    c1 = norm_vector(jnp.cross(c2, c3))
    c2 = norm_vector(jnp.cross(c3, c1))
    return jnp.column_stack((c1, c2, c3))


def robust_polar_decomposition(
    A: ArrayLike, n_iter: int = 20, eps: float = 1e-8
) -> jax.Array:
    r"""Orthonormalize rotation matrix with robust polar decomposition.

    Robust polar decomposition [1]_ [2]_ is a computationally more costly
    method, but it spreads the error more evenly between the basis vectors
    in comparison to Gram-Schmidt orthonormalization (as in
    :func:`norm_matrix`).

    Robust polar decomposition finds an orthonormal matrix that minimizes the
    Frobenius norm

    .. math::

        ||\boldsymbol{A} - \boldsymbol{R}||^2

    between the input :math:`\boldsymbol{A}` that is not orthonormal and the
    output :math:`\boldsymbol{R}` that is orthonormal.

    Parameters
    ----------
    A : array-like, shape (3, 3)
        Matrix that contains a basis vector in each column. The basis does not
        have to be orthonormal.

    n_iter : int, optional (default: 20)
        Maximum number of iterations for which we refine the estimation of the
        rotation matrix.

    eps : float, optional (default: 1e-8)
        Precision for termination criterion of iterative refinement.

    Returns
    -------
    R : array, shape (3, 3)
        Orthonormalized rotation matrix.

    See Also
    --------
    norm_matrix
        The cheaper default orthonormalization method that uses Gram-Schmidt
        orthonormalization optimized for 3 dimensions.

    References
    ----------
    .. [1] Selstad, J. (2019). Orthonormalization.
       https://zalo.github.io/blog/polar-decomposition/

    .. [2] Müller, M., Bender, J., Chentanez, N., Macklin, M. (2016).
       A Robust Method to Extract the Rotational Part of Deformations.
       In MIG '16: Proceedings of the 9th International Conference on Motion in
       Games, pp. 55-60, doi: 10.1145/2994258.2994269.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtransform3d.rotations import robust_polar_decomposition
    >>> robust_polar_decomposition(
    ...     jnp.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]]))
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)
    """
    A = jnp.asarray(A)

    chex.assert_shape(A, (3, 3))
    chex.assert_type(n_iter, expected_types=int)
    chex.assert_scalar_positive(eps)

    current_R = jnp.eye(3)

    for _ in range(n_iter):
        column_vector_cross_products = jnp.cross(
            current_R, A, axisa=0, axisb=0, axisc=1
        )
        column_vector_dot_products_sum = jnp.sum(current_R * A)
        omega = column_vector_cross_products.sum(axis=0) / (
            abs(column_vector_dot_products_sum) + eps
        )
        current_R = jnp.dot(matrix_from_compact_axis_angle(omega), current_R)
    return current_R
