import jax
import pytransform3d.rotations as pr
from numpy.testing import assert_array_almost_equal

import jaxtransform3d.rotations as jr


def test_left_jacobian_SO3():
    key = jax.random.key(42)
    axis_angle = jax.random.normal(key, shape=(10, 3))
    jac = jr.left_jacobian_SO3(axis_angle)
    for a, j in zip(axis_angle, jac, strict=False):
        jac_gt = pr.left_jacobian_SO3(a)
        assert_array_almost_equal(j, jac_gt)
