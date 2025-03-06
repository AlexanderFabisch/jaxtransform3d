import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def compact_axis_angle_from_matrix(R: ArrayLike) -> jax.Array:
    """Compute axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (..., 3, 3)
        Rotation matrix.

    Returns
    -------
    a : array, shape (..., 3)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    R = jnp.asarray(R)
    if not jnp.issubdtype(R.dtype, jnp.floating):
        R = R.astype(jnp.float64)

    instances_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    traces = jnp.einsum("nii", R)
    cos_angle = jnp.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    axis_unnormalized = jnp.column_stack(
        (R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1])
    )

    R_diag = jnp.clip(jnp.einsum("nii->ni", R), -1.0, 1.0)
    eeT_diag = 0.5 * (R_diag + 1.0)
    signs = 2.0 * (axis_unnormalized >= 0.0).astype(R.dtype) - 1.0
    axis_close_to_pi = jnp.sqrt(eeT_diag) * signs

    angle_close_to_pi = abs(angle - jnp.pi) < 1e-4
    axis = jnp.where(
        angle_close_to_pi[:, jnp.newaxis], axis_close_to_pi, axis_unnormalized
    )
    axis = axis / jnp.linalg.norm(axis, axis=-1)[..., jnp.newaxis]

    angle_nonzero = angle != 0.0
    axis = jnp.where(angle_nonzero[:, jnp.newaxis], axis, 0.0)

    rot_vec = axis * angle[:, jnp.newaxis]

    if instances_shape:
        rot_vec = rot_vec.reshape(*instances_shape + (3,))
    else:
        rot_vec = rot_vec[0]

    return rot_vec
