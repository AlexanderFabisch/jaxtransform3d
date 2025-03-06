import jax
import numpy as np
import jax.numpy as jnp
import jaxtransform3d.transformations as jt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
import matplotlib.pyplot as plt

rng = np.random.default_rng(23)
pointcloud_A = jnp.asarray(rng.normal(size=(15, 3)))
exp_coords_BA_actual = 0.3 * rng.normal(size=6)
T_BA = jt.transform_from_exponential_coordinates(exp_coords_BA_actual)
pointcloud_B = jt.apply_transform(T_BA, pointcloud_A)


def error(exp_coords_BA):
    T_BA = jt.transform_from_exponential_coordinates(exp_coords_BA)
    pointcloud_B_expected = jt.apply_transform(T_BA, pointcloud_A)
    return jnp.mean(jnp.linalg.norm(pointcloud_B - pointcloud_B_expected, axis=1))


error_with_grad = jax.jit(jax.value_and_grad(error))

exp_coords_BA = jnp.asarray(rng.normal(size=6))

for it in range(1000):
    e, grad = error_with_grad(exp_coords_BA)
    print(f"{it=}, {e=:.3f}, {jnp.linalg.norm(grad)=:.3f}")
    if e == 0:
        break
    exp_coords_BA = exp_coords_BA - 0.005 * grad

T_BA_estimated = jt.transform_from_exponential_coordinates(exp_coords_BA)

print(f"{T_BA=}")
print(f"{T_BA_estimated=}")

pointcloud_B_estimated = jt.apply_transform(T_BA_estimated, pointcloud_A)

ax = ppu.make_3d_axis(ax_s=2.0)
ax.scatter(
    pointcloud_A[:, 0],
    pointcloud_A[:, 1],
    pointcloud_A[:, 2],
    c="k",
    label="Untransformed",
)
ax.scatter(
    pointcloud_B[:, 0],
    pointcloud_B[:, 1],
    pointcloud_B[:, 2],
    c="r",
    alpha=0.5,
    s=50,
    label="With actual transform",
)
ax.scatter(
    pointcloud_B_estimated[:, 0],
    pointcloud_B_estimated[:, 1],
    pointcloud_B_estimated[:, 2],
    c="b",
    label="With estimated transform",
)
pt.plot_transform(ax=ax, A2B=T_BA, name="Actual", alpha=0.3, lw=5, s=1.2)
pt.plot_transform(ax=ax, A2B=T_BA_estimated, name="Estimated")
plt.legend()
plt.show()
