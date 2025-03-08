.. _api:

=================
API Documentation
=================

:mod:`jaxtransform3d.rotations`
===============================

.. automodule:: jaxtransform3d.rotations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/

   ~matrix_from_compact_axis_angle
   ~quaternion_from_compact_axis_angle
   ~matrix_inverse
   ~apply_matrix
   ~compose_matrices
   ~compact_axis_angle_from_matrix
   ~norm_quaternion
   ~compose_quaternions
   ~quaternion_conjugate
   ~apply_quaternion
   ~compact_axis_angle_from_quaternion
   ~norm_matrix
   ~robust_polar_decomposition


:mod:`jaxtransform3d.transformations`
=====================================

.. automodule:: jaxtransform3d.transformations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/

   ~transform_inverse
   ~apply_transform
   ~compose_transforms
   ~create_transform
   ~exponential_coordinates_from_transform
   ~transform_from_exponential_coordinates
   ~dual_quaternion_from_exponential_coordinates
   ~norm_dual_quaternion
   ~dual_quaternion_squared_norm
   ~compose_dual_quaternions
   ~dual_quaternion_quaternion_conjugate
   ~apply_dual_quaternion
   ~exponential_coordinates_from_dual_quaternion
