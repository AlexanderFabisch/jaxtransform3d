.. _api:

=================
API Documentation
=================

:mod:`jaxtransform3d.rotations`
===============================

.. automodule:: jaxtransform3d.rotations
    :no-members:
    :no-inherited-members:

Rotation Matrices
-----------------

.. autosummary::
   :toctree: _apidoc/

   ~norm_matrix
   ~robust_polar_decomposition
   ~matrix_inverse
   ~compose_matrices
   ~apply_matrix
   ~matrix_from_compact_axis_angle
   ~compact_axis_angle_from_matrix

Quaternions
-----------

.. autosummary::
   :toctree: _apidoc/

   ~norm_quaternion
   ~quaternion_conjugate
   ~compose_quaternions
   ~apply_quaternion
   ~quaternion_from_compact_axis_angle
   ~compact_axis_angle_from_quaternion


:mod:`jaxtransform3d.transformations`
=====================================

.. automodule:: jaxtransform3d.transformations
    :no-members:
    :no-inherited-members:

Transformation Matrices
-----------------------

.. autosummary::
   :toctree: _apidoc/

   ~transform_inverse
   ~apply_transform
   ~compose_transforms
   ~create_transform
   ~transform_from_exponential_coordinates
   ~exponential_coordinates_from_transform

Dual Quaternions
----------------

.. autosummary::
   :toctree: _apidoc/

   ~norm_dual_quaternion
   ~dual_quaternion_squared_norm
   ~dual_quaternion_quaternion_conjugate
   ~compose_dual_quaternions
   ~apply_dual_quaternion
   ~dual_quaternion_from_exponential_coordinates
   ~exponential_coordinates_from_dual_quaternion
