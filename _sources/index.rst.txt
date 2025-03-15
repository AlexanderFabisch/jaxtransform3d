.. jaxtransform3d documentation master file, created by
   sphinx-quickstart on Sat Mar  8 20:54:32 2025.

jaxtransform3d
==============

.. toctree::
   :hidden:

   api
   _auto_examples/index

This library is still experimental and a lot of things are subject to change,
e.g., the name. The core idea is to create a version of
https://github.com/dfki-ric/pytransform3d
that is JIT-compiled, executable on GPU, differentiable, and inherently
vectorized.

**Why should it be inherently vectorized?**

The decision is not final yet. As long as it is not too complicated to write
the code to accept any-dimensional input arrays I will do it. There are only a
few exceptions: rotation matrix normalization methods only accept one rotation
matrix. If it gets to complicated or I see significant performance issues
in comparison to vmapping the code, I will abandon this idea. The advantage
over using vmap is that I don't have to call vmap multiple times when I want
to vectorize multiple times (e.g., a list of a list of a list of rotation
matrices).

--------
Citation
--------

If you use jaxtransform3d for a scientific publication, I would appreciate
citation of the following paper for now, since jaxtransform3d is a spin-off
of pytransform3d:

Fabisch, A. (2019). pytransform3d: 3D Transformations for Python.
Journal of Open Source Software, 4(33), 1159,
https://doi.org/10.21105/joss.01159
