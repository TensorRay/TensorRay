.. _Inverse diff render:


Differentiable rendering
=========================

We have showed how to perform forward rendering using TensorRay in the :ref:`previous section <Python render>`.
This section focuses on the use of TensorRay's Python bindings for *differentiable* and *inverse* rendering applications.

Similar to the forward-rendering case, after being loaded and configured, a scene can be rendered in a differentiable fashion by calling an integrator's ``renderD()`` method. The return value of this method is a TensorRay tensor of the type ``TR.Tensorf``.


Generating derivative image
---------------------------------------

The following example generates derivative images with respect to the translation of the first mesh about the Z-axis.

.. code-block:: batch

   cd TensorRay\example\validation

   rem Usage: python validate_gpu.py <config_file_name> backward
   python validate_gpu.py cbox_bunny.conf backward


Shape optimization using inverse rendering
----------------------------------------------

What follows is another example that optimizes all vertex positions of a triangular mesh.

.. code-block:: batch

   cd TensorRay\example\inverse_rendering

   rem Generate target images
   python preprocess_multi_pose.py kitty_in_cbox_diffuse_multi_pose.conf

   rem Inverse rendering
   python optimize_multi_pose.py kitty_in_cbox_diffuse_multi_pose.conf
