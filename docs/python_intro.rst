.. _Python intro:


Introduction
====================

Importing the renderer
------------------------------

To import TensorRay into Python, you only need to import its central module named ``TensorRay``:

.. code-block:: python3

   import TensorRay as TR


Examples and scene assets
------------------------------

We created a shared `folder <https://drive.google.com/drive/folders/1mxgyqNBgS8HTEeLDFD0yy0nyEt6q2OJC?usp=sharing>`_ to host all the scene assets used in the differentiable and inverse rendering Python scripts.
This folder should be copied to and merged with ``TensorRay\example``.


Additional packages
------------------------------

Additional packages are needed to run the included Python scripts under ``TensorRay\example\validation`` and ``TensorRay\example\inverse_rendering``. You can install those packages by running the following commands:

.. code-block:: batch

   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   pip install OpenEXR scikit-image gin-config
   pip install opencv-python==4.3.0.38
   pip install cholespy
   pip install largesteps
