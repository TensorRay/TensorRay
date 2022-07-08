.. _Core compile:

Compiling the system
=========================

Compiling TensorRay from scratch requires recent versions of CMake (at least 3.8.8), Python (at least 3.8), CUDA 11.5, OptiX 7.4, as well as a C++ compiler that supports the C++17 standard.


Windows
--------------------

To make the compilation under Windows more streamlined, we created a `package <https://drive.google.com/drive/folders/1-H8knPaY5HTia2nQusWF3LZoKVg5eU2U?usp=sharing>`_ containing precompiled binaries of most of the dependencies. Utilizing this package requires having Visual Studio 2019, CUDA 11.5, and Python 3.8.

After decompressing this package to ``ext_win64`` under TensorRay's root directory, assuming Python to be installed under ``C:/Users/User/Anaconda3`` and having an RTX 3090 GPU (the value of `CUDA_NVCC_FLAGS` needs to be changed according to the GPU, see `this <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`_), TensorRay can be built by running the following commands under the command prompt:

.. code-block:: batch

   rem Create a directory where build products are stored
   mkdir build
   cd build
   cmake -DCUDA_NVCC_FLAGS="-arch=sm_86" -DPYTHON_ROOT="C:/Users/User/Anaconda3" -G "Visual Studio 16 2019" -A x64 ..
   cmake --build . -j --config Release
   copy /y lib\Release\*.pyd lib\


After compiling TensorRay, add ``TensorRay\build\lib`` to the `PYTHONPATH` environment variable.


Tested version
^^^^^^^^^^^^^^
* NVIDIA GeForce RTX 3090
* Windows 10
* Visual Studio 2019
* Python 3.8.12 (Anaconda)
* CUDA 11.5
* OptiX 7.4


Linux
--------------------

We haven't tested building TensorRay under Linux.


Mac OS
--------------------

Unfortunately, TensorRay does not work under the Mac OS due to the lack of CUDA support.
