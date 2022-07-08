.. _Components:

Components
====================

TensorRay contains the following main components:

- A CUDA meta-programming library for vectorized computations and automatic differentiation;

- Monte Carlo path-space differential integral estimators;

- Python bindings and scripts.


CUDA meta-programming library
-----------------------------------

Our library traces computational graphs and uses NVIDIA jitify for just-in-time (JIT) CUDA kernel compilation and execution. It uses similar tensor-based programming used in deep learning frameworks such as PyTorch. To make the computations more efficient, we greedily fuse complicated arithmetic and memory indexing operations to reduce the number of kernels launched and reduce memory bandwidth overhead. We also cache the CUDA kernels to avoid repeated code compilation.


Differential integral estimators
-----------------------------------

We implemented the following differential integrators for estimating unbiased scene derivatives:

- ``PathTracer``,

- ``PrimaryEdgeIntegrator``,

- ``DirectEdgeIntegrator``,

- ``IndirectEdgeIntegrator``,

- ``PixelBoundaryIntegrator``.

Discontinuities (usually caused by visibility) are handled by estimating edge/boundary integrals and explicit edge sampling. We also apply variance reduction techniques such as antithetic sampling and primary-sample-space guiding for efficient gradient estimation. These integrators are carefully validated with the CPU-based implementation of PSDR.




Python bindings and scripts
-----------------------------------

TensorRay can be easily used in Python for inverse rendering applications. We have provided examples in the :ref:`previous section <Inverse diff render>`. Many useful Python functions are located under ``TensorRay\pyTensorRay``.