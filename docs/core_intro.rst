.. _Core intro:

Getting started
====================

TensorRay is a physics-based differenetiable renderer written in C++17/CUDA and running on the GPU. 
It is based on the theoretical framework introduced in the paper titled "`Path-Space Differentiable Rendering <https://shuangz.com/projects/psdr-sg20/>`_".

TensorRay is light-weight (compared to other modern renderers like `Mitsuba 2 <https://www.mitsuba-renderer.org/>`_) yet powerful (similar to `PSDR-CUDA <https://github.com/uci-rendering/psdr-cuda>`_), offering the following features:

\1. **Vectorized computation:** TensorRay uses a CUDA-based meta-programming framework to perform vectorized computations and automatic differentiation (including both forward-mode and reserve-mode AD) on the GPU using `CUDA 11.5 <https://developer.nvidia.com/cuda-toolkit>`_ and `Jitify <https://github.com/NVIDIA/jitify>`_ for just-in-time (JIT) compilation.

\2. **Fast GPU-based ray tracing:** TensorRay uses `OptiX 7.4 <https://developer.nvidia.com/optix/>`_ that leverages RTX ray tracing on modern NVIDIA graphics hardware.

\3. **Fast and unbiased gradients:** TensorRay implements state-of-the-art algorithms that produce unbiased gradient estimates with respect to *arbitrary* scene parameters (e.g., object geometry and spatially varying reflectance).

\4. **Python bindings:** TensorRay provides fine-grained Python bindings using `pybind11 <https://github.com/pybind/pybind11/>`_ and can be easily integrated into Python-based scripts and learning pipelines.


About
--------------------

This project was created at NVIDIA by Cheng Zhang and Edward Liu.

Significant features and/or improvements to the code were contributed by Lifan Wu, Kai Yan, and Shuang Zhao.
