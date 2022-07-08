cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

project(TensorRay VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DNDEBUG)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Wall -fPIC")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

if (WIN32)
    set(PYBIND11_ROOT ${CMAKE_SOURCE_DIR}/ext_win64/pybind11)
    set(PYTHON_ROOT D:/Anaconda)
else()
    if (NOT PYTHON_INCLUDE_PATH)
        message(FATAL_ERROR "PYTHON_INCLUDE_PATH not set!")
    endif()
endif()

set(USE_PROFILING 0)
add_compile_definitions(USE_PROFILING=${USE_PROFILING})

set(TENSOR_RAY_CUDA_FILE TensorRay)
add_compile_definitions(TENSOR_RAY_CUDA_FILE="${TENSOR_RAY_CUDA_FILE}")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING
       "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE
       "Generate dependencies during configure time instead of only during build time." OFF)

find_package(CUDA 11.0 REQUIRED)

if (USE_PROFILING EQUAL 1)
    find_package(NvToolExt REQUIRED)
endif()

mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx")


if (WIN32)
    set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/ext_win64/optix" CACHE PATH "Path to OptiX installed location.")
else()
    set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/ext/optix" CACHE PATH "Path to OptiX installed location.")
endif()
find_package(OptiX 7.4.0 REQUIRED)

set(LibTensorRay ${CMAKE_SOURCE_DIR}/LibTensorRay)

if (USE_PROFILING)
    include_directories(${NvToolExt_INCLUDE_DIR})
    link_libraries(${NvToolExt_LIBRARIES})
endif()

set(LibEDX ${LibTensorRay}/EDXUtil)
set(LibTRRenderer ${LibTensorRay}/Renderer)
set(LibTensor ${LibTensorRay}/Tensor)
set(LibExamples ${LibTensorRay}/Examples)
set(LibTest ${LibTensorRay}/Test)


if (WIN32)
    include_directories(
        ${LibTensorRay}
        ${LibEDX}
        ${OptiX_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${PYTHON_ROOT}/include
        ${PYBIND11_ROOT}/include
    )
    link_directories(
        ${PYTHON_ROOT}/libs
    )
else()
    include_directories(
        "include/"
        "${OptiX_INCLUDE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_BINARY_DIR}/include"
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_PATH}
    )
endif()

function(build_ptx target_name_base target_name_var)
    set(target_name ${target_name_base})
    set(${target_name_var} ${target_name} PARENT_SCOPE)

    source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")

    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
    CUDA_WRAP_SRCS(${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options})

    add_executable(${target_name}
        ${source_files}
        ${generated_files}
        ${cmake_options}
    )
    set_target_properties(${target_name} PROPERTIES
        COMPILE_DEFINITIONS OPTIX_SAMPLE_NAME_DEFINE=${target_name}
    )
endfunction()

build_ptx(ptx target_name
    ${LibTRRenderer}/ptx.cu
    ${LibTRRenderer}/ptx.cpp
    OPTIONS -rdc true
)

set(CORE_SOURCE_FILES
    ${LibTensor}/Tensor.cpp
    ${LibTensor}/Tensor.cu

    ${LibEDX}/Graphics/BaseCamera.cpp
    ${LibEDX}/Graphics/Color.cpp
    ${LibEDX}/Graphics/ObjMesh.cpp
    ${LibEDX}/Graphics/Texture.cpp

    ${LibEDX}/Math/Matrix.cpp
    ${LibEDX}/Math/FFT.cpp

    ${LibEDX}/Windows/Bitmap.cpp
    ${LibEDX}/Windows/Debug.cpp
    ${LibEDX}/Windows/stb_image.c

    ${LibTRRenderer}/BSDF.h
    ${LibTRRenderer}/BSDF.cpp
    ${LibTRRenderer}/Camera.h
    ${LibTRRenderer}/Camera.cpp
    ${LibTRRenderer}/Config.h
    ${LibTRRenderer}/Diffuse.h
    ${LibTRRenderer}/Diffuse.cpp
    ${LibTRRenderer}/Distribution.h
    ${LibTRRenderer}/Distribution.cpp
    ${LibTRRenderer}/Distribution.cu
    ${LibTRRenderer}/Edge.h
    ${LibTRRenderer}/Edge.cpp
    ${LibTRRenderer}/Integrator.h
    ${LibTRRenderer}/Integrator.cpp
    ${LibTRRenderer}/Light.h
    ${LibTRRenderer}/Light.cpp
    ${LibTRRenderer}/Microfacet.h
    ${LibTRRenderer}/Microfacet.cpp
    ${LibTRRenderer}/miniz.h
    ${LibTRRenderer}/miniz.cpp
    ${LibTRRenderer}/Optix.h
    ${LibTRRenderer}/Optix.cpp
    ${LibTRRenderer}/PathTracer.h
    ${LibTRRenderer}/PathTracer.cpp
    ${LibTRRenderer}/ParticleTracer.h
    ${LibTRRenderer}/ParticleTracer.cpp
    ${LibTRRenderer}/PathTracerDebug.h
    ${LibTRRenderer}/PathTracerDebug.cpp
    ${LibTRRenderer}/Primitive.h
    ${LibTRRenderer}/Primitive.cpp
    ${LibTRRenderer}/pybind_utils.h
    ${LibTRRenderer}/pybind_utils.cpp
    ${LibTRRenderer}/Ray.h
    ${LibTRRenderer}/Ray.cpp
    ${LibTRRenderer}/Records.h
    ${LibTRRenderer}/Records.cpp
    ${LibTRRenderer}/Roughconductor.h
    ${LibTRRenderer}/Roughconductor.cpp
    ${LibTRRenderer}/Scene.h
    ${LibTRRenderer}/Scene.cpp
    ${LibTRRenderer}/SceneLoader.h
    ${LibTRRenderer}/SceneLoader.cpp
    ${LibTRRenderer}/Utils.h
    ${LibTRRenderer}/Utils.cpp
    ${LibTRRenderer}/Boundary.h
    ${LibTRRenderer}/Boundary.cpp
    ${LibTRRenderer}/BoundaryDirect.cpp
    ${LibTRRenderer}/BoundaryIndirect.cpp
    ${LibTRRenderer}/BoundaryPrimary.cpp
    ${LibTRRenderer}/BoundaryPixel.cpp

    ${LibTRRenderer}/NEE.cu
    ${LibTRRenderer}/NEE.cuh

    ${LibTRRenderer}/AQ_distrb.h

    ${LibTest}/TestCuda.cu
    ${LibTest}/Test.cpp

    ${LibExamples}/Validations/ValidationExamples.cpp
    ${LibExamples}/Validations/CboxExample.cpp

    ${LibExamples}/InvRendering/InvRenderingExamples.cpp
    ${LibExamples}/InvRendering/BunnyShadow.cpp
    ${LibExamples}/InvRendering/BunnyTexture.cpp
)

set(TENSOR_RAY_SOURCE_FILES
    TensorRay.cpp
)

set(DEBUG_SOURCE_FILES
    TensorRay_debug.cpp
    TensorVis.natvis
)

message(${CUDA_CUBLAS_LIBRARIES})
message(${CUDA_curand_LIBRARY})
message(${CUDA_LIBRARIES})
message(${CUDA_TOOLKIT_ROOT_DIR})
message(${CUDA_INCLUDE_DIRS})

set(cuda_lib C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5/lib/x64/cuda.lib) # WARN: FIX THIS!!!
add_definitions(-D_SOURCE_DIR="${CMAKE_SOURCE_DIR}/")
if (WIN32)
    add_subdirectory(${PYBIND11_ROOT})
    cuda_add_library(cu_library STATIC ${CORE_SOURCE_FILES})
    target_compile_options(cu_library PRIVATE /MP)
    target_link_libraries(cu_library ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${cuda_lib} )

    pybind11_add_module(TensorRay ${TENSOR_RAY_SOURCE_FILES})
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(TensorRay PRIVATE /O2 /wd4251 /MP)
    else()
        target_compile_options(TensorRay PRIVATE /MP)
    endif()
    target_link_libraries(TensorRay PRIVATE cu_library)

    add_executable(Debug ${DEBUG_SOURCE_FILES})
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(Debug PRIVATE /O2 /wd4251 /MP)
    else()
        target_compile_options(Debug PRIVATE /MP)
    endif()

    target_link_libraries(Debug PRIVATE cu_library)
else()
    cuda_add_library(TensorRay MODULE ${TENSOR_RAY_CUDA_FILE})
endif()


# target build
target_compile_definitions(cu_library PRIVATE PTX_OUTPUT_DIR="${CUDA_GENERATED_OUTPUT_DIR}")


set_target_properties(TensorRay PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(TensorRay PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_property(TARGET TensorRay PROPERTY CXX_STANDARD 17)
set_target_properties(TensorRay PROPERTIES PREFIX "")

set_target_properties(Debug PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(Debug PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_property(TARGET Debug PROPERTY CXX_STANDARD 17)
set_target_properties(Debug PROPERTIES PREFIX "")
