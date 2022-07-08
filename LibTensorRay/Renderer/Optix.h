/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <optix.h>
#include "device_launch_parameters.h"
#include "../Tensor/Tensor.h"
#include "Ray.h"
#include "Records.h"

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw std::runtime_error(ss.str().c_str());                        \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed with error: "           \
               << optixGetErrorName(res)                                       \
               << ", " __FILE__ << ":" << __LINE__ << "\n"                     \
               << "Log:\n" << log                                              \
               << ( sizeof_log > sizeof( log ) ? "<TRUNCATED>" : "" )          \
               << "\n";                                                        \
            throw std::runtime_error(ss.str().c_str());                        \
        }                                                                      \
    } while( 0 )

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call '" << #call << "' failed with error: "            \
               << cudaGetErrorString( error )                                  \
               << ", " __FILE__ << ":" << __LINE__ << "\n";                    \
            throw std::runtime_error(ss.str().c_str());                        \
        }                                                                      \
    } while( 0 )


using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		struct OptixState
		{
			OptixDeviceContext          context = 0;

			OptixPipelineCompileOptions pipeline_compile_options = {};
			OptixModule                 ptx_module = 0;
			OptixPipeline               pipeline = 0;

			OptixProgramGroup           raygen_prog_group = 0;
			OptixProgramGroup           miss_prog_group = 0;
			OptixProgramGroup           hit_prog_group = 0;

			OptixTraversableHandle		blas;
			OptixTraversableHandle		tlas;
			CUdeviceptr					d_gas_output_buffer = 0;  // Triangle AS memory
			size_t						gas_output_buffer_size = 0;
			CUdeviceptr					d_ias_output_buffer = 0;  // Instance AS memory
			size_t						ias_output_buffer_size = 0;
			OptixShaderBindingTable     sbt = {};

			CUdeviceptr					d_temp_buffer = 0;
			size_t						temp_buffer_size = 0;

			void CreateModule();
			void CreateProgramGroups();
			void CreatePipelines();
			void Release();
		};

		// Singleton for storing cublas handle
		class Optix
		{
		public:
			static OptixState& GetHandle()
			{
				static OptixState Handle;
				return Handle;
			}
		public:
			Optix(Optix const&) = delete;
			void operator = (Optix const&) = delete;
		};

		template <typename T>
		struct Record
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};

		struct EmptyData {};
		typedef Record<EmptyData> EmptyRecord;

		struct HitData
		{
			int bsdfId;
		};
		typedef Record<HitData> HitRecord;

		void RayIntersectOptix(const Ray& rays, Intersection& isect);
	}
}