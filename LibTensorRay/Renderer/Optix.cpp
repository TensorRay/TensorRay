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

#include "Optix.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <iterator>
#include <map>

#include <iomanip>
#include <fstream>
#include <iostream>
#include <string>

namespace EDX
{
	namespace TensorRay
	{
        void OptixState::CreateModule()
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(nullptr));

            CUcontext          cuCtx = nullptr;  // zero means take the current context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            //options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void* /*cbdata */)
            //{
            //    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
            //    << message << "\n";
            //};
            //options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

            char   log[2048];  // For error reporting from OptiX creation functions
            size_t sizeof_log = sizeof(log);

            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipeline_compile_options.numPayloadValues = 5;
            pipeline_compile_options.numAttributeValues = 2;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            static const char* ptx_directories[] = {
                PTX_OUTPUT_DIR
            };

            std::string ptx_file;
            for( const char* directory : ptx_directories )
            {
                if( directory )
                {
                    ptx_file = directory;
                    ptx_file += '/';
                    ptx_file += "ptx";
                    ptx_file += "_generated_";
                    ptx_file += "ptx.cu";
                    ptx_file += ".ptx";
                    std::cout << ptx_file << std::endl;
                }
            }

            std::cout << "load ptx: " << ptx_file << std::endl;
            ifstream fileStream(ptx_file);
            
            string ptx((std::istreambuf_iterator<char>(fileStream)),
                std::istreambuf_iterator<char>());

            OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options,
                ptx.c_str(), ptx.size(), log, &sizeof_log, &ptx_module));
        }

        void OptixState::CreateProgramGroups()
        {
            char   log[2048];  // For error reporting from OptiX creation functions
            size_t sizeof_log = sizeof(log);

            OptixProgramGroupOptions program_group_options = {};

            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = ptx_module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__visibility_raygen";

            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &raygen_prog_group_desc,
                1,  // num program groups
                &program_group_options, log, &sizeof_log, &raygen_prog_group));

            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module = ptx_module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__visibility_miss";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
                1,  // num program groups
                &program_group_options, log, &sizeof_log, &miss_prog_group));


            OptixProgramGroupDesc hit_prog_group_desc = {};
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = ptx_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__visibility_hit";
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &hit_prog_group_desc,
                1,  // num program groups
                &program_group_options, log, &sizeof_log, &hit_prog_group));
        }

        void OptixState::CreatePipelines()
        {
            char   log[2048];  // For error reporting from OptiX creation functions
            size_t sizeof_log = sizeof(log);

            const uint32_t    max_trace_depth = 1;
            OptixProgramGroup program_groups[3] = { raygen_prog_group, miss_prog_group, hit_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

            OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options,
                program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
                &sizeof_log, &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
            {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state, &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state, continuation_stack_size,
                2  // maxTraversableDepth
            ));
        }

        void OptixState::Release()
        {
            if (pipeline) {
                OPTIX_CHECK(optixPipelineDestroy(pipeline));
                pipeline = 0;
            }
            if (raygen_prog_group) {
                OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
                raygen_prog_group = 0;
            }
            if (miss_prog_group) {
                OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
                miss_prog_group = 0;
            }
            if (hit_prog_group) {
                OPTIX_CHECK(optixProgramGroupDestroy(hit_prog_group));
                hit_prog_group = 0;
            }
            if (ptx_module) {
                OPTIX_CHECK(optixModuleDestroy(ptx_module));
                ptx_module = 0;
            }
            if (context) {
                OPTIX_CHECK(optixDeviceContextDestroy(context));
                context = 0;
            }

            if (sbt.raygenRecord) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
                sbt.raygenRecord = 0;
            }
            if (sbt.missRecordBase) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
                sbt.missRecordBase = 0;
            }
            if (sbt.hitgroupRecordBase) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
                sbt.hitgroupRecordBase = 0;
            }
            if (d_gas_output_buffer) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
                d_gas_output_buffer = 0;
                gas_output_buffer_size = 0;
            }
            if (d_ias_output_buffer) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_output_buffer)));
                d_ias_output_buffer = 0;
                ias_output_buffer_size = 0;
            }
            if (d_temp_buffer) {
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
                d_temp_buffer = 0;
                temp_buffer_size = 0;
            }

        }

        struct LaunchParams
        {
            OptixTraversableHandle bvh;

            TensorJit<float> rayOrg;
            TensorJit<float> rayDir;
            TensorJit<float> tMin;
            TensorJit<float> tMax;

            TensorJit<int> bsdfId;
            TensorJit<int> triangleId;
            TensorJit<float> baryU;
            TensorJit<float> baryV;
            TensorJit<float> tHit;
        };

        void RayIntersectOptix(const Ray& ray, Intersection& isect)
        {
            Tensorf rayOrg = ray.mOrg;
            Tensorf rayDir = ray.mDir;
            Tensorf tMin = ray.mMin;
            Tensorf tMax = ray.mMax;

            LaunchParams hostParams;
            hostParams.bvh = Optix::GetHandle().tlas;
            hostParams.rayOrg = rayOrg.ToJit();
            hostParams.rayDir = rayDir.ToJit();
            hostParams.tMin = tMin.ToJit();
            hostParams.tMax = tMax.ToJit();

            Tensori bsdfId;
            Tensori triangleId;
            Tensorf baryU;
            Tensorf baryV;
            Tensorf tHit;

            bsdfId.Resize(ray.mNumRays);
            triangleId.Resize(ray.mNumRays);
            baryU.Resize(ray.mNumRays);
            baryV.Resize(ray.mNumRays);
            tHit.Resize(ray.mNumRays);

            hostParams.bsdfId = bsdfId.ToJit();
            hostParams.triangleId = triangleId.ToJit();
            hostParams.baryU = baryU.ToJit();
            hostParams.baryV = baryV.ToJit();
            hostParams.tHit = tHit.ToJit();

            LaunchParams* deviceParams = 0;
            CuAllocator::GetHandle().DeviceAllocate(reinterpret_cast<void**>(&deviceParams), sizeof(LaunchParams));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(deviceParams), &hostParams, sizeof(LaunchParams),
                cudaMemcpyHostToDevice));

            OPTIX_CHECK(optixLaunch(Optix::GetHandle().pipeline, 0, reinterpret_cast<CUdeviceptr>(deviceParams), sizeof(LaunchParams),
                &Optix::GetHandle().sbt, ray.mNumRays, 1, 1));

            CuAllocator::GetHandle().DeviceFree(reinterpret_cast<void*>(deviceParams));

            bsdfId.CopyToHost();
            triangleId.CopyToHost();
            baryU.CopyToHost();
            baryV.CopyToHost();
            tHit.CopyToHost();

            isect.mBsdfId = bsdfId;
            isect.mTriangleId = triangleId;
            isect.mBaryU = baryU;
            isect.mBaryV = baryV;
            isect.mTHit = tHit;
        }
	}
}