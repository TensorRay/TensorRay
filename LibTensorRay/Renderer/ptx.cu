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

#include <optix.h>

#include <Tensor/TensorJitKernel.cuh>

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

struct HitData
{
    int bsdfId;
};

extern "C" {
    __constant__ LaunchParams params;
}


extern "C" __global__ void __raygen__visibility_raygen()
{
    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    const int linearIdx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    float3 rayOrg = make_float3(
        params.rayOrg.X(linearIdx, params.rayOrg.mParams),
        params.rayOrg.Y(linearIdx, params.rayOrg.mParams),
        params.rayOrg.Z(linearIdx, params.rayOrg.mParams));

    float3 rayDir = make_float3(
        params.rayDir.X(linearIdx, params.rayDir.mParams),
        params.rayDir.Y(linearIdx, params.rayDir.mParams),
        params.rayDir.Z(linearIdx, params.rayDir.mParams));

    float tmin = params.tMin[linearIdx];
    float tmax = params.tMax[linearIdx];


    unsigned int bsdfId, triangleId, u, v, t;
    optixTrace(params.bvh, rayOrg, rayDir, tmin, tmax, 0.0f, OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE, 0u/*RAY_TYPE*/, 1u/*RAY_TYPE_COUNT*/, 0u/*RAY_TYPE*/, bsdfId, triangleId, u, v, t);

    params.bsdfId[linearIdx] = bsdfId;
    params.triangleId[linearIdx] = triangleId;
    params.tHit[linearIdx] = __int_as_float(t);
    params.baryU[linearIdx] = __int_as_float(u);
    params.baryV[linearIdx] = __int_as_float(v);
}

extern "C" __global__ void __miss__visibility_miss()
{
    optixSetPayload_0(static_cast<unsigned int>(-1));
    optixSetPayload_1(static_cast<unsigned int>(-1));
    optixSetPayload_2(__float_as_int(0.0f));
    optixSetPayload_3(__float_as_int(0.0f));
    optixSetPayload_4(__float_as_int(-1.0f));
}

extern "C" __global__ void __closesthit__visibility_hit()
{
    HitData* hitData = (HitData*)optixGetSbtDataPointer();
    const float2   barys = optixGetTriangleBarycentrics();
    const float tHit = optixGetRayTmax();

    // Set the hit data
    optixSetPayload_0(hitData->bsdfId);
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_int(barys.x));
    optixSetPayload_3(__float_as_int(barys.y));
    optixSetPayload_4(__float_as_int(tHit));
}

