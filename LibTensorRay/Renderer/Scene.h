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

#include "Graphics/BaseCamera.h"
#include "Graphics/ObjMesh.h"
#include "../Tensor/Tensor.h"
#include "Config.h"
#include "Distribution.h"
#include "Light.h"
#include "Camera.h"
#include "BSDF.h"
#include "Primitive.h"
#include "Optix.h"
#include <optix.h>


using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
    namespace TensorRay
    {
        class Scene
        {
        public:
            vector<shared_ptr<Primitive>> mPrims;
            vector<shared_ptr<BSDF>> mBsdfs;
            vector<shared_ptr<Light>> mLights;
            vector<shared_ptr<Camera>> mSensors;

            int mVertexCount;
            int mTexcoorCount;
            int mVertexNormalCount;
            int mTriangleCount;
            int mBSDFCount;
            int mSensorCount;

            int mAreaLightIndex;

            Tensorf mPositionBuffer;
            Tensorf mFaceNormalBuffer;
            Tensorf mVertexNormalBuffer;
            Tensori mEmitterIDBuffer;
            Tensori mTriIdToEmitTriIdBuffer;   // help to map from global triangle ID to emitter triangle ID
            Tensorf mTexcoordBuffer;
            Tensorui mIndexPosBuffer;
            Tensorui mIndexNormalBuffer;
            Tensorui mIndexTexBuffer;
            Tensorb mUseSmoothShadingBuffer;
            Tensorf mTriangleAreaBuffer;
            Tensorui mSbtIndexBuffer;
            Tensorf mPositionBufferT;       // construct Optix BVH

        public:
            Scene(): mVertexCount(0), mTexcoorCount(0), mVertexNormalCount(0), mTriangleCount(0), mBSDFCount(0), mSensorCount(0), mAreaLightIndex(-1)
            {
            }

            ~Scene()
            {
            }

            template <typename T, typename... TArgs>
            void AddPrimitive(TArgs&&... Args) { mPrims.push_back(make_shared<T>(std::forward<TArgs>(Args)...)); }

            template <typename T, typename... TArgs>
            void AddLight(TArgs&&... Args) { mLights.push_back(make_unique<T>(std::forward<TArgs>(Args)...)); }

            template <typename T, typename... TArgs>
            void AddBsdf(TArgs&&... Args) {
                mBsdfs.push_back(make_shared<T>(std::forward<TArgs>(Args)...));
                mBsdfs.back()->mId = mBsdfs.size() - 1;
                mBSDFCount++;
            }

            template <typename T, typename... TArgs>
            void AddSensor(TArgs&&... Args) { mSensors.push_back(make_unique<T>(std::forward<TArgs>(Args)...)); mSensorCount++; }

            void Configure();
            void Update();
            void ExtractGeometry();
            void ConfigureLight();
            void CreateSBT(OptixState& state);
            void ConstructAccelerationStructure(OptixState& state);
            void UpdateAccelerationStructure(OptixState& state);

            void Intersect(const Ray& ray, Intersection& isect) const;
            void IntersectHit(Ray& rays, Intersection& isect) const;
            void PostIntersect(Intersection& isect) const;
            void PostIntersectPrimary(const Ray& rays, Intersection& isect) const;
            void Occluded(Ray& ray, Tensorb& bHits) const;
            void Occluded(Ray& ray, Expr& bHits) const;
            void LoadFromFile(const char* filename);

            const vector<shared_ptr<Primitive>>& GetPrimitives() const { return mPrims; }
            const vector<shared_ptr<Light>>& GetLights() const { return mLights; }
            int GetImageWidth(int i) const { return mSensors[i]->mResX; }
            int GetImageHeight(int i) const { return mSensors[i]->mResY; }
        };
    }
}