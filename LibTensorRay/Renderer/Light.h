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
#include "../Tensor/Tensor.h"
#include "Primitive.h"
#include "Records.h"
#include "Utils.h"
#include "Ray.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		class Light
		{
		public:
			virtual ~Light() { }

			// TODO! Unified light interface!!
			virtual Expr Eval(const Intersection& isect, const Expr& wi) const
			{
				return Zeros(1);
			}
			virtual Expr Eval(const PositionSample& record, const Expr& wi) const
			{
				return Zeros(1);
			}
			virtual void Sample(const Expr& samples, PositionSample& lightP) const
			{
			}
			virtual Expr Pdf(const Expr& refP, const Intersection& isect) const
			{
				return Zeros(1);
			}
			virtual void Pdf(const Tensorf& refP, const Tensorf& samples, const Intersection& isect, Tensorf& pdf) const
			{
			}

			virtual void Eval(const Intersection& isect, const Tensorf& samples, Ray& rays, Tensorf& intens, Tensorf& pdf) const
			{
			}
			virtual void Emit(const Tensorf& dir, Tensorf& emitted) const
			{
			}
			virtual Tensorf Pdf(const Tensorf& dir) const
			{
				return Zeros(1);
			}
		};

		class PointLight : public Light
		{
		public:
			Tensorf mPos;
			Tensorf mIntensity;

		public:
			PointLight(const float intens, const Tensorf& pos);


			void Eval(const Intersection& isect, const Tensorf& samples, Ray& rays, Tensorf& intens, Tensorf& pdf) const;
			void Emit(const Tensorf& dir, Tensorf& emitted) const;
			Tensorf Pdf(const Tensorf& dir) const;
		};

		class DirectionalLight : public Light
		{
		public:
			Tensorf mDir;
			Tensorf mIntensity;
			float mConeDegree;

			Tensorf mTangent;
			Tensorf mBitangent;

		public:
			DirectionalLight(const float intens, const Tensorf& dir, const float coneDeg);
			void Eval(const Intersection& isect, const Tensorf& samples, Ray& rays, Tensorf& intens, Tensorf& pdf) const;
			void Emit(const Tensorf& dir, Tensorf& emitted) const;
			Tensorf Pdf(const Tensorf& dir) const;
		};


		struct lightInfoCPU
		{
			vector<int> mShapeId;
			vector<Vector3> mIntensity;
		};

		class AreaLight : public Light
		{
		public:
			lightInfoCPU mInfo;
			Tensorf mIntensity;										// light_id -> intensity
			// For sampling
			std::unique_ptr<Distribution1D> mpLightDistrb;			// intensity * tri.area
			Tensorui mLightIdBuffer;								// tri_id -> light_id
			Tensorf mPositionBuffer;
			Tensorui mIndexPosBuffer;
			Tensorf mFaceNormalBuffer;
			Tensorf mTriangleAreaBuffer;
			Tensori mTriIdToEmitTriIdBuffer;                        // help to map its.triangleId (global) to emitterTriangleId
			float mInvTotArea;

			AreaLight(const Vector3& intens, int shapeId);
			void Append(const Vector3& intens, int shapeId);

			Expr Eval(const Intersection& isect, const Expr& wi) const;			// Direct lighting from path tracing
			Expr Eval(const PositionSample& record, const Expr& wi) const;		// Direct lighting from light samples
			void Sample(const Expr& samples, PositionSample& lightP) const;
			Expr Pdf(const Expr& refP, const Intersection& isect) const;
		};

		//class EnvironmentLight : public Light
		//{
		//private:
		//	Tensorf mMap;
		//	unique_ptr<Distribution2D> mDist;
		//	int mWidth, mHeight, mChannels;

		//public:
		//	EnvironmentLight(const char* path);
		//	void Eval(const Intersection& isect, const Tensorf& samples, Ray& rays, Tensorf& intens, Tensorf& pdf) const;
		//	void Emit(const Tensorf& dir, Tensorf& emitted) const;
		//	Tensorf Pdf(const Tensorf& dir) const;
		//};

	}
}