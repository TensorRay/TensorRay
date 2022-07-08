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
#include "Graphics/BaseCamera.h"
#include "Utils.h"
#include "Records.h"
#include "Ray.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		class ReconstructionFilter
		{
		public:
			virtual Expr Eval(const Expr& x) const = 0;
			virtual Expr Sample(const Expr& rnd) const = 0;
			virtual Expr Sample(const Expr& rnd, Expr& pdf) const
			{
				auto ret = Sample(rnd);
				pdf = Eval(ret);
				return ret;
			}
		};

		class TentFilter : public ReconstructionFilter
		{
		public:
			TentFilter() : mPadding(0.5f - EPSILON) {};
			TentFilter(float padding) : mPadding(padding) {};
			Expr Eval(const Expr& x) const;
			Expr Sample(const Expr& rnd) const;
			float mPadding;
		};

		class BoxFilter : public ReconstructionFilter
		{
		public:
			BoxFilter() {};
			Expr Eval(const Expr& x) const;
			Expr Sample(const Expr& rnd) const;
			Expr Sample(const Expr& rnd, Expr& pdf) const;
		};

		class Camera : public EDX::Camera
		{
		public:
			Tensorf mPosTensor;
			Tensorf mDirTensor;
			Tensorf mViewTensor;
			Tensorf mViewInvTensor;

			Tensorf mProjTensor;

			Tensorf mScreenToRasterTensor;
			Tensorf mRasterToCameraTensor;
			Tensorf mCameraToRasterTensor;
			Tensorf mRasterToWorldTensor;
			Tensorf mWorldToRasterTensor;
			Tensorf mWorldToSampleTensor;
			Tensorf mSampleToCameraTensor;
			
			float mImagePlaneLengthX, mImagePlaneLengthY;
			float mInvArea;
			int mResX, mResY;
			float mFilterPad;
			bool mRequireGrad;
			unique_ptr<ReconstructionFilter> mFilter;

		public:
			Camera() { }

			Camera(const Vector3& pos, const Vector3& tar, const Vector3& up, int resX, int resY, float fov)
			{
				Init(pos, tar, up, resX, resY, fov);
			}

			void Update(const Camera& c)
			{
				Init(c.mPos, c.mTarget, c.mUp, c.mResX, c.mResY, c.mFOV);
			}

			void Init(const Vector3& pos,
				const Vector3& tar,
				const Vector3& up,
				const int resX,
				const int resY,
				const float FOV = 35.0f,
				const float nearClip = 1.0f,
				const float farClip = 1000.0f,
				const float filterPad = 0.5f - EPSILON);			

			void Resize(int width, int height);
			void GenerateRay(Ray& ray) const;
			void GenerateRayPair(Ray& rayPrimal, Ray& rayDual) const;
			void GenerateAntitheticRays(Ray& rays, int batchSize, int antitheticSpp) const;
			Expr EvalFilter(const Expr& pixelId, const Intersection& isect) const;
			bool requireGrad() const { return mRequireGrad; }

			void GenerateBoundaryRays(const Tensorf& samples, const Tensorf& normal, Ray& rayP, Ray& rayN) const;
			void GenerateBoundaryRays(const SensorDirectSample& sds, Ray& ray) const;
			SensorDirectSample sampleDirect(const Expr& p) const;
			Tensorf WriteToImage(const Expr& rayContrib, const Expr& pixelCoor) const;

			void GeneratePixelBoundarySamples(const Tensorf& rnd, Tensorf& p0, Tensorf& edge, Tensorf& edge2, Tensorf& pdf) const;
		};
	}
}