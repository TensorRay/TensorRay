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

#include "Ray.h"
#include "Utils.h"

namespace EDX
{
	namespace TensorRay
	{
		Ray::Ray(const Expr& org, const Expr& dir)
		{
			Assert(org->GetShape().VectorSize() > 1 && dir->GetShape().VectorSize() > 1);
			Assert(org->GetShape()[0] == dir->GetShape()[0]);
			mNumRays = org->GetShape()[0];
			mOrg = org;
			mDir = dir;
			mMin = Ones(mNumRays) * Scalar(SHADOW_EPSILON);
			mMax = Ones(mNumRays) * Scalar(1e32f);
		}

		void Ray::Eval()
		{
			if (mNumRays > 0)
			{
				mOrg = Tensorf(mOrg);
				mDir = Tensorf(mDir);
				mThroughput = Tensorf(mThroughput);
				mPrevPdf = Tensorf(mPrevPdf);
				mSpecular = Tensorf(mSpecular);
				mMin = Tensorf(mMin);
				mMax = Tensorf(mMax);
				mPixelIdx = Tensori(mPixelIdx);
				mRayIdx = Tensori(mRayIdx);
			}
		}

		void Ray::Append(const Ray& ray)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mNumRays == 0)
			{
				*this = ray;
			}
			else
			{
				mNumRays = mNumRays + ray.mNumRays;
				mOrg = Concat(mOrg, ray.mOrg, 0);
				mDir = Concat(mDir, ray.mDir, 0);
				mThroughput = Concat(mThroughput, ray.mThroughput, 0);
				mPrevPdf = Concat(mPrevPdf, ray.mPrevPdf, 0);
				mSpecular = Concat(mSpecular, ray.mSpecular, 0);
				mMin = Concat(mMin, ray.mMin, 0);
				mMax = Concat(mMax, ray.mMax, 0);
				mPixelIdx = Concat(mPixelIdx, ray.mPixelIdx, 0);
				mRayIdx = Concat(mRayIdx, ray.mRayIdx, 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
			return;
		}

		Ray Ray::GetMaskedCopy(const IndexMask& mask, bool complete) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			Ray ret;
			ret.mNumRays = mask.sum;
			ret.mOrg = Mask(mOrg, mask, 0);
			ret.mDir = Mask(mDir, mask, 0);
			ret.mMin = Mask(mMin, mask, 0);
			ret.mMax = Mask(mMax, mask, 0);
			if (complete)
			{
				ret.mPixelIdx = Mask(mPixelIdx, mask, 0);
				ret.mRayIdx = Mask(mRayIdx, mask, 0);
				ret.mPrevPdf = Mask(mPrevPdf, mask, 0);
				ret.mSpecular = Mask(mSpecular, mask, 0);
				ret.mThroughput = Mask(mThroughput, mask, 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
			return ret;
		}
	}
}