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

#include "Records.h"

namespace EDX
{
	namespace TensorRay
	{
		void PositionSample::Append(const PositionSample& record)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mNumSample == 0)
			{
				*this = record;
			}
			else
			{
				mNumSample = mNumSample + record.mNumSample;
				J = Concat(J, record.J, 0);
				n = Concat(n, record.n, 0);
				p = Concat(p, record.p, 0);
				pdf = Concat(pdf, record.pdf, 0);
				lightId = Concat(lightId, record.lightId, 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
			return;
		}

		PositionSample PositionSample::GetMaskedCopy(const IndexMask& mask) const
		{
			PositionSample ret;
			ret.mNumSample = mask.sum;
			ret.lightId = Mask(lightId, mask, 0);
			ret.J = Mask(J, mask, 0);
			ret.n = Mask(n, mask, 0);
			ret.p = Mask(p, mask, 0);
			ret.pdf = Mask(pdf, mask, 0);
			return ret;
		}

		void PositionSample::Eval()
		{
			lightId = Tensorui(lightId);
			J = Tensorf(J);
			n = Tensorf(n);
			p = Tensorf(p);
			pdf = Tensorf(pdf);
		}

		void Intersection::Eval()
		{
			if (mTriangleId)
			{
				mBsdfId = Tensori(mBsdfId);
				mTriangleId = Tensori(mTriangleId);
				mEmitterId = Tensori(mEmitterId);
				mBaryU = Tensorf(mBaryU);
				mBaryV = Tensorf(mBaryV);
				mTHit = Tensorf(mTHit);

				mPosition = Tensorf(mPosition);
				mNormal = Tensorf(mNormal);
				mGeoNormal = Tensorf(mGeoNormal);
				mTexcoord = Tensorf(mTexcoord);
				mTangent = Tensorf(mTangent);
				mBitangent = Tensorf(mBitangent);
				mJ = Tensorf(mJ);
			}
		}

		void Intersection::Append(const Intersection& its)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if ( !mTriangleId )
			{
				*this = its;
			}
			else
			{
				mBsdfId = Concat(mBsdfId, its.mBsdfId, 0);
				mTriangleId = Concat(mTriangleId, its.mTriangleId, 0);
				mEmitterId = Concat(mEmitterId, its.mEmitterId, 0);
				mBaryU = Concat(mBaryU, its.mBaryU, 0);
				mBaryV = Concat(mBaryV, its.mBaryV, 0);
				mTHit = Concat(mTHit, its.mTHit, 0);
				
				mPosition = Concat(mPosition, its.mPosition, 0);
				mNormal = Concat(mNormal, its.mNormal, 0);
				mGeoNormal = Concat(mGeoNormal, its.mGeoNormal, 0);
				mTexcoord = Concat(mTexcoord, its.mTexcoord, 0);
				mTangent = Concat(mTangent, its.mTangent, 0);
				mBitangent = Concat(mBitangent, its.mBitangent, 0);
				mJ = Concat(mJ, its.mJ, 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
			return;
		}

		Intersection Intersection::GetMaskedCopy(const IndexMask& mask, bool copyAll) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			Intersection ret;

			ret.mBsdfId = Mask(mBsdfId, mask, 0);
			ret.mTriangleId = Mask(mTriangleId, mask, 0);
			ret.mEmitterId = Mask(mEmitterId, mask, 0);
			ret.mBaryU = Mask(mBaryU, mask, 0);
			ret.mBaryV = Mask(mBaryV, mask, 0);
			ret.mTHit = Mask(mTHit, mask, 0);

			if (copyAll)
			{
				ret.mPosition = Mask(mPosition, mask, 0);
				ret.mNormal = Mask(mNormal, mask, 0);
				ret.mGeoNormal = Mask(mGeoNormal, mask, 0);
				ret.mTexcoord = Mask(mTexcoord, mask, 0);
				ret.mTangent = Mask(mTangent, mask, 0);
				ret.mBitangent = Mask(mBitangent, mask, 0);
				ret.mJ = Mask(mJ, mask, 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
			return ret;
		}

		Expr Intersection::LocalToWorld(const Expr& vec) const
		{
			Assert(vec->GetShape().VectorSize() == 3);
			auto x = X(vec);
			auto y = Y(vec);
			auto z = Z(vec);
			return x * mTangent + y * mBitangent + z * mNormal;
		}

		Expr Intersection::WorldToLocal(const Expr& vec) const
		{
			Assert(vec->GetShape().VectorSize() == 3);
			auto x = VectorDot(vec, mTangent);
			auto y = VectorDot(vec, mBitangent);
			auto z = VectorDot(vec, mNormal);

			return MakeVector3(x, y, z);
		}
	}
}