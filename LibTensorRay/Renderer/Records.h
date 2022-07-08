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
#include "Utils.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		struct PositionSample
		{
			PositionSample() :mNumSample(0) {}
			int mNumSample;
			Expr p;			// Sampled position
			Expr n;			// Surface normal at the sample
			Expr pdf;		// Probability density at the sample (1D tensor)
			Expr J;			// (2D tensor)
			Expr lightId;
			void Append(const PositionSample& record);
			PositionSample GetMaskedCopy(const IndexMask& mask) const;
			void Eval();
		};

		struct SensorDirectSample
		{
			Expr q;
			Expr isValid;
			Expr pixelIdx;
			Expr sensorVal;
		};

		class Intersection
		{
		public:
			// computed in RayIntersect (no AD)
			Expr mBaryU;
			Expr mBaryV;
			Expr mTHit;
			Expr mTriangleId;
			Expr mBsdfId;
			Expr mEmitterId;
			// Requires AD (computed in post processing)
			Expr mPosition;
			Expr mNormal;
			Expr mGeoNormal;
			Expr mTexcoord;
			Expr mTangent;
			Expr mBitangent;
			Expr mJ;

			Intersection GetMaskedCopy(const IndexMask& mask, bool copyAll = false) const;
			void Append(const Intersection& its);
			Expr LocalToWorld(const Expr& vec) const;
			Expr WorldToLocal(const Expr& vec) const;
			void Eval();
		};
	}
}