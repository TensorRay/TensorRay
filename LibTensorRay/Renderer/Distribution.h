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
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		// Distribution function
		__global__ void SampleDiscrete1D(const TensorJit<float> cdf, const TensorJit<float> pdf, const TensorJit<float> u, const float integral, TensorJit<int> offset, TensorJit<float> sampledPdf);
		__global__ void SampleContinuous1D(const TensorJit<float> cdf, const TensorJit<float> pdf, const TensorJit<float> u, const float integral, TensorJit<int> offset, TensorJit<float> sampledVal, TensorJit<float> sampledPdf);
		__global__ void SampleContinuous2D(const TensorJit<float> condPdf, const TensorJit<float> condCdf,
			const TensorJit<float> marginalPdf, const TensorJit<float> marginalCdf,
			const float marginalIntegral, const TensorJit<float> uv,
			TensorJit<float> sampledU, TensorJit<float> sampledV, TensorJit<float> sampledPdf);
		__global__ void Pdf2D(const TensorJit<float> condPdf, const TensorJit<float> condCdf,
			const TensorJit<float> marginalPdf, const TensorJit<float> marginalCdf,
			const float marginalIntegral,
			const TensorJit<float> u, const TensorJit<float> v,
			TensorJit<float> sampledPdf);

		class Distribution1D
		{
		public:
			Tensorf mPDF;
			Tensorf mCDF;
			int mSize = INDEX_NONE;
			float mIntegralVal;
			friend class Distribution2D;

			Distribution1D() = default;
			Distribution1D(const Tensorf& func) { SetFunction(func); }
			void SetFunction(const Tensorf& func);
			void SampleContinuous(const Tensorf& u, Tensori* pOffset, Tensorf* pSampled, Tensorf* pPdf) const;
			Expr Distribution1D::ReuseSample(const Tensorf& rnd, const Tensori& index) const;
			void SampleDiscrete(const Tensorf& u, Tensori* pOffset, Tensorf* pPdf) const;
			float GetIntegral() const { return mIntegralVal; }
			float GetPdf(int i) const { return mPDF.Get(i) / (mIntegralVal * mPDF.LinearSize()); }
		};

		class Distribution2D
		{
		private:
			Tensorf mPdf;
			Tensorf mCdf;
			Tensorf mMarginalPdf;
			Tensorf mMarginalCdf;
			float mMarginalIntegral;

		public:
			Distribution2D(const Tensorf& func);
			void SampleContinuous(const Tensorf& uv, Tensorf* pSampledU, Tensorf* pSampledV, Tensorf* pPdf) const;
			Tensorf Pdf(const Tensorf& u, const Tensorf& v) const;
		};
	}
}
