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

#include "Distribution.h"

namespace EDX
{
	namespace TensorRay
	{
		__global__ void SampleDiscrete1D(const TensorJit<float> cdf, const TensorJit<float> pdf, const TensorJit<float> u, const float integral, TensorJit<int> offset, TensorJit<float> sampledPdf)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i >= u.LinearSize())
				return;

			int index = Algorithm::UpperBound(cdf.Data(), int(cdf.LinearSize()), u[i], Algorithm::LessEQ<float>());
			index = Math::Clamp(index, 0, cdf.LinearSize() - 1);
			sampledPdf[i] = pdf[index] / (integral * cdf.LinearSize());
			offset[i] = index;
		}

		__global__ void SampleContinuous1D(const TensorJit<float> cdf, const TensorJit<float> pdf, const TensorJit<float> u, const float integral, TensorJit<int> offset, TensorJit<float> sampledVal, TensorJit<float> sampledPdf)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i >= u.LinearSize())
				return;

			int index = Algorithm::UpperBound(cdf.Data(), int(cdf.LinearSize()), u[i], Algorithm::LessEQ<float>());
			index = Math::Clamp(index - 1, 0, cdf.LinearSize() - 1);
			sampledPdf[i] = pdf[index] / integral;
			offset[i] = index;

			float du = (u[i] - cdf[index]) / (cdf[index] - cdf[index - 1] + 1e-4f);
			sampledVal[i] = du;
		}

		__global__ void SampleContinuous2D(
			const TensorJit<float> condPdf, const TensorJit<float> condCdf,
			const TensorJit<float> marginalPdf, const TensorJit<float> marginalCdf,
			const float marginalIntegral, const TensorJit<float> uv,
			TensorJit<float> sampledU, TensorJit<float> sampledV, TensorJit<float> sampledPdf)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i >= uv.mParams.mShape.x1)
				return;

			float v = uv[ShapeJit(1, i)];
			int idxV = Algorithm::UpperBound(marginalCdf.Data(), int(marginalCdf.LinearSize()), v, Algorithm::LessEQ<float>());
			idxV = Math::Clamp(idxV - 1, 0, marginalCdf.LinearSize() - 1);
			float pdfV = marginalPdf[idxV] / (marginalIntegral);
			float baseV = (idxV - 1 > 0) ? marginalCdf[idxV - 1] : 0.0f;
			float dv = (v - marginalCdf[idxV]) / (marginalCdf[idxV] - baseV + 1e-4f);
			sampledV[i] = (idxV + dv) / float(marginalCdf.LinearSize());

			float u = uv[ShapeJit(0, i)];
			int idxU = Algorithm::UpperBound(condCdf.Data() + idxV * condCdf.mParams.mShape.x1, condCdf.mParams.mShape.x1, u, Algorithm::LessEQ<float>());
			idxU = Math::Clamp(idxU - 1, 0, condCdf.mParams.mShape.x1 - 1);
			float pdfU = condPdf[ShapeJit(idxV, idxU)] / marginalPdf[idxV];
			float baseU = (idxU - 1 > 0) ? condCdf[ShapeJit(idxV, idxU - 1)] : 0.0f;
			float du = (u - condCdf[ShapeJit(idxV, idxU)]) / (condCdf[ShapeJit(idxV, idxU)] - baseU + 1e-4f);
			sampledU[i] = (idxU + du) / float(condCdf.mParams.mShape.x1);

			sampledPdf[i] = pdfU * pdfV;
		}

		__global__ void Pdf2D(
			const TensorJit<float> condPdf, const TensorJit<float> condCdf,
			const TensorJit<float> marginalPdf, const TensorJit<float> marginalCdf,
			const float marginalIntegral,
			const TensorJit<float> u, const TensorJit<float> v,
			TensorJit<float> sampledPdf)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;

			if (i >= u.mParams.mShape.x1)
				return;

			int idxV = v[i] * marginalPdf.LinearSize();
			idxV = Math::Clamp(idxV - 1, 0, marginalCdf.LinearSize() - 1);
			float pdfV = marginalPdf[idxV] / (marginalIntegral);

			int idxU = u[i] * condPdf.mParams.mShape.x1;
			idxU = Math::Clamp(idxU - 1, 0, condPdf.mParams.mShape.x1 - 1);
			float pdfU = condPdf[ShapeJit(idxV, idxU)] / marginalPdf[idxV];

			sampledPdf[i] = pdfU * pdfV;
		}

		void Distribution1D::SampleContinuous(const Tensorf& u, Tensori* pOffset, Tensorf* pSampled, Tensorf* pPdf) const
		{
			pOffset->Resize(u.LinearSize());
			pSampled->Resize(u.LinearSize());
			pPdf->Resize(u.LinearSize());

			const int linearSize = u.LinearSize();
			const int blockDim = 256;
			const int gridDim = (linearSize + blockDim - 1) / blockDim;

			SampleContinuous1D << <gridDim, blockDim >> > (mCDF.ToJit(), mPDF.ToJit(), u.ToJit(), mIntegralVal, pOffset->ToJit(), pSampled->ToJit(), pPdf->ToJit());
		}

		void Distribution1D::SampleDiscrete(const Tensorf& u, Tensori* pOffset, Tensorf* pPdf) const
		{
			pOffset->Resize(u.LinearSize());
			pPdf->Resize(u.LinearSize());

			const int linearSize = u.LinearSize();
			const int blockDim = 256;
			const int gridDim = (linearSize + blockDim - 1) / blockDim;
			SampleDiscrete1D << <gridDim, blockDim >> > (mCDF.ToJit(), mPDF.ToJit(), u.ToJit(), mIntegralVal, pOffset->ToJit(), pPdf->ToJit());

			pOffset->CopyToHost();
			pPdf->CopyToHost();
		}

		Distribution2D::Distribution2D(const Tensorf& func)
		{
			Assert(func.Dim() == 2);

			mPdf = func;
			mCdf.Resize(mPdf.GetShape());

			Tensor<int> keys = Tensor<int>::ArrayRange(mPdf.GetShape(0));
			keys.Reshape(mPdf.GetShape(0), 1);
			keys = Broadcast(keys, mPdf.GetShape());
			Tensor<float> pdfCpu = mPdf;
			Tensor<float> cdfCpu = mCdf;

			thrust::inclusive_scan_by_key(
				thrust::host,
				keys.Data(),
				keys.Data() + keys.LinearSize(),
				pdfCpu.Data(), cdfCpu.Data()); // in-place scan

			Tensorf cdfGpu = cdfCpu;
			mCdf = cdfGpu;

			mCdf /= Scalar(mPdf.GetShape(1));
			mMarginalPdf = Slice(mCdf, { 0, mCdf.GetShape(1) - 1 }, mCdf.GetShape());
			mMarginalPdf.Reshape(mPdf.GetShape(0), 1);
			mCdf /= mMarginalPdf;

			// Build marginal distribution
			mMarginalPdf.Reshape(mPdf.GetShape(0));
			int marginalSize = mMarginalPdf.LinearSize();
			float invMarginalSize = 1.0f / float(marginalSize);
			mMarginalCdf = Tensorf::InclusiveScan(mMarginalPdf) * Scalar(invMarginalSize);

			mMarginalIntegral = mMarginalCdf.Get(marginalSize - 1);
			if (mMarginalIntegral > 0.0f)
			{
				mMarginalCdf /= Scalar(mMarginalIntegral);
			}
		}

		void Distribution2D::SampleContinuous(const Tensorf& uv, Tensorf* pSampledU, Tensorf* pSampledV, Tensorf* pPdf) const
		{
			const int N = uv.GetShape(1);
			pSampledU->Resize(N);
			pSampledV->Resize(N);
			pPdf->Resize(N);

			const int linearSize = N;
			const int blockDim = 256;
			const int gridDim = (linearSize + blockDim - 1) / blockDim;

			SampleContinuous2D << <gridDim, blockDim >> > (
				mPdf.ToJit(), mCdf.ToJit(),
				mMarginalPdf.ToJit(), mMarginalCdf.ToJit(),
				mMarginalIntegral, uv.ToJit(),
				pSampledU->ToJit(), pSampledV->ToJit(), pPdf->ToJit());
		}

		Tensorf Distribution2D::Pdf(const Tensorf& u, const Tensorf& v) const
		{
			Tensorf pdf;
			pdf.Resize(u.LinearSize());

			const int linearSize = u.LinearSize();
			const int blockDim = 256;
			const int gridDim = (linearSize + blockDim - 1) / blockDim;

			Pdf2D << <gridDim, blockDim >> > (
				mPdf.ToJit(), mCdf.ToJit(),
				mMarginalPdf.ToJit(), mMarginalCdf.ToJit(),
				mMarginalIntegral, u.ToJit(), v.ToJit(), pdf.ToJit());

			return pdf;
		}
	}
}