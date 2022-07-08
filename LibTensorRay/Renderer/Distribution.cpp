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
		Expr Distribution1D::ReuseSample(const Tensorf& rnd, const Tensori& index) const
		{
			auto cdf = Concat(Tensorf({ 0.0f }), mCDF, 0);
			auto pmin = IndexedRead(cdf, index, 0);
			auto pmax = IndexedRead(cdf, index + Scalar(1), 0);
			return Tensorf((rnd.Reshape(rnd.LinearSize()) - pmin) / (pmax - pmin)).Reshape(rnd.GetShape());
		}

		void Distribution1D::SetFunction(const Tensorf& func)
		{
			mPDF = Detach(func);
			int size = mPDF.LinearSize();
			mPDF = mPDF.Reshape(size);
			float invSize = 1.0f / float(size);
			mCDF = Tensorf::InclusiveScan(mPDF) * Tensorf({ invSize });

			mIntegralVal = mCDF.Get(size - 1);
			if (mIntegralVal > 0.0f)
			{
				mCDF /= Tensorf({ mIntegralVal });
			}
		}
	}
}