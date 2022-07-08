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

#include "../Core/Types.h"
#include "EDXMath.h"
#include "../Core/Memory.h"
#include "Constants.h"

namespace EDX
{
	namespace Math
	{
		struct FouriorData
		{
			float x;
			float y;
		};

		class FFT
		{
		private:
			uint miDimention;
			uint miNumButterflies;
			float* mpButterFlyData;

			mutable bool mbIsPingTarget;
			mutable FouriorData* mpFDataPing;
			mutable FouriorData* mpFDataPong;

		public:
			FFT()
				: mpButterFlyData(NULL)
			{
			}
			
			void Init1D(int iDim)
			{
				Assert(IsPowOfTwo(iDim));

				miDimention = iDim;

				miNumButterflies = logf(iDim) / logf(2.0f);
				mbIsPingTarget = true;

				CreateButterflyRes();
				mpFDataPing = new FouriorData[miDimention];
				mpFDataPong = new FouriorData[miDimention];
			}

			void Init2D(int iDim)
			{
				Assert(IsPowOfTwo(iDim));

				miDimention = iDim;

				miNumButterflies = logf(iDim) / logf(2.0f);
				mbIsPingTarget = true;

				CreateButterflyRes();
				mpFDataPing = new FouriorData[miDimention * miDimention];
				mpFDataPong = new FouriorData[miDimention * miDimention];
			}

			void PerformForward1D(float* pfDataIn, float* pfDataOut) const;
			void Perform1D() const;

			void PerformForward2D(float* pfDataIn, float* pfDataOut) const;
			void PerformInverse2D(float* pfDataIn, float* pfDataOut) const;
			void Perform2D() const;

			void SetDim(uint iDim)
			{
				Assert(IsPowOfTwo(iDim));

				miDimention = iDim;
				miNumButterflies = logf(iDim) / logf(2.0f);
				mbIsPingTarget = true;

				Memory::SafeDeleteArray(mpButterFlyData);
				Memory::SafeDeleteArray(mpFDataPing);
				Memory::SafeDeleteArray(mpFDataPong);

				CreateButterflyRes();

				mpFDataPing = new FouriorData[miDimention * miDimention];
				mpFDataPong = new FouriorData[miDimention * miDimention];
			}
			~FFT()
			{
				Memory::SafeDeleteArray(mpButterFlyData);
				Memory::SafeDeleteArray(mpFDataPing);
				Memory::SafeDeleteArray(mpFDataPong);
			}

		private:
			void CreateButterflyRes();
			void CalcIndices(float* pfIndices) const;
			void CalcWeights(float* pfWeights) const;
			void SwitchPingPongTarget(FouriorData*& pSrc, FouriorData*& pDest) const;
			void BitReverse(float* pfIndices, int N, int n) const;
		};
	}
}

