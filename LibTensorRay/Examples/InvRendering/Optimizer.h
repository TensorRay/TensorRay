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
#include "../../Tensor/Tensor.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace DeepLearning
	{
		class Optimizer
		{
		public:
			virtual void Step(Tensorf& x, const Tensorf& dx) = 0;
		};

		class StochasticGradientDescent : public Optimizer
		{
		private:
			float mLearningRate;

		public:
			StochasticGradientDescent(float learningRate)
				: mLearningRate(learningRate)
			{

			}

			virtual void Step(Tensorf& x, const Tensorf& dx) override
			{
				Tensorf updatedValue = Detach(x) - Scalar(mLearningRate) * dx;
				x.Assign(updatedValue.HostData(), x.GetShape());
			}
		};

		class Adam : public Optimizer
		{
		public:
			float mLearningRate = 1e-3f;
			float mBeta1 = 0.9f;
			float mBeta2 = 0.999f;
			float mEpsilon = 1e-6f;
			std::map<Tensorf*, Tensorf> mMapM;
			std::map<Tensorf*, Tensorf> mMapV;
			int mSteps = 0;

		public:
			Adam(const float learningRate = 1e-3f,
				const float beta1 = 0.9f,
				const float beta2 = 0.999f,
				const float eps = 1e-6f,
				const int steps = 0)
				: mLearningRate(learningRate)
				, mBeta1(beta1)
				, mBeta2(beta2)
				, mEpsilon(eps)
				, mSteps(steps)
			{

			}

			void SetLearningRate(const float learningRate)
			{
				mLearningRate = learningRate;
			}

			virtual void Step(Tensorf& x, const Tensorf& dx) override
			{
				if (mMapM.find(&x) == mMapM.end())
				{
					mMapM[&x] = Zeros(x.GetShape());
				}
				if (mMapV.find(&x) == mMapV.end())
				{
					mMapV[&x] = Zeros(x.GetShape());
				}

				Tensorf m = mMapM[&x];
				Tensorf v = mMapV[&x];

				mSteps++;
				m = Scalar(mBeta1) * m + Scalar(1 - mBeta1) * dx;
				v = Scalar(mBeta2) * v + Scalar(1 - mBeta2) * (dx * dx);
				Tensorf mt_hat = m / Scalar(1 - Math::Pow(mBeta1, mSteps));
				Tensorf vt_hat = v / Scalar(1 - Math::Pow(mBeta2, mSteps));
				Tensorf realStep = Scalar(mLearningRate) * mt_hat / (Sqrt(vt_hat) + Scalar(mEpsilon));
				Tensorf updatedValue = Detach(x) - realStep;
				
				x.Assign(updatedValue.HostData(), x.GetShape());
				mMapM[&x].Assign(m.HostData(), m.GetShape());
				mMapV[&x].Assign(v.HostData(), v.GetShape());
			}
		};
	}
}