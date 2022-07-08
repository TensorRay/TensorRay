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
#include "BSDF.h"

namespace EDX
{
	namespace TensorRay
	{
		class RoughConductor : public BSDF
		{
		public:
			Texture* mRoughness;
			Texture* mEta;
			Texture* mK;

			RoughConductor(Texture* pRoughness, Texture* eta, Texture* k, Texture* pTex = nullptr)
				: BSDF(pTex), mRoughness(pRoughness), mEta(eta), mK(k) { }

			~RoughConductor()
			{
				Memory::SafeDelete(mRoughness);
				Memory::SafeDelete(mEta);
				Memory::SafeDelete(mK);
			}

			virtual Expr EvalInner(const Intersection& isect, const Expr& wo, const Expr& wi) const;
			virtual Expr SampleInner(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const;
			void SampleOnly(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const;
			virtual Expr PdfInner(const Intersection& isect, const Expr& wo, const Expr& wi) const;

			virtual Texture* GetTexture(const char* type) const
			{
				if (strcmp(type, "albedo") == 0)
					return mpTexture;
				else if (strcmp(type, "roughness") == 0)
					return mRoughness;
				else if (strcmp(type, "eta") == 0)
					return mEta;
				else if (strcmp(type, "k") == 0)
					return mK;
				else
				{
					std::cout << "[ERROR] Texture " << type << " not supported by roughconductor!" << std::endl;
					return nullptr;
				}
			}
		};

		void RoughConductorTests();
	}
}