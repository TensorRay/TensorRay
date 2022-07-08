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
#include "Records.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		class Texture
		{
		public:
			virtual ~Texture()
			{
			}

			virtual Expr Eval(const Expr& uv) const = 0;
			virtual Tensorf& GetTextureTensor() = 0;
		};

		class ConstantTexture : public Texture
		{
		protected:
			Tensorf mAlbedo;

		public:
			ConstantTexture(const Vector3& albedo);
			ConstantTexture(float albedo);
			ConstantTexture(const Tensorf& albedo);

			virtual Expr Eval(const Expr& uv) const
			{
				return mAlbedo;
			}

			Tensorf& GetTextureTensor() { return mAlbedo; }

		};

		class ImageTexture : public Texture
		{
		protected:
			Tensorf mTexMap;
			int mWidth, mHeight, mChannels;

		public:
			ImageTexture(const char* path);

			virtual Expr Eval(const Expr& uv) const;

			Tensorf& GetTextureTensor() { return mTexMap; }
		};

		namespace BSDFCoords
		{
			Expr CosTheta(const Expr& vec);
			Expr CosTheta2(const Expr& vec);
			Expr AbsCosTheta(const Expr& vec);
			Expr SinTheta2(const Expr& vec);
			Expr SinTheta(const Expr& vec);
			Expr TanTheta(const Expr& vec);
			Expr TanTheta2(const Expr& vec);
			Expr AbsTanTheta(const Expr& vec);
			Expr SameHemisphere(const Expr& v1, const Expr& v2);

			Expr CosPhi(const Expr& vec);
			Expr SinPhi(const Expr& vec);
			Expr CosPhi2(const Expr& vec);
			Expr SinPhi2(const Expr& vec);
		}

		class BSDF
		{
		protected:
			Texture* mpTexture;

		public:
			int mId;
			const char* mStrId;

		public:
			BSDF(Texture* pTex)
				: mpTexture(pTex)
			{
				mStrId = "";
				mId = 0;
			}
			virtual ~BSDF()
			{
				Memory::SafeDelete(mpTexture);
			}
			virtual Expr Eval(const Intersection& isect, const Expr& wo, const Expr& wi, bool correction = false) const;
			virtual Expr Sample(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const;
			virtual void SampleOnly(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const { Assert(false); };
			virtual Expr Pdf(const Intersection& isect, const Expr& wo, const Expr& wi) const;

			virtual Expr EvalInner(const Intersection& isect, const Expr& wo, const Expr& wi) const = 0;
			virtual Expr SampleInner(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const = 0;
			virtual Expr PdfInner(const Intersection& isect, const Expr& wo, const Expr& wi) const = 0;
			virtual Expr OutDirValid(const Intersection& isect, const Expr& wo) const { return VectorDot(isect.mNormal, wo) * VectorDot(isect.mGeoNormal, wo) > Scalar(0.0f); }
			virtual bool IsDelta() const { return false; }

			static Expr GGX_D(const Expr& wh, const Expr& alpha);
			static Expr Smith_G(const Expr& v, const Expr& wh, const Expr& alpha);
			static Expr GGX_G(const Expr& wo, const Expr& wi, const Expr& wh, const Expr& alpha);
			static Expr GGX_VNDF_Pdf(const Expr& v, const Expr& wh, const Expr& alpha);
			static Expr GGX_VNDF_Sample(const Expr& v, const Expr& alpha, const Expr& samples);
			static Expr GGX_VNDF_SampleVisible11(const Expr& cos_theta, const Expr& samples);

			static Expr FresnelConductor(const Expr& cosThetaI, const Expr& eta, const Expr& k);
			//static Expr FresnelDielectric(const Expr& cosi, const float _etai, const float _etat);

			virtual Texture* GetTexture(const char* type = "albedo") const
			{
				if (strcmp(type, "albedo") == 0)
					return mpTexture;
				else
				{
					std::cout << "[ERROR] Texture " << type << " not supported!" << std::endl;
					return nullptr;
				}
			}

			// python binding
			void DiffTexture(const std::string& name);
		};
	}
}