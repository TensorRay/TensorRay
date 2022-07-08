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

#include "BSDF.h"
#include "Windows/Bitmap.h"
#include "Core/Random.h"

namespace EDX
{
	namespace TensorRay
	{
		ConstantTexture::ConstantTexture(const Vector3& albedo)
			: Texture()
		{
			mAlbedo = Vector3ToTensor(albedo);
		}

		ConstantTexture::ConstantTexture(float albedo)
			: Texture()
		{
			mAlbedo = Tensorf({ albedo }, false);
		}

		ConstantTexture::ConstantTexture(const Tensorf& albedo)
			: Texture()
		{
			mAlbedo = albedo;
		}

		ImageTexture::ImageTexture(const char* path)
			: Texture()
		{
			float* pMap = Bitmap::ReadFromFile<float>(path, &mWidth, &mHeight, &mChannels, 3);

			mChannels = 3;
			mTexMap.Assign(pMap, { mHeight * mWidth, mChannels });

			// Transpose it into channels x texels
			mTexMap = Tensorf::Transpose(mTexMap);
			mTexMap = mTexMap.Reshape(mTexMap.GetShape().Vectorize());

			Memory::SafeDeleteArray(pMap);
		}

		Expr ImageTexture::Eval(const Expr& uv) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			int N = uv->GetShape()[1];
			auto u = X(uv);
			auto v = Y(uv);

			auto tiledU = u - Floor(u);
			auto tiledV = v - Floor(v);

			Tensori offsetX = tiledU * Scalar(mWidth - 1);
			Tensori offsetY = tiledV * Scalar(mHeight - 1);
			offsetX = offsetX.Reshape(offsetX.LinearSize());
			offsetY = offsetY.Reshape(offsetY.LinearSize());
			auto offset = offsetY * Scalar(mWidth) + offsetX;

			Expr ret = IndexedRead(mTexMap, offset, 0);

#if USE_PROFILING
			nvtxRangePop();
#endif

			return ret;
		}

		Expr BSDFCoords::CosTheta(const Expr& vec)
		{
			Assert(vec->GetShape().VectorSize() == 3);
			return Z(vec);
		}

		Expr BSDFCoords::CosTheta2(const Expr& vec)
		{
			return Square(CosTheta(vec));
		}

		Expr BSDFCoords::AbsCosTheta(const Expr& vec)
		{
			return Abs(CosTheta(vec));
		}

		Expr BSDFCoords::SinTheta2(const Expr& vec)
		{
			return Maximum(Scalar(1.0f) - CosTheta2(vec), Scalar(0.0f));
		}

		Expr BSDFCoords::SinTheta(const Expr& vec)
		{
			return Sqrt(SinTheta2(vec));
		}

		Expr BSDFCoords::TanTheta(const Expr& vec)
		{
            return SinTheta(vec) / CosTheta(vec);
		}

		Expr BSDFCoords::TanTheta2(const Expr& vec)
		{
			auto cosTheta2 = CosTheta2(vec);
			auto temp = Maximum(Scalar(1.0f) - cosTheta2, Scalar(0.0f));
			return temp / cosTheta2;
		}

		Expr BSDFCoords::AbsTanTheta(const Expr& vec)
		{
			return Abs(TanTheta(vec));
		}

		Expr BSDFCoords::SameHemisphere(const Expr& v1, const Expr& v2)
		{
			return (CosTheta(v1) * CosTheta(v2)) > Scalar(0.0f);
		}

		Expr BSDFCoords::CosPhi(const Expr& vec)
		{
			auto sinTheta = SinTheta(vec);
			auto ret = Clamp(X(vec) / sinTheta, Scalar(-1.0f), Scalar(1.0f));
			return Where(sinTheta == Scalar(0.0f), Scalar(1.0f), ret);
		}

		Expr BSDFCoords::SinPhi(const Expr& vec)
		{
			auto sinTheta = SinTheta(vec);
			auto ret = Clamp(Y(vec) / sinTheta, Scalar(-1.0f), Scalar(1.0f));
			return Where(sinTheta == Scalar(0.0f), Scalar(0.0f), ret);
		}

		Expr BSDFCoords::CosPhi2(const Expr& vec)
		{
			return Square(CosPhi(vec));
		}

		Expr BSDFCoords::SinPhi2(const Expr& vec)
		{
			return Square(SinPhi(vec));
		}

		Expr BSDF::Eval(const Intersection& isect, const Expr& wo, const Expr& wi, bool correction) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			Expr localWo = isect.WorldToLocal(wo);
			Expr localWi = isect.WorldToLocal(wi);
			Expr correctTerm = correction ? Abs( Z(localWo) * VectorDot(isect.mGeoNormal, wi) / (Z(localWi) * VectorDot(isect.mGeoNormal, wo)) )
										  : Scalar(1.0f);

#if USE_PROFILING
			nvtxRangePop();
#endif

			if (mpTexture == nullptr)
			{
				return EvalInner(isect, localWo, localWi) * correctTerm;
			}
			else
			{
				return mpTexture->Eval(isect.mTexcoord) * EvalInner(isect, localWo, localWi) * correctTerm;
			}
		}

		Expr BSDF::Sample(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			Expr localWo = isect.WorldToLocal(wo);

			Expr localWi;
			Expr f = SampleInner(isect, localWo, samples, &localWi, pPdf);

			*pWi = isect.LocalToWorld(localWi);

#if USE_PROFILING
			nvtxRangePop();
#endif

			if (mpTexture == nullptr)
			{
				return f;
			}
			else
			{
				return mpTexture->Eval(isect.mTexcoord) * f;
			}
		}

		Expr BSDF::Pdf(const Intersection& isect, const Expr& wo, const Expr& wi) const
		{
			Expr localWo = isect.WorldToLocal(wo);
			Expr localWi = isect.WorldToLocal(wi);

			return PdfInner(isect, localWo, localWi);
		}

		Expr BSDF::GGX_D(const Expr& wh, const Expr& alpha)
		{
			auto tanTheta2 = BSDFCoords::TanTheta2(wh);
			auto isValid = (tanTheta2 < Scalar(FLT_MAX));
			auto cosTheta2 = BSDFCoords::CosTheta2(wh);
			auto e = Scalar(1.0f) / (alpha * alpha) * tanTheta2;
			auto root = alpha * cosTheta2 * (Scalar(1.0f) + e);
			auto D = Scalar(float(Math::EDX_INV_PI)) / Square(root);
			return Where(isValid, D, Scalar(0.0f));
		}

		Expr BSDF::Smith_G(const Expr& v, const Expr& wh, const Expr& alpha)
		{
			auto tanTheta2 = BSDFCoords::TanTheta2(wh);
			auto isValid = (tanTheta2 < Scalar(FLT_MAX));
			auto root = Square(alpha) * tanTheta2;
			auto G = Scalar(2.0f) / (Scalar(1.0f) + Sqrt(Scalar(1.0f) + Square(root)));
			G = Where(isValid, G, Scalar(0.0f));
            
			auto VoH = VectorDot(v, wh);
            auto cosThetaV = BSDFCoords::CosTheta(v);
			G = Where(tanTheta2 == Scalar(0.0f), Scalar(1.0f), G);
			G = Where((VoH * cosThetaV) <= Scalar(0.0f), Scalar(0.0f), G);
			return G;
		}

		Expr BSDF::GGX_G(const Expr& wo, const Expr& wi, const Expr& wh, const Expr& alpha)
		{
			return Smith_G(wo, wh, alpha) * Smith_G(wi, wh, alpha);
		}

		Expr BSDF::GGX_VNDF_Pdf(const Expr& v, const Expr& wh, const Expr& alpha)
		{
			auto D = GGX_D(wh, alpha);
			auto G = Smith_G(v, wh, alpha);
			auto VoH = Abs(VectorDot(v, wh));
			auto pdf = D * G * VoH / BSDFCoords::AbsCosTheta(v);  //Maximum(BSDFCoords::AbsCosTheta(v), Scalar(1e-4f));
			return Detach(pdf);
		}

		Expr BSDF::GGX_VNDF_Sample(const Expr& v, const Expr& alpha, const Expr& samples)
		{
			auto alpha_scale = MakeVector3(alpha, alpha, Ones(1));
			auto V = v * alpha_scale;
			V = VectorNormalize(V);

			auto sin_phi = BSDFCoords::SinPhi(V);
			auto cos_phi = BSDFCoords::CosPhi(V);
			auto cos_theta = BSDFCoords::CosTheta(V);

			auto slope = GGX_VNDF_SampleVisible11(cos_theta, samples);

			auto slope_x = (cos_phi * X(slope) - sin_phi * Y(slope)) * alpha;
			auto slope_y = (sin_phi * X(slope) + cos_phi * Y(slope)) * alpha;
			auto res = VectorNormalize(MakeVector3(-slope_x, -slope_y, Scalar(1.0f)));
			return res;
		}

		Expr BSDF::GGX_VNDF_SampleVisible11(const Expr& cos_theta, const Expr& samples)
		{
			auto p = Sampling::SquareToUniformDiskConcentric(samples);
			auto s = Scalar(0.5f) * (Scalar(1.0f) + cos_theta);
			auto x = X(p);
			auto y = Lerp(SafeSqrt(Scalar(1.0f) - Square(x)), Y(p), s);
			auto z = SafeSqrt(Scalar(1.0f) - VectorSquaredLength(p));

			auto sin_theta = SafeSqrt(Scalar(1.0f) - Square(cos_theta));
			auto norm = Scalar(1.0f) / (sin_theta * y + cos_theta * z);
			auto res = MakeVector2(cos_theta * y - sin_theta * z, x);
			return res * norm;
		}

		Expr BSDF::FresnelConductor(const Expr& cosThetaI, const Expr& eta, const Expr& k)
		{
			auto cosThetaI2 = cosThetaI * cosThetaI;
			auto sinThetaI2 = Scalar(1.0f) - cosThetaI2;
			auto sinThetaI4 = sinThetaI2 * sinThetaI2;
			auto eta2 = Square(eta), k2 = Square(k);

			auto temp1 = eta2 - k2 - sinThetaI2;
			auto a2pb2 = SafeSqrt(Square(temp1) + k2 * eta2 * Scalar(4.0f));
			auto a = SafeSqrt((a2pb2 + temp1) * Scalar(0.5f));
			auto term1 = a2pb2 + cosThetaI2;
			auto term2 = a * (Scalar(2.0f) * cosThetaI);
			auto Rs2 = (term1 - term2) / (term1 + term2);

			auto term3 = a2pb2 * cosThetaI2 + sinThetaI4,
				 term4 = term2 * sinThetaI2;
			auto Rp2 = Rs2 * (term3 - term4) / (term3 + term4);

			return Scalar(0.5f) * (Rp2 + Rs2);
		}

		/*
		Expr BSDF::FresnelDielectric(const Expr& cosi, const float _etai, const float _etat)
		{
			auto entering = cosi > Zeros(1);
			auto etai = Scalar(_etai);
			auto etat = Scalar(_etat);
			auto ei = Where(entering, Scalar(etai), Scalar(etat));
			auto et = Where(entering, Scalar(etat), Scalar(etai));

			auto sint = ei / et * Sqrt(Maximum(Ones(1) - cosi * cosi, Scalar(1e-4f)));
			auto cost = Sqrt(Maximum(Ones(1) - sint * sint, Scalar(1e-4f)));

			auto absCosi = Abs(cosi);

			auto para = ((etat * absCosi) - (etai * cost)) /
				((etat * absCosi) + (etai * cost));
			auto perp = ((etai * absCosi) - (etat * cost)) /
				((etai * absCosi) + (etat * cost));

			auto fr = Where(sint < Ones(1), Scalar(0.5f)* (Square(para) + Square(perp)), Ones(1));

			return fr;
		}
		*/

		void BSDF::DiffTexture(const std::string& name)
		{
			if (name.length() == 0)
				GetTexture()->GetTextureTensor().SetRequiresGrad(true);
			else
				GetTexture(name.c_str())->GetTextureTensor().SetRequiresGrad(true);
		}
	}
}