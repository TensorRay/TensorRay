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
#include "Roughconductor.h"

namespace EDX
{
	namespace TensorRay
	{
		Expr RoughConductor::EvalInner(const Intersection& isect, const Expr& wo, const Expr& wi) const
		{
			auto active = (BSDFCoords::CosTheta(wo) > Scalar(0.0f) && BSDFCoords::CosTheta(wi) > Scalar(0.0f));

			auto wh = VectorNormalize(wo + wi);
			
			auto roughness = mRoughness->Eval(isect.mTexcoord);
			auto D = GGX_D(wh, roughness);
			active = (active && D != Scalar(0.0f));

			auto G = GGX_G(wo, wi, wh, roughness);
			
			auto coso = VectorDot(wo, wh);
			auto eta = mEta->Eval(isect.mTexcoord);
			auto k = mK->Eval(isect.mTexcoord);
			auto Fr = FresnelConductor(coso, eta, k);

			auto f = Fr * D * G / (Scalar(4.0f) * BSDFCoords::AbsCosTheta(wi) * BSDFCoords::AbsCosTheta(wo));
			return Where(active, f, Scalar(0.0f));
		}

		Expr RoughConductor::SampleInner(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const
		{
			auto reflect = BSDFCoords::CosTheta(wo) > Scalar(0.0f);
			
			auto roughness = mRoughness->Eval(isect.mTexcoord);
			auto wh = GGX_VNDF_Sample(wo, roughness, samples);
			auto microfacetPdf = GGX_VNDF_Pdf(wo, wh, roughness);

			auto wi = VectorReflect(-wo, wh);
			reflect = (reflect && microfacetPdf != Scalar(0.0f) && BSDFCoords::CosTheta(wi) > Scalar(0.0f));

			auto dwh_dwi = Scalar(1.0f) / (Scalar(4.0f) * VectorDot(wi, wh));
			auto pdf = Abs(microfacetPdf * dwh_dwi);

			auto D = GGX_D(wh, roughness);
			auto coso = VectorDot(wo, wh);
			auto eta = mEta->Eval(isect.mTexcoord);
			auto k = mK->Eval(isect.mTexcoord);
			auto Fr = FresnelConductor(coso, eta, k);
			auto G = GGX_G(wo, wi, wh, roughness);

			auto f = Fr * D * G / (Scalar(4.0f) * BSDFCoords::AbsCosTheta(wi) * BSDFCoords::AbsCosTheta(wo));

			f = Where(reflect, f, Scalar(0.0f));
			*pPdf = Where(reflect, pdf, Scalar(0.0f));
			*pWi = wi;

			return f;
		}

		void RoughConductor::SampleOnly(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			Expr localWo = isect.WorldToLocal(wo);
			auto reflect = BSDFCoords::CosTheta(localWo) > Scalar(0.0f);

			auto roughness = mRoughness->Eval(isect.mTexcoord);
			auto wh = GGX_VNDF_Sample(localWo, roughness, samples);
			auto microfacetPdf = GGX_VNDF_Pdf(localWo, wh, roughness);

			Expr localWi = VectorReflect(-localWo, wh);
			reflect = (reflect && microfacetPdf != Scalar(0.0f) && BSDFCoords::CosTheta(localWi) > Scalar(0.0f));

			auto dwh_dwi = Scalar(1.0f) / (Scalar(4.0f) * VectorDot(localWi, wh));
			auto pdf = Abs(microfacetPdf * dwh_dwi);

			*pPdf = Where(reflect, pdf, Scalar(0.0f));
			*pWi = isect.LocalToWorld(localWi);
#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		Expr RoughConductor::PdfInner(const Intersection& isect, const Expr& wo, const Expr& wi) const
		{
			auto roughness = mRoughness->Eval(isect.mTexcoord);
			auto wh = VectorNormalize(wo + wi);
			auto microfacetPdf = GGX_VNDF_Pdf(wo, wh, roughness);

			auto dwh_dwi = Scalar(1.0f) / (Scalar(4.0f) * VectorDot(wi, wh));
			auto pdf = Abs(microfacetPdf * dwh_dwi);

			auto reflect = (BSDFCoords::CosTheta(wo) > Scalar(0.0f) && 
				microfacetPdf != Scalar(0.0f) && BSDFCoords::CosTheta(wi) > Scalar(0.0f));
			pdf = Where(reflect, pdf, Scalar(0.0f));
			return pdf;
		}

		void RoughConductorTests()
		{
			Tensorf roughness({ 0.2f });
			roughness.SetRequiresGrad(true);

			RoughConductor bsdf = RoughConductor(new ConstantTexture(roughness),
												 new ConstantTexture(Vector3(0.200438f, 0.924033f, 1.10221f)),
												 new ConstantTexture(Vector3(3.91295f, 2.45285f, 2.14219f)),
												 new ConstantTexture(Vector3(1.0f, 1.0f, 1.0f)));

			int N = 1;
			Tensorf sampleIn = Tensorf::RandomFloat(Shape({ N }, VecType::Vec2));
			Tensorf sampleOut = Tensorf::RandomFloat(Shape({ N }, VecType::Vec2));

			Tensorf vIn = Sampling::CosineSampleHemisphere(sampleIn);
			Tensorf vOut = Sampling::CosineSampleHemisphere(sampleOut);

			Intersection isect;

			Tensorf temp = Tensorf({ Vector3(0.0f, 0.0f, 1.0f) }, false);
			Tensorf normal = Broadcast(temp, Shape({ N }, VecType::Vec3));
			isect.mNormal = normal;
			isect.mNormal = VectorNormalize(isect.mNormal);
			CoordinateSystem(isect.mNormal, &isect.mTangent, &isect.mBitangent);

			Tensorf texcoord = Tensorf::RandomFloat(Shape({ N }, VecType::Vec2));
			isect.mTexcoord = texcoord;
			Tensorf f = bsdf.Eval(isect, VectorNormalize(vIn), VectorNormalize(vOut));
			std::cout << "f: " << f << "\n";

			ValidateBackwardDiff(f, bsdf.mRoughness->GetTextureTensor());
			ValidateForwardDiff(f, bsdf.mRoughness->GetTextureTensor());

			bsdf.mRoughness->GetTextureTensor().ClearGrad();

			Tensorf pdf = bsdf.Pdf(isect, VectorNormalize(vIn), VectorNormalize(vOut));
			std::cout << "pdf: " << pdf << "\n";

			Tensorf samples = Tensorf::RandomFloat(Shape({ N }, VecType::Vec3));
			Expr scatterPdfExpr;
			Expr scatterDirExpr;
			Tensorf f_sample = bsdf.Sample(isect, vIn, samples, &scatterDirExpr, &scatterPdfExpr);
			Tensorf scatterPdf = scatterPdfExpr;
			Tensorf scatterDir = scatterDirExpr;

			std::cout << "f_sample: " << f_sample << "\n";
			std::cout << "dir_sample: " << scatterDir << "\n";
			std::cout << "pdf_sample: " << scatterPdf << "\n";

			ValidateBackwardDiff(scatterDir, bsdf.mRoughness->GetTextureTensor());
			ValidateForwardDiff(scatterDir, bsdf.mRoughness->GetTextureTensor());

			bsdf.mRoughness->GetTextureTensor().ClearGrad();

			ParameterPool::GetHandle().clear();
		}
	}
}