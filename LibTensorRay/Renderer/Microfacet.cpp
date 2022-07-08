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

#include "Microfacet.h"

namespace EDX
{
    namespace TensorRay
    {
        Expr Microfacet::EvalInner(const Intersection& isect, const Expr& wo, const Expr& wi) const
        {
            auto nDotV = BSDFCoords::CosTheta(wo); // view dir
            auto nDotL = BSDFCoords::CosTheta(wi); // light dir
            auto active = (nDotV > Scalar(0.0f) && nDotL > Scalar(0.0f));

            auto diffuseRefl = mpDiffuseReflectance->Eval(isect.mTexcoord) * Scalar(float(Math::EDX_INV_PI));

            auto wh = VectorNormalize(wo + wi);
            auto nDotH = BSDFCoords::CosTheta(wh);
            auto vDotH = VectorDot(wo, wh);

            auto F0 = mpSpecularReflectance->Eval(isect.mTexcoord);
            auto roughness = mpRoughness->Eval(isect.mTexcoord);
            auto alpha = Square(roughness);
            auto k = Square(roughness + Scalar(1.f)) * Scalar(0.125f);

            // GGX NDF term
            auto tmp = alpha / (nDotH * nDotH * (Square(alpha) - Scalar(1.f)) + Scalar(1.f));
            auto ggx = tmp * tmp * Scalar(float(Math::EDX_INV_PI));

            // Fresnel term
            auto coeff = vDotH * (vDotH * Scalar(-5.55473f) - Scalar(6.8316f));
            auto fresnel = F0 + (Scalar(1.f) - F0) * Exponent(Log(Scalar(2.f)) * coeff); // Note: Pow function is not implemented!

            // Geometry term
            auto smithG1 = nDotV / (nDotV * (Scalar(1.f) - k) + k);
            auto smithG2 = nDotL / (nDotL * (Scalar(1.f) - k) + k);
            auto smithG = smithG1 * smithG2;

            auto numerator = ggx * smithG * fresnel;
            auto denominator = Scalar(4.f) * Abs(nDotV) * Abs(nDotL);
            auto specularRefl = numerator / (denominator + Scalar(1e-6f));

            auto value = diffuseRefl + specularRefl;
            return Where(active, value, Scalar(0.0f));
        }

        // TODO: Also consider sampling the diffuse lobe.
        // Currently, importance sampling only consider the GGX lobe.
        Expr Microfacet::SampleInner(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const
        {
            auto nDotV = BSDFCoords::CosTheta(wo); // view dir
            auto active = nDotV > Scalar(0.0f);

            auto diffuseRefl = mpDiffuseReflectance->Eval(isect.mTexcoord) * Scalar(float(Math::EDX_INV_PI));
            
            auto F0 = mpSpecularReflectance->Eval(isect.mTexcoord);
            auto roughness = mpRoughness->Eval(isect.mTexcoord);
            auto alpha = Square(roughness);
            auto k = Square(roughness + Scalar(1.f)) * Scalar(0.125f);

            auto wh = GGX_VNDF_Sample(wo, roughness, samples);
            auto microfacetPdf = GGX_VNDF_Pdf(wo, wh, roughness);

            auto wi = VectorReflect(-wo, wh);
            auto nDotL = BSDFCoords::CosTheta(wi); // light dir
            active = (active && microfacetPdf != Scalar(0.0f) && nDotL > Scalar(0.0f));

            auto dwh_dwi = Scalar(1.0f) / (Scalar(4.0f) * VectorDot(wi, wh));
            auto pdf = Abs(microfacetPdf * dwh_dwi);

            auto nDotH = BSDFCoords::CosTheta(wh);
            auto vDotH = VectorDot(wo, wh);

            // GGX NDF term
            auto tmp = alpha / (nDotH * nDotH * (Square(alpha) - Scalar(1.f)) + Scalar(1.f));
            auto ggx = tmp * tmp * Scalar(float(Math::EDX_INV_PI));

            // Fresnel term
            auto coeff = vDotH * (vDotH * Scalar(-5.55473f) - Scalar(6.8316f));
            auto fresnel = F0 + (Scalar(1.f) - F0) * Exponent(Log(Scalar(2.f)) * coeff); // Note: Pow function is not implemented!

            // Geometry term
            auto smithG1 = nDotV / (nDotV * (Scalar(1.f) - k) + k);
            auto smithG2 = nDotL / (nDotL * (Scalar(1.f) - k) + k);
            auto smithG = smithG1 * smithG2;

            auto numerator = ggx * smithG * fresnel;
            auto denominator = Scalar(4.f) * Abs(nDotV) * Abs(nDotL);
            auto specularRefl = numerator / (denominator + Scalar(1e-6f));

            auto value = diffuseRefl + specularRefl;
            value = Where(active, value, Scalar(0.0f));
            *pPdf = Where(active, pdf, Scalar(0.0f));
            *pWi = wi;

            return value;
        }

        void Microfacet::SampleOnly(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const
        {
#if USE_PROFILING
            nvtxRangePushA(__FUNCTION__);
#endif
            Expr localWo = isect.WorldToLocal(wo);
            auto reflect = BSDFCoords::CosTheta(localWo) > Scalar(0.0f);

            auto roughness = mpRoughness->Eval(isect.mTexcoord);
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

        Expr Microfacet::PdfInner(const Intersection& isect, const Expr& wo, const Expr& wi) const
        {
            auto roughness = mpRoughness->Eval(isect.mTexcoord);
            auto wh = VectorNormalize(wo + wi);
            auto microfacetPdf = GGX_VNDF_Pdf(wo, wh, roughness);

            auto dwh_dwi = Scalar(1.0f) / (Scalar(4.0f) * VectorDot(wi, wh));
            auto pdf = Abs(microfacetPdf * dwh_dwi);

            auto reflect = (BSDFCoords::CosTheta(wo) > Scalar(0.0f) &&
                microfacetPdf != Scalar(0.0f) && BSDFCoords::CosTheta(wi) > Scalar(0.0f));
            pdf = Where(reflect, pdf, Scalar(0.0f));
            return pdf;
        }

        void MicrofacetTests()
        {
            Tensorf roughness({ 0.2f });
            roughness.SetRequiresGrad(true);

            Microfacet bsdf = Microfacet(new ConstantTexture(Vector3(0.2f, 0.3f, 0.25f)),
                new ConstantTexture(Vector3(0.8f, 0.5f, 0.2f)),
                new ConstantTexture(roughness));

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

            Tensorf texcoord = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec2));
            isect.mTexcoord = texcoord;
            Tensorf f = bsdf.Eval(isect, VectorNormalize(vIn), VectorNormalize(vOut));
            std::cout << "f: " << f << "\n";

            ValidateBackwardDiff(f, bsdf.mpRoughness->GetTextureTensor());
            ValidateForwardDiff(f, bsdf.mpRoughness->GetTextureTensor());

            bsdf.mpRoughness->GetTextureTensor().ClearGrad();

            Tensorf pdf = bsdf.Pdf(isect, VectorNormalize(vIn), VectorNormalize(vOut));
            std::cout << "pdf: " << pdf << "\n";

            Tensorf samples = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec3));
            Expr scatterPdfExpr;
            Expr scatterDirExpr;
            Tensorf f_sample = bsdf.Sample(isect, vIn, samples, &scatterDirExpr, &scatterPdfExpr);
            Tensorf scatterPdf = scatterPdfExpr;
            Tensorf scatterDir = scatterDirExpr;

            std::cout << "f_sample: " << f_sample << "\n";
            std::cout << "dir_sample: " << scatterDir << "\n";
            std::cout << "pdf_sample: " << scatterPdf << "\n";

            ValidateBackwardDiff(scatterDir, bsdf.mpRoughness->GetTextureTensor());
            ValidateForwardDiff(scatterDir, bsdf.mpRoughness->GetTextureTensor());

            bsdf.mpRoughness->GetTextureTensor().ClearGrad();

            ParameterPool::GetHandle().clear();
        }
    }
}