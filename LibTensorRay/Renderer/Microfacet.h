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
        class Microfacet : public BSDF
        {
        public:
            Texture* mpDiffuseReflectance;
            Texture* mpSpecularReflectance;
            Texture* mpRoughness;

            Microfacet(Texture* pDiffuseReflectance, Texture* pSpecularReflectance, Texture* pRoughness)
                : BSDF(nullptr), mpDiffuseReflectance(pDiffuseReflectance), mpSpecularReflectance(pSpecularReflectance), mpRoughness(pRoughness) { }

            ~Microfacet()
            {
                Memory::SafeDelete(mpDiffuseReflectance);
                Memory::SafeDelete(mpSpecularReflectance);
                Memory::SafeDelete(mpRoughness);
            }

            virtual Expr EvalInner(const Intersection& isect, const Expr& wo, const Expr& wi) const;
            virtual Expr SampleInner(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const;
            void SampleOnly(const Intersection& isect, const Expr& wo, const Expr& samples, Expr* pWi, Expr* pPdf) const;
            virtual Expr PdfInner(const Intersection& isect, const Expr& wo, const Expr& wi) const;

            virtual Texture* GetTexture(const char* type) const
            {
                if (strcmp(type, "diffuseReflectance") == 0)
                    return mpDiffuseReflectance;
                else if (strcmp(type, "specularReflectance") == 0)
                    return mpSpecularReflectance;
                else if (strcmp(type, "roughness") == 0)
                    return mpRoughness;
                else
                {
                    std::cout << "[ERROR] Texture " << type << " not supported by microfacet!" << std::endl;
                    return nullptr;
                }
            }
        };

        void MicrofacetTests();
    }
}