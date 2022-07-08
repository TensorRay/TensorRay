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
#include "Math/Matrix.h"
#include "Records.h"
#include "Config.h"

#define TENSORRAY_ASSERT(cond)											   \
	do																		   \
	{																		   \
		if (!(cond))															   \
		{																	   \
			std::stringstream ss;											   \
			ss << "\n File \"" << __FILE__ << "\", line " <<__LINE__;		   \
			throw std::runtime_error(ss.str().c_str());						   \
		}																	   \
	} while ( 0 )


#define TENSORRAY_ASSERT_MSG(cond, msg)											   \
	do																		   \
	{																		   \
		if (!(cond))															   \
		{																	   \
			std::stringstream ss;											   \
			ss << "\n File \"" << __FILE__ << "\", line " <<__LINE__;		   \
			throw std::runtime_error((std::string(msg) + ss.str()).c_str());    \
		}																	   \
	} while ( 0 )															   \

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		namespace Sampling
		{
			Expr UniformSampleCone(const Expr& samples, const float coneDeg, const Expr& coneDir, const Expr& tangent, const Expr& bitangent);
			Expr UniformSampleSphere(const Expr& samples);
			Expr CosineSampleHemisphere(const Expr& samples);
			inline float UniformSpherePdf() { return float(Math::EDX_INV_4PI); }
			inline float UniformConePdf(float cosThetaMax) {
				if (cosThetaMax == 1.0f)
					return 1.0f;
				return 1.0f / (2.0f * float(Math::EDX_PI) * (1.0f - cosThetaMax));
			}

			Expr SquareToUniformDiskConcentric(const Expr& samples);
		}
		void CoordinateSystem(const Expr& normal, Expr* pTangent, Expr* pBitangent);
		inline Expr PowerHeuristics(const Expr& nf, const Expr& fpdf, const Expr& ng, const Expr& gpdf)
		{
			auto f = nf * fpdf;
			auto g = ng * gpdf;
			return Square(f) / (Square(f) + Square(g) + Scalar(1e-4f));
		}

		Expr SphericalTheta(const Expr& vec);
		Expr SphericalPhi(const Expr& vec);
		Expr SphericalDir(const Expr& sinTheta, const Expr& cosTheta, const Expr& phi);

		Tensorf MatrixToTensor(const Matrix& mat);
		Matrix TensorToMatrix(const Tensorf& m);
		Tensorf Vector3ToTensor(const Vector3& v);
		Tensorf Vector4ToTensor(const Vector4& v);
		Tensorf CalcRotateAxisAngle(const Tensorf& fAngle, const Tensorf& vAxis);
		Tensorf CalcTransform(const Tensorf& vPos, const Tensorf& vScl, const Tensorf& vRot);
		Tensorf CalcInvTransform(const Tensorf& vPos, const Tensorf& vScl, const Tensorf& vRot);

		Expr NaNToZero(const Expr& val);
		void AccumulateGradsAndReleaseGraph();

		void RayIntersectAD(const Expr& rayDir, const Expr& rayOrg,
							const Expr& p0, const Expr& e1, const Expr& e2,
							Expr& u, Expr& v, Expr& t);
		bool SaveEXR(const float* rgb, int width, int height, const char* outfilename, bool isTransposed = false);
		void ExportDerivative(const Tensorf& val, int indexParam, std::vector<Tensorf>& deriv);

		void UtilityTest();
	}
}