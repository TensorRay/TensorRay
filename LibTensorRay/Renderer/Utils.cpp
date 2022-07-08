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

#include "Utils.h"
#define TINYEXR_USE_MINIZ 0
#include "miniz.h"
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace EDX
{
	namespace TensorRay
	{
		namespace Sampling
		{
			Expr UniformSampleCone(const Expr& samples, const float coneDeg, const Expr& coneDir, const Expr& tangent, const Expr& bitangent)
			{
				auto samples0 = X(samples);
				auto samples1 = Y(samples);
				auto phi = Scalar(2.0f * float(Math::EDX_PI)) * samples0;
				const float cosThetaMax = Math::Cos(Math::ToRadians(coneDeg));
				auto cosTheta = (Scalar(1.0f) - samples1) * Scalar(cosThetaMax) + samples1;
				auto sinTheta = Sqrt(Maximum(Scalar(1.0f) - cosTheta * cosTheta, Scalar(1e-4f)));
				auto cosPhi = Cos(phi);
				auto sinPhi = Sin(phi);

				return cosPhi * sinTheta * tangent + sinPhi * sinTheta * bitangent + cosTheta * coneDir;
			}

			Expr UniformSampleSphere(const Expr& samples)
			{
				auto samples0 = X(samples);
				auto samples1 = Y(samples);
				auto z = Ones(1) - Scalar(2.0f) * samples0;
				auto r = Sqrt(Maximum(Scalar(1e-4f), Ones(1) - Square(z)));
				auto phi = Scalar(2.0f * float(Math::EDX_PI)) * samples1;
				auto x = r * Cos(phi);
				auto y = r * Sin(phi);

				return MakeVector3(x, y, z, 0);
			}

			Expr CosineSampleHemisphere(const Expr& samples)
			{
				auto samples0 = X(samples);
				auto samples1 = Y(samples);

				auto r1 = Scalar(2.0f) * samples0 - Scalar(1.0f);
				auto r2 = Scalar(2.0f) * samples1 - Scalar(1.0f);

				auto cond = (r1 * r1) > (r2 * r2);
				auto r = Where(cond, r1, r2);
				auto phi = Where(
					cond,
					Scalar(float(Math::EDX_PI_4)) * (r2 / r1),
					Scalar(float(Math::EDX_PI_2)) - (r1 / r2) * Scalar(float(Math::EDX_PI_4))
				);

				auto condZero = (r1 == Zeros(1)) && (r2 == Zeros(1));
				auto r_ = Where(condZero, Zeros(1), r);
				auto phi_ = Where(condZero, Zeros(1), phi);

				auto x = r_ * Cos(phi_);
				auto y = r_ * Sin(phi_);
				auto z = Sqrt(Maximum(Scalar(1e-4f), Scalar(1.0f) - Square(x) - Square(y)));

				return MakeVector3(x, y, z, 0);
			}

			Expr SquareToUniformDiskConcentric(const Expr& samples)
			{
				auto x = Scalar(2.0f) * X(samples) - Scalar(1.0f);
				auto y = Scalar(2.0f) * Y(samples) - Scalar(1.0f);

				auto is_zero = (x == Scalar(0.0f) && y == Scalar(0.0f));
				auto quadrant_1_or_3 = (Abs(x) < Abs(y));

				auto r = Where(quadrant_1_or_3, y, x);
				auto rp = Where(quadrant_1_or_3, x, y);

				auto phi_ = Scalar(0.25f * float(Math::EDX_PI)) * rp / r;
				auto phi = Where(is_zero, Scalar(0.0f),
					Where(quadrant_1_or_3, Scalar(0.5f * float(Math::EDX_PI)) - phi_, phi_));

				auto res_x = r * Cos(phi);
				auto res_y = r * Sin(phi);
				return MakeVector2(res_x, res_y);
			}
		}

		void CoordinateSystem(const Expr& normal, Expr* pTangent, Expr* pBitangent)
		{
			Assert(normal->GetShape().VectorSize() == 3);
			auto normalX = X(normal);
			auto normalY = Y(normal);
			auto normalZ = Z(normal);

			auto cond = Abs(normalX) > Abs(normalY);

			auto invLen = Where(
				cond,
				Inv(Sqrt(Maximum(Square(normalX) + Square(normalZ), Scalar(1e-3f)))),
				Inv(Sqrt(Maximum(Square(normalY) + Square(normalZ), Scalar(1e-3f))))
			);

			*pTangent = Where(
				cond,
				MakeVector3(-normalZ * invLen, Zeros(normalZ->GetShape()), normalX * invLen),
				MakeVector3(Zeros(normalZ->GetShape()), normalZ * invLen, -normalY * invLen)
			);

			*pBitangent = VectorCross(normal, *pTangent);
		}

		Expr SphericalTheta(const Expr& vec)
		{
			Assert(vec->GetShape().VectorSize() == 3);

			return Acos(Y(vec));
		}

		Expr SphericalPhi(const Expr& vec)
		{
			Assert(vec->GetShape().VectorSize() == 3);

			auto p = Atan2(Z(vec), X(vec));
			auto neg = p < Scalar(0.0f);
			return Where(neg, p + Scalar(2.0f * 3.141592653f), p);
		}

		Expr SphericalDir(const Expr& sinTheta, const Expr& cosTheta, const Expr& phi)
		{
			auto x = sinTheta * Cos(phi);
			auto y = cosTheta;
			auto z = sinTheta * Sin(phi);
			return MakeVector3(x, y, z, 0);
		}
		
		Tensorf MatrixToTensor(const Matrix& mat)
		{
			Tensorf ret;
			ret.Assign((float*)mat.m, { 4, 4 });

			return ret;
		}

		Matrix TensorToMatrix(const Tensorf& m)
		{
			Assert(m.GetShape(0) == 4);
			Assert(m.VectorSize() == 4);
			Matrix ret(m.Get(0), m.Get(1), m.Get(2), m.Get(3),
				m.Get(4), m.Get(5), m.Get(6), m.Get(7),
				m.Get(8), m.Get(9), m.Get(10), m.Get(11),
				m.Get(12), m.Get(13), m.Get(14), m.Get(15)
			);
			return ret;
		}

		Tensorf Vector3ToTensor(const Vector3& v)
		{
			Tensorf ret = Tensorf({ v });

			return ret;
		}

		Tensorf Vector4ToTensor(const Vector4& v)
		{
			Tensorf ret = Tensorf({ v });

			return ret;
		}

        Tensorf CalcRotateAxisAngle(const Tensorf& fAngle, const Tensorf& vAxis)
        {
            Assert(vAxis.VectorSize() == 3);

            Expr axis = VectorNormalize(MakeVector3(X(vAxis), Y(vAxis), Z(vAxis)));
			Expr angle = fAngle; //* Scalar(3.14159265358979323846f / 180.f);
            Expr s = Sin(angle);
            Expr c = Cos(angle);

            Tensorf matRotate = { Vector4::ZERO, Vector4::ZERO, Vector4::ZERO, Vector4::UNIT_W };
            auto c0 = MakeVector4(
                X(axis) * X(axis) + (Scalar(1.f) - X(axis) * X(axis)) * c,
                X(axis) * Y(axis) * (Scalar(1.f) - c) + Z(axis) * s,
                X(axis) * Z(axis) * (Scalar(1.f) - c) - Y(axis) * s,
                Zeros(1)
            );
            matRotate = matRotate + IndexedWrite(c0, { 0 }, Shape({ 4 }, VecType::Vec4), 0);
            auto c1 = MakeVector4(
                X(axis) * Y(axis) * (Scalar(1.f) - c) - Z(axis) * s,
                Y(axis) * Y(axis) + (Scalar(1.f) - Y(axis) * Y(axis)) * c,
                Y(axis) * Z(axis) * (Scalar(1.f) - c) + X(axis) * s,
                Zeros(1)
            );
            matRotate = matRotate + IndexedWrite(c1, { 1 }, Shape({ 4 }, VecType::Vec4), 0);
            auto c2 = MakeVector4(
                X(axis) * Z(axis) * (Scalar(1.f) - c) + Y(axis) * s,
                Y(axis) * Z(axis) * (Scalar(1.f) - c) - X(axis) * s,
                Z(axis) * Z(axis) + (Scalar(1.f) - Z(axis) * Z(axis)) * c,
                Zeros(1)
            );
            matRotate = matRotate + IndexedWrite(c2, { 2 }, Shape({ 4 }, VecType::Vec4), 0);
            return matRotate;
        }

		Tensorf CalcTransform(const Tensorf& vPos, const Tensorf& vScl, const Tensorf& vRot)
		{
			Assert(vPos.VectorSize() == 3);
			Assert(vScl.VectorSize() == 3);
			Assert(vRot.VectorSize() == 3);

			Tensorf matTranslate = { Vector4::UNIT_X, Vector4::UNIT_Y, Vector4::UNIT_Z, Vector4::UNIT_W };
			matTranslate = matTranslate + IndexedWrite(MakeVector4(X(vPos), Y(vPos), Z(vPos), Zeros(1)), { 3 }, Shape({ 4 }, VecType::Vec4), 0);
			Tensorf matScale = { Vector4::ZERO, Vector4::ZERO, Vector4::ZERO, Vector4::UNIT_W };
			matScale = matScale + IndexedWrite(MakeVector4(X(vScl), Zeros(1), Zeros(1), Zeros(1)), { 0 }, Shape({ 4 }, VecType::Vec4), 0);
			matScale = matScale + IndexedWrite(MakeVector4(Zeros(1), Y(vScl), Zeros(1), Zeros(1)), { 1 }, Shape({ 4 }, VecType::Vec4), 0);
			matScale = matScale + IndexedWrite(MakeVector4(Zeros(1), Zeros(1), Z(vScl), Zeros(1)), { 2 }, Shape({ 4 }, VecType::Vec4), 0);

			// TODO: differentiate the rotation vector
			Tensorf matRotate = MatrixToTensor(Matrix::YawPitchRow(vRot.Get(1), vRot.Get(0), vRot.Get(2)));

			return Dot(matTranslate, Dot(matScale, matRotate));
		}

		Tensorf CalcInvTransform(const Tensorf& vPos, const Tensorf& vScl, const Tensorf& vRot)
		{
			Assert(vPos.VectorSize() == 3);
			Assert(vScl.VectorSize() == 3);
			Assert(vRot.VectorSize() == 3);

			Tensorf matTranslate = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
			matTranslate = matTranslate + IndexedWrite(MakeVector4(X(-vPos), Y(-vPos), Z(-vPos), Zeros(1)), { 3 }, { 4, 4 }, 0);
			Tensorf matScale = { {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1} };
			Tensorf invScale = Scalar(1.0f) / vScl;
			matScale = matScale + IndexedWrite(MakeVector4(X(invScale), Zeros(1), Zeros(1), Zeros(1)), { 0 }, { 4, 4 }, 0);
			matScale = matScale + IndexedWrite(MakeVector4(Zeros(1), Y(invScale), Zeros(1), Zeros(1)), { 1 }, { 4, 4 }, 0);
			matScale = matScale + IndexedWrite(MakeVector4(Zeros(1), Zeros(1), Z(invScale), Zeros(1)), { 2 }, { 4, 4 }, 0);

			// Todo: differentiate the rotation vector
			Tensorf matRotate = MatrixToTensor(Matrix::Inverse(Matrix::YawPitchRow(vRot.Get(1), vRot.Get(0), vRot.Get(2))));

			return Dot(matRotate, Dot(matScale, matTranslate));
		}

		Expr NaNToZero(const Expr& val)
		{
			Expr isFinite = IsFinite(X(val)) && IsFinite(Y(val)) && IsFinite(Z(val));
			return Where(isFinite, val, Zeros(1));
		}

		void AccumulateGradsAndReleaseGraph()
		{
			for (auto& it : ParameterPool::GetHandle())
			{
				Tensorf* pTensor = dynamic_cast<Tensorf*>(it.ptr.get());
				pTensor->EvalAndReleaseGradGraph();
			}
		}

		void RayIntersectAD(const Expr& rayDir, const Expr& rayOrg, const Expr& p0, const Expr& e1, const Expr& e2, Expr& u, Expr& v, Expr& t)
		{
			auto h = VectorCross(rayDir, e2);
			auto f = Scalar(1.0f) / VectorDot(e1, h);
			auto s = rayOrg - p0;
			u = f * VectorDot(s, h);
			auto q = VectorCross(s, e1);
			v = f * VectorDot(rayDir, q);
			t = f * VectorDot(e2, q);
		}

		bool SaveEXR(const float* rgb, int width, int height, const char* outfilename, bool isTransposed) 
		{
			EXRHeader header;
			InitEXRHeader(&header);

			EXRImage image;
			InitEXRImage(&image);

			image.num_channels = 3;

			std::vector<float> images[3];
			images[0].resize(width * height);
			images[1].resize(width * height);
			images[2].resize(width * height);

			if (!isTransposed)
			{
				// Split RGBRGBRGB... into R, G and B layer
				for (int i = 0; i < width * height; i++) {
					images[0][i] = rgb[3 * i + 0];
					images[1][i] = rgb[3 * i + 1];
					images[2][i] = rgb[3 * i + 2];
				}
			}
			else
			{
				// Split R..RG..GB..B into R, G and B layer
				for (int i = 0; i < width * height; i++) {
					images[0][i] = rgb[i];
					images[1][i] = rgb[i + width * height];
					images[2][i] = rgb[i + width * height * 2];
				}
			}

			float* image_ptr[3];
			image_ptr[0] = &(images[2].at(0)); // B
			image_ptr[1] = &(images[1].at(0)); // G
			image_ptr[2] = &(images[0].at(0)); // R

			image.images = (unsigned char**)image_ptr;
			image.width = width;
			image.height = height;

			header.num_channels = 3;
			header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
			// Must be (A)BGR order, since most of EXR viewers expect this channel order.
			strncpy_s(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
			strncpy_s(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
			strncpy_s(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

			header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
			header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
			for (int i = 0; i < header.num_channels; i++) {
				header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
				header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
			}

			const char* err = NULL; // or nullptr in C++11 or later.
			int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
			if (ret != TINYEXR_SUCCESS) {
				fprintf(stderr, "Save EXR err: %s\n", err);
				FreeEXRErrorMessage(err); // free's buffer for an error message
				return ret;
			}

			free(header.channels);
			free(header.pixel_types);
			free(header.requested_pixel_types);

			return ret;
		}

		void ExportDerivative(const Tensorf& val, int indexParam, std::vector<Tensorf>& deriv)
		{
			int numElements = val.LinearSize();
			int numDeriv = deriv.size();
			for (int iElement = 0; iElement < numElements; iElement++)
			{
				Tensorf mask = Zeros(val.GetShape());
				mask.Set(iElement, 1);
				val.Backward(mask);
				Tensorf* pTensor = dynamic_cast<Tensorf*>(ParameterPool::GetHandle()[indexParam].ptr.get());
				Assert(pTensor->LinearSize() == numDeriv);
				Tensorf grad = pTensor->Grad();
				if (grad.LinearSize() == numDeriv) 
				{
					for (int j = 0; j < numDeriv; j++)
						deriv[j].Set(iElement, grad.Get(j));
					std::cout << "Derivative computation [Element  " << iElement << "/" << numElements << "] completed\r" << std::flush;
				}
				for (auto& it : ParameterPool::GetHandle())
					it->Update(0.0f);		// Clear Grad
			}
			std::cout << std::endl;
		}

		void UtilityTest()
		{
			{
				std::cout << "Utility Test 1: " << std::endl;
				Tensorf vecTrans({ 0.0f, 3.0f, 0.0f });
				vecTrans = vecTrans.Reshape(Shape({ 1 }, VecType::Vec3));
				vecTrans.SetRequiresGrad(true);
				Tensorf vecScale = Tensorf({ 1.0f, 1.0f, 1.0f });
				vecScale = vecScale.Reshape(Shape({ 1 }, VecType::Vec3));
				Tensorf vecRotate = Tensorf({0.0f, 0.0f, 0.0f });
				vecRotate = vecRotate.Reshape(Shape({ 1 }, VecType::Vec3));
				Tensorf target = CalcTransform(vecTrans, vecScale, vecRotate);

				ValidateBackwardDiff(target, vecTrans);
				ValidateForwardDiff(target, vecTrans);

				ParameterPool::GetHandle().clear();
			}
			{
				std::cout << "Test 3:\n";

				Tensorf A = Tensorf::LinSpace(0, 10, 10, true/*requires_grad*/);

				// Expression built from (nested) function

				// Forward evaluation, this will take more than 1 cuda kernel calls because of reduction (Sum) is involved when calculating standard deviation
				// Parts of the evaluation will still get evaluated in a fused cuda kernel when viable
				Tensorf std = StandardDeviation(A);
				std.Backward(Ones(std.GetShape()));

				std::cout << "results: " << std << "\n";

				Tensorf diff = A.Grad();
				std::cout << "autodiff derivative: " << diff << "\n";

				Tensorf numericalDiff = NumericalGradientEval(std, A);
				std::cout << "numerical derivative: " << numericalDiff << "\n\n";

				ValidateForwardDiff(std, A);

				/*
				results: 3.19143
				autodiff derivative: -0.15667 -0.121854 -0.0870388 -0.0522233 -0.0174078 0.0174078 0.0522233 0.0870388 0.121854 0.15667
				numerical derivative: -0.15676 -0.121832 -0.0870228 -0.0522137 -0.0172853 0.0172853 0.0522137 0.087142 0.121951 0.15676
				*/

				ParameterPool::GetHandle().clear();
			}
			{
				std::cout << "Test 11:\n";
				Tensorf A = Ones(Shape({ 1 }, VecType::Vec3));;
				A.SetRequiresGrad(true);

				Tensorf C = VectorNormalize(A);
				std::cout << "results: " << C << "\n";

				ValidateBackwardDiff(C, A);
				ValidateForwardDiff(C, A);

				ParameterPool::GetHandle().clear();
			}
			{
				std::cout << "Test 12:\n";

				Tensorf A = Tensorf::ArrayRange(3, 6, 1, true).Reshape(Shape({ 1 }, VecType::Vec3));
				Tensorf B = Tensorf::ArrayRange(6, 9, 1, false).Reshape(Shape({ 1 }, VecType::Vec3));

				Tensorf C = VectorCross(A, B);
				std::cout << "results: " << C << "\n";

				ValidateBackwardDiff(C, A);
				ValidateForwardDiff(C, A);

				ParameterPool::GetHandle().clear();
			}

			{
				std::cout << "Test 24:\n";


				Tensorf A = Tensorf::RandomFloat(Shape({ 10 }, VecType::Vec3));
				Tensorf mat = Tensorf::RandomFloat(4, 4);
				mat.SetRequiresGrad(true);

				Tensorf transformed = TransformPoints(A, mat);

				std::cout << "results: " << transformed << "\n";

				ValidateBackwardDiff(transformed, mat);
				ValidateForwardDiff(transformed, mat);

				ParameterPool::GetHandle().clear();
			}
			{
				// ------------------------------------------------------------------
				// World to local
				// ------------------------------------------------------------------
				Intersection isect;
				Tensorf normal = Tensorf::RandomFloat(Shape({ 5 }, VecType::Vec3));
				normal.SetRequiresGrad(true);
				isect.mNormal = VectorNormalize(normal);
				CoordinateSystem(isect.mNormal, &isect.mTangent, &isect.mBitangent);

				Tensorf vec = Tensorf::RandomFloat(Shape({ 5 }, VecType::Vec3));

				Tensorf localVec = isect.WorldToLocal(vec);

				std::cout << "World to local: " << localVec << "\n";

				ValidateBackwardDiff(localVec, normal);
				ValidateForwardDiff(localVec, normal);

				normal.ClearGrad();

				// ------------------------------------------------------------------
				// Local to world
				// ------------------------------------------------------------------

				Tensorf worldVec = isect.LocalToWorld(vec);
				std::cout << "Local to world: " << worldVec << "\n";

				ValidateBackwardDiff(worldVec, normal);
				ValidateForwardDiff(worldVec, normal);

				normal.ClearGrad();

				// ------------------------------------------------------------------
				// Tangent
				// ------------------------------------------------------------------
				Tensorf tangent = isect.mTangent;

				std::cout << "Tangent: " << isect.mTangent << "\n";

				ValidateBackwardDiff(tangent, normal);
				ValidateForwardDiff(tangent, normal);

				normal.ClearGrad();


				// ------------------------------------------------------------------
				// Bitangent
				// ------------------------------------------------------------------
				Tensorf bitangent = isect.mBitangent;

				std::cout << "Bitangent: " << bitangent << "\n";

				ValidateBackwardDiff(tangent, normal);
				ValidateForwardDiff(isect.mBitangent, normal);

				normal.ClearGrad();

				ParameterPool::GetHandle().clear();
			}
			{
				// ------------------------------------------------------------------
				// Ray Triangle intersect
				// ------------------------------------------------------------------

				Tensorf position0 = Tensorf::RandomFloat(Shape({ 5 }, VecType::Vec3));

				std::cout << "Utility Test 1: " << std::endl;
				Tensorf vecTrans = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec3));
				vecTrans.SetRequiresGrad(true);
				Tensorf vecScale = Ones(Shape({ 1 }, VecType::Vec3));
				Tensorf vecRotate = Zeros(Shape({ 1 }, VecType::Vec3));
				Tensorf mat = CalcTransform(vecTrans, vecScale, vecRotate);
				auto pos0 = TransformPoints(position0, mat);


				Tensorf dir = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec3));
				Tensorf org = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec3));
				dir.SetRequiresGrad(true);
				org.SetRequiresGrad(true);

				auto p0 = IndexedRead(pos0, { 0 }, 0);
				auto p1 = IndexedRead(pos0, { 1 }, 0);
				auto p2 = IndexedRead(pos0, { 2 }, 0);

				Expr u, v, w, t;
				RayIntersectAD(dir, org, p0, p1 - p0, p2 - p0, u, v, t);

				Tensorf tU = u;

				std::cout << "u: " << tU << "\n";

				ValidateBackwardDiff(tU, vecTrans);
				ValidateForwardDiff(tU, vecTrans);

				vecTrans.ClearGrad();

				ParameterPool::GetHandle().clear();
			}
			{
				Tensorf dir = Tensorf::RandomFloat(Shape({ 1 }, VecType::Vec3));
				dir.SetRequiresGrad(true);

				auto normal = VectorNormalize(dir);

				Expr tangent, bitangent;
				CoordinateSystem(normal, &tangent, &bitangent);

				Tensorf samples = Tensorf::RandomFloat(Shape({ 120 }, VecType::Vec2));
				Tensorf out = Sampling::UniformSampleCone(samples, 5.0f, normal, tangent, bitangent);

				std::cout << "Sampled dir: " << out << "\n";
				ValidateBackwardDiff(out, dir);
				ValidateForwardDiff(out, dir);

				dir.ClearGrad();

				ParameterPool::GetHandle().clear();
			}
		}
	}
}