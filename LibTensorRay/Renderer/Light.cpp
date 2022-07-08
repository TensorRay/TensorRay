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

#include "Light.h"

namespace EDX
{
	namespace TensorRay
	{
		PointLight::PointLight(const float intens, const Tensorf& pos)
		{
			mIntensity = intens;
			// TODO: Check whether this works without setting vector axis
			mPos = pos;
		}

		void PointLight::Eval(const Intersection& isect, const Tensorf& samples, Ray& neeRays, Tensorf& intens, Tensorf& pdf) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			const int numRays = isect.mPosition->GetShape()[0];

			neeRays.mNumRays = numRays;
			neeRays.mOrg = isect.mPosition;
			neeRays.mDir = mPos - isect.mPosition;
			neeRays.mMin = Ones(numRays) * Scalar(1e-2f);
			neeRays.mMax = Ones(numRays) * Scalar(1e32f);

			intens = mIntensity;
			pdf = Scalar(1.0f);

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void PointLight::Emit(const Tensorf& dir, Tensorf& emitted) const
		{
			const int numRays = dir.GetShape(1);
			emitted.Resize(Shape({ numRays }, VecType::Vec3));
		}

		Tensorf PointLight::Pdf(const Tensorf& dir) const
		{
			return Scalar(0.0f);
		}

		DirectionalLight::DirectionalLight(const float intens, const Tensorf& dir, const float coneDeg)
		{
			mIntensity = intens;
			mDir = dir;
			mConeDegree = coneDeg;
		}

		void DirectionalLight::Eval(const Intersection& isect, const Tensorf& samples, Ray& neeRays, Tensorf& intens, Tensorf& pdf) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			const int numRays = isect.mPosition->GetShape()[0];

			neeRays.mNumRays = numRays;
			neeRays.mOrg = isect.mPosition;

			auto dir = VectorNormalize(mDir);
			Expr tangent, bitangent;
			CoordinateSystem(dir, &tangent, &bitangent);
			neeRays.mDir = Sampling::UniformSampleCone(samples, mConeDegree, dir, tangent, bitangent);
			neeRays.mMin = Ones(numRays) * Scalar(1e-2f);
			neeRays.mMax = Ones(numRays) * Scalar(1e32f);

			intens = mIntensity;
			pdf = Scalar(Sampling::UniformConePdf(Math::Cos(Math::ToRadians(mConeDegree))));

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void DirectionalLight::Emit(const Tensorf& dir, Tensorf& emitted) const
		{
			const int numRays = dir.GetShape(1);
			emitted.Resize(Shape({ numRays }, VecType::Vec3));
		}

		Tensorf DirectionalLight::Pdf(const Tensorf& dir) const
		{
			return Scalar(Sampling::UniformConePdf(Math::Cos(Math::ToRadians(mConeDegree))));
		}

		AreaLight::AreaLight(const Vector3& intens, int shape_id)
		{
			mInfo.mIntensity.push_back(intens);
			mInfo.mShapeId.push_back(shape_id);
			mInvTotArea = 0.f;
		}

		void AreaLight::Append(const Vector3& intens, int shape_id)
		{
			mInfo.mIntensity.push_back(intens);
			mInfo.mShapeId.push_back(shape_id);
		}

		Expr AreaLight::Eval(const Intersection& isect, const Expr& wi) const
		{
			Expr reflect = VectorDot(wi, isect.mGeoNormal) > Zeros(1);
			return IndexedRead(mIntensity, isect.mEmitterId, 0) * Ones(wi->GetShape()) * reflect;
		}

		Expr AreaLight::Eval(const PositionSample& record, const Expr& wi) const
		{
			Expr reflect = VectorDot(wi, record.n) > Zeros(1);
			return IndexedRead(mIntensity, record.lightId, 0) * Ones(wi->GetShape()) * reflect;
		}

		void AreaLight::Sample(const Expr& samples, PositionSample& lightP) const
		{
			Tensorf pdf;
			auto samples_x = X(samples);
			auto samples_y = Y(samples);
			Tensori triangleID;
			mpLightDistrb->SampleDiscrete(samples_x, &triangleID, &pdf);
			auto sample_reuse = mpLightDistrb->ReuseSample(samples_x, triangleID);
			auto t = Sqrt(sample_reuse);
			// Convert a uniformly distributed square sample into barycentric coordinate
			auto baryU = Scalar(1.0f) - t;
			auto baryV = t * samples_y;
			auto triId0 = Scalar(3) * triangleID;
			auto triId1 = Scalar(3) * triangleID + Scalar(1);
			auto triId2 = Scalar(3) * triangleID + Scalar(2);
			auto indices_pos0 = IndexedRead(mIndexPosBuffer, triId0, 0);
			auto indices_pos1 = IndexedRead(mIndexPosBuffer, triId1, 0);
			auto indices_pos2 = IndexedRead(mIndexPosBuffer, triId2, 0);
			auto position0 = IndexedRead(mPositionBuffer, indices_pos0, 0);
			auto position1 = IndexedRead(mPositionBuffer, indices_pos1, 0);
			auto position2 = IndexedRead(mPositionBuffer, indices_pos2, 0);
			auto baryW = Scalar(1.0f) - baryU - baryV;
			auto areaAD = IndexedRead(mTriangleAreaBuffer, triangleID, 0);
			lightP.mNumSample = samples->GetShape()[0];
			lightP.p = baryW * position0 + baryU * position1 + baryV * position2;
			lightP.n = IndexedRead(mFaceNormalBuffer, triangleID, 0);
			lightP.J = areaAD / Detach(areaAD);
			lightP.pdf = pdf / Detach(areaAD);
			lightP.lightId = IndexedRead(mLightIdBuffer, triangleID, 0);
		}

		Expr AreaLight::Pdf(const Expr& refP, const Intersection& isect) const
		{
			Shape shape = isect.mTriangleId->GetShape();
			Expr triangleId = IndexedRead(mTriIdToEmitTriIdBuffer, isect.mTriangleId, 0);
			Expr pdf = IndexedRead(mpLightDistrb->mPDF, triangleId, 0) / Scalar(mpLightDistrb->mIntegralVal * mpLightDistrb->mCDF.LinearSize());
			Expr area = IndexedRead(mTriangleAreaBuffer, triangleId, 0);
			return Detach(pdf / area);
		}

		//EnvironmentLight::EnvironmentLight(const char* path)
		//	: Light()
		//{
		//	float* pMap = Bitmap::ReadFromFile<float>(path, &mWidth, &mHeight, &mChannels, 3);

		//	mMap.Assign(pMap, { mHeight * mWidth, mChannels });

		//	// Transpose it into channels x texels
		//	mMap = Tensorf::Transpose(mMap);
		//	mMap = mMap.Reshape(mChannels, mHeight * mWidth);

		//	Tensorf luminance = Luminance(mMap, true);
		//	luminance = luminance.Reshape(mHeight, mWidth);

		//	Tensorf yCoord = Tensorf::ArrayRange(mHeight);
		//	Tensorf sinTheta = Sin(Scalar(float(Math::EDX_PI)) * (yCoord + Scalar(0.5f)) / Scalar(float(mHeight)));
		//	sinTheta = sinTheta.Reshape(mHeight, 1);

		//	luminance *= sinTheta;
		//	mDist = make_unique<Distribution2D>(luminance);

		//	mMap = mMap.Reshape(mChannels, mHeight, mWidth);

		//	Memory::SafeDeleteArray(pMap);
		//}

		//void EnvironmentLight::Eval(const Intersection& isect, const Tensorf& samples, Ray& neeRays, Tensorf& intens, Tensorf& pdf) const
		//{
		//	nvtxRangePushA(__FUNCTION__);

		//	const int numRays = isect.mPosition.GetShape(1);

		//	neeRays.mNumRays = numRays;
		//	neeRays.mOrg = isect.mPosition;
		//	Tensorf dirU, dirV;
		//	mDist->SampleContinuous(samples, &dirU, &dirV, &pdf);
		//	dirU = dirU.Reshape(1, numRays);
		//	dirV = dirV.Reshape(1, numRays);
		//	auto phi = dirU * Scalar(float(Math::EDX_TWO_PI));
		//	auto theta = dirV * Scalar(float(Math::EDX_PI));
		//	auto sinTheta = Sin(theta);
		//	neeRays.mDir = SphericalDir(sinTheta, Cos(theta), phi);
		//	neeRays.mMin = Ones(1, numRays) * Scalar(1e-2f);
		//	neeRays.mMax = Ones(1, numRays) * Scalar(1e32f);

		//	Emit(neeRays.mDir, intens);

		//	pdf = Where(sinTheta != Zeros(1), pdf / (Scalar(2.0f * float(Math::EDX_PI) * float(Math::EDX_PI)) * sinTheta), Zeros(1));

		//	nvtxRangePop();

		//}

		//void EnvironmentLight::Emit(const Tensorf& dir, Tensorf& emitted) const
		//{
		//	nvtxRangePushA(__FUNCTION__);

		//	auto phi = SphericalPhi(dir);
		//	auto theta = SphericalTheta(dir);

		//	auto u = phi / Scalar(float(Math::EDX_TWO_PI));
		//	auto v = theta / Scalar(float(Math::EDX_PI));

		//	auto offsetX = u * Scalar(mWidth - 1);
		//	auto offsetY = v * Scalar(mHeight - 1);

		//	const int numRays = dir.GetShape(1);
		//	emitted.Resize(3, numRays);
		//	Tensorf::IndexedRead(mMap, emitted, offsetY, offsetX, 1, 2);

		//	nvtxRangePop();
		//}

		//Tensorf EnvironmentLight::Pdf(const Tensorf& dir) const
		//{
		//	nvtxRangePushA(__FUNCTION__);

		//	auto phi = SphericalPhi(dir);
		//	auto theta = SphericalTheta(dir);

		//	auto u = phi / Scalar(float(Math::EDX_TWO_PI));
		//	auto v = theta / Scalar(float(Math::EDX_PI));

		//	Tensorf ret = mDist->Pdf(u, v);

		//	nvtxRangePop();

		//	return ret;
		//}
	}
}