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

#include "Camera.h"

namespace EDX
{
	namespace TensorRay
	{
		Expr TentFilter::Eval(const Expr& x) const
		{
			auto dist2center = x - Scalar(0.5f);
			auto k = Scalar(0.5f / mPadding);
			auto val = Scalar(0.5f + mPadding) * k - Abs(k * dist2center);
			return Maximum(Minimum(val, Scalar(1.0f)), Scalar(0.0f));
		}

		Expr TentFilter::Sample(const Expr& rnd) const
		{
			auto ret = Zeros(rnd->GetShape());
			float segLimit0 = mPadding;
			float segLimit1 = mPadding + 1.0f - 2.0f * mPadding;
			IndexMask mask0 = rnd <= Scalar(segLimit0);
			if (mask0.sum > 0)
			{
				auto val = Sqrt(Scalar(4.0f * mPadding) * Mask(rnd, mask0, 0)) - Scalar(mPadding);
				ret = ret + IndexedWrite(val, mask0.index, rnd->GetShape(), 0);
			}
			IndexMask mask1 = rnd > Scalar(segLimit0) && rnd <= Scalar(segLimit1);
			if (mask1.sum > 0)
			{
				ret = ret + IndexedWrite(Mask(rnd, mask1, 0), mask1.index, rnd->GetShape(), 0);
			}
			IndexMask mask2 = rnd > Scalar(segLimit1);
			if (mask2.sum > 0)
			{
				auto val = Mask(rnd, mask2, 0) - Scalar(segLimit1);
				val = -Sqrt(Scalar(4.0f * mPadding) * val) + Scalar(mPadding + 1.0f);
				ret = ret + IndexedWrite(val, mask2.index, rnd->GetShape(), 0);
			}
			return ret;
		}

		Expr BoxFilter::Eval(const Expr& x) const
		{
            Expr ret = Where(x >= Scalar(0.f) && x <= Scalar(1.f), Scalar(1.f), Scalar(0.f));
			return ret;
		}

		Expr BoxFilter::Sample(const Expr& rnd) const
		{
			return rnd;
		}

        Expr BoxFilter::Sample(const Expr& rnd, Expr& pdf) const
        {
            auto ret = Sample(rnd);
            pdf = Scalar(1.f);
            return ret;
        }

		void Camera::Init(const Vector3& pos, const Vector3& tar, const Vector3& up, 
						  const int resX, const int resY, 
						  const float FOV /* = 35.0f */, 
						  const float nearClip /* = 1.0f */, 
						  const float farClip /* = 1000.0f */, 
						  const float filterPad /* = 0.5f - EPSILON */)
		{
            EDX::Camera::Init(pos, tar, up, resX, resY, FOV, nearClip, farClip);
            Resize(resX, resY);
            mFilterPad = filterPad;
#if USE_BOX_FILTER
            mFilter = make_unique<BoxFilter>();
#else
            mFilter = make_unique<TentFilter>(filterPad);
#endif
            mRequireGrad = false;
		}

		void Camera::Resize(int width, int height)
		{
			EDX::Camera::Resize(width, height);
			mPosTensor = Vector3ToTensor(mPos);
			mDirTensor = Vector3ToTensor(mDir);
			mViewTensor = MatrixToTensor(mView);
			mViewInvTensor = MatrixToTensor(mViewInv);
			mProjTensor = MatrixToTensor(mProj);
			mScreenToRasterTensor = MatrixToTensor(mScreenToRaster);
			mRasterToCameraTensor = MatrixToTensor(mRasterToCamera);
			mCameraToRasterTensor = MatrixToTensor(mCameraToRaster);
			mRasterToWorldTensor = MatrixToTensor(mRasterToWorld);
			mWorldToRasterTensor = MatrixToTensor(mWorldToRaster);
			Matrix cameraToSample = Matrix::Scale(1.0 / float(mFilmResX), 1.0 / float(mFilmResY), 1.0f) * mCameraToRaster;
			{
				Matrix sampleToCamera = mRasterToCamera * Matrix::Scale(float(mFilmResX), float(mFilmResY), 1.0f);
				auto v00 = Matrix::TransformPoint(Vector4(0.0f, 0.0f, 0.0f, 1.0f), sampleToCamera);
				auto v10 = Matrix::TransformPoint(Vector4(1.0f, 0.0f, 0.0f, 1.0f), sampleToCamera);
				auto v11 = Matrix::TransformPoint(Vector4(1.0f, 1.0f, 0.0f, 1.0f), sampleToCamera);
				auto vc  = Matrix::TransformPoint(Vector4(0.5f, 0.5f, 0.0f, 1.0f), sampleToCamera);
				Vector3 horizontal = Vector3(v00.x/v00.w, v00.y/v00.w, v00.z/v00.w) - Vector3(v10.x/v10.w, v10.y/v10.w, v10.z/v10.w);
				Vector3 vertical = Vector3(v11.x/v11.w, v11.y/v11.w, v11.z/v11.w) - Vector3(v10.x/v10.w, v10.y/v10.w, v10.z/v10.w);
				mImagePlaneLengthX = Math::Length(horizontal);
				mImagePlaneLengthY = Math::Length(vertical);
				mInvArea = 1.0 / (mImagePlaneLengthX * mImagePlaneLengthY) * Math::LengthSquared(Vector3(vc.x/vc.w, vc.y/vc.w, vc.z/vc.w));
			}
			mWorldToSampleTensor = MatrixToTensor(Matrix::Mul(cameraToSample, mView));
			mSampleToCameraTensor = MatrixToTensor(Matrix::Inverse(cameraToSample));
			mResX = width;
			mResY = height;
		}

		void Camera::GenerateRay(Ray& ray) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			const int numRays = mFilmResX * mFilmResY;
			ray.mNumRays = numRays;

			Tensorf cameraSamples = Tensorf::RandomFloat(Shape({ mFilmResX * mFilmResY }, VecType::Vec2));
			auto cameraSamplesPadded = MakeVector4(X(cameraSamples), Y(cameraSamples), Zeros(1), Zeros(1));

			Tensorf grid = (Expr(make_shared<PixelCoordExp>(mFilmResX, mFilmResY)) + cameraSamplesPadded);
			auto camCoord = TransformPointsHomogeneous(grid, mRasterToCameraTensor);

			auto origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
			auto directions_view = VectorNormalize(camCoord);

			auto origins = TransformPoints(origins_view, mViewInvTensor);
			auto directions = TransformVectors(directions_view, mViewInvTensor);

			ray.mOrg = origins;
			ray.mDir = directions;
			ray.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			ray.mPrevPdf = Ones(numRays);
			ray.mSpecular = True(numRays);
			ray.mPixelIdx = Tensori::ArrayRange(numRays);
			ray.mMin = Ones(numRays) * Scalar(1e-4f);
			ray.mMax = Ones(numRays) * Scalar(1e32f);

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void Camera::GenerateRayPair(Ray& rayPrimal, Ray& rayDual) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			const int numRays = mFilmResX * mFilmResY;
			Tensorf rnd = Tensorf::RandomFloat(Shape({ numRays }, VecType::Vec2));
			auto origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
			auto origins = TransformPoints(origins_view, mViewInvTensor);
			rayPrimal.mNumRays = numRays;
			rayPrimal.mOrg = Tensorf(origins);
			rayPrimal.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			rayPrimal.mPrevPdf = Ones(numRays);
			rayPrimal.mSpecular = True(numRays);
			rayPrimal.mPixelIdx = Tensori::ArrayRange(numRays);
			rayPrimal.mRayIdx = Tensori::ArrayRange(numRays);
			rayPrimal.mMin = Ones(numRays) * Scalar(SHADOW_EPSILON);
			rayPrimal.mMax = Ones(numRays) * Scalar(1e32f);
			rayDual.mNumRays = numRays;
			rayDual.mOrg = origins;
			rayDual.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			rayDual.mPrevPdf = Ones(numRays);
			rayDual.mSpecular = True(numRays);
			rayDual.mPixelIdx = Tensori::ArrayRange(numRays);
			rayDual.mRayIdx = Tensori::ArrayRange(numRays);
			rayDual.mMin = Ones(numRays) * Scalar(SHADOW_EPSILON);
			rayDual.mMax = Ones(numRays) * Scalar(1e32f);
			auto px = mFilter->Sample(X(rnd));
			auto py = mFilter->Sample(Y(rnd));
			{
				auto camSamples = MakeVector4(px, py, Zeros(1), Zeros(1));
				Tensorf grid = (Expr(make_shared<PixelCoordExp>(mFilmResX, mFilmResY)) + camSamples);
				auto camCoord = TransformPointsHomogeneous(grid, mRasterToCameraTensor);
				auto directions_view = VectorNormalize(camCoord);
				auto directions = TransformVectors(directions_view, mViewInvTensor);
				rayPrimal.mDir = Tensorf(directions);
			}
			{
				auto camSamples = MakeVector4(Scalar(1.0f) - px, Scalar(1.0f) - py, Zeros(1), Zeros(1));
				Tensorf grid = (Expr(make_shared<PixelCoordExp>(mFilmResX, mFilmResY)) + camSamples);
				auto camCoord = TransformPointsHomogeneous(grid, mRasterToCameraTensor);
				auto directions_view = VectorNormalize(camCoord);
				auto directions = TransformVectors(directions_view, mViewInvTensor);
				rayDual.mDir = Tensorf(directions);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void Camera::GenerateAntitheticRays(Ray& ray, int batchSize, int antitheticSpp) const
		{
			assert(antitheticSpp == 1 || antitheticSpp == 2 || antitheticSpp == 4);
#if USE_PROFILING
            nvtxRangePushA(__FUNCTION__);
#endif
            const int numRays = mFilmResX * mFilmResY * batchSize;
            Tensorf rnd = Tensorf::RandomFloat(Shape({ numRays / antitheticSpp }, VecType::Vec2));
			if (antitheticSpp >= 2)
				rnd = Concat(rnd, rnd, 0);
			if (antitheticSpp >= 4)
				rnd = Concat(rnd, rnd, 0);
            auto origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
            auto origins = TransformPoints(origins_view, mViewInvTensor);

			ray.mNumRays = numRays;
			ray.mOrg = Tensorf(origins);
			ray.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			ray.mPrevPdf = Ones(numRays);
			ray.mSpecular = True(numRays);
			ray.mPixelIdx = Tensori::ArrayRange(numRays) % Scalar(mFilmResX * mFilmResY);
			ray.mRayIdx = Tensori::ArrayRange(numRays);
			ray.mMin = Ones(numRays) * Scalar(SHADOW_EPSILON);
			ray.mMax = Ones(numRays) * Scalar(1e32f);

			auto px = mFilter->Sample(X(rnd));
			auto py = mFilter->Sample(Y(rnd));

			Expr camSamples0 = MakeVector4(px, py, Zeros(1), Zeros(1));
			Expr camSamples1 = MakeVector4(Scalar(1.0f) - px, Scalar(1.0f) - py, Zeros(1), Zeros(1));
			Expr camSamples2 = MakeVector4(Scalar(1.0f) - px, py, Zeros(1), Zeros(1));
			Expr camSamples3 = MakeVector4(px, Scalar(1.0f) - py, Zeros(1), Zeros(1));

			Expr antitheticIndex = Floor(ray.mRayIdx / Scalar(numRays / antitheticSpp));
			Expr camSamples = Where(antitheticIndex == Scalar(0),
				camSamples0,
				Where(antitheticIndex == Scalar(1),
					camSamples1,
					Where(antitheticIndex == Scalar(2),
						camSamples2,
						camSamples3
					)
				)
			);

			Tensorf grid = (Expr(make_shared<PixelCoordExp>(mFilmResX, mFilmResY, batchSize)) + camSamples);
			auto camCoord = TransformPointsHomogeneous(grid, mRasterToCameraTensor);
			auto directions_view = VectorNormalize(camCoord);
			auto directions = TransformVectors(directions_view, mViewInvTensor);
			ray.mDir = Tensorf(directions);
#if USE_PROFILING
            nvtxRangePop();
#endif
		}

		Expr Camera::EvalFilter(const Expr& pixelId, const Intersection& isect) const
		{
#if USE_BOX_FILTER
			// The sample is always in the pixel with pixelId.
			Expr f = Scalar(1.f);
#else
			auto pyIndex = Floor(pixelId / Scalar(mResX));
			auto pxIndex = pixelId - pyIndex * Scalar(mResX);
			auto pProj = TransformPoints(isect.mPosition, mWorldToSampleTensor);
			auto pxProj = X(pProj) * Scalar(mResX);
			auto pyProj = Y(pProj) * Scalar(mResY);
			auto xOffset = pxProj - Detach(pxIndex);
			auto yOffset = pyProj - Detach(pyIndex);
			auto fx = mFilter->Eval(xOffset);
			auto fy = mFilter->Eval(yOffset);
			auto f = fx * fy;
#endif
			auto dir = isect.mPosition - mPosTensor;
			auto dist = VectorLength(dir);
			dir = dir / dist;
			auto cosy = VectorDot(dir, mDirTensor);
			auto cosx = VectorDot(-dir, isect.mGeoNormal);
			auto G = Abs(cosx) / (cosy * cosy * cosy * Square(dist));
			return G * f / (Detach(G) * Detach(f));
		}

		void Camera::GenerateBoundaryRays(const Tensorf& samples, const Tensorf& normal, Ray& rayP, Ray& rayN) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			int numRays = samples.GetShape(1);
			rayP.mNumRays = numRays;
			rayN.mNumRays = numRays;
			auto origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
			auto origins = TransformPoints(origins_view, mViewInvTensor);
			rayP.mOrg = origins;
			rayN.mOrg = origins;
			auto samplesP = samples + Scalar(EDGE_EPSILON) * normal;
			auto camCoordP = TransformPoints(MakeVector3( X(samplesP), Y(samplesP), Zeros(1), 0), mSampleToCameraTensor);
			rayP.mDir = TransformVectors(VectorNormalize(camCoordP), mViewInvTensor);
			auto samplesN = samples - Scalar(EDGE_EPSILON) * normal;
			auto camCoordN = TransformPoints(MakeVector3( X(samplesN), Y(samplesN), Zeros(1), 0), mSampleToCameraTensor);
			rayN.mDir = TransformVectors(VectorNormalize(camCoordN), mViewInvTensor);

			rayP.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			rayP.mPrevPdf = Ones(numRays);
			rayP.mSpecular = True(numRays);
			rayP.mPixelIdx = Floor(Y(samples) * Scalar(mResY)) * Scalar(mResX) + Floor(X(samples) * Scalar(mResX));
			rayP.mRayIdx = Tensori::ArrayRange(numRays);
			rayP.mMin = Ones(numRays) * Scalar(1e-4f);
			rayP.mMax = Ones(numRays) * Scalar(1e32f);

			rayN.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
			rayN.mPrevPdf = Ones(numRays);
			rayN.mSpecular = True(numRays);
			rayN.mPixelIdx = Floor(Y(samples) * Scalar(mResY)) * Scalar(mResX) + Floor(X(samples) * Scalar(mResX));
			rayN.mRayIdx = Tensori::ArrayRange(numRays);
			rayN.mMin = Ones(numRays) * Scalar(1e-4f);
			rayN.mMax = Ones(numRays) * Scalar(1e32f);

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void Camera::GenerateBoundaryRays(const SensorDirectSample& sds, Ray& ray) const
		{
			IndexMask maskValid = IndexMask(sds.isValid);
			int numRays = maskValid.sum;
			ray.mNumRays = numRays;
			auto origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
			auto origins = TransformPoints(origins_view, mViewInvTensor);
			ray.mOrg = origins;
			auto samples = Mask(sds.q, maskValid, 0);
			auto camCoord = TransformPoints(MakeVector3(X(samples), Y(samples), Zeros(1), 0), mSampleToCameraTensor);
			ray.mDir = TransformVectors(VectorNormalize(camCoord), mViewInvTensor);

			ray.mPrevPdf = Ones(numRays);
			ray.mSpecular = True(numRays);

			ray.mThroughput = Mask(sds.sensorVal, maskValid, 0) * Ones(Shape({ numRays }, VecType::Vec3));
			ray.mPixelIdx = Mask(sds.pixelIdx, maskValid, 0);
			ray.mRayIdx = Tensori::ArrayRange(numRays);
			ray.mMin = Ones(numRays) * Scalar(SHADOW_EPSILON);
			ray.mMax = Ones(numRays) * Scalar(1e32f);
		}

		SensorDirectSample Camera::sampleDirect(const Expr& p) const
		{
			SensorDirectSample ret;
			auto pProj = TransformPoints(p, mWorldToSampleTensor);
			auto pxProj = X(pProj);
			auto pyProj = Y(pProj);
			ret.q = MakeVector2(pxProj, pyProj, 0);
			ret.isValid = (pxProj >= Scalar(0.0f)) && (pxProj < Scalar(1.0f)) &&
						  (pyProj >= Scalar(0.0f)) && (pyProj < Scalar(1.0f));
			ret.pixelIdx = Floor(pyProj * Scalar(mResY)) * Scalar(mResX) + Floor(pxProj * Scalar(mResX));
			auto dir = Detach(p) - Detach(mPosTensor);
			auto dist2 = VectorSquaredLength(dir);
			dir = dir / Sqrt(dist2);
			auto cosTheta = VectorDot(Detach(mDirTensor), dir);
			auto invCosTheta = Scalar(1.0f) / cosTheta;
			ret.sensorVal = Scalar(1.0f) / dist2 * invCosTheta * invCosTheta * invCosTheta * Scalar(mInvArea); // // Note: Pow function is not implemented for backward.
			return ret;
		}

		Tensorf Camera::WriteToImage(const Expr& rayContrib, const Expr& pixelCoor) const
		{
			auto ret = Zeros(Shape({ mResX * mResY }, VecType::Vec3));
			auto px = Detach(X(pixelCoor) * Scalar(mResX));
			auto px_index0 = Floor(px);
			auto px_offset0 = px - px_index0;
			auto py = Detach(Y(pixelCoor) * Scalar(mResY));
			auto py_index0 = Floor(py);
			auto py_offset0 = py - py_index0;
#if USE_BOX_FILTER
			auto xIndexValid = px_index0 >= Scalar(0) && px_index0 < Scalar(mResX);
			auto yIndexValid = py_index0 >= Scalar(0) && py_index0 < Scalar(mResY);
            auto contrib = rayContrib * mFilter->Eval(px_offset0) * mFilter->Eval(py_offset0) * xIndexValid * yIndexValid;
            auto pixelIndex = py_index0 * Scalar(mResX) + px_index0;
            IndexMask _nonZero = VectorLength(contrib) > Scalar(0.0f);
            if (_nonZero.sum > 0)
            {
                contrib = Mask(contrib, _nonZero, 0);
                pixelIndex = Mask(pixelIndex, _nonZero, 0);
                ret = ret + IndexedWrite(contrib, pixelIndex, Shape({ mResX * mResY }, VecType::Vec3), 0);
            }
#else
			for (int ix = -1; ix <= 1; ix++)
			{
				auto px_index = px_index0 + Scalar(ix);
				auto px_offset = px_offset0 + Scalar(-ix);
				auto xIndexValid = px_index >= Scalar(0) && px_index < Scalar(mResX);
				for (int iy = -1; iy <= 1; iy++)
				{
					auto py_index = py_index0 + Scalar(iy);
					auto py_offset = py_offset0 + Scalar(-iy);
					auto yIndexValid = py_index >= Scalar(0) && py_index < Scalar(mResY);
					auto contrib = rayContrib * mFilter->Eval(px_offset) * mFilter->Eval(py_offset) * xIndexValid * yIndexValid;
					auto pixelIndex = py_index * Scalar(mResX) + px_index;
					IndexMask _nonZero = VectorLength(contrib) > Scalar(0.0f);
					if (_nonZero.sum > 0)
					{
						contrib = Mask(contrib, _nonZero, 0);
						pixelIndex = Mask(pixelIndex, _nonZero, 0);
						ret = ret + IndexedWrite(contrib, pixelIndex, Shape({ mResX * mResY }, VecType::Vec3), 0);
					}
				}
			}
#endif
			return ret;
		}

		void Camera::GeneratePixelBoundarySamples(const Tensorf& rnd, Tensorf& p0, Tensorf& edge, Tensorf& edge2, Tensorf& pdf) const
		{
			Tensorf rnd4 = rnd * Scalar(4.0f);
			Tensori pixel_boundary_index = Floor(rnd4);

			Expr offset0 = MakeVector4(Zeros(1), rnd4, Zeros(1), Zeros(1));
			Expr offset1 = MakeVector4(rnd4 - Scalar(1.f), Ones(1), Zeros(1), Zeros(1));
			Expr offset2 = MakeVector4(Ones(1), Scalar(3.f) - rnd4, Zeros(1), Zeros(1));
			Expr offset3 = MakeVector4(Scalar(4.f) - rnd4, Zeros(1), Zeros(1), Zeros(1));

            Expr camSamples = Where(pixel_boundary_index == Scalar(0),
                offset0,
                Where(pixel_boundary_index == Scalar(1),
                    offset1,
                    Where(pixel_boundary_index == Scalar(2),
                        offset2,
                        offset3
                    )
                )
            );

			Tensorf pdf_edge = Tensorf({ 0.25f / (mImagePlaneLengthY * mFilmResX), 0.25f / (mImagePlaneLengthX * mFilmResY), 
				0.25f / (mImagePlaneLengthY * mFilmResX), 0.25f / (mImagePlaneLengthX * mFilmResY) });
			//Tensorf pdf_edge = Scalar(0.25f / (2.f * tan(mFOV * 0.5f * M_PI / 180.f)) / float(mFilmResY)) * Ones(4);
            Expr dir_edge = Tensorf({ Vector3(0.f, 1.f, 0.f), Vector3(1.f, 0.f, 0.f), Vector3(0.f, 1.f, 0.f), Vector3(1.f, 0.f, 0.f) });
			dir_edge = VectorNormalize(TransformVectors(dir_edge, mRasterToWorldTensor));
			Expr dir_visible = Tensorf({ Vector3(1.f, 0.f, 0.f), Vector3(0.f, -1.f, 0.f), Vector3(-1.f, 0.f, 0.f), Vector3(0.f, 1.f, 0.f) });
			dir_visible = VectorNormalize(TransformVectors(dir_visible, mRasterToWorldTensor));

            Tensorf grid = (Expr(make_shared<PixelCoordExp>(mFilmResX, mFilmResY, rnd.GetShape(0) / (mFilmResX * mFilmResY))) + camSamples);
            p0 = TransformPointsHomogeneous(grid, mRasterToWorldTensor);
			edge = IndexedRead(dir_edge, pixel_boundary_index, 0);
			edge2 = IndexedRead(dir_visible, pixel_boundary_index, 0);
			pdf = IndexedRead(pdf_edge, pixel_boundary_index, 0);
		}
	}
}