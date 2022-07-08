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

#include "Boundary.h"
namespace EDX
{
	namespace TensorRay
	{
		void PrimaryBoundaryIntegrator::Integrate(const Scene& scene, Tensorf& image) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mSpp == 0) return;
			const Camera& camera = *scene.mSensors[0];

            PrimaryEdgeInfo2 primEdges;
            if (ConstructPrimEdgeList(scene, camera, primEdges) == 0)
                return;

			Timer timer;
			timer.Start();
			Shape imageShape = Shape({ camera.mResX * camera.mResY }, VecType::Vec3);
			int npass = mSpp / mSppBatch;
			for (int ipass = 0; ipass < npass; ipass++) {
				int numPrimarySamples = imageShape[0] * mSppBatch;
				BoundarySegSamplePrimary bss;
				// Step 1: Sample point on the edge
				if (SampleBoundarySegmentPrimary(scene, primEdges, numPrimarySamples, bss) == 0)
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}

				// Step 2: Check if the ray hit anything
				Intersection its;
				TensorRay::Ray rayFromEdge(bss.p0, VectorNormalize(bss.p0 - camera.mPosTensor));
				scene.Intersect(rayFromEdge, its);
				bss.maskValid = IndexMask(its.mBsdfId != Scalar(-1));
				if (bss.maskValid.sum == 0)
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}

				Tensorf hitT = Mask(its.mTHit, bss.maskValid, 0);
				auto hitP = Mask(rayFromEdge.mOrg, bss.maskValid, 0) + hitT * Mask(rayFromEdge.mDir, bss.maskValid, 0);
				its = its.GetMaskedCopy(bss.maskValid);
				bss = bss.getValidCopy();

				// Check if hitP is visible to camera
				SensorDirectSample sds = camera.sampleDirect(hitP);
				bss.maskValid = IndexMask(sds.isValid);
				if (bss.maskValid.sum == 0)
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}
				TensorRay::Ray rays;		//Ray from camera
				camera.GenerateBoundaryRays(sds, rays);
				auto coorImg = Mask(sds.q, bss.maskValid, 0);
				auto dist2cam = Mask(VectorLength(bss.p0 - camera.mPosTensor), bss.maskValid, 0);
				rays.mMax = dist2cam - Scalar(SHADOW_EPSILON);
				Tensorb samePoint;
				scene.Occluded(rays, samePoint);
				Tensori isVisible = IndexedWrite(samePoint.Reshape(samePoint.LinearSize()), bss.maskValid.index, bss.maskValid.mask.GetShape(), 0);
				bss.maskValid = IndexMask(bss.maskValid.mask * isVisible);
				if (bss.maskValid.sum == 0)
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}
				its = its.GetMaskedCopy(bss.maskValid);
				IndexMask isSamePoint(Where(samePoint, Ones(1), Zeros(1)));
				rays = rays.GetMaskedCopy(isSamePoint, true);
				Tensorf pixelCoor = Mask(coorImg, isSamePoint, 0);
				bss = bss.getValidCopy();

				// Step 4: Compute the boundary contribution
				Expr baseVal;
				scene.PostIntersect(its);
				auto dist = VectorLength(its.mPosition - camera.mPosTensor);
				auto dist1 = VectorLength(bss.p0 - camera.mPosTensor);
				auto cos2 = Abs(VectorDot(its.mGeoNormal, -rays.mDir));
				auto e = VectorCross(bss.edge, rays.mDir);
				auto sinphi = VectorLength(e);
				auto proj = VectorNormalize(VectorCross(e, its.mGeoNormal));
				auto sinphi2 = VectorLength(VectorCross(rays.mDir, proj));
				auto n = Detach(VectorNormalize(VectorCross(its.mGeoNormal, proj)));
				auto sign0 = VectorDot(e, bss.edge2) > Scalar(0.0f);
				auto sign1 = VectorDot(e, n) > Scalar(0.0f);
				baseVal = (dist / dist1) * (sinphi / sinphi2) * cos2;
				baseVal = baseVal * (sinphi > Scalar(EPSILON)) * (sinphi2 > Scalar(EPSILON));
				baseVal = baseVal * Where(sign0 == sign1, Ones(bss.pdf.GetShape()), -Ones(bss.pdf.GetShape()));

				auto indicesTri0 = Scalar(3) * its.mTriangleId;
				auto indicesTri1 = Scalar(3) * its.mTriangleId + Scalar(1);
				auto indicesTri2 = Scalar(3) * its.mTriangleId + Scalar(2);
				Expr u, v, w, t;
				auto indicesPos0 = IndexedRead(scene.mIndexPosBuffer, indicesTri0, 0);
				auto indicesPos1 = IndexedRead(scene.mIndexPosBuffer, indicesTri1, 0);
				auto indicesPos2 = IndexedRead(scene.mIndexPosBuffer, indicesTri2, 0);
				auto position0 = IndexedRead(scene.mPositionBuffer, indicesPos0, 0);
				auto position1 = IndexedRead(scene.mPositionBuffer, indicesPos1, 0);
				auto position2 = IndexedRead(scene.mPositionBuffer, indicesPos2, 0);
				RayIntersectAD(VectorNormalize(bss.p0 - camera.mPosTensor), camera.mPosTensor,
					position0, position1 - position0, position2 - position0, u, v, t);
				w = Scalar(1.0f) - u - v;
				auto u2 = w * Detach(position0) + u * Detach(position1) + v * Detach(position2);
				auto xDotN = VectorDot(n, u2);

				Tensorf radiance = Zeros(Shape({ rays.mNumRays }, VecType::Vec3));
				rays.mRayIdx = Tensori::ArrayRange(rays.mNumRays);
				radiance = radiance + Detach(EvalRadianceEmitted(scene, rays, its));
				for (int iBounce = 0; iBounce < mMaxBounces; iBounce++)
				{
					if (rays.mNumRays == 0) break;
					Ray raysNext;
					Intersection itsNext;
					Tensorf rnd_light = Tensorf::RandomFloat(Shape({ rays.mNumRays }, VecType::Vec2));
					Tensorf rnd_bsdf = Tensorf::RandomFloat(Shape({ rays.mNumRays }, VecType::Vec3));
					auto val = Detach(EvalRadianceDirect(scene, rays, its, rnd_light, rnd_bsdf, raysNext, itsNext));
					radiance = radiance + IndexedWrite(val, rays.mRayIdx, radiance.GetShape(), 0);
					rays = raysNext;
					its = itsNext;
				}

				Tensorf boundaryTerm = Detach(radiance * baseVal / bss.pdf) * xDotN;
				boundaryTerm = Tensorf({ 1.0f / float(mSpp) }) * camera.WriteToImage(boundaryTerm, pixelCoor);
				mGradHandler.AccumulateDeriv(boundaryTerm);
				if (!mDLoss.Empty()) 
				{
					boundaryTerm.Backward(mDLoss);
					AccumulateGradsAndReleaseGraph();
				}

                if (mVerbose)
                    std::cout << string_format("[PrimaryBoundary] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
                KernelLaunchCounter::Reset();
			}
			if (mVerbose)
				std::cout << string_format("[PrimaryBoundary] Total Elapsed time = %f (%d samples/pass, %d pass)", timer.GetElapsedTime(), mSppBatch, npass) << std::endl;
#if USE_PROFILING
			nvtxRangePop();
#endif
		}
	}
}
