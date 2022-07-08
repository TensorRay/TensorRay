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

#include "Boundary.h"
namespace EDX
{
	namespace TensorRay
	{
		void IndirectBoundaryIntegrator::Integrate(const Scene& scene, Tensorf& image) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mSpp == 0) return;
			const Camera& camera = *scene.mSensors[0];
			
			SecondaryEdgeInfo secEdges;
			if (ConstructSecEdgeList(scene, secEdges) == 0)
			{
#if USE_PROFILING
				nvtxRangePop();
#endif
				return;
			}

			Timer timer;
			timer.Start();
			Shape imageShape = Shape({ camera.mResX * camera.mResY }, VecType::Vec3);
			int npass = mSpp / mSppBatch;
			for (int ipass = 0; ipass < npass; ipass++) 
			{
				int numSecondarySamples = imageShape[0] * mSppBatch;
				BoundarySegSampleIndirect bss;
				// Step 1: Sample point on the edge & emitter
				if (SampleBoundarySegmentIndirect(scene, secEdges, numSecondarySamples, bss) == 0)
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}
				// Step 2: Compute the contrib from valid boundary segments (AD)
				Intersection its2sensor, its2emitter;
				Tensorf baseVal, xDotN;
				Ray ray2sensor(bss.p0, bss.dir);
				Ray ray2emitter(bss.p0, -bss.dir);
				{
					scene.Intersect(ray2emitter, its2emitter);
					scene.Intersect(ray2sensor, its2sensor);
					bss.maskValid = IndexMask(its2emitter.mTriangleId != Scalar(-1) && its2sensor.mTriangleId != Scalar(-1));
					if (bss.maskValid.sum == 0)
					{
#if USE_PROFILING
						nvtxRangePop();
#endif
						return;
					}
					its2emitter = its2emitter.GetMaskedCopy(bss.maskValid);
					its2sensor = its2sensor.GetMaskedCopy(bss.maskValid);
					ray2emitter = ray2emitter.GetMaskedCopy(bss.maskValid);
					ray2sensor = ray2sensor.GetMaskedCopy(bss.maskValid);
					bss = bss.getValidCopy();

					scene.PostIntersect(its2emitter);
					scene.PostIntersect(its2sensor);
					auto dist = VectorLength(its2emitter.mPosition - its2sensor.mPosition);
					auto cos2 = Abs(VectorDot(its2emitter.mGeoNormal, ray2sensor.mDir));
					auto e = VectorCross(bss.edge, -ray2sensor.mDir);
					auto sinphi = VectorLength(e);
					auto proj = VectorNormalize(VectorCross(e, its2emitter.mGeoNormal));
					auto sinphi2 = VectorLength(VectorCross(-ray2sensor.mDir, proj));
					auto itsT = VectorLength(its2sensor.mPosition - bss.p0);
					auto n = Detach(VectorNormalize(VectorCross(its2emitter.mGeoNormal, proj)));
					auto sign0 = VectorDot(e, bss.edge2) > Scalar(0.0f);
					auto sign1 = VectorDot(e, n) > Scalar(0.0f);
					baseVal = Detach((itsT / dist) * (sinphi / sinphi2) * cos2);
					baseVal = baseVal * (sinphi > Scalar(EPSILON)) * (sinphi2 > Scalar(EPSILON));
					baseVal = baseVal * Where(sign0 == sign1, Ones(bss.pdf.GetShape()), -Ones(bss.pdf.GetShape()));

					auto indicesTri0 = Scalar(3) * its2emitter.mTriangleId;
					auto indicesTri1 = Scalar(3) * its2emitter.mTriangleId + Scalar(1);
					auto indicesTri2 = Scalar(3) * its2emitter.mTriangleId + Scalar(2);
					Expr u, v, w, t;
					auto indicesPos0 = IndexedRead(scene.mIndexPosBuffer, indicesTri0, 0);
					auto indicesPos1 = IndexedRead(scene.mIndexPosBuffer, indicesTri1, 0);
					auto indicesPos2 = IndexedRead(scene.mIndexPosBuffer, indicesTri2, 0);
					auto position0 = IndexedRead(scene.mPositionBuffer, indicesPos0, 0);
					auto position1 = IndexedRead(scene.mPositionBuffer, indicesPos1, 0);
					auto position2 = IndexedRead(scene.mPositionBuffer, indicesPos2, 0);
					RayIntersectAD(VectorNormalize(bss.p0 - its2sensor.mPosition), its2sensor.mPosition,
						position0, position1 - position0, position2 - position0, u, v, t);
					w = Scalar(1.0f) - u - v;
					auto u2 = w * Detach(position0) + u * Detach(position1) + v * Detach(position2);
					xDotN = VectorDot(n, u2);
				}

				int numRays = ray2sensor.mNumRays;
				Tensorf value0 = Detach(baseVal / bss.pdf) * xDotN;
				std::vector<Tensorf> radiance;
				// Step 3: trace toward the emitter
				Tensorf currentRadiance = Zeros(Shape({ numRays }, VecType::Vec3));
				ray2emitter.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
				ray2emitter.mRayIdx = Tensori::ArrayRange(numRays);
				ray2emitter.mPrevPdf = Ones(numRays);
				ray2emitter.mSpecular = True(numRays);
				ray2emitter.mPixelIdx = Tensori::ArrayRange(numRays).Reshape(numRays);
				for (int iBounce = 0; iBounce < mMaxBounces - 1; iBounce++)
				{
					if (ray2emitter.mNumRays == 0)
						break;
					Ray raysNext;
					Intersection itsNext;
					Tensorf rnd_light = Tensorf::RandomFloat(Shape({ ray2emitter.mNumRays }, VecType::Vec2));
					Tensorf rnd_bsdf = Tensorf::RandomFloat(Shape({ ray2emitter.mNumRays }, VecType::Vec3));
					Tensorf value0 = EvalRadianceDirect(scene, ray2emitter, its2emitter, rnd_light, rnd_bsdf, raysNext, itsNext);
					currentRadiance = currentRadiance + IndexedWrite(Detach(value0), ray2emitter.mRayIdx, currentRadiance.GetShape(), 0);
					radiance.push_back(currentRadiance);
					ray2emitter = raysNext;
					its2emitter = itsNext;
				}

				// Step 4: trace towards the sensor
				ray2sensor.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
				ray2sensor.mRayIdx = Tensori::ArrayRange(numRays);
				ray2sensor.mPrevPdf = Ones(numRays);
				ray2sensor.mSpecular = True(numRays);
				ray2sensor.mPixelIdx = Tensori::ArrayRange(numRays).Reshape(numRays);
				Tensorf boundaryTerm = Zeros(Shape({ camera.mResX * camera.mResY }, VecType::Vec3));
				for (int iBounce = 0; iBounce < mMaxBounces - 1; iBounce++)
				{
					if (ray2sensor.mNumRays == 0)
						break;
					Ray rayNext;
					Intersection itsNext;
					Expr pixelCoor;
					Tensori rayIdx;
					Tensorf pathContrib = EvalImportance(scene, camera, ray2sensor, its2sensor, rayNext, itsNext, pixelCoor, rayIdx);
					if (rayIdx.LinearSize() > 0 && mMaxBounces - iBounce - 2 < radiance.size())
					{
						Tensorf rad = IndexedRead(radiance[mMaxBounces - iBounce - 2], rayIdx, 0);
						Tensorf val = Tensorf({ 1.0f / float(mSpp) }) * IndexedRead(value0, rayIdx, 0) * rad * Detach(pathContrib);
						boundaryTerm = boundaryTerm + camera.WriteToImage(val, pixelCoor);
					}
					ray2sensor = rayNext;
					its2sensor = itsNext;
				}
				boundaryTerm = boundaryTerm - Detach(boundaryTerm);
				mGradHandler.AccumulateDeriv(boundaryTerm);
				if (!mDLoss.Empty()) 
				{
					boundaryTerm.Backward(mDLoss);
					AccumulateGradsAndReleaseGraph();
				}
				if (mVerbose)
					std::cout << string_format("[IndirectBoundary] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
				KernelLaunchCounter::Reset();
			}
			if (mVerbose)
				std::cout << string_format("[IndirectBoundary] Total Elapsed time = %f (%d samples/pass, %d pass)", timer.GetElapsedTime(), mSppBatch, npass) << std::endl;
#if USE_PROFILING
			nvtxRangePop();
#endif
		}
	}
}
