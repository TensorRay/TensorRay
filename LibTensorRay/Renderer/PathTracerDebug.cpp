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

#include "PathTracerDebug.h"

namespace EDX
{
	namespace TensorRay
	{
		void PathTracerDebug::GenCameraRays(const Camera& camera, Ray& ray) const
		{
			const int numRays = camera.mResX * camera.mResY;
			Tensorf rnd = Tensorf::RandomFloat(Shape({ numRays }, VecType::Vec2));
            Tensorf origins_view = Zeros(Shape({ numRays }, VecType::Vec3));
            Tensorf origins = TransformPoints(origins_view, camera.mViewInvTensor);

            ray.mNumRays = numRays;
            ray.mOrg = Tensorf(origins);
            ray.mThroughput = Ones(Shape({ numRays }, VecType::Vec3));
            ray.mPrevPdf = Ones(numRays);
            ray.mSpecular = True(numRays);
            ray.mPixelIdx = Tensori::ArrayRange(numRays) % Scalar(camera.mResX * camera.mResY);
            ray.mRayIdx = Tensori::ArrayRange(numRays);
            ray.mMin = Ones(numRays) * Scalar(SHADOW_EPSILON);
            ray.mMax = Ones(numRays) * Scalar(1e32f);

            Tensorf px = camera.mFilter->Sample(X(rnd));
            Tensorf py = camera.mFilter->Sample(Y(rnd));
			Expr camSamples = MakeVector4(px, py, Zeros(1), Zeros(1));
			Tensorf grid = (Expr(make_shared<PixelCoordExp>(camera.mResX, camera.mResY, 1)) + camSamples);
            auto camCoord = TransformPointsHomogeneous(grid, camera.mRasterToCameraTensor);
            auto directions_view = VectorNormalize(camCoord);
            auto directions = TransformVectors(directions_view, camera.mViewInvTensor);
            ray.mDir = Tensorf(directions);
		}

		void PathTracerDebug::GetDebugMask(const Camera& camera, const Expr& pixelId, const Intersection& isect, IndexMask& mask) const
		{
            auto pyIndex = Floor(pixelId / Scalar(camera.mResX));
            auto pxIndex = pixelId - pyIndex * Scalar(camera.mResX);
            auto pProj = TransformPoints(isect.mPosition, camera.mWorldToSampleTensor);
            auto pxProj = X(pProj) * Scalar(camera.mResX);
            auto pyProj = Y(pProj) * Scalar(camera.mResY);
            auto xOffset = pxProj - Detach(pxIndex);
            auto yOffset = pyProj - Detach(pyIndex);
            auto fx = camera.mFilter->Eval(xOffset);
            auto fy = camera.mFilter->Eval(yOffset);
            auto f = fx * fy;
            mask = IndexMask(f < Scalar(1.f));
		}

        Tensorf PathTracerDebug::EvalFilter(const Camera& camera, const Expr& pixelId, const Intersection& isect) const
        {
            Tensorf pyIndex = Floor(pixelId / Scalar(camera.mResX));
            Tensorf pxIndex = pixelId - pyIndex * Scalar(camera.mResX);
            Tensorf pProj = TransformPoints(isect.mPosition, camera.mWorldToSampleTensor);
            Tensorf pxProj = X(pProj) * Scalar(camera.mResX);
            Tensorf pyProj = Y(pProj) * Scalar(camera.mResY);
            Tensorf xOffset = pxProj - Detach(pxIndex);
            Tensorf yOffset = pyProj - Detach(pyIndex);
            Tensorf fx = camera.mFilter->Eval(xOffset);
            Tensorf fy = camera.mFilter->Eval(yOffset);
            Tensorf f = fx * fy;
            Tensorf dir = isect.mPosition - camera.mPosTensor;
            Tensorf dist = VectorLength(dir);
            dir = dir / dist;
            Tensorf cosy = VectorDot(dir, camera.mDirTensor);
            Tensorf cosx = VectorDot(-dir, isect.mGeoNormal);
            Tensorf G = Abs(cosx) / (cosy * cosy * cosy * Square(dist));
            // f could be zero!
            return G * f / (Detach(G) * Detach(f));
        }

        Tensorf DebugEvalRadianceDirect(const Scene& scene, const Ray& rays, const Intersection& its,
            const Tensorf& rnd_light, const Tensorf& rnd_bsdf, Ray& raysNext, Intersection& itsNext)
        {
            Tensorf ret = Zeros(Shape({ rays.mNumRays }, VecType::Vec3));
            // Sample point on area light
            PositionSample lightSamples;
            scene.mLights[scene.mAreaLightIndex]->Sample(rnd_light, lightSamples);
            lightSamples.Eval();

            for (int iBSDF = 0; iBSDF < scene.mBSDFCount; iBSDF++)
            {
                IndexMask mask_bsdf = its.mBsdfId == Scalar(iBSDF);
                if (mask_bsdf.sum == 0) continue;
                Intersection its_nee = its.GetMaskedCopy(mask_bsdf, true);
                PositionSample light_sample = lightSamples.GetMaskedCopy(mask_bsdf);
                Tensorf wi = light_sample.p - its_nee.mPosition;
                Tensorf dist_sqr = VectorSquaredLength(wi);
                Tensorf dist = Sqrt(dist_sqr);

                wi = wi / dist;
                Tensorf dotShNorm = VectorDot(its_nee.mNormal, wi);
                Tensorf dotGeoNorm = VectorDot(its_nee.mGeoNormal, wi);

                // check if the connection is valid
                Tensorf connectValid;
                {
                    auto numStable = (dist > Scalar(SHADOW_EPSILON) && Abs(dotShNorm) > Scalar(EDGE_EPSILON));
                    auto noLightLeak = ((dotShNorm * dotGeoNorm) > Scalar(0.0f));
                    Expr visible;
                    {
                        Ray shadowRays;
                        shadowRays.mNumRays = mask_bsdf.sum;
                        shadowRays.mOrg = its_nee.mPosition;
                        shadowRays.mDir = wi;
                        shadowRays.mMin = Ones(mask_bsdf.sum) * Scalar(SHADOW_EPSILON);
                        shadowRays.mMax = dist - Scalar(SHADOW_EPSILON);
                        scene.Occluded(shadowRays, visible);
                    }
                    connectValid = numStable * noLightLeak * visible;
                }

                // evaluate the direct lighting from light sampling
                Tensorf tmpThroughput = rays.mThroughput;
                Tensorf throughput = Mask(rays.mThroughput, mask_bsdf, 0);
                Tensorf wo = Mask(-rays.mDir, mask_bsdf, 0);
                Tensorf G = Abs(VectorDot(light_sample.n, -wi)) / dist_sqr;
                Tensorf bsdfVal = scene.mBsdfs[iBSDF]->Eval(its_nee, wo, wi) * Abs(dotShNorm);
                Tensorf Le = scene.mLights[scene.mAreaLightIndex]->Eval(light_sample, -wi);
                Tensorf val = throughput * G; // connectValid * throughput * bsdfVal * light_sample.J * G * Le / light_sample.pdf;
                ret = ret + IndexedWrite(val, mask_bsdf.index, ret.GetShape(), 0);
            }
            return ret;
        }

// 		Expr PathTracer::Radiance(const Scene& scene, Ray& rays, Tensorf& image) const
// 		{
// 			Expr contrib = Zeros(image.GetShape());
//             int nRaysInit = (mSppBatch / mAntitheticSpp) * image.GetShape(0);
// 			// Path tracing
// 			Intersection its;
// 			// Handle primary rays
// 			scene.IntersectHit(rays, its);
// 			if (rays.mNumRays > 0)
// 			{
// 				scene.PostIntersect(its);
// 				const Camera& camera = *scene.mSensors[0];
// 				rays.mThroughput = camera.EvalFilter(rays.mPixelIdx, its) * its.mJ;
// 				rays.mDir = VectorNormalize(its.mPosition - rays.mOrg);
// 				Expr Le = EvalRadianceEmitted(scene, rays, its);
// 				contrib = contrib + IndexedWrite(Le, rays.mPixelIdx, image.GetShape(), 0);
// 			}
//             // Handle secondary rays
//             for (int iBounce = 0; iBounce < mMaxBounces; iBounce++)
//             {
//                 if (rays.mNumRays == 0)
//                     break;
//                 Tensorf antitheticRnd_light = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec2));
//                 Tensorf antitheticRnd_bsdf = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec3));
// 
//                 if (rays.mNumRays > 0)
//                 {
//                     Ray raysNext;
//                     Intersection itsNext;
//                     Expr rnd_light = IndexedRead(antitheticRnd_light, rays.mRayIdx % Scalar(nRaysInit), 0);
// 					Expr rnd_bsdf = IndexedRead(antitheticRnd_bsdf, rays.mRayIdx % Scalar(nRaysInit), 0);
// 					Expr value = EvalRadianceDirect(scene, rays, its, rnd_light, rnd_bsdf, raysNext, itsNext);
//                     contrib = contrib + IndexedWrite(value, rays.mPixelIdx, image.GetShape(), 0);
//                     rays = raysNext;
//                     its = itsNext;
//                 }
//             }
//             contrib = contrib * Scalar(1.0f / float(mSpp));
// 			return contrib;
// 		}

		void PathTracerDebug::Integrate(const Scene& scene, Tensorf& image) const
		{
			if (mSpp == 0) return;
			const Camera& camera = *scene.mSensors[0];

			int npass = mSpp;
			for (int ipass = 0; ipass < npass; ipass++)
			{
                Tensorf contrib = Zeros(image.GetShape());

                Ray rays;
				GenCameraRays(camera, rays);

				Intersection its;
				scene.IntersectHit(rays, its);

				if (rays.mNumRays > 0)
				{
					scene.PostIntersect(its);

					IndexMask mask_debug;
					GetDebugMask(camera, rays.mPixelIdx, its, mask_debug);
					
					rays = rays.GetMaskedCopy(mask_debug, true);
					its = its.GetMaskedCopy(mask_debug, true);
                    rays.mThroughput = EvalFilter(camera, rays.mPixelIdx, its) * its.mJ;
					rays.mDir = VectorNormalize(its.mPosition - rays.mOrg);
				}

				int nRaysInit = camera.mResX * camera.mResY;
				Tensorf antitheticRnd_light = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec2));
                Tensorf antitheticRnd_bsdf = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec3));

                if (rays.mNumRays > 0)
                {
                    Ray raysNext;
                    Intersection itsNext;
                    Tensorf rnd_light = IndexedRead(antitheticRnd_light, rays.mRayIdx % Scalar(nRaysInit), 0);
                    Tensorf rnd_bsdf = IndexedRead(antitheticRnd_bsdf, rays.mRayIdx % Scalar(nRaysInit), 0);

                    Tensorf value = DebugEvalRadianceDirect(scene, rays, its, rnd_light, rnd_bsdf, raysNext, itsNext);
                    contrib = contrib + IndexedWrite(value, rays.mPixelIdx, image.GetShape(), 0);
                }
                contrib = contrib * Scalar(1.f / float(mSpp));

				Tensorf result = contrib;
				result.Backward(mDLoss);
				AccumulateGradsAndReleaseGraph();

                if (mVerbose)
                    std::cout << string_format("[PathTracerDebug] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
                KernelLaunchCounter::Reset();
			}
		}
	}
}
