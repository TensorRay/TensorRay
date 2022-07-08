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

#include "PathTracer.h"

namespace EDX
{
	namespace TensorRay
	{
		Expr PathTracer::Radiance(const Scene& scene, Ray& rays, Tensorf& image) const
		{
			Expr contrib = Zeros(image.GetShape());
            int nRaysInit = (mSppBatch / mAntitheticSpp) * image.GetShape(0);
			// Path tracing
			Intersection its;
			// Handle primary rays
			scene.IntersectHit(rays, its);
			if (rays.mNumRays > 0)
			{
				scene.PostIntersect(its);
				const Camera& camera = *scene.mSensors[0];
				rays.mThroughput = camera.EvalFilter(rays.mPixelIdx, its) * its.mJ;
				rays.mDir = VectorNormalize(its.mPosition - rays.mOrg);
				Expr Le = EvalRadianceEmitted(scene, rays, its);
				contrib = contrib + IndexedWrite(Le, rays.mPixelIdx, image.GetShape(), 0);
			}
            // Handle secondary rays
            for (int iBounce = 0; iBounce < mMaxBounces; iBounce++)
            {
                if (rays.mNumRays == 0)
                    break;
                Tensorf antitheticRnd_light = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec2));
                Tensorf antitheticRnd_bsdf = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec3));

                if (rays.mNumRays > 0)
                {
                    Ray raysNext;
                    Intersection itsNext;
                    Expr rnd_light = IndexedRead(antitheticRnd_light, rays.mRayIdx % Scalar(nRaysInit), 0);
					Expr rnd_bsdf = IndexedRead(antitheticRnd_bsdf, rays.mRayIdx % Scalar(nRaysInit), 0);
					Expr value = EvalRadianceDirect(scene, rays, its, rnd_light, rnd_bsdf, raysNext, itsNext);
                    contrib = contrib + IndexedWrite(value, rays.mPixelIdx, image.GetShape(), 0);
                    rays = raysNext;
                    its = itsNext;
                }
            }
            contrib = contrib * Scalar(1.0f / float(mSpp));
			return contrib;
		}

		void PathTracer::Integrate(const Scene& scene, Tensorf& image) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mSpp == 0) return;
			const Camera& camera = *scene.mSensors[0];
			Timer timer;
			timer.Start();
			// For output derivative image
			int npass = mSpp / mSppBatch;

			for (int ipass = 0; ipass < npass; ipass++)
			{
				Expr contrbPass = Zeros(image.GetShape());

				Ray rays;
                // Generate antithetic rays
				camera.GenerateAntitheticRays(rays, mSppBatch, mAntitheticSpp);
				contrbPass = contrbPass + Radiance(scene, rays, image);
				
				Tensorf result = contrbPass;
				mGradHandler.AccumulateDeriv(result);
				if (mDLoss.Empty()) 
				{
					// RenderC: update returned image
					image = image + Detach(result);
				} 
				else 
				{
					// RenderD: backward + update dervaitive image (optional)
					result.Backward(mDLoss);
					AccumulateGradsAndReleaseGraph();
				}

                if (mVerbose)
                    std::cout << string_format("[PathTracer] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
                KernelLaunchCounter::Reset();
			}
			if (mVerbose)
				std::cout << string_format("[PathTracer] Total Elapsed time = %f (%d samples/pass, %d pass)", timer.GetElapsedTime(), mSppBatch, npass) << std::endl;

#if USE_PROFILING
			nvtxRangePop();
#endif
		}
	}
}
