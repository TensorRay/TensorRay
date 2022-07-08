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

#include "ParticleTracer.h"
#include "Utils.h"

namespace EDX
{
	namespace TensorRay
	{
        // Debug purpose only!
		void ParticleTracer::Integrate(const Scene& scene, Tensorf& image) const
		{
            if (mSpp == 0) return;
            const Camera& camera = *scene.mSensors[0];
            Timer timer;
            timer.Start();
            // For output derivative image
            int npass = mSpp / mSppBatch;
            int nRaysInit = mSppBatch * image.GetShape(0);

            for (int ipass = 0; ipass < npass; ipass++)
            {
                Tensorf contrib = Zeros(image.GetShape());
                
                // Sample position on emitters
                Tensorf rnd_light = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec2));
                PositionSample lightSamples;
                scene.mLights[scene.mAreaLightIndex]->Sample(rnd_light, lightSamples);
                //lightSamples.Eval();

                // TODO: Connect emitter to camera directly

                // Generate initial rays
                Tensorf rnd_dir = Tensorf::RandomFloat(Shape({ nRaysInit }, VecType::Vec2));
                Expr dir_local = Sampling::CosineSampleHemisphere(rnd_dir);
                Expr pdf = BSDFCoords::CosTheta(dir_local) * Scalar(1.0f / M_PI);

                Expr tangent, bitangent;
                CoordinateSystem(lightSamples.n, &tangent, &bitangent);
                Expr dir = X(dir_local) * tangent + Y(dir_local) * bitangent + Z(dir_local) * lightSamples.n;

                // Only works for renderC!
                //Tensorf Le = scene.mLights[scene.mAreaLightIndex]->Eval(lightSamples, dir);
                //Tensorf cosine = VectorDot(dir, lightSamples.n);
                Tensorf pdf_val = pdf * lightSamples.pdf;
                Expr power = scene.mLights[scene.mAreaLightIndex]->Eval(lightSamples, dir) * VectorDot(dir, lightSamples.n) / pdf_val;
                //Tensorf power = Le * cosine / pdf_val;

                Ray rays;
                rays.mNumRays = nRaysInit;
                rays.mOrg = lightSamples.p;
                rays.mDir = dir;
                rays.mThroughput = power; // Ones(Shape({ nRaysInit }, VecType::Vec3));
                rays.mPrevPdf = Ones(nRaysInit);
                rays.mSpecular = True(nRaysInit);
                rays.mPixelIdx = Tensori::ArrayRange(nRaysInit);
                rays.mRayIdx = Tensori::ArrayRange(nRaysInit);
                rays.mMin = Ones(nRaysInit) * Scalar(SHADOW_EPSILON);
                rays.mMax = Ones(nRaysInit) * Scalar(1e32f);

                // Continue tracing
                Intersection its;
                scene.Intersect(rays, its);
                IndexMask mask_hit = its.mTriangleId != Scalar(-1);
                rays = rays.GetMaskedCopy(mask_hit, true);
                its = its.GetMaskedCopy(mask_hit);
                scene.PostIntersect(its);                
                //its.Eval();

                {
                    //Expr dist2 = VectorSquaredLength(its.mPosition - rays.mOrg);
                    //Tensorf G = Abs(VectorDot(its.mPosition, -rays.mDir)) / dist2;
                    //rays.mThroughput = rays.mThroughput * G;
                }
                //rays.Eval();

                for (int iBounce = 0; iBounce < mMaxBounces; iBounce++)
                {
                    if (rays.mNumRays == 0)
                        break;
                    Ray rayNext;
                    Intersection itsNext;
                    Expr pixelCoor;
                    Tensori rayIdx;
                    Tensorf pathContrib = EvalImportance(scene, camera, rays, its, rayNext, itsNext, pixelCoor, rayIdx);
                    if (rayIdx.LinearSize() == 0)
                        continue;
                    Tensorf val = Tensorf({ 1.0f / float(mSpp) }) * Detach(pathContrib);
                    contrib = contrib + camera.WriteToImage(val, pixelCoor);
                    rays = rayNext;
                    its = itsNext;
                }

                if (mVerbose)
                    std::cout << string_format("[ParticleTracer] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
                KernelLaunchCounter::Reset();
                //Tensorf result = contrbPass;
                //mGradHandler.AccumulateDeriv(result);
                if (mDLoss.Empty()) {
                    // RenderC: update returned image
                    image = image + Detach(contrib);
                }
                else {
                    // RenderD: backward + update dervaitive image (optional)
                    //result.Backward(mDLoss);
                    //AccumulateGradsAndReleaseGraph();
                }
            }
            if (mVerbose)
                std::cout << string_format("[ParticleTracer] Total Elapsed time = %f (%d samples/pass, %d pass)", timer.GetElapsedTime(), mSppBatch, npass) << std::endl;
		}
	}
}
