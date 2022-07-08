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
		void DirectBoundaryIntegrator::Integrate(const Scene& scene, Tensorf& image) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (mSpp == 0)
			{
#if USE_PROFILING
				nvtxRangePop();
#endif
				return;
			}
			const Camera& camera = *scene.mSensors[0];
			
			SecondaryEdgeInfo secEdges;
			if (ConstructSecEdgeList(scene, secEdges) == 0) 
			{
#if USE_PROFILING
				nvtxRangePop();
#endif
				return;
			}

			// TODO: Guiding Parameters
			Tensorf edgeLength = VectorLength(secEdges.e1);
			Distribution1D secEdgeDistrb(edgeLength);
			int edge_size = edgeLength.LinearSize();
			Tensori idx = Tensori::ArrayRange(0, edge_size-1, 1);
			
			if (mVerbose)
				std::cout << "Total edge_size: " << edge_size << std::endl;

			AdaptiveQuadratureDistribution m_aq;
			Tensorf cut_cdf = secEdgeDistrb.mCDF;

			if (g_direct) 
			{
				Timer timer_guide;
				timer_guide.Start();
				m_aq.setup(camera, scene, secEdges, cut_cdf, g_options);
				if (mVerbose)
					std::cout << string_format("[DirectBoundary Guiding] Total Elapsed time = %f seconds", timer_guide.GetElapsedTime()) << std::endl;
			}
				
			Timer timer;
			timer.Start();
			Shape imageShape = Shape({ camera.mResX * camera.mResY }, VecType::Vec3);
			int npass = mSpp / mSppBatch;
			for (int ipass = 0; ipass < npass; ipass++) 
			{
				int numSecondarySamples = imageShape[0] * mSppBatch;
				BoundarySegSampleDirect bss;
				// Step 1: Sample point on the edge & emitter
				if (!g_direct) 
				{
					Tensorf rnd_b = Tensorf::RandomFloat(Shape({ numSecondarySamples }, VecType::Vec3)); // no guiding
					if (SampleBoundarySegmentDirect(scene, secEdges, numSecondarySamples, rnd_b, Scalar(1.0), bss, false) == 0)
					{
#if USE_PROFILING
						nvtxRangePop();
#endif
						return;
					}
				} 
				else 
				{
					Tensorf rnd_a = Tensorf::RandomFloat(Shape({ numSecondarySamples }, VecType::Vec3)); // no guiding
					Tensorf aq_pdf = Zeros(Shape({ numSecondarySamples }));
					Tensorf aq_rnd = m_aq.sample(rnd_a, aq_pdf);
					if (SampleBoundarySegmentDirect(scene, secEdges, numSecondarySamples, aq_rnd, aq_pdf, bss, false) == 0) 
					{
#if USE_PROFILING
						nvtxRangePop();
#endif
						return;
					}
				}
				// 
				// std::cout << numSecondarySamples << std::endl;
				// std::cout << bss.pdf.LinearSize() << std::endl;
				Tensorf boundaryTerm = Zeros(Shape({ camera.mResX * camera.mResY }, VecType::Vec3));
				if (EvalBoundarySegmentDirect(camera, scene, mSpp, mMaxBounces, bss, boundaryTerm, false) == 0) 
				{
#if USE_PROFILING
					nvtxRangePop();
#endif
					return;
				}
				boundaryTerm = boundaryTerm - Detach(boundaryTerm);
				mGradHandler.AccumulateDeriv(boundaryTerm);
				if (!mDLoss.Empty()) 
				{
					boundaryTerm.Backward(mDLoss);
					AccumulateGradsAndReleaseGraph();
				}

                if (mVerbose)
                    std::cout << string_format("[DirectBoundary] #Pass %d / %d, %d kernels launched\r", ipass + 1, npass, KernelLaunchCounter::GetHandle());
                KernelLaunchCounter::Reset();
			}
			if (mVerbose)
				std::cout << string_format("[DirectBoundary] Total Elapsed time = %f (%d samples/pass, %d pass)", timer.GetElapsedTime(), mSppBatch, npass) << std::endl;
#if USE_PROFILING
			nvtxRangePop();
#endif
		}
		
	}
}
