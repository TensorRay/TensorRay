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

#include "Integrator.h"
#include "Ray.h"
#include "Records.h"

#include "AQ_distrb.h"

namespace EDX
{
	namespace TensorRay
	{
		class PrimaryBoundaryIntegrator : public Integrator
		{
		public:
			PrimaryBoundaryIntegrator() {}

			void SetParam(const RenderOptions& options) 
			{
				mSpp = options.mSppPrimary;
				mSppBatch = options.mSppPrimaryBatch;
				mMaxBounces = options.mMaxBounces;
				mVerbose = !options.mQuiet;
			}

			void Integrate(const Scene& scene, Tensorf& image) const;
		};

		class DirectBoundaryIntegrator : public Integrator
		{
		public:
			bool  g_direct = false;
			GuidingOption g_options;

			DirectBoundaryIntegrator() {}

			void SetParam(const RenderOptions& options) 
			{
				mSpp = options.mSppDirect;
				mSppBatch = options.mSppDirectBatch;
				mMaxBounces = options.mMaxBounces;
				mVerbose = !options.mQuiet;

				g_direct = options.g_direct;
				g_options.depth = options.g_direct_depth;
				g_options.max_size = options.g_direct_max_size;
				g_options.spp = options.g_direct_spp;
				g_options.thold = options.g_direct_thold;
				g_options.eps = options.g_eps;

			}

			void Integrate(const Scene& scene, Tensorf& image) const;
		};

		class IndirectBoundaryIntegrator : public Integrator
		{
		public:
			IndirectBoundaryIntegrator() {}

			void SetParam(const RenderOptions& options) 
			{
				mSpp = options.mSppIndirect;
				mSppBatch = options.mSppIndirectBatch;
				mMaxBounces = options.mMaxBounces;
				mVerbose = !options.mQuiet;
			}
			
			void Integrate(const Scene& scene, Tensorf& image) const;
		};

		class PixelBoundaryIntegrator : public Integrator
		{
		public:
			PixelBoundaryIntegrator() : mAntitheticSpp(4) {}

			void SetParam(const RenderOptions& options)
			{
				mSpp = options.mSppPixelBoundary;
				mSppBatch = options.mSppPixelBoundaryBatch;
				mMaxBounces = options.mMaxBounces;
				mVerbose = !options.mQuiet;
			}

			void Integrate(const Scene& scene, Tensorf& image) const;

			int mAntitheticSpp;
		};
	}
}
