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

namespace EDX
{
	namespace TensorRay
	{
		class PathTracer : public Integrator
		{
		public:
			PathTracer()
			{
#if USE_BOX_FILTER
				mAntitheticSpp = 1;
#else
				mAntitheticSpp = 4;
#endif
			}

			void SetParam(const RenderOptions& options) 
			{
				mSpp = options.mSppInterior;
				mSppBatch = options.mSppInteriorBatch;
				mMaxBounces = options.mMaxBounces;
				mVerbose = !options.mQuiet;
			}

			Tensorf RenderC(const Scene& scene, const RenderOptions& options)
			{
				const Camera& camera = *scene.mSensors[0];
				Tensorf ret = Zeros(Shape({ camera.GetFilmSizeX() * camera.GetFilmSizeY() }, VecType::Vec3));
				SetParam(options);
				mDLoss.Free();
				Integrate(scene, ret);
				return ret;
			}

			void Integrate(const Scene& scene, Tensorf& image) const;

			Expr Radiance(const Scene& scene, Ray& rays, Tensorf& image) const;
			
		private:
			int mAntitheticSpp;
		};
	}
}
