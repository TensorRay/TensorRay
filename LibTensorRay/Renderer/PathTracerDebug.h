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
#include "PathTracer.h"
#include "Ray.h"
#include "Records.h"

namespace EDX
{
	namespace TensorRay
	{
		class PathTracerDebug : public PathTracer
		{
		public:
			PathTracerDebug() {}

            Tensorf RenderC(const Scene& scene, const RenderOptions& options)
            {
                const Camera& camera = *scene.mSensors[0];
                Tensorf ret = Zeros(Shape({ camera.GetFilmSizeX() * camera.GetFilmSizeY() }, VecType::Vec3));
                SetParam(options);
                mDLoss.Free();
                PathTracer::Integrate(scene, ret);
                return ret;
            }

            Tensorf RenderD(const Scene& scene, const RenderOptions& options, const Tensorf& dLdI)
            {
                const Camera& camera = *scene.mSensors[0];
                mDLoss = dLdI;
                SetParam(options);
                Tensorf gradImage = Zeros(Shape({ camera.GetFilmSizeX() * camera.GetFilmSizeY() }, VecType::Vec3));
                Integrate(scene, gradImage);
                mGradHandler.ClearGradImages();
                mGradHandler.ClearIndex();
                mDLoss.Free();
                return gradImage;
            }

			void GenCameraRays(const Camera& camera, Ray& rays) const;

            Tensorf EvalFilter(const Camera& camera, const Expr& pixelId, const Intersection& isect) const;

            void GetDebugMask(const Camera& camera, const Expr& pixelId, const Intersection& isect, IndexMask& mask) const;

			void Integrate(const Scene& scene, Tensorf& image) const;

			//Expr Radiance(const Scene& scene, Ray& rays, Tensorf& image) const;
		};
	}
}
