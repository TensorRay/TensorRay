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
#include "Scene.h"
#include "Edge.h"
using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		Expr EvalRadianceEmitted(const Scene& scene, const Ray& rays, const Intersection& its);
		Expr EvalImportance(const Scene& scene, const Camera& camera, const Ray& rays, const Intersection& its,
			Ray& raysNext, Intersection& itsNext, Expr& pixelCoor, Tensori& rayIdx);
		Expr EvalRadianceDirect(const Scene& scene, const Ray& rays, const Intersection& its, const Tensorf& rnd_light, const Tensorf& rnd_bsdf,
			Ray& raysNext, Intersection& itsNext);

		struct RenderOptions {
			RenderOptions() {}

			RenderOptions(int seed, int maxBounces, int spp, int sppe, int sppse0, int sppse1, int sppe0 = 0)
				: mRndSeed(seed), mMaxBounces(maxBounces),
				  mSppInterior(spp), mSppPrimary(sppe), mSppDirect(sppse0), mSppIndirect(sppse1), mSppPixelBoundary(sppe0),
				  mSppInteriorBatch(spp), mSppPrimaryBatch(sppe), mSppDirectBatch(sppse0), mSppIndirectBatch(sppse1), mSppPixelBoundaryBatch(sppe0),
				  mExportDerivative(false), mQuiet(true)
			{}

			int mRndSeed;
			int mMaxBounces;
			int mSppInterior;
			int mSppInteriorBatch;
			int mSppPrimary;
			int mSppPrimaryBatch;
			int mSppDirect;
			int mSppDirectBatch;
			int mSppIndirect;
			int mSppIndirectBatch;
			int mSppPixelBoundary;
			int mSppPixelBoundaryBatch;
			bool mExportDerivative;
			bool mQuiet;

			bool  g_direct = false;
			int   g_direct_depth = 0;
			int   g_direct_max_size = 100000;
			int   g_direct_spp = 16;
			float g_direct_thold = 0.01f;
			float g_eps = 0.01f;
		};

		struct GradiantImageHandler
		{
			int mResX, mResY;
			std::vector<Tensorf> mGradientImages;
			std::unordered_map<int, int> mGradientIndexMap;

			void SpecifyIndex(int iGrad, int iElement) { mGradientIndexMap.insert({ iGrad, iElement }); }
			void ClearIndex() { mGradientIndexMap.clear(); }
			void ClearGradImages() { mGradientImages.clear(); }
			void GetGradientImages(Tensorf& gradImage) const;
			void InitGradient(int resX, int resY);
			void AccumulateDeriv(const Tensorf& target);
		};

		void ExportDeriv(const Tensorf& deriv, int resY, const std::string& fn);

		class Integrator
		{
		public:
			virtual Tensorf RenderC(const Scene& scene, const RenderOptions& options)
			{
				std::cout << "[INFO] renderC not implemented!" << std::endl;
				Assert(false);
				return Tensorf();
			}
			
			virtual Tensorf RenderD(const Scene& scene, const RenderOptions& options, const Tensorf& dLdI)
			{
				const Camera& camera = *scene.mSensors[0];
				mDLoss = dLdI;
				SetParam(options);
				if (options.mExportDerivative)
					mGradHandler.InitGradient(camera.GetFilmSizeX(), camera.GetFilmSizeY());
				Tensorf gradImage = Zeros(Shape({ camera.GetFilmSizeX() * camera.GetFilmSizeY() }, VecType::Vec3));
				Integrate(scene, gradImage);
				if (options.mExportDerivative)
					mGradHandler.GetGradientImages(gradImage);
				mGradHandler.ClearGradImages();
				mGradHandler.ClearIndex();
				mDLoss.Free();
				return gradImage;
			}

			virtual void SetParam(const RenderOptions& options) = 0;

			virtual void Integrate(const Scene& scene, Tensorf& image) const = 0;

			mutable GradiantImageHandler mGradHandler;
			int mSpp;
			int mSppBatch;
			int mMaxBounces;
			Tensorf mDLoss;
			bool mVerbose;
		};
	}
}

