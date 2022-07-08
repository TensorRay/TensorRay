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
#include "../Tensor/Tensor.h"
#include "Config.h"

using namespace EDX;
using namespace EDX::DeepLearning;
namespace EDX
{
	namespace TensorRay
	{
		class Scene;
		class Camera;
		class Ray;
		class Intersection;
		// Old boundary term
		class EdgeIndexInfo
		{
		public:
			int numEdges;
			Tensori indexVert0;
			Tensori indexVert1;
			Tensori indexTri0;
			Tensori indexTri1;
			Tensori indexVert2;
		};

		// Dircect Boundary
		struct SecondaryEdgeInfo
		{
			int numTot;
			Expr isBoundary;
			Expr p0;
			Expr e1;
			Expr n0;
			Expr n1;
			Expr p2;
		};

		struct BoundarySegSampleSecondary
		{
			Tensorf p0;
			Tensorf edge;
			Tensorf edge2;
			Tensorf pdf;
		};
		
		struct BoundarySegSampleDirect : BoundarySegSampleSecondary
		{
			Tensorf p2;
			Tensorf n;
			IndexMask maskValid;

			BoundarySegSampleDirect getValidCopy() const
			{
				BoundarySegSampleDirect ret;
				ret.p0 = Mask(p0, maskValid, 0);
				ret.edge = Mask(edge, maskValid, 0);
				ret.edge2 = Mask(edge2, maskValid, 0);
				ret.p2 = Mask(p2, maskValid, 0);
				ret.n = Mask(n, maskValid, 0);
				ret.pdf = Mask(pdf, maskValid, 0);
				ret.maskValid = IndexMask(Ones(maskValid.sum));
				return ret;
			}
		};
		int ConstructSecEdgeList(const Scene& scene, SecondaryEdgeInfo& list);
		Tensori SampleFromSecEdges(const SecondaryEdgeInfo& list, const Tensorf& rnd1, BoundarySegSampleSecondary& samples);
		int SampleBoundarySegmentDirect(const Scene& scene, const SecondaryEdgeInfo &secEdges, int numSamples, const Tensorf& rnd_b, const Tensorf& pdf_b, BoundarySegSampleDirect& samples, bool guiding_mode);
		int EvalBoundarySegmentDirect(const Camera& camera, const Scene& scene, int mSpp, int mMaxBounce, BoundarySegSampleDirect& bss, Tensorf& boundaryTerm, bool guiding_mode);

		// Indirect Boundary
		struct BoundarySegSampleIndirect : BoundarySegSampleSecondary
		{
			Tensorf dir;
			IndexMask maskValid;

			BoundarySegSampleIndirect getValidCopy() const
			{
				BoundarySegSampleIndirect ret;
				ret.p0 = Mask(p0, maskValid, 0);
				ret.edge = Mask(edge, maskValid, 0);
				ret.edge2 = Mask(edge2, maskValid, 0);
				ret.dir = Mask(dir, maskValid, 0);
				ret.pdf = Mask(pdf, maskValid, 0);
				ret.maskValid = IndexMask(Ones(maskValid.sum));
				return ret;
			}
		};
		int SampleBoundarySegmentIndirect(const Scene& scene, const SecondaryEdgeInfo& secEdges, int numSamples, BoundarySegSampleIndirect& samples);

		// New primary edge evaluation
		struct PrimaryEdgeInfo2
		{
			int numTot;
			Expr isBoundary;
			Expr p0;
			Expr e1;
			Expr n0;
			Expr n1;
			Expr p2;
		};
		struct BoundarySegSamplePrimary
		{
			Tensorf p0;
			Tensorf edge;
			Tensorf edge2;
			Tensorf pdf;
			IndexMask maskValid;

			BoundarySegSamplePrimary getValidCopy() const
			{
				BoundarySegSamplePrimary ret;
				ret.p0 = Mask(p0, maskValid, 0);
				ret.edge = Mask(edge, maskValid, 0);
				ret.edge2 = Mask(edge2, maskValid, 0);
				ret.pdf = Mask(pdf, maskValid, 0);
				ret.maskValid = IndexMask(Ones(maskValid.sum));
				return ret;
			}
		};
		int ConstructPrimEdgeList(const Scene& scene, const Camera& camera, PrimaryEdgeInfo2& list);
		Tensori SampleFromPrimEdges(const PrimaryEdgeInfo2& list, int numSamples, BoundarySegSamplePrimary& samples);
		int SampleBoundarySegmentPrimary(const Scene& scene, const PrimaryEdgeInfo2& primEdges, int numSamples, BoundarySegSamplePrimary& samples);

		// Pixel boundary
		struct BoundarySegSamplePixel : BoundarySegSamplePrimary
		{
			Tensori rayIdx;
			Tensori pixelIdx;

			BoundarySegSamplePixel getValidCopy() const
			{
				BoundarySegSamplePixel ret;
                ret.p0 = Mask(p0, maskValid, 0);
                ret.edge = Mask(edge, maskValid, 0);
                ret.edge2 = Mask(edge2, maskValid, 0);
                ret.pdf = Mask(pdf, maskValid, 0);
                ret.rayIdx = Mask(rayIdx, maskValid, 0);
				ret.pixelIdx = Mask(pixelIdx, maskValid, 0);
                ret.maskValid = IndexMask(Ones(maskValid.sum));
                return ret;
			}
		};

		void SampleBoundarySegmentPixel(const Camera& camera, int spp, int antitheticSpp, BoundarySegSamplePixel& samples);
	}
}
