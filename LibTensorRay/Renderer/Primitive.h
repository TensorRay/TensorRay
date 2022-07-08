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
#include "Graphics/ObjMesh.h"
#include "Utils.h"
#include "BSDF.h"
#include "Records.h"
#include "Distribution.h"
#include "Edge.h"

using namespace EDX;
using namespace EDX::DeepLearning;

namespace EDX
{
	namespace TensorRay
	{
		class TriangleMesh
		{
		public:
			Tensorf mPositionBufferRaw;
			Tensorf mPositionBuffer;				// Dim = (3, mVertexCount)
			Tensorf mFaceNormalBuffer;				// Dim = (3, mTriangleCount)
			Tensorui mIndexPosBuffer;				// Dim = (   mTriangleCount x 3)
			Tensorui mIndexNormalBuffer;			// Dim = (   mTriangleCount x 3)
			Tensorui mIndexTexBuffer;				// Dim = (   mTriangleCount x 3)
			Tensorb mUseSmoothShadingBuffer;		// Dim = (   mTriangleCount)
			Tensorf mTexcoordBuffer;				// Dim = (2, mTexcoorCount)				(mTexcoorCount >= mVertexCount)
			Tensorf mVertexNormalBufferRaw;
			Tensorf mVertexNormalBuffer;			// Dim = (3, mVertexNormalCount)		(mVertexNormalCount >= mVertexCount)
			Tensorf mTriangleAreaBuffer;			// Dim = (	 mTriangleCount)
			EdgeIndexInfo mEdgeInfo;

			int			mVertexCount;
			int			mTriangleCount;
			int			mTexcoorCount;
			int			mVertexNormalCount;
			bool		mTextured;
			float		mInvTotArea;
			float		mTotalArea;

			std::unique_ptr<const ObjMesh> mpObjMesh;
			std::unique_ptr<Distribution1D> mpDist;

		public:
			TriangleMesh() : mVertexCount(0),
							 mTriangleCount(0),
							 mTexcoorCount(0),
							 mVertexNormalCount(0),
							 mTextured(false)  { }
			~TriangleMesh() { }
			void ComputeFaceNormalAndDistrib();
			void ComputeVertexNormal(const ObjMesh* pObjMesh);

			int GetTriangleCount() const { return mTriangleCount; }
			int GetVertexCount() const { return mVertexCount; }
			const ObjMesh* GetObjMeshHandle() const { return mpObjMesh.get(); }
			void LoadRawBuffer(const ObjMesh* pObjMesh, bool flat_shading);
			void SamplePosition(const Expr& samples, PositionSample& lightSample);
			Expr PdfPosition(const Tensorf& ref, const Intersection& its) const { return Scalar(mInvTotArea); }
		};

		class Primitive
		{
		public:
			Tensorf mTrans;
			Tensorf mScale;
			Tensorf mRotate; // YawPitchRow representation
			Tensorf mRotateAxis;
			Tensorf mRotateAngle;

			// For multi-pose optimization
			Tensorf mObjCenter;              // Given by python scripts
			Tensorf mRotateMatrix;           // Given by python scripts

			Tensorf mWorldTensor;
			Tensorf mWorldInvTensor;
			Tensorf mWorldTensorRaw;
			Tensorf mWorldInvTensorRaw;

			// Allow setting translation for a single vertex
			// for visualizing forward-mode gradient image
			int mVertexId = -1;
			Tensorf mVertexTrans;

			unique_ptr<TriangleMesh> mpMesh;
			Tensorui mMaterialIndices;

			bool		mIsEmitter;
			bool		mIsFlatShading;

		public:
			virtual ~Primitive() { }
			Primitive(const char* path, const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading = false);
			Primitive(const char* path, const BSDF& bsdf, const Expr& toWorld, const Expr& toWorldInv, bool flat_shading = false);
			Primitive(const float radius, const int slices, const int stacks, const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading = false);
			Primitive(const int width, const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading = false);
			Primitive(const int width, const BSDF& bsdf, const Expr& toWorld, const Expr& toWorldInv, bool flat_shading = false);
			void Configure();

			virtual void Intersect(Ray& rays, Intersection& isect) const { }
			virtual void Occluded(Ray& rays, Tensorb& bHits) const { }

			// python binding

			void DiffTranslation();
			void DiffRotation(const Tensorf& rotateAxis);
			void DiffAllVertexPos();

			int GetVertexCount();
			int GetFaceCount();
			int GetEdgeCount();
			void GetVertexPos(ptr_wrapper<float> vPosPtr);
			void GetVertexGrad(ptr_wrapper<float> vGradPtr);
			void GetFaceIndices(ptr_wrapper<int> fPtr);
			void GetEdgeData(ptr_wrapper<int> edgePtr);

			// Center position in the object space
			Tensorf GetObjCenter();

			void ExportMesh(const char* filename);
		};
	}
}