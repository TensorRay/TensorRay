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

#include "../Core/Types.h"
#include "../Math/Vector.h"
#include "../Math/BoundingBox.h"
#include "../Math/Matrix.h"
#include "Color.h"
#include "Core/Memory.h"
#include <vector>
using std::vector;

#define MAX_PATH 260

namespace EDX
{
	struct MeshFace
	{
		int posIndices[3];
		int normIndices[3];
		int texIndices[3];
		int iSmoothingGroup;
	};
	struct ObjMaterial
	{
		char strName[MAX_PATH];
		char strTexturePath[MAX_PATH];
		char strBumpPath[MAX_PATH];
		Color color;
		Color specColor;
		Color transColor;
		float bumpScale;

		ObjMaterial(const char* name = "")
			: color(0.85f, 0.85f, 0.85f)
			, specColor(0.0f, 0.0f, 0.0f)
			, transColor(0.0f, 0.0f, 0.0f)
			, bumpScale(1.0f)
		{
			strcpy_s(strName, MAX_PATH, name);
			Memory::Memset(strTexturePath, 0, MAX_PATH);
			Memory::Memset(strBumpPath, 0, MAX_PATH);
		}

		bool operator == (const ObjMaterial& rhs) const
		{
			return strcmp(strName, rhs.strName) == 0;
		}
	};

	class EdgeList
	{
	public:
		int numEdges;
		vector<int> indexVertex0;
		vector<int> indexVertex1;
		vector<int> indexFace0;
		vector<int> indexFace1;
		vector<int> indexVertex2;
	};

	class ObjMesh
	{
	protected:
		uint mVertexCount, mTriangleCount;
		vector<Vector3> mPositionBuf;
		vector<Vector3> mNormalBuf;
		vector<Vector2> mTexBuf;
		vector<MeshFace> mFaces;
		EdgeList mEdges;

		vector<ObjMaterial> mMaterials;
		vector<uint> mMaterialIdx;
		vector<uint> mSubsetStartIdx;
		vector<uint> mSubsetMtlIdx;
		uint mNumSubsets;

		bool mNormaled;
		bool mTextured;
		bool mHasSmoothGroup;

	public:
		ObjMesh()
			: mVertexCount(0)
			, mTriangleCount(0)
			, mNumSubsets(0)
			, mNormaled(false)
			, mTextured(false)
			, mHasSmoothGroup(false)
		{
		}

		bool LoadFromObj(const char* path, const bool makeLeftHanded = true);
		void LoadPlane(const float length);
		void LoadSphere(const float radius, const int slices = 64, const int stacks = 64);
		void LoadMaterialsFromMtl(const char* path);
		void processEdges();

		inline const vector<Vector3>& GetVertexPositionBuf() const { return mPositionBuf; }
		inline const vector<Vector2>& GetVertexTextureBuf() const { return mTexBuf; }
		inline const vector<Vector3>& GetVertexNormalBuf() const { return mNormalBuf; }
		inline const MeshFace& GetFaceAt(int iIndex) const { return mFaces[iIndex]; }
		inline uint GetVertexCount() const { return mVertexCount; }
		inline uint GetTriangleCount() const { return mTriangleCount; }
		inline bool HasSmoothGroup() const { return mHasSmoothGroup; }
		inline bool IsNormaled() const { return mNormaled; }
		inline bool IsTextured() const { return mTextured; }
		inline const EdgeList& getEdgeList() const { return mEdges; }

		const vector<ObjMaterial>& GetMaterialInfo() const { return mMaterials; }
		inline const vector<uint>& GetMaterialIdxBuf() const { return mMaterialIdx; }
		inline uint GetMaterialIdx(int iTri) const { return mMaterialIdx[iTri]; }

		const uint GetSubsetCount() const { return mNumSubsets; }
		const uint GetSubsetStartIdx(int setIdx) { return mSubsetStartIdx[setIdx]; }
		const uint GetSubsetMtlIndex(int setIdx) { return mSubsetMtlIdx[setIdx]; }

		void Release();
	};
}