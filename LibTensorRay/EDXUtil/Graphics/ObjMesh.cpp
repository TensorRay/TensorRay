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

#include "ObjMesh.h"
#include "../Math/Matrix.h"
#include "../Core/Memory.h"
#include <map>
#include <vector>

namespace EDX
{
	bool ObjMesh::LoadFromObj(const char* strPath, const bool makeLeftHanded)
	{
		int iSmoothingGroup = 0;		
		mNumSubsets = 0;
		mHasSmoothGroup = false;
		mNormaled = false;
		mTextured = false;
		int iCurrentMtl = 0;
		char strMaterialFilename[MAX_PATH] = { 0 };
		//Matrix leftHandedTransform = makeLeftHanded ? Matrix::Scale(-1.0f, 1.0f, 1.0f)
		//											: Matrix::IDENTITY;
		//Matrix invLeftHandedTransform = Matrix::Inverse(leftHandedTransform);

		char strCommand[MAX_PATH] = { 0 };
		FILE* pInFile = 0;
		fopen_s(&pInFile, strPath, "rt");
		assert(pInFile);

		while (!feof(pInFile))
		{
			fscanf_s(pInFile, "%s", strCommand, MAX_PATH);

			if (0 == strcmp(strCommand, "#"))
			{
				// Comment
			}
			else if (0 == strcmp(strCommand, "v"))
			{
				// Vertex Position
				float x, y, z;
				fscanf_s(pInFile, "%f %f %f", &x, &y, &z);
				//mPositionBuf.push_back(Matrix::TransformPoint(Vector3(x, y, z), leftHandedTransform));
				mPositionBuf.push_back(Vector3(x, y, z));
			}
			else if (0 == strcmp(strCommand, "vt"))
			{
				// Vertex TexCoord
				float u, v;
				fscanf_s(pInFile, "%f %f", &u, &v);
				mTexBuf.push_back(Vector2(u, 1.0f-v));
				mTextured = true;
			}
			else if (0 == strcmp(strCommand, "vn"))
			{
				// Vertex Normal
				float x, y, z;
				fscanf_s(pInFile, "%f %f %f", &x, &y, &z);
				//mNormalBuf.push_back(Matrix::TransformNormal(Vector3(x, y, z), invLeftHandedTransform));
				mNormalBuf.push_back(Vector3(x, y, z));
				mNormaled = true;
			}
			else if (0 == strcmp(strCommand, "f"))
			{
				// Parse face
				fgets(strCommand, MAX_PATH, pInFile);
				int length = strlen(strCommand) + 1;

				// Face
				int posIdx, texIdx, normalIdx;
				MeshFace Face, quadFace;
				uint facePosIdx[4] = { 0, 0, 0, 0 };
				uint faceTexIdx[4] = { 0, 0, 0, 0 };
				uint faceVertexNormalIdx[4] = { 0, 0, 0, 0 };
				int vertexCount = 0;

				int slashCount = -1;
				bool doubleSlash = false;
				auto startIdx = 0;
				for (auto i = 0; i < length; i++)
				{
					auto c = strCommand[i];
					if (strCommand[i] != ' ' && strCommand[i] != '\t' && strCommand[i] != '\n' && strCommand[i] != '\0')
						continue;
					if (startIdx == i)
					{
						startIdx++;
						continue;
					}

					if (slashCount == -1)
					{
						slashCount = 0;
						bool prevIsSlash = false;
						for (auto cur = startIdx; cur < i; cur++)
						{
							if (strCommand[cur] == '/')
							{
								if (prevIsSlash)
									doubleSlash = true;

								slashCount++;
								prevIsSlash = true;
							}
							else
							{
								prevIsSlash = false;
							}
						}
					}

					if (!doubleSlash)
					{
						if (slashCount == 0)
						{
							sscanf_s(strCommand + startIdx, "%d", &posIdx);
							if (posIdx < 0)
								posIdx = mPositionBuf.size() + posIdx + 1;
						}
						else if (slashCount == 1)
						{
							sscanf_s(strCommand + startIdx, "%d/%d", &posIdx, &texIdx);
							if (posIdx < 0)
								posIdx = mPositionBuf.size() + posIdx + 1;
							if (texIdx < 0)
								texIdx = mTexBuf.size() + texIdx + 1;
						}
						else if (slashCount == 2)
						{
							sscanf_s(strCommand + startIdx, "%d/%d/%d", &posIdx, &texIdx, &normalIdx);
							if (posIdx < 0)
								posIdx = mPositionBuf.size() + posIdx + 1;
							if (texIdx < 0)
								texIdx = mTexBuf.size() + texIdx + 1;
							if (normalIdx < 0)
								normalIdx = mNormalBuf.size() + normalIdx + 1;
						}
					}
					else
					{
						sscanf_s(strCommand + startIdx, "%d//%d", &posIdx, &normalIdx);
						if (posIdx < 0)
							posIdx = mPositionBuf.size() + posIdx + 1;
						if (normalIdx < 0)
							normalIdx = mNormalBuf.size() + normalIdx + 1;
					}

					facePosIdx[vertexCount] = posIdx - 1;
					if (mTextured)
						faceTexIdx[vertexCount] = texIdx - 1;
					if (mNormaled)
					faceVertexNormalIdx[vertexCount] = normalIdx - 1;

					vertexCount++;
					Assert(vertexCount <= 4);
					startIdx = i + 1;
				}


				if (makeLeftHanded)
				{
					Face.posIndices[0] = facePosIdx[0];
					Face.posIndices[1] = facePosIdx[2];
					Face.posIndices[2] = facePosIdx[1];
					if (mTextured)
					{
						Face.texIndices[0] = faceTexIdx[0];
						Face.texIndices[1] = faceTexIdx[2];
						Face.texIndices[2] = faceTexIdx[1];
					}
					if (mNormaled)
					{
						Face.normIndices[0] = faceVertexNormalIdx[0];
						Face.normIndices[1] = faceVertexNormalIdx[2];
						Face.normIndices[2] = faceVertexNormalIdx[1];
					}
				}
				else
				{
					for (int j = 0; j < 3; j++)
					{
						Face.posIndices[j] = facePosIdx[j];
						if (mTextured)
							Face.texIndices[j] = faceTexIdx[j];
						if (mNormaled)
							Face.normIndices[j] = faceVertexNormalIdx[j];
					}
				}

				// Add face
				Face.iSmoothingGroup = iSmoothingGroup;
				mFaces.push_back(Face);
				mMaterialIdx.push_back(iCurrentMtl);

				if (vertexCount == 4)
				{
					// Triangularize quad
					{
						if (makeLeftHanded)
						{
							quadFace.posIndices[0] = facePosIdx[3];
							quadFace.posIndices[1] = Face.posIndices[1];
							quadFace.posIndices[2] = Face.posIndices[0];
							if (mTextured)
							{
								Face.texIndices[0] = faceTexIdx[3];
								Face.texIndices[1] = Face.texIndices[1];
								Face.texIndices[2] = Face.texIndices[0];
							}
							if (mNormaled)
							{
								Face.normIndices[0] = faceTexIdx[3];
								Face.normIndices[1] = Face.normIndices[1];
								Face.normIndices[2] = Face.normIndices[0];
							}
						}
						else
						{
							quadFace.posIndices[0] = facePosIdx[3];
							quadFace.posIndices[1] = Face.posIndices[0];
							quadFace.posIndices[2] = Face.posIndices[2];
							if (mTextured)
							{
								Face.texIndices[0] = faceTexIdx[3];
								Face.texIndices[1] = Face.texIndices[0];
								Face.texIndices[2] = Face.texIndices[2];
							}
							if (mNormaled)
							{
								Face.normIndices[0] = faceTexIdx[3];
								Face.normIndices[1] = Face.normIndices[0];
								Face.normIndices[2] = Face.normIndices[2];
							}
						}
					}

					quadFace.iSmoothingGroup = iSmoothingGroup;
					mFaces.push_back(quadFace);
					mMaterialIdx.push_back(iCurrentMtl);
				}
			}
			else if (0 == strcmp(strCommand, "s")) // Handle smoothing group for normal computation
			{
				fscanf_s(pInFile, "%s", strCommand, MAX_PATH);

				if (strCommand[0] >= '1' && strCommand[0] <= '9')
				{
					mHasSmoothGroup = true;
					sscanf_s(strCommand, "%d", &iSmoothingGroup);
				}
				else
					iSmoothingGroup = 0;
			}
			else if (0 == strcmp(strCommand, "mtllib"))
			{
				// Material library
				fscanf_s(pInFile, "%s", strMaterialFilename, MAX_PATH);
			}
			else if (0 == strcmp(strCommand, "usemtl"))
			{
				// Material
				char strName[MAX_PATH] = { 0 };
				fscanf_s(pInFile, "%s", strName, MAX_PATH);

				ObjMaterial currMtl = ObjMaterial(strName);
				auto itMtl = std::find(mMaterials.begin(), mMaterials.end(), currMtl);
				if (itMtl != mMaterials.end())
				{
					iCurrentMtl = itMtl - mMaterials.begin();
				}
				else
				{
					iCurrentMtl = mMaterials.size();
					mMaterials.push_back(currMtl);
				}

				mSubsetStartIdx.push_back(mFaces.size()*3);
				mSubsetMtlIdx.push_back(iCurrentMtl);
				mNumSubsets++;
			}
			else
			{
				while (!feof(pInFile) && fgetc(pInFile) != '\n');
			}
		}

		fclose(pInFile);

		// Correct subsets index
		if (mNumSubsets == 0)
		{
			mSubsetStartIdx.push_back(0);
			mNumSubsets = 1;
			mSubsetMtlIdx.push_back(0);
		}
		mSubsetStartIdx.push_back(mFaces.size() * 3);

		mVertexCount = mPositionBuf.size();
		mTriangleCount = mFaces.size();

		if (strMaterialFilename[0])
		{
			const char* path1 = strrchr(strPath, '/');
			const char* path2 = strrchr(strPath, '\\');
			int idx = (path1 ? path1 : path2) - strPath + 1;
			char strMtlPath[MAX_PATH] = { 0 };
			strncpy_s(strMtlPath, MAX_PATH, strPath, idx);
			strcat_s(strMtlPath, MAX_PATH, strMaterialFilename);

			LoadMaterialsFromMtl(strMtlPath);
		}

		if (mMaterials.size() == 0)
			mMaterials.push_back(ObjMaterial(""));

		processEdges();

		return true;
	}

	void ObjMesh::LoadMaterialsFromMtl(const char* strPath)
	{
		char strCommand[MAX_PATH] = { 0 };
		FILE* pInFile = 0;
		fopen_s(&pInFile, strPath, "rt");
		assert(pInFile);

		int itCurrMaterial = INDEX_NONE;
		while (!feof(pInFile))
		{
			fscanf_s(pInFile, "%s", strCommand, MAX_PATH);

			if (0 == strcmp(strCommand, "#"))
			{
				// Comment
			}
			else if (0 == strcmp(strCommand, "newmtl"))
			{
				// Switching active materials
				char strName[MAX_PATH] = { 0 };
				fscanf_s(pInFile, "%s", strName, MAX_PATH);

				ObjMaterial tmpMtl = ObjMaterial(strName);
				
				auto itMtl = std::find(mMaterials.begin(), mMaterials.end(), tmpMtl);
				if (itMtl != mMaterials.end())
				{
					// TODO: Switch to use iterator
					itCurrMaterial = itMtl - mMaterials.begin();
				}
				else
					itCurrMaterial = INDEX_NONE;
			}

			if (itCurrMaterial == INDEX_NONE)
				continue;

			else if (0 == strcmp(strCommand, "Kd"))
			{
				// Diffuse color
				float r, g, b;
				fscanf_s(pInFile, "%f %f %f", &r, &g, &b);
				mMaterials[itCurrMaterial].color = Color(r, g, b);
			}
			else if (0 == strcmp(strCommand, "Ks"))
			{
				// Diffuse color
				float r, g, b;
				fscanf_s(pInFile, "%f %f %f", &r, &g, &b);
				mMaterials[itCurrMaterial].specColor = Color(r, g, b);
			}
			else if (0 == strcmp(strCommand, "Tf"))
			{
				// Diffuse color
				float r, g, b;
				fscanf_s(pInFile, "%f %f %f", &r, &g, &b);
				mMaterials[itCurrMaterial].transColor = Color(r, g, b);
			}
			else if (0 == strcmp(strCommand, "d") || 0 == strcmp(strCommand, "Tr"))
			{
				// Alpha
				fscanf_s(pInFile, "%f", &mMaterials[itCurrMaterial].color.a);
			}
			else if (0 == strcmp(strCommand, "map_Kd"))
			{
				if (!mMaterials[itCurrMaterial].strTexturePath[0])
				{
					// Texture
					char strTexName[MAX_PATH] = { 0 };
					fgets(strTexName, MAX_PATH, pInFile);

					if (strTexName[strlen(strTexName) - 1] == '\n')
						strTexName[strlen(strTexName) - 1] = '\0';

					const char* path1 = strrchr(strPath, '/');
					const char* path2 = strrchr(strPath, '\\');
					int idx = (path1 ? path1 : path2) - strPath + 1;

					char strMtlPath[MAX_PATH] = { 0 };
					strncpy_s(mMaterials[itCurrMaterial].strTexturePath, MAX_PATH, strPath, idx);
					strcat_s(mMaterials[itCurrMaterial].strTexturePath, MAX_PATH, strTexName + 1);
				}
			}
			else if (0 == strcmp(strCommand, "bump"))
			{
				if (!mMaterials[itCurrMaterial].strBumpPath[0])
				{
					// Texture
					char strTexName[MAX_PATH] = { 0 };
					fgets(strTexName, MAX_PATH, pInFile);

					if (strTexName[strlen(strTexName) - 1] == '\n')
						strTexName[strlen(strTexName) - 1] = '\0';

					const char* path1 = strrchr(strPath, '/');
					const char* path2 = strrchr(strPath, '\\');
					int idx = (path1 ? path1 : path2) - strPath + 1;

					char strMtlPath[MAX_PATH] = { 0 };
					strncpy_s(mMaterials[itCurrMaterial].strBumpPath, MAX_PATH, strPath, idx);
					strcat_s(mMaterials[itCurrMaterial].strBumpPath, MAX_PATH, strTexName + 1);
				}
			}
			else
			{
				while (!feof(pInFile) && fgetc(pInFile) != '\n');
			}
		}

		fclose(pInFile);

		return;
	}

	void ObjMesh::LoadPlane(const float length)
	{
		float length_2 = length * 0.5f;

		Vector3 pt(-length_2, 0.0f, length_2);		// First point
		mPositionBuf.push_back(pt);
		mNormalBuf.push_back(Vector3::UNIT_Y);
		mTexBuf.push_back(Vector2(0.0f, 0.0f));
		pt = Vector3(-length_2, 0.0f, -length_2);	// Second point
		mPositionBuf.push_back(pt);
		mNormalBuf.push_back(Vector3::UNIT_Y);
		mTexBuf.push_back(Vector2(0.0f, 1.0f));
		pt = Vector3(length_2, 0.0f, -length_2);	// Third point
		mPositionBuf.push_back(pt);
		mNormalBuf.push_back(Vector3::UNIT_Y);
		mTexBuf.push_back(Vector2(1.0f, 1.0f));
		pt = Vector3(length_2, 0.0f, length_2);		// Fourth point
		mPositionBuf.push_back(pt);
		mNormalBuf.push_back(Vector3::UNIT_Y);
		mTexBuf.push_back(Vector2(1.0f, 0.0f));

		MeshFace face;
		int indices0[3] = { 0, 2, 1 };
		for (int i = 0; i < 3; i++)
		{
			face.posIndices[i] = indices0[i];
			face.texIndices[i] = indices0[i];
			face.normIndices[i] = indices0[i];
		}
		mFaces.push_back(face);
		int indices1[3] = { 2, 0, 3 };
		for (int i = 0; i < 3; i++)
		{
			face.posIndices[i] = indices1[i];
			face.texIndices[i] = indices1[i];
			face.normIndices[i] = indices1[i];
		}
		mFaces.push_back(face);

		mTriangleCount = 2;
		mVertexCount = 4;
		mMaterialIdx = vector<uint>(mTriangleCount, 0);
		mNormaled = mTextured = true;
		mHasSmoothGroup = false;

		mMaterials.push_back(ObjMaterial());

		mNumSubsets = 1;
		mSubsetMtlIdx.push_back(0);
		mSubsetStartIdx.push_back(0);
		mSubsetStartIdx.push_back(mTriangleCount*3);

		processEdges();
	}

	void ObjMesh::LoadSphere(const float fRadius, const int slices, const int stacks)
	{
		const float fThetaItvl = float(Math::EDX_PI) / float(stacks);
		const float fPhiItvl = float(Math::EDX_TWO_PI) / float(slices);

		float fTheta = 0.0f;
		for (int i = 0; i <= stacks; i++)
		{
			float fPhi = 0.0f;
			for (int j = 0; j <= slices; j++)
			{
				Vector3 vDir = Math::SphericalDirection(Math::Sin(fTheta), Math::Cos(fTheta), fPhi);

				Vector3 pt = fRadius * vDir;
				mPositionBuf.push_back(pt);
				mNormalBuf.push_back(vDir);
				mTexBuf.push_back(Vector2(fPhi / float(Math::EDX_TWO_PI),
									fTheta / float(Math::EDX_PI)));
				fPhi += fPhiItvl;
			}

			fTheta += fThetaItvl;
		}

		for (int i = 0; i < stacks; i++)
		{
			for (int j = 0; j < slices; j++)
			{
				MeshFace face;
				int indices0[3] = {    i    * (slices + 1) + j,
									   i    * (slices + 1) + j + 1,
									(i + 1) * (slices + 1) + j };
				for (int k = 0; k < 3; k++)
				{
					face.posIndices[k] = indices0[k];
					face.texIndices[k] = indices0[k];
					face.normIndices[k] = indices0[k];
				}
				mFaces.push_back(face);
				int indices1[3] = {    i    * (slices + 1) + j + 1,
									(i + 1) * (slices + 1) + j + 1,
									(i + 1) * (slices + 1) + j };
				for (int k = 0; k < 3; k++)
				{
					face.posIndices[k] = indices1[k];
					face.texIndices[k] = indices1[k];
					face.normIndices[k] = indices1[k];
				}
				mFaces.push_back(face);
			}
		}

		mTriangleCount = mFaces.size();
		mVertexCount = mPositionBuf.size();
		mMaterialIdx = vector<uint>(mTriangleCount, 0);
		mNormaled = mTextured = true;
		mHasSmoothGroup = false;
		mMaterials.push_back(ObjMaterial());

		mNumSubsets = 1;
		mSubsetMtlIdx.push_back(0);
		mSubsetStartIdx.push_back(0);
		mSubsetStartIdx.push_back(mTriangleCount*3);

		processEdges();
	}

	void ObjMesh::processEdges()
	{
		mEdges.numEdges = 0;
		std::map<std::pair<int, int>, std::vector<int>> edgeMap;
		for (int i = 0; i < mTriangleCount; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				int indexV1 = mFaces[i].posIndices[j];
				int indexV2 = mFaces[i].posIndices[(j + 1) % 3];
				int indexV3 = mFaces[i].posIndices[(j + 2) % 3];
				auto key = indexV1 < indexV2 ? std::make_pair(indexV1, indexV2) : std::make_pair(indexV2, indexV1);
				if (edgeMap.find(key) == edgeMap.end())
				{
					auto it = edgeMap.insert(edgeMap.end(), { key, std::vector<int>() });
					it->second.push_back(indexV3);
				}
				edgeMap[key].push_back(i);
			}
		}
		for (auto it : edgeMap)
		{
			//if (it.second.size() > 3)
			//{
			//	std::cout << "[WARN] Mesh non-manifold, contains vertex with more than 2 neighbor edges..." << std::endl;
			//}
			mEdges.indexVertex0.push_back(it.first.first);
			mEdges.indexVertex1.push_back(it.first.second);
			if (it.second.size() == 3)
			{
				//if (it.second[1] == it.second[2]);	// Duplicated faces is not allowed
				//{
				//	std::cout << "[WARN] Mesh non-manifold, contains duplicated faces..." << std::endl;
				//}
				mEdges.indexFace0.push_back(it.second[1]);
				mEdges.indexFace1.push_back(it.second[2]);
				mEdges.indexVertex2.push_back(it.second[0]);
				mEdges.numEdges++;
			}
			else
			{
				//Assert(it.second.size() == 2);	// Edge should be boundary
				mEdges.indexFace0.push_back(it.second[1]);
				mEdges.indexFace1.push_back(-1);
				mEdges.indexVertex2.push_back(it.second[0]);
				mEdges.numEdges++;
			}
		}
	}

	void ObjMesh::Release()
	{
		mPositionBuf.clear();
		mNormalBuf.clear();
		mTexBuf.clear();
		mFaces.clear();
	}
}