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

#include "Primitive.h"
#include <iostream>
#include <fstream>

namespace EDX
{
	namespace TensorRay
	{
        void TriangleMesh::LoadRawBuffer(const ObjMesh* pObjMesh, bool flat_shading)
        {
            mpObjMesh.reset(pObjMesh);
            mVertexCount = pObjMesh->GetVertexCount();
            mTriangleCount = pObjMesh->GetTriangleCount();

            // Init position buffer & texture buffer on GPU
            const vector<Vector3>& pbuffer = pObjMesh->GetVertexPositionBuf();
            Tensor<float> positionBuffer;
            positionBuffer.Resize(3, mVertexCount);
            positionBuffer.Assign((float*)pbuffer.data(), { mVertexCount, 3 });
            mPositionBufferRaw = Tensorf::Transpose(positionBuffer);
            mPositionBufferRaw = mPositionBufferRaw.Reshape(Shape({ mVertexCount }, VecType::Vec3));

            // Init texture and texture-index buffer on GPU (if textured)
            if (mTextured = pObjMesh->IsTextured())
            {
                // Texture buffer
                const vector<Vector2>& tbuffer = pObjMesh->GetVertexTextureBuf();
                mTexcoorCount = tbuffer.size();
                Tensor<float> texcoordBuffer;
                texcoordBuffer.Assign((float*)tbuffer.data(), { mTexcoorCount, 2 });
                mTexcoordBuffer = Tensorf::Transpose(texcoordBuffer);
                mTexcoordBuffer = mTexcoordBuffer.Reshape(Shape({ mTexcoorCount }, VecType::Vec2));

                vector<uint> flattenedIndicesTex;
                flattenedIndicesTex.resize(3 * size_t(mTriangleCount));
                for (size_t i = 0; i < mTriangleCount; i++)
                {
                    flattenedIndicesTex[3 * i + 0] = pObjMesh->GetFaceAt(i).texIndices[0];
                    flattenedIndicesTex[3 * i + 1] = pObjMesh->GetFaceAt(i).texIndices[1];
                    flattenedIndicesTex[3 * i + 2] = pObjMesh->GetFaceAt(i).texIndices[2];
                }
                // Texture-index buffer
                mIndexTexBuffer.Resize(3 * mTriangleCount);
                mIndexTexBuffer.Assign(flattenedIndicesTex.data(), { 3 * mTriangleCount });
            }
            else
            {
                // Assign zero texcoord
                mTexcoorCount = 1;
                mTexcoordBuffer = Zeros(Shape({ mTexcoorCount }, VecType::Vec2));
                // Texture-index buffer
                mIndexTexBuffer = Zeros(3 * mTriangleCount);
            }

            // Init position index buffer
            vector<uint> flattenedIndicesPos;
            flattenedIndicesPos.resize(3 * mTriangleCount);
            for (auto i = 0; i < mTriangleCount; i++)
            {
                flattenedIndicesPos[3 * i + 0] = pObjMesh->GetFaceAt(i).posIndices[0];
                flattenedIndicesPos[3 * i + 1] = pObjMesh->GetFaceAt(i).posIndices[1];
                flattenedIndicesPos[3 * i + 2] = pObjMesh->GetFaceAt(i).posIndices[2];
            }
            // Texture-index buffer
            mIndexPosBuffer.Assign(flattenedIndicesPos.data(), { 3 * mTriangleCount });

            if (flat_shading)
            {
                mVertexNormalCount = 0;
                mUseSmoothShadingBuffer = False(mTriangleCount);
                mIndexNormalBuffer.Resize(3 * mTriangleCount);
            }
            else if (pObjMesh->IsNormaled())
            {
                // If normal is specified in obj file, take the normal directly
                mUseSmoothShadingBuffer = True(mTriangleCount);
                const vector<Vector3>& nbuffer = pObjMesh->GetVertexNormalBuf();
                mVertexNormalCount = nbuffer.size();
                Tensor<float> normalBuffer;
                normalBuffer.Assign((float*)nbuffer.data(), { mVertexNormalCount, 3 });
                mVertexNormalBufferRaw = Tensorf::Transpose(normalBuffer);
                mVertexNormalBufferRaw = mVertexNormalBufferRaw.Reshape(Shape({ mVertexNormalCount }, VecType::Vec3));
                // vertex-normal-index buffer
                vector<uint> flattenedIndicesNormal;
                flattenedIndicesNormal.resize(3 * mTriangleCount);
                for (auto i = 0; i < mTriangleCount; i++)
                {
                    flattenedIndicesNormal[3 * i + 0] = pObjMesh->GetFaceAt(i).normIndices[0];
                    flattenedIndicesNormal[3 * i + 1] = pObjMesh->GetFaceAt(i).normIndices[1];
                    flattenedIndicesNormal[3 * i + 2] = pObjMesh->GetFaceAt(i).normIndices[2];
                }
                // Texture-index buffer
                mIndexNormalBuffer.Assign(flattenedIndicesNormal.data(), { 3 * mTriangleCount });
            }

            mEdgeInfo.numEdges = pObjMesh->getEdgeList().numEdges;
            mEdgeInfo.indexVert0.Assign(pObjMesh->getEdgeList().indexVertex0.data(), { mEdgeInfo.numEdges });
            mEdgeInfo.indexVert1.Assign(pObjMesh->getEdgeList().indexVertex1.data(), { mEdgeInfo.numEdges });
            mEdgeInfo.indexTri0.Assign(pObjMesh->getEdgeList().indexFace0.data(), { mEdgeInfo.numEdges });
            mEdgeInfo.indexTri1.Assign(pObjMesh->getEdgeList().indexFace1.data(), { mEdgeInfo.numEdges });
            mEdgeInfo.indexVert2.Assign(pObjMesh->getEdgeList().indexVertex2.data(), { mEdgeInfo.numEdges });
        }

        void Primitive::Configure()
        {
            const Tensorf identity = { Vector4::UNIT_X, Vector4::UNIT_Y, Vector4::UNIT_Z, Vector4::UNIT_W };

            Expr curWorld = identity,
                 curWorldInv = identity;

            // Set world transformation from raw transform (read from xml)
            if (!mWorldTensorRaw.Empty())
            {
                Assert(!mWorldInvTensorRaw.Empty());
                curWorld = mWorldTensorRaw;
                curWorldInv = mWorldInvTensorRaw;
            }

            Expr transToCenterMatrix = identity,
                 transToCenterInvMatrix = identity;
            if (!mObjCenter.Empty())
            {
                Expr tmpTrans = IndexedWrite(MakeVector4(-X(mObjCenter), -Y(mObjCenter), -Z(mObjCenter), Zeros(1)), { 3 }, Shape({ 4 }, VecType::Vec4), 0);
                transToCenterMatrix = identity + tmpTrans;
                transToCenterInvMatrix = identity - tmpTrans;
            }

            // For multi-pose optimization
            if (!mRotateMatrix.Empty())
            {
                Expr preXform = Dot(transToCenterInvMatrix, Dot(mRotateMatrix, transToCenterMatrix));
                Matrix rot = TensorToMatrix(mRotateMatrix);
                Matrix rotInv = Matrix::Inverse(rot);
                Expr preXformInv = Dot(transToCenterInvMatrix, Dot(MatrixToTensor(rotInv), transToCenterMatrix));
                curWorld = Dot(mWorldTensorRaw, preXform);
                curWorldInv = Dot(preXformInv, mWorldInvTensorRaw);
            }

            // For forward-mode auto diff visualization
            if (mTrans.Empty() && mRotateAxis.Empty())
            {
                mWorldTensor = curWorld;
                mWorldInvTensor = curWorldInv;
            }
            else
            {
                // Only one of mTrans or mRotate would be non-empty!
                if (!mTrans.Empty())
                {
                    Expr tmp = IndexedWrite(MakeVector4(X(mTrans), Y(mTrans), Z(mTrans), Zeros(1)), { 3 }, Shape({ 4 }, VecType::Vec4), 0);
                    mWorldTensor = Dot(identity + tmp, curWorld);
                    mWorldInvTensor = Dot(curWorldInv, identity - tmp);
                }
                else if (!mRotateAxis.Empty())
                {
                    Tensorf tmpRotate = CalcRotateAxisAngle(mRotateAngle, mRotateAxis);
                    Tensorf tmpRotateInv = CalcRotateAxisAngle(mRotateAngle * Scalar(-1.f), mRotateAxis);
                    mWorldTensor = Dot(tmpRotate, curWorld);
                    mWorldInvTensor = Dot(curWorldInv, tmpRotateInv);
                }
            }

            mpMesh->mPositionBuffer = TransformPoints(mpMesh->mPositionBufferRaw, mWorldTensor);

            if (mVertexId != -1 && !mVertexTrans.Empty())
            {
                Tensorf tmpTrans = Zeros(mpMesh->mPositionBufferRaw.GetShape());
                tmpTrans = IndexedWrite(mVertexTrans, { mVertexId }, tmpTrans.GetShape(), 0);
                mpMesh->mPositionBuffer = mpMesh->mPositionBuffer + tmpTrans;            
            }

            mpMesh->ComputeFaceNormalAndDistrib();
            if (!mIsFlatShading)
            {
                if (!mpMesh->mpObjMesh->IsNormaled())
                    mpMesh->ComputeVertexNormal(mpMesh->mpObjMesh.get());
                else
                    mpMesh->mVertexNormalBuffer = VectorNormalize(TransformNormals(mpMesh->mVertexNormalBufferRaw, mWorldInvTensor));
            }
        }

        // Mesh
        Primitive::Primitive(const char* path, const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading)
        {
            mIsFlatShading = flat_shading;
            mIsEmitter = false;
            bool makeLeftHanded = false;
            mTrans = pos;
            mScale = scl * (makeLeftHanded ? Tensorf({ {-1.0f}, {1.0f}, {1.0f} }) : Tensorf({ {1.0f}, {1.0f}, {1.0f} }));
            mRotate = rot;

            ObjMesh* pObjMesh = new ObjMesh;
            pObjMesh->LoadFromObj(path, makeLeftHanded);
            mpMesh = make_unique<TriangleMesh>();
            mpMesh->LoadRawBuffer(pObjMesh, flat_shading);
            mMaterialIndices = Ones((int)mpMesh->GetTriangleCount()) * Tensori({bsdf.mId});
            mWorldTensor = CalcTransform(mTrans, mScale, mRotate);
            mWorldInvTensor = CalcInvTransform(mTrans, mScale, mRotate);
            // Configure();
        }

        Primitive::Primitive(const char* path, const BSDF& bsdf, const Expr& toWorld, const Expr& toWorldInv, bool flat_shading)
        {
            mIsFlatShading = flat_shading;
            mIsEmitter = false;
            bool makeLeftHanded = false;
            ObjMesh* pObjMesh = new ObjMesh;
            pObjMesh->LoadFromObj(path, makeLeftHanded);
            mpMesh = make_unique<TriangleMesh>();
            mpMesh->LoadRawBuffer(pObjMesh, flat_shading);
            mMaterialIndices = Ones((int)mpMesh->GetTriangleCount()) * Tensori({ bsdf.mId });
            mWorldTensorRaw = toWorld;
            mWorldInvTensorRaw = toWorldInv;
            // Configure();
        }

        // Sphere
        Primitive::Primitive(const float radius, const int slices, const int stacks,
                             const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading)
        {
            mIsFlatShading = flat_shading;
            mIsEmitter = false;
            mTrans = pos;
            mScale = scl;
            mRotate = rot;

            ObjMesh* pObjMesh = new ObjMesh;
            pObjMesh->LoadSphere(radius, slices, stacks);
            mpMesh = make_unique<TriangleMesh>();
            mpMesh->LoadRawBuffer(pObjMesh, flat_shading);
            mMaterialIndices = Ones((int)mpMesh->GetTriangleCount()) * Tensori({ bsdf.mId });
            mWorldTensor = CalcTransform(mTrans, mScale, mRotate);
            mWorldInvTensor = CalcInvTransform(mTrans, mScale, mRotate);
            // Configure();
        }

        // Plane
        Primitive::Primitive(const int width, const BSDF& bsdf, const Tensorf& pos, const Tensorf& scl, const Tensorf& rot, bool flat_shading)
        {
            mIsFlatShading = flat_shading;
            mIsEmitter = false;
            mTrans = pos;
            mScale = scl;
            mRotate = rot;

            ObjMesh* pObjMesh = new ObjMesh;
            pObjMesh->LoadPlane(width);
            mpMesh = make_unique<TriangleMesh>();
            mpMesh->LoadRawBuffer(pObjMesh, flat_shading);
            mMaterialIndices = Ones((int)mpMesh->GetTriangleCount()) * Tensori({ bsdf.mId });
            mWorldTensor = CalcTransform(mTrans, mScale, mRotate);
            mWorldInvTensor = CalcInvTransform(mTrans, mScale, mRotate);
            // Configure();
        }

        Primitive::Primitive(const int width, const BSDF& bsdf, const Expr& toWorld, const Expr& toWorldInv, bool flat_shading)
        {
            mIsFlatShading = flat_shading;
            mIsEmitter = false;
            ObjMesh* pObjMesh = new ObjMesh;
            pObjMesh->LoadPlane(width);
            mpMesh = make_unique<TriangleMesh>();
            mpMesh->LoadRawBuffer(pObjMesh, flat_shading);
            mMaterialIndices = Ones((int)mpMesh->GetTriangleCount()) * Tensori({ bsdf.mId });
            mWorldTensorRaw = toWorld;
            mWorldInvTensorRaw = toWorldInv;
        }

        void TriangleMesh::ComputeFaceNormalAndDistrib()
        {
            auto triId0 = Scalar(3) * Tensorf::ArrayRange(mTriangleCount);
            auto triId1 = triId0 + Scalar(1);
            auto triId2 = triId0 + Scalar(2);
            auto indices0 = IndexedRead(mIndexPosBuffer, triId0, 0);
            auto indices1 = IndexedRead(mIndexPosBuffer, triId1, 0);
            auto indices2 = IndexedRead(mIndexPosBuffer, triId2, 0);
            auto position0 = IndexedRead(mPositionBuffer, indices0, 0);
            auto position1 = IndexedRead(mPositionBuffer, indices1, 0);
            auto position2 = IndexedRead(mPositionBuffer, indices2, 0);
            auto edge0 = position1 - position0;
            auto edge1 = position2 - position0;
            auto crossP = VectorCross(edge0, edge1);
            auto length = VectorLength(crossP);
            mFaceNormalBuffer = crossP / length;
            mTriangleAreaBuffer = length * Scalar(0.5f);
            Tensorf areaTot = Sum(mTriangleAreaBuffer);
            mTotalArea = areaTot.Get(0);
            Assert(mTotalArea > 0.0f);
            mInvTotArea = 1.0 / mTotalArea;
            mpDist = make_unique<Distribution1D>(mTriangleAreaBuffer);
        }

        void TriangleMesh::ComputeVertexNormal(const ObjMesh* pObjMesh)
        {
            // In the current implementation, we always averaging neighboring triangle normals (smooth group is ignored)
            mVertexNormalCount = mVertexCount;
            mUseSmoothShadingBuffer = True(mTriangleCount);
            mIndexNormalBuffer = mIndexPosBuffer;
            auto triID0 = Scalar(3) * Tensorf::ArrayRange(mTriangleCount);
            mVertexNormalBuffer = Zeros(Shape({ mVertexCount }, VecType::Vec3));
            Tensori neighborFaceCount = Zeros(mVertexCount);
            for (int i = 0; i < 3; i++)
            {
                auto triId = triID0 + Scalar(i);
                auto indices = IndexedRead(mIndexPosBuffer, triId, 0);
                mVertexNormalBuffer = mVertexNormalBuffer + IndexedWrite(mFaceNormalBuffer, indices, mVertexNormalBuffer.GetShape(), 0);
                neighborFaceCount = neighborFaceCount + IndexedWrite(Ones(mTriangleCount), indices, neighborFaceCount.GetShape(), 0);
            }
            mVertexNormalBuffer = VectorNormalize(mVertexNormalBuffer / neighborFaceCount);
        }

        void TriangleMesh::SamplePosition(const Expr& samples, PositionSample& lightSample)
        {
            Tensorf pdf;
            auto samples_x = X(samples);
            auto samples_y = Y(samples);
            Tensori triangleID;
            mpDist->SampleDiscrete(samples_x, &triangleID, &pdf);
            auto sample_reuse = mpDist->ReuseSample(samples_x, triangleID);
            auto t = Sqrt(sample_reuse);
            // Convert a uniformly distributed square sample into barycentric coordinate
            auto baryU = Scalar(1.0f) - t;
            auto baryV = t * samples_y;
            auto triId0 = Scalar(3) * triangleID;
            auto triId1 = Scalar(3) * triangleID + Scalar(1);
            auto triId2 = Scalar(3) * triangleID + Scalar(2);
            auto indices_pos0 = IndexedRead(mIndexPosBuffer, triId0, 0);
            auto indices_pos1 = IndexedRead(mIndexPosBuffer, triId1, 0);
            auto indices_pos2 = IndexedRead(mIndexPosBuffer, triId2, 0);
            auto position0 = IndexedRead(mPositionBuffer, indices_pos0, 0);
            auto position1 = IndexedRead(mPositionBuffer, indices_pos1, 0);
            auto position2 = IndexedRead(mPositionBuffer, indices_pos2, 0);
            auto baryW = Scalar(1.0f) - baryU - baryV;
            auto areaAD = IndexedRead(mTriangleAreaBuffer, triangleID, 0);
            lightSample.mNumSample = samples->GetShape()[0];
            lightSample.p = baryW * position0 + baryU * position1 + baryV * position2;
            lightSample.n = IndexedRead(mFaceNormalBuffer, triangleID, 0);
            lightSample.J = areaAD / Detach(areaAD);
            lightSample.pdf = Tensorf({ mInvTotArea }) * Ones(samples->GetShape()[0]);
        }

        // python binding

        void Primitive::DiffTranslation()
        {
            mTrans = Tensorf({ Vector3(0.f, 0.f, 0.f) }, true);
        }

        void Primitive::DiffRotation(const Tensorf& rotateAxis)
        {
            mRotateAxis = rotateAxis;
            mRotateAngle = Tensorf({ 0.f }, true);
        }

        void Primitive::DiffAllVertexPos()
        {
            mpMesh->mPositionBufferRaw.SetRequiresGrad(true);
        }

        int Primitive::GetVertexCount()
        {
            return mpMesh->GetVertexCount();
        }

        int Primitive::GetFaceCount()
        {
            return mpMesh->GetTriangleCount();
        }

        int Primitive::GetEdgeCount()
        {
            return mpMesh->mEdgeInfo.numEdges;
        }

        void Primitive::GetVertexPos(ptr_wrapper<float> vPosPtr)
        {
            Tensorf vertexPositionBufferRaw = Tensorf::Transpose(mpMesh->mPositionBufferRaw); // Should it be mPositionBufferRaw?
            auto val = vertexPositionBufferRaw.HostData();
            std::copy(val, val + vertexPositionBufferRaw.LinearSize(), vPosPtr.get());
        }

        void Primitive::GetVertexGrad(ptr_wrapper<float> vGradPtr)
        {
            Tensorf vertexPositionBufferRaw = Tensorf::Transpose(mpMesh->mPositionBufferRaw);
            auto gradVal = Tensorf(vertexPositionBufferRaw.Grad()).HostData();
            std::copy(gradVal, gradVal + vertexPositionBufferRaw.LinearSize(), vGradPtr.get());
        }

        void Primitive::GetFaceIndices(ptr_wrapper<int> fPtr)
        {
            int faceCount = GetFaceCount();
            int indexCount = 3 * faceCount;
            vector<int> faceIndices;
            faceIndices.resize(indexCount);
            for (size_t i = 0; i < faceCount; i++)
            {
                faceIndices[3 * i + 0] = mpMesh->mpObjMesh->GetFaceAt(i).posIndices[0];
                faceIndices[3 * i + 1] = mpMesh->mpObjMesh->GetFaceAt(i).posIndices[1];
                faceIndices[3 * i + 2] = mpMesh->mpObjMesh->GetFaceAt(i).posIndices[2];
            }
            int* val = (int*)(faceIndices.data());
            std::copy(val, val + indexCount, fPtr.get());
        }

        void Primitive::GetEdgeData(ptr_wrapper<int> edgePtr)
        {
            int numEdges = mpMesh->mEdgeInfo.numEdges;
            int* indexVert0Val = mpMesh->mEdgeInfo.indexVert0.HostData();
            std::copy(indexVert0Val, indexVert0Val + numEdges, &edgePtr[0]);
            auto indexVert1Val = mpMesh->mEdgeInfo.indexVert1.HostData();
            std::copy(indexVert1Val, indexVert1Val + numEdges, &edgePtr[numEdges]);
        }

        Tensorf Primitive::GetObjCenter()
        {
            Expr pMin = Detach(Min(mpMesh->mPositionBufferRaw, { 0 }));
            Expr pMax = Detach(Max(mpMesh->mPositionBufferRaw, { 0 }));
            Tensorf res = Scalar(0.5f) * (pMin + pMax);
            return res;
        }

        void Primitive::ExportMesh(const char* filename)
        {
            Tensorf vpos = Tensorf::Transpose(mpMesh->mPositionBufferRaw);
            Tensorui triIdx = mpMesh->mIndexPosBuffer;

            std::ofstream myfile;
            myfile.open(filename);
            for (int i = 0; i < vpos.GetShape(0); i++)
            {
                myfile << "v " << vpos.Get(i * 3) << " " << vpos.Get(i * 3 + 1) << " " << vpos.Get(i * 3 + 2) << std::endl;
            }

            for (int i = 0; i < triIdx.LinearSize() / 3; i++)
            {
                myfile << "f " << triIdx.Get(i * 3) + 1 << " " << triIdx.Get(i * 3 + 1) + 1 << " " << triIdx.Get(i * 3 + 2) + 1 << std::endl;
            }
            myfile.close();
        }
	}
}