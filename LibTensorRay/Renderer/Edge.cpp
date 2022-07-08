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

#include "Edge.h"
#include "Scene.h"
#include "Camera.h"
#include "Integrator.h"

using namespace EDX;
using namespace EDX::DeepLearning;
namespace EDX
{
	namespace TensorRay
	{
		int ConstructSecEdgeList(const Scene& scene, SecondaryEdgeInfo& list)
		{
			list.numTot = 0;
			for (int iMesh = 0; iMesh < scene.mPrims.size(); iMesh++)
			{
				const Tensorf& vertexBuffer = scene.mPrims[iMesh]->mpMesh->mPositionBuffer;
				const Tensorf& faceNormalBuffer = scene.mPrims[iMesh]->mpMesh->mFaceNormalBuffer;
				const EdgeIndexInfo& edgeInfo = scene.mPrims[iMesh]->mpMesh->mEdgeInfo;
				auto isBoundaryAll = edgeInfo.indexTri1 == Scalar(-1);
				IndexMask notBoundary = edgeInfo.indexTri1 != Scalar(-1);
				
				auto normal0 = IndexedRead(faceNormalBuffer, edgeInfo.indexTri0, 0);
				auto normal1 = Where(isBoundaryAll, Zeros(Shape({ edgeInfo.numEdges }, VecType::Vec3)), IndexedRead(faceNormalBuffer, edgeInfo.indexTri1 * notBoundary.mask, 0));
				auto p0All = IndexedRead(vertexBuffer, edgeInfo.indexVert0, 0);
				auto p2All = IndexedRead(vertexBuffer, edgeInfo.indexVert2, 0);
				Tensori validNotBoundary;
				if (notBoundary.sum > 0)
				{
					// Check if two face is coplanar
					auto normal1_nb = Mask(normal1, notBoundary, 0);
					auto valid0 = VectorDot(Mask(normal0, notBoundary, 0), normal1_nb) < Scalar(1.0f - 1e-6f);
					// Check if two face is convex (exception when surface is transmissive)
					auto edge2 = Mask(VectorNormalize(p2All - p0All), notBoundary, 0);
					auto valid1 = VectorDot(normal1_nb, edge2) < Scalar(-EDGE_EPSILON);
					validNotBoundary = valid0 && valid1;
					validNotBoundary = validNotBoundary.Reshape(notBoundary.sum);
				}
				IndexMask validEdge = (notBoundary.sum > 0) ? IndexedWrite(validNotBoundary, notBoundary.index, notBoundary.mask.GetShape(), 0) + isBoundaryAll : isBoundaryAll;
				if (validEdge.sum > 0)
				{
					auto p0 = Tensorf(Mask(p0All, validEdge, 0));
					auto e1 = Tensorf(IndexedRead(vertexBuffer, Mask(edgeInfo.indexVert1, validEdge, 0), 0) - p0);
					auto n0 = Tensorf(Mask(normal0, validEdge, 0));
					auto n1 = Tensorf(Mask(normal1, validEdge, 0));
					auto p2 = Tensorf(Mask(p2All, validEdge, 0));
					auto isBoundary = Mask(isBoundaryAll, validEdge, 0);
					list.p0 = list.numTot == 0 ? p0 : Concat(list.p0, p0, 0);
					list.e1 = list.numTot == 0 ? e1 : Concat(list.e1, e1, 0);
					list.n0 = list.numTot == 0 ? n0 : Concat(list.n0, n0, 0);
					list.n1 = list.numTot == 0 ? n1 : Concat(list.n1, n1, 0);
					list.p2 = list.numTot == 0 ? p2 : Concat(list.p2, p2, 0);
					list.isBoundary = list.numTot == 0 ? isBoundary : Concat(list.isBoundary, isBoundary, 0);
					list.numTot += validEdge.sum;
				}
			}
			return list.numTot;
		}

		Tensori SampleFromSecEdges(const SecondaryEdgeInfo& list, const Tensorf& rnd1, BoundarySegSampleSecondary& samples)
		{
			auto edgeLength = VectorLength(list.e1);
			Distribution1D secEdgeDistrb(edgeLength);
			Tensori edgeIdx;
			Tensorf edgePdf, pdfFinal;
			secEdgeDistrb.SampleDiscrete(rnd1, &edgeIdx, &edgePdf);
			Expr samples_reuse = secEdgeDistrb.ReuseSample(rnd1, edgeIdx);
			auto p0 = IndexedRead(list.p0, edgeIdx, 0);
			auto e1 = IndexedRead(list.e1, edgeIdx, 0);
			auto p2 = IndexedRead(list.p2, edgeIdx, 0);
			samples.pdf = edgePdf / Detach(IndexedRead(edgeLength, edgeIdx, 0));
			samples.p0 = p0 + samples_reuse * e1;
			samples.edge = VectorNormalize(e1);
			samples.edge2 = VectorNormalize(Detach(p2) - Detach(p0));
			return edgeIdx;
		}

		int SampleBoundarySegmentDirect(const Scene& scene, const SecondaryEdgeInfo &secEdges, int numSamples, const Tensorf& rnd_b, const Tensorf& pdf_b, BoundarySegSampleDirect& secEdgeSamples, bool guiding_mode)
		{
			auto samples_x = X(rnd_b);
			auto samples_y = Y(rnd_b);
			auto samples_z = Z(rnd_b);

			Tensori edgeIdx = SampleFromSecEdges(secEdges, samples_x, secEdgeSamples);

			// Sample a point on the light sources
			auto rnd_light = MakeVector2(samples_y, samples_z);
			PositionSample lightSamples;
			scene.mLights[scene.mAreaLightIndex]->Sample(rnd_light, lightSamples);

			secEdgeSamples.p2 = lightSamples.p;
			secEdgeSamples.n = lightSamples.n;
			auto e = secEdgeSamples.p2 - Detach(secEdgeSamples.p0);
			auto distSqr = VectorSquaredLength(e);
			auto dist = Sqrt(distSqr);
			auto eNormalized = e / dist;
			auto cosTheta = VectorDot(secEdgeSamples.n, -eNormalized);
			auto n0 = IndexedRead(secEdges.n0, edgeIdx, 0);
			auto n1 = IndexedRead(secEdges.n1, edgeIdx, 0);
			auto isBoundary = IndexedRead(secEdges.isBoundary, edgeIdx, 0);
			auto cosine0 = VectorDot(eNormalized, n0);
			auto cosine1 = VectorDot(eNormalized, n1);
			auto valid0 = Abs(cosine0) > Scalar(EDGE_EPSILON);
			auto valid1 = (cosine0 > Scalar(EDGE_EPSILON) && cosine1 < Scalar(-EDGE_EPSILON)) || (cosine0 < Scalar(-EDGE_EPSILON) && cosine1 > Scalar(EDGE_EPSILON));
			auto rightSide = (isBoundary && valid0) || (~isBoundary && valid1);
			auto valid2 = cosTheta > Scalar(EPSILON);
			auto valid3 = dist > Scalar(SHADOW_EPSILON);
			secEdgeSamples.pdf = secEdgeSamples.pdf * lightSamples.pdf * distSqr / cosTheta * pdf_b;
			secEdgeSamples.maskValid = IndexMask(rightSide && valid2 && valid3);

			if (!guiding_mode && secEdgeSamples.maskValid.sum > 0)
				secEdgeSamples = secEdgeSamples.getValidCopy();
			return secEdgeSamples.maskValid.sum;
		}

		int SampleBoundarySegmentIndirect(const Scene& scene, const SecondaryEdgeInfo& secEdges, int numSamples, BoundarySegSampleIndirect& samples)
		{
			// Sample position
			Tensorf rnd = Tensorf::RandomFloat(Shape({ numSamples }, VecType::Scalar1));
			Tensori edgeIdx = SampleFromSecEdges(secEdges, rnd, samples);

			// Sample direction
			IndexMask mask_boundary = IndexedRead(secEdges.isBoundary, edgeIdx, 0);
			IndexMask mask_notBoundary = Scalar(1) - mask_boundary.mask;
			auto dir = Zeros(Shape({ numSamples }, VecType::Vec3));
			auto pdf = Zeros(numSamples);
			if (mask_boundary.sum > 0)
			{
				auto n0 = IndexedRead(secEdges.n0, edgeIdx, 0);
				auto normal = VectorCross(Mask(n0, mask_boundary, 0), Mask(samples.edge, mask_boundary, 0));
				Expr tangent, bitangent;
				CoordinateSystem(normal, &tangent, &bitangent);
				auto rnd = Tensorf::RandomFloat(Shape({ mask_boundary.sum }, VecType::Vec2));
				auto z = Minimum(Ones(1) - Scalar(2.0f) * X(rnd), Scalar(1.0f-EDGE_EPSILON));
				auto r = Sqrt(Maximum(Scalar(0.0f), Ones(1) - Square(z)));
				auto phi = Scalar(2.0f * float(Math::EDX_PI)) * Y(rnd);
				auto x = r * Cos(phi);
				auto y = r * Sin(phi);
				auto dir_val = x * tangent + y * bitangent + z * normal;
				dir = dir + IndexedWrite(dir_val, mask_boundary.index, dir->GetShape(), 0);
				Tensorf pdf_val = Scalar(0.25f / float(Math::EDX_PI)) * Ones(mask_boundary.sum);
				pdf = pdf + IndexedWrite(pdf_val, mask_boundary.index, pdf->GetShape(), 0);
			}
			if (mask_notBoundary.sum > 0) 
			{
				auto n0 = Mask(IndexedRead(secEdges.n0, edgeIdx, 0), mask_notBoundary, 0);
				auto n1 = Mask(IndexedRead(secEdges.n1, edgeIdx, 0), mask_notBoundary, 0);
				auto phi0 = Acos(VectorDot(n0, n1));
				auto pdf_val = Scalar(0.25f)/phi0 * Ones(mask_notBoundary.sum);
				pdf = pdf + IndexedWrite(pdf_val, mask_notBoundary.index, pdf->GetShape(), 0);

				auto z = VectorNormalize(n0 + n1);
				auto y = VectorNormalize(VectorCross(n0, z));
				auto x = VectorCross(y, z);
				auto rnd = Tensorf::RandomFloat(Shape({ mask_notBoundary.sum }, VecType::Vec2));
				auto phi = (X(rnd) - Scalar(0.5f)) * phi0;
				phi = Minimum(Maximum(phi, Scalar(-0.5f) * phi0 + Scalar(EDGE_EPSILON)),
					          Scalar(0.5f) * phi0 - Scalar(EDGE_EPSILON));
				auto x1 = x * Cos(phi) + z * Sin(phi);
				IndexMask mask0 = Y(rnd) > Scalar(0.5f);
				auto b = Scalar(4.0f) * Y(rnd) - Where(mask0.mask, Scalar(3.0f), Scalar(1.0f));
				auto a = Sqrt(Scalar(1.0f) - Square(b)) * (Scalar(1.0f) - mask0.mask * Scalar(2.0f));
				auto dir_val = x1 * a + y * b;
				dir = dir + IndexedWrite(dir_val, mask_notBoundary.index, dir->GetShape(), 0);
			}
			samples.pdf = samples.pdf * Detach(pdf);
			samples.dir = Detach(dir);
			return numSamples;
		}

		int EvalBoundarySegmentDirect(const Camera& camera, const Scene& scene, int mSpp, int mMaxBounce, BoundarySegSampleDirect& bss, Tensorf& boundaryTerm, bool guiding_mode) 
		{
			// Step 2: Compute the contrib from valid boundary segments (AD)
			Tensorf rayDir = Detach(VectorNormalize(bss.p0 - bss.p2));
			Ray ray(bss.p0, rayDir);
			Intersection its;
			Expr emittedRadiance, baseVal, xDotN;
			{
				Intersection its0;
				Ray ray0(bss.p0, -rayDir);
				scene.Intersect(ray0, its0);
				scene.Intersect(ray, its);
				auto hitP = ray0.mOrg + its0.mTHit * ray0.mDir;
				Tensorb samePoint = VectorLength(hitP - bss.p2) < Scalar(SHADOW_EPSILON);
				bss.maskValid = IndexMask(samePoint && its0.mTriangleId != Scalar(-1)
					&& its.mTriangleId != Scalar(-1));
				if (bss.maskValid.sum == 0)
				{
					return 0;
				}

				ray = ray.GetMaskedCopy(bss.maskValid);
				its0 = its0.GetMaskedCopy(bss.maskValid);
				its = its.GetMaskedCopy(bss.maskValid);
				Tensori validIndex;
				if (guiding_mode) 
				{
					validIndex = bss.maskValid.index;
				}
				bss = bss.getValidCopy();

				emittedRadiance = Zeros(Shape({ ray.mNumRays }, VecType::Vec3));
				IndexMask mask_light = its0.mEmitterId != Scalar(-1);
				if (mask_light.sum > 0) 
				{
					Intersection its_light = its0.GetMaskedCopy(mask_light);
					auto dir = Mask(ray.mDir, mask_light, 0);
					scene.PostIntersect(its_light);
					auto val = Detach(scene.mLights[scene.mAreaLightIndex]->Eval(its_light, dir));
					emittedRadiance = emittedRadiance + IndexedWrite(val, mask_light.index, emittedRadiance->GetShape(), 0);
				}

				scene.PostIntersect(its);
				Tensorf d_position = its.mPosition;
				auto dist = VectorLength(bss.p2 - its.mPosition);
				auto cos2 = Abs(VectorDot(bss.n, ray.mDir));
				auto e = VectorCross(bss.edge, -ray.mDir);
				auto sinphi = VectorLength(e);
				auto proj = VectorNormalize(VectorCross(e, bss.n));
				auto sinphi2 = VectorLength(VectorCross(-ray.mDir, proj));
				auto itsT = VectorLength(its.mPosition - bss.p0);
				auto n = Detach(VectorNormalize(VectorCross(bss.n, proj)));
				auto sign0 = VectorDot(e, bss.edge2) > Scalar(0.0f);
				auto sign1 = VectorDot(e, n) > Scalar(0.0f);
				baseVal = Detach((itsT / dist) * (sinphi / sinphi2) * cos2);
				baseVal = baseVal * (sinphi > Scalar(EPSILON)) * (sinphi2 > Scalar(EPSILON));
				baseVal = baseVal * Where(sign0 == sign1, Ones(bss.pdf.GetShape()), -Ones(bss.pdf.GetShape()));

				if (guiding_mode) 
				{
					boundaryTerm = IndexedWrite(baseVal, validIndex, validIndex.GetShape(), 0);
					return 1;
				}

				auto indicesTri0 = Scalar(3) * its0.mTriangleId;
				auto indicesTri1 = Scalar(3) * its0.mTriangleId + Scalar(1);
				auto indicesTri2 = Scalar(3) * its0.mTriangleId + Scalar(2);
				Expr u, v, w, t;
				auto indicesPos0 = IndexedRead(scene.mIndexPosBuffer, indicesTri0, 0);
				auto indicesPos1 = IndexedRead(scene.mIndexPosBuffer, indicesTri1, 0);
				auto indicesPos2 = IndexedRead(scene.mIndexPosBuffer, indicesTri2, 0);
				auto position0 = IndexedRead(scene.mPositionBuffer, indicesPos0, 0);
				auto position1 = IndexedRead(scene.mPositionBuffer, indicesPos1, 0);
				auto position2 = IndexedRead(scene.mPositionBuffer, indicesPos2, 0);
				RayIntersectAD(VectorNormalize(bss.p0 - its.mPosition), its.mPosition,
					position0, position1 - position0, position2 - position0, u, v, t);
				w = Scalar(1.0f) - u - v;
				auto u2 = w * Detach(position0) + u * Detach(position1) + v * Detach(position2);
				xDotN = VectorDot(n, u2);
			}
			// Step 3: trace towards the sensor
			ray.mThroughput = Ones(Shape({ ray.mNumRays }, VecType::Vec3));
			ray.mRayIdx = Tensori::ArrayRange(ray.mNumRays);
			ray.mPrevPdf = Ones(ray.mNumRays);
			ray.mSpecular = True(ray.mNumRays);
			ray.mPixelIdx = Tensori::ArrayRange(ray.mNumRays);
			Tensorf value0 = Detach(emittedRadiance * baseVal / bss.pdf) * xDotN;

			for (int iBounce = 0; iBounce < mMaxBounce; iBounce++)
			{
				if (ray.mNumRays == 0) break;
				Ray rayNext;
				Intersection itsNext;
				Expr pixelCoor;
				Tensori rayIdx;
				Tensorf pathContrib = EvalImportance(scene, camera, ray, its, rayNext, itsNext, pixelCoor, rayIdx);

				if (rayIdx.LinearSize() > 0)
				{
					Tensorf val = Scalar(1.0f / float(mSpp)) * IndexedRead(value0, rayIdx, 0) * Detach(pathContrib);
					boundaryTerm = boundaryTerm + camera.WriteToImage(val, pixelCoor);
				}

				ray = rayNext;
				its = itsNext;
			}

			return 1;
		}


		// Primary boundary
		int ConstructPrimEdgeList(const Scene& scene, const Camera& camera, PrimaryEdgeInfo2& list)
		{
			list.numTot = 0;
			for (int iMesh = 0; iMesh < scene.mPrims.size(); iMesh++)
			{
				const Tensorf& vertexBuffer = scene.mPrims[iMesh]->mpMesh->mPositionBuffer;
				const Tensorf& faceNormalBuffer = scene.mPrims[iMesh]->mpMesh->mFaceNormalBuffer;
				const EdgeIndexInfo& edgeInfo = scene.mPrims[iMesh]->mpMesh->mEdgeInfo;
				auto normal0 = IndexedRead(faceNormalBuffer, edgeInfo.indexTri0, 0);
				auto isBoundaryAll = edgeInfo.indexTri1 == Scalar(-1);
				auto normal1 = Where(isBoundaryAll,
					Zeros(Shape({ edgeInfo.numEdges }, VecType::Vec3)),
					IndexedRead(faceNormalBuffer, edgeInfo.indexTri1 * (~isBoundaryAll), 0));  // For non-boundary edges, n1 is zero vector
				auto e = camera.mPosTensor - IndexedRead(vertexBuffer, edgeInfo.indexVert0, 0);
				auto frontFacing0 = VectorDot(e, normal0) > Scalar(0.0f);
				auto frontFacing1 = VectorDot(e, normal1) > Scalar(0.0f);
				IndexMask validEdge = (frontFacing0 != frontFacing1) * (~isBoundaryAll) + isBoundaryAll;
				if (validEdge.sum > 0)
				{
					auto p0 = Tensorf(IndexedRead(vertexBuffer, Mask(edgeInfo.indexVert0, validEdge, 0), 0));
					auto e1 = Tensorf(IndexedRead(vertexBuffer, Mask(edgeInfo.indexVert1, validEdge, 0), 0) - p0);
					auto n0 = Tensorf(Mask(normal0, validEdge, 0));
					auto n1 = Tensorf(Mask(normal1, validEdge, 0));
					auto p2 = Tensorf(IndexedRead(vertexBuffer, Mask(edgeInfo.indexVert2, validEdge, 0), 0));
					auto isBoundary = Mask(isBoundaryAll, validEdge, 0);
					list.p0 = list.numTot == 0 ? p0 : Concat(list.p0, p0, 0);
					list.e1 = list.numTot == 0 ? e1 : Concat(list.e1, e1, 0);
					list.n0 = list.numTot == 0 ? n0 : Concat(list.n0, n0, 0);
					list.n1 = list.numTot == 0 ? n1 : Concat(list.n1, n1, 0);
					list.p2 = list.numTot == 0 ? p2 : Concat(list.p2, p2, 0);
					list.isBoundary = list.numTot == 0 ? isBoundary : Concat(list.isBoundary, isBoundary, 0);
					list.numTot += validEdge.sum;
				}
			}
			return list.numTot;
		}

		Tensori SampleFromPrimEdges(const PrimaryEdgeInfo2& list, int numSamples, BoundarySegSamplePrimary& samples)
		{
			Tensorf edgeLength = VectorLength(list.e1);
			Distribution1D secEdgeDistrb(edgeLength);
			Tensorf rnd = Tensorf::RandomFloat(Shape({ numSamples }, VecType::Vec2));
			Tensori edgeIdx;
			Tensorf edgePdf, pdfFinal;
			secEdgeDistrb.SampleDiscrete(X(rnd), &edgeIdx, &edgePdf);
			auto p0 = IndexedRead(list.p0, edgeIdx, 0);
			auto e1 = IndexedRead(list.e1, edgeIdx, 0);
			auto p2 = IndexedRead(list.p2, edgeIdx, 0);
			samples.pdf = edgePdf / Detach(IndexedRead(edgeLength, edgeIdx, 0));
			samples.p0 = p0 + Y(rnd) * e1;
			samples.edge = VectorNormalize(e1);
			samples.edge2 = VectorNormalize(Detach(p2) - Detach(p0));
			samples.maskValid = IndexMask(Ones(numSamples));
			return edgeIdx;
		}

		int SampleBoundarySegmentPrimary(const Scene& scene, const PrimaryEdgeInfo2& primEdges, int numSamples, BoundarySegSamplePrimary& samples)
		{
			Tensori edgeIdx = SampleFromPrimEdges(primEdges, numSamples, samples);
			return numSamples;
		}


		// Pixel boundary
		void SampleBoundarySegmentPixel(const Camera& camera, int spp, int antitheticSpp, BoundarySegSamplePixel& samples)
		{
			int numSamples = camera.mResX * camera.mResY * spp;
			int numSamplesPerAnti = numSamples / antitheticSpp;
            samples.maskValid = IndexMask(Ones(numSamples));
            samples.rayIdx = Tensori::ArrayRange(numSamples);
            samples.pixelIdx = Tensori::ArrayRange(numSamples) % Scalar(camera.mResX * camera.mResY);

			Tensorf rnd = Tensorf::RandomFloat(Shape({ numSamplesPerAnti }, VecType::Scalar1));
			if (antitheticSpp >= 2)
				rnd = Concat(rnd, rnd, 0);
			if (antitheticSpp >= 4)
				rnd = Concat(rnd, rnd, 0);

			Expr rnd0 = rnd;
			Expr rnd1 = Where(rnd < Scalar(0.5f), rnd + Scalar(0.5f), rnd - Scalar(0.5f));
			Expr rnd2 = Where(rnd < Scalar(0.75f), Scalar(0.75f) - rnd, Scalar(1.75f) - rnd);
			Expr rnd3 = Where(rnd < Scalar(0.25f), Scalar(0.25f) - rnd, Scalar(1.25f) - rnd);

			Expr antitheticIndex = Floor(samples.rayIdx / Scalar(numSamplesPerAnti));
            Expr rnd_pixel_edge = Where(antitheticIndex == Scalar(0),
                rnd0,
                Where(antitheticIndex == Scalar(1),
                    rnd1,
                    Where(antitheticIndex == Scalar(2),
                        rnd2,
                        rnd3
                    )
                )
            );

			camera.GeneratePixelBoundarySamples(rnd_pixel_edge, samples.p0, samples.edge, samples.edge2, samples.pdf);
		}
	}
}
