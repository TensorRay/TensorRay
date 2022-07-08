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

#include "Integrator.h"

#define COLOR_MAP_GRADIENT 0
#define REMOVE_NAN 1

namespace EDX
{
	namespace TensorRay
	{
		Expr ColorMapGradient(const Tensorf& grad)
		{
			auto clampedGrad = Minimum(Maximum(grad + Scalar(0.5f), Scalar(0.0f)), Scalar(1.0f));

			// Red/Blue/White
			auto ratio = clampedGrad;
			auto r = Lerp(Scalar(2.0f), Scalar(0.0f), ratio);
			auto b = Lerp(Scalar(0.0f), Scalar(2.0f), ratio);
			auto ret = Where(ratio >= Scalar(0.0f) && ratio < Scalar(0.5f),
				MakeVector3(b, b, Ones(1), 0),
				MakeVector3(Ones(1), r, r, 0));

			return ret;
		}

		Expr EvalRadianceEmitted(const Scene& scene, const Ray& rays, const Intersection& its)
		{
			Expr ret = Zeros(Shape({ rays.mNumRays }, VecType::Vec3));
			IndexMask is_emitter = its.mEmitterId != Scalar(-1);
			if (is_emitter.sum > 0) 
			{
				Intersection its_emitter = its.GetMaskedCopy(is_emitter, true);
				Expr throughput = Mask(rays.mThroughput, is_emitter, 0);
				Expr dir = Mask(-rays.mDir, is_emitter, 0);
				Expr val = throughput * scene.mLights[scene.mAreaLightIndex]->Eval(its_emitter, dir);
				ret = ret + IndexedWrite(val, is_emitter.index, ret->GetShape(), 0);
			}

#if REMOVE_NAN
			return NaNToZero(ret);
#else
			return ret;
#endif
		}

		Expr EvalImportance(const Scene& scene, const Camera& camera, const Ray& rays, const Intersection& its,
			Ray& raysNext, Intersection& itsNext, Expr& pixelCoor, Tensori& rayIdx)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			Expr ret = Zeros(Shape({ rays.mNumRays }, VecType::Vec3));

			SensorDirectSample sds = camera.sampleDirect(its.mPosition);
			Tensorf d_mPosition = its.mPosition;
			IndexMask mask_visible = IndexMask(sds.isValid);		// Inside the screen

			if (mask_visible.sum > 0)
			{
                Ray primaryRay;
                camera.GenerateBoundaryRays(sds, primaryRay);
                auto dist2cam = VectorLength(Mask(its.mPosition, mask_visible, 0) - camera.mPosTensor);
                primaryRay.mMax = dist2cam - Scalar(SHADOW_EPSILON);
                Tensorb isVisible;
                scene.Occluded(primaryRay, isVisible);
                IndexMask mask_samepoint = Tensori(isVisible);
                if (mask_samepoint.sum > 0)
                {
                    mask_visible = IndexMask(IndexedWrite(mask_samepoint.mask, mask_visible.index, mask_visible.mask.GetShape(), 0));
                    pixelCoor = Mask(sds.q, mask_visible, 0);
                    rayIdx = Mask(rays.mRayIdx, mask_visible, 0);

                    // Contrib from current vertex
                    auto throughput = Mask(primaryRay.mThroughput, mask_samepoint, 0) * Mask(rays.mThroughput, mask_visible, 0);
                    Expr bsdfVal = Zeros(Shape({ mask_visible.sum }, VecType::Vec3));
                    Intersection its_visible = its.GetMaskedCopy(mask_visible, true);
                    Tensorf visDir = Mask(-rays.mDir, mask_visible, 0);
                    for (int iBSDF = 0; iBSDF < scene.mBSDFCount; iBSDF++)
                    {
                        auto woNoLeak = scene.mBsdfs[iBSDF]->OutDirValid(its_visible, visDir);
                        IndexMask mask_bsdf = (its_visible.mBsdfId == Scalar(iBSDF)) * woNoLeak;
                        if (mask_bsdf.sum == 0) continue;
                        Intersection its_bsdf = its_visible.GetMaskedCopy(mask_bsdf, true);
                        auto wo = Mask(visDir, mask_bsdf, 0);
                        {
                            auto wi = VectorNormalize(camera.mPosTensor - its_bsdf.mPosition);
                            auto wiNoLeak = scene.mBsdfs[iBSDF]->OutDirValid(its_bsdf, wi);
                            auto val = scene.mBsdfs[iBSDF]->Eval(its_bsdf, wo, wi, true) * Abs(VectorDot(its_bsdf.mNormal, wi)) * wiNoLeak;
                            bsdfVal = bsdfVal + IndexedWrite(val, mask_bsdf.index, bsdfVal->GetShape(), 0);
                        }
                    }
                    ret = throughput * bsdfVal;
                }
			}

			auto rnd_bsdf = Tensorf::RandomFloat(Shape({ rays.mNumRays }, VecType::Vec3));
			// Further trace the path towards the camera
			for (int iBSDF = 0; iBSDF < scene.mBSDFCount; iBSDF++)
			{
				auto woNoLeak = scene.mBsdfs[iBSDF]->OutDirValid(its, -rays.mDir);
				IndexMask mask_bsdf = (its.mBsdfId == Scalar(iBSDF)) * woNoLeak;
				if (mask_bsdf.sum == 0) continue;
				Intersection its_bsdf = its.GetMaskedCopy(mask_bsdf, true);
				auto wo = Mask(-rays.mDir, mask_bsdf, 0);
				Expr wi;
				Expr bsdfPdf;
				scene.mBsdfs[iBSDF]->SampleOnly(its_bsdf, wo, Mask(rnd_bsdf, mask_bsdf, 0), &wi, &bsdfPdf);
				auto bsdfVal = scene.mBsdfs[iBSDF]->Eval(its_bsdf, wo, wi, true) * Abs(VectorDot(its_bsdf.mNormal, wi));

				Ray scatteredRays;
				Intersection scatteredIts;
				scatteredRays.mNumRays = mask_bsdf.sum;
				scatteredRays.mOrg = Tensorf(its_bsdf.mPosition);
				scatteredRays.mDir = Tensorf(wi);
				scatteredRays.mPrevPdf = Detach(bsdfPdf);
				scatteredRays.mSpecular = make_shared<ConstantExp<bool>>(scene.mBsdfs[iBSDF]->IsDelta(), Shape({ mask_bsdf.sum }));
				scatteredRays.mMin = Ones(mask_bsdf.sum) * Scalar(SHADOW_EPSILON);
				scatteredRays.mMax = Ones(mask_bsdf.sum) * Scalar(1e32f);
				scatteredRays.mPixelIdx = Mask(rays.mPixelIdx, mask_bsdf, 0);
				scatteredRays.mRayIdx = Mask(rays.mRayIdx, mask_bsdf, 0);
				scatteredRays.mThroughput = Mask(rays.mThroughput, mask_bsdf, 0) * bsdfVal / scatteredRays.mPrevPdf;
				scene.Intersect(scatteredRays, scatteredIts);
				IndexMask mask_hit = scatteredIts.mTriangleId != Scalar(-1) &&
					bsdfPdf > Scalar(0.0f) &&
					scene.mBsdfs[iBSDF]->OutDirValid(its_bsdf, wi);
				if (mask_hit.sum > 0)
				{
					scatteredRays = scatteredRays.GetMaskedCopy(mask_hit, true);
					scatteredIts = scatteredIts.GetMaskedCopy(mask_hit);
					scene.PostIntersect(scatteredIts);
					scatteredIts.Eval();
					scatteredRays.Eval();
					raysNext.Append(scatteredRays);
					itsNext.Append(scatteredIts);
				}
			}
			
#if USE_PROFILING
			nvtxRangePop();
#endif

#if REMOVE_NAN
            return NaNToZero(ret);
#else
            return ret;
#endif
		}

		Expr EvalRadianceDirect(const Scene& scene, const Ray& rays, const Intersection& its, const Tensorf& rnd_light, const Tensorf& rnd_bsdf,
								Ray& raysNext, Intersection& itsNext)
		{
			Expr ret = Zeros(Shape({ rays.mNumRays }, VecType::Vec3));
			// Sample point on area light
			PositionSample lightSamples;
			scene.mLights[scene.mAreaLightIndex]->Sample(rnd_light, lightSamples);
			for (int iBSDF = 0; iBSDF < scene.mBSDFCount; iBSDF++) 
			{
				IndexMask mask_bsdf = its.mBsdfId == Scalar(iBSDF);
				if (mask_bsdf.sum == 0) continue;
				Intersection its_nee = its.GetMaskedCopy(mask_bsdf, true);
				PositionSample light_sample = lightSamples.GetMaskedCopy(mask_bsdf);
				Expr wi = light_sample.p - its_nee.mPosition;
				auto dist_sqr = VectorSquaredLength(wi);
				auto dist = Sqrt(dist_sqr);

				wi = wi / dist;
				auto dotShNorm = VectorDot(its_nee.mNormal, wi);
				auto dotGeoNorm = VectorDot(its_nee.mGeoNormal, wi);

				// check if the connection is valid
				Expr connectValid;
				{
					auto numStable = (dist > Scalar(SHADOW_EPSILON) && Abs(dotShNorm) > Scalar(EDGE_EPSILON));
					auto noLightLeak = ((dotShNorm * dotGeoNorm) > Scalar(0.0f));
					Expr visible;
					{
						Ray shadowRays;
						shadowRays.mNumRays = mask_bsdf.sum;
						shadowRays.mOrg = its_nee.mPosition;
						shadowRays.mDir = wi;
						shadowRays.mMin = Ones(mask_bsdf.sum) * Scalar(SHADOW_EPSILON);
						shadowRays.mMax = dist - Scalar(SHADOW_EPSILON);
						scene.Occluded(shadowRays, visible);
					}
					connectValid = numStable * noLightLeak * visible;
				}
				// evaluate the direct lighting from light sampling
				Expr throughput = Mask(rays.mThroughput, mask_bsdf, 0);
				Expr wo = Mask(-rays.mDir, mask_bsdf, 0);
				Expr G = Abs(VectorDot(light_sample.n, -wi)) / dist_sqr;
				Expr bsdfVal = scene.mBsdfs[iBSDF]->Eval(its_nee, wo, wi) * Abs(dotShNorm);
				Expr Le = scene.mLights[scene.mAreaLightIndex]->Eval(light_sample, -wi);
				Expr val = connectValid * throughput * bsdfVal * light_sample.J * G * Le / light_sample.pdf;
#if USE_MIS
				// MIS weight
				Expr mis_pdf_bsdf = Detach(scene.mBsdfs[iBSDF]->Pdf(its_nee, wo, wi) * G);
				Expr mis_weight = Square(light_sample.pdf) / (Square(light_sample.pdf) + Square(mis_pdf_bsdf));
				val = val * mis_weight;
#endif
				ret = ret + IndexedWrite(val, mask_bsdf.index, ret->GetShape(), 0);
				
				// keep tracing
				Expr wi_bsdf;
				Expr pdf_bsdf;
				scene.mBsdfs[iBSDF]->SampleOnly(its_nee, wo, Mask(rnd_bsdf, mask_bsdf, 0), &wi_bsdf, &pdf_bsdf);
				{
					Ray scatteredRays;
					Intersection scatteredIts;
					scatteredRays.mNumRays = mask_bsdf.sum;
					scatteredRays.mOrg = its_nee.mPosition;
					scatteredRays.mDir = wi_bsdf;
					scatteredRays.mPrevPdf = Detach(pdf_bsdf);
					scatteredRays.mSpecular = make_shared<ConstantExp<bool>>(scene.mBsdfs[iBSDF]->IsDelta(), Shape({ mask_bsdf.sum }));
					scatteredRays.mMin = Ones(mask_bsdf.sum) * Scalar(SHADOW_EPSILON);
					scatteredRays.mMax = Ones(mask_bsdf.sum) * Scalar(1e32f);
					scatteredRays.mPixelIdx = Mask(rays.mPixelIdx, mask_bsdf, 0);
					scatteredRays.mRayIdx = Mask(rays.mRayIdx, mask_bsdf, 0);
					scatteredRays.mThroughput = Mask(rays.mThroughput, mask_bsdf, 0);
					scene.Intersect(scatteredRays, scatteredIts);

					IndexMask mask_hit = scatteredIts.mTriangleId != Scalar(-1) &&
						pdf_bsdf > Scalar(0.0f) &&
						scene.mBsdfs[iBSDF]->OutDirValid(its_nee, wi_bsdf);
					if (mask_hit.sum > 0)
					{
						scatteredRays = scatteredRays.GetMaskedCopy(mask_hit, true);
						scatteredIts = scatteredIts.GetMaskedCopy(mask_hit);
						its_nee = its_nee.GetMaskedCopy(mask_hit, true);
						scene.PostIntersect(scatteredIts);
						auto dir = scatteredIts.mPosition - scatteredRays.mOrg;
						auto dist2 = VectorSquaredLength(dir);
						scatteredRays.mDir = dir / Sqrt(dist2);
						Expr G = Abs(VectorDot(scatteredIts.mGeoNormal, -scatteredRays.mDir)) / dist2;
						auto bsdfVal = scene.mBsdfs[iBSDF]->Eval(its_nee, Mask(wo, mask_hit, 0), scatteredRays.mDir);
						bsdfVal = bsdfVal * Abs(VectorDot(its_nee.mNormal, scatteredRays.mDir));
						scatteredRays.mPrevPdf = scatteredRays.mPrevPdf * Detach(G);
						scatteredRays.mThroughput = scatteredRays.mThroughput * bsdfVal * G * scatteredIts.mJ / scatteredRays.mPrevPdf;
						scatteredRays.mThroughput = Where(G > Scalar(0.f), scatteredRays.mThroughput, Scalar(0.f));

#if USE_MIS
						IndexMask mask_hit_emitter = scatteredIts.mEmitterId != Scalar(-1);
						if (mask_hit_emitter.sum > 0)
						{
							Intersection its_emitter = scatteredIts.GetMaskedCopy(mask_hit_emitter, true);
							Ray rays_hit = scatteredRays.GetMaskedCopy(mask_hit_emitter, true);
							Expr val = rays_hit.mThroughput * scene.mLights[scene.mAreaLightIndex]->Eval(its_emitter, -rays_hit.mDir);

							// MIS weight
							Expr pdf_bsdf = Detach(Mask(scatteredRays.mPrevPdf, mask_hit_emitter, 0));
							Expr pdf_nee = Detach(scene.mLights[scene.mAreaLightIndex]->Pdf(rays_hit.mOrg, its_emitter));
							Expr mis_weight = Square(pdf_bsdf) / (Square(pdf_bsdf) + Square(pdf_nee));
							val = val * mis_weight;

							//Expr val1 = IndexedWrite(val, mask_hit_emitter.index, ret->GetShape(), 0);
							//Expr val2 = IndexedWrite(val1, mask_hit.index, ret->GetShape(), 0);
							//Expr val3 = IndexedWrite(val2, mask_bsdf.index, ret->GetShape(), 0);
							Expr hit_emitter_index = IndexedRead(IndexedRead(mask_bsdf.index, mask_hit.index, 0), mask_hit_emitter.index, 0);
							Expr val3 = IndexedWrite(val, hit_emitter_index, ret->GetShape(), 0);
							ret = ret + val3;
						}
#endif

						raysNext.Append(scatteredRays);
						itsNext.Append(scatteredIts);
					}
				}
			}

#if REMOVE_NAN
            return NaNToZero(ret);
#else
            return ret;
#endif
		}

		void GradiantImageHandler::InitGradient(int resX, int resY) 
		{
			mResX = resX; mResY = resY;
			int nDeriv = ParameterPool::GetHandle().size();
#if COLOR_MAP_GRADIENT
			for (int i = 0; i < ParameterPool::GetHandle().size(); i++)
				mGradientImages.push_back(Zeros(Shape({ camera.GetFilmSizeX() * camera.GetFilmSizeY() }, VecType::Vec3)));
#else
			int gradIdx = 0;
			for (auto& it : ParameterPool::GetHandle()) 
			{
				if (mGradientIndexMap.find(gradIdx++) != mGradientIndexMap.end())
					mGradientImages.push_back(Zeros(Shape({ resX * resY }, VecType::Vec3)));
				else 
				{
					int nElement = it.ptr->GetShape().LinearSize();
					mGradientImages.push_back(Zeros(Shape({ resX * resY * nElement}, VecType::Vec3)));
				}
			}
#endif
		}

		void ExportDeriv(const Tensorf& deriv, int resX, const std::string& fn) 
		{
#if COLOR_MAP_GRADIENT
			Tensorf gradientImg = ColorMapGradient(deriv);
#else
			int resY = deriv.VectorLinearSize() / resX;
			Tensorf gradientImg = MakeVector3(X(deriv), Zeros(deriv.VectorLinearSize()), Zeros(deriv.VectorLinearSize()));
#endif
			SaveEXR((float*)Tensorf::Transpose(gradientImg).HostData(), resX, resY, fn.c_str());
		}

		void GradiantImageHandler::AccumulateDeriv(const Tensorf& target) 
		{
			if (mGradientImages.size() > 0)
			{
				int gradImgIdx = 0;
				for (auto& it : ParameterPool::GetHandle())
				{
					Tensorf& param = *dynamic_cast<Tensorf*>(it.ptr.get());
					ForwardDiffVariableCache::GetHandle().clear();
					Tensorf diff_i;
					if (mGradientIndexMap.find(gradImgIdx) != mGradientIndexMap.end())
						diff_i = Detach(target.ForwardDiff(param.Data(), mGradientIndexMap[gradImgIdx]));
					else 
					{
						int nElement = it.ptr->GetShape().LinearSize();
						diff_i = Detach(target.ForwardDiff(param.Data(), 0));
						for (int i = 1; i < nElement; i++) 
						{
							ForwardDiffVariableCache::GetHandle().clear();
							diff_i = Concat(diff_i, Detach(target.ForwardDiff(param.Data(), i)), 0);
						}
					}
					Tensorf& accumGradImg = mGradientImages[gradImgIdx++];
#if COLOR_MAP_GRADIENT
					accumGradImg += Tensorf(Luminance(diff_i));
#else
#if REMOVE_NAN
					Expr isFinite = IsFinite(X(diff_i)) && IsFinite(Y(diff_i)) && IsFinite(Z(diff_i));
					accumGradImg = Where(isFinite, accumGradImg + diff_i, accumGradImg);
#else
					accumGradImg = accumGradImg + diff_i;
#endif
#endif
				}
			}
		}
	
		void GradiantImageHandler::GetGradientImages(Tensorf& gradImage) const {
			gradImage = mGradientImages[0];
			for (int i = 1; i < mGradientImages.size(); i++)
				gradImage = Concat(gradImage, mGradientImages[i], 0);
		}
	}
}
