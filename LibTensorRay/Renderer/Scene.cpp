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

#include "Scene.h"
#include "Windows/Bitmap.h"
#include "Windows/Timer.h"
#include <optix_stubs.h>
#include "SceneLoader.h"

namespace EDX
{
	namespace TensorRay
	{
		void Scene::LoadFromFile(const char* filename)
		{
			SceneLoader::LoadFromFile(filename, *this);
		}

		void Scene::ExtractGeometry()
		{
			mVertexCount = 0;
			mTriangleCount = 0;
			mVertexNormalCount = 0;
			mTexcoorCount = 0;
			int emitterID = 0;
			int emitterTriOffset = 0;

			mPositionBuffer = mPrims[0]->mpMesh->mPositionBuffer;
			mFaceNormalBuffer = mPrims[0]->mpMesh->mFaceNormalBuffer;
			mUseSmoothShadingBuffer = mPrims[0]->mpMesh->mUseSmoothShadingBuffer;
			mIndexPosBuffer = mPrims[0]->mpMesh->mIndexPosBuffer;
			mIndexTexBuffer = mPrims[0]->mpMesh->mIndexTexBuffer;
			mIndexNormalBuffer = mPrims[0]->mpMesh->mIndexNormalBuffer;
			mTriangleAreaBuffer = mPrims[0]->mpMesh->mTriangleAreaBuffer;
			mSbtIndexBuffer = mPrims[0]->mMaterialIndices;
			mEmitterIDBuffer = mPrims[0]->mIsEmitter ?
				Ones(mPrims[0]->mpMesh->GetTriangleCount()) * Scalar(emitterID) :
				Ones(mPrims[0]->mpMesh->GetTriangleCount()) * Scalar(-1.0f);
			mTriIdToEmitTriIdBuffer = mPrims[0]->mIsEmitter ?
				Tensori::ArrayRange(mPrims[0]->mpMesh->GetTriangleCount()) + Scalar(emitterTriOffset) :
				Ones(mPrims[0]->mpMesh->GetTriangleCount()) * Scalar(-1.0f);
			if (mPrims[0]->mIsEmitter)
			{
				emitterID++;
				emitterTriOffset += mPrims[0]->mpMesh->GetTriangleCount();
			}
			mTexcoordBuffer.Free();
			mVertexNormalBuffer.Free();
			for (auto i = 0; i < mPrims.size(); i++)
			{
				if (i > 0)
				{
					mPositionBuffer = Concat(mPositionBuffer, mPrims[i]->mpMesh->mPositionBuffer, 0);
					mFaceNormalBuffer = Concat(mFaceNormalBuffer, mPrims[i]->mpMesh->mFaceNormalBuffer, 0);
					mUseSmoothShadingBuffer = Concat(mUseSmoothShadingBuffer, mPrims[i]->mpMesh->mUseSmoothShadingBuffer, 0);
					mIndexPosBuffer = Concat(mIndexPosBuffer, mPrims[i]->mpMesh->mIndexPosBuffer + Scalar(mVertexCount), 0);
					mIndexTexBuffer = Concat(mIndexTexBuffer, mPrims[i]->mpMesh->mIndexTexBuffer + Scalar(mTexcoorCount), 0);
					mIndexNormalBuffer = Concat(mIndexNormalBuffer, mPrims[i]->mpMesh->mIndexNormalBuffer + Scalar(mVertexNormalCount), 0);
					mTriangleAreaBuffer = Concat(mTriangleAreaBuffer, mPrims[i]->mpMesh->mTriangleAreaBuffer, 0);
					mSbtIndexBuffer = Concat(mSbtIndexBuffer, mPrims[i]->mMaterialIndices, 0);
					mEmitterIDBuffer = Concat(mEmitterIDBuffer, mPrims[i]->mIsEmitter ?
						Ones(mPrims[i]->mpMesh->GetTriangleCount()) * Scalar(emitterID) :
						Ones(mPrims[i]->mpMesh->GetTriangleCount()) * Scalar(-1), 0);
					mTriIdToEmitTriIdBuffer = Concat(mTriIdToEmitTriIdBuffer, mPrims[i]->mIsEmitter ?
						Tensori::ArrayRange(mPrims[i]->mpMesh->GetTriangleCount()) + Scalar(emitterTriOffset) :
						Ones(mPrims[i]->mpMesh->GetTriangleCount()) * Scalar(-1.0f), 0);
					if (mPrims[i]->mIsEmitter)
					{
						emitterID++;
						emitterTriOffset += mPrims[i]->mpMesh->GetTriangleCount();
					}
				}

				mVertexCount += mPrims[i]->mpMesh->GetVertexCount();
				mTriangleCount += mPrims[i]->mpMesh->GetTriangleCount();

				if (mPrims[i]->mpMesh->mTexcoorCount > 0)
				{
					mTexcoorCount += mPrims[i]->mpMesh->mTexcoorCount;
					if (mTexcoordBuffer.Empty())
						mTexcoordBuffer = mPrims[i]->mpMesh->mTexcoordBuffer;
					else
						mTexcoordBuffer = Concat(mTexcoordBuffer, mPrims[i]->mpMesh->mTexcoordBuffer, 0);
				}

				if (mPrims[i]->mpMesh->mVertexNormalCount > 0)
				{
					mVertexNormalCount += mPrims[i]->mpMesh->mVertexNormalCount;
					if (mVertexNormalBuffer.Empty())
						mVertexNormalBuffer = mPrims[i]->mpMesh->mVertexNormalBuffer;
					else
						mVertexNormalBuffer = Concat(mVertexNormalBuffer, mPrims[i]->mpMesh->mVertexNormalBuffer, 0);
				}
			}

			mPositionBufferT = Tensorf::Transpose(mPositionBuffer);
		}

		void Scene::CreateSBT(OptixState& state)
		{
			// raygen
			CUdeviceptr  d_raygen_record = 0;
			const size_t raygen_record_size = sizeof(EmptyRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

			EmptyRecord rg_record;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_record));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &rg_record, raygen_record_size, cudaMemcpyHostToDevice));

			// miss
			CUdeviceptr  d_miss_record = 0;
			const size_t miss_record_size = sizeof(EmptyRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));

			EmptyRecord ms_record;
			OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_record));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &ms_record, miss_record_size, cudaMemcpyHostToDevice));

			// hit group
			std::vector<HitRecord> hitgroup_records;
			for (int i = 0; i < mBSDFCount; i++)
			{
				HitRecord rec = {};
				OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &rec));
				rec.data.bsdfId = i;
				hitgroup_records.push_back(rec);
			}

			CUdeviceptr  d_hitgroup_record = 0;
			const size_t hitgroup_record_size = sizeof(HitRecord);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hitgroup_record_size * hitgroup_records.size()));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), hitgroup_records.data(),
				hitgroup_record_size * hitgroup_records.size(), cudaMemcpyHostToDevice));

			state.sbt.raygenRecord = d_raygen_record;
			state.sbt.missRecordBase = d_miss_record;
			state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
			state.sbt.missRecordCount = 1; // RAY_TYPE_COUNT;
			state.sbt.hitgroupRecordBase = d_hitgroup_record;
			state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
			state.sbt.hitgroupRecordCount = static_cast<int>(hitgroup_records.size());
		}

		void Scene::ConstructAccelerationStructure(OptixState& state)
		{
			// Build triangle accel
			{
				OptixBuildInput triangle_input = {};
				triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

				triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
				triangle_input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
				triangle_input.triangleArray.numVertices = mVertexCount;
				const float* pVertexBufData = mPositionBufferT.Data();
				triangle_input.triangleArray.vertexBuffers = (CUdeviceptr*)(&pVertexBufData);

				triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
				triangle_input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
				triangle_input.triangleArray.numIndexTriplets = mTriangleCount;
				triangle_input.triangleArray.indexBuffer = (CUdeviceptr)mIndexPosBuffer.Data();

				vector<uint> flags;
				flags.assign(mBSDFCount, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
				triangle_input.triangleArray.flags = flags.data();

				triangle_input.triangleArray.numSbtRecords = mBSDFCount;
				triangle_input.triangleArray.sbtIndexOffsetBuffer = (CUdeviceptr)mSbtIndexBuffer.Data();
				triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint);
				triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint);

				OptixAccelBuildOptions accel_options = {};
				accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
				accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

				OptixAccelBufferSizes gas_buffer_sizes;
				OPTIX_CHECK(optixAccelComputeMemoryUsage(
					state.context,
					&accel_options,
					&triangle_input,
					1,  // num_build_inputs
					&gas_buffer_sizes
				));
				state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;

				if (state.d_temp_buffer)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_temp_buffer)));
					state.d_temp_buffer = 0;
				}
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

				// non-compacted output
				CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
				size_t      compactedSizeOffset = Math::RoundUpTo(gas_buffer_sizes.outputSizeInBytes, 8ull);
				CUDA_CHECK(cudaMalloc(
					reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
					compactedSizeOffset + 8
				));

				OptixAccelEmitDesc emitProperty = {};
				emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
				emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

				OPTIX_CHECK(optixAccelBuild(
					state.context,
					0,                                  // CUDA stream
					&accel_options,
					&triangle_input,
					1,                                  // num build inputs
					state.d_temp_buffer,
					gas_buffer_sizes.tempSizeInBytes,
					d_buffer_temp_output_gas_and_compacted_size,
					gas_buffer_sizes.outputSizeInBytes,
					&state.blas,
					&emitProperty,                      // emitted property list
					1                                   // num emitted properties
				));

				//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

				size_t compacted_gas_size;
				CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
				if (state.d_gas_output_buffer)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
					state.d_gas_output_buffer = 0;
				}
				if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
				{
					CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));
					// use handle as input and output
					OPTIX_CHECK(optixAccelCompact(state.context, 0, state.blas, state.d_gas_output_buffer, compacted_gas_size, &state.blas));
					CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
					state.gas_output_buffer_size = compacted_gas_size;
				}
				else
				{
					state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
					state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
				}
			}
			// Build instance accel
			{
				const size_t num_instances = 1;
				const int rayTypeCount = 1;
				std::vector<OptixInstance> optix_instances(num_instances);

				unsigned int sbt_offset = 0;
				for (size_t i = 0; i < num_instances; ++i)
				{
					auto& optix_instance = optix_instances[i];
					memset(&optix_instance, 0, sizeof(OptixInstance));

					optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
					optix_instance.instanceId = static_cast<unsigned int>(i);
					optix_instance.sbtOffset = sbt_offset;
					optix_instance.visibilityMask = 1;
					optix_instance.traversableHandle = state.blas;
					Matrix identity = Matrix::IDENTITY;
					memcpy(optix_instance.transform, (float*)&identity, sizeof(float) * 12);

					sbt_offset += rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
				}

				const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
				CUdeviceptr  d_instances;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(d_instances),
					optix_instances.data(),
					instances_size_in_bytes,
					cudaMemcpyHostToDevice
				));

				OptixBuildInput instance_input = {};
				instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
				instance_input.instanceArray.instances = d_instances;
				instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

				OptixAccelBuildOptions ias_accel_options = {};
				ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
				ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

				OptixAccelBufferSizes ias_buffer_sizes;
				OPTIX_CHECK(optixAccelComputeMemoryUsage(
					state.context,
					&ias_accel_options,
					&instance_input,
					1, // num build inputs
					&ias_buffer_sizes
				));

				CUdeviceptr d_ias_temp_buffer;
				if (ias_buffer_sizes.tempSizeInBytes > state.temp_buffer_size)
				{
					CUDA_CHECK(cudaMalloc((void**)&d_ias_temp_buffer, ias_buffer_sizes.tempSizeInBytes));
				}
				else
				{
					d_ias_temp_buffer = state.d_temp_buffer;
				}

				if (state.d_ias_output_buffer)
				{
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_ias_output_buffer)));
					state.d_ias_output_buffer = 0;
				}
				CUDA_CHECK(cudaMalloc(
					reinterpret_cast<void**>(&state.d_ias_output_buffer),
					ias_buffer_sizes.outputSizeInBytes
				));

				OPTIX_CHECK(optixAccelBuild(
					state.context,
					nullptr,                  // CUDA stream
					&ias_accel_options,
					&instance_input,
					1,                  // num build inputs
					d_ias_temp_buffer,
					ias_buffer_sizes.tempSizeInBytes,
					state.d_ias_output_buffer,
					ias_buffer_sizes.outputSizeInBytes,
					&state.tlas,
					nullptr,            // emitted property list
					0                   // num emitted properties
				));
				state.ias_output_buffer_size = ias_buffer_sizes.outputSizeInBytes;

				if (ias_buffer_sizes.tempUpdateSizeInBytes > state.temp_buffer_size)
				{
					CUDA_CHECK(cudaFree((void*)state.d_temp_buffer));
					state.temp_buffer_size = ias_buffer_sizes.tempUpdateSizeInBytes;
					CUDA_CHECK(cudaMalloc((void**)&state.d_temp_buffer, state.temp_buffer_size));
					CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_temp_buffer)));
				}

				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
			}
		}
		
		void Scene::UpdateAccelerationStructure(OptixState& state)
		{
			OptixAccelBuildOptions gas_accel_options = {};
			gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
			gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

			OptixBuildInput triangle_input = {};
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
			triangle_input.triangleArray.numVertices = mVertexCount;
			const float* pVertexBufData = mPositionBufferT.Data();
			triangle_input.triangleArray.vertexBuffers = (CUdeviceptr*)(&pVertexBufData);

			triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangle_input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
			triangle_input.triangleArray.numIndexTriplets = mTriangleCount;
			triangle_input.triangleArray.indexBuffer = (CUdeviceptr)mIndexPosBuffer.Data();

			vector<uint> flags;
			flags.assign(mBSDFCount, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
			triangle_input.triangleArray.flags = flags.data();

			triangle_input.triangleArray.numSbtRecords = mBSDFCount;
			triangle_input.triangleArray.sbtIndexOffsetBuffer = (CUdeviceptr)mSbtIndexBuffer.Data();
			triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint);
			triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint);

			OPTIX_CHECK(optixAccelBuild(
				state.context,
				0,									// CUDA stream
				&gas_accel_options,
				&triangle_input,
				1,                                  // num build inputs
				state.d_temp_buffer,
				state.temp_buffer_size,
				state.d_gas_output_buffer,
				state.gas_output_buffer_size,
				&state.blas,
				nullptr,                            // emitted property list
				0                                   // num emitted properties
			));


			const size_t num_instances = 1;
			const int rayTypeCount = 1;
			std::vector<OptixInstance> optix_instances(num_instances);

			unsigned int sbt_offset = 0;
			for (size_t i = 0; i < num_instances; ++i)
			{
				auto& optix_instance = optix_instances[i];
				memset(&optix_instance, 0, sizeof(OptixInstance));

				optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
				optix_instance.instanceId = static_cast<unsigned int>(i);
				optix_instance.sbtOffset = sbt_offset;
				optix_instance.visibilityMask = 1;
				optix_instance.traversableHandle = state.blas;
				Matrix identity = Matrix::IDENTITY;
				memcpy(optix_instance.transform, (float*)&identity, sizeof(float) * 12);

				sbt_offset += rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
			}

			const size_t instances_size_in_bytes = sizeof(OptixInstance) * num_instances;
			CUdeviceptr  d_instances;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size_in_bytes));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_instances),
				optix_instances.data(),
				instances_size_in_bytes,
				cudaMemcpyHostToDevice
			));

			OptixBuildInput instance_input = {};
			instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
			instance_input.instanceArray.instances = d_instances;
			instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

			OptixAccelBuildOptions ias_accel_options = {};
			ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
			ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

			OPTIX_CHECK(optixAccelBuild(
				state.context,
				nullptr,                  // CUDA stream
				&ias_accel_options,
				&instance_input,
				1,                  // num build inputs
				state.d_temp_buffer,
				state.temp_buffer_size,
				state.d_ias_output_buffer,
				state.ias_output_buffer_size,
				&state.tlas,
				nullptr,            // emitted property list
				0                   // num emitted properties
			));

			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instances)));
		}

		void Scene::ConfigureLight()
		{
			// Merge all area lights to one.
			for (int i = 0; i < mLights.size(); i++) 
			{
				AreaLight* ptr_light = dynamic_cast<AreaLight*>(mLights[i].get());
				if (!ptr_light) continue;
				const vector<int>& eShapeId = ptr_light->mInfo.mShapeId;
				const vector<Vector3>& intensity = ptr_light->mInfo.mIntensity;
				int offset = 0;
				Tensorf lightBuffer;
				int light_index = 0;
				for (auto shape_id : eShapeId)
				{
					float luminance = 0.212671f * intensity[light_index].x +
						0.715160f * intensity[light_index].y +
						0.072169f * intensity[light_index].z;
					const TriangleMesh* ptrMesh = mPrims[shape_id]->mpMesh.get();
					if (offset == 0) 
					{
						ptr_light->mIndexPosBuffer = ptrMesh->mIndexPosBuffer;
						ptr_light->mFaceNormalBuffer = ptrMesh->mFaceNormalBuffer;
						ptr_light->mPositionBuffer = ptrMesh->mPositionBuffer;
						ptr_light->mTriangleAreaBuffer = ptrMesh->mTriangleAreaBuffer;
						ptr_light->mLightIdBuffer = Tensorui(light_index) * Ones(ptrMesh->mTriangleCount);
						lightBuffer = Tensorf(luminance) * ptrMesh->mTriangleAreaBuffer;
					}
					else 
					{
						ptr_light->mIndexPosBuffer = Concat(ptr_light->mIndexPosBuffer, ptrMesh->mIndexPosBuffer + Scalar(offset), 0);
						ptr_light->mFaceNormalBuffer = Concat(ptr_light->mFaceNormalBuffer, ptrMesh->mFaceNormalBuffer, 0);
						ptr_light->mPositionBuffer = Concat(ptr_light->mPositionBuffer, ptrMesh->mPositionBuffer, 0);
						ptr_light->mTriangleAreaBuffer = Concat(ptr_light->mTriangleAreaBuffer, ptrMesh->mTriangleAreaBuffer, 0);
						ptr_light->mLightIdBuffer = Concat(ptr_light->mLightIdBuffer, Tensorui(light_index) * Ones(ptrMesh->mTriangleCount), 0);
						lightBuffer = Concat(lightBuffer, Tensorf(luminance) * ptrMesh->mTriangleAreaBuffer, 0);
					}
					offset += ptrMesh->mVertexCount;
					light_index++;
				}
				ptr_light->mpLightDistrb = make_unique<Distribution1D>(lightBuffer);
				Tensorf intensityBuffer;
				intensityBuffer.Assign((float*)intensity.data(), { (int)intensity.size(), 3 });
				ptr_light->mIntensity = Tensorf::Transpose(intensityBuffer);
				ptr_light->mIntensity = ptr_light->mIntensity.Reshape(Shape({ (int)intensity.size() }, VecType::Vec3));
				Tensorf areaTot = Sum(ptr_light->mTriangleAreaBuffer);
				ptr_light->mInvTotArea = 1.f / areaTot.Get(0);

				ptr_light->mTriIdToEmitTriIdBuffer = mTriIdToEmitTriIdBuffer;
			}
		}

		void Scene::Configure()
		{
			for (auto i = 0; i < mPrims.size(); i++)
			{
				mPrims[i]->Configure();
			}
			ExtractGeometry();
			ConfigureLight();

			ConstructAccelerationStructure(Optix::GetHandle());
			CreateSBT(Optix::GetHandle());
		}

		void Scene::Update()
		{
			for (auto i = 0; i < mPrims.size(); i++)
			{
				mPrims[i]->Configure();
			}
			ExtractGeometry();
			ConfigureLight();

			UpdateAccelerationStructure(Optix::GetHandle());
			CreateSBT(Optix::GetHandle());
		}

		void Scene::Occluded(Ray& ray, Tensorb& bHit) const
		{
			Intersection isect;
			this->Intersect(ray, isect);
			bHit = isect.mBsdfId == Scalar(-1);

		}

		void Scene::Occluded(Ray& ray, Expr& bHit) const
		{
			Intersection isect;
			this->Intersect(ray, isect);
			bHit = isect.mBsdfId == Scalar(-1);

		}

		void Scene::Intersect(const Ray& ray, Intersection& isect) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			RayIntersectOptix(ray, isect);
			IndexMask maskHit = isect.mTriangleId != Scalar(-1.0f);
			isect.mEmitterId = Ones(isect.mTriangleId->GetShape()) * Scalar(-1.0f);
			if (maskHit.sum > 0)
			{
				auto triIndices = Mask(isect.mTriangleId, maskHit, 0);
				auto emitterID = IndexedRead(mEmitterIDBuffer, triIndices, 0) + Scalar(1.0f);
				isect.mEmitterId = isect.mEmitterId + IndexedWrite(emitterID, maskHit.index, isect.mEmitterId->GetShape(), 0);
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void Scene::IntersectHit(Ray& rays, Intersection& isect) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			if (rays.mNumRays > 0)
			{
				RayIntersectOptix(rays, isect);
				IndexMask mask = isect.mTriangleId != Scalar(-1);
				isect.mEmitterId = Ones(isect.mTriangleId->GetShape()) * Scalar(-1);
				if (mask.sum > 0)
				{
					rays = rays.GetMaskedCopy(mask, true);
					isect = isect.GetMaskedCopy(mask);
					isect.mEmitterId = isect.mEmitterId + IndexedRead(mEmitterIDBuffer, isect.mTriangleId, 0) + Scalar(1);
				}
				else
				{
					rays.mNumRays = 0;
				}
			}
#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		void Scene::PostIntersectPrimary(const Ray& rays, Intersection& isect) const
		{
			auto indicesTri0 = Scalar(3) * isect.mTriangleId;
			auto indicesTri1 = Scalar(3) * isect.mTriangleId + Scalar(1);
			auto indicesTri2 = Scalar(3) * isect.mTriangleId + Scalar(2);
			Expr u, v, w, t;
			{
				auto indicesPos0 = IndexedRead(mIndexPosBuffer, indicesTri0, 0);
				auto indicesPos1 = IndexedRead(mIndexPosBuffer, indicesTri1, 0);
				auto indicesPos2 = IndexedRead(mIndexPosBuffer, indicesTri2, 0);
				auto position0 = IndexedRead(mPositionBuffer, indicesPos0, 0);
				auto position1 = IndexedRead(mPositionBuffer, indicesPos1, 0);
				auto position2 = IndexedRead(mPositionBuffer, indicesPos2, 0);

				RayIntersectAD(rays.mDir, rays.mOrg, position0, position1 - position0, position2 - position0, u, v, t);

				w = Scalar(1.0f) - u - v;
				isect.mPosition = rays.mOrg + t * rays.mDir;
			}

			if (mTexcoordBuffer.LinearSize() > 0)
			{
				auto indicesTex0 = IndexedRead(mIndexTexBuffer, indicesTri0, 0);
				auto indicesTex1 = IndexedRead(mIndexTexBuffer, indicesTri1, 0);
				auto indicesTex2 = IndexedRead(mIndexTexBuffer, indicesTri2, 0);

				auto texcoord0 = IndexedRead(mTexcoordBuffer, indicesTex0, 0);
				auto texcoord1 = IndexedRead(mTexcoordBuffer, indicesTex1, 0);
				auto texcoord2 = IndexedRead(mTexcoordBuffer, indicesTex2, 0);
				isect.mTexcoord = w * texcoord0 + u * texcoord1 + v * texcoord2;
			}

			isect.mGeoNormal = IndexedRead(mFaceNormalBuffer, isect.mTriangleId, 0);
			auto useShadingNormal = IndexedRead(mUseSmoothShadingBuffer, isect.mTriangleId, 0);
			isect.mNormal = Zeros(isect.mGeoNormal->GetShape());
			IndexMask shNormalMask = useShadingNormal == True(useShadingNormal->GetShape());
			if (shNormalMask.sum > 0)
			{
				auto indicesNorm0 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri0, shNormalMask, 0), 0);
				auto indicesNorm1 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri1, shNormalMask, 0), 0);
				auto indicesNorm2 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri2, shNormalMask, 0), 0);
				auto normal0 = IndexedRead(mVertexNormalBuffer, indicesNorm0, 0);
				auto normal1 = IndexedRead(mVertexNormalBuffer, indicesNorm1, 0);
				auto normal2 = IndexedRead(mVertexNormalBuffer, indicesNorm2, 0);
				auto shNormal = Mask(w, shNormalMask, 0) * normal0 +
					Mask(u, shNormalMask, 0) * normal1 +
					Mask(v, shNormalMask, 0) * normal2;
				isect.mNormal = isect.mNormal + IndexedWrite(shNormal, shNormalMask.index, isect.mGeoNormal->GetShape(), 0);
			}
			IndexMask geoNormalMask = useShadingNormal == False(useShadingNormal->GetShape());
			if (geoNormalMask.sum > 0)
			{
				auto geoNormal = Mask(isect.mGeoNormal, geoNormalMask, 0);
				isect.mNormal = isect.mNormal + IndexedWrite(geoNormal, geoNormalMask.index, isect.mGeoNormal->GetShape(), 0);
			}
			isect.mJ = Ones(isect.mTriangleId->GetShape().LinearSize());
			CoordinateSystem(isect.mNormal, &isect.mTangent, &isect.mBitangent);
		}

		void Scene::PostIntersect(Intersection& isect) const
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif
			auto indicesTri0 = Scalar(3) * isect.mTriangleId;
			auto indicesTri1 = Scalar(3) * isect.mTriangleId + Scalar(1);
			auto indicesTri2 = Scalar(3) * isect.mTriangleId + Scalar(2);
			auto baryW = Scalar(1.0f) - isect.mBaryU - isect.mBaryV;
			{
				auto indicesPos0 = IndexedRead(mIndexPosBuffer, indicesTri0, 0);
				auto indicesPos1 = IndexedRead(mIndexPosBuffer, indicesTri1, 0);
				auto indicesPos2 = IndexedRead(mIndexPosBuffer, indicesTri2, 0);
				auto position0 = IndexedRead(mPositionBuffer, indicesPos0, 0);
				auto position1 = IndexedRead(mPositionBuffer, indicesPos1, 0);
				auto position2 = IndexedRead(mPositionBuffer, indicesPos2, 0);
				isect.mPosition = baryW * position0 + isect.mBaryU * position1 + isect.mBaryV * position2;
			}

			if (mTexcoordBuffer.LinearSize() > 0)
			{
				auto indicesTex0 = IndexedRead(mIndexTexBuffer, indicesTri0, 0);
				auto indicesTex1 = IndexedRead(mIndexTexBuffer, indicesTri1, 0);
				auto indicesTex2 = IndexedRead(mIndexTexBuffer, indicesTri2, 0);
				auto texcoord0 = IndexedRead(mTexcoordBuffer, indicesTex0, 0);
				auto texcoord1 = IndexedRead(mTexcoordBuffer, indicesTex1, 0);
				auto texcoord2 = IndexedRead(mTexcoordBuffer, indicesTex2, 0);
				isect.mTexcoord = baryW * texcoord0 + isect.mBaryU * texcoord1 + isect.mBaryV * texcoord2;
			}

			isect.mGeoNormal = IndexedRead(mFaceNormalBuffer, isect.mTriangleId, 0);
			auto useShadingNormal = IndexedRead(mUseSmoothShadingBuffer, isect.mTriangleId, 0);
			isect.mNormal = Zeros(isect.mGeoNormal->GetShape());
			IndexMask shNormalMask = useShadingNormal == True(1);
			if (shNormalMask.sum > 0)
			{
				auto indicesNorm0 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri0, shNormalMask, 0), 0);
				auto indicesNorm1 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri1, shNormalMask, 0), 0);
				auto indicesNorm2 = IndexedRead(mIndexNormalBuffer, Mask(indicesTri2, shNormalMask, 0), 0);
				auto normal0 = IndexedRead(mVertexNormalBuffer, indicesNorm0, 0);
				auto normal1 = IndexedRead(mVertexNormalBuffer, indicesNorm1, 0);
				auto normal2 = IndexedRead(mVertexNormalBuffer, indicesNorm2, 0);
				auto shNormal = Mask(baryW, shNormalMask, 0) * normal0 +
					Mask(isect.mBaryU, shNormalMask, 0) * normal1 +
					Mask(isect.mBaryV, shNormalMask, 0) * normal2;
				isect.mNormal = isect.mNormal + IndexedWrite(shNormal, shNormalMask.index, isect.mGeoNormal->GetShape(), 0);
			}
			IndexMask geoNormalMask = useShadingNormal == False(1);
			if (geoNormalMask.sum > 0)
			{
				auto geoNormal = Mask(isect.mGeoNormal, geoNormalMask, 0);
				isect.mNormal = isect.mNormal + IndexedWrite(geoNormal, geoNormalMask.index, isect.mGeoNormal->GetShape(), 0);
			}

			isect.mJ = IndexedRead(mTriangleAreaBuffer, isect.mTriangleId, 0);
			isect.mJ = isect.mJ / Detach(isect.mJ);

			CoordinateSystem(isect.mNormal, &isect.mTangent, &isect.mBitangent);
			isect.Eval();
#if USE_PROFILING
			nvtxRangePop();
#endif
		}

	}

}