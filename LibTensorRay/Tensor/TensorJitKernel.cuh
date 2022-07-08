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

struct ShapeJit
{
	static const int MaxArraySize = 4;
	int x0;
	int x1;
	int x2;
	int x3;
	int mSize;
	int mVecSize;

	__host__ __device__ inline ShapeJit()
	{
	}

	__host__ __device__ inline void Clear()
	{
		mSize = x0 = x1 = x2 = x3 = 0;
		mVecSize = 1;
	}

	__host__ __device__ ShapeJit(const int _x0, const int _x1)
	{
		Clear();

		x0 = _x0;
		x1 = _x1;
		mSize = 2;
	}

	__host__ __device__ inline void Resize(const int size)
	{
		mSize = size;
	}

	__host__ __device__ inline int Size() const
	{
		return mSize;
	}

	__host__ __device__ bool Add(const int val)
	{
		(*this)[mSize++] = val;

		return false;
	}

	__host__ __device__ inline bool operator == (const ShapeJit& rhs) const
	{
		if (mSize != rhs.mSize)
		{
			return false;
		}

		return x0 == rhs.x0 &&
			x1 == rhs.x1 &&
			x2 == rhs.x2 &&
			x3 == rhs.x3 &&
			mVecSize == rhs.mVecSize;
	}

	__host__ __device__ inline int& operator [] (const int idx)
	{
		switch (idx)
		{
		case 0:
			return x0;
		case 1:
			return x1;
		case 2:
			return x2;
		case 3:
			return x3;
		};

		return x0;
	}

	__host__ __device__ inline int operator [] (const int idx) const
	{
		switch (idx)
		{
		case 0:
			return x0;
		case 1:
			return x1;
		case 2:
			return x2;
		case 3:
			return x3;
		};

		return x0;
	}

	__host__ __device__ inline bool WithinRange(const ShapeJit& shape) const
	{
		if (mSize == 1)
		{
			return shape.x0 >= 0 && shape.x0 < x0;
		}
		else if (mSize == 2)
		{
			return shape.x0 >= 0 && shape.x0 < x0&&
				shape.x1 >= 0 && shape.x1 < x1;
		}
		else if (mSize == 3)
		{
			return shape.x0 >= 0 && shape.x0 < x0&&
				shape.x1 >= 0 && shape.x1 < x1&&
				shape.x2 >= 0 && shape.x2 < x2;
		}
		else if (mSize == 4)
		{
			return shape.x0 >= 0 && shape.x0 < x0&&
				shape.x1 >= 0 && shape.x1 < x1&&
				shape.x2 >= 0 && shape.x2 < x2&&
				shape.x3 >= 0 && shape.x3 < x3;
		}

		return false;
	}

	__host__ __device__ inline bool operator < (const ShapeJit& rhs) const
	{
		return (mSize != rhs.mSize ?
			mSize < rhs.mSize :
			(x0 != rhs.x0 ?
				x0 < rhs.x0 :
				(x1 != rhs.x1 ?
					x1 < rhs.x1 :
					(x2 != rhs.x2 ?
						x2 < rhs.x2 :
						(x3 != rhs.x3 ?
							x3 < rhs.x3 :
							(mVecSize != rhs.mVecSize ?
								mVecSize < rhs.mVecSize :
								false))))));
	}

	__host__ __device__ inline bool operator != (const ShapeJit& rhs) const
	{
		return !(*this == rhs);
	}
};

struct TensorParamsJit
{
	ShapeJit mShape;
	ShapeJit mStrides;
	int mLinearSize;
	int mNumElements;

	__host__ __device__ inline void Clear()
	{
		mShape.Clear();
		mStrides.Clear();
		mLinearSize = 0;
		mNumElements = 0;
	}

	__host__ __device__ inline int LinearIndex(const ShapeJit& idx) const
	{
		int ret = 0;
		ret = idx.x0 * mStrides.x0 +
			idx.x1 * mStrides.x1 +
			idx.x2 * mStrides.x2 +
			idx.x3 * mStrides.x3;

		return ret;
	}

	__host__ __device__ __noinline__ ShapeJit ProjectedIndex(const ShapeJit idx, const int skipAxis = -1) const
	{
		ShapeJit projectedIdx;
		projectedIdx.Clear();
		projectedIdx.Resize(Dim());

		int dimOffset = idx.Size() - Dim();

		for (int i = dimOffset; i < dimOffset + Dim(); i++)
		{
			int localDim = i - dimOffset;
			if (idx[i] >= mShape[localDim] && localDim != skipAxis)
			{
				projectedIdx[localDim] = 0;
			}
			else
			{
				projectedIdx[localDim] = idx[i];
			}
		}

		return projectedIdx;
	}

	__host__ __device__ __noinline__ int ProjectedLinearIndex(const ShapeJit idx) const
	{
		ShapeJit projectedIdx;
		projectedIdx.Clear();
		projectedIdx.Resize(Dim());

		int dimOffset = idx.Size() - Dim();

		for (int i = dimOffset; i < dimOffset + Dim(); i++)
		{
			int localDim = i - dimOffset;
			if (idx[i] >= mShape[localDim])
			{
				projectedIdx[localDim] = 0;
			}
			else
			{
				projectedIdx[localDim] = idx[i];
			}
		}

		return LinearIndex(projectedIdx);
	}

	__host__ __device__ inline int Dim() const
	{
		return mShape.mSize;
	}

	__host__ __device__ __noinline__ ShapeJit Index(int linearIdx) const
	{
		ShapeJit vRet;
		vRet.Clear();
		int dim = mShape.mSize;


		vRet.Resize(dim);
		if (dim == 1)
		{
			vRet.x0 = linearIdx;
		}
		else if (dim == 2)
		{
			vRet.x0 = linearIdx / mStrides.x0;
			linearIdx %= mStrides.x0;

			vRet.x1 = linearIdx;
		}
		else if (dim == 3)
		{
			vRet.x0 = linearIdx / mStrides.x0;
			linearIdx %= mStrides.x0;

			vRet.x1 = linearIdx / mStrides.x1;
			linearIdx %= mStrides.x1;

			vRet.x2 = linearIdx;
		}
		else if (dim == 4)
		{
			vRet.x0 = linearIdx / mStrides.x0;
			linearIdx %= mStrides.x0;

			vRet.x1 = linearIdx / mStrides.x1;
			linearIdx %= mStrides.x1;

			vRet.x2 = linearIdx / mStrides.x2;
			linearIdx %= mStrides.x2;

			vRet.x3 = linearIdx;
		}

		return vRet;
	}


	__host__ __device__ inline int X(const int idx) const
	{
		ShapeJit index = Index(idx);
		return LinearIndex(index);
	}

	__host__ __device__ inline int Y(const int idx) const
	{
		ShapeJit index = Index(idx);
		return LinearIndex(index) + mNumElements;
	}

	__host__ __device__ inline int Z(const int idx) const
	{
		ShapeJit index = Index(idx);
		return LinearIndex(index) + 2 * mNumElements;
	}

	__host__ __device__ inline int W(const int idx) const
	{
		ShapeJit index = Index(idx);
		return LinearIndex(index) + 3 * mNumElements;
	}

	__host__ __device__ inline int LinearSize() const
	{
		return mLinearSize;
	}

	__host__ __device__ inline int NumElements() const
	{
		return mNumElements;
	}

	__host__ __device__ void IterateIndex(ShapeJit& index) const
	{
		for (int i = mShape.Size() - 1; i >= 0; i--)
		{
			index[i]++;

			if (index[i] < mShape[i])
				break;
			else
				index[i] = 0;
		}
	}

	__host__ __device__ __noinline__ bool IterateIndex(ShapeJit& index, const ShapeJit axes /*axes to iterate through*/) const
	{
		for (int i = axes.Size() - 1; i >= 0; i--)
		{
			int axis = axes[i];

			index[axis]++;

			if (index[axis] < mShape[axis])
			{
				break;
			}
			else
			{
				if (i == 0)
					return false;

				index[axis] = 0;
			}
		}

		return true;
	}

	__host__ __device__ inline bool operator < (const TensorParamsJit& rhs) const
	{
		return (mLinearSize != rhs.mLinearSize ?
			mLinearSize < rhs.mLinearSize :
			(mShape != rhs.mShape ?
				mShape < rhs.mShape :
				(mStrides != rhs.mStrides ?
					(mStrides < rhs.mStrides) :
						(mNumElements < rhs.mNumElements))));
	}

	__host__ __device__ inline bool operator != (const TensorParamsJit& rhs) const
	{
		return mLinearSize != rhs.mLinearSize || mNumElements != rhs.mNumElements || mShape != rhs.mShape || mStrides != rhs.mStrides;
	}
};

template<class T>
struct TensorJit
{
	T* mpData;

	TensorParamsJit mParams;

	__host__ __device__ inline T Eval(const int idx, const TensorParamsJit& broadcastIndex) const
	{
		ShapeJit index = broadcastIndex.Index(idx);
		int linearIndex = ProjectedLinearIndex(index);

		return mpData[linearIndex];
	}

	__host__ __device__ inline T X(const int idx, const TensorParamsJit& broadcastIndex) const
	{
		ShapeJit index = broadcastIndex.Index(idx);
		int linearIndex = ProjectedLinearIndex(index);

		return mpData[linearIndex];
	}

	__host__ __device__ inline T Y(const int idx, const TensorParamsJit& broadcastIndex) const
	{
		ShapeJit index = broadcastIndex.Index(idx);
		int linearIndex = ProjectedLinearIndex(index) + mParams.NumElements();

		return mpData[linearIndex];
	}

	__host__ __device__ inline T Z(const int idx, const TensorParamsJit& broadcastIndex) const
	{
		ShapeJit index = broadcastIndex.Index(idx);
		int linearIndex = ProjectedLinearIndex(index) + 2 * mParams.NumElements();

		return mpData[linearIndex];
	}

	__host__ __device__ inline T W(const int idx, const TensorParamsJit& broadcastIndex) const
	{
		ShapeJit index = broadcastIndex.Index(idx);
		int linearIndex = ProjectedLinearIndex(index) + 3 * mParams.NumElements();

		return mpData[linearIndex];
	}

	__host__ __device__ inline int Dim() const
	{
		return mParams.mShape.mSize;
	}
	__host__ __device__ int LinearIndex(const ShapeJit& idx) const
	{
		return mParams.LinearIndex(idx);
	}
	__host__ __device__ int ProjectedLinearIndex(const ShapeJit& idx) const
	{
		return mParams.ProjectedLinearIndex(idx);
	}
	__host__ __device__ ShapeJit Index(int linearIdx) const
	{
		return mParams.Index(linearIdx);
	}
	__host__ __device__ int LinearSize() const
	{
		return mParams.LinearSize();
	}

	__host__ __device__ T& operator [] (const ShapeJit& idx)
	{
		return mpData[ProjectedLinearIndex(idx)];
	}
	__host__ __device__ const T& operator [] (const ShapeJit& idx) const
	{
		return mpData[ProjectedLinearIndex(idx)];
	}
	__host__ __device__ void IterateIndex(ShapeJit& index) const
	{
		return mParams.IterateIndex(index);
	}
	__host__ __device__ bool IterateIndex(ShapeJit& index, const ShapeJit& axes) const
	{
		return mParams.IterateIndex(index, axes);
	}
	__host__ __device__ const T* Data() const
	{
		return mpData;
	}
	__host__ __device__ T* Data()
	{
		return mpData;
	}
	__host__ __device__ T& operator [] (const int idx)
	{
		return mpData[idx];
	}
	__host__ __device__ const T& operator [] (const int idx) const
	{
		return mpData[idx];
	}
};

struct TensorJitArg
{
	int mType;
	void* mpData;

	TensorParamsJit mParams;

	template<class T>
	__host__ __device__ inline TensorJit<T> ToTensorJit() const
	{
		TensorJit<T> ret;
		ret.mpData = (T*)mpData;
		ret.mParams = mParams;

		return ret;
	}

	__host__ __device__ inline bool operator < (const TensorJitArg& rhs) const
	{
		return mpData < rhs.mpData;
	}

	__host__ __device__ inline bool operator != (const TensorJitArg& rhs) const
	{
		return mpData != rhs.mpData;
	}
};

struct IndexedReadArg
{
	void* pIndices;
	TensorParamsJit opParams;
	TensorParamsJit indicesParams;
	int axis;

	__host__ __device__ inline bool operator < (const IndexedReadArg& rhs) const
	{
		return (pIndices != rhs.pIndices ?
			pIndices < rhs.pIndices :
			(axis != rhs.axis ?
				axis < rhs.axis :
				opParams != rhs.opParams ?
				opParams < rhs.opParams : indicesParams != rhs.indicesParams));
	}

	__host__ __device__ inline bool operator != (const IndexedReadArg& rhs) const
	{
		return pIndices != rhs.pIndices || opParams != rhs.opParams || indicesParams != rhs.indicesParams || axis != rhs.axis;
	}

	__host__ __device__ inline int GetAxisIndex(const int linearIdx, const TensorParamsJit& inParams) const
	{
		ShapeJit index = inParams.Index(linearIdx);
		return opParams.ProjectedIndex(index, axis)[axis];
	}

	__host__ __device__ inline int ConvertLinearIndex(const int linearIdx, const int dimIdx, const TensorParamsJit& inParams) const
	{
		ShapeJit index = opParams.ProjectedIndex(inParams.Index(linearIdx), axis);
		index[axis] = dimIdx;
		return opParams.ProjectedLinearIndex(index);
	}
};

struct ConcatIndex
{
	TensorParamsJit params;
	TensorParamsJit params1;
	TensorParamsJit params2;
	int dim;

	__host__ __device__ inline bool operator < (const ConcatIndex& rhs) const
	{
		return (dim != rhs.dim ?
			dim < rhs.dim :
			(params != rhs.params ?
				params < rhs.params :
				(params1 != rhs.params1 ?
					params1 < rhs.params1 :
					(params2 < rhs.params2))));
	}

	__host__ __device__ inline bool operator != (const ConcatIndex& rhs) const
	{
		return params != rhs.params || params1 != rhs.params1 || params2 != rhs.params2 || dim != rhs.dim;
	}
};

struct SliceIndex
{
	TensorParamsJit oldParam;
	TensorParamsJit newParam;
	ShapeJit begin;
	ShapeJit end;
	bool bBackward;

	__host__ __device__ inline int CalcLinearIndex(int i, const TensorParamsJit& broadcastIndex)
	{
		if (bBackward)
		{
			ShapeJit idx = newParam.Index(i);
			idx = DeriveIndex(idx);
			if (oldParam.mShape.WithinRange(idx))
			{
				return oldParam.LinearIndex(idx);
			}
			else
				return -1;
		}
		else
		{
			ShapeJit idx = broadcastIndex.Index(i);
			if (idx.x0 >= newParam.mShape.x0)
				idx.x0 = 0;
			if (idx.x1 >= newParam.mShape.x1)
				idx.x1 = 0;
			if (idx.x2 >= newParam.mShape.x2)
				idx.x2 = 0;
			if (idx.x3 >= newParam.mShape.x3)
				idx.x3 = 0;
			idx = DeriveIndex(idx);
			return oldParam.LinearIndex(idx);
		}
	}

	__host__ __device__ inline ShapeJit DeriveIndex(const ShapeJit& idx) const
	{
		ShapeJit ret;
		ret.Clear();
		ret.Resize(idx.mSize);
		ret.x0 = idx.x0 + begin.x0;
		ret.x1 = idx.x1 + begin.x1;
		ret.x2 = idx.x2 + begin.x2;
		ret.x3 = idx.x3 + begin.x3;

		return ret;
	}


	__host__ __device__ inline bool operator < (const SliceIndex& rhs) const
	{
		return (bBackward != rhs.bBackward ?
			int(bBackward) < int(rhs.bBackward) :
			(oldParam != rhs.oldParam ?
				oldParam < rhs.oldParam :
				(newParam != rhs.newParam ?
					newParam < rhs.newParam :
					(begin != rhs.begin ?
						begin < rhs.begin :
						(end < rhs.end)))));
	}

	__host__ __device__ inline bool operator != (const SliceIndex& rhs) const
	{
		return bBackward != rhs.bBackward || oldParam != rhs.oldParam || newParam != rhs.newParam || begin != rhs.begin || end != rhs.end;
	}
};

#ifdef __CUDACC_RTC__

__device__ float4 PixelCoord(const int i, const int width, const int height)
{
	int linearIdx = i % (width * height);
	int h = linearIdx / width;
	int w = linearIdx % width;

	return make_float4(w, h, 0.0f, 1.0f);
}

#include "CudaMath.h"

#endif

#ifndef __CUDACC_RTC__

static const char* JitKernelTemplate0 = R"===(

#include "LibTensorRay/Tensor/TensorJitKernel.cuh"

)===";

static const char* JitKernelTemplate1 = R"===(

__global__ void ExpressionJitKernel(
	TensorParamsJit broadcastParams)===";


static const char* JitKernelTemplate2 = R"===()
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= broadcastParams.NumElements())
		return;

)===";

static const char* JitKernelTemplate3 = R"===(
	pDest[i] = )===";


static const char* JitKernelTemplate4 = R"===(
}

)===";


static const char* JitInplaceOpKernelTemplate = R"===(

#include "LibTensorRay/Tensor/TensorJitKernel.cuh"

__global__ void ExpressionInplaceJitKernel(
	TensorParamsJit broadcastParams)===";


static const char* JitInplaceOpKernelTemplate2 = R"===()
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= broadcastParams.NumElements())
		return;

)===";

static const char* JitInplaceOpKernelTemplate3 = R"===(
	pDest[i] = (pDest[i])===";



static const char* JitInplaceOpKernelTemplate4 = R"===(
}

)===";

#endif