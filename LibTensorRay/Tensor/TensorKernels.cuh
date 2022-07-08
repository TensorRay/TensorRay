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


template<typename TensorT, typename TensorI, typename TensorB>
__global__ void MaskedSelectionKernel(TensorT dst, TensorT val, TensorI offset, TensorB mask, const int axis)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	int numItems = val.LinearSize();
	if (i >= numItems)
		return;

	ShapeJit index = val.Index(i);
	ShapeJit maskIndex;
	maskIndex.Clear();
	for (int j = 0; j < mask.Dim(); j++)
	{
		maskIndex.Add(index[axis + j]);
	}

	if (mask[maskIndex])
	{
		ShapeJit dstIndex;
		dstIndex.Clear();
		for (int j = 0; j < index.mSize; j++)
		{
			if (j == axis)
			{
				dstIndex.Add(offset[maskIndex]);
				j += mask.Dim() - 1;
			}
			else
			{
				dstIndex.Add(index[j]);
			}
		}

		dst[dstIndex] = val[index];
	}
}

template<typename TensorT, typename TensorI, typename TensorB>
void InvokeMaskedSelection(TensorT& dst, const TensorT& val, const TensorI& offset, const TensorB& mask, const int axis)
{
	const int linearSize = val.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	MaskedSelectionKernel<<<gridDim, blockDim >>>(dst.ToJit(), val.ToJit(), offset.ToJit(), mask.ToJit(), axis);
}

template<typename TensorI, typename TensorB>
__global__ void MaskedSelectionIndexKernel(TensorI dst, const int maskSize, TensorI offset, TensorB mask)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= maskSize)
		return;

	if (mask[i])
	{
		dst[offset[i]] = i;
	}
}

template<typename TensorI, typename TensorB>
void InvokeMaskedSelectionIndex(TensorI& dst, const int maskSize, const TensorI& offset, const TensorB& mask)
{
	const int linearSize = maskSize;
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	MaskedSelectionIndexKernel<<<gridDim, blockDim>>>(dst.ToJit(), maskSize, offset.ToJit(), mask.ToJit());
}

template<typename TensorT, typename TensorI>
__global__ void IndexedReadKernel(TensorT dst, TensorT val, TensorI indices, TensorParamsJit dstParam, const int axis)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= indices.LinearSize() * dstParam.LinearSize())
		return;

	int j = i % dstParam.LinearSize();
	i = i / dstParam.LinearSize();

	ShapeJit dstIdx = dstParam.Index(j);
	ShapeJit valIdx = dstIdx;
	valIdx.Resize(val.Dim());

	dstIdx[axis] = indices[i];
	valIdx[axis] = i;
	val[valIdx] = dst[dstIdx];
}

template<typename TensorT, typename TensorI>
void InvokeIndexedRead(const TensorT& dst, TensorT& val, const TensorI& indices, const int axis)
{
	Shape dstShape = dst.GetShape();
	dstShape[axis] = 1;
	TensorParams dstParam;
	dstParam.Resize(dstShape);

	const int linearSize = indices.LinearSize() * dstParam.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	IndexedReadKernel<<<gridDim, blockDim>>>(dst.ToJit(), val.ToJit(), indices.ToJit(), dstParam.ToJit(), axis);
}


template<typename TensorT, typename TensorI>
__global__ void IndexedReadKernel(TensorT dst, TensorT val, TensorI indices0, TensorI indices1, TensorParamsJit dstParam, const int axis0, const int axis1)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= indices0.LinearSize() * dstParam.LinearSize())
		return;

	int j = i % dstParam.LinearSize();
	i = i / dstParam.LinearSize();

	ShapeJit dstIdx = dstParam.Index(j);
	ShapeJit valIdx = dstIdx;
	valIdx.Resize(val.Dim());

	dstIdx[axis0] = indices0[i];
	dstIdx[axis1] = indices1[i];
	valIdx[axis0] = i;
	val[valIdx] = dst[dstIdx];
}

template<typename TensorT, typename TensorI>
void InvokeIndexedRead(const TensorT& dst, TensorT& val, const TensorI& indices0, const TensorI& indices1, const int axis0, const int axis1)
{
	Shape dstShape = dst.GetShape();
	dstShape[axis0] = 1;
	dstShape[axis1] = 1;
	TensorParams dstParam;
	dstParam.Resize(dstShape);

	const int linearSize = indices0.LinearSize() * dstParam.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	IndexedReadKernel<<<gridDim, blockDim>>>(dst.ToJit(), val.ToJit(), indices0.ToJit(), indices1.ToJit(), dstParam.ToJit(), axis0, axis1);
}

template<typename TensorT, typename TensorI>
__global__ void IndexedWriteKernel(TensorT dst, TensorT val, TensorI indices, TensorParamsJit dstParam, const int axis)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= indices.LinearSize() * dstParam.LinearSize())
		return;

	int j = i % dstParam.LinearSize();
	i = i / dstParam.LinearSize();

	ShapeJit dstIdx = dstParam.Index(j);
	ShapeJit valIdx = dstIdx;

	dstIdx[axis] = indices[i];
	valIdx[axis] = i;
	atomicAdd(&dst[dstIdx], val[valIdx]);
}

template<typename TensorT, typename TensorI>
void InvokeIndexedWrite(TensorT& dst, const TensorT& val, const TensorI& indices, const int axis)
{
	Shape dstShape = dst.GetShape();
	dstShape[axis] = 1;
	TensorParams dstParam;
	dstParam.Resize(dstShape);

	const int linearSize = indices.LinearSize() * dstParam.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	IndexedWriteKernel<<<gridDim, blockDim>>>(dst.ToJit(), val.ToJit(), indices.ToJit(), dstParam.ToJit(), axis);
}

template<typename TensorT, typename Op, typename T>
__global__ void TensorProjectionOpKernel(TensorT ret, const TensorT lhs, const TensorParamsJit Params, const ShapeJit axises, Op op, T initVal)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= ret.LinearSize())
		return;

	T reduced = initVal;

	ShapeJit projIndex = Params.Index(i);

	do
	{
		reduced = op(reduced, lhs[projIndex]);
	} while (lhs.IterateIndex(projIndex, axises));

	ret[i] = reduced;
}

template<typename TensorT, typename Op, typename T>
void InvokeTensorProjectionOp(TensorT& ret, const TensorT& lhs, const TensorParams& params, const Shape& axises, Op op, T initVal)
{
	const int linearSize = ret.LinearSize();
	const int blockDim = 256;
	const int gridDim = (linearSize + blockDim - 1) / blockDim;

	TensorProjectionOpKernel<<<gridDim, blockDim>>>(ret.ToJit(), lhs.ToJit(), params.ToJit(), axises.ToJit(), op, initVal);
}