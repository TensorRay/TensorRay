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

#include "Tensor.h"
#include <cub/cub.cuh>


namespace EDX
{
	namespace DeepLearning
	{
		
		#include "TensorKernels.cuh"

		template<class T>
		Tensor<T> Tensor<T>::ExclusiveScan(const Tensor<T>& val)
		{
			Tensor<T> ret;
			ret.Resize(val.GetShape());

			// Declare, allocate, and initialize device-accessible pointers for input and output
			int numItems = val.LinearSize();
			const T* pIn = val.Data();
			T* pOut = ret.Data();

			// Determine temporary device storage requirements
			void* pTempStorage = NULL;
			size_t tempStorageBytes = 0;
			cub::DeviceScan::ExclusiveSum(pTempStorage, tempStorageBytes, pIn, pOut, numItems);
			// Allocate temporary storage
			CuAllocator::GetHandle().DeviceAllocate(&pTempStorage, tempStorageBytes);
			// Run exclusive prefix sum
			cub::DeviceScan::ExclusiveSum(pTempStorage, tempStorageBytes, pIn, pOut, numItems);

			// Release device memory
			CuAllocator::GetHandle().DeviceFree(pTempStorage);

			ret.CopyToHost();

			return ret;
		}

		template Tensor<bool> Tensor<bool>::ExclusiveScan(const Tensor<bool>& val);
		template Tensor<int> Tensor<int>::ExclusiveScan(const Tensor<int>& val);
		template Tensor<uint> Tensor<uint>::ExclusiveScan(const Tensor<uint>& val);
		template Tensor<float> Tensor<float>::ExclusiveScan(const Tensor<float>& val);
		template Tensor<double> Tensor<double>::ExclusiveScan(const Tensor<double>& val);

		template<class T>
		Tensor<T> Tensor<T>::InclusiveScan(const Tensor<T>& val)
		{
			Tensor<T> ret;
			ret.Resize(val.GetShape());

			// Declare, allocate, and initialize device-accessible pointers for input and output
			int numItems = val.LinearSize();
			const T* pIn = val.Data();
			T* pOut = ret.Data();

			// Determine temporary device storage requirements
			void* pTempStorage = NULL;
			size_t tempStorageBytes = 0;
			cub::DeviceScan::InclusiveSum(pTempStorage, tempStorageBytes, pIn, pOut, numItems);
			// Allocate temporary storage
			CuAllocator::GetHandle().DeviceAllocate(&pTempStorage, tempStorageBytes);
			// Run inclusive prefix sum
			cub::DeviceScan::InclusiveSum(pTempStorage, tempStorageBytes, pIn, pOut, numItems);

			// Release device memory
			CuAllocator::GetHandle().DeviceFree(pTempStorage);
			ret.CopyToHost();

			return ret;
		}

		template Tensor<bool> Tensor<bool>::InclusiveScan(const Tensor<bool>& val);
		template Tensor<int> Tensor<int>::InclusiveScan(const Tensor<int>& val);
		template Tensor<uint> Tensor<uint>::InclusiveScan(const Tensor<uint>& val);
		template Tensor<float> Tensor<float>::InclusiveScan(const Tensor<float>& val);
		template Tensor<double> Tensor<double>::InclusiveScan(const Tensor<double>& val);

		template<class T>
		void Tensor<T>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			Assert(mask.Dim() == 1);

			const Shape& maskShape = mask.GetShape();

			// Resize return tensor
			pResult->Resize(maskSum);

			InvokeMaskedSelectionIndex(*pResult, maskSize, offset, mask);
			pResult->CopyToHost();

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		template void Tensor<bool>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);
		template void Tensor<int>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);
		template void Tensor<uint>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);
		template void Tensor<float>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);
		template void Tensor<double>::MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);

		template<class T>
		void Tensor<T>::IndexedWrite(Tensor<T>& dst, const Tensor<T>& lhs, const Tensor<int>& indices, const int axis)
		{
#if USE_PROFILING
			nvtxRangePushA(__FUNCTION__);
#endif

			Tensor<T> val;
			int ax = axis;
			if (lhs.VectorSize() > 1)
			{
				Shape newShape = lhs.GetShape().LinearizeVector();
				val = lhs.Reshape(newShape);
				dst = dst.Reshape(dst.GetShape().LinearizeVector());
				ax++;
			}
			else
				val = lhs;
			dst.Clear();

			InvokeIndexedWrite(dst, val, indices, ax);

			if (lhs.VectorSize() > 1)
			{
				dst = dst.Reshape(dst.GetShape().Vectorize());
			}
			dst.CopyToHost();

#if USE_PROFILING
			nvtxRangePop();
#endif
		}

		//template void Tensor<bool>::IndexedWrite(Tensor<bool>& dst, const Tensor<bool>& val, const Tensor<int>& indices, const int axis);
		template void Tensor<int>::IndexedWrite(Tensor<int>& dst, const Tensor<int>& val, const Tensor<int>& indices, const int axis);
		template void Tensor<uint>::IndexedWrite(Tensor<uint>& dst, const Tensor<uint>& val, const Tensor<int>& indices, const int axis);
		template void Tensor<float>::IndexedWrite(Tensor<float>& dst, const Tensor<float>& val, const Tensor<int>& indices, const int axis);
		template void Tensor<double>::IndexedWrite(Tensor<double>& dst, const Tensor<double>& val, const Tensor<int>& indices, const int axis);


		template<class T>
		template<typename Op>
		Tensor<T> Tensor<T>::ProjectionOp(const Tensor<T>& lhs, const Shape& axes, const bool keepDim, Op op, T initVal)
		{
			if (axes.Size() == 0)
				return lhs;

			if (axes.Size() == 1 && axes[0] == -2)
			{
				// Declare, allocate, and initialize device-accessible pointers for input and output
				int numItems = lhs.LinearSize();
				const T* pIn = lhs.Data();
				Tensor<T> ret;
				ret.Resize(1);
				T* pOut = ret.Data();

				// Determine temporary device storage requirements
				void* pTempStorage = NULL;
				size_t tempStorageBytes = 0;
				cub::DeviceReduce::Reduce(pTempStorage, tempStorageBytes, pIn, pOut, numItems, op, initVal);
				// Allocate temporary storage
				CuAllocator::GetHandle().DeviceAllocate(&pTempStorage, tempStorageBytes);
				// Run exclusive prefix sum
				cub::DeviceReduce::Reduce(pTempStorage, tempStorageBytes, pIn, pOut, numItems, op, initVal);

				// Release device memory
				CuAllocator::GetHandle().DeviceFree(pTempStorage);

				ret.CopyToHost();

				return ret;
			}

			Tensor<T> val;
			Shape ax;
			if (lhs.VectorSize() > 1)
			{
				Shape newShape = lhs.GetShape().LinearizeVector();
				val = lhs.Reshape(newShape);

				ax = axes;
				for (int i = 0; i < axes.Size(); i++)
				{
					ax[i]++;
				}
			}
			else
			{
				val = lhs;
				ax = axes;
			}

			const Shape& inShape = val.GetShape();
			Shape projShape;
			Shape projShapeKeepDim;
			for (int i = 0; i < inShape.Size(); i++)
			{
				if (!keepDim)
				{
					if (!ax.Contains(i))
						projShape.Add(inShape[i]);
				}
				else
				{
					if (ax.Contains(i))
						projShape.Add(1);
					else
						projShape.Add(inShape[i]);
				}

				if (ax.Contains(i))
					projShapeKeepDim.Add(1);
				else
					projShapeKeepDim.Add(inShape[i]);
			}

			if (projShape.Empty())
				projShape.Add(1);

			if (projShapeKeepDim.Empty())
				projShapeKeepDim.Add(1);

			TensorParams tensorParamsKeepDim;
			tensorParamsKeepDim.Resize(projShapeKeepDim);

			Tensor<T> ret;
			ret.Resize(projShape);
			InvokeTensorProjectionOp(ret, lhs, tensorParamsKeepDim, axes, op, initVal);

			if (lhs.VectorSize() > 1)
				ret = ret.Reshape(projShape.Vectorize());

			ret.CopyToHost();

			return ret;
		}

		template Tensor<float> Tensor<float>::ProjectionOp(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, float initVal);
		template Tensor<float> Tensor<float>::ProjectionOp(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, float initVal);
		template Tensor<float> Tensor<float>::ProjectionOp(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, float initVal);
		template Tensor<float> Tensor<float>::ProjectionOp(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, float initVal);
		template Tensor<int> Tensor<int>::ProjectionOp(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, int initVal);
		template Tensor<int> Tensor<int>::ProjectionOp(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, int initVal);
		template Tensor<int> Tensor<int>::ProjectionOp(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, int initVal);
		template Tensor<int> Tensor<int>::ProjectionOp(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, int initVal);


		template<class T>
		template<typename Op>
		void Tensor<T>::ProjectionOpInplace(const Tensor<T>& lhs, const Shape& axes, const bool keepDim, Op op, T initVal, Tensor<T>* pResult)
		{
			if (axes.Size() == 1 && axes[0] == -2)
			{
				// Declare, allocate, and initialize device-accessible pointers for input and output
				int numItems = lhs.LinearSize();
				const T* pIn = lhs.Data();
				pResult->Resize(1);
				T* pOut = pResult->Data();

				// Determine temporary device storage requirements
				void* pTempStorage = NULL;
				size_t tempStorageBytes = 0;
				cub::DeviceReduce::Reduce(pTempStorage, tempStorageBytes, pIn, pOut, numItems, op, initVal);
				// Allocate temporary storage
				CuAllocator::GetHandle().DeviceAllocate(&pTempStorage, tempStorageBytes);
				// Run exclusive prefix sum
				cub::DeviceReduce::Reduce(pTempStorage, tempStorageBytes, pIn, pOut, numItems, op, initVal);

				// Release device memory
				CuAllocator::GetHandle().DeviceFree(pTempStorage);

				pResult->CopyToHost();
				return;
			}

			Tensor<T> val;
			Shape ax;
			if (lhs.VectorSize() > 1)
			{
				Shape newShape = lhs.GetShape().LinearizeVector();
				val = lhs.Reshape(newShape);

				ax = axes;
				for (int i = 0; i < axes.Size(); i++)
				{
					ax[i]++;
				}
			}
			else
			{
				val = lhs;
				ax = axes;
			}

			const Shape& inShape = val.GetShape();
			Shape projShape;
			Shape projShapeKeepDim;
			for (int i = 0; i < inShape.Size(); i++)
			{
				if (!keepDim)
				{
					if (!ax.Contains(i))
						projShape.Add(inShape[i]);
				}
				else
				{
					if (ax.Contains(i))
						projShape.Add(1);
					else
						projShape.Add(inShape[i]);
				}

				if (ax.Contains(i))
					projShapeKeepDim.Add(1);
				else
					projShapeKeepDim.Add(inShape[i]);
			}

			if (projShape.Empty())
				projShape.Add(1);

			if (projShapeKeepDim.Empty())
				projShapeKeepDim.Add(1);

			TensorParams tensorParamsKeepDim;
			tensorParamsKeepDim.Resize(projShapeKeepDim);

			pResult->Resize(projShape);
			InvokeTensorProjectionOp(*pResult, val, tensorParamsKeepDim, ax, op, initVal);

			if (lhs.VectorSize() > 1)
				*pResult = pResult->Reshape(projShape.Vectorize());

			pResult->CopyToHost();
		}

		template void Tensor<bool>::ProjectionOpInplace(const Tensor<bool>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, bool initVal, Tensor<bool>* pResult);
		template void Tensor<bool>::ProjectionOpInplace(const Tensor<bool>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, bool initVal, Tensor<bool>* pResult);
		template void Tensor<bool>::ProjectionOpInplace(const Tensor<bool>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, bool initVal, Tensor<bool>* pResult);
		template void Tensor<bool>::ProjectionOpInplace(const Tensor<bool>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, bool initVal, Tensor<bool>* pResult);

		template void Tensor<int>::ProjectionOpInplace(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, int initVal, Tensor<int>* pResult);
		template void Tensor<int>::ProjectionOpInplace(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, int initVal, Tensor<int>* pResult);
		template void Tensor<int>::ProjectionOpInplace(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, int initVal, Tensor<int>* pResult);
		template void Tensor<int>::ProjectionOpInplace(const Tensor<int>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, int initVal, Tensor<int>* pResult);

		template void Tensor<uint>::ProjectionOpInplace(const Tensor<uint>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, uint initVal, Tensor<uint>* pResult);
		template void Tensor<uint>::ProjectionOpInplace(const Tensor<uint>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, uint initVal, Tensor<uint>* pResult);
		template void Tensor<uint>::ProjectionOpInplace(const Tensor<uint>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, uint initVal, Tensor<uint>* pResult);
		template void Tensor<uint>::ProjectionOpInplace(const Tensor<uint>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, uint initVal, Tensor<uint>* pResult);

		template void Tensor<float>::ProjectionOpInplace(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, float initVal, Tensor<float>* pResult);
		template void Tensor<float>::ProjectionOpInplace(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, float initVal, Tensor<float>* pResult);
		template void Tensor<float>::ProjectionOpInplace(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, float initVal, Tensor<float>* pResult);
		template void Tensor<float>::ProjectionOpInplace(const Tensor<float>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, float initVal, Tensor<float>* pResult);

		template void Tensor<double>::ProjectionOpInplace(const Tensor<double>& lhs, const Shape& axes, const bool keepDim, Algorithm::Plus<> op, double initVal, Tensor<double>* pResult);
		template void Tensor<double>::ProjectionOpInplace(const Tensor<double>& lhs, const Shape& axes, const bool keepDim, Algorithm::Multiply<> op, double initVal, Tensor<double>* pResult);
		template void Tensor<double>::ProjectionOpInplace(const Tensor<double>& lhs, const Shape& axes, const bool keepDim, Algorithm::Min<> op, double initVal, Tensor<double>* pResult);
		template void Tensor<double>::ProjectionOpInplace(const Tensor<double>& lhs, const Shape& axes, const bool keepDim, Algorithm::Max<> op, double initVal, Tensor<double>* pResult);

	}
}