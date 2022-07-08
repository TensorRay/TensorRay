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
#include "../Core/Memory.h"
#include "../Math/Vector.h"

#include <vector>

namespace EDX
{
	template<size_t Dimension>
	class ArrayIndex
	{
	protected:
		Vec<Dimension, size_t> mDim;
		Vec<Dimension, size_t> mStrides;
		size_t mArraySize;

	public:
		void Init(const Vec<Dimension, uint>& size)
		{
			mDim = size;
			CalcStrides();

			mArraySize = 1;
			for (auto i = 0; i < Dimension; i++)
				mArraySize *= mDim[i];
		}

		__forceinline size_t LinearIndex(const Vec<Dimension, uint>& idx) const
		{
			size_t ret = idx[0];
			for (auto i = 1; i < Dimension; i++)
			{
				ret += idx[i] * mStrides[i];
			}
			Assert(ret < mArraySize);
			return ret;
		}
		__forceinline Vec<Dimension, uint> Index(size_t linearIdx) const
		{
			Assert(linearIdx < mArraySize);
			Vec<Dimension, uint> vRet;
			for (int i = Dimension - 1; i >= 0; i--)
			{
				vRet[i] = linearIdx / mStrides[i];
				linearIdx %= mStrides[i];
			}
			return vRet;
		}

		void CalcStrides()
		{
			for (auto i = 0; i < Dimension; i++)
			{
				mStrides[i] = 1;
				for (auto dim = 0; dim < i; dim++)
					mStrides[i] *= mDim[dim];
			}
		}
		void SetDim(const Vec<Dimension, uint>& size)
		{
			auto oldSize = mArraySize;

			mDim = size;
			CalcStrides();
			mArraySize = 1;
			for (auto i = 0; i < Dimension; i++)
			{
				mArraySize *= mDim[i];
			}

			Assert(mArraySize == oldSize);
		}

		__forceinline size_t LinearSize() const
		{
			return mArraySize;
		}
		__forceinline size_t Size(uint iDim) const
		{
			Assert(iDim < Dimension);
			return mDim[iDim];
		}
		__forceinline Vec<Dimension, uint> Size() const
		{
			return mDim;
		}
		__forceinline size_t Stride(uint iDim) const
		{
			Assert(iDim < Dimension);
			return mStrides[iDim];
		}
	};

	template<size_t Dimension, class T>
	class DimensionalArray
	{
	protected:
		ArrayIndex<Dimension> mIndex;
		T* mpData;

	public:
		DimensionalArray()
			: mpData(nullptr)
		{
		}

		DimensionalArray(const Vec<Dimension, uint>& size, bool bClear = true)
			: mpData(nullptr)
		{
			this->Init(size, bClear);
		}

		virtual ~DimensionalArray()
		{
			Free();
		}

		DimensionalArray(const DimensionalArray& rhs)
			: mpData(NULL)
		{
			this->operator=(rhs);
		}

		DimensionalArray(DimensionalArray&& rhs)
			: mpData(NULL)
		{
			this->operator=(std::move(rhs));
		}

		void Init(const Vec<Dimension, uint>& size, bool bClear = true)
		{
			Memory::SafeFree(mpData);
			mIndex.Init(size);

			mpData = Memory::AlignedAlloc<T>(mIndex.LinearSize());
			Assert(mpData);

			if (bClear)
				Clear();
		}

		void SetData(const T* pData)
		{
			Memory::Memcpy(mpData, pData, LinearSize() * sizeof(T));
		}

		void SetDim(const Vec<Dimension, uint>& size)
		{
			mIndex.SetDim(size);
		}

		__forceinline void Clear()
		{
			Memory::SafeClear(mpData, mIndex.LinearSize());
		}

		DimensionalArray& operator = (const DimensionalArray& rhs)
		{
			if (Size() != rhs.Size())
			{
				Free();
				Init(rhs.Size());
			}
			Memory::Memcpy(mpData, rhs.mpData, LinearSize() * sizeof(T));
			return *this;
		}
		DimensionalArray& operator = (DimensionalArray&& rhs)
		{
			if (Size() != rhs.Size())
			{
				Free();
				Init(rhs.Size());
			}
			mpData = rhs.mpData;
			rhs.mpData = NULL;
			return *this;
		}

		__forceinline virtual size_t LinearIndex(const Vec<Dimension, uint>& idx) const
		{
			return mIndex.LinearIndex(idx);
		}
		__forceinline virtual Vec<Dimension, uint> Index(size_t linearIdx) const
		{
			return mIndex.Index(linearIdx);
		}
		__forceinline size_t LinearSize() const
		{
			return mIndex.LinearSize();
		}
		__forceinline size_t Size(uint iDim) const
		{
			return mIndex.Size(iDim);
		}
		__forceinline Vec<Dimension, uint> Size() const
		{
			return mIndex.Size();
		}
		__forceinline size_t Stride(uint iDim) const
		{
			return mIndex.Stride(iDim);
		}

		__forceinline T& operator [] (const Vec<Dimension, uint>& idx)
		{
			return mpData[LinearIndex(idx)];
		}
		__forceinline const T operator [] (const Vec<Dimension, uint>& idx) const
		{
			return mpData[LinearIndex(idx)];
		}
		__forceinline T& operator [] (const size_t idx)
		{
			Assert(idx < mIndex.LinearSize());
			return mpData[idx];
		}
		__forceinline const T operator [] (const size_t idx) const
		{
			Assert(idx < mIndex.LinearSize());
			return mpData[idx];
		}
		__forceinline const T* Data() const
		{
			return mpData;
		}
		__forceinline T* ModifiableData()
		{
			return mpData;
		}

		void Free()
		{
			Memory::SafeFree(mpData);
		}
	};

	template<size_t Dimension, class T>
	void ToStlVector(std::vector<T>& lhs, const DimensionalArray<Dimension, T>& rhs)
	{
		lhs.clear();
		lhs.resize(rhs.LinearSize());

		Memory::Memcpy(lhs.data(), rhs.Data(), rhs.LinearSize() * sizeof(T));
	}

	typedef DimensionalArray<1, float> Array1f;
	typedef DimensionalArray<2, float> Array2f;
	typedef DimensionalArray<3, float> Array3f;

	typedef DimensionalArray<1, double> Array1d;
	typedef DimensionalArray<2, double> Array2d;
	typedef DimensionalArray<3, double> Array3d;
}