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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "Core/Types.h"
#include "Containers/Algorithm.h"
#include "Core/Memory.h"
#include "Math/Vector.h"
#include "Core/Random.h"

#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include "TensorJitKernel.cuh"
#include "PtrWrapper.h"
#include "../LibTensorRay/Renderer/Config.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <map>
using namespace std;

#include <ppl.h>
using namespace concurrency;

#include <cub/util_allocator.cuh>

#if USE_PROFILING
#include "nvToolsExt.h"
#endif

#define TENSOR_INLINE inline

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	Assert(size > 0);
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

#ifdef _DEBUG
static const bool HOST_DEBUG = true;
#else
static const bool HOST_DEBUG = false;
#endif



namespace EDX
{
	/**
	* Does a boolean AND of the ::Value static members of each type, but short-circuits if any Type::Value == false.
	*/
	template <typename... Types>
	struct And;

	template <bool LHSValue, typename... RHS>
	struct AndValue
	{
		enum { Value = And<RHS...>::Value };
	};

	template <typename... RHS>
	struct AndValue<false, RHS...>
	{
		enum { Value = false };
	};

	template <typename LHS, typename... RHS>
	struct And<LHS, RHS...> : AndValue<LHS::Value, RHS...>
	{
	};

	template <>
	struct And<>
	{
		enum { Value = true };
	};

	/**
	* Does a boolean OR of the ::Value static members of each type, but short-circuits if any Type::Value == true.
	*/
	template <typename... Types>
	struct Or;

	template <bool LHSValue, typename... RHS>
	struct OrValue
	{
		enum { Value = Or<RHS...>::Value };
	};

	template <typename... RHS>
	struct OrValue<true, RHS...>
	{
		enum { Value = true };
	};

	template <typename LHS, typename... RHS>
	struct Or<LHS, RHS...> : OrValue<LHS::Value, RHS...>
	{
	};

	template <>
	struct Or<>
	{
		enum { Value = false };
	};

	/**
	* Does a boolean NOT of the ::Value static members of the type.
	*/
	template <typename Type>
	struct Not
	{
		enum { Value = !Type::Value };
	};


	/** Tests whether two typenames refer to the same type. */
	template<typename A, typename B>
	struct AreTypesEqual;

	template<typename, typename>
	struct AreTypesEqual
	{
		enum { Value = false };
	};

	template<typename A>
	struct AreTypesEqual<A, A>
	{
		enum { Value = true };
	};

	/**
	* IsFloatType
	*/
	template<typename T> struct IsFloatType { enum { Value = false }; };

	template<> struct IsFloatType<float> { enum { Value = true }; };
	template<> struct IsFloatType<double> { enum { Value = true }; };
	template<> struct IsFloatType<long double> { enum { Value = true }; };

	/**
	* IsIntegralType
	*/
	template<typename T> struct IsIntegralType { enum { Value = false }; };

	template<> struct IsIntegralType<uint8> { enum { Value = true }; };
	template<> struct IsIntegralType<uint16> { enum { Value = true }; };
	template<> struct IsIntegralType<uint32> { enum { Value = true }; };
	template<> struct IsIntegralType<uint64> { enum { Value = true }; };

	template<> struct IsIntegralType<int8> { enum { Value = true }; };
	template<> struct IsIntegralType<int16> { enum { Value = true }; };
	template<> struct IsIntegralType<int32> { enum { Value = true }; };
	template<> struct IsIntegralType<int64> { enum { Value = true }; };

	template<> struct IsIntegralType<bool> { enum { Value = true }; };

	template<> struct IsIntegralType<WIDECHAR> { enum { Value = true }; };
	template<> struct IsIntegralType<ANSICHAR> { enum { Value = true }; };

	/**
	* IsSignedIntegralType
	*/
	template<typename T> struct IsSignedIntegralType { enum { Value = false }; };

	template<> struct IsSignedIntegralType<int8> { enum { Value = true }; };
	template<> struct IsSignedIntegralType<int16> { enum { Value = true }; };
	template<> struct IsSignedIntegralType<int32> { enum { Value = true }; };
	template<> struct IsSignedIntegralType<int64> { enum { Value = true }; };

	/**
	* IsArithmeticType
	*/
	template<typename T> struct IsArithmeticType
	{
		enum { Value = IsIntegralType<T>::Value || IsFloatType<T>::Value };
	};


	// NestedInitializerList

	template <class T, SIZE_T I>
	struct NestedInitializerListImpl
	{
		using Type = std::initializer_list<typename NestedInitializerListImpl<T, I - 1>::Type>;
	};

	template <class T>
	struct NestedInitializerListImpl<T, 0>
	{
		using Type = T;
	};

	template <class T, SIZE_T I>
	using NestedInitializerList = typename NestedInitializerListImpl<T, I>::Type;

	// InitListNestedCopy implementation

	template <class T, class S>
	inline void InitListNestedCopy(T&& iter, const S& s)
	{
		*iter++ = s;
	}

	template <class T, class S>
	inline void InitListNestedCopy(T&& iter, std::initializer_list<S> s)
	{
		for (auto it = s.begin(); it != s.end(); ++it)
		{
			InitListNestedCopy(std::forward<T>(iter), *it);
		}
	}


	// InitializerListDimension implementation
	template <class U>
	struct InitializerListDimension
	{
		static constexpr SIZE_T Value = 0;
	};

	template <class T>
	struct InitializerListDimension<std::initializer_list<T>>
	{
		static constexpr SIZE_T Value = 1 + InitializerListDimension<T>::Value;
	};


	// InitializerListShape implementation

	template <SIZE_T I>
	struct InitializerListShapeImpl
	{
		template <class T>
		static constexpr SIZE_T Value(T t)
		{
			return t.size() == 0 ? 0 : InitializerListShapeImpl<I - 1>::Value(*t.begin());
		}
	};

	template <>
	struct InitializerListShapeImpl<0>
	{
		template <class T>
		static constexpr SIZE_T Value(T t)
		{
			return t.size();
		}
	};

	template <class Ret, class T, SIZE_T... I>
	constexpr Ret InitializerListShape(T t, std::index_sequence<I...>)
	{
		using SizeType = typename Ret::ElementType;
		return { SizeType(InitializerListShapeImpl<I>::Value(t))... };
	}

	template <class Ret, class T>
	constexpr Ret DeriveShapeFromNestedInitList(T t)
	{
		return InitializerListShape<Ret, decltype(t)>(t, std::make_index_sequence<InitializerListDimension<decltype(t)>::Value>());
	}

	namespace DeepLearning
	{
		template<typename First, typename... Rest>
		struct AllIntegralType
		{
			enum { Value = And<IsIntegralType<First>, AllIntegralType<Rest...>>::Value };
		};

		template<typename Last>
		struct AllIntegralType<Last>
		{
			enum { Value = IsIntegralType<Last>::Value };
		};

		enum class Type
		{
			Bool = 0,
			Int,
			Uint,
			Float,
			Double,
			Undefined = -1
		};

		static Type DeriveType(const Type t1, const Type t2)
		{
			Assert(t1 != Type::Undefined);
			Assert(t2 != Type::Undefined);
			return Type(max(t1, t2));
		}

		template<typename T>
		static Type DeriveType()
		{
			if (std::is_same<T, bool>::value)
				return Type::Bool;
			else if (std::is_same<T, uint>::value)
				return Type::Uint;
			else if (std::is_same<T, int>::value)
				return Type::Int;
			else if (std::is_same<T, float>::value)
				return Type::Float;
			else if (std::is_same<T, double>::value)
				return Type::Double;
			else
			{
				AssertNoEntry();
				return Type::Undefined;
			}
		}

		static size_t SizeOf(Type t)
		{
			switch (t)
			{
			case Type::Bool:
				return sizeof(bool);
			case Type::Int:
				return sizeof(int);
			case Type::Uint:
				return sizeof(uint);
			case Type::Float:
				return sizeof(float);
			case Type::Double:
				return sizeof(double);
			}

			AssertNoEntry();

			return 0;
		}
		

		enum VecType
		{
			Scalar1 = 1,
			Vec2,
			Vec3,
			Vec4,
			Mat4x4 = 16,
			Undefined = -1
		};

		struct Shape
		{
			typedef int ElementType;

			static const int MaxArraySize = 4;
			int x0 = 0;
			int x1 = 0;
			int x2 = 0;
			int x3 = 0;
			int mSize = 0;
			VecType mVecType = VecType::Scalar1;
			bool mTransposed = false;

			TENSOR_INLINE Shape()
			{
				mSize = x0 = x1 = x2 = x3 = 0;
			}


			inline Shape(std::initializer_list<int> InitList, const int vecType = 1)
			{
				SetVectorType(VecType(vecType));
				this->operator=(InitList);
			}

			inline Shape(std::vector<int> InitList, const int vecType = 1)
			{
				SetVectorType(VecType(vecType));
				this->operator=(InitList);
			}

			inline void operator = (std::initializer_list<int> InitList)
			{
				Assign(InitList.begin(), InitList.size());
			}

			inline void operator = (std::vector<int> InitList)
			{
				Assign(InitList.data(), InitList.size());
			}

			TENSOR_INLINE Shape(const int _x0, const int _x1)
			{
				x0 = _x0;
				x1 = _x1;
				mSize = 2;
			}

			void SetVectorType(const VecType vecType)
			{
				mVecType = vecType;
				Assert(int(mVecType) >= 1);
			}

			void SetVectorType(const int vecType)
			{
				mVecType = VecType(vecType);
				Assert(int(mVecType) >= 1);
			}

			TENSOR_INLINE int VectorSize() const
			{
				return int(mVecType);
			}

			TENSOR_INLINE bool operator == (const Shape& rhs) const
			{
				if (mSize != rhs.Size())
				{
					return false;
				}

				return x0 == rhs.x0 &&
					x1 == rhs.x1 &&
					x2 == rhs.x2 &&
					x3 == rhs.x3 &&
					mVecType == rhs.mVecType;
			}

			TENSOR_INLINE bool operator != (const Shape& rhs) const
			{
				return !(*this == rhs);
			}

			TENSOR_INLINE bool operator < (const Shape& rhs) const
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
									(mVecType != rhs.mVecType ?
										mVecType < rhs.mVecType :
											false))))));
			}


			TENSOR_INLINE int operator [] (const int idx) const
			{
				Assert(idx < MaxArraySize);
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

				Assert(0);
				return -1;
			}

			TENSOR_INLINE int& operator [] (const int idx)
			{
				Assert(idx < MaxArraySize);
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

				Assert(0);
				return x0;
			}


			TENSOR_INLINE int Size() const
			{
				return mSize;
			}

			TENSOR_INLINE bool Empty() const
			{
				return mSize == 0;
			}

			inline void Clear(int32 Slack = 0)
			{
				ResizeZeroed(Slack);
			}

			TENSOR_INLINE void Resize(const int size)
			{
				Assert(size <= MaxArraySize);
				mSize = size;
			}

			inline void ResizeZeroed(const int size)
			{
				Assert(size <= MaxArraySize);

				mSize = size;
				x0 = 0;
				x1 = 0;
				x2 = 0;
				x3 = 0;
			}

			inline void Assign(const int* Vals, const int32 Count)
			{
				mSize = x0 = x1 = x2 = x3 = 0;

				Resize(Count);

				if (mSize == 1)
				{
					x0 = Vals[0];
				}
				else if (mSize == 2)
				{
					x0 = Vals[0];
					x1 = Vals[1];
				}
				else if (mSize == 3)
				{
					x0 = Vals[0];
					x1 = Vals[1];
					x2 = Vals[2];
				}
				else if (mSize == 4)
				{
					x0 = Vals[0];
					x1 = Vals[1];
					x2 = Vals[2];
					x3 = Vals[3];
				}
			}

			inline void Assign(const Shape& Vals, const int32 InitDim, const int32 Count)
			{
				mSize = x0 = x1 = x2 = x3 = 0;

				Resize(Count);

				if (mSize == 1)
				{
					x0 = Vals[0 + InitDim];
				}
				else if (mSize == 2)
				{
					x0 = Vals[0 + InitDim];
					x1 = Vals[1 + InitDim];
				}
				else if (mSize == 3)
				{
					x0 = Vals[0 + InitDim];
					x1 = Vals[1 + InitDim];
					x2 = Vals[2 + InitDim];
				}
				else if (mSize == 4)
				{
					x0 = Vals[0 + InitDim];
					x1 = Vals[1 + InitDim];
					x2 = Vals[2 + InitDim];
					x3 = Vals[3 + InitDim];
				}
			}

			TENSOR_INLINE bool Contains(const int val) const
			{

				if (mSize == 1)
				{
					return x0 == val;
				}
				else if (mSize == 2)
				{
					return x0 == val ||
						x1 == val;
				}
				else if (mSize == 3)
				{
					return x0 == val ||
						x1 == val ||
						x2 == val;
				}
				else if (mSize == 4)
				{
					return x0 == val ||
						x1 == val ||
						x2 == val ||
						x3 == val;
				}

				return false;
			}

			TENSOR_INLINE bool Add(const int val)
			{
				Assert(mSize < 4);
				(*this)[mSize++] = val;

				return false;
			}
			inline int LinearSize() const
			{
				if (mSize == 1)
				{
					return x0 * VectorSize();
				}
				else if (mSize == 2)
				{
					return x0 * x1 * VectorSize();
				}
				else if (mSize == 3)
				{
					return x0 * x1 * x2 * VectorSize();
				}
				else if (mSize == 4)
				{
					return x0 * x1 * x2 * x3 * VectorSize();
				}
				else if (mSize == 0)
				{
					return 0;
				}

				Assert(0);
				return -1;
			}

			inline int VectorLinearSize() const
			{
				if (mSize == 1)
				{
					return x0;
				}
				else if (mSize == 2)
				{
					return x0 * x1;
				}
				else if (mSize == 3)
				{
					return x0 * x1 * x2;
				}
				else if (mSize == 4)
				{
					return x0 * x1 * x2 * x3;
				}
				else if (mSize == 0)
				{
					return 0;
				}

				Assert(0);
				return -1;
			}

			inline int NumElements() const
			{
				if (mSize == 1)
				{
					return x0;
				}
				else if (mSize == 2)
				{
					return x0 * x1;
				}
				else if (mSize == 3)
				{
					return x0 * x1 * x2;
				}
				else if (mSize == 4)
				{
					return x0 * x1 * x2 * x3;
				}

				Assert(0);
				return -1;
			}


			inline Shape LinearizeVector() const
			{
				if (mVecType == VecType::Scalar1)
				{
					return *this;
				}
				else
				{
					if (mSize == 1)
					{
						return { VectorSize(), x0 };
					}
					else if (mSize == 2)
					{
						return { VectorSize(), x0, x1 };
					}
					else if (mSize == 3)
					{
						return { VectorSize(), x0, x1, x2 };
					}
					else if (mSize == 4)
					{
						return { VectorSize(), x0, x1, x2, x3 };
					}
				}

				Assert(0);
				return { 0 };
			}

			inline Shape Vectorize() const
			{
				if (mVecType == VecType::Scalar1)
				{
					if (mSize == 1)
					{
						return Shape({ 1 }, x0);
					}
					else if (mSize == 2)
					{
						return Shape({ x1 }, x0);
					}
					else if (mSize == 3)
					{
						return Shape({ x1, x2 }, x0);
					}
					else if (mSize == 4)
					{
						return Shape({ x1, x2, x3 }, x0);
					}
				}

				return *this;
			}

			Shape Transpose(const Shape& transposeDim = {})
			{
				Shape ret = *this;
				if (transposeDim.Empty())
				{
					for (int i = 0; i < Size(); i++)
					{
						ret[i] = (*this)[Size() - 1 - i];
					}
				}
				else
				{
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						ret[i] = (*this)[transposeDim[i]];
					}
				}

				ret.mTransposed = true;

				return ret;
			}

			TENSOR_INLINE bool WithinRange(const Shape& shape) const
			{
				if (mSize == 1)
				{
					return shape.x0 >= 0 && shape.x0 < x0;
				}
				else if (mSize == 2)
				{
					return shape.x0 >= 0 && shape.x0 < x0 &&
						shape.x1 >= 0 && shape.x1 < x1;
				}
				else if (mSize == 3)
				{
					return shape.x0 >= 0 && shape.x0 < x0 &&
						shape.x1 >= 0 && shape.x1 < x1 &&
						shape.x2 >= 0 && shape.x2 < x2;
				}
				else if (mSize == 4)
				{
					return shape.x0 >= 0 && shape.x0 < x0 &&
						shape.x1 >= 0 && shape.x1 < x1 &&
						shape.x2 >= 0 && shape.x2 < x2 &&
						shape.x3 >= 0 && shape.x3 < x3;
				}

				return false;
			}

			ShapeJit ToJit() const
			{
				ShapeJit ret;

				ret.mSize = mSize;
				ret.x0 = x0;
				ret.x1 = x1;
				ret.x2 = x2;
				ret.x3 = x3;
				ret.mVecSize = int(mVecType);

				return ret;
			}
		};

		class TensorParams
		{
		public:
			Shape mShape;
			Shape mStrides;
			Shape mVectorStrides;
			int mLinearSize;
			int mNumElements;

		public:
			inline TensorParams()
				: mLinearSize(0)
				, mNumElements(0)
			{
			}

			template<typename... TShape>
			void Resize(TShape... shape)
			{
				static_assert(AllIntegralType<TShape...>::Value, "All parameters have to be integral type.");

				mShape = { shape... };
				Resize_Common();
			}

			void Resize(const Shape& shape)
			{
				mShape = shape;
				Resize_Common();
			}

			template<typename... TShape>
			void Reshape(TShape... shape)
			{
				static_assert(AllIntegralType<TShape...>::Value, "All parameters have to be integral type.");

#ifdef _DEBUG
				auto oldSize = mLinearSize;
#endif

				mShape = { shape... };
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}

			void Reshape(const Shape& shape)
			{
#ifdef _DEBUG
				auto oldSize = mLinearSize;
#endif

				mShape = shape;
				Resize_Common();

				Assert(mLinearSize == oldSize);
			}
			
			void Transpose(const Shape& transposeDim = {})
			{
				if (transposeDim.Empty())
				{
					// TODO: Implement Transpose in Shape
					Shape shapeCopy = mShape;
					Shape strideCopy = mStrides;
					for (int i = 0; i < mShape.Size(); i++)
					{
						mShape[i] = shapeCopy[mShape.Size() - 1 - i];
						mStrides[i] = strideCopy[mShape.Size() - 1 - i];
					}
				}
				else
				{
					Shape shapeCopy = mShape;
					Shape strideCopy = mStrides;
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						mShape[i] = shapeCopy[transposeDim[i]];
						mStrides[i] = strideCopy[transposeDim[i]];
					}
				}

				mShape.mTransposed = true;
			}

			template<typename... Index>
			TENSOR_INLINE int LinearIndex(Index... idx) const
			{
				constexpr int size = sizeof...(idx);
				Assertf(size == mShape.Size(), "Input index dimension does not match with array dimension.");

				int _idx[size] = { idx... };
				return LinearIndex<size>(_idx);
			}

			TENSOR_INLINE int LinearIndex(const Shape& idx) const
			{
				const int dim = idx.Size();
				Assertf(dim == mShape.Size(), "Input index dimension does not match with array dimension.");

				int ret = 0;
				ret = idx.x0 * mStrides.x0 +
					idx.x1 * mStrides.x1 +
					idx.x2 * mStrides.x2 +
					idx.x3 * mStrides.x3;

				Assert(ret < mLinearSize);
				return ret;
			}

			template<int dim>
			TENSOR_INLINE int LinearIndex(const int idx[dim]) const
			{
				Assertf(dim == mShape.Size(), "Input index dimension does not match with array dimension.");

				int ret = 0;
				if (dim == 1)
				{
					ret = idx[0] * mStrides.x0;
				}
				else if (dim == 2)
				{
					ret = idx[0] * mStrides.x0 +
						idx[1] * mStrides.x1;
				}
				else if (dim == 3)
				{
					ret = idx[0] * mStrides.x0 +
						idx[1] * mStrides.x1 +
						idx[2] * mStrides.x2;
				}
				else if (dim == 4)
				{
					ret = idx[0] * mStrides.x0 +
						idx[1] * mStrides.x1 +
						idx[2] * mStrides.x2 +
						idx[3] * mStrides.x3;
				}

				Assert(ret < mLinearSize);
				return ret;
			}

			TENSOR_INLINE Shape Index(int linearIdx) const
			{
				Assert(linearIdx < mLinearSize);

				Shape vRet;
				int dim = mShape.Size();

				const Shape& strides = mStrides;

				vRet.Resize(dim);
				if (dim == 1)
				{
					vRet.x0 = linearIdx / strides.x0;
				}
				else if (dim == 2)
				{
					vRet.x0 = linearIdx / strides.x0;
					linearIdx %= strides.x0;

					vRet.x1 = linearIdx / strides.x1;
				}
				else if (dim == 3)
				{
					vRet.x0 = linearIdx / strides.x0;
					linearIdx %= strides.x0;

					vRet.x1 = linearIdx / strides.x1;
					linearIdx %= strides.x1;

					vRet.x2 = linearIdx / strides.x2;
				}
				else if (dim == 4)
				{
					vRet.x0 = linearIdx / strides.x0;
					linearIdx %= strides.x0;

					vRet.x1 = linearIdx / strides.x1;
					linearIdx %= strides.x1;

					vRet.x2 = linearIdx / strides.x2;
					linearIdx %= strides.x2;

					vRet.x3 = linearIdx / strides.x3;
				}

				return vRet;
			}

			TENSOR_INLINE Shape ShiftedIndex(int linearIdx) const
			{
				Assert(linearIdx < mLinearSize);

				Shape vRet;
				int dim = mShape.Size();

				vRet.Resize(dim);
				if (dim == 1)
				{
					vRet.x3 = linearIdx / mStrides.x0;
				}
				else if (dim == 2)
				{
					vRet.x2 = linearIdx / mStrides.x0;
					linearIdx %= mStrides.x0;

					vRet.x3 = linearIdx / mStrides.x1;
				}
				else if (dim == 3)
				{
					vRet.x1 = linearIdx / mStrides.x0;
					linearIdx %= mStrides.x0;

					vRet.x2 = linearIdx / mStrides.x1;
					linearIdx %= mStrides.x1;

					vRet.x3 = linearIdx / mStrides.x2;
				}
				else if (dim == 4)
				{
					vRet.x0 = linearIdx / mStrides.x0;
					linearIdx %= mStrides.x0;

					vRet.x1 = linearIdx / mStrides.x1;
					linearIdx %= mStrides.x1;

					vRet.x2 = linearIdx / mStrides.x2;
					linearIdx %= mStrides.x2;

					vRet.x3 = linearIdx / mStrides.x3;
				}

				return vRet;
			}

			TENSOR_INLINE int LinearSize() const
			{
				return mShape.LinearSize();
			}

			TENSOR_INLINE int NumElements() const
			{
				return mShape.NumElements();
			}
			
			TENSOR_INLINE bool IndexRangeCheck(const Shape& index) const
			{
				for (int i = 0; i < mShape.Size(); i++)
				{
					if (index[i] >= mShape[i])
						return false;
				}
				return true;
			}

			TENSOR_INLINE void IterateIndex(Shape& index) const
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

			TENSOR_INLINE bool IterateIndex(Shape& index, const Shape& axes /*axes to iterate through*/) const
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

			TENSOR_INLINE int GetShape(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mShape[iDim];
			}

			TENSOR_INLINE const Shape& GetShape() const
			{
				return mShape;
			}

			TENSOR_INLINE Shape GetShape()
			{
				return mShape;
			}

			TENSOR_INLINE int Stride(int iDim) const
			{
				Assert(iDim < mShape.Size());
				return mStrides[iDim];
			}

			TENSOR_INLINE const Shape& Stride() const
			{
				return mStrides;
			}

			TENSOR_INLINE int VectorSize() const
			{
				return mShape.VectorSize();
			}

			inline TensorParams GetSliceIndex(int subDim) const
			{
				Assert(subDim <= mShape.Size());
				TensorParams ret;

				if (subDim < mShape.Size())
				{
					ret.mShape.Assign(mShape, subDim, mShape.Size() - subDim);
				}
				else
				{
					ret.mShape = { 1 };
				}

				ret.Resize_Common();
				return ret;
			}

			inline TensorParams GetSectionIndex(int num) const
			{
				Assert(num < mShape[0]);
				TensorParams ret;

				ret.mShape = mShape;
				ret.mShape[0] = num;

				ret.Resize_Common();
				return ret;
			}

			TENSOR_INLINE bool IsTransposed() const
			{
				return mShape.mTransposed;
			}

			TensorParamsJit ToJit() const
			{
				TensorParamsJit ret;

				ret.mShape = mShape.ToJit();
				ret.mStrides = mStrides.ToJit();
				ret.mLinearSize = mLinearSize;
				ret.mNumElements = mNumElements;

				return ret;
			}

		private:
			void CalcStrides()
			{
				mStrides.Resize(4);

				for (auto i = 0; i < 4; i++)
				{
					mStrides[i] = 1;
					for (auto dim = mShape.Size() - 1; dim > i; dim--)
						mStrides[i] *= mShape[dim];
				}
			}

			inline void Resize_Common()
			{
				CalcStrides();

				mLinearSize = mShape.Size() > 0 ? 1 : 0;
				for (auto i = 0; i < mShape.Size(); i++)
					mLinearSize *= mShape[i];

				mNumElements = mLinearSize;
				mLinearSize *= int(mShape.mVecType);
			}
		};

		static Shape BroadcastShape(const Shape& leftShape, const Shape& rightShape)
		{
			if (leftShape == rightShape) // Trivially broadcasted
			{
				return leftShape;
			}

			const int leftDim = leftShape.Size();
			const int rightDim = rightShape.Size();
			const auto& greaterShape = leftDim > rightDim ? leftShape : rightShape;
			const int retDim = greaterShape.Size();

			Shape ret;
			ret.Resize(retDim);

			int k = retDim - 1;
			for (int i = leftDim - 1, j = rightDim - 1;
				i >= 0 && j >= 0;
				i--, j--, k--)
			{
				Assertf(leftShape[i] == rightShape[j] ||
					leftShape[i] == 1 ||
					rightShape[j] == 1, "Tensor dimensions not aligned");

				ret[k] = Math::Max(leftShape[i], rightShape[j]);
			}

			while (k >= 0)
			{
				ret[k] = greaterShape[k];
				k--;
			}

			if (leftShape.mVecType == VecType::Scalar1 || rightShape.mVecType == VecType::Scalar1)
			{
				ret.SetVectorType((VecType)Math::Max((int)leftShape.mVecType, (int)rightShape.mVecType));
			}
			else
			{
				Assert(leftShape.mVecType == rightShape.mVecType);
				ret.SetVectorType(leftShape.mVecType);
			}

			return ret;
		}

		struct Exp;

		struct Expr
		{
			shared_ptr<Exp> ptr;

			~Expr()
			{
				ptr = nullptr;
			}

			Expr()
			{
			}

			Expr(const shared_ptr<Exp>& p)
				: ptr(p)
			{
			}

			Expr(shared_ptr<Exp>&& p)
				: ptr(std::move(p))
			{
			}

			Expr(const shared_ptr<Exp>&& p)
				: ptr(std::move(p))
			{
			}

			template<typename T>
			Expr(const shared_ptr<T>& p)
				: ptr(static_pointer_cast<Exp>(p))
			{
			}

			template<typename T>
			Expr(shared_ptr<T>&& p)
				: ptr(std::move(static_pointer_cast<Exp>(p)))
			{
			}

			template<typename T>
			Expr(const shared_ptr<T>&& p)
				: ptr(std::move(static_pointer_cast<Exp>(p)))
			{
			}

			Expr(const Exp& p);

			const shared_ptr<Exp>& operator->() const
			{
				return ptr;
			}
			shared_ptr<Exp>& operator->()
			{
				return ptr;
			}
			operator bool() const
			{
				return bool(ptr);
			}
		};

		// Singleton for storing cublas handle
		class Cublas
		{
		public:
			static cublasHandle_t& GetHandle()
			{
				static cublasHandle_t Handle;
				return Handle;
			}
		public:
			Cublas(Cublas const&) = delete;
			void operator = (Cublas const&) = delete;
		};

		// Singleton for storing cublas handle
		class Curand
		{
		public:
			static curandGenerator_t& GetHandle()
			{
				static curandGenerator_t Handle;
				return Handle;
			}
		public:
			Curand(Curand const&) = delete;
			void operator = (Curand const&) = delete;
		};


		// Singleton for storing cublas handle
		class CuAllocator
		{
		public:
			static cub::CachingDeviceAllocator& GetHandle()
			{
				static cub::CachingDeviceAllocator Handle(2u);
				return Handle;
			}
		public:
			CuAllocator(CuAllocator const&) = delete;
			void operator = (CuAllocator const&) = delete;
		};

		class ParameterPool
		{
		public:
			static vector<Expr>& GetHandle()
			{
				static vector<Expr> ParamPool;
				return ParamPool;
			}
		public:
			ParameterPool(ParameterPool const&) = delete;
			void operator = (ParameterPool const&) = delete;
		};

		class VisitedSet
		{
		public:
			static set<const Exp*>& GetHandle()
			{
				static set<const Exp*> Visited;
				return Visited;
			}
		public:
			static bool Visited(const Exp* node)
			{
				bool ret = true;
				if (GetHandle().find(node) == GetHandle().end())
					ret = false;

				GetHandle().insert(node);
				return ret;
			}

			VisitedSet(VisitedSet const&) = delete;
			void operator = (VisitedSet const&) = delete;
		};


		class GlobalVisitedSet
		{
		public:
			static set<const Exp*>& GetHandle()
			{
				static set<const Exp*> Visited;
				return Visited;
			}
		public:
			static bool Visited(const Exp* node)
			{
				bool ret = true;
				if (GetHandle().find(node) == GetHandle().end())
				{
					ret = false;
				}
				return ret;
			}
			static void SetVisited(const Exp* node)
			{
				GetHandle().insert(node);
			}
			static void Remove(const Exp* node)
			{
				GetHandle().erase(node);
			}
			static void Clear()
			{
				GetHandle().clear();
				return;
			}

			GlobalVisitedSet(GlobalVisitedSet const&) = delete;
			void operator = (GlobalVisitedSet const&) = delete;
		};

		class GraphProcessContext
		{
		};

		class VariableCache
		{
		public:
			static map<std::tuple<const Exp*, string, string>, int>& GetHandle()
			{
				static map<std::tuple<const Exp*, string, string>, int> Cache;
				return Cache;
			}
		public:
			VariableCache(VariableCache const&) = delete;
			void operator = (VariableCache const&) = delete;
		};


		class TensorVariableCache
		{
		public:
			static map<std::tuple<const void*, string, string>, int>& GetHandle()
			{
				static map<std::tuple<const void*, string, string>, int> Cache;
				return Cache;
			}
		public:
			TensorVariableCache(TensorVariableCache const&) = delete;
			void operator = (TensorVariableCache const&) = delete;
		};

		class ForwardDiffVariableCache
		{
		public:
			static map<const void*, Expr>& GetHandle()
			{
				static map<const void*, Expr> Cache;
				return Cache;
			}
		public:
			ForwardDiffVariableCache(ForwardDiffVariableCache const&) = delete;
			void operator = (ForwardDiffVariableCache const&) = delete;
		};


		class KernelLaunchCounter
		{
		public:
			static int& GetHandle()
			{
				static int counter = 0;
				return counter;
			}
			static void Increment()
			{
				int& counter = GetHandle();
				counter++;
			}
			static void Reset()
			{
				int& counter = GetHandle();
				counter = 0;
			}
		public:
			KernelLaunchCounter(KernelLaunchCounter const&) = delete;
			void operator = (KernelLaunchCounter const&) = delete;
		};

		struct VariableMap
		{
			map<TensorJitArg, int> tensorArgs;
			map<ConcatIndex, int> concatArgs;
			map<SliceIndex, int> sliceArgs;
			map<IndexedReadArg, int> indexedReadArgs;
		};


		struct Exp
		{
			bool mbRequiresGrad = false;
			Shape mShape;
			Type mType;

			mutable Expr value;
			mutable bool mValueCached = false;

			virtual ~Exp()
			{
				GlobalVisitedSet::Remove(this);
			}

			virtual void Backward(const struct Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const
			{
			}

			virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const;

			virtual TENSOR_INLINE Shape GetShape() const
			{
				return mShape;
			}

			virtual Type GetType() const
			{
				return mType;
			}

			virtual shared_ptr<Exp> ToShared() const
			{
				return make_shared<Exp>();
			}


			virtual bool Empty() const
			{
				return true;
			}

			virtual void Resize(const Shape& shape)
			{
			}

			virtual void Forward(const Expr& rhs, const bool bForceRecompute)
			{
				Forward(rhs.ptr.get(), bForceRecompute);
			}

			virtual void Forward(Exp* rhs, const bool bForceRecompute)
			{
			}

			virtual void GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace = false, const string& op = "=")
			{
			}

			virtual void RecursiveProcess(class GraphProcessContext& context, const bool bForceRecompute = false)
			{
			}

			virtual string EmitCuda(struct VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false)
			{
				return "";
			}

			virtual void TopologicalSort(vector<const Exp*>& sorted) const
			{
			}

			virtual void Update(const float scale = 1.0f)
			{
			}
		};

		enum DeviceType
		{
			CPU, GPU
		};

		template<typename T>
		struct Deleter
		{
			void operator () (T* ptr)
			{
				if (ptr)
					CuAllocator::GetHandle().DeviceFree(ptr);
			}
		};

		template<typename T>
		struct DeleterHost
		{
			void operator () (T* ptr)
			{
				Memory::SafeFree(ptr);
			}
		};

		template<class T>
		class Tensor : public Exp
		{
		protected:
			shared_ptr<T> mpData;
			shared_ptr<T> mpHostData;
			Expr mpExp;
			mutable shared_ptr<Expr> mpGrad;

		public:
			TensorParams mParams;

			bool RequiresGrad() const
			{
				return mbRequiresGrad;
			}

			void SetRequiresGrad(const bool b)
			{
				mbRequiresGrad = b;

				if (!mpGrad && mbRequiresGrad)
				{
					mpGrad = make_shared<Expr>();
					*mpGrad = Zeros(1);
					ParameterPool::GetHandle().push_back(*this);
				}
			}

			void ReleaseGraph()
			{
				mpExp.ptr = nullptr;
			}

			void EvalAndReleaseGradGraph()
			{
				Tensor<T> gradTensor = this->Grad();
				gradTensor.ReleaseGraph();
				*mpGrad = gradTensor;
			}

			// For memory debug
			int GetGradPtrUseCount()
			{
				return mpGrad.use_count();
			}

			Tensor()
			{
				mbRequiresGrad = false;
			}

			virtual ~Tensor()
			{
				Free();
			}

			Tensor(const Tensor& rhs)
			{
				this->operator=(rhs);
			}

			Tensor(Tensor&& rhs)
			{
				this->operator=(std::move(rhs));
			}

			Tensor(const Tensor&& rhs)
			{
				this->operator=(std::move(rhs));
			}

			Tensor(const T& val)
			{
				this->operator=(val);
			}

			Tensor& operator = (const Tensor& rhs)
			{
				Free();
				mParams = rhs.mParams;
				mpData = rhs.mpData;
				mpHostData = rhs.mpHostData;
				mpGrad = rhs.mpGrad;
				mbRequiresGrad = rhs.mbRequiresGrad;
				mpExp = rhs.mpExp;
				return *this;
			}

			Tensor& operator = (Tensor&& rhs)
			{
				Free();
				mParams = rhs.mParams;
				mpData = std::move(rhs.mpData);
				mpHostData = rhs.mpHostData;
				mpExp = std::move(rhs.mpExp);
				mpGrad = std::move(rhs.mpGrad);
				mbRequiresGrad = rhs.mbRequiresGrad;
				return *this;
			}

			Tensor& operator = (const Tensor&& rhs)
			{
				Free();
				mParams = rhs.mParams;
				mpData = std::move(rhs.mpData);
				mpHostData = std::move(rhs.mpHostData);
				mpExp = std::move(rhs.mpExp);
				mpGrad = std::move(rhs.mpGrad);
				mbRequiresGrad = rhs.mbRequiresGrad;
				return *this;
			}

			Tensor& operator = (const T val)
			{
				if (LinearSize() != 1)
				{
					Free();
					Resize(1);
				}

				cudaMemcpy(mpData.get(), &val, sizeof(T), cudaMemcpyHostToDevice);
				CopyToHost();

				return *this;
			}


			virtual shared_ptr<Exp> ToShared() const
			{
				return make_shared<Tensor<T>>(*this);
			}

			// ----------------------------------------------------------------
			// Constructor and assignment override for template expression
			// ----------------------------------------------------------------
			Tensor(const Expr& rhs)
			{
				this->operator=(rhs);
			}

			Tensor<T>& operator = (const Expr& rhs)
			{
				return EvalExpr(rhs);
			}

			Tensor<T>& EvalExpr(const Expr& rhs);

			void Forward(const Expr& rhs, const bool bForceRecompute = false) override;
			void Forward(Exp* rhs, const bool bForceRecompute = false) override;
			void GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace = false, const string& op = "=") override;

			virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override;

			virtual void TopologicalSort(vector<const Exp*>& sorted) const override
			{
				if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
					return;

				VisitedSet::GetHandle().insert(this);

				if (mpExp)
				{
					mpExp->TopologicalSort(sorted);
				}

				sorted.push_back(this);
			}

			virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;
			void Backward(const Expr& grad) const;
			void Backward() const;
			virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const;

			inline void RecursiveProcess(
				GraphProcessContext& context,
				const bool bForceRecompute = false) override
			{
				if (GlobalVisitedSet::Visited(this))
					return;

				value.ptr = nullptr;
				mValueCached = false;

				if (bForceRecompute && mpExp)
				{
					Forward(mpExp, bForceRecompute);
				}

				GlobalVisitedSet::SetVisited(this);
			}

			virtual void Update(const float scale = 1.0f) override;

			TENSOR_INLINE void Set(const int idx, const TensorParams& broadcastIndex, const T val)
			{
				Shape selfIndex;
				selfIndex.Resize(Dim());

				Shape index = broadcastIndex.Index(idx);
				for (int j = 0; j < Dim(); j++)
				{
					selfIndex[j] = index[j + broadcastIndex.GetShape().Size() - Dim()];
					if (selfIndex[j] >= GetShape(j))
						selfIndex[j] = 0;
				}

				this->operator[](selfIndex) = val;
			}

			Tensor(NestedInitializerList<T, 1> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);

				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<T, 2> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);

				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<T, 3> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);

				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<T, 4> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);

				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<T, 5> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);

				SetRequiresGrad(requiresGrad);
			}

			Tensor& operator = (NestedInitializerList<T, 1> initList)
			{
				NestedInitializerListHelper<1>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 2> initList)
			{
				NestedInitializerListHelper<2>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 3> initList)
			{
				NestedInitializerListHelper<3>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 4> initList)
			{
				NestedInitializerListHelper<4>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<T, 5> initList)
			{
				NestedInitializerListHelper<5>(initList);

				return *this;
			}

			template<int level>
			void NestedInitializerListHelper(NestedInitializerList<T, level> initList)
			{
				Resize(DeriveShapeFromNestedInitList<Shape>(initList));

				{
					T* pTempData = new T[LinearSize()];
					T* pIter = pTempData;
					InitListNestedCopy(pIter, initList);

					cudaMemcpy(Data(), pTempData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
					if (HOST_DEBUG)
						memcpy(mpHostData.get(), pTempData, LinearSize() * sizeof(T));

					Memory::SafeDeleteArray(pTempData);
				}
			}

			Tensor(NestedInitializerList<Vec<2, T>, 1> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<2, T>, 2> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<2, T>, 3> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<2, T>, 4> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor& operator = (NestedInitializerList<Vec<2, T>, 1> initList)
			{
				NestedInitializerListHelper_Vector<2, 1>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<2, T>, 2> initList)
			{
				NestedInitializerListHelper_Vector<2, 2>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<2, T>, 3> initList)
			{
				NestedInitializerListHelper_Vector<2, 3>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<2, T>, 4> initList)
			{
				NestedInitializerListHelper_Vector<2, 4>(initList);

				return *this;
			}


			Tensor(NestedInitializerList<Vec<3, T>, 1> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<3, T>, 2> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<3, T>, 3> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<3, T>, 4> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor& operator = (NestedInitializerList<Vec<3, T>, 1> initList)
			{
				NestedInitializerListHelper_Vector<3, 1>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<3, T>, 2> initList)
			{
				NestedInitializerListHelper_Vector<3, 2>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<3, T>, 3> initList)
			{
				NestedInitializerListHelper_Vector<3, 3>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<3, T>, 4> initList)
			{
				NestedInitializerListHelper_Vector<3, 4>(initList);

				return *this;
			}


			Tensor(NestedInitializerList<Vec<4, T>, 1> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<4, T>, 2> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<4, T>, 3> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor(NestedInitializerList<Vec<4, T>, 4> initList, const bool requiresGrad = false)
			{
				this->operator=(initList);
				SetRequiresGrad(requiresGrad);
			}

			Tensor& operator = (NestedInitializerList<Vec<4, T>, 1> initList)
			{
				NestedInitializerListHelper_Vector<4, 1>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<4, T>, 2> initList)
			{
				NestedInitializerListHelper_Vector<4, 2>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<4, T>, 3> initList)
			{
				NestedInitializerListHelper_Vector<4, 3>(initList);

				return *this;
			}

			Tensor& operator = (NestedInitializerList<Vec<4, T>, 4> initList)
			{
				NestedInitializerListHelper_Vector<4, 4>(initList);

				return *this;
			}

			template<int N, int level>
			void NestedInitializerListHelper_Vector(NestedInitializerList<Vec<N, T>, level> initList)
			{
				Shape shape = DeriveShapeFromNestedInitList<Shape>(initList);
				shape.SetVectorType(N);
				Resize(shape);

				{
					Vec<N, T>* pTempData = new Vec<N, T>[NumElements()];
					Vec<N, T>* pIter = pTempData;
					InitListNestedCopy(pIter, initList);

					T* pTransposedData = new T[LinearSize()];
					for (int i = 0; i < NumElements(); i++)
					{
						for (int v = 0; v < N; v++)
							pTransposedData[i + NumElements() * v] = pTempData[i][v];
					}

					cudaMemcpy(Data(), pTransposedData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
					if (HOST_DEBUG)
						memcpy(mpHostData.get(), pTransposedData, LinearSize() * sizeof(T));

					Memory::SafeDeleteArray(pTempData);
					Memory::SafeDeleteArray(pTransposedData);
				}
			}

			Type GetType() const
			{
				return DeriveType<T>();
			}

			template<typename... TShape>
			void Resize(TShape... shape)
			{
				Resize({ shape... });
			}

			void Resize(const Shape& shape)
			{
				int newLinSize = shape.LinearSize();
				if (LinearSize() != newLinSize)
				{
					Free();
					mParams.Resize(shape);

					{
						T* pAlloc;
						if (CuAllocator::GetHandle().DeviceAllocate((void**)&pAlloc, mParams.LinearSize() * sizeof(T)) != cudaSuccess)
						{
							AssertNoEntry();
						}

						mpData = { pAlloc, Deleter<T>() };
						if (HOST_DEBUG)
							mpHostData = { Memory::AlignedAlloc<T>(mParams.LinearSize()), DeleterHost<T>() };
					}

					Assert(mpData.get());

					if (mbRequiresGrad)
					{
						if (!mpGrad)
						{
							mpGrad = make_shared<Expr>();
						}
						*mpGrad = Zeros(1);
					}

					Clear();
				}
				else if (GetShape() != shape)
				{
					mParams.Reshape(shape);
				}
			}

			void Assign(const T* pData, const Shape& shape)
			{
				Resize(shape);

				{
					cudaMemcpy(mpData.get(), pData, LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
					if (HOST_DEBUG)
						memcpy(mpHostData.get(), pData, LinearSize() * sizeof(T));
				}
			}

			void ValueToTorch(ptr_wrapper<T> torchPtr)
			{
				T* hostData = HostData();
				std::copy(hostData, hostData + LinearSize(), torchPtr.get());
			}

			void GradToTorch(ptr_wrapper<T> torchPtr)
			{
				Tensor<T> gradTensor = Grad();
				T* gradData = gradTensor.HostData();
				std::copy(gradData, gradData + gradTensor.LinearSize(), torchPtr.get());
			}

			Tensor GetTransposed(const Shape& transposeDim = {}) const
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mpData = mpData;
				ret.mpHostData = mpHostData;
				ret.mpGrad = mpGrad;
				ret.mbRequiresGrad = mbRequiresGrad;
				ret.mpExp = mpExp;

				ret.mParams.Transpose(transposeDim);

				return ret;
			}

			template<typename... TShape>
			Tensor Reshape(TShape... shape)
			{
				static_assert(AllIntegralType<TShape...>::Value, "All parameters have to be integral type.");

				return Reshape({ shape... });
			}

			Tensor Reshape(const Shape& shape)
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mParams.Reshape(shape);
				ret.mpData = mpData;
				ret.mpHostData = mpHostData;
				ret.mpGrad = mpGrad;
				ret.mbRequiresGrad = mbRequiresGrad;
				ret.mpExp = mpExp;
				return ret;
			}

			template<typename... TShape>
			const Tensor Reshape(TShape... shape) const
			{
				static_assert(AllIntegralType<TShape...>::Value, "All parameters have to be integral type.");

				return Reshape({ shape... });
			}

			const Tensor Reshape(const Shape& shape) const
			{
				Tensor ret;
				ret.mParams = mParams;
				ret.mParams.Reshape(shape);
				ret.mpData = mpData;
				ret.mpHostData = mpHostData;
				ret.mpGrad = mpGrad;
				ret.mbRequiresGrad = mbRequiresGrad;
				ret.mpExp = mpExp;
				return ret;
			}

			inline bool IsTransposed() const
			{
				return mParams.IsTransposed();
			}

			inline void CopyToHost()
			{
				if (HOST_DEBUG)
				{
					cudaMemcpy(mpHostData.get(), mpData.get(), mParams.LinearSize() * sizeof(T), cudaMemcpyDeviceToHost);
				}
			}

			inline void Clear()
			{
				if (mpData)
					cudaMemset(mpData.get(), 0, mParams.LinearSize() * sizeof(T));

				if (HOST_DEBUG)
				{
					if (mpHostData)
						Memory::SafeClear(mpHostData.get(), mParams.LinearSize());
				}
				
				ClearGrad();

				mpExp.ptr = nullptr;
			}

			inline void ClearGrad()
			{
				if (mpGrad)
					*mpGrad = Zeros(1);
			}

			inline Tensor Clone() const
			{
				Tensor cloned;

				cloned.mParams = mParams;
				cloned.mpExp = mpExp;
				cloned.mpGrad = mpGrad;
				cloned.mbRequiresGrad = mbRequiresGrad;

				cudaMemcpy(cloned.Data(), Data(), LinearSize() * sizeof(T), cudaMemcpyDeviceToDevice);
				if (HOST_DEBUG)
					memcpy(cloned.HostData(), HostData(), LinearSize() * sizeof(T));

				return cloned;
			}

			template<typename... Index>
			TENSOR_INLINE int LinearIndex(Index... idx) const
			{
				return mParams.LinearIndex(idx...);
			}
			TENSOR_INLINE int LinearIndex(const Shape& idx) const
			{
				return mParams.LinearIndex(idx);
			}
			TENSOR_INLINE virtual Shape Index(int linearIdx) const
			{
				return mParams.Index(linearIdx);
			}
			TENSOR_INLINE int LinearSize() const
			{
				return mParams.LinearSize();
			}
			TENSOR_INLINE int VectorLinearSize() const
			{
				if (VectorSize() > 0)
					return mParams.LinearSize() / VectorSize();
				else
					return mParams.LinearSize();
			}
			TENSOR_INLINE int NumElements() const
			{
				return mParams.NumElements();
			}
			TENSOR_INLINE bool IndexRangeCheck(const Shape& index) const
			{
				return mParams.IndexRangeCheck(index);
			}
			TENSOR_INLINE void IterateIndex(Shape& index) const
			{
				return mParams.IterateIndex(index);
			}
			TENSOR_INLINE bool IterateIndex(Shape& index, const Shape& axes) const
			{
				return mParams.IterateIndex(index, axes);
			}
			TENSOR_INLINE int GetShape(int iDim) const
			{
				return mParams.GetShape(iDim);
			}
			TENSOR_INLINE Shape GetShape() const
			{
				return mParams.GetShape();
			}
			TENSOR_INLINE int Dim() const
			{
				return GetShape().Size();
			}
			TENSOR_INLINE int Stride(int iDim) const
			{
				return mParams.Stride(iDim);
			}
			TENSOR_INLINE int VectorSize() const
			{
				return mParams.VectorSize();
			}
			TENSOR_INLINE bool Empty() const
			{
				return LinearSize() == 0;
			}

			void Set(const Shape& idx, const T val)
			{
				cudaMemcpy(&(mpData.get()[LinearIndex(idx)]), &val, sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					mpHostData.get()[LinearIndex(idx)] = val;
				}
			}
			const T Get(const Shape& idx) const
			{
				T ret;
				cudaMemcpy(&ret, &(mpData.get()[LinearIndex(idx)]), sizeof(T), cudaMemcpyDeviceToHost);
				return ret;
			}
			void Set(const int idx, const T val)
			{
				Assert(idx < mParams.LinearSize());

				cudaMemcpy(&(mpData.get()[idx]), &val, sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					mpHostData.get()[idx] = val;
				}
			}
			const T Get(const int idx) const
			{
				Assert(idx < mParams.LinearSize());

				T ret;
				cudaMemcpy(&ret, &(mpData.get()[idx]), sizeof(T), cudaMemcpyDeviceToHost);
				return ret;
			}

			TENSOR_INLINE const T* Data() const
			{
				return mpData.get();
			}
			TENSOR_INLINE T* Data()
			{
				return mpData.get();
			}


			TENSOR_INLINE const T* HostData() const
			{
				return mpHostData.get();
			}
			TENSOR_INLINE T* HostData()
			{
				if (!mpHostData)
				{
					mpHostData = { Memory::AlignedAlloc<T>(mParams.LinearSize()), DeleterHost<T>() };
					cudaMemcpy(mpHostData.get(), mpData.get(), mParams.LinearSize() * sizeof(T), cudaMemcpyDeviceToHost);
				}
				return mpHostData.get();
			}

			const Expr Grad() const
			{
				return *mpGrad;
			}
			Expr Grad()
			{
				return *mpGrad;
			}

			void Free()
			{
				mpData = nullptr;
				mpHostData = nullptr;
				mpGrad = nullptr;
				mpExp.ptr = nullptr;

				mParams = TensorParams();
			}

			TensorJit<T> ToJit() const
			{
				TensorJit<T> ret;

				ret.mpData = mpData.get();
				ret.mParams = mParams.ToJit();

				return ret;
			}

			TensorJitArg ToJitArg() const
			{
				TensorJitArg ret;

				ret.mpData = mpData.get();
				ret.mParams = mParams.ToJit();
				if (std::is_same<T, float>::value)
				{
					ret.mType = 0;
				}
				else if (std::is_same<T, int>::value)
				{
					ret.mType = 1;
				}
				else if (std::is_same<T, uint>::value)
				{
					ret.mType = 2;
				}
				else if (std::is_same<T, bool>::value)
				{
					ret.mType = 3;
				}

				return ret;
			}


			friend std::ostream& operator << (std::ostream& stream, Tensor& A)
			{
				const auto numSamples = A.LinearSize();
				vector<T> arr;
				arr.resize(numSamples);
				cudaMemcpy(arr.data(), A.Data(), numSamples * sizeof(T), cudaMemcpyDeviceToHost);

				for (auto it : arr)
					stream << it << " ";

				return stream;
			}
		public:
			// ----------------------------------------------------------------
			// Common utilities for tensors
			// ----------------------------------------------------------------
			static Tensor<T> LinSpace(const T& start, const T& stop, const int& numSamples, const bool requiresGrad = false)
			{
				Tensor<T> ret;
				ret.Resize(numSamples);
				ret.SetRequiresGrad(requiresGrad);

				T step = (stop - start) / T(numSamples - 1);

				vector<T> arr;
				arr.resize(numSamples);

				for (int i = 0; i < numSamples; i++)
					arr[i] = start + i * step;

				cudaMemcpy(ret.Data(), arr.data(), numSamples * sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					memcpy(ret.HostData(), arr.data(), numSamples * sizeof(T));
				}

				return ret;
			}

			static Tensor<T> ArrayRange(const T& start, const T& stop, const T& step, const bool requiresGrad = false)
			{
				Tensor<T> ret;

				if (stop <= start)
					return ret;

				int numSamples = (stop - start) / step;
				ret.Resize(numSamples);
				ret.SetRequiresGrad(requiresGrad);

				vector<T> arr;
				arr.resize(numSamples);

				for (int i = 0; i < numSamples; i++)
					arr[i] = start + i * step;

				cudaMemcpy(ret.Data(), arr.data(), numSamples * sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					memcpy(ret.HostData(), arr.data(), numSamples * sizeof(T));
				}

				return ret;
			}

			static Tensor<T> ArrayRange(const T& stop, const bool requiresGrad = false)
			{
				return ArrayRange(0, stop, 1, requiresGrad);
			}

			static Tensor<T> Identity(const int N)
			{
				Tensor<T> ret;
				ret.Resize(N, N);

				vector<T> arr;
				arr.resize(N * N);

				for (int i = 0; i < N; i++)
					arr[ret.LinearIndex(i, i)] = T(1);

				cudaMemcpy(ret.Data(), arr.data(), N * N * sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					memcpy(ret.HostData(), arr.data(), N * N * sizeof(T));
				}

				return ret;
			}

			static Tensor<T> Normalize(const Tensor<T>& inTensor)
			{
				const Tensor<T> tensorSqr = inTensor * inTensor;
				Tensor<T> denorm = Tensor<T>::Sqrt(Tensor<T>::Sum(tensorSqr));

				return inTensor / denorm;
			}

			static Tensor<T> Dot(const Tensor<T>& lhs, const Tensor<T>& rhs)
			{
				const Shape& leftShape = lhs.GetShape().LinearizeVector();
				const Shape& rightShape = rhs.GetShape().LinearizeVector();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				Tensor<T> ret;
				Shape retShape = { leftShape[0], rightShape[1] };
				if (lhs.GetShape().VectorSize() > 1 || rhs.GetShape().VectorSize() > 1)
					retShape = retShape.Vectorize();

				ret.Resize(retShape);
				DotInplace(lhs, rhs, &ret);

				return ret;
			}

			static void DotInplace(const Tensor<T>& lhs, const Tensor<T>& rhs, const bool leftTransposed, const bool rightTransposed, Tensor<T>* pResult, const float alpha = 1.0f, const float beta = 0.0f)
			{
				const Shape& leftShape = leftTransposed ? lhs.GetShape().LinearizeVector().Transpose() : lhs.GetShape().LinearizeVector();
				const Shape& rightShape = rightTransposed ? rhs.GetShape().LinearizeVector().Transpose() : rhs.GetShape().LinearizeVector();

				Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
				Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

				Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

				cublasSgemm(Cublas::GetHandle(),
					rightTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
					leftTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
					rightShape[1],
					leftShape[0],
					leftShape[1],
					&alpha,
					rhs.Data(),
					rightTransposed ? rightShape[0] : rightShape[1],
					lhs.Data(),
					leftTransposed ? leftShape[0] : leftShape[1],
					&beta,
					pResult->Data(),
					pResult->GetShape().LinearizeVector()[1]);

				pResult->CopyToHost();
			}

			static Tensor<T> Transpose(const Tensor<T>& inTensor, const Shape& transposeDim = {})
			{
				const Shape& inShape = inTensor.GetShape().LinearizeVector();

				Shape transposedShape = inShape;
				if (transposeDim.Empty())
				{
					Shape shapeCopy = transposedShape;
					for (int i = 0; i < transposedShape.Size(); i++)
					{
						transposedShape[i] = shapeCopy[transposedShape.Size() - 1 - i];
					}
				}
				else
				{
					for (int i = 0; i < transposeDim.Size(); i++)
					{
						transposedShape[i] = inShape[transposeDim[i]];
					}
				}

				Tensor<T> ret;
				ret.Resize(transposedShape);

				Shape index;
				index.ResizeZeroed(inTensor.Dim());

				const float alpha = 1.0f;
				const float beta = 0.0f;
				cublasSgeam(Cublas::GetHandle(),
					CUBLAS_OP_T,
					CUBLAS_OP_T,
					inShape[0],
					inShape[1],
					&alpha,
					inTensor.Data(),
					inTensor.IsTransposed() ? inShape[0] : inShape[1],
					&beta,
					inTensor.Data(),
					inTensor.IsTransposed() ? inShape[0] : inShape[1],
					ret.Data(),
					ret.GetShape(1));

				ret.CopyToHost();

				return ret;
			}

			static Tensor<T> Inverse(const Tensor<T>& inTensor)
			{
				Assertf(inTensor.Dim() == 2 && inTensor.GetShape(0) == inTensor.GetShape(1),
					"Matrix inversion dimension mismatch.");

				const int N = inTensor.GetShape(0);

				Tensor<T> ret = inTensor;

				// TODO: Add CUDA kernel for matrix inversion
				AssertNoEntry();

				return ret;
			}

			static Tensor<T> Sum(const Tensor<T>& inTensor, const Shape& axes = { -2 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axes, keepDim, Algorithm::Plus<>(), T(0));
			}

			static Tensor<T> Product(const Tensor<T>& inTensor, const Shape& axes = { -2 })
			{
				return ProjectionOp(inTensor, axes, keepDim, Algorithm::Multiply<>(), T(1));
			}

			static Tensor<T> Max(const Tensor<T>& inTensor, const Shape& axes = { -2 }, const bool keepDim = false)
			{
				return ProjectionOp(inTensor, axes, keepDim, Algorithm::Max<>(), T(Math::EDX_NEG_INFINITY));
			}


			static void SumInplace(const Tensor<T>& inTensor, Tensor<T>* pResult, const Shape& axes = { -2 }, const bool keepDim = false)
			{
				ProjectionOpInplace(inTensor, axes, keepDim, Algorithm::Plus<>(), T(0), pResult);
			}

			static void ProductInplace(const Tensor<T>& inTensor, Tensor<T>* pResult, const Shape& axes = { -2 })
			{
				ProjectionOpInplace(inTensor, axes, keepDim, Algorithm::Multiply<>(), T(1), pResult);
			}

			static void MaxInplace(const Tensor<T>& inTensor, Tensor<T>* pResult, const Shape& axes = { -2 }, const bool keepDim = false)
			{
				ProjectionOpInplace(inTensor, axes, keepDim, Algorithm::Max<>(), T(Math::EDX_NEG_INFINITY), pResult);
			}


			static Tensor<T> ExclusiveScan(const Tensor<T>& val);
			static Tensor<T> InclusiveScan(const Tensor<T>& val);
			static void MaskedSelectionIndex(const int maskSize, const Tensor<int>& mask, const Tensor<int>& offset, const int maskSum, Tensor<int>* pResult);
			static void IndexedWrite(Tensor<T>& dst, const Tensor<T>& val, const Tensor<int>& indices, const int axis = 0);

			static Tensor<T> Mean(const Tensor<T>& X, const Shape& axes = { -2 }, const bool keepDim = false)
			{
				Tensor<T> ret = Sum(X, axes);

				float invDivisor = ret.LinearSize() / float(X.LinearSize());
				ret *= invDivisor;

				return ret;
			}

			//static Tensor<T> StandardDeviation(const Tensor<T>& X, const Shape& axes = { -1 }, const bool keepDim = false)
			//{
			//	Tensor<T> mean = Mean(X, axes, keepDim);
			//	Tensor<T> centeredX = X - mean;

			//	Tensor<T> variance = Tensorf::Mean(centeredX * centeredX, axes, keepDim);

			//	return Sqrt(variance + Scalar(1e-5f));
			//}

			template<typename... TShape>
			static Tensor<T> RandomInt(const int high, TShape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				RandomGen random;
				vector<T> arr;
				arr.Resize(ret.LinearSize());
				for (auto& it : arr)
				{
					it = random.UnsignedInt() % high;
				}

				cudaMemcpy(ret.Data(), arr.data(), ret.LinearSize() * sizeof(T), cudaMemcpyHostToDevice);
				if (HOST_DEBUG)
				{
					memcpy(ret.HostData(), arr.data(), ret.LinearSize() * sizeof(T));
				}

				return ret;
			}

			template<typename... TShape>
			static Tensor<T> RandomFloat(TShape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				curandGenerateUniform(Curand::GetHandle(), ret.Data(), ret.LinearSize());
				ret.CopyToHost();

				return ret;
			}

			template<typename... TShape>
			static Tensor<T> RandomNormalDistribution(const float std, TShape... shape)
			{
				Tensor<T> ret;
				ret.Resize(shape...);

				curandGenerateNormal(Curand::GetHandle(), ret.Data(), ret.LinearSize(), 0.0f, std);
				ret.CopyToHost();

				return ret;
			}

			inline void operator += (const Expr& rhs)
			{
				ElementWiseBinaryOpInplaceExpr(*this, rhs.ptr, Algorithm::Plus<>());
			}
			inline void operator -= (const Expr& rhs)
			{
				ElementWiseBinaryOpInplaceExpr(*this, rhs.ptr, Algorithm::Substract<>());
			}
			inline void operator *= (const Expr& rhs)
			{
				ElementWiseBinaryOpInplaceExpr(*this, rhs.ptr, Algorithm::Multiply<>());
			}
			inline void operator /= (const Expr& rhs)
			{
				ElementWiseBinaryOpInplaceExpr(*this, rhs.ptr, Algorithm::Divide<>());
			}

		public:
			template<typename Op>
			static void ElementWiseBinaryOpInplaceExpr(Tensor<T>& lhs, const Expr& rhs, Op op);

			static Shape ProjectionShape(const Shape& inShape, const Shape& axes, const bool keepDim)
			{
				if (axes.Size() == 0)
					return inShape;

				if (axes.Size() == 1 && axes[0] == -2)
				{
					return { 1 };
				}

				Shape projShape;
				for (int i = 0; i < inShape.Size(); i++)
				{
					if (!keepDim)
					{
						if (!axes.Contains(i))
							projShape.Add(inShape[i]);
					}
					else
					{
						if (axes.Contains(i))
							projShape.Add(1);
						else
							projShape.Add(inShape[i]);
					}
				}

				if (projShape.Empty())
					projShape.Add(1);

				if (axes.Contains(-1))
					projShape.SetVectorType(VecType::Scalar1);
				else
					projShape.SetVectorType(inShape.VectorSize());

				return projShape;
			}

			template<typename Op>
			static Tensor<T> ProjectionOp(const Tensor<T>& lhs, const Shape& axes, const bool keepDim, Op op, T initVal);

			template<typename Op>
			static void ProjectionOpInplace(const Tensor<T>& lhs, const Shape& axes, const bool keepDim, Op op, T initVal, Tensor<T>* pResult);

			static Tensor<T> Unbroadcast(const Tensor<T>& tensor, const Shape& target)
			{
				Shape shape = tensor.GetShape();
				if (shape == target)
					return tensor;

				Shape axes;

				if (target.LinearSize() == 1)
				{
					axes.Add(-1);
				}
				else
				{
					for (int i = 0; i < shape.Size(); i++)
					{
						if (shape[i] > target[i])
							axes.Add(i);
					}
				}

				Tensor<T> ret = Tensor<T>::Sum(tensor, axes, true);
				ret.Reshape(target);

				return ret;
			}

	private:

			// --------------------------------------------------------------------------------------------
			// DO NOT USE DIRECTLY
			// STL-like iterators to enable range-based for loop support.
			// --------------------------------------------------------------------------------------------
			inline friend		T*	begin(Tensor<T>& tensor) { return tensor.Data(); }
			inline friend const	T*	begin(const Tensor<T>& tensor) { return tensor.Data(); }
			inline friend		T*	end(Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
			inline friend const	T*	end(const Tensor<T>& tensor) { return tensor.Data() + tensor.LinearSize(); }
		};

		using Tensorf = Tensor<float>;
		using Tensord = Tensor<double>;
		using Tensori = Tensor<int>;
		using Tensorui = Tensor<uint>;
		using Tensorb = Tensor<bool>;

		template<typename T>
		inline Tensor<T> operator + (const Tensor<T>& lhs, const Tensor<T>& rhs)
		{
			Tensor<T> ret;
			ret = Expr(lhs) + Expr(rhs);
			return ret;
		}

		template<typename T>
		inline Tensor<T> operator * (const Tensor<T>& lhs, const Tensor<T>& rhs)
		{
			Tensor<T> ret;
			ret = Expr(lhs) * Expr(rhs);
			return ret;
		}

		template<typename T>
		inline Tensor<T> operator - (const Tensor<T>& lhs, const Tensor<T>& rhs)
		{
			Tensor<T> ret;
			ret = Expr(lhs) - Expr(rhs);
			return ret;
		}

		template<typename T>
		inline Tensor<T> operator / (const Tensor<T>& lhs, const Tensor<T>& rhs)
		{
			Tensor<T> ret;
			ret = Expr(lhs) / Expr(rhs);
			return ret;
		}

		template<typename T>
		inline Tensor<T> operator - (const Tensor<T>& param)
		{
			Tensor<T> ret;
			ret = -Expr(param);
			return ret;
		}
		
		#include "TemplateExpression.h"

		static Tensorf NumericalGradientEval(const Expr& exp, Tensorf& x, const float step = 1e-3f)
		{
			Tensorf gradient;
			gradient.Resize(x.GetShape());


			for (int i = 0; i < x.LinearSize(); i++)
			{
				float originalVal = x.Get(i);

				x.Set(i, originalVal + step);
				Tensorf positive;
				GlobalVisitedSet::Clear();
				positive.Forward(exp.ptr, true);

				x.Set(i, originalVal - step);
				Tensorf negative;
				GlobalVisitedSet::Clear();
				negative.Forward(exp.ptr, true);

				x.Set(i, originalVal);

				gradient.Set(i, Tensorf::Sum((positive - negative)).Get(0) / (2.0f * step));
			}

			// Clear the visited set
			GlobalVisitedSet::Clear();

			return gradient;
		}

		static void ValidateBackwardDiff(const Tensorf& val, Tensorf& param)
		{
			// Backpropagation
			// This call automatically backprops across two statically compiled expressions all the way to the tensor A
			// In this case one cuda kernel will get called to for each leaf node in the expression that requires gradient
			val.Backward(Ones(val.GetShape()));

			Tensorf diff = param.Grad();
			std::cout << "autodiff derivative: " << diff << "\n";

			Tensorf numericalDiff = NumericalGradientEval(val, param);
			std::cout << "numerical derivative: " << numericalDiff << "\n";

			for (int i = 0; i < diff.LinearSize(); i++)
			{
				float v1 = diff.Get(i);
				float v2 = numericalDiff.Get(i);
				float d = Math::Abs(v1 - v2);
				float r = Math::Max(Math::Abs(v1), Math::Abs(v2));

				Assertf(d / Math::Max(r, 1e-6f) < 1e-3f || d < 1e-3f, "Backward diff gradient mismatch!");
			}

			std::cout << '\n';

			return;
		}

		static void ValidateForwardDiff(const Tensorf& val, Tensorf& param)
		{
			std::vector<Tensorf> forwardGrad;

			int paramNumEl = param.LinearSize();
			for (int i = 0; i < paramNumEl; i++)
			{
				forwardGrad.push_back(Tensorf(Zeros(val.GetShape())));
			}

			int numElements = val.LinearSize();
			for (int i = 0; i < numElements; i++)
			{
				param.ClearGrad();		// Clear Grad

				Tensorf mask = Zeros(val.GetShape());
				mask.Set(i, 1);
				val.Backward(mask);
				const Tensorf& grad = param.Grad();
				for (int j = 0; j < paramNumEl; j++)
					forwardGrad[j].Set(i, grad.Get(j));
			}

			std::vector<Tensorf> forwardGrad2;
			for (int i = 0; i < paramNumEl; i++)
			{
				ForwardDiffVariableCache::GetHandle().clear();
				Tensorf diff_i = val.ForwardDiff(param.Data(), i);
				cout << "Forward Diff Gradients:  " << diff_i << '\n';
				cout << "Backward Diff Gradients: " << forwardGrad[i] << '\n';

				for (int j = 0; j < numElements; j++)
				{
					Assertf(Math::Abs(diff_i.Get(j) - forwardGrad[i].Get(j)) < 1e-3f, "Forward diff gradient mismatch!");
				}
			}

			std::cout << '\n';

			return;
		}

		Expr SafeSqrt(const Expr& x);
		Expr Clamp(const Expr& x, const Expr& val_min, const Expr& val_max);
		Expr Lerp(const Expr& a, const Expr& b, const Expr& ratio);
		Expr VectorNormalize(const Expr& x);
		Expr VectorLength(const Expr& x);
		Expr VectorSquaredLength(const Expr& x);
		Expr Mean(const Expr& x, const Shape& axes = { -2 }, const bool keepDim = false);
		Expr Variance(const Expr& x, const Shape& axes = { -2 }, const bool keepDim = false);
		Expr StandardDeviation(const Expr& x, const Shape& axes = { -2 }, const bool keepDim = false);
		Expr TransformPointsHomogeneous(const Expr& pt, const Expr& mat);
		Expr TransformPoints(const Expr& pt, const Expr& mat);
		Expr TransformVectors(const Expr& vec, const Expr& mat);
		Expr TransformNormals(const Expr& vec, const Expr& matInv);
		Expr Luminance(const Expr& val);
		Expr VectorDot(const Expr& vec1, const Expr& vec2);
		Expr VectorCross(const Expr& vec1, const Expr& vec2);
		Expr VectorReflect(const Expr& in, const Expr& norm);
	}
}