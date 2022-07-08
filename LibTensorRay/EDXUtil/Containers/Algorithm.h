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

#include <initializer_list>
#include <Math/EDXMath.h>


namespace EDX
{
	namespace Algorithm
	{
		template <typename T, typename PredType>
		bool IsSorted(const T* Range, int32 RangeSize, PredType Pred)
		{
			if (RangeSize == 0)
			{
				return true;
			}

			// When comparing N elements, we do N-1 comparisons
			--RangeSize;

			const T* Next = Range + 1;
			for (;;)
			{
				if (RangeSize == 0)
				{
					return true;
				}

				if (Pred(*Next, *Range))
				{
					return false;
				}

				++Range;
				++Next;
				--RangeSize;
			}
		}

		template <typename T>
		struct Less
		{
			EDX_INLINE bool operator()(const T& Lhs, const T& Rhs) const
			{
				return Lhs < Rhs;
			}
		};

		template <typename T>
		struct LessEQ
		{
			EDX_INLINE bool operator()(const T& Lhs, const T& Rhs) const
			{
				return Lhs <= Rhs;
			}
		};

		template <typename T>
		__forceinline void Reverse(T* Array, int32 ArraySize)
		{
			for (int32 i = 0, i2 = ArraySize - 1; i < ArraySize / 2 /*rounding down*/; ++i, --i2)
			{
				Swap(Array[i], Array[i2]);
			}
		}

		template <typename T, int32 ArraySize>
		__forceinline bool IsSorted(const T(&Array)[ArraySize])
		{
			return IsSorted((const T*)Array, ArraySize, Less<T>());
		}

		template <typename T, int32 ArraySize, typename PredType>
		__forceinline bool IsSorted(const T(&Array)[ArraySize], PredType Pred)
		{
			return IsSorted((const T*)Array, ArraySize, Pred);
		}

		template <typename T>
		__forceinline bool IsSorted(const T* Array, int32 ArraySize)
		{
			return IsSorted(Array, ArraySize, Less<T>());
		}

		template <typename ContainerType>
		__forceinline bool IsSorted(const ContainerType& Container)
		{
			return IsSorted(Container.Data(), Container.Size(), Less<typename ContainerType::ElementType>());
		}

		template <typename ContainerType, typename PredType>
		__forceinline bool IsSorted(const ContainerType& Container, PredType Pred)
		{
			return IsSorted(Container.Data(), Container.Size(), Pred);
		}

		template<class T, class UnaryPredicate>
		int32 Partition(T* Elements, const int32 Num, const UnaryPredicate& Predicate)
		{
			T* First = Elements;
			T* Last = Elements + Num;

			while (First != Last)
			{
				while (Predicate(*First))
				{
					++First;
					if (First == Last)
					{
						return First - Elements;
					}
				}

				do
				{
					--Last;
					if (First == Last)
					{
						return First - Elements;
					}
				} while (!Predicate(*Last));

				Exchange(*First, *Last);
				++First;
			}

			return First - Elements;
		}

		template<class T, class BinaryPredicate>
		EDX_INLINE int32 LowerBound(const T* Elements, const int32 Num, const T& Val, const BinaryPredicate& Predicate)
		{
			const T* First = Elements;
			const T* Last = Elements + Num;

			// Find first element not before _Val, using _Pred
			int32 Count = Num;

			while (Count > 0)
			{
				// Divide and conquer, find half that contains answer
				int32 Count2 = Count / 2;
				const T* Mid = First;
				Mid += Count2;

				if (Predicate(*Mid, Val))
				{
					// Try top half
					First = ++Mid;
					Count -= Count2 + 1;
				}
				else
					Count = Count2;
			}

			return First - Elements;
		}

		template<class T>
		EDX_INLINE int32 LowerBound(const T* Elements, const int32 Num, const T& Val)
		{
			return LowerBound(Elements, Num, Val, Less<T>());
		}

		template<class T, class BinaryPredicate>
		EDX_INLINE int32 UpperBound(const T* Elements, const int32 Num, const T& Val, const BinaryPredicate& Predicate)
		{
			const T* First = Elements;
			//const T* Last = Elements + Num;

			// Find first element not before _Val, using _Pred
			int32 Count = Num;

			while (Count > 0)
			{
				// Divide and conquer, find half that contains answer
				int32 Count2 = Count / 2;
				const T* Mid = First;
				Mid += Count2;

				if (!Predicate(Val, *Mid))
				{
					// Try top half
					First = ++Mid;
					Count -= Count2 + 1;
				}
				else
					Count = Count2;
			}

			return First - Elements;
		}

		template<class T>
		EDX_INLINE int32 UpperBound(const T* Elements, const int32 Num, const T& Val)
		{
			return UpperBound(Elements, Num, Val, Less<T>());
		}

		/**
		* Reverses a range
		*
		* @param  Array  The array to reverse.
		*/
		template <typename T, int32 ArraySize>
		__forceinline void Reverse(T(&Array)[ArraySize])
		{
			return Reverse((T*)Array, ArraySize);
		}

		/**
		* Reverses a range
		*
		* @param  Container  The container to reverse
		*/
		template <typename ContainerType>
		__forceinline void Reverse(ContainerType& Container)
		{
			return Reverse(Container.Data(), Container.Size());
		}

		// Plus<T> specifically takes const T& and returns T.
		// Plus<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Plus
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A + B; }
		};

		template<>
		struct Plus<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A + B) { return A + B; }
		};

		// Substract<T> specifically takes const T& and returns T.
		// Substract<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A-B)
		template<typename T = void>
		struct Substract
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A - B; }
		};

		template<>
		struct Substract<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A - B) { return A - B; }
		};

		// Multiply<T> specifically takes const T& and returns T.
		// Multiply<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Multiply
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A * B; }
		};

		template<>
		struct Multiply<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A * B) { return A * B; }
		};

		// Divide<T> specifically takes const T& and returns T.
		// Divide<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Divide
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A / B; }
		};

		template<>
		struct Divide<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A / B) { return A / B; }
		};

		// Min<T> specifically takes const T& and returns T.
		// Min<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Min
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A < B ? A : B; }
		};

		template<>
		struct Min<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A + B) { return A < B ? A : B; }
		};
		
		// Max<T> specifically takes const T& and returns T.
		// Max<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Max
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return A > B ? A : B; }
		};

		template<>
		struct Max<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A + B) { return A > B ? A : B; }
		};

		// Pow<T> specifically takes const T& and returns T.
		// Pow<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Pow
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return Math::Pow(A, B); }
		};

		template<>
		struct Pow<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A * B) { return Math::Pow(A, B); }
		};

		// Exp<T> specifically takes const T& and returns T.
		// Exp<> (empty angle brackets) is late-binding, taking whatever is passed and returning the correct result type for (A+B)
		template<typename T = void>
		struct Exp
		{
			EDX_INLINE T operator()(const T& A, const T& B) { return Math::Pow(A, B); }
		};

		template<>
		struct Exp<void>
		{
			template<typename U, typename V>
			EDX_INLINE auto operator()(U&& A, V&& B) -> decltype(A * B) { return Math::Pow(A, B); }
		};

		/**
		* Sums a range by successively applying Op.
		*
		* @param  Input  Any iterable type
		* @param  Init  Initial value for the summation
		* @param  Op  Summing Operation (the default is Plus<>)
		*
		* @return the result of summing all the elements of Input
		*/
		template <typename T, typename A, typename OpT>
		__forceinline T Accumulate(const A& Input, T init, OpT Op)
		{
			T result = init;
			for (auto&& i : Input)
			{
				result = Op(result, i);
			}
			return result;
		}

		template <typename T, typename A>
		__forceinline T Accumulate(const A& Input, T init)
		{
			return Accumulate(Input, init, Plus<>());
		}

		template <typename T, typename A, typename MapT, typename OpT>
		__forceinline T TransformAccumulate(const A& Input, MapT MapOp, T init, OpT Op)
		{
			T result = init;
			for (auto&& i : Input)
			{
				result = Op(result, MapOp(i));
			}
			return result;
		}

		template <typename T, typename A, typename MapT>
		__forceinline T TransformAccumulate(const A& Input, MapT MapOp, T init)
		{
			return TransformAccumulate(Input, MapOp, init, Plus<>());
		}
	}
}