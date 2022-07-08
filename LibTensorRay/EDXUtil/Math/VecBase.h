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

namespace EDX
{
	template<unsigned int N, class T>
	class Vec
	{
	public:
		T val[N];

		Vec()
		{
		}
		__forceinline Vec(const T val)
		{
			for (auto i = 0; i < N; i++)
				(*this)[i] = val;
		}
		__forceinline Vec(const Vec& rhs)
		{
			this->operator=(rhs);
		}
		__forceinline Vec& operator = (const Vec& rhs)
		{
			for (auto i = 0; i < N; i++)
				(*this)[i] = rhs[i];
			return *this;
		}
		template<class T1>
		__forceinline Vec(const Vec<N, T1>& rhs)
		{
			this->operator=<T1>(rhs);
		}
		template<class T1>
		__forceinline Vec& operator = (const Vec<N, T1>& vOther)
		{
			for (auto i = 0; i < N; i++)
				(*this)[i] = T(vOther[i]);
			return *this;
		}
		
		__forceinline const T& operator [] (const size_t idx) const { return val[idx]; }
		__forceinline		T& operator [] (const size_t idx)		{ return val[idx]; }

		__forceinline bool operator == (const Vec& rhs) const
		{
			for (auto i = 0; i < N; i++)
			{
				if ((*this)[i] != rhs[i])
					return false;
			}
			return true;
		}
		__forceinline bool operator != (const Vec& rhs) const
		{
			for (auto i = 0; i < N; i++)
			{
				if ((*this)[i] == rhs[i])
					return false;
			}
			return true;
		}
	};
}