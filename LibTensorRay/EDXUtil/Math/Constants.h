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

#include <limits>

namespace EDX
{
	namespace Constants
	{
		static struct Null {
		} EDX_NULL;

		static struct True {
			inline operator bool() const { return true; }
		} EDX_TRUE;

		static struct False {
			inline operator bool() const { return false; }
		} EDX_FALSE;

		static struct Emp {
		} EDX_EMP;

		static struct Full {
		} EDX_FULL;
	}

	namespace Math
	{
		static struct Zero
		{
			inline operator double() const { return 0.0; }
			inline operator float() const { return 0.0f; }
			inline operator long long() const { return 0; }
			inline operator unsigned long long() const { return 0; }
			inline operator long() const { return 0; }
			inline operator unsigned long() const { return 0; }
			inline operator int() const { return 0; }
			inline operator unsigned int() const { return 0; }
			inline operator short() const { return 0; }
			inline operator unsigned short() const { return 0; }
			inline operator char() const { return 0; }
			inline operator unsigned char() const { return 0; }
		} EDX_ZERO;

		static struct One
		{
			inline operator double() const { return 1.0; }
			inline operator float() const { return 1.0f; }
			inline operator long long() const { return 1; }
			inline operator unsigned long long() const { return 1; }
			inline operator long() const { return 1; }
			inline operator unsigned long() const { return 1; }
			inline operator int() const { return 1; }
			inline operator unsigned int() const { return 1; }
			inline operator short() const { return 1; }
			inline operator unsigned short() const { return 1; }
			inline operator char() const { return 1; }
			inline operator unsigned char() const { return 1; }
		} EDX_ONE;

		static struct NegInf
		{
			inline operator double() const { return -std::numeric_limits<double>::infinity(); }
			inline operator float() const { return -std::numeric_limits<float>::infinity(); }
			inline operator long long() const { return std::numeric_limits<long long>::min(); }
			inline operator unsigned long long() const { return std::numeric_limits<unsigned long long>::min(); }
			inline operator long() const { return std::numeric_limits<long>::min(); }
			inline operator unsigned long() const { return std::numeric_limits<unsigned long>::min(); }
			inline operator int() const { return std::numeric_limits<int>::min(); }
			inline operator unsigned int() const { return std::numeric_limits<unsigned int>::min(); }
			inline operator short() const { return std::numeric_limits<short>::min(); }
			inline operator unsigned short() const { return std::numeric_limits<unsigned short>::min(); }
			inline operator char() const { return std::numeric_limits<char>::min(); }
			inline operator unsigned char() const { return std::numeric_limits<unsigned char>::min(); }

		} EDX_NEG_INFINITY;

		static struct PosInf
		{
			inline operator double() const { return std::numeric_limits<double>::infinity(); }
			inline operator float() const { return std::numeric_limits<float>::infinity(); }
			inline operator long long() const { return std::numeric_limits<long long>::max(); }
			inline operator unsigned long long() const { return std::numeric_limits<unsigned long long>::max(); }
			inline operator long() const { return std::numeric_limits<long>::max(); }
			inline operator unsigned long() const { return std::numeric_limits<unsigned long>::max(); }
			inline operator int() const { return std::numeric_limits<int>::max(); }
			inline operator unsigned int() const { return std::numeric_limits<unsigned int>::max(); }
			inline operator short() const { return std::numeric_limits<short>::max(); }
			inline operator unsigned short() const { return std::numeric_limits<unsigned short>::max(); }
			inline operator char() const { return std::numeric_limits<char>::max(); }
			inline operator unsigned char() const { return std::numeric_limits<unsigned char>::max(); }
		} EDX_INFINITY, EDX_POS_INFINITY;

		static struct NaN
		{
			inline operator double() const { return std::numeric_limits<double>::quiet_NaN(); }
			inline operator float() const { return std::numeric_limits<float>::quiet_NaN(); }
		} EDX_NAN;

		static struct Epsilon
		{
			inline operator double() const { return std::numeric_limits<double>::epsilon(); }
			inline operator float() const { return std::numeric_limits<float>::epsilon(); }
		} EDX_EPSILON;

		static struct OneMinusEpsilon
		{
			inline operator double() const { return 1.0 - std::numeric_limits<double>::epsilon(); }
			inline operator float() const { return 1.0f - std::numeric_limits<float>::epsilon(); }
		} EDX_ONE_MINUS_EPS;

		static struct Pi
		{
			inline operator double() const { return 3.14159265358979323846; }
			inline operator float() const { return 3.14159265358979323846f; }
		} EDX_PI;

		static struct InvPi
		{
			inline operator double() const { return 0.31830988618379069122; }
			inline operator float() const { return 0.31830988618379069122f; }
		} EDX_INV_PI;

		static struct TwoPi
		{
			inline operator double() const { return 6.283185307179586232; }
			inline operator float() const { return 6.283185307179586232f; }
		} EDX_TWO_PI;

		static struct PiOverTwo
		{
			inline operator double() const { return 1.57079632679489661923; }
			inline operator float() const { return 1.57079632679489661923f; }
		} EDX_PI_2;

		static struct InvTwoPi
		{
			inline operator double() const { return 0.15915494309189534561; }
			inline operator float() const { return 0.15915494309189534561f; }
		} EDX_INV_2PI;

		static struct FourPi
		{
			inline operator double() const { return 12.566370614359172464; }
			inline operator float() const { return 12.566370614359172464f; }
		} EDX_FOUR_PI;

		static struct PiOverFour
		{
			inline operator double() const { return 0.785398163397448309616; }
			inline operator float() const { return 0.785398163397448309616f; }
		} EDX_PI_4;

		static struct InvFourPi
		{
			inline operator double() const { return 0.079577471545947672804; }
			inline operator float() const { return 0.079577471545947672804f; }
		} EDX_INV_4PI;

		static struct Step {
		} EDX_STEP;
	}
}
