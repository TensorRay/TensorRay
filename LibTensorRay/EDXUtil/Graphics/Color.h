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

#include "../Math/Vec3.h"

namespace EDX
{
	class Color
	{
	public:
		float r, g, b, a;

	public:
		Color()
			: r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}
		Color(float fR, float fG, float fB, float fA = 1.0f)
			: r(fR), g(fG), b(fB), a(fA)
		{
			NumericValid();
		}
		Color(float val)
			: r(val), g(val), b(val), a(1.0f)
		{
			NumericValid();
		}
		Color(const class Color4b& c);

		explicit Color(const Vector3& vVec)
			: r(vVec.x), g(vVec.y), b(vVec.z), a(1.0f)
		{
			NumericValid();
		}

		~Color()
		{
		}

		inline bool IsBlack() const { return *this == BLACK; }
		inline float Luminance() const { return r * 0.212671f + g * 0.715160f + b * 0.072169f; }

		__forceinline const float& operator [] (const size_t idx) const { Assert(idx < 4); return (&r)[idx]; }
		__forceinline		float& operator [] (const size_t idx)		{ Assert(idx < 4); return (&r)[idx]; }

		Color operator + (const Color& color) const
		{
			return Color(r + color.r, g + color.g, b + color.b);
		}

		Color& operator += (const Color& color)
		{
			r += color.r; g += color.g; b += color.b; a = 1.0f;
			NumericValid();

			return *this;
		}
		Color operator - (const Color& color) const
		{
			return Color(r - color.r, g - color.g, b - color.b);
		}

		Color& operator -= (const Color& color)
		{
			r -= color.r; g -= color.g; b -= color.b; a = 1.0f;
			NumericValid();

			return *this;
		}
		Color operator * (float val) const
		{
			return Color(val * r, val * g, val * b);
		}

		Color operator * (const Color& color) const
		{
			return Color(r * color.r, g * color.g, b * color.b);
		}

		Color& operator *= (float val)
		{
			r *= val; g *= val; b *= val; a = 1.0f;
			NumericValid();

			return *this;
		}

		Color& operator *= (const Color& color)
		{
			r *= color.r; g *= color.g; b *= color.b; a = 1.0f;
			NumericValid();

			return *this;
		}

		Color operator / (float val) const
		{
			float fInv = 1.0f / val;
			return Color(r * fInv, g * fInv, b * fInv);
		}

		Color operator / (const Color& color) const
		{
			return Color(r / color.r, g / color.g, b / color.b);
		}

		Color& operator /= (float val)
		{
			float fInv = 1.0f / val;
			r *= fInv; g *= fInv; b *= fInv; a = 1.0f;
			NumericValid();

			return *this;
		}


		bool operator == (const Color& color) const
		{
			return r == color.r && g == color.g && b == color.b;
		}
		bool operator != (const Color& color) const
		{
			return r != color.r || g != color.g || b != color.b;
		}

		void NumericValid() const
		{
			Assert(Math::NumericValid(r));
			Assert(Math::NumericValid(g));
			Assert(Math::NumericValid(b));
		}

		static const Color BLACK;
		static const Color WHITE;

		static const Color RED;
		static const Color GREEN;
		static const Color BLUE;

	};

	inline Color operator * (float f, const Color& color) { return color * f; }

	namespace Math
	{
		Color Pow(const Color& color, float fP);
		Color Exp(const Color& color);
	}

	class Color4b
	{
	public:
		_byte r, g, b, a;

	public:
		Color4b()
			: r(0), g(0), b(0), a(255) {}
		Color4b(_byte R, _byte G, _byte B, _byte A = 255)
			: r(R), g(G), b(B), a(A) {}
		Color4b(_byte val)
			: r(val), g(val), b(val), a(255) {}
		Color4b(const Color& c)
		{
			r = Math::Clamp(Math::RoundToInt(255 * c.r), 0, 255);
			g = Math::Clamp(Math::RoundToInt(255 * c.g), 0, 255);
			b = Math::Clamp(Math::RoundToInt(255 * c.b), 0, 255);
			a = Math::Clamp(Math::RoundToInt(255 * c.a), 0, 255);
		}

	public:
		__forceinline Color4b operator * (float val) const
		{
			return Color4b(val * r, val * g, val * b);
		}

		__forceinline Color4b operator + (const Color4b& color) const
		{
			return Color4b(r + color.r, g + color.g, b + color.b);
		}

		__forceinline Color4b& operator += (const Color4b& color)
		{
			r += color.r; g += color.g; b += color.b; a = 1.0f;
			return *this;
		}

		__forceinline void FromFloats(float R, float G, float B, float A = 1.0f)
		{
			r = Math::Clamp(Math::RoundToInt(255 * R), 0, 255);
			g = Math::Clamp(Math::RoundToInt(255 * G), 0, 255);
			b = Math::Clamp(Math::RoundToInt(255 * B), 0, 255);
			a = Math::Clamp(Math::RoundToInt(255 * A), 0, 255);
		}

		__forceinline Color4b operator / (float val) const
		{
			float fInv = 1.0f / val;
			return Color4b(r * fInv, g * fInv, b * fInv);
		}

		__forceinline Color4b operator >> (const int shift) const { return Color4b(r >> shift, g >> shift, b >> shift, a); }
	};
}