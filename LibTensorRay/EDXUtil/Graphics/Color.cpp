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

#include "Color.h"

namespace EDX
{
	const Color Color::BLACK(0.0f, 0.0f, 0.0f);
	const Color Color::WHITE(1.0f, 1.0f, 1.0f);
	const Color Color::RED(1.0f, 0.0f, 0.0f);
	const Color Color::GREEN(0.0f, 1.0f, 0.0f);
	const Color Color::BLUE(0.0f, 0.0f, 1.0f);

	Color::Color(const Color4b& c)
		: r(c.r * 0.00390625f)
		, g(c.g * 0.00390625f)
		, b(c.b * 0.00390625f)
		, a(c.a * 0.00390625f)
	{
		NumericValid();
	}

	namespace Math
	{
		Color Pow(const Color& color, float pow)
		{
			float r = Math::Pow(color.r, pow);
			float g = Math::Pow(color.g, pow);
			float b = Math::Pow(color.b, pow);
			float a = color.a;
			return Color(r, g, b, a);
		}

		Color4b Pow(const Color4b& color, float pow)
		{
			return Pow(Color(color), pow);
		}

		Color Exp(const Color& color)
		{
			return Color(Math::Exp(color.r), Math::Exp(color.g), Math::Exp(color.b));
		}
	}
}