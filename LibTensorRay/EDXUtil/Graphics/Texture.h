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
#include "../Graphics/Color.h"
#include "../Math/Vector.h"
#include "../Containers/BlockedDimensionalArray.h"

namespace EDX
{
	enum class TextureFilter
	{
		Nearest = 0,
		Linear = 1,
		TriLinear = 2,
		Anisotropic4x = 3,
		Anisotropic8x = 4,
		Anisotropic16x = 5
	};

	enum class TextureWrapMode
	{
		Clamp, Repeat, Mirror
	};

	template<uint Dim, class T>
	class Texture
	{
	public:
		virtual ~Texture() {}
		virtual T Sample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim]) const = 0;
		virtual T Sample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim], TextureFilter filter) const = 0;
		virtual bool HasAlpha() const { return false; }
		virtual bool IsConstant() const { return false; }
		virtual int Width() const { return 0; }
		virtual int Height() const { return 0; }
		virtual void SetFilter(const TextureFilter filter)
		{
		}

		virtual T GetValue() const
		{
			return T(0);
		}
		virtual void SetValue(const T& value)
		{
		}
	};

	template<class T>
	using Texture2D = Texture < 2, T >;
	template<class T>
	using Texture3D = Texture < 3, T >;

	template<uint Dim, class T>
	class ConstantTexture : public Texture < Dim, T >
	{
	private:
		T mVal;

	public:
		ConstantTexture(const T& val)
			: mVal(val) {}

		__forceinline T Sample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim]) const
		{
			return mVal;
		}
		__forceinline T Sample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim], TextureFilter filter) const
		{
			return mVal;
		}

		bool IsConstant() const { return true; }

		T GetValue() const
		{
			return mVal;
		}
		void SetValue(const T& value)
		{
			this->mVal = value;
		}
	};

	template<class T>
	using ConstantTexture2D = ConstantTexture < 2, T >;
	template<class T>
	using ConstantTexture3D = ConstantTexture < 3, T >;

	template<uint Dim, typename T, typename Container = DimensionalArray<Dim, T>>
	class Mipmap
	{
	private:
		Vec<Dim, int> mOffsetTable[Math::Pow2<Dim>::Value];
		Vec<Dim, int> mTexDims;
		int mNumLevels;

	public:
		Container* mpLeveledTexels;
		Mipmap()
		{
			for (uint i = 0; i < Math::Pow2<Dim>::Value; i++)
				for (uint d = 0; d < Dim; d++)
					mOffsetTable[i][d] = (i & (1 << d)) != 0;
		}

		~Mipmap()
		{
			Memory::SafeDeleteArray(mpLeveledTexels);
		}

		void Generate(const Vec<Dim, int>& dims, const T* pRawTex);

		T LinearSample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim]) const;
		T TrilinearSample(const Vec<Dim, float>& texCoord, const Vec<Dim, float> differentials[Dim]) const;
		T SampleLevel_Linear(const Vec<Dim, float>& texCoord, const int level) const;
		T Sample_Nearest(const Vec<Dim, float>& texCoord) const;

		const T* GetMemoryPtr(const int level = 0) const
		{
			Assert(level < mNumLevels);
			return mpLeveledTexels[level].Data();
		}
		const int GetNumLevels() const
		{
			return mNumLevels;
		}
	};

	template<class T>
	using Mipmap2D = Mipmap < 2, T >;
	template<class T>
	using Mipmap3D = Mipmap < 3, T >;

	template<typename TRet, typename TMem>
	class ImageTexture : public EDX::Texture2D<TRet>
	{
	private:
		int mTexWidth;
		int mTexHeight;
		float mTexInvWidth, mTexInvHeight;
		bool mHasAlpha;
		TextureFilter mTexFilter;
		Mipmap2D<TMem> mTexels;

	public:
		ImageTexture(const char* strFile, const float gamma = 2.2f);
		ImageTexture(const TMem* pTexels, const int width, const int height);
		~ImageTexture()
		{
		}

		TRet Sample(const Vector2& texCoord, const Vector2 differentials[2]) const;
		TRet Sample(const Vector2& texCoord, const Vector2 differentials[2], TextureFilter filter) const;
		TRet AnisotropicSample(const Vector2& texCoord, const Vector2 differentials[2], const int maxRate) const;

		void SetFilter(const TextureFilter filter)
		{
			mTexFilter = filter;
		}
		int Width() const
		{
			return mTexWidth;
		}
		int Height() const
		{
			return mTexHeight;
		}
		bool HasAlpha() const
		{
			return mHasAlpha;
		}

		static TMem GammaCorrect(TMem tIn, float fGamma = 2.2)
		{
			return Math::Pow(tIn, fGamma);
		}
		static const TMem* GetLevelMemoryPtr(const ImageTexture<TRet, TMem>& tex, const int level = 0)
		{
			return tex.mTexels.GetMemoryPtr(level);
		}
		//static Color ConvertOut(const Color4b& in)
		//{
		//	return in * 0.00390625f;
		//}
		//static Color4b ConvertIn(const Color4b& in)
		//{
		//	return in;
		//}
	};

}