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

#include "Base.h"
#include "../Graphics/Color.h"
#include <WinGDI.h>

#include "Bitmap.h"
#include "../Core/Memory.h"

//#define STBI_HEADER_FILE_ONLY
#define STB_IMAGE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS 1
#include "stb_image.h"

namespace EDX
{
	// write data to bmp file
	void Bitmap::SaveBitmapFile(const char* strFilename, const float* pData, int iWidth, int iHeight, EDXImageFormat format)
	{
		int iPixelBytes = int(format);
		int iSize = iWidth * iHeight * iPixelBytes; // the byte of pixel, data size

		const float fInvGamma = 1.f / 2.2f; // Gamma

		unsigned char* pPixels = new unsigned char[iSize];
		for (int i = 0; i < iHeight; i++)
		{
			for (int j = 0; j < iWidth; j++)
			{
				int iWriteIndex = int(format) * ((iHeight - i - 1) * iWidth + j);
				int iIndex = int(format) * (i * iWidth + j);
				pPixels[iWriteIndex + 0] = Math::Clamp(Math::RoundToInt(255 * pData[iIndex + 2]), 0, 255);
				pPixels[iWriteIndex + 1] = Math::Clamp(Math::RoundToInt(255 * pData[iIndex + 1]), 0, 255);
				pPixels[iWriteIndex + 2] = Math::Clamp(Math::RoundToInt(255 * pData[iIndex + 0]), 0, 255);

				// Set alpha
				if (format == EDX_RGBA_32)
					pPixels[iWriteIndex + 3] = 255;
			}
		}

		// Bmp first part, file information
		BITMAPFILEHEADER bmpHeader;
		bmpHeader.bfType = 0x4d42; //Bmp
		bmpHeader.bfSize = iSize // data size
			+ sizeof(BITMAPFILEHEADER) // first section size
			+ sizeof(BITMAPINFOHEADER); // second section size

		bmpHeader.bfReserved1 = 0; // reserved 
		bmpHeader.bfReserved2 = 0; // reserved
		bmpHeader.bfOffBits = bmpHeader.bfSize - iSize;

		// Bmp second part, data information
		BITMAPINFOHEADER bmpInfo;
		bmpInfo.biSize = sizeof(BITMAPINFOHEADER);
		bmpInfo.biWidth = iWidth;
		bmpInfo.biHeight = iHeight;
		bmpInfo.biPlanes = 1;
		bmpInfo.biBitCount = 8 * iPixelBytes;
		bmpInfo.biCompression = 0;
		bmpInfo.biSizeImage = iSize;
		bmpInfo.biXPelsPerMeter = 0;
		bmpInfo.biYPelsPerMeter = 0;
		bmpInfo.biClrUsed = 0;
		bmpInfo.biClrImportant = 0;

		FILE* pFile = NULL;
		fopen_s(&pFile, strFilename, "wb");
		Assert(pFile);


		fwrite(&bmpHeader, 1, sizeof(BITMAPFILEHEADER), pFile);
		fwrite(&bmpInfo, 1, sizeof(BITMAPINFOHEADER), pFile);

		fwrite(pPixels, 1, iSize, pFile);
		fclose(pFile);

		Memory::SafeDeleteArray(pPixels);
	}

	void Bitmap::SaveBitmapFile(const char* strFilename, const _byte* pData, int iWidth, int iHeight)
	{
		int iPixelBytes = 4;
		int iSize = iWidth * iHeight * iPixelBytes; // the byte of pixel, data size

		const float fInvGamma = 1.f / 2.2f; // Gamma

		unsigned char* pPixels = new unsigned char[iSize];
		for (int i = 0; i < iHeight; i++)
		{
			for (int j = 0; j < iWidth; j++)
			{
				int iWriteIndex = iPixelBytes * ((iHeight - i - 1) * iWidth + j);
				int iIndex = iPixelBytes * (i * iWidth + j);
				pPixels[iWriteIndex + 0] = pData[iIndex + 2];
				pPixels[iWriteIndex + 1] = pData[iIndex + 1];
				pPixels[iWriteIndex + 2] = pData[iIndex + 0];
				pPixels[iWriteIndex + 3] = pData[iIndex + 3];
			}
		}

		// Bmp first part, file information
		BITMAPFILEHEADER bmpHeader;
		bmpHeader.bfType = 0x4d42; //Bmp
		bmpHeader.bfSize = iSize // data size
			+ sizeof(BITMAPFILEHEADER) // first section size
			+ sizeof(BITMAPINFOHEADER); // second section size

		bmpHeader.bfReserved1 = 0; // reserved 
		bmpHeader.bfReserved2 = 0; // reserved
		bmpHeader.bfOffBits = bmpHeader.bfSize - iSize;

		// Bmp second part, data information
		BITMAPINFOHEADER bmpInfo;
		bmpInfo.biSize = sizeof(BITMAPINFOHEADER);
		bmpInfo.biWidth = iWidth;
		bmpInfo.biHeight = iHeight;
		bmpInfo.biPlanes = 1;
		bmpInfo.biBitCount = 8 * iPixelBytes;
		bmpInfo.biCompression = 0;
		bmpInfo.biSizeImage = iSize;
		bmpInfo.biXPelsPerMeter = 0;
		bmpInfo.biYPelsPerMeter = 0;
		bmpInfo.biClrUsed = 0;
		bmpInfo.biClrImportant = 0;

		FILE* pFile = NULL;
		fopen_s(&pFile, strFilename, "wb");
		Assert(pFile);

		fwrite(&bmpHeader, 1, sizeof(BITMAPFILEHEADER), pFile);
		fwrite(&bmpInfo, 1, sizeof(BITMAPINFOHEADER), pFile);

		fwrite(pPixels, 1, iSize, pFile);
		fclose(pFile);

		Memory::SafeDeleteArray(pPixels);
	}

	template<>
	float* Bitmap::ReadFromFile(const char* strFile, int* pWidth, int* pHeight, int* pChannel)
	{
		return (float*)stbi_loadf(strFile, pWidth, pHeight, pChannel, 4);
	}

	template<>
	Color* Bitmap::ReadFromFile(const char* strFile, int* pWidth, int* pHeight, int* pChannel)
	{
		return (Color*)stbi_loadf(strFile, pWidth, pHeight, pChannel, 4);
	}

	template<>
	Color4b* Bitmap::ReadFromFile(const char* strFile, int* pWidth, int* pHeight, int* pChannel)
	{
		return (Color4b*)stbi_load(strFile, pWidth, pHeight, pChannel, 4);
	}

	template<>
	float* Bitmap::ReadFromFile(const char* strFile, int* pWidth, int* pHeight, int* pChannel, int requiredChannel)
	{
		return (float*)stbi_loadf(strFile, pWidth, pHeight, pChannel, requiredChannel);
	}

	template<>
	uint8* Bitmap::ReadFromFile(const char* strFile, int* pWidth, int* pHeight, int* pChannel, int requiredChannel)
	{
		return (uint8*)stbi_load(strFile, pWidth, pHeight, pChannel, requiredChannel);
	}
}

