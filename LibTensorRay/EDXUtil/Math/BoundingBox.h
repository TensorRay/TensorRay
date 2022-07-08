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

#include "Vector.h"
#include "Ray.h"
#include "EDXMath.h"

namespace EDX
{
	class BoundingBox
	{
	public:
		Vector3 mMin, mMax;

	public:
		BoundingBox()
			: mMin(float(Math::EDX_INFINITY), float(Math::EDX_INFINITY), float(Math::EDX_INFINITY))
			, mMax(float(Math::EDX_NEG_INFINITY), float(Math::EDX_NEG_INFINITY), float(Math::EDX_NEG_INFINITY))
		{}

		BoundingBox(const Vector3& pt)
			: mMin(pt)
			, mMax(pt)
		{}

		BoundingBox(
			const Vector3& pt1,
			const Vector3& pt2)
		{
			mMin = Vector3(Math::Min(pt1.x, pt2.x), Math::Min(pt1.y, pt2.y), Math::Min(pt1.z, pt2.z));
			mMax = Vector3(Math::Max(pt1.x, pt2.x), Math::Max(pt1.y, pt2.y), Math::Max(pt1.z, pt2.z));
		}

		bool Occluded(const Ray& ray, float* pfHitNear = NULL, float* pfHitFar = NULL) const;

		inline bool Inside(const Vector3& point) const
		{
			return point.x >= mMin.x && point.x <= mMax.x &&
				point.y >= mMin.y && point.y <= mMax.y &&
				point.z >= mMin.z && point.z <= mMax.z;
		}

		inline bool Overlaps(const BoundingBox& bbox) const
		{
			return mMin.x <= bbox.mMax.x && mMax.x >= bbox.mMin.x &&
				mMin.y <= bbox.mMax.y && mMax.y >= bbox.mMin.y &&
				mMin.z <= bbox.mMax.z && mMax.z >= bbox.mMin.x;
		}

		inline int MaximumExtent() const
		{
			Vector3 vDiag = mMax - mMin;
			if (vDiag.x > vDiag.y && vDiag.x > vDiag.z)
				return 0;
			else if (vDiag.y > vDiag.z)
				return 1;
			else
				return 2;
		}

		inline Vector3 Centroid() const
		{
			return 0.5f * (mMin + mMax);
		}

		inline Vector3 Offset(const Vector3& pt) const
		{
			return Vector3((pt.x - mMin.x) / (mMax.x - mMin.x + 1e-6f),
				(pt.y - mMin.y) / (mMax.y - mMin.y + 1e-6f),
				(pt.z - mMin.z) / (mMax.z - mMin.z + 1e-6f));
		}

		inline float Area() const
		{
			Vector3 vDiag = mMax - mMin;
			return 2.0f * (vDiag.x * vDiag.y + vDiag.y * vDiag.z + vDiag.x * vDiag.z);
		}

		inline float Volume() const
		{
			Vector3 vDiag = mMax - mMin;
			return vDiag.x * vDiag.y * vDiag.z;
		}

		inline void BoundingSphere(Vector3* pvCenter, float* pfRadius) const
		{
			*pvCenter = 0.5f * (mMin + mMax);
			*pfRadius = Inside(*pvCenter) ? Math::Distance(*pvCenter, mMax) : 0.0f;
		}

		inline void Expand(const float fDelta)
		{
			mMin -= Vector3(fDelta);
			mMax += Vector3(fDelta);
		}

		inline bool operator == (const BoundingBox& bbox)
		{
			return mMin == bbox.mMin && mMax == bbox.mMax;
		}

		inline bool operator != (const BoundingBox& bbox)
		{
			return mMin != bbox.mMin || mMax != bbox.mMax;
		}

		inline const Vector3& operator [] (int i) const
		{
			Assert(i == 0 || i == 1);
			return (&mMin)[i];
		}

		inline Vector3 operator [] (int i)
		{
			Assert(i == 0 || i == 1);
			return (&mMin)[i];
		}

		void RenderInGL() const;
	};

	namespace Math
	{
		inline BoundingBox Union(const BoundingBox& bbox1, const BoundingBox& bbox2)
		{
			BoundingBox ret;

			ret.mMin.x = Math::Min(bbox1.mMin.x, bbox2.mMin.x);
			ret.mMin.y = Math::Min(bbox1.mMin.y, bbox2.mMin.y);
			ret.mMin.z = Math::Min(bbox1.mMin.z, bbox2.mMin.z);
			ret.mMax.x = Math::Max(bbox1.mMax.x, bbox2.mMax.x);
			ret.mMax.y = Math::Max(bbox1.mMax.y, bbox2.mMax.y);
			ret.mMax.z = Math::Max(bbox1.mMax.z, bbox2.mMax.z);

			return ret;
		}

		inline BoundingBox Union(const BoundingBox& bbox, const Vector3& point)
		{
			BoundingBox ret;

			ret.mMin.x = Math::Min(bbox.mMin.x, point.x);
			ret.mMin.y = Math::Min(bbox.mMin.y, point.y);
			ret.mMin.z = Math::Min(bbox.mMin.z, point.z);
			ret.mMax.x = Math::Max(bbox.mMax.x, point.x);
			ret.mMax.y = Math::Max(bbox.mMax.y, point.y);
			ret.mMax.z = Math::Max(bbox.mMax.z, point.z);

			return ret;
		}
	}
}