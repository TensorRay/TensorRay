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

#include "BaseCamera.h"

namespace EDX
{
	Camera::Camera()
	{
		Init(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f), Vector3(0.0f, 1.0f, 0.0f), 1024, 640, float(Math::EDX_PI_4), 1.0f, 1000.0f);
	}

	Camera::Camera(const Vector3& vPos, const Vector3& vTar, const Vector3& vUp, int iResX, int iResY, float fFOV, float fNear, float fFar)
	{
		Init(vPos, vTar, vUp, iResX, iResY, fFOV, fNear, fFar);
	}

	void Camera::Init(const Vector3& vPos, const Vector3& vTar, const Vector3& vUp, int iResX, int iResY,
		float fFOV, float fNear, float fFar)
	{
		mPos = vPos;
		mTarget = vTar;
		mUp = vUp;
		mFilmResX = iResX;
		mFilmResY = iResY;
		mFOV = fFOV;
		mNearClip = fNear;
		mFarClip = fFar;

		mMoveScaler = 1.0f;
		mRotateScaler = 0.0025f;

		mFOV_2 = mFOV / 2.0f;

		mDir = Math::Normalize(mTarget - mPos);

		mYaw = Math::Atan2(mDir.x, mDir.z);
		mPitch = -Math::Atan2(mDir.y, Math::Sqrt(mDir.x * mDir.x + mDir.z * mDir.z));

		mView = Matrix::LookAt(mPos, mTarget, mUp);
		mViewInv = Matrix::Inverse(mView);

		Resize(mFilmResX, mFilmResY);
	}

	void Camera::Resize(int iWidth, int iHeight)
	{
		mFilmResX = iWidth;
		mFilmResY = iHeight;

		mRatio = mFilmResX / float(mFilmResY);
		mProj = Matrix::Perspective(mFOV, mRatio, mNearClip, mFarClip);
		mScreenToRaster = Matrix::Scale(float(mFilmResX), float(mFilmResY), 1.0f) *
			Matrix::Scale(-0.5f, -0.5f, 1.0f) *
			Matrix::Translate(Vector3(-1.0f, -1.0f, 0.0f));

		mRasterToCamera = Matrix::Mul(Matrix::Inverse(mProj), Matrix::Inverse(mScreenToRaster));
		mCameraToRaster = Matrix::Inverse(mRasterToCamera);
		mRasterToWorld = Matrix::Mul(mViewInv, mRasterToCamera);
		mWorldToRaster = Matrix::Inverse(mRasterToWorld);
	}
}