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

#include "../Windows/Base.h"
#include "../Math/Matrix.h"
#include "../Windows/Timer.h"

namespace EDX
{
	class Camera
	{
	public:
		// Camera parameters
		Vector3 mPos;
		Vector3 mTarget;
		Vector3 mUp;
		Vector3 mDir;

		// First person movement params
		float mMoveScaler;
		float mRotateScaler;
		float mYaw, mPitch;

		float mFOV;
		float mRatio;
		float mNearClip;
		float mFarClip;

	protected:
		float mFOV_2;

		// Screen resolution
		int mFilmResX;
		int mFilmResY;

		// Matrices
		Matrix mView;
		Matrix mViewInv;

		Matrix mProj;

		Matrix mScreenToRaster;
		Matrix mRasterToCamera;
		Matrix mCameraToRaster;
		Matrix mRasterToWorld;
		Matrix mWorldToRaster;

		// User input
		Vector3 mMovementVelocity;
		Vector3 mMovementImpulse;
		Vector2 mRotateVelocity;
		Timer mTimer;

	public:
		Camera();
		Camera(const Vector3& ptPos, const Vector3& ptTar, const Vector3& vUp, int iResX, int iResY,
			float fFOV = 35.0f, float fNear = 0.1f, float fFar = 1000.0f);

		virtual ~Camera(void)
		{
		}

		virtual void Init(const Vector3& ptPos, const Vector3& ptTar, const Vector3& vUp, int iResX, int iResY,
			float fFOV = 35.0f, float fNear = 0.1f, float fFar = 1000.0f);

		// Handling the resize event
		virtual void Resize(int width, int height);

		// Getters
		const Matrix& GetViewMatrix() const { return mView; }
		const Matrix& GetViewInvMatrix() const { return mViewInv; }
		const Matrix& GetProjMatrix() const { return mProj; }
		const Matrix& GetRasterMatrix() const { return mScreenToRaster; }

		// Given a point in world space, return the raster coordinate
		inline Vector3 WorldToRaster(const Vector3 ptWorld) const { return Matrix::TransformPoint(ptWorld, mWorldToRaster); }
		inline Vector3 RasterToWorld(const Vector3 ptRas) const { return Matrix::TransformPoint(ptRas, mRasterToWorld); }
		inline Vector3 RasterToCamera(const Vector3 ptRas) const { return Matrix::TransformPoint(ptRas, mRasterToCamera); }
		inline Vector3 CameraToRaster(const Vector3 ptCam) const { return Matrix::TransformPoint(ptCam, mCameraToRaster); }
		inline bool CheckRaster(const Vector3& ptRas) const { return ptRas.x < float(mFilmResX) && ptRas.x >= 0.0f && ptRas.y < float(mFilmResY) && ptRas.y >= 0.0f; }

		int GetFilmSizeX() const { return mFilmResX; }
		int GetFilmSizeY() const { return mFilmResY; }
	};
}