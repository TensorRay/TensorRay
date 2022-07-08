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
#include "Base.h"

namespace EDX
{
	class Timer
	{
	public:
		double mLastElapsedAbsoluteTime;
		double mBaseAbsoluteTime;

		double mLastElapsedTime;
		double mBaseTime;
		double mStopTime;
		bool mbTimerStopped;

		char mFrameRate[16];
		dword mNumFrames;
		double mLastFPSTime;

		LARGE_INTEGER mPerfFreq;

		Timer()
		{
			QueryPerformanceFrequency(&mPerfFreq);
			double fTime = GetAbsoluteTime();

			mBaseAbsoluteTime = fTime;
			mLastElapsedAbsoluteTime = fTime;

			mBaseTime = fTime;
			mStopTime = 0.0;
			mLastElapsedTime = fTime;
			mbTimerStopped = false;

			mFrameRate[0] = '\0';
			mNumFrames = 0;
			mLastFPSTime = fTime;
		}

		double GetAbsoluteTime()
		{
			LARGE_INTEGER Time;
			QueryPerformanceCounter(&Time);
			double fTime = (double)Time.QuadPart / (double)mPerfFreq.QuadPart;
			return fTime;
		}

		double GetTime()
		{
			// Get either the current time or the stop time, depending
			// on whether we're stopped and what comand was sent
			return (mStopTime != 0.0) ? mStopTime : GetAbsoluteTime();
		}

		double GetElapsedTime()
		{
			double fTime = GetAbsoluteTime();

			double fElapsedAbsoluteTime = (double)(fTime - mLastElapsedAbsoluteTime);
			mLastElapsedAbsoluteTime = fTime;
			return fElapsedAbsoluteTime;
		}

		// Return the current time
		double GetAppTime()
		{
			return GetTime() - mBaseTime;
		}

		// Reset the timer
		double Reset()
		{
			double fTime = GetTime();

			mBaseTime = fTime;
			mLastElapsedTime = fTime;
			mStopTime = 0;
			mbTimerStopped = false;
			return 0.0;
		}

		// Start the timer
		void Start()
		{
			double fTime = GetAbsoluteTime();

			if (mbTimerStopped)
				mBaseTime += fTime - mStopTime;
			mStopTime = 0.0;
			mLastElapsedTime = fTime;
			mbTimerStopped = false;
		}

		// Stop the timer
		void Stop()
		{
			double fTime = GetTime();

			if (!mbTimerStopped)
			{
				mStopTime = fTime;
				mLastElapsedTime = fTime;
				mbTimerStopped = true;
			}
		}

		// Advance the timer by 1/10th second
		void SingleStep(double fTimeAdvance)
		{
			mStopTime += fTimeAdvance;
		}

		void MarkFrame()
		{
			mNumFrames++;
		}

		char* GetFrameRate()
		{
			double fTime = GetAbsoluteTime();

			// Only re-compute the FPS (frames per second) once per second
			if (fTime - mLastFPSTime > 1.0)
			{
				double fFPS = mNumFrames / (fTime - mLastFPSTime);
				mLastFPSTime = fTime;
				mNumFrames = 0L;
				sprintf_s(mFrameRate, "FPS: %0.02f", (float)fFPS);
			}
			return mFrameRate;
		}
	};
}