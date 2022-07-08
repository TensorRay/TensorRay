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

#include "Optimizer.h"
#include "InvRenderingExamples.h"

void InvRenderingExample::RenderTarget()
{
    Timer timer;
    timer.Start();

	// Load scene
	auto curPath = std::filesystem::current_path();
	std::filesystem::current_path(curPath / mSceneDir);
	TensorRay::Scene scene;
	std::string fn = "sceneT.xml";
	SceneLoader::LoadFromFile(fn.c_str(), scene);
	scene.Configure();
	std::filesystem::current_path(curPath);

	// Render target image
	int optSpp = mConfig.mSppInterior;
	int optSppBatch = mConfig.mSppInteriorBatch;
	mConfig.mSppInterior = mTargetSpp;
	mConfig.mSppInteriorBatch = mTargetSppBatch;
	mTargetImg = mpIntegrators[0]->RenderC(scene, mConfig);
	mTargetImg = Detach(mTargetImg);
	mConfig.mSppInterior = optSpp;
	mConfig.mSppInteriorBatch = optSppBatch;

	string filename = mOutDir + string_format("target.exr");
	SaveEXR((float*)Tensorf::Transpose(mTargetImg).HostData(), scene.GetImageWidth(0), scene.GetImageHeight(0), filename.c_str());
	std::cout << "[INFO] Target image rendering takes " << timer.GetElapsedTime() << " seconds." << std::endl;
}

void InvRenderingExample::Optimize()
{
	// TODO: avoid negative values when optimizing textures

	auto curPath = std::filesystem::current_path();
	std::filesystem::current_path(curPath / mSceneDir);
	TensorRay::Scene scene;
	std::string fn = "scene.xml";
	SceneLoader::LoadFromFile(fn.c_str(), scene);
	std::filesystem::current_path(curPath);

	ConfigureVars(&scene);	
	scene.Configure();

	Adam optimizer(mConfig.learningRate);
	Timer timer;
	for (int iter = 0; iter < mConfig.numIters; iter++)
	{
		timer.Start();

		// Render forward image for diff. image
		curandSetGeneratorOffset(Curand::GetHandle(), iter * 1e5);
		Tensorf fwdImg = mpIntegrators[0]->RenderC(scene, mConfig);
		fwdImg = Detach(fwdImg);

		curandSetGeneratorOffset(Curand::GetHandle(), iter * 1e5);
		Tensorf dLdI = fwdImg - mTargetImg;

		for (auto& integrator : mpIntegrators)
		{
			integrator->RenderD(scene, mConfig, dLdI);
		}

		for (auto& it : ParameterPool::GetHandle())
		{
			Tensorf* pTensor = dynamic_cast<Tensorf*>(it.ptr.get());
			Tensorf gradTensor = pTensor->Grad();
			if (gradTensor.LinearSize() < 10)
				std::cout << "[INFO] Parameter = (" << *pTensor << ")  " << "Grad = (" << gradTensor << ")" << std::endl;
			optimizer.Step(*pTensor, gradTensor);
			pTensor->ClearGrad();
		}
		scene.Configure();
		
		Tensorf loss = Mean(Square(dLdI)) * Scalar(0.5f);
		string filename = mOutDir + string_format("iteration_%d.exr", iter);
		SaveEXR((float*)Tensorf::Transpose(fwdImg).HostData(), scene.GetImageWidth(0), scene.GetImageHeight(0), filename.c_str());
		std::cout << string_format("[INFO] Iter = %d, imageLoss = %f, elapsedTime = %f", iter, loss.Get(0), timer.GetElapsedTime()) << std::endl;
	}
	ParameterPool::GetHandle().clear();
}
