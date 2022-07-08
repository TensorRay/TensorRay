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

#include "ValidationExamples.h"
#include "Renderer/ParticleTracer.h"

void ValidationExample::RenderOrig()
{
    Timer timer;
    timer.Start();
    const TensorRay::Camera& camera = *(mpScene->mSensors[0]);
    unique_ptr<Integrator> pIntegrator = make_unique<PathTracer>();
    RenderOptions options = mConfig;
    options.mSppInterior = mTargetSpp;
    options.mSppInteriorBatch = mTargetSppBatch;
    mpScene->Configure();
    curandSetGeneratorOffset(Curand::GetHandle(), 1234U);
    Tensorf origImg = pIntegrator->RenderC(*mpScene, options);
    string filename = mOutDir + string_format("forward.exr");
    SaveEXR((float*)Tensorf::Transpose(origImg).HostData(), camera.GetFilmSizeX(), camera.GetFilmSizeY(), filename.c_str());
    std::cout << "[INFO] " << "Forward rendering takes " << timer.GetElapsedTime() << " seconds [" << mName << "]" << std::endl;
}

void ValidationExample::RenderFD()
{
    Timer timer;
    timer.Start();
    const TensorRay::Camera& camera = *(mpScene->mSensors[0]);
    unique_ptr<Integrator> pIntegrator = make_unique<PathTracer>();
    RenderOptions options = mConfig;
    options.mSppInterior = mFDspp;
    options.mSppInteriorBatch = mTargetSppBatch;
    int idx = 0;
    for (auto& it : ParameterPool::GetHandle())
    {
        Tensorf x = Detach(*dynamic_cast<Tensorf*>(it.ptr.get()));
        x += Scalar(mFDstep);
        Configure(x);
        curandSetGeneratorOffset(Curand::GetHandle(), 1234U);
        Tensorf positiveImg = pIntegrator->RenderC(*mpScene, options);
        x -= Scalar(2.0 * mFDstep);
        Configure(x);
        curandSetGeneratorOffset(Curand::GetHandle(), 1234U);
        Tensorf negativeImg = pIntegrator->RenderC(*mpScene, options);
        x += Scalar(mFDstep);

        string filename = mOutDir + string_format("finite_diff_grad_%d.exr", idx++);
        Tensorf gradientImg = (positiveImg - negativeImg) / Scalar(2.0f * mFDstep);
        gradientImg = MakeVector3(X(gradientImg), Zeros(camera.GetFilmSizeX() * camera.GetFilmSizeY()), Zeros(camera.GetFilmSizeX() * camera.GetFilmSizeY()));
        SaveEXR((float*)Tensorf::Transpose(gradientImg).HostData(), camera.GetFilmSizeX(), camera.GetFilmSizeY(), filename.c_str());
    }
    std::cout << "[INFO] " << "FD rendering takes " << timer.GetElapsedTime() << " seconds [" << mName << "]" << std::endl;
}

void ValidationExample::Validate()
{
    Tensorf tVar = Tensorf({ 0.f }, true);
    //RenderFD();     // FD

    Timer timer;
    timer.Start();
    // Render gradient image
    const TensorRay::Camera& camera = *(mpScene->mSensors[0]);
    Configure(tVar);
    int imgResX = camera.GetFilmSizeX();
    //int imgResY = camera.GetFilmSizeY();

    unique_ptr<Integrator> pIntegrator;
    mConfig.mExportDerivative = true;

    Tensorf dInterior = Scalar(0.f);
    Tensorf dPrimary = Scalar(0.f);
    Tensorf dDirect = Scalar(0.f);
    Tensorf dIndirect = Scalar(0.f);
    Tensorf dPixel = Scalar(0.f);

    if (mConfig.mSppInterior > 0) {
        pIntegrator = make_unique<PathTracer>();
        dInterior = pIntegrator->RenderD(*mpScene, mConfig, Tensorf());
        string filename = mOutDir + string_format("grad_interior.exr");
        ExportDeriv(dInterior, imgResX, filename);
    }
    if (mConfig.mSppPrimary > 0) {
        pIntegrator = make_unique<PrimaryBoundaryIntegrator>();
        dPrimary = pIntegrator->RenderD(*mpScene, mConfig, Tensorf());
        string filename = mOutDir + string_format("grad_primaryB.exr");
        ExportDeriv(dPrimary, imgResX, filename);
    }
    if (mConfig.mSppDirect > 0) {
        pIntegrator = make_unique<DirectBoundaryIntegrator>();
        dDirect = pIntegrator->RenderD(*mpScene, mConfig, Tensorf());
        string filename = mOutDir + string_format("grad_directB.exr");
        ExportDeriv(dDirect, imgResX, filename);
    }
    if (mConfig.mSppIndirect > 0) {
        pIntegrator = make_unique<IndirectBoundaryIntegrator>();
        dIndirect = pIntegrator->RenderD(*mpScene, mConfig, Tensorf());
        string filename = mOutDir + string_format("grad_indirectB.exr");
        ExportDeriv(dIndirect, imgResX, filename);
    }
    if (mConfig.mSppPixelBoundary > 0) {
        pIntegrator = make_unique<PixelBoundaryIntegrator>();
        dPixel = pIntegrator->RenderD(*mpScene, mConfig, Tensorf());
        string filename = mOutDir + string_format("grad_pixelB.exr");
        ExportDeriv(dPixel, imgResX, filename);
    }

    Tensorf dAll = dInterior + dPrimary + dDirect + dIndirect + dPixel;
    if (!dAll.Empty()) {
        string filename = mOutDir + string_format("grad.exr");
        ExportDeriv(dAll, imgResX, filename);
    }

    ParameterPool::GetHandle().clear();
    std::cout << "[INFO] " << "Gradient rendering takes " << timer.GetElapsedTime() << " seconds [" << mName << "]" << std::endl;
    std::cout << std::endl;
}
