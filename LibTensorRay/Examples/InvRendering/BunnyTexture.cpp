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

BunnyTextureExample::BunnyTextureExample()
{
    mSceneDir = "../example/inverse_rendering/scenes/bunny_texture/";
    mOutDir = "../example/inverse_rendering/output/bunny_texture/";
    mConfig.mMaxBounces = 3;

    mTargetSpp = 128;
    mTargetSppBatch = 32;

    mConfig.mSppInterior = 4;
    mConfig.mSppInteriorBatch = 4;
    mConfig.mSppPrimary = 4;
    mConfig.mSppPrimaryBatch = 4;
    mConfig.mSppDirect = 4;
    mConfig.mSppDirectBatch = 4;
    mConfig.mSppIndirect = 4;
    mConfig.mSppIndirectBatch = 4;
    mConfig.mSppPixelBoundary = 4;
    mConfig.mSppPixelBoundaryBatch = 4;
    mConfig.mQuiet = true;
    mConfig.mExportDerivative = false;

    mConfig.learningRate = 0.05f;
    mConfig.numIters = 100;

    mpIntegrators.push_back(std::make_unique<PathTracer>());
    mpIntegrators.push_back(std::make_unique<PrimaryBoundaryIntegrator>());
    mpIntegrators.push_back(std::make_unique<DirectBoundaryIntegrator>());
    mpIntegrators.push_back(std::make_unique<IndirectBoundaryIntegrator>());
    mpIntegrators.push_back(std::make_unique<PixelBoundaryIntegrator>());

    std::filesystem::create_directory(mOutDir);
}

void BunnyTextureExample::ConfigureVars(TensorRay::Scene* mpScene)
{
    mpScene->mPrims[2]->mTrans = Tensorf({ Vector3(0.f, 0.f, 0.f) }, true);
    mpScene->mBsdfs[1]->DiffTexture("");
}
