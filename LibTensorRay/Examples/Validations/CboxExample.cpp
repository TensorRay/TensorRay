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

CboxExample::CboxExample()
{
    mName = "Cbox";
    
    mConfig.mMaxBounces = 5;

    mFDspp = 128;
    mFDstep = 1e-2f;
    mTargetSpp = 2048;
    mTargetSppBatch = 64;

    mConfig.mSppInterior = 16; // 128;
    mConfig.mSppInteriorBatch = 16;
    mConfig.mSppPrimary = 16; // 128;
    mConfig.mSppPrimaryBatch = 16;
    mConfig.mSppDirect = 16; // 128;
    mConfig.mSppDirectBatch = 16;
    mConfig.mSppIndirect = 16;
    mConfig.mSppIndirectBatch = 16;
    mConfig.mSppPixelBoundary = 16; // 16;
    mConfig.mSppPixelBoundaryBatch = 16;
    mConfig.mQuiet = false;
    
    mpScene = make_unique<TensorRay::Scene>();

    auto curPath = std::filesystem::current_path();
    std::filesystem::path scenePath("../example/validation_box_rfilter/scenes/kitty_in_cbox");
    std::filesystem::current_path(curPath / scenePath);
    SceneLoader::LoadFromFile("tar.xml", *mpScene);
    
    mOutDir = "../../output/kitty_in_cbox/";
    std::filesystem::create_directory(mOutDir);
}

void CboxExample::Configure(Tensorf& x)
{
    mpScene->mPrims[0]->mTrans = Tensorf({ Vector3(1.f, 1.f, 1.f) }, false) * x;
    mpScene->Configure();
}
