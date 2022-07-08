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

#include "Test.cuh"

#include "Examples/Example.h"
#include "Examples/Validations/ValidationExamples.h"
#include "Examples/InvRendering/InvRenderingExamples.h"
#include "Renderer/SceneLoader.h"

#include "Tensor/Tensor.h"
#include "Math/Constants.h"

#include <iostream>
using namespace std;

using namespace DeepLearning;

using namespace EDX;
using namespace EDX::DeepLearning;
using namespace EDX::TensorRay;


void TensorRay::test_memory()
{
    Tensorf A = Tensorf::ArrayRange(2000000, true);
    Tensorf B = Tensorf::ArrayRange(2000000) * Scalar(5.0f);

    for (int i = 0; i < 1000000; i++)
    {
        printf("iter %d: %d\n", i, A.GetGradPtrUseCount());
        auto exp = Sin(A + B * A * A * B * A * B + B) + Cos(A * A / B - A + B) + Log(A * B);
        printf("iter %d: %d\n", i, A.GetGradPtrUseCount());
        Tensorf results = exp;
        printf("iter %d: %d\n", i, A.GetGradPtrUseCount());
        results.Backward(Ones(results.GetShape()));
        printf("iter %d: %d\n", i, A.GetGradPtrUseCount());
        AccumulateGradsAndReleaseGraph();
        printf("iter %d: %d\n", i, A.GetGradPtrUseCount());
    }
}


void TensorRay::test_examples()
{
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_WNDW);

    // Initialize cublas
    cublasStatus_t status;
    status = cublasCreate(&Cublas::GetHandle());

    // Initialize curand
    curandCreateGenerator(&Curand::GetHandle(), CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(Curand::GetHandle(), 1234ULL);

    Optix::GetHandle().CreateModule();
    Optix::GetHandle().CreateProgramGroups();
    Optix::GetHandle().CreatePipelines();

    std::cout << "Derivative image validation." << std::endl;

    std::vector<ValidationExample*> validations;

    validations.push_back(new CboxExample());

    for (size_t i = 0; i < validations.size(); i++)
    {
        //validations[i]->RenderOrig();
        validations[i]->Validate();
    }

    std::cout << "Inverse rendering optimization." << std::endl;

    std::vector<InvRenderingExample*> invExamples;

    //invExamples.push_back(new BunnyShadowExample());
    //invExamples.push_back(new BunnyTextureExample());

    for (size_t i = 0; i < invExamples.size(); i++)
    {
        std::cout << "RenderTarget" << std::endl;
        invExamples[i]->RenderTarget();
        std::cout << "Optimize" << std::endl;
        invExamples[i]->Optimize();
    }

    std::cout << "Finish running validation examples." << std::endl;

    cublasDestroy(Cublas::GetHandle());
    curandDestroyGenerator(Curand::GetHandle());
    CuAllocator::GetHandle().FreeAllCached();
    Optix::GetHandle().Release();
}


void TensorRay::unit_tests() 
{
    std::cout << "Running unit tests..." << std::endl;

    {
        std::cout << "Test 1:\n";
        
        Tensorf A = Tensorf::LinSpace(1, 10, 10, true/*requires_grad*/);
        Expr B = Scalar(5.0f) * Tensorf::LinSpace(1, 10, 10);
    
        // Forward evalulation
        // In this case only a single fused cuda kernel will get called to evaluate the above expression in parallel
        Expr results = Sin(A + B * A) + Cos(A) + Log(A * B) / Scalar(0.5f);

        Expr C = Tensorf::LinSpace(1, 10, 10) * Scalar(2.0f);
        // Forward evalulation, using the results computed in the previous expressions
        // In this case only a single fused cuda kernel will get called to evaluate the above expression in parallel
        Tensorf results2 = results * C * C + Sin(Log(results) * B) + C;

        std::cout << "results: " << results2 << "\n";
        ValidateBackwardDiff(results2, A);
        ValidateForwardDiff(results2, A);

        ParameterPool::GetHandle().clear();

    }

    {
        std::cout << "Test 2:\n";

        // A: 1x5
        Tensorf A = { { 9,2,3,4,5 } };
        // B: 5x1
        Tensorf B(NestedInitializerList<float, 2>({ {1},{2},{3},{4},{5} }), true/*requires_grad*/);

        // Forward evaluation for broadcasting, single kernel will be called
        // C: 5x5
        Tensorf C = A + B;

        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        /*
        results: 10 3 4 5 6 11 4 5 6 7 12 5 6 7 8 13 6 7 8 9 14 7 8 9 10
        autodiff derivative: 5 5 5 5 5
        numerical derivative: 5.00011 4.99964 4.99892 4.99845 4.99797
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 3:\n";

        Tensorf A = Tensorf::LinSpace(1, 10, 10, true);
        Expr B = Exponent(A * Log(Scalar(2.f)));
        Tensorf results = B;

        std::cout << "results: " << results << "\n";
        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 4:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(5, 2);
        Tensorf C({ 7 });

        // Forward evaluation
        // This takes multiple cuda kernel calls because of reduction (Sum) and matrix multiplication
        // Parts of the expression will still get evaluated in a fused kernel when viable
        Tensorf results = Dot(Dot(B, A) * Log(C), Sin(B)) * Sum(Square(Cos(A * C))) + C;

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 81.1383 86.3454 284.036 285.058 486.933 483.77 689.83 682.483 892.727 881.195
        autodiff derivative: 182.759 -5715.61 -1998.1 5581.23 3433.75 -4414.98 -4114.05 2974.56 5442.58 -1599.5
        numerical derivative: 182.693 -5715.48 -1998.12 5580.89 3433.36 -4414.93 -4113.77 2974.32 5443.99 -1600.14
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 5:\n";

        Tensorf A = Tensorf::ArrayRange(5, true/*requires_grad*/).Reshape(5, 1);
        Tensorf B = Tensorf::ArrayRange(5).Reshape(5, 1);
        Tensorf C = Tensorf::ArrayRange(5).Reshape(5, 1);
        Tensorf D = Tensorf::ArrayRange(5).Reshape(5, 1);

        // Forward evaluation
        // This takes multiple cuda kernel calls because of reduction (Sum) and matrix multiplication
        // Parts of the expression will still get evaluated in a fused kernel when viable
        Tensorf results = Concat(Concat(Concat(A, B, 1), C, 1), D, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
        autodiff derivative: 1 1 1 1 1
        numerical derivative: 1 1.00002 0.999987 0.999927 0.999927
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 6:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(5, 2);
        Tensorf C({ 7 });

        auto exp = Dot(Dot(B, A) * Log(C), Sin(B)) * Sum(Square(Cos(A * C))) + C;

        Tensorf results = Slice(exp, { 0, 0 }, { 5, 1 }) / Slice(exp, { 0, 1 }, { 5, 2 });

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 0.939695 0.996413 1.00654 1.01077 1.01309
        autodiff derivative: -0.305178 0.282004 0.0768819 -0.344239 0.203691 -0.475087 0.446328 0.108358 -0.540227 0.335506
        numerical derivative: -0.304967 0.282049 0.0769496 -0.343919 0.20355 -0.475079 0.446171 0.108421 -0.540495 0.335723
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 7:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(2, 5);

        IndexMask mask = (Tensorf::ArrayRange(5) > Scalar(1));

        Tensorf results = Mask(A * B, mask, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 4 9 16 49 64 81
        autodiff derivative: 0 0 2 3 4 0 0 7 8 9
        numerical derivative: 0 0 1.99997 3.00026 3.99971 0 0 6.99997 8.00133 9.00269
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 8:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(2, 5);

        Tensori indices = { 2, 3, 4 };

        Tensorf results = IndexedRead(A * B, indices, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 4 9 16 49 64 81
        autodiff derivative: 0 0 2 3 4 0 0 7 8 9
        numerical derivative: 0 0 1.99997 3.00026 3.99971 0 0 6.99997 8.00133 9.00269
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 9:\n";

        Tensorf A = Tensorf::ArrayRange(10, true).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(2, 5) * Scalar(3.0f);

        IndexMask mask = Tensori::ArrayRange(10) % Scalar(2);

        Tensorf results = IndexedWrite(A * B, mask.index, { 2, 10 }, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 0 0 0 3 0 12 0 27 0 48 0 75 0 108 0 147 0 192 0 243
        autodiff derivative: 0 3 6 9 12 15 18 21 24 27
        numerical derivative: 0 3.00014 6.00004 8.99887 12.001 14.9994 17.9977 20.9961 24.0097 27.0081
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 10:\n";

        Tensorf A = Tensorf({ Vector4::UNIT_X }, true);
        Tensorf B = { Vector4::UNIT_Y };

        Tensorf results = Concat(A, B, 0);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
        autodiff derivative: 1 1 1 1 1
        numerical derivative: 1 1.00002 0.999987 0.999927 0.999927
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 14:\n";

        Tensorf A = Tensorf({ Vector3::UNIT_Y }, true);
        Tensorf B = Tensorf::ArrayRange(4, false).Reshape(1, 4);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, A);
        ValidateForwardDiff(C, A);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 16:\n";

        Tensorf A = Tensorf::ArrayRange(3, true).Reshape(Shape({ 1 }, VecType::Vec3));
        Tensorf B = Tensorf::ArrayRange(4, 8, 1, false);

        Tensorf C = Z(A) * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, A);
        ValidateForwardDiff(C, A);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 17:\n";
        Tensorf A = Tensorf::ArrayRange(3, 6, 1, false).Reshape(3, 1);
        Tensorf B = Tensorf::ArrayRange(5, 6, 1, true).Reshape(1, 1);

        Tensorf C = A * W(MakeVector4(Zeros(1), Zeros(1), Zeros(1), B, 1));
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 18:\n";
        Tensorf A = Tensorf::ArrayRange(3, false).Reshape(3, 1);
        Tensorf B = Tensorf::ArrayRange(3, true).Reshape(3);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }
    
    {
        std::cout << "Test 19:\n";

        Tensorf A = Tensorf::ArrayRange(3, false).Reshape(1, 3);
        Tensorf B = Tensorf::ArrayRange(3, true).Reshape(3);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 20:\n";

        Tensorf A = Tensorf::ArrayRange(3, false).Reshape(1, 3);
        Tensorf B = Tensorf::ArrayRange(3, true).Reshape(1, 3);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 21:\n";

        Tensorf A = Tensorf::ArrayRange(3, false).Reshape(3, 1);
        Tensorf B = Tensorf::ArrayRange(3, true).Reshape(3, 1);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 22:\n";

        Tensorf A = Tensorf::ArrayRange(15, false).Reshape(3, 5);
        Tensorf B = Tensorf::ArrayRange(5, true);

        Tensorf C = A * B;
        std::cout << "results: " << C << "\n";

        ValidateBackwardDiff(C, B);
        ValidateForwardDiff(C, B);

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 23:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/).Reshape(2, 5);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(5, 2);
        Tensorf C({ 7 });

        auto exp = Dot(B, A * C);

        Tensorf results = exp;

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 0.939695 0.996413 1.00654 1.01077 1.01309
        autodiff derivative: -0.305178 0.282004 0.0768819 -0.344239 0.203691 -0.475087 0.446328 0.108358 -0.540227 0.335506
        numerical derivative: -0.304967 0.282049 0.0769496 -0.343919 0.20355 -0.475079 0.446171 0.108421 -0.540495 0.335723
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 24:\n";

        Tensorf A = Tensorf::ArrayRange(10, true/*requires_grad*/);
        Tensorf B = Tensorf::ArrayRange(10, true/*requires_grad*/);

        Tensori id = { 1 };
        Tensori id2 = { 2, 3, 4, 5, 6 };
        Expr indices = id2 + id;
        auto results0 = IndexedRead(A, indices, 0);
        auto results1 = IndexedRead(B, indices, 0);
        Tensori indices1 = { 2, 3 };
        Tensorf results = IndexedRead(results0 * results1, indices1, 0);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);
        ValidateForwardDiff(results, B);

        /*
        results: 4 9 16 49 64 81
        autodiff derivative: 0 0 2 3 4 0 0 7 8 9
        numerical derivative: 0 0 1.99997 3.00026 3.99971 0 0 6.99997 8.00133 9.00269
        */
        
        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 25:\n";

        Tensorf A = Tensorf::ArrayRange(5, true/*requires_grad*/);
        Tensorf B = Tensorf::ArrayRange(10).Reshape(2, 5);

        Tensori indicesBuf = { 0, 1, 2, 3, 4 };
        Expr indices = IndexedRead(indicesBuf, { 2, 3, 4 }, 0);

        Expr AA = IndexedRead(A, indices, 0);
        Expr BB = IndexedRead(B, indices, 1);

        Tensori indices2 = { 1, 2 };
        Tensorf results = IndexedRead(AA * BB, indices2, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, A);
        ValidateForwardDiff(results, A);

        /*
        results: 9 16 24 36
        autodiff derivative: 0 0 0 11 13
        numerical derivative: 0 0 0 10.9997 12.9986
        */

        ParameterPool::GetHandle().clear();
    }

    {
        std::cout << "Test 26:\n";

        Tensorf B = Tensorf::ArrayRange(5, true).Reshape(1, 5);

        Tensorf results = IndexedWrite(B, {0, 2, 4, 6, 8}, { 3, 10 }, 1);

        std::cout << "results: " << results << "\n";

        ValidateBackwardDiff(results, B);
        ValidateForwardDiff(results, B);

        ParameterPool::GetHandle().clear();
    }

    {
        /*
        std::cout << "Perf Test:\n";

        Tensorf A = Tensorf::ArrayRange(2000000, true);
        Tensorf B = Tensorf::ArrayRange(2000000) * Scalar(5.0f);

        auto exp = Sin(A + B * A * A * B * A * B + B) + Cos(A * A / B - A + B) + Log(A * B);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Forward evalulation
        // In this case only a single fused cuda kernel will get called to evaluate the above expression in parallel
        Tensorf results = exp;

        // Backpropagation
        // In this case one cuda kernel will get called to for each leaf node in the expression that requires gradient
        results.Backward(Ones(results.GetShape()));

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        std::cout << milliseconds << "\n";
        */

        /*
        To test same expression in pytorch:

        torch.set_default_tensor_type(torch.cuda.FloatTensor)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        A = torch.arange(2000000)
        B = torch.arange(2000000) * 5.0

        A = A.float()
        B = B.float()
        A.requires_grad = True

        start.record()

        exp = torch.sin(A + B * A * A * B * A * B + B) + torch.cos(A * A / B - A + B) + torch.log(A * B)
        exp.backward(torch.ones_like(exp))

        end.record()

        torch.cuda.synchronize()

        print(start.elapsed_time(end))
        */
    }
}
