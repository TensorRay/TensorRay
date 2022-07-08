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
#include "Tensor/Tensor.h"
#include <pybind11/pybind11.h>
#include <iostream>
#include <fstream>
#include "Scene.h"

void EnvCreate()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_WNDW);

	cublasStatus_t status;
	status = cublasCreate(&Cublas::GetHandle());

	// Initialize curand
	curandCreateGenerator(&Curand::GetHandle(), CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(Curand::GetHandle(), 1234ULL);

	Optix::GetHandle().CreateModule();
	Optix::GetHandle().CreateProgramGroups();
	Optix::GetHandle().CreatePipelines();
}

void EnvRelease()
{
	cublasDestroy(Cublas::GetHandle());
	curandDestroyGenerator(Curand::GetHandle());
	CuAllocator::GetHandle().FreeAllCached();
	Optix::GetHandle().Release();
}

void DebugTensor(const Tensorf& t, int index)
{
	printf("tshape: %d %d %d\n", t.GetShape(0), t.GetShape(1), t.GetShape(2));
	printf("%d: %f\n", index, t.Get(index));
}

void DebugImage(Tensorf& t, const std::string& fn)
{
	printf("tshape: %d %d %d\n", t.GetShape(0), t.GetShape(1), t.GetShape(2));
	float* rgb = (float*)t.HostData();
	printf("%f %f %f\n", rgb[0], rgb[1], rgb[2]);
	int resX = 1400; // t.GetShape(1);
	int resY = 1000; // t.GetShape(0);
	SaveEXR(rgb, resX, resY, fn.c_str(), true);
}

Tensorf toTensor(ptr_wrapper<float> ptr, const Shape& shape)
{
	Tensorf ret;
	ret.Assign(ptr.get(), shape);
	return ret;
}

void AssignTorchToTensor(ptr_wrapper<float> ptr, Tensorf& t)
{
	t.Assign(ptr.get(), t.GetShape());
}

void AssignTensorToTorch(Tensorf& t, ptr_wrapper<float> ptr)
{
	t.ValueToTorch(ptr);
}

Tensorf DetachTensor(const Tensorf& t)
{
	return Detach(t);
}

int GetNumParam() {
	return ParameterPool::GetHandle().size();
}

int GetParamSize(pybind11::list l)
{
	int tot = 0;
	for (auto& it : ParameterPool::GetHandle())
	{
		Tensorf* pTensor = dynamic_cast<Tensorf*>(it.ptr.get());
		int size = pTensor->LinearSize();
		l.append(size);
		tot += size;
	}
	return tot;
}

void GetVariable(int i, ptr_wrapper<float> ptr, int offset)
{
	auto pool = ParameterPool::GetHandle();
	Tensorf* pTensor = dynamic_cast<Tensorf*>(pool[i].ptr.get());
	auto valTensor = pTensor->HostData();
	std::copy(valTensor, valTensor + pTensor->LinearSize(), &ptr[offset]);
}

void SetVariable(int i, ptr_wrapper<float> ptr, int offset)
{
	auto pool = ParameterPool::GetHandle();
	Tensorf* pTensor = dynamic_cast<Tensorf*>(pool[i].ptr.get());
	pTensor->Assign(&ptr[offset], pTensor->GetShape());
}

void GetGradient(int i, ptr_wrapper<float> ptr, int offset, bool clearGrad)
{
	auto pool = ParameterPool::GetHandle();
	Tensorf* pTensor = dynamic_cast<Tensorf*>(pool[i].ptr.get());
	Tensorf gradTensor = pTensor->Grad();
	auto gradVal = gradTensor.HostData();
	std::copy(gradVal, gradVal + gradTensor.LinearSize(), &ptr[offset]);
	if (clearGrad)
		pTensor->ClearGrad();
}

void SetRandomSeed(size_t seed)
{
	curandSetGeneratorOffset(Curand::GetHandle(), seed);
}
