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

#include "Tensor.h"

#include "jitify2.hpp"
static const size_t CacheSize = 8192;
static const char* const CachePath = "JitKernelCache";
static const char* const CommonSource = R"(
// Tensor expression kernel
)";

#ifdef _SOURCE_DIR
string srcDir = _SOURCE_DIR;
#define PATH _SOURCE_DIR
#else
string srcDir = "../";
#endif


namespace EDX
{
	namespace DeepLearning
	{
		Expr::Expr(const Exp& p)
			: ptr(p.ToShared())
		{
		}

		Expr Exp::ForwardDiff(const void* dx, const int elementLinearIdx) const
		{
			return Zeros(1);
		}

		template<typename T>
		Tensor<T>& Tensor<T>::EvalExpr(const Expr& rhs)
		{
			mbRequiresGrad = false;

			Free();
			Forward(rhs);

			if (mbRequiresGrad)
			{
				auto* pTensorRHS = dynamic_cast<Tensor*>(rhs.ptr.get());
				if (!pTensorRHS) // Only track rhs if rhs is not a tensor
					mpExp = rhs;
			}

			return *this;
		}

		template Tensor<float>& Tensor<float>::EvalExpr(const Expr& rhs);
		template Tensor<double>& Tensor<double>::EvalExpr(const Expr& rhs);
		template Tensor<int>& Tensor<int>::EvalExpr(const Expr& rhs);
		template Tensor<uint>& Tensor<uint>::EvalExpr(const Expr& rhs);
		template Tensor<bool>& Tensor<bool>::EvalExpr(const Expr& rhs);

		template<typename T>
		void Tensor<T>::Forward(const Expr& rhs, const bool bForceRecompute)
		{
			this->Forward(rhs.ptr.get(), bForceRecompute);
		}

		template void Tensor<float>::Forward(const Expr& rhs, const bool bForceRecompute);
		template void Tensor<double>::Forward(const Expr& rhs, const bool bForceRecompute);
		template void Tensor<int>::Forward(const Expr& rhs, const bool bForceRecompute);
		template void Tensor<uint>::Forward(const Expr& rhs, const bool bForceRecompute);
		template void Tensor<bool>::Forward(const Expr& rhs, const bool bForceRecompute);

		template<typename T>
		void Tensor<T>::Forward(Exp* rhs, const bool bForceRecompute)
		{
			if (!bForceRecompute)
			{
				const auto* pTensorRHS = dynamic_cast<const Tensor<T>*>(rhs);
				if (pTensorRHS) // This will fail when the T and TDeviceType don't match
				{
					this->operator=(*pTensorRHS);
					return;
				}
			}

			{
#if USE_PROFILING
				nvtxRangePushA("Jit source gen.");
#endif
				GraphProcessContext context;
				rhs->RecursiveProcess(context, bForceRecompute);

				GenerateAndLaunchCUDAKernel(rhs);

#if USE_PROFILING
				nvtxRangePop();
#endif
			}

			rhs->value = *this;
			rhs->mValueCached = true;
		}

		template void Tensor<float>::Forward(Exp* rhs, const bool bForceRecompute);
		template void Tensor<double>::Forward(Exp* rhs, const bool bForceRecompute);
		template void Tensor<int>::Forward(Exp* rhs, const bool bForceRecompute);
		template void Tensor<uint>::Forward(Exp* rhs, const bool bForceRecompute);
		template void Tensor<bool>::Forward(Exp* rhs, const bool bForceRecompute);

		template<typename T>
		void Tensor<T>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op)
		{
			if (!bInplace)
			{
				Resize(rhs->GetShape());
				mbRequiresGrad = rhs->mbRequiresGrad;
			}

			string argsStr;
			// Print output pointers
			if (std::is_same<T, float>::value)
			{
				argsStr += ",\n\tfloat* pDest";
			}
			else if (std::is_same<T, int>::value)
			{
				argsStr += ",\n\tint* pDest";
			}
			else if (std::is_same<T, uint>::value)
			{
				argsStr += ",\n\tunsigned int* pDest";
			}
			else if (std::is_same<T, bool>::value)
			{
				argsStr += ",\n\tbool* pDest";
			}
			else if (std::is_same<T, double>::value)
			{
				argsStr += ",\n\tdouble* pDest";
			}
			VariableCache::GetHandle().clear();
			TensorVariableCache::GetHandle().clear();
			
			VariableMap localVariableMap;
			string exprStr;
			string retVarName = rhs->EmitCuda(localVariableMap, "i", "broadcastParams", GetShape(), exprStr, 1, false);


			const map<TensorJitArg, int>& tensorArgs = localVariableMap.tensorArgs;
			const map<ConcatIndex, int>& concatArgs = localVariableMap.concatArgs;
			const map<SliceIndex, int>& sliceArgs = localVariableMap.sliceArgs;
			const map<IndexedReadArg, int>& indexedReadArgs = localVariableMap.indexedReadArgs;

			std::vector<std::pair<int, TensorJitArg>> tensorArgsSorted;
			for (auto it = tensorArgs.begin(); it != tensorArgs.end(); ++it)
				tensorArgsSorted.push_back({ it->second, it->first });

			sort(tensorArgsSorted.begin(), tensorArgsSorted.end(), [=](std::pair<int, TensorJitArg>& a, std::pair<int, TensorJitArg>& b)
			{
				return a.first < b.first;
			});

			std::vector<std::pair<int, IndexedReadArg>> indexedReadArgsSorted;
			for (auto it = indexedReadArgs.begin(); it != indexedReadArgs.end(); ++it)
				indexedReadArgsSorted.push_back({ it->second, it->first });

			sort(indexedReadArgsSorted.begin(), indexedReadArgsSorted.end(), [=](std::pair<int, IndexedReadArg>& a, std::pair<int, IndexedReadArg>& b)
			{
				return a.first < b.first;
			});

			string assignStr;
			int vectorSize = VectorSize();
			if (vectorSize == 1)
			{
				NewLine(assignStr);
				Indent(assignStr, 1); assignStr += "pDest[i] " + op + " " + retVarName + ";";
			}
			else
			{
				if (rhs->GetShape().mVecType > 1)
				{
					NewLine(assignStr);
					if (vectorSize >= 2)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.X(i)] " + op + " " + retVarName + ".x;\n";
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.Y(i)] " + op + " " + retVarName + ".y;\n";
					}
					if (vectorSize >= 3)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.Z(i)] " + op + " " + retVarName + ".z;\n";
					}
					if (vectorSize >= 4)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.W(i)] " + op + " " + retVarName + ".w;\n";
					}
				}
				else
				{
					NewLine(assignStr);
					if (vectorSize >= 2)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.X(i)] " + op + " " + retVarName + ";\n";
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.Y(i)] " + op + " " + retVarName + ";\n";
					}
					if (vectorSize >= 3)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.Z(i)] " + op + " " + retVarName + ";\n";
					}
					if (vectorSize >= 4)
					{
						Indent(assignStr, 1); assignStr += "pDest[broadcastParams.W(i)] " + op + " " + retVarName + ";\n";
					}
				}
			}

			string constantsStr;

			// Print tensor arguments
			for (const auto& it : tensorArgsSorted)
			{
				if (it.second.mType == 0)
				{
					constantsStr += "__constant__ TensorJit<float> Tensor";
				}
				else if (it.second.mType == 1)
				{
					constantsStr += "__constant__ TensorJit<int> Tensor";
				}
				else if (it.second.mType == 2)
				{
					constantsStr += "__constant__ TensorJit<unsigned int> Tensor";
				}
				else if (it.second.mType == 3)
				{
					constantsStr += "__constant__ TensorJit<bool> Tensor";
				}
				constantsStr += to_string(it.first) + ";\n";
			}

			// Print tensor arguments
			for (const auto& it : concatArgs)
			{
				constantsStr += "__constant__ ConcatIndex ";
				constantsStr += "concat" + to_string(it.second) + ";\n";
			}

			// Print tensor arguments
			for (const auto& it : sliceArgs)
			{
				constantsStr += "__constant__ SliceIndex ";
				constantsStr += "slice" + to_string(it.second) + ";\n";
			}

			// Print tensor arguments
			for (const auto& it : indexedReadArgsSorted)
			{
				constantsStr += "__constant__ IndexedReadArg ";
				constantsStr += "indexedRead" + to_string(it.first) + ";\n";
			}


			string kernelSrc = JitKernelTemplate0 + constantsStr + JitKernelTemplate1 + argsStr + JitKernelTemplate2 + exprStr + assignStr + JitKernelTemplate4;

#if USE_PROFILING
			nvtxRangePushA("Jit compile.");
#endif

			static jitify2::ProgramCache<> cache(
				CacheSize,
				*jitify2::Program("program0", CommonSource)->preprocess(),
				nullptr,
				(srcDir + CachePath).c_str()
			);


			jitify2::StringMap generatedSource = { {"GeneratedKernel.cu", kernelSrc} };

			jitify2::LoadedProgram program = cache.get_program(
				{ "ExpressionJitKernel" },
				generatedSource,
				{
					"-include=GeneratedKernel.cu",
					"--use_fast_math",
					"--generate-line-info",
					("--include-path=" + srcDir).c_str()
					//, "--device-debug"
				}
			);

#if USE_PROFILING
			nvtxRangePop();
#endif

			std::vector<void*> arg_ptrs;
			auto broadcastParamsJit = mParams.ToJit();
			arg_ptrs.push_back((void*)&broadcastParamsJit);
			T* pData = mpData.get();
			arg_ptrs.push_back((void*)&pData);

			vector<TensorJit<float>> floatTensorArgs(tensorArgs.size() * 4);
			vector<TensorJit<int>> intTensorArgs(tensorArgs.size() * 4);
			vector<TensorJit<uint>> uintTensorArgs(tensorArgs.size() * 4);
			vector<TensorJit<bool>> boolTensorArgs(tensorArgs.size() * 4);
			for (const auto& it : tensorArgsSorted)
			{
				if (it.second.mType == 0)
				{
					floatTensorArgs.push_back(it.second.ToTensorJit<float>());

					program->set_global_data("Tensor" + to_string(it.first), &floatTensorArgs.back(), 1);
				}
				else if (it.second.mType == 1)
				{
					intTensorArgs.push_back(it.second.ToTensorJit<int>());

					program->set_global_data("Tensor" + to_string(it.first), &intTensorArgs.back(), 1);
				}
				else if (it.second.mType == 2)
				{
					uintTensorArgs.push_back(it.second.ToTensorJit<uint>());

					program->set_global_data("Tensor" + to_string(it.first), &uintTensorArgs.back(), 1);
				}
				else if (it.second.mType == 3)
				{
					boolTensorArgs.push_back(it.second.ToTensorJit<bool>());

					program->set_global_data("Tensor" + to_string(it.first), &boolTensorArgs.back(), 1);
				}
			}

			for (const auto& it : concatArgs)
			{
				program->set_global_data("concat" + to_string(it.second), &it.first, 1);
			}

			for (const auto& it : sliceArgs)
			{
				program->set_global_data("slice" + to_string(it.second), &it.first, 1);
			}

			for (const auto& it : indexedReadArgsSorted)
			{
				program->set_global_data("indexedRead" + to_string(it.first), &it.second, 1);
			}

			const int nElems = NumElements();
			const int blockDim = 256;
			const int gridDim = (nElems + blockDim - 1) / blockDim;

			program->get_kernel("ExpressionJitKernel")
				->configure(gridDim, blockDim)
				->launch(arg_ptrs);

			KernelLaunchCounter::Increment();

			CopyToHost();
		}

		template void Tensor<float>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op);
		template void Tensor<double>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op);
		template void Tensor<int>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op);
		template void Tensor<uint>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op);
		template void Tensor<bool>::GenerateAndLaunchCUDAKernel(Exp* rhs, const bool bInplace, const string& op);

		template<typename T>
		string Tensor<T>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute)
		{
			auto cachedVar = TensorVariableCache::GetHandle().find(std::make_tuple((const void*)mpData.get(), indexName, paramsName));
			if (cachedVar != TensorVariableCache::GetHandle().end())
			{
				return "tensorVar" + to_string(cachedVar->second);
			}

			int varId = TensorVariableCache::GetHandle().size();
			TensorVariableCache::GetHandle().insert({ std::make_tuple((const void*)mpData.get(), indexName, paramsName), varId });

			auto arg = ToJitArg();
			int id;
			auto it = variableMap.tensorArgs.find(arg);
			if (it == variableMap.tensorArgs.end())
			{
				id = variableMap.tensorArgs.size();
				variableMap.tensorArgs.insert({ arg, id });
			}
			else
			{
				id = it->second;
			}

			string varName = "tensorVar" + to_string(varId);

			int vectorSize = VectorSize();
			Assertf(vectorSize <= 4, "Only vector dim less than 4 is currently supported");

			if (vectorSize == 1)
			{
				NewLine(exprStr);
				Indent(exprStr, indentLevels);
				exprStr += "auto " + varName + " = ";
				if (GetShape() != broadcastShape)
				{
					exprStr += "Tensor" + to_string(id) + ".Eval(" + indexName + ", " + paramsName + ");";
				}
				else
				{
					exprStr += "Tensor" + to_string(id) + ".mpData[" + indexName + "];";
				}
			}
			else
			{
				NewLine(exprStr);
				Indent(exprStr, indentLevels);
				exprStr += "float" + to_string(vectorSize) + " " + varName + " = make_float" + to_string(vectorSize) + "(";
				if (vectorSize == 2)
				{
					exprStr += "Tensor" + to_string(id) + ".X(" + indexName + ", " + paramsName + "), ";
					exprStr += "Tensor" + to_string(id) + ".Y(" + indexName + ", " + paramsName + ")); ";
				}
				else if (vectorSize == 3)
				{
					exprStr += "Tensor" + to_string(id) + ".X(" + indexName + ", " + paramsName + "), ";
					exprStr += "Tensor" + to_string(id) + ".Y(" + indexName + ", " + paramsName + "), ";
					exprStr += "Tensor" + to_string(id) + ".Z(" + indexName + ", " + paramsName + "));";
				}
				else if (vectorSize == 4)
				{
					exprStr += "Tensor" + to_string(id) + ".X(" + indexName + ", " + paramsName + "), ";
					exprStr += "Tensor" + to_string(id) + ".Y(" + indexName + ", " + paramsName + "), ";
					exprStr += "Tensor" + to_string(id) + ".Z(" + indexName + ", " + paramsName + "),";
					exprStr += "Tensor" + to_string(id) + ".W(" + indexName + ", " + paramsName + "));";
				}
			}

			return varName;
		}


		template string Tensor<float>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute);
		template string Tensor<double>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute);
		template string Tensor<int>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute);
		template string Tensor<uint>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute);
		template string Tensor<bool>::EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels, const bool bForceRecompute);

		template<typename T>
		template<typename Op>
		void Tensor<T>::ElementWiseBinaryOpInplaceExpr(Tensor<T>& lhs, const Expr& rhs, Op op)
		{
			{
#if USE_PROFILING
				nvtxRangePushA("Jit source gen.");
#endif
				VariableMap variableMap;
				GraphProcessContext context;
				rhs->RecursiveProcess(context, false);

				string opStr;
				if (std::is_same<Op, Algorithm::Plus<void>>::value)
				{
					opStr = "+=";
				}
				else if (std::is_same<Op, Algorithm::Substract<void>>::value)
				{
					opStr += "-=";
				}
				else if (std::is_same<Op, Algorithm::Multiply<void>>::value)
				{
					opStr += "*=";
				}
				else if (std::is_same<Op, Algorithm::Divide<void>>::value)
				{
					opStr += "/=";
				}
				lhs.GenerateAndLaunchCUDAKernel(rhs.ptr.get(), true, opStr);
#if USE_PROFILING
				nvtxRangePop();
#endif
			}

		}

		template void Tensor<float>::ElementWiseBinaryOpInplaceExpr(Tensor<float>& lhs, const Expr& rhs, Algorithm::Plus<> op);
		template void Tensor<double>::ElementWiseBinaryOpInplaceExpr(Tensor<double>& lhs, const Expr& rhs, Algorithm::Plus<> op);
		template void Tensor<int>::ElementWiseBinaryOpInplaceExpr(Tensor<int>& lhs, const Expr& rhs, Algorithm::Plus<> op);
		template void Tensor<uint>::ElementWiseBinaryOpInplaceExpr(Tensor<uint>& lhs, const Expr& rhs, Algorithm::Plus<> op);
		template void Tensor<bool>::ElementWiseBinaryOpInplaceExpr(Tensor<bool>& lhs, const Expr& rhs, Algorithm::Plus<> op);

		template void Tensor<float>::ElementWiseBinaryOpInplaceExpr(Tensor<float>& lhs, const Expr& rhs, Algorithm::Substract<> op);
		template void Tensor<double>::ElementWiseBinaryOpInplaceExpr(Tensor<double>& lhs, const Expr& rhs, Algorithm::Substract<> op);
		template void Tensor<int>::ElementWiseBinaryOpInplaceExpr(Tensor<int>& lhs, const Expr& rhs, Algorithm::Substract<> op);
		template void Tensor<uint>::ElementWiseBinaryOpInplaceExpr(Tensor<uint>& lhs, const Expr& rhs, Algorithm::Substract<> op);
		template void Tensor<bool>::ElementWiseBinaryOpInplaceExpr(Tensor<bool>& lhs, const Expr& rhs, Algorithm::Substract<> op);

		template void Tensor<float>::ElementWiseBinaryOpInplaceExpr(Tensor<float>& lhs, const Expr& rhs, Algorithm::Multiply<> op);
		template void Tensor<double>::ElementWiseBinaryOpInplaceExpr(Tensor<double>& lhs, const Expr& rhs, Algorithm::Multiply<> op);
		template void Tensor<int>::ElementWiseBinaryOpInplaceExpr(Tensor<int>& lhs, const Expr& rhs, Algorithm::Multiply<> op);
		template void Tensor<uint>::ElementWiseBinaryOpInplaceExpr(Tensor<uint>& lhs, const Expr& rhs, Algorithm::Multiply<> op);
		template void Tensor<bool>::ElementWiseBinaryOpInplaceExpr(Tensor<bool>& lhs, const Expr& rhs, Algorithm::Multiply<> op);

		template void Tensor<float>::ElementWiseBinaryOpInplaceExpr(Tensor<float>& lhs, const Expr& rhs, Algorithm::Divide<> op);
		template void Tensor<double>::ElementWiseBinaryOpInplaceExpr(Tensor<double>& lhs, const Expr& rhs, Algorithm::Divide<> op);
		template void Tensor<int>::ElementWiseBinaryOpInplaceExpr(Tensor<int>& lhs, const Expr& rhs, Algorithm::Divide<> op);
		template void Tensor<uint>::ElementWiseBinaryOpInplaceExpr(Tensor<uint>& lhs, const Expr& rhs, Algorithm::Divide<> op);
		template void Tensor<bool>::ElementWiseBinaryOpInplaceExpr(Tensor<bool>& lhs, const Expr& rhs, Algorithm::Divide<> op);


		template<typename T>
		Expr Tensor<T>::ForwardDiff(const void* dx, const int elementLinearIdx) const
		{
			auto cached = ForwardDiffVariableCache::GetHandle().find(mpData.get());
			if (cached != ForwardDiffVariableCache::GetHandle().end())
			{
				return cached->second;
			}

			Expr ret;
			if (mpExp)
			{
				ret = mpExp->ForwardDiff(dx, elementLinearIdx);
			}
			else
			{
				if (dx == (void*)Data())
				{
					if (elementLinearIdx == -1)
					{
						ret = Ones(GetShape());
					}
					else
					{
						Tensorf grad = Zeros(GetShape());
						grad.Set(elementLinearIdx, 1.0f);
						ret = grad;
					}
				}
				else
				{
					ret = Zeros(GetShape());
				}
			}

			ForwardDiffVariableCache::GetHandle().insert({ mpData.get(), ret });
			return ret;
		}


		template Expr Tensor<float>::ForwardDiff(const void* dx, const int elementLinearIdx) const;
		template Expr Tensor<double>::ForwardDiff(const void* dx, const int elementLinearIdx) const;
		template Expr Tensor<int>::ForwardDiff(const void* dx, const int elementLinearIdx) const;
		template Expr Tensor<uint>::ForwardDiff(const void* dx, const int elementLinearIdx) const;
		template Expr Tensor<bool>::ForwardDiff(const void* dx, const int elementLinearIdx) const;


		template<class T>
		void Tensor<T>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const
		{
			if (mbRequiresGrad && mpGrad)
			{
				*mpGrad = *mpGrad + grad;
			}

			if (mpExp)
			{
				upperGradientsMap.insert({ mpExp.ptr.get(), grad.ptr });
			}
		}

		template void Tensor<float>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;
		template void Tensor<double>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;
		template void Tensor<int>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;
		template void Tensor<uint>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;
		template void Tensor<bool>::Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const;


		template<class T>
		void Tensor<T>::Backward(const Expr& grad) const
		{
			vector<const Exp*> nodes;
			if (mpExp)
			{
				VisitedSet::GetHandle().clear();
				TopologicalSort(nodes);
			}
			else if (mbRequiresGrad)
			{
				multimap<const Exp*, shared_ptr<Exp>> dummy;
				Backward(grad, dummy);
				return;
			}

			multimap<const Exp*, shared_ptr<Exp>> upperGradientsMap;
			upperGradientsMap.insert({ this, grad.ptr });
			for (int i = nodes.size() - 1; i >= 0; i--)
			{
				const Exp* current = nodes[i];
				auto grads = upperGradientsMap.equal_range(current);

				Expr gradExpr;
				for (auto it = grads.first; it != grads.second; ++it)
				{
					if (it == grads.first)
						gradExpr = Expr(it->second);
					else
						gradExpr = gradExpr + Expr(it->second);
				}

				current->Backward(gradExpr, upperGradientsMap);
			}
		}

		template void Tensor<float>::Backward(const Expr& grad) const;
		template void Tensor<double>::Backward(const Expr& grad) const;
		template void Tensor<int>::Backward(const Expr& grad) const;
		template void Tensor<uint>::Backward(const Expr& grad) const;
		template void Tensor<bool>::Backward(const Expr& grad) const;

		template<class T>
		void Tensor<T>::Backward() const
		{
			auto grad = Ones(GetShape());
			Backward(grad);
		}

		template void Tensor<float>::Backward() const;
		template void Tensor<double>::Backward() const;
		template void Tensor<int>::Backward() const;
		template void Tensor<uint>::Backward() const;
		template void Tensor<bool>::Backward() const;

		template<class T>
		void Tensor<T>::Update(const float scale)
		{
			*this -= *mpGrad * Scalar(scale);
			ClearGrad();
		}

		template void Tensor<float>::Update(const float);
		template void Tensor<double>::Update(const float);
		template void Tensor<int>::Update(const float);
		template void Tensor<uint>::Update(const float);
		template void Tensor<bool>::Update(const float);

		Expr SafeSqrt(const Expr& x)
		{
			return Sqrt(Maximum(x, Scalar(0.0f)));
		}

		Expr Clamp(const Expr& x, const Expr& val_min, const Expr& val_max)
		{
			return Minimum(Maximum(x, val_min), val_max);
		}

		Expr Lerp(const Expr& a, const Expr& b, const Expr& ratio)
		{
			return (Ones(1) - ratio) * a + ratio * b;
		}

		Expr VectorNormalize(const Expr& x)
		{
			Assert(x->GetShape().VectorSize() == 3);

			auto sqrX = Square(x);
			auto length = Sqrt(X(sqrX) + Y(sqrX) + Z(sqrX));
			auto normalized = x / length;

			return normalized;
		}

		Expr VectorLength(const Expr& x)
		{
			Assert(x->GetShape().VectorSize() == 3 || x->GetShape().VectorSize() == 2);
			auto sqrX = Square(x);
			if (x->GetShape().VectorSize() == 2)
				return Sqrt(X(sqrX) + Y(sqrX));
			else
				return Sqrt(X(sqrX) + Y(sqrX) + Z(sqrX));
		}

		Expr VectorSquaredLength(const Expr& x)
		{
			Assert(x->GetShape().VectorSize() == 3 || x->GetShape().VectorSize() == 2);
			auto sqrX = Square(x);
			if (x->GetShape().VectorSize() == 2)
				return X(sqrX) + Y(sqrX);
			else
				return X(sqrX) + Y(sqrX) + Z(sqrX);
		}

		Expr Mean(const Expr& x, const Shape& axes, const bool keepDim)
		{
			auto sum = Sum(x, axes, keepDim);

			float invDivisor = sum->GetShape().LinearSize() / float(x->GetShape().LinearSize());
			auto mean = sum * Scalar(invDivisor);

			return mean;
		}

		Expr Variance(const Expr& x, const Shape& axes, const bool keepDim)
		{
			auto mean = Mean(x, axes, keepDim);
			auto centeredX = x - mean;
			auto variance = Mean(centeredX * centeredX, axes, keepDim);

			return variance;
		}

		Expr StandardDeviation(const Expr& x, const Shape& axes, const bool keepDim)
		{
			auto variance = Variance(x, axes, keepDim);

			return Sqrt(variance + Scalar(1e-5f));
		}

		Expr TransformPointsHomogeneous(const Expr& pt, const Expr& mat)
		{
			Assert(pt->GetShape().VectorSize() == 4);

			auto transformed_homo = Dot(mat, pt);
			auto x = X(transformed_homo);
			auto y = Y(transformed_homo);
			auto z = Z(transformed_homo);
			auto w = W(transformed_homo);
			auto x_divided = x / w;
			auto y_divided = y / w;
			auto z_divided = z / w;
			auto transformed = MakeVector3(x_divided, y_divided, z_divided);

			return transformed;
		}

		Expr TransformPoints(const Expr& pt, const Expr& mat)
		{
			Assert(pt->GetShape().VectorSize() == 3);

			auto pt_homo = MakeVector4(X(pt), Y(pt), Z(pt), Ones(1));
			auto transformed_homo = Dot(mat, pt_homo);
			auto x = X(transformed_homo);
			auto y = Y(transformed_homo);
			auto z = Z(transformed_homo);
			auto w = W(transformed_homo);
			auto x_divided = x / w;
			auto y_divided = y / w;
			auto z_divided = z / w;
			auto transformed = MakeVector3(x_divided, y_divided, z_divided);

			return transformed;
		}

		Expr TransformVectors(const Expr& vec, const Expr& mat)
		{
			Assert(vec->GetShape().VectorSize() == 3);

			auto vec_homo = MakeVector4(X(vec), Y(vec), Z(vec), Zeros(1));
			auto transformed_homo = Dot(mat, vec_homo);
			auto x = X(transformed_homo);
			auto y = Y(transformed_homo);
			auto z = Z(transformed_homo);
			auto transformed = MakeVector3(x, y, z);

			return transformed;
		}

		Expr TransformNormals(const Expr& vec, const Expr& matInv)
		{
			Assert(vec->GetShape().VectorSize() == 3);
			auto vec_homo = MakeVector4(X(vec), Y(vec), Z(vec), Zeros(1));
			auto transformed_homo = Dot(matInv, vec_homo, true);
			auto x = X(transformed_homo);
			auto y = Y(transformed_homo);
			auto z = Z(transformed_homo);
			auto transformed = MakeVector3(x, y, z);
			return transformed;
		}

		Expr Luminance(const Expr& val)
		{
			Assert(val->GetShape().VectorSize() == 3);
			auto x = X(val);
			auto y = Y(val);
			auto z = Z(val);
			return Scalar(0.212671f) * x + Scalar(0.715160f) * y + Scalar(0.072169f) * z;
		}

		Expr VectorDot(const Expr& vec1, const Expr& vec2)
		{
			Assert((vec1->GetShape().VectorSize() == 3 && vec2->GetShape().VectorSize() == 3) ||
				(vec1->GetShape().VectorSize() == 2 && vec2->GetShape().VectorSize() == 2));
			auto vec1_x = X(vec1);
			auto vec2_x = X(vec2);
			auto vec1_y = Y(vec1);
			auto vec2_y = Y(vec2);
			if (vec1->GetShape().VectorSize() == 2)
			{
				return vec1_x * vec2_x + vec1_y * vec2_y;
			}
			else
			{
				auto vec1_z = Z(vec1);
				auto vec2_z = Z(vec2);
				return vec1_x * vec2_x + vec1_y * vec2_y + vec1_z * vec2_z;
			}
		}

		Expr VectorCross(const Expr& vec1, const Expr& vec2)
		{
			Assert(vec1->GetShape().VectorSize() == 3 && vec2->GetShape().VectorSize() == 3);
			auto vec1_x = X(vec1);
			auto vec1_y = Y(vec1);
			auto vec1_z = Z(vec1);
			auto vec2_x = X(vec2);
			auto vec2_y = Y(vec2);
			auto vec2_z = Z(vec2);

			auto cross_x = vec1_y * vec2_z - vec1_z * vec2_y;
			auto cross_y = vec1_z * vec2_x - vec1_x * vec2_z;
			auto cross_z = vec1_x * vec2_y - vec1_y * vec2_x;

			return MakeVector3(cross_x, cross_y, cross_z);
		}

		Expr VectorReflect(const Expr& in, const Expr& norm)
		{
			return in + (Scalar(2.0f) * VectorDot(-in, norm) * norm);
		}
	}
}