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

// Template expression definition
// This file should be included before Tensor class declaration in Tensor.h

// Helper functions for cuda code generation
inline void NewLine(string& str)
{
	str += "\n";
}

inline void Indent(string& str, const int levels = 0)
{
	for (int i = 0; i < levels; i++)
	{
		str += "\t";
	}
}


inline Expr Where(const Expr& cond, const Expr& t, const Expr& f);
inline Expr Slice(const Expr& param, const Shape& begin, const Shape& end, const bool backward = false);

inline Expr operator + (const Expr& lhs, const Expr& rhs);

inline Expr operator * (const Expr& lhs, const Expr& rhs);

inline Expr operator > (const Expr& lhs, const Expr& rhs);


inline Expr operator < (const Expr& lhs, const Expr& rhs);


inline Expr operator == (const Expr& lhs, const Expr& rhs);


inline Expr operator != (const Expr& lhs, const Expr& rhs);

inline Expr operator && (const Expr& lhs, const Expr& rhs);
inline Expr operator || (const Expr& lhs, const Expr& rhs);

inline Expr operator - (const Expr& lhs, const Expr& rhs);

inline Expr operator / (const Expr& lhs, const Expr& rhs);

inline Expr operator - (const Expr& param);
inline Expr operator ~ (const Expr& param);

inline Expr IsFinite(const Expr& param);


inline Expr Pow(const Expr& lhs, const Expr& rhs);


inline Expr Maximum(const Expr& lhs, const Expr& rhs);


inline Expr Minimum(const Expr& lhs, const Expr& rhs);


inline Expr Atan2(const Expr& lhs, const Expr& rhs);

inline Expr Inv(const Expr& param);

inline Expr Exponent(const Expr& param);

inline Expr Sqrt(const Expr& param);


inline Expr Square(const Expr& param);


inline Expr Log(const Expr& param);

inline Expr Sin(const Expr& param);

inline Expr Asin(const Expr& param);

inline Expr Cos(const Expr& param);

inline Expr Acos(const Expr& param);

inline Expr Tan(const Expr& param);

inline Expr Atan(const Expr& param);

inline Expr Sigmoid(const Expr& param);

inline Expr Abs(const Expr& param);

inline Expr Dot(const Expr& lhs, const Expr& rhs, const bool leftTransposed = false, const bool rightTransposed = false);

inline Expr IndexedWrite(const Expr& x, const Tensori& _indices, const Shape& _shape, const int _axis);
inline Expr Concat(const Expr& op1, const Expr& op2, const int dim);
inline Expr IndexedRead(const Expr& x, const Expr& indices, const int axis, const bool keepDim = true);
inline Expr IndexedRead(const Expr& x, const Tensori& indices, const int axis, const bool keepDim = true);

template<typename T>
inline Expr Mask(const Expr& x, const struct IndexMask& mask, const int axis);

inline Expr GetComponent(const Expr& param, const int axis);
inline Expr X(const Expr& param);
inline Expr Y(const Expr& param);
inline Expr Z(const Expr& param);
inline Expr W(const Expr& param);
inline Expr MakeVector1(const Expr& _param1, const int _axis = 0);
inline Expr MakeVector2(const Expr& _param1, const Expr& _param2, const int _axis = 0);
inline Expr MakeVector3(const Expr& _param1, const Expr& _param2, const Expr& _param3, const int _axis = 0);
inline Expr MakeVector4(const Expr& _param1, const Expr& _param2, const Expr& _param3, const Expr& _param4, const int _axis = 0);


template<typename T>
struct ScalarExp : public Exp
{
	const T val;

	ScalarExp(const T _val)
		: val(_val)
	{
		mShape = { 1 };
		mType = DeriveType<T>();
	}

	virtual shared_ptr<Exp> ToShared() const
	{
		return make_shared<ScalarExp<T>>(*this);
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		return Scalar(0.0f);
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		string ret;
		ret += to_string(val);
		if (std::is_same<T, float>::value)
			ret += 'f';
		return ret;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		sorted.push_back(this);
	}
};

template<typename T>
static Expr Scalar(const T& val)
{
	return make_shared<ScalarExp<T>>(val);
}


template<typename T>
struct ConstantExp : public Exp
{
	T val;

	ConstantExp(const T _val, const Shape& _shape)
		: val(_val)
	{
		mShape = _shape;
		mType = DeriveType<T>();
	}

	template<typename... TShape>
	ConstantExp(const T _val, TShape... shape)
		: ConstantExp(_val, { shape... })
	{
	}

	virtual shared_ptr<Exp> ToShared() const
	{
		return make_shared<ConstantExp<T>>(*this);
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		return Zeros(mShape);
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		string ret;

		int vecSize = mShape.VectorSize();
		if (vecSize <= 1)
		{
			ret += to_string(val);
			if (std::is_same<T, float>::value)
				ret += 'f';
		}
		else
		{
			ret += "make_float" + to_string(vecSize) + "(" + to_string(val);
			if (std::is_same<T, float>::value)
				ret += 'f';
			ret += +")";
		}
		return ret;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		sorted.push_back(this);
	}
};

template<typename... TShape>
static Expr Zeros(TShape&&... shape)
{
	return make_shared<ConstantExp<float>>(0.0f, std::forward<TShape>(shape)...);
}

template<typename... TShape>
static Expr Ones(TShape&&... shape)
{
	return make_shared<ConstantExp<float>>(1.0f, std::forward<TShape>(shape)...);
}

template<typename... TShape>
static Expr False(TShape&&... shape)
{
	return make_shared<ConstantExp<bool>>(false, std::forward<TShape>(shape)...);
}

template<typename... TShape>
static Expr True(TShape&&... shape)
{
	return make_shared<ConstantExp<bool>>(true, std::forward<TShape>(shape)...);
}

static Expr Constant(const float val, const Shape& shape)
{
	return make_shared<ConstantExp<float>>(val, shape);
}


inline Expr Unbroadcast(const Expr& tensor, const Shape& target);

template<typename TOp>
struct BinaryExp : public Exp
{
	Expr lhs;
	Expr rhs;

	BinaryExp(const Expr& _lhs, const Expr& _rhs)
		: lhs(_lhs)
		, rhs(_rhs)
	{
		mbRequiresGrad = lhs->mbRequiresGrad || rhs->mbRequiresGrad;
		mShape = BroadcastShape(lhs->GetShape(), rhs->GetShape());
		mType = DeriveType(lhs->GetType(), rhs->GetType());
	}

	virtual ~BinaryExp()
	{
	}


	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Shape leftShape = lhs->GetShape();
		auto leftGrad = Unbroadcast(TOp::Backward(rhs, lhs, grad), leftShape);
		
		Shape rightShape = rhs->GetShape();
		auto rightGrad = Unbroadcast(TOp::Backward(lhs, rhs, grad), rightShape);

		upperGradientsMap.insert({ lhs.ptr.get(), leftGrad.ptr });
		upperGradientsMap.insert({ rhs.ptr.get(), rightGrad.ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		auto leftDiff = lhs->ForwardDiff(dx, elementLinearIdx);
		auto rightDiff = rhs->ForwardDiff(dx, elementLinearIdx);

		Expr ret = TOp::ForwardDiff(leftDiff, lhs, rightDiff, rhs);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		lhs->RecursiveProcess(context, bForceRecompute);
		rhs->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		string lhsVar = lhs->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		string rhsVar = rhs->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		exprStr += "auto " + varName + " = ";
		exprStr += TOp::EmitCuda(lhsVar, rhsVar);
		exprStr += ";";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		lhs->TopologicalSort(sorted);
		rhs->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};

struct AddOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static auto Exec(T1 a, T2 b)
	{
		return a + b;
	}


	static Expr Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return inGrad;
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return da + db;
	}


	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " + ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct MulOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static auto Exec(T1 a, T2 b)
	{
		return a * b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return v1 * inGrad;
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return da * b + db * a;
	}


	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " * ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct PowOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static auto Exec(T1 a, T2 b)
	{
		return Math::Pow(a, b);
	}

	// TODO: Fix gradient of pow
	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return inGrad;
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Ones(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "powf(";
		ret += v1;
		ret += ", ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct MaxOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static auto Exec(T1 a, T2 b)
	{
		return a > b ? a : b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return (v1 < v2) * inGrad;
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Where(a > b, da, db);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "fmaxf(";
		ret += v1;
		ret += ", ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct MinOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static auto Exec(T1 a, T2 b)
	{
		return a < b ? a : b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return (v1 > v2) * inGrad;
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Where(a < b, da, db);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "fminf(";
		ret += v1;
		ret += ", ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct GTOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a > b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}


	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " > ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct GTEqOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a >= b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}


	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " >= ";
		ret += v2;
		ret += ")";
		return ret;
	}
};


struct LTOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a < b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " < ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct LTEqOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a <= b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " <= ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct EQOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a == b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " == ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct NotEQOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return a != b;
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " != ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct AndOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return bool(a) && bool(b);
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}


	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " && ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct OrOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return bool(a) || bool(b);
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}

	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += v1;
		ret += " || ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

struct ModOp
{
	template<typename T1, typename T2>
	TENSOR_INLINE static bool Exec(T1 a, T2 b)
	{
		return int(a) % int(b);
	}

	// TODO: Fix mod gradient
	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		return Zeros(inGrad->GetShape());
	}


	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return da;
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "(";
		ret += "int(";
		ret += v1;
		ret += ")";
		ret += " % ";
		ret += "int(";
		ret += v2;
		ret += ")";
		ret += ")";
		return ret;
	}
};

struct Atan2Op
{
	template<typename T1, typename T2>
	TENSOR_INLINE static float Exec(T1 a, T2 b)
	{
		return Math::Atan2(a, b);
	}


	static auto Backward(const Expr& v1, const Expr& v2, const Expr& inGrad)
	{
		// TODO: Fix backwards for Atan2
		return Zeros(1);
	}


	static Expr ForwardDiff(const Expr& da, const Expr& a, const Expr& db, const Expr& b)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& v1, const string& v2)
	{
		string ret;
		ret += "atan2(";
		ret += v1;
		ret += ", ";
		ret += v2;
		ret += ")";
		return ret;
	}
};

inline Expr operator + (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<AddOp>>(lhs, rhs);
}

inline Expr operator * (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<MulOp>>(lhs, rhs);
}


inline Expr operator > (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<GTOp>>(lhs, rhs);
}

inline Expr operator >= (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<GTEqOp>>(lhs, rhs);
}

inline Expr operator < (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<LTOp>>(lhs, rhs);
}

inline Expr operator <= (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<LTEqOp>>(lhs, rhs);
}

inline Expr operator == (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<EQOp>>(lhs, rhs);
}


inline Expr operator != (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<NotEQOp>>(lhs, rhs);
}

inline Expr operator && (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<AndOp>>(lhs, rhs);
}

inline Expr operator || (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<OrOp>>(lhs, rhs);
}

inline Expr operator % (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<ModOp>>(lhs, rhs);
}

inline Expr Pow(const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<PowOp>>(lhs, rhs);
}


inline Expr Maximum(const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<MaxOp>>(lhs, rhs);
}


inline Expr Minimum(const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<MinOp>>(lhs, rhs);
}


inline Expr Atan2(const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<Atan2Op>>(lhs, rhs);
}

template<typename TOp>
struct UnaryExp : public Exp
{
	Expr param;

	UnaryExp(const Expr& _param)
		: param(_param)
	{
		mbRequiresGrad = param->mbRequiresGrad;
		mShape = param->GetShape();
		mType = param->GetType();
	}

	UnaryExp(Expr&& _param)
		: param(std::move(_param))
	{
		mbRequiresGrad = param->mbRequiresGrad;
		mShape = param->GetShape();
		mType = param->GetType();
	}

	virtual ~UnaryExp()
	{
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Assert(grad->GetShape() == GetShape());

		upperGradientsMap.insert({ param.ptr.get(), TOp::Backward(param, grad).ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr diff = param->ForwardDiff(dx, elementLinearIdx);

		Expr localDiff = TOp::ForwardDiff(param);

		Expr ret = localDiff * diff;
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		param->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		string paramVar = param->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		exprStr += "auto " + varName + " = ";
		exprStr += TOp::EmitCuda(paramVar);
		exprStr += ";";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		param->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};

struct NegateOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return -val;
	}

	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Scalar(-1) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Scalar(-1);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "(";
		ret += "-";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct NotOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return !val;
	}

	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Zeros(grad->GetShape());
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "(";
		ret += "!";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct IsFiniteOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return std::isfinite(val);
	}

	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Zeros(grad->GetShape());
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "isfinite(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct InvOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return 1.0f / val;
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Scalar(-1.0f) / (value * value) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Scalar(-1.0f) / (value * value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "(";
		ret += "1.0f / ";
		ret += "(";
		ret += val;
		ret += ")";
		ret += ")";
		return ret;
	}
};

struct ExpOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Exp(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Exponent(value) * grad;
	}


	static Expr ForwardDiff(const Expr& value)
	{
		return Exponent(value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "expf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct SqrtOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Sqrt(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Scalar(0.5f) / Sqrt(value) * grad;
	}


	static Expr ForwardDiff(const Expr& value)
	{
		return Scalar(0.5f) / Sqrt(value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "sqrtf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct SquareOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Square(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Scalar(2.0f) * value * grad;
	}


	static Expr ForwardDiff(const Expr& value)
	{
		return Scalar(2.0f) * value;
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "squaref(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct LogOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Log(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Inv(value) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Inv(value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "logf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct SinOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Sin(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Cos(value) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Cos(value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "sinf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct AsinOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Asin(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Inv(Sqrt(Scalar(1.0f) - Square(value))) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Inv(Sqrt(Scalar(1.0f) - Square(value)));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "asinf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct CosOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Cos(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return -Sin(value) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return -Sin(value);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "cosf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct AcosOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Acos(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Scalar(-1.0f) * Inv(Sqrt(Scalar(1.0f) - Square(value))) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Scalar(-1.0f) * Inv(Sqrt(Scalar(1.0f) - Square(value)));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "acosf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct TanOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Tan(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Inv(Cos(value) * Cos(value)) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Inv(Cos(value) * Cos(value));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "tanf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct AtanOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Atan(val);
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Inv(Scalar(1.0f) + Square(value)) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Inv(Scalar(1.0f) + Square(value));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "atanf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct SigmoidOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return 1.0f / (1.0f + Math::Exp(-val));
	}


	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Sigmoid(value) * (Scalar(1.0f) - Sigmoid(value)) * grad;
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Sigmoid(value) * (Scalar(1.0f) - Sigmoid(value));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "(";
		ret += "1.0f / (1.0f + expf(-1.0f * ";
		ret += val;
		ret += "))";
		ret += ")";
		return ret;
	}
};

//struct ReluOp
//{
//	template<typename T>
//	TENSOR_INLINE static auto Exec(T val)
//	{
//		return val > 0.0f ? val : 0.0f;
//	}
//
//	
//	static string EmitCuda(const string& val)
//	{
//		exprStr += "(";
//		value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
//		exprStr += " > 0.0f ? ";
//		value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
//		exprStr += " 0.0f)";
//	}
//};

struct AbsOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::Abs(val);
	}

	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Where(value > Zeros(1), Scalar(1.0f) * grad, Scalar(-1.0f) * grad);
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Where(value > Zeros(1), Scalar(1.0f), Scalar(-1.0f));
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "fabs(";
		ret += val;
		ret += ")";
		return ret;
	}
};

struct FloorOp
{
	template<typename T>
	TENSOR_INLINE static auto Exec(T val)
	{
		return Math::FloorToInt(val);
	}

	static auto Backward(const Expr& value, const Expr& grad)
	{
		return Zeros(grad->GetShape());
	}

	static Expr ForwardDiff(const Expr& value)
	{
		return Zeros(1);
	}

	static string EmitCuda(const string& val)
	{
		string ret;
		ret += "floorf(";
		ret += val;
		ret += ")";
		return ret;
	}
};

inline Expr operator - (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<AddOp>>(lhs, make_shared<UnaryExp<NegateOp>>(rhs));
}

inline Expr operator / (const Expr& lhs, const Expr& rhs)
{
	return make_shared<BinaryExp<MulOp>>(lhs, make_shared<UnaryExp<InvOp>>(rhs));
}


inline Expr operator - (const Expr& param)
{
	return make_shared<UnaryExp<NegateOp>>(param);
}

inline Expr operator ~ (const Expr& param)
{
	return make_shared<UnaryExp<NotOp>>(param);
}

inline Expr IsFinite(const Expr& param)
{
	return make_shared<UnaryExp<IsFiniteOp>>(param);
}


inline Expr Inv(const Expr& param)
{
	return make_shared<UnaryExp<InvOp>>(param);
}


inline Expr Exponent(const Expr& param)
{
	return make_shared<UnaryExp<ExpOp>>(param);
}


inline Expr Sqrt(const Expr& param)
{
	return make_shared<UnaryExp<SqrtOp>>(param);
}


inline Expr Square(const Expr& param)
{
	return make_shared<UnaryExp<SquareOp>>(param);
}


inline Expr Log(const Expr& param)
{
	return make_shared<UnaryExp<LogOp>>(param);
}


inline Expr Sin(const Expr& param)
{
	return make_shared<UnaryExp<SinOp>>(param);
}


inline Expr Asin(const Expr& param)
{
	return make_shared<UnaryExp<AsinOp>>(param);
}


inline Expr Cos(const Expr& param)
{
	return make_shared<UnaryExp<CosOp>>(param);
}


inline Expr Acos(const Expr& param)
{
	return make_shared<UnaryExp<AcosOp>>(param);
}


inline Expr Tan(const Expr& param)
{
	return make_shared<UnaryExp<TanOp>>(param);
}


inline Expr Atan(const Expr& param)
{
	return make_shared<UnaryExp<AtanOp>>(param);
}


inline Expr Abs(const Expr& param)
{
	return make_shared<UnaryExp<AbsOp>>(param);
}

inline Expr Floor(const Expr& param)
{
	return make_shared<UnaryExp<FloorOp>>(param);
}

inline Expr Sigmoid(const Expr& param)
{
	return make_shared<UnaryExp<SigmoidOp>>(param);
}

struct DotExp : public Exp
{
	Expr lhs;
	Expr rhs;
	const bool leftTransposed;
	const bool rightTransposed;

	DotExp(const Expr& _lhs, const Expr& _rhs, const bool lTrans = false, const bool rTrans = false)
		: lhs(_lhs)
		, rhs(_rhs)
		, leftTransposed(lTrans)
		, rightTransposed(rTrans)
	{
		mbRequiresGrad = lhs->mbRequiresGrad || rhs->mbRequiresGrad;

		mShape = DeriveShape();
		mType = Type::Float;
	}

	virtual ~DotExp()
	{
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		Shape leftShape = lhs->GetShape().LinearizeVector();
		Shape rightShape = rhs->GetShape().LinearizeVector();

		if (leftTransposed)
		{
			Shape shapeCopy = leftShape;
			for (int i = 0; i < leftShape.Size(); i++)
			{
				leftShape[i] = shapeCopy[leftShape.Size() - 1 - i];
			}
		}

		if (rightTransposed)
		{
			Shape shapeCopy = rightShape;
			for (int i = 0; i < rightShape.Size(); i++)
			{
				rightShape[i] = shapeCopy[rightShape.Size() - 1 - i];
			}
		}

		Assertf(leftShape.Size() == rightShape.Size(), "Number of dimensions has to match between left and right tensors in dot product.");
		Assertf(leftShape.Size() <= 2, "Dot product only supports tensors less than 2 dimensions.");

		Assertf(!(leftShape.Size() == 2 && leftShape[1] != rightShape[0]), "Dimension mismatch for tensor multiply.");

		Shape shape = { leftShape[0], rightShape[1] };

		if (lhs->GetShape().VectorSize() > 1 || rhs->GetShape().VectorSize() > 1)
		{
			shape = shape.Vectorize();
		}

		return shape;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
		{
			Assert(value && !value->Empty());
			return;
		}

		value.ptr = nullptr;
		mValueCached = false;


		lhs->RecursiveProcess(context, bForceRecompute);
		rhs->RecursiveProcess(context, bForceRecompute);
		Tensorf left;
		left.GenerateAndLaunchCUDAKernel(lhs.ptr.get());
		lhs->value = left;
		lhs->mValueCached = true;

		Tensorf right;
		right.GenerateAndLaunchCUDAKernel(rhs.ptr.get());
		rhs->value = right;
		rhs->mValueCached = true;

		value = Tensorf();
		value->Resize(GetShape());
		auto* pTensor = dynamic_cast<Tensorf*>(value.ptr.get());
		Tensorf::DotInplace(left, right, leftTransposed, rightTransposed, pTensor);

		mValueCached = true;
		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		Assert(value && !value->Empty());
		return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Expr leftGrad = Dot(grad, rhs, false, true);
		Expr rightGrad = Dot(lhs, grad, true, false);

		if (lhs->GetShape().VectorSize() > 1)
			leftGrad->mShape = leftGrad->mShape.Vectorize();
		if (rhs->GetShape().VectorSize() > 1)
			rightGrad->mShape = rightGrad->mShape.Vectorize();

		upperGradientsMap.insert({ lhs.ptr.get(), leftGrad.ptr });
		upperGradientsMap.insert({ rhs.ptr.get(), rightGrad.ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		auto leftDiff = lhs->ForwardDiff(dx, elementLinearIdx);
		auto rightDiff = rhs->ForwardDiff(dx, elementLinearIdx);

		Expr ret = Dot(leftDiff, rhs) + Dot(lhs, rightDiff);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		lhs->TopologicalSort(sorted);
		rhs->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};

inline Expr Dot(const Expr& lhs, const Expr& rhs, const bool leftTransposed, const bool rightTransposed)
{
	return make_shared<DotExp>(lhs, rhs, leftTransposed, rightTransposed);
}


struct BroadcastExp : public Exp
{
	Expr param;
	const Shape shape;
	const Shape oldShape;

	BroadcastExp(const Expr& _param, const Shape& _shape)
		: param(_param)
		, oldShape(_param->GetShape())
	{
		mShape = _shape;
		if (oldShape.VectorSize() > 1)
			mShape = mShape.Vectorize();

		mType = param->GetType();

		mbRequiresGrad = param->mbRequiresGrad;
	}

	template<typename... TShape>
	BroadcastExp(const Expr& _param, TShape... shape)
		: BroadcastExp(_param, { shape... })
	{
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		upperGradientsMap.insert({ param.ptr.get(), Unbroadcast(grad, oldShape).ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr ret = param->ForwardDiff(dx, elementLinearIdx);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;
		
		param->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		return param->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		param->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	Type GetType() const
	{
		return param->GetType();
	}
};


inline auto Broadcast(const Expr& param, const Shape& shape)
{
	return make_shared<BroadcastExp>(param, shape);
}

template<typename TOp>
struct ProjectExp : public Exp
{
	Expr operand;
	TOp op;
	float initVal;
	Shape axes;
	bool keepDim;

	ProjectExp(const Expr& _operand, TOp _op, const float _initVal, const Shape& _axes, const bool& _keepDim)
		: operand(_operand)
		, op(_op)
		, initVal(_initVal)
		, axes(_axes)
		, keepDim(_keepDim)
	{
		mbRequiresGrad = operand->mbRequiresGrad;

		mShape = DeriveShape();
		mType = operand->GetType();
	}

	virtual ~ProjectExp()
	{
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		if (axes.Size() > 0)
		{
			return Tensorf::ProjectionShape(operand->GetShape(), axes, keepDim);
		}
		else
			return operand->GetShape();
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		if (axes.Size() > 0)
		{
			if (mType == Type::Bool || mType == Type::Int)
				value = Tensori();
			else if (mType == Type::Uint)
				value = Tensorui();
			else if (mType == Type::Float)
				value = Tensorf();
			else if (mType == Type::Double)
				value = Tensord();

			value->Resize(GetShape());
			operand->RecursiveProcess(context, bForceRecompute);

			if (mType == Type::Bool || mType == Type::Int)
			{
				Tensori operandVal;
				operandVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
				operand->value = operandVal;
				operand->mValueCached = true;
				auto* pTensor = dynamic_cast<Tensori*>(value.ptr.get());
				Tensori::ProjectionOpInplace<TOp>(operandVal, axes, keepDim, op, initVal, pTensor);
			}
			else if (mType == Type::Uint)
			{
				Tensorui operandVal;
				operandVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
				operand->value = operandVal;
				operand->mValueCached = true;
				auto* pTensor = dynamic_cast<Tensorui*>(value.ptr.get());
				Tensorui::ProjectionOpInplace<TOp>(operandVal, axes, keepDim, op, initVal, pTensor);
			}
			else if (mType == Type::Float)
			{
				Tensorf operandVal;
				operandVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
				operand->value = operandVal;
				operand->mValueCached = true;
				auto* pTensor = dynamic_cast<Tensorf*>(value.ptr.get());
				Tensorf::ProjectionOpInplace<TOp>(operandVal, axes, keepDim, op, initVal, pTensor);
			}
			else if (mType == Type::Double)
			{
				Tensord operandVal;
				operandVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
				operand->value = operandVal;
				operand->mValueCached = true;
				auto* pTensor = dynamic_cast<Tensord*>(value.ptr.get());
				Tensord::ProjectionOpInplace<TOp>(operandVal, axes, keepDim, op, initVal, pTensor);
			}
			else
				AssertNoEntry();
			mValueCached = true;
		}
		else
		{
			operand->RecursiveProcess(context, bForceRecompute);
		}

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (axes.Size() > 0)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}
		else
		{
			return operand->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		if (operand->GetShape().VectorSize() > 1)
			grad->mShape = grad->mShape.Vectorize();
		Expr broadcast = Broadcast(grad, operand->GetShape());
		upperGradientsMap.insert({ operand.ptr.get(), broadcast.ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff = operand->ForwardDiff(dx, elementLinearIdx);
		Expr ret = make_shared<ProjectExp<TOp>>(localDiff, TOp(), initVal, axes, keepDim);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		operand->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};


inline Expr Sum(const Expr& operand, const Shape& axes = { -2 }, const bool keepDim = false)
{
	return make_shared<ProjectExp<Algorithm::Plus<>>>(operand, Algorithm::Plus<>(), 0.0f, axes, keepDim);
}

inline Expr Product(const Expr& operand, const Shape& axes = { -2 }, const bool keepDim = false)
{
	return make_shared<ProjectExp<Algorithm::Multiply<>>>(operand, Algorithm::Multiply<>(), 1.0f, axes, keepDim);
}

inline Expr Min(const Expr& operand, const Shape& axes = { -2 }, const bool keepDim = false)
{
	return make_shared<ProjectExp<Algorithm::Min<>>>(operand, Algorithm::Min<>(), float(Math::EDX_POS_INFINITY), axes, keepDim);
}

inline Expr Max(const Expr& operand, const Shape& axes = { -2 }, const bool keepDim = false)
{
	return make_shared<ProjectExp<Algorithm::Max<>>>(operand, Algorithm::Max<>(), float(Math::EDX_NEG_INFINITY), axes, keepDim);
}

inline Expr Unbroadcast(const Expr& tensor, const Shape& target)
{
	const auto& tens = tensor;
	Shape shape = tens->GetShape();

	Shape axes;
	if (shape == target)
		return tensor;

	if (shape.VectorSize() > target.VectorSize() && !(target.LinearSize() == 1 && target.Size() == 1))
		axes.Add(-1);

	if (shape.Size() == target.Size())
	{
		if (target.LinearSize() == 1 && target.Size() == 1)
		{
			axes.Add(-2);
		}
		else
		{
			for (int i = 0; i < shape.Size(); i++)
			{
				if (shape[i] > target[i])
					axes.Add(i);
			}
		}

		return Sum(tens, axes, true);
	}
	else
	{
		if (target.LinearSize() == 1 && target.Size() == 1)
		{
			axes.Add(-2);
		}
		else
		{
			int shapeEnd = shape.Size() - 1;
			int targetEnd = target.Size() - 1;
			for (int i = shapeEnd, j = targetEnd; i >= 0; i--, j--)
			{
				if (j < 0 || shape[i] > target[j])
					axes.Add(i);
			}
		}

		return Sum(tens, axes, false);
	}
}


struct SliceExp : public Exp
{
	Expr param;
	TensorParams oldParam;
	TensorParams newParam;
	const Shape begin;
	const Shape end;

	// TODO: Remove this?
	bool bBackward;

	SliceExp(const Expr& _param, const Shape& _begin, const Shape& _end, const bool _backward = false)
		: param(_param)
		, begin(_begin)
		, end(_end)
		, bBackward(_backward)
	{
		mShape = DeriveShape();

		mType = param->GetType();

		newParam.Resize(mShape);
		oldParam.Resize(param->GetShape());

		mbRequiresGrad = param->mbRequiresGrad;
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Shape backBegin = begin;
		for (int i = 0; i < backBegin.mSize; i++)
		{
			backBegin[i] *= -1;
		}
		Shape backEnd = end;
		for (int i = 0; i < backEnd.mSize; i++)
		{
			backEnd[i] = backBegin[i] + oldParam.GetShape(i);
		}

		upperGradientsMap.insert({ param.ptr.get(), Slice(grad, backBegin, backEnd, true).ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff = param->ForwardDiff(dx, elementLinearIdx);

		Expr ret = Slice(localDiff, begin, end, bBackward);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		param->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		SliceIndex arg = ToJit();
		int id;
		auto it = variableMap.sliceArgs.find(arg);
		if (it == variableMap.sliceArgs.end())
		{
			id = variableMap.sliceArgs.size();
			variableMap.sliceArgs.insert({ arg, id });
		}
		else
		{
			id = it->second;
		}


		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		int vectorSize = GetShape().VectorSize();
		if (vectorSize > 1)
			exprStr += "float" + to_string(vectorSize) + " " + varName + ";\n";
		else
			exprStr += "float " + varName + ";\n";

		string sliceVarName = "slice" + to_string(id);
		string sliceLinearIdxName = indexName + "_s" + to_string(varId);
		string sliceBoolName = "sliceBool" + to_string(varId);

		Indent(exprStr, indentLevels); exprStr += "int " + sliceLinearIdxName + " = " + sliceVarName + ".CalcLinearIndex(" + indexName + ", " + paramsName + ");\n";
		Indent(exprStr, indentLevels); exprStr += "bool " + sliceBoolName + " = " + sliceLinearIdxName + " >= 0;\n";
		Indent(exprStr, indentLevels); exprStr += "if (" + sliceBoolName + ")\n";
		Indent(exprStr, indentLevels); exprStr += "{\n";
		string paramString = param->EmitCuda(variableMap, sliceLinearIdxName, sliceVarName + ".oldParam", oldParam.GetShape(), exprStr, indentLevels + 1, bForceRecompute);
		NewLine(exprStr);
		Indent(exprStr, indentLevels + 1); exprStr += varName + " = " + paramString + ";\n";
		Indent(exprStr, indentLevels); exprStr += "}\n";
		Indent(exprStr, indentLevels); exprStr += "else\n";
		Indent(exprStr, indentLevels); exprStr += "{\n";
		Indent(exprStr, indentLevels + 1);
		if (vectorSize > 1)
			exprStr += varName + " = make_float" + to_string(vectorSize) + "(0.0f);\n";
		else
			exprStr += varName + " = 0.0f;\n";
		Indent(exprStr, indentLevels); exprStr += "}\n";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		param->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	SliceIndex ToJit() const
	{
		SliceIndex ret;

		ret.oldParam = oldParam.ToJit();
		ret.newParam = newParam.ToJit();
		ret.begin = begin.ToJit();
		ret.end = end.ToJit();
		ret.bBackward = bBackward;

		return ret;
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		Shape ret;
		ret.Resize(begin.mSize);
		for (int i = 0; i < begin.mSize; i++)
		{
			ret[i] = end[i] - begin[i];
		}
		ret.SetVectorType(param->GetShape().VectorSize());

		return ret;
	}

	TENSOR_INLINE Shape DeriveIndex(const Shape& idx) const
	{
		Shape ret;
		ret.Resize(idx.mSize);
		for (int i = 0; i < begin.mSize; i++)
		{
			ret[i] = idx[i] + begin[i];
		}

		return ret;
	}
};

inline Expr Slice(const Expr& param, const Shape& begin, const Shape& end, const bool backward)
{
	return make_shared<SliceExp>(param, begin, end, backward);
}


struct ConcatenateExp : public Exp
{
	Expr operand1;
	Expr operand2;
	TensorParams params;
	TensorParams params1;
	TensorParams params2;
	const int dim;

	ConcatenateExp(const Expr& op1, const Expr& op2, const int _dim)
		: dim(_dim)
		, operand1(op1)
		, operand2(op2)
	{
		params1.Resize(operand1->GetShape());
		params2.Resize(operand2->GetShape());

		Assertf(params1.GetShape().mSize == params2.GetShape().mSize, "Concatenate requires two tensor to have the same dimension.");
		Assertf(params1.GetShape().mSize > dim, "Concatenate dimension must be smaller than tensor dimension.");

		Shape shape = operand1->GetShape();
		shape[dim] += params2.GetShape()[dim];
		params.Resize(shape);

		mShape = shape;
		mType = DeriveType(operand1->GetType(), operand2->GetType());

		mbRequiresGrad = operand1->mbRequiresGrad || operand2->mbRequiresGrad;
	}

	virtual ~ConcatenateExp()
	{
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Shape begin1;
		begin1.ResizeZeroed(params.GetShape().mSize);
		begin1.SetVectorType(params.GetShape().VectorSize());
		Shape end1 = params1.GetShape();

		Shape begin2;
		begin2.ResizeZeroed(params2.GetShape().mSize);
		begin2[dim] = end1[dim];
		begin2.SetVectorType(params.GetShape().VectorSize());
		Shape end2 = params.GetShape();

		Expr grad1 = Slice(grad, begin1, end1);
		Expr grad2 = Slice(grad, begin2, end2);

		upperGradientsMap.insert({ operand1.ptr.get(), grad1.ptr });
		upperGradientsMap.insert({ operand2.ptr.get(), grad2.ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff1 = operand1->ForwardDiff(dx, elementLinearIdx);
		Expr localDiff2 = operand2->ForwardDiff(dx, elementLinearIdx);

		Expr ret = Concat(localDiff1, localDiff2, dim);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		operand1->RecursiveProcess(context, bForceRecompute);
		operand2->RecursiveProcess(context, bForceRecompute);

        if (mType == Type::Bool || mType == Type::Int)
            value = Tensori();
        else if (mType == Type::Uint)
            value = Tensorui();
        else if (mType == Type::Float)
            value = Tensorf();
        else if (mType == Type::Double)
            value = Tensord();

        value->GenerateAndLaunchCUDAKernel(this);
        mValueCached = true;

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		ConcatIndex arg = ToJit();
		int id;
		auto it = variableMap.concatArgs.find(arg);
		if (it == variableMap.concatArgs.end())
		{
			id = variableMap.concatArgs.size();
			variableMap.concatArgs.insert({ arg, id });
		}
		else
		{
			id = it->second;
		}


		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		int vectorSize = GetShape().VectorSize();
		if (vectorSize > 1)
		{
			exprStr += "float" + to_string(vectorSize) + " " + varName + ";\n";
		}
		else
			exprStr += "float " + varName + ";\n";

		string concatVarName = "concat" + to_string(id);
		string concatLinearIdxName_T = indexName + "_c" + to_string(varId) + "t";
		string concatLinearIdxName_F = indexName + "_c" + to_string(varId) + "f";
		string concatIdxName = "concatIdx" + to_string(varId);
		string concatCondName = "concatBool" + to_string(varId);

		Indent(exprStr, indentLevels); exprStr += "ShapeJit " + concatIdxName + " = " + concatVarName + ".params.Index(" + indexName + ");\n";
		Indent(exprStr, indentLevels); exprStr += "bool " + concatCondName + " = " + concatIdxName + ".x" + to_string(dim) + " < " + concatVarName + ".params1.mShape.x" + to_string(dim) + ";\n";
		Indent(exprStr, indentLevels); exprStr += "if (" + concatCondName + ")\n";
		Indent(exprStr, indentLevels); exprStr += "{\n";
		Indent(exprStr, indentLevels + 1); exprStr += "int " + concatLinearIdxName_T + " = " + concatVarName + ".params1.LinearIndex(" + concatIdxName + ");\n";
		string operandName1 = operand1->EmitCuda(variableMap, concatLinearIdxName_T, concatVarName + ".params1", params1.GetShape(), exprStr, indentLevels + 1, bForceRecompute);
		NewLine(exprStr);
		Indent(exprStr, indentLevels + 1); exprStr += varName + " = " + operandName1 + ";\n";
		Indent(exprStr, indentLevels); exprStr += "}\n";
		Indent(exprStr, indentLevels); exprStr += "else\n";
		Indent(exprStr, indentLevels); exprStr += "{\n";
		Indent(exprStr, indentLevels + 1); exprStr += concatIdxName + ".x" + to_string(dim) + " -= " + concatVarName + ".params1.mShape.x" + to_string(dim) + ";\n";
		Indent(exprStr, indentLevels + 1); exprStr += concatIdxName + ".mSize = " + to_string(params2.GetShape().Size()) + ";\n";
		Indent(exprStr, indentLevels + 1); exprStr += "int " + concatLinearIdxName_F + " = " + concatVarName + ".params2.LinearIndex(" + concatIdxName + ");\n";
		string operandName2 = operand2->EmitCuda(variableMap, concatLinearIdxName_F, concatVarName + ".params2", params2.GetShape(), exprStr, indentLevels + 1, bForceRecompute);
		NewLine(exprStr);
		Indent(exprStr, indentLevels + 1); exprStr += varName + " = " + operandName2 + ";\n";
		Indent(exprStr, indentLevels); exprStr += "}\n";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		operand1->TopologicalSort(sorted);
		operand2->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	ConcatIndex ToJit() const
	{
		ConcatIndex ret;

		ret.params = params.ToJit();
		ret.params1 = params1.ToJit();
		ret.params2 = params2.ToJit();
		ret.dim = dim;

		return ret;
	}

	Type GetType() const
	{
		return DeriveType(operand1->GetType(), operand2->GetType());
	}
};

inline Expr Concat(const Expr& op1, const Expr& op2, const int dim)
{
	return make_shared<ConcatenateExp>(op1, op2, dim);
}

struct ComponentExp : public Exp
{
	Expr param;
	const int comp;
	TensorParams oldParams;
	TensorParams newParams;

	ComponentExp(const Expr& _param, const int _comp)
		: param(_param)
		, comp(_comp)
	{
		oldParams.Resize(param->GetShape());

		Assertf(oldParams.VectorSize() > 1, "Input parameter have to have valid vector axis.");
		Assertf(oldParams.VectorSize() > comp, "Vector size should be larger than component index.");
		Assertf(comp < 4, "Component index has to be less than 4 since there are only 4 components in 4D vectors.");

		mShape = DeriveShape();
		mType = param->GetType();

		newParams.Resize(mShape);

		mbRequiresGrad = param->mbRequiresGrad;
	}


	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Expr vecGrad;
		if (oldParams.VectorSize() == 1)
		{
			vecGrad = grad;
		}
		else if (oldParams.VectorSize() == 2)
		{
			if (comp == 0)
			{
				vecGrad = MakeVector2(grad, Zeros(1));
			}
			else if (comp == 1)
			{
				vecGrad = MakeVector2(Zeros(1), grad);
			}

		}
		else if (oldParams.VectorSize() == 3)
		{
			if (comp == 0)
			{
				vecGrad = MakeVector3(grad, Zeros(1), Zeros(1));
			}
			else if (comp == 1)
			{
				vecGrad = MakeVector3(Zeros(1), grad, Zeros(1));
			}
			else if (comp == 2)
			{
				vecGrad = MakeVector3(Zeros(1), Zeros(1), grad);
			}
		}
		else if (oldParams.VectorSize() == 4)
		{
			if (comp == 0)
			{
				vecGrad = MakeVector4(grad, Zeros(1), Zeros(1), Zeros(1));
			}
			else if (comp == 1)
			{
				vecGrad = MakeVector4(Zeros(1), grad, Zeros(1), Zeros(1));
			}
			else if (comp == 2)
			{
				vecGrad = MakeVector4(Zeros(1), Zeros(1), grad, Zeros(1));
			}
			else if (comp == 3)
			{
				vecGrad = MakeVector4(Zeros(1), Zeros(1), Zeros(1), grad);
			}
		}

		upperGradientsMap.insert({ param.ptr.get(), vecGrad.ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff = param->ForwardDiff(dx, elementLinearIdx);

		Expr ret = GetComponent(localDiff, comp);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		param->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		string paramVar = param->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		exprStr += "auto " + varName + " = ";
		exprStr += paramVar;
		switch (comp)
		{
		case 0: exprStr += ".x"; break;
		case 1: exprStr += ".y"; break;
		case 2: exprStr += ".z"; break;
		case 3: exprStr += ".w"; break;
		}
		exprStr += ";";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		param->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		Shape ret = oldParams.GetShape();
		ret.SetVectorType(VecType::Scalar1);

		return ret;
	}

	Type GetType() const
	{
		return param->GetType();
	}
};

inline Expr GetComponent(const Expr& param, const int axis)
{
	return make_shared<ComponentExp>(param, axis);
}

inline Expr X(const Expr& param)
{
	return make_shared<ComponentExp>(param, 0);
}

inline Expr Y(const Expr& param)
{
	return make_shared<ComponentExp>(param, 1);
}

inline Expr Z(const Expr& param)
{
	return make_shared<ComponentExp>(param, 2);
}

inline Expr W(const Expr& param)
{
	return make_shared<ComponentExp>(param, 3);
}

struct VectorExp : public Exp
{
	Expr param[4];
	const int axis;
	const int vectorSize;
	TensorParams oldParams;
	TensorParams newParams;

	VectorExp(const Expr& _param1, const int _axis)
		: param{ _param1 }
		, axis(_axis)
		, vectorSize(1)
	{
		oldParams.Resize(param[0]->GetShape());

		Assertf(oldParams.VectorSize() == 1, "Only support creating vectors from a single scalar.");
		Assertf(vectorSize <= 4, "Vector size has to be less than 4 since there are only 4 components in 4D vectors.");

		mShape = DeriveShape();
		mType = param[0]->GetType();

		newParams.Resize(mShape);

		mbRequiresGrad = param[0]->mbRequiresGrad;
	}

	VectorExp(const Expr& _param1, const Expr& _param2, const int _axis)
		: param{ _param1, _param2 }
		, axis(_axis)
		, vectorSize(2)
	{
		oldParams.Resize(BroadcastShape(param[0]->GetShape(), param[1]->GetShape()));

		Assertf(oldParams.VectorSize() == 1, "Only support creating vectors from a single scalar.");
		Assertf(vectorSize <= 4, "Vector size has to be less than 4 since there are only 4 components in 4D vectors.");

		mShape = DeriveShape();
		mType = param[0]->GetType();
		newParams.Resize(mShape);

		mbRequiresGrad = param[0]->mbRequiresGrad || param[1]->mbRequiresGrad;
	}

	VectorExp(const Expr& _param1, const Expr& _param2, const Expr& _param3, const int _axis)
		: param{ _param1, _param2, _param3 }
		, axis(_axis)
		, vectorSize(3)
	{
		oldParams.Resize(BroadcastShape(param[0]->GetShape(), BroadcastShape(param[1]->GetShape(), param[2]->GetShape())));

		Assertf(oldParams.VectorSize() == 1, "Only support creating vectors from a single scalar.");
		Assertf(vectorSize <= 4, "Vector size has to be less than 4 since there are only 4 components in 4D vectors.");

		mShape = DeriveShape();
		mType = param[0]->GetType();
		newParams.Resize(mShape);

		mbRequiresGrad = param[0]->mbRequiresGrad || param[1]->mbRequiresGrad || param[2]->mbRequiresGrad;
	}

	VectorExp(const Expr& _param1, const Expr& _param2, const Expr& _param3, const Expr& _param4, const int _axis)
		: param{ _param1, _param2, _param3, _param4 }
		, axis(_axis)
		, vectorSize(4)
	{
		oldParams.Resize(BroadcastShape(param[0]->GetShape(), BroadcastShape(param[1]->GetShape(), BroadcastShape(param[2]->GetShape(), param[3]->GetShape()))));

		Assertf(oldParams.VectorSize() == 1, "Only support creating vectors from a single scalar.");
		Assertf(vectorSize <= 4, "Vector size has to be less than 4 since there are only 4 components in 4D vectors.");

		mShape = DeriveShape();
		mType = param[0]->GetType();
		newParams.Resize(mShape);

		mbRequiresGrad = param[0]->mbRequiresGrad || param[1]->mbRequiresGrad || param[2]->mbRequiresGrad || param[3]->mbRequiresGrad;
	}


	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		for (int i = 0; i < vectorSize; i++)
		{
			if (param[i]->mbRequiresGrad)
			{
				Expr compGrad = GetComponent(grad, i);

				upperGradientsMap.insert({ param[i].ptr.get(), compGrad.ptr });
			}
		}
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr ret;
		if (vectorSize == 1)
		{
			Expr localDiff0 = param[0]->ForwardDiff(dx, elementLinearIdx);
			ret = MakeVector1(localDiff0, axis);
		}
		else if (vectorSize == 2)
		{
			Expr localDiff0 = param[0]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff1 = param[1]->ForwardDiff(dx, elementLinearIdx);
			ret = MakeVector2(localDiff0, localDiff1, axis);
		}
		else if (vectorSize == 3)
		{
			Expr localDiff0 = param[0]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff1 = param[1]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff2 = param[2]->ForwardDiff(dx, elementLinearIdx);
			ret = MakeVector3(localDiff0, localDiff1, localDiff2, axis);
		}
		else if (vectorSize == 4)
		{
			Expr localDiff0 = param[0]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff1 = param[1]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff2 = param[2]->ForwardDiff(dx, elementLinearIdx);
			Expr localDiff3 = param[3]->ForwardDiff(dx, elementLinearIdx);
			ret = MakeVector4(localDiff0, localDiff1, localDiff2, localDiff3, axis);
		}
		else
		{
			AssertNoEntry();
			ret = Zeros(1);
		}

		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;

		for (int i = 0; i < vectorSize; i++)
		{
			param[i]->RecursiveProcess(context, bForceRecompute);
		}

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		string paramVar[4];
		for (int i = 0; i < vectorSize; i++)
		{
			paramVar[i] = param[i]->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		if (vectorSize == 1)
		{
			exprStr += "auto " + varName + " = " + paramVar[0] + ";";
		}
		else if (vectorSize > 1)
		{
			exprStr += "float" + to_string(vectorSize) + " " + varName + " = " + "make_float" + to_string(vectorSize) + "(";

			exprStr += paramVar[0];
			for (int i = 1; i < vectorSize; i++)
			{
				exprStr += ", " + paramVar[i];
			}
		}
		exprStr += ");";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		for (int i = 0; i < vectorSize; i++)
			param[i]->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		Shape ret = oldParams.GetShape();
		ret.SetVectorType(vectorSize);

		return ret;
	}
};

inline Expr MakeVector1(const Expr& _param1, const int _axis)
{
	return make_shared<VectorExp>(_param1, _axis);
}

inline Expr MakeVector2(const Expr& _param1, const Expr& _param2, const int _axis)
{
	return make_shared<VectorExp>(_param1, _param2, _axis);
}

inline Expr MakeVector3(const Expr& _param1, const Expr& _param2, const Expr& _param3, const int _axis)
{
	return make_shared<VectorExp>(_param1, _param2, _param3, _axis);
}

inline Expr MakeVector4(const Expr& _param1, const Expr& _param2, const Expr& _param3, const Expr& _param4, const int _axis)
{
	return make_shared<VectorExp>(_param1, _param2, _param3, _param4, _axis);
}


struct TenaryExp : public Exp
{
	Expr cond;
	Expr paramT;
	Expr paramF;

	TenaryExp(const Expr& c, const Expr& parT, const Expr& parF)
		: cond(c)
		, paramT(parT)
		, paramF(parF)
	{
		mbRequiresGrad = paramT->mbRequiresGrad || paramF->mbRequiresGrad;

		mShape = DeriveShape();
		mType = DeriveType(paramT->GetType(), paramF->GetType());
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		upperGradientsMap.insert({ cond.ptr.get(), Zeros(cond->GetShape()).ptr });
		upperGradientsMap.insert({ paramT.ptr.get(), Unbroadcast(Where(cond, grad, Zeros(grad->GetShape())), paramT->GetShape()).ptr });
		upperGradientsMap.insert({ paramF.ptr.get(), Unbroadcast(Where(cond, Zeros(grad->GetShape()), grad), paramF->GetShape()).ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiffT = paramT->ForwardDiff(dx, elementLinearIdx);
		Expr localDiffF = paramF->ForwardDiff(dx, elementLinearIdx);

		Expr ret = Where(cond, localDiffT, localDiffF);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		// TODO: Make sure broadcasting works correctly
		Shape shape = BroadcastShape(paramT->GetShape(), paramF->GetShape());
		shape = BroadcastShape(shape, cond->GetShape());

		return shape;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;


		cond->RecursiveProcess(context, bForceRecompute);
		paramT->RecursiveProcess(context, bForceRecompute);
		paramF->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		string condVarName = cond->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		string trueVarName = paramT->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		string falseVarName = paramF->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		NewLine(exprStr);
		Indent(exprStr, indentLevels); exprStr += "auto " + varName + " = where(" + condVarName + ", " + trueVarName + ", " + falseVarName + ");\n";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		cond->TopologicalSort(sorted);
		paramT->TopologicalSort(sorted);
		paramF->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};

inline Expr Where(const Expr& cond, const Expr& t, const Expr& f)
{
	return make_shared<TenaryExp>(cond, t, f);
}

struct IndexMask
{
	Tensor<int> mask;
	Tensor<int> index;
	int sum;

	IndexMask() = default;

	IndexMask(const Tensori& _mask)
	{
		Init(_mask);
	}

	void Init(const Tensori& _mask)
	{
		mask = _mask.Reshape(_mask.LinearSize());
		sum = Tensori::Sum(mask).Get(0);
		if (sum > 0)
		{
			Tensor<int> offset = Tensori::ExclusiveScan(mask);
			Tensori::MaskedSelectionIndex(mask.LinearSize(), mask, offset, sum, &index);
		}
	}
};

struct IndexedReadExp : public Exp
{
	Expr operand;
	Expr indices;
	int axis;
	bool keepDim;

	IndexedReadExp(const Expr& op, const Expr& _indices, const int _axis, const bool _keepDim = true)
		: operand(op)
		, indices(_indices)
		, axis(_axis)
		, keepDim(_keepDim)
	{
		mbRequiresGrad = operand->mbRequiresGrad;
		mShape = DeriveShape();
		mType = operand->GetType();

		int opDim = operand->GetShape().Size();
		Assertf(axis < opDim, "Axis must be smaller than the dimension of the operator.");
		Assertf(indices->GetShape().mSize == 1, "Indices of IndexedRead can only have 1 dimension");
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		upperGradientsMap.insert({ operand.ptr.get(), IndexedWrite(grad, Tensori(indices), operand->GetShape(), axis).ptr });
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff = operand->ForwardDiff(dx, elementLinearIdx);

		Expr ret = IndexedRead(localDiff, indices, axis, keepDim);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	TENSOR_INLINE Shape DeriveShape() const
	{
		Shape valShape = operand->GetShape();
		Shape retShape = valShape;
		int numIndices = indices->GetShape().LinearSize();
		if (numIndices > 1)
		{
			retShape[axis] = indices->GetShape().LinearSize();
		}
		else if (keepDim)
		{
			retShape[axis] = indices->GetShape().LinearSize();
		}
		else
		{
			retShape[axis] = 0;
			retShape.Resize(retShape.Size() - 1);
		}

		return retShape;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
		{
			return;
		}

		value.ptr = nullptr;
		mValueCached = false;

		operand->RecursiveProcess(context, bForceRecompute);
		indices->RecursiveProcess(context, bForceRecompute);

		//if (mType == Type::Bool || mType == Type::Int)
		//	value = Tensori();
		//else if (mType == Type::Uint)
		//	value = Tensorui();
		//else if (mType == Type::Float)
		//	value = Tensorf();
		//else if (mType == Type::Double)
		//	value = Tensord();

		//value->GenerateAndLaunchCUDAKernel(this);
		//mValueCached = true;

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		if (mValueCached)
		{
			Assert(value && !value->Empty());
			return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
		}

		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		IndexedReadArg arg = ToJit();
		int id;
		auto it = variableMap.indexedReadArgs.find(arg);
		if (it == variableMap.indexedReadArgs.end())
		{
			id = variableMap.indexedReadArgs.size();
			variableMap.indexedReadArgs.insert({ arg, id });
		}
		else
		{
			id = it->second;
		}


		string indexVar;
		auto cachedIndexVar = VariableCache::GetHandle().find(std::make_tuple((const Exp*)(size_t(this) + 1), indexName, paramsName));
		if (cachedIndexVar != VariableCache::GetHandle().end())
		{
			indexVar = "var" + to_string(cachedIndexVar->second);
		}
		else
		{
			int indexVarId = VariableCache::GetHandle().size();
			// Use this + 1 to cache the index variable
			VariableCache::GetHandle().insert({ std::make_tuple((const Exp*)(size_t(this) + 1), indexName, paramsName), indexVarId });

			NewLine(exprStr);
			Indent(exprStr, indentLevels);
			exprStr += "int var" + to_string(indexVarId) + " = indexedRead" + to_string(id) + ".GetAxisIndex(" + indexName + ", " + paramsName + ");\n";
			indexVar = "var" + to_string(indexVarId);
		}

		string indexVar2 = indices->EmitCuda(variableMap, indexVar, "indexedRead" + to_string(id) + ".indicesParams", indices->GetShape(), exprStr, indentLevels, bForceRecompute);

		string linearIndexVar;
		auto cachedLinearIndexVar = VariableCache::GetHandle().find(std::make_tuple((const Exp*)(size_t(this) + 2), indexName, paramsName));
		if (cachedLinearIndexVar != VariableCache::GetHandle().end())
		{
			linearIndexVar = "var" + to_string(cachedLinearIndexVar->second);
		}
		else
		{
			int linearIndexVarId = VariableCache::GetHandle().size();
			// Use this + 2 to cache the index variable
			VariableCache::GetHandle().insert({ std::make_tuple((const Exp*)(size_t(this) + 2), indexName, paramsName), linearIndexVarId });

			NewLine(exprStr);
			Indent(exprStr, indentLevels);
			exprStr += "int var" + to_string(linearIndexVarId) + " = indexedRead" + to_string(id) + ".ConvertLinearIndex(" + indexName + ", " + indexVar2 + ", " + paramsName + ");\n";
			linearIndexVar = "var" + to_string(linearIndexVarId);
		}

		string opVar = operand->EmitCuda(variableMap, linearIndexVar, "indexedRead" + to_string(id) + ".opParams", operand->GetShape(), exprStr, indentLevels, bForceRecompute);

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels);
		exprStr += "auto " + varName + " = ";
		exprStr += opVar;
		exprStr += ";";

		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		operand->TopologicalSort(sorted);

		sorted.push_back(this);
	}

	IndexedReadArg ToJit() const
	{
		IndexedReadArg ret;


		ret.pIndices = (void*)indices.ptr.get();
		TensorParams params;
		params.Resize(operand->GetShape());
		ret.opParams = params.ToJit();
		
		params.Resize(indices->GetShape());
		ret.indicesParams = params.ToJit();
		ret.axis = axis;

		return ret;
	}
};

inline Expr IndexedRead(const Expr& x, const Expr& indices, const int axis, const bool keepDim)
{
	return make_shared<IndexedReadExp>(x, indices, axis, keepDim);
}

inline Expr IndexedRead(const Expr& x, const Tensori& indices, const int axis, const bool keepDim)
{
	return make_shared<IndexedReadExp>(x, indices, axis, keepDim);
}

inline Expr Mask(const Expr& x, const IndexMask& mask, const int axis)
{
	Assertf(x->GetShape()[axis] == mask.mask.GetShape(0), "Tensor size doesn't match mask size before reduction!");
	return make_shared<IndexedReadExp>(x, mask.index, axis, true);
}

struct IndexedWriteExp : public Exp
{
	Expr operand;
	Tensori indices;
	int axis;
	bool dimIncrease;

	IndexedWriteExp(const Expr& op, const Tensori& _indices, const Shape& _shape, const int _axis)
		: operand(op)
		, indices(_indices)
		, axis(_axis)
	{
		indices = indices.Reshape(indices.LinearSize());

		mShape = _shape;
		mType = operand->GetType();

		Shape operandShape = operand->GetShape();

		Assertf(indices.LinearSize() == operandShape[axis], "Invalid index size in IndexedWrite!");

		dimIncrease = mShape.Size() > operandShape.Size();

		mbRequiresGrad = operand->mbRequiresGrad;
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
		if (!mbRequiresGrad)
			return;

		Shape unboardcastedShape = operand->GetShape();
		unboardcastedShape[axis] = mShape[axis];
		upperGradientsMap.insert({ operand.ptr.get(), IndexedRead(Unbroadcast(grad, unboardcastedShape), indices, axis, !dimIncrease).ptr });
	}


	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		auto cached = ForwardDiffVariableCache::GetHandle().find(this);
		if (cached != ForwardDiffVariableCache::GetHandle().end())
		{
			return cached->second;
		}

		Expr localDiff = operand->ForwardDiff(dx, elementLinearIdx);

		Expr ret = IndexedWrite(localDiff, indices, mShape, axis);
		ForwardDiffVariableCache::GetHandle().insert({ this, ret });

		return ret;
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
		{
			Assert(value && !value->Empty());
			return;
		}

		value.ptr = nullptr;
		mValueCached = false;

		if (mType == Type::Bool || mType == Type::Int)
			value = Tensori();
		else if (mType == Type::Uint)
			value = Tensorui();
		else if (mType == Type::Float)
			value = Tensorf();
		else if (mType == Type::Double)
			value = Tensord();
		value->Resize(mShape);

		operand->RecursiveProcess(context, bForceRecompute);

		if (mType == Type::Bool || mType == Type::Int)
		{
			Tensori tempVal;
			tempVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
			operand->value = tempVal;
			operand->mValueCached = true;

			auto* pTensor = dynamic_cast<Tensori*>(value.ptr.get());
			Tensori::IndexedWrite(*pTensor, tempVal, indices, axis);
		}
		else if (mType == Type::Uint)
		{
			Tensorui tempVal;
			tempVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
			operand->value = tempVal;
			operand->mValueCached = true;

			auto* pTensor = dynamic_cast<Tensorui*>(value.ptr.get());
			Tensorui::IndexedWrite(*pTensor, tempVal, indices, axis);
		}
		else if (mType == Type::Float)
		{
			Tensorf tempVal;
			tempVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
			operand->value = tempVal;
			operand->mValueCached = true;

			auto* pTensor = dynamic_cast<Tensorf*>(value.ptr.get());
			Tensorf::IndexedWrite(*pTensor, tempVal, indices, axis);
		}
		else if (mType == Type::Double)
		{
			Tensord tempVal;
			tempVal.GenerateAndLaunchCUDAKernel(operand.ptr.get());
			operand->value = tempVal;
			operand->mValueCached = true;

			auto* pTensor = dynamic_cast<Tensord*>(value.ptr.get());
			Tensord::IndexedWrite(*pTensor, tempVal, indices, axis);

		}
		else
			AssertNoEntry();

		mValueCached = true;
		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		Assert(value && !value->Empty());
		return value->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		operand->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};

inline Expr IndexedWrite(const Expr& x, const Tensori& _indices, const Shape& _shape, const int _axis)
{
	return make_shared<IndexedWriteExp>(x, _indices, _shape, _axis);
}

struct DetachExp : public Exp
{
	Expr param;

	DetachExp(const Expr& _param)
		: param(_param)
	{
		mShape = param->GetShape();
		mType = param->GetType();
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		return Zeros(mShape);
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
		if (GlobalVisitedSet::Visited(this))
			return;

		value.ptr = nullptr;
		mValueCached = false;
		
		param->RecursiveProcess(context, bForceRecompute);

		GlobalVisitedSet::SetVisited(this);
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		return param->EmitCuda(variableMap, indexName, paramsName, broadcastShape, exprStr, indentLevels, bForceRecompute);
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		param->TopologicalSort(sorted);

		sorted.push_back(this);
	}
};


inline auto Detach(const Expr& param)
{
	return make_shared<DetachExp>(param);
}

struct PixelCoordExp : public Exp
{
	int width, height;
	int batch;
	TensorParams params;

	PixelCoordExp(int w, int h, int b = 1)
		: width(w)
		, height(h)
		, batch(b)
	{
		params.Resize(Shape({ height * width * batch }, VecType::Vec4));
		mShape = params.GetShape();
		mType = Type::Int;
	}

	virtual void Backward(const Expr& grad, multimap<const Exp*, shared_ptr<Exp>>& upperGradientsMap) const override
	{
	}

	virtual Expr ForwardDiff(const void* dx, const int elementLinearIdx = -1) const
	{
		return Zeros(1);
	}

	virtual void RecursiveProcess(GraphProcessContext& context, const bool bForceRecompute = false) override
	{
	}

	virtual string EmitCuda(VariableMap& variableMap, const string& indexName, const string& paramsName, const Shape& broadcastShape, string& exprStr, const int indentLevels = 0, const bool bForceRecompute = false) override
	{
		auto cachedVar = VariableCache::GetHandle().find(std::make_tuple(this, indexName, paramsName));
		if (cachedVar != VariableCache::GetHandle().end())
		{
			return "var" + to_string(cachedVar->second);
		}

		int varId = VariableCache::GetHandle().size();
		VariableCache::GetHandle().insert({ std::make_tuple(this, indexName, paramsName), varId });

		string varName = "var" + to_string(varId);

		NewLine(exprStr);
		Indent(exprStr, indentLevels); exprStr += "float4 " + varName + " = PixelCoord(" + indexName + ", " + to_string(width) + ", " + to_string(height) + ");\n";
		return varName;
	}

	virtual void TopologicalSort(vector<const Exp*>& sorted) const override
	{
		if (!mbRequiresGrad || VisitedSet::GetHandle().find(this) != VisitedSet::GetHandle().end())
			return;

		VisitedSet::GetHandle().insert(this);

		sorted.push_back(this);
	}
};