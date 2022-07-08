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

#include "Examples/Example.h"
#include "Examples/Validations/ValidationExamples.h"
#include "Renderer/SceneLoader.h"
#include "Renderer/pybind_utils.h"
#include "Tensor/Tensor.h"
#include "Renderer/ParticleTracer.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(TensorRay, m) {
	m.doc() = "Path-space differentiable renderer: TensorRay";
	m.def("env_create", &EnvCreate);
	m.def("env_release", &EnvRelease);
	m.def("to_tensor", &toTensor);
	m.def("torch_to_tensor", &AssignTorchToTensor);
	m.def("tensor_to_torch", &AssignTensorToTorch);
	m.def("detach", &DetachTensor);
	m.def("get_num_param", &GetNumParam);
	m.def("get_size_param", &GetParamSize);
	m.def("get_var", &GetVariable, "index"_a, "ptr_data"_a, "offset"_a = 0);
	m.def("set_var", &SetVariable, "index"_a, "ptr_data"_a, "offset"_a = 0);
	m.def("get_grad", &GetGradient, "index"_a, "ptr_data"_a, "offset"_a = 0, "clear_grad"_a = true);
	m.def("set_rnd_seed", &SetRandomSeed);

	m.def("debug_tensor", &DebugTensor);
	m.def("debug_image", &DebugImage);

	py::class_<ptr_wrapper<float>>(m, "float_ptr")
		.def(py::init<std::size_t>());
	py::class_<ptr_wrapper<int>>(m, "int_ptr")
		.def(py::init<std::size_t>());

	py::class_<Vector3>(m, "Vector3f")
		.def(py::init<float, float, float>());

	py::class_<Expr>(m, "Expr")
		.def(py::init([](const Tensorf& t) { return Expr(t); }));

	py::class_<Tensorf>(m, "Tensorf")
		.def("resize", static_cast<void (Tensorf::*)(const Shape&)>(&Tensorf::Resize))
		.def("size", &Tensorf::LinearSize)
		.def("requires_grad", &Tensorf::RequiresGrad)
		.def("set_requires_grad", &Tensorf::SetRequiresGrad)
		.def("backward", py::overload_cast<>(&Tensorf::Backward, py::const_))
		.def("clear", &Tensorf::Clear)
		.def("clear_grad", &Tensorf::ClearGrad)
		.def("value_to_torch", &Tensorf::ValueToTorch)
		.def("grad_to_torch", &Tensorf::GradToTorch)
		.def(py::init<>())
		.def(py::init([](float val, bool requiresGrad) { return Tensorf({ val }, requiresGrad); }), "val"_a = 0, "requiresGrad"_a = false)
		.def(py::init([](float x, float y, float z, bool requiresGrad = false) { return Tensorf({ Vector3(x, y, z) }, requiresGrad); }), "x"_a = 0, "y"_a = 0, "z"_a = 0, "requiresGrad"_a = false)
		.def(py::init([](float m00, float m01, float m02, float m03,
			float m10, float m11, float m12, float m13,
			float m20, float m21, float m22, float m23,
			float m30, float m31, float m32, float m33) { 
				return Tensorf({ 
					Vector4(m00, m10, m20, m30), 
					Vector4(m01, m11, m21, m31), 
					Vector4(m02, m12, m22, m32),
					Vector4(m03, m13, m23, m33)}, false); }))
		.def(py::init([](ptr_wrapper<float> ptr, const Shape& shape) { return toTensor(ptr, shape); }))
		.def(py::self + py::self)
		.def(py::self * py::self)
		.def(py::self - py::self)
		.def(py::self / py::self)
		.def(-py::self)
		.def("eval_expr", &Tensorf::EvalExpr);

	py::enum_<VecType>(m, "VecType")
		.value("Scalar1", VecType::Scalar1)
		.value("Vec2", VecType::Vec2)
		.value("Vec3", VecType::Vec3)
		.value("Vec4", VecType::Vec4)
		.value("Mat4x4", VecType::Mat4x4)
		.export_values();

	py::class_<Shape>(m, "TensorShape")
		.def(py::init<std::vector<int>, int>())
		.def("size", &Shape::Size);

	py::class_<Primitive, std::shared_ptr<Primitive>>(m, "Shape")
		.def_readwrite("obj_center", &Primitive::mObjCenter)        // For multi-pose optimization
		.def_readwrite("rotate_matrix", &Primitive::mRotateMatrix)  // For multi-pose optimization
		.def_readwrite("translation", &Primitive::mTrans)
		.def_readwrite("rotate_axis", &Primitive::mRotateAxis)
		.def_readwrite("rotate_angle", &Primitive::mRotateAngle)
		.def_readwrite("vertex_id", &Primitive::mVertexId)
		.def_readwrite("vertex_translation", &Primitive::mVertexTrans)
		.def("diff_translation", &Primitive::DiffTranslation)
		.def("diff_rotation", &Primitive::DiffRotation, "rotate_axis"_a)
		.def("diff_all_vertex_pos", &Primitive::DiffAllVertexPos)
		.def("get_vertex_count", &Primitive::GetVertexCount)
		.def("get_face_count", &Primitive::GetFaceCount)
		.def("get_edge_count", &Primitive::GetEdgeCount)
		.def("get_face_indices", &Primitive::GetFaceIndices)
		.def("get_edge_data", &Primitive::GetEdgeData)
		.def("get_obj_center", &Primitive::GetObjCenter)
		.def("export_mesh", &Primitive::ExportMesh, "filename"_a);

	py::class_<BSDF, std::shared_ptr<BSDF>>(m, "BSDF")
		.def("diff_texture", &BSDF::DiffTexture, "name"_a = std::string());

	py::class_<Light, std::shared_ptr<Light>>(m, "Light");

	py::class_<TensorRay::Camera, std::shared_ptr<TensorRay::Camera>>(m, "Camera")
		.def(py::init<>())
		.def(py::init<const Vector3&, const Vector3&, const Vector3&, int, int, float>())
		.def_readonly("width", &TensorRay::Camera::mResX)
		.def_readonly("height", &TensorRay::Camera::mResY)
		.def_readonly("posTensor", &TensorRay::Camera::mPosTensor)
		.def("init", &TensorRay::Camera::Init)
		.def("update", &TensorRay::Camera::Update)
		.def("resize", &TensorRay::Camera::Resize);

	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def("load_file", &Scene::LoadFromFile, "filename"_a)
		.def("configure", &Scene::Configure)
		.def("get_width", &Scene::GetImageWidth)
		.def("get_height", &Scene::GetImageHeight)
		.def_readonly("num_bsdf", &Scene::mBSDFCount)
		.def_readwrite("shapes", &Scene::mPrims)
		.def_readwrite("bsdfs", &Scene::mBsdfs)
		.def_readwrite("emitters", &Scene::mLights)
		.def_readwrite("cameras", &Scene::mSensors);

	py::class_<RenderOptions>(m, "RenderOptions")
		.def(py::init<int, int, int, int, int, int, int>())
		.def_readwrite("seed", &RenderOptions::mRndSeed)
		.def_readwrite("max_bounces", &RenderOptions::mMaxBounces)
		.def_readwrite("spp", &RenderOptions::mSppInterior)
		.def_readwrite("spp_batch", &RenderOptions::mSppInteriorBatch)
		.def_readwrite("sppe", &RenderOptions::mSppPrimary)
		.def_readwrite("sppe_batch", &RenderOptions::mSppPrimaryBatch)
		.def_readwrite("sppse0", &RenderOptions::mSppDirect)
		.def_readwrite("sppse0_batch", &RenderOptions::mSppDirectBatch)
		.def_readwrite("sppse1", &RenderOptions::mSppIndirect)
		.def_readwrite("sppse1_batch", &RenderOptions::mSppIndirectBatch)
		.def_readwrite("sppe0", &RenderOptions::mSppPixelBoundary)
		.def_readwrite("sppe0_batch", &RenderOptions::mSppPixelBoundaryBatch)
		.def_readwrite("export_deriv", &RenderOptions::mExportDerivative)
		.def_readwrite("quiet", &RenderOptions::mQuiet)

		// guiding options
		.def_readwrite("g_direct", &RenderOptions::g_direct)
		.def_readwrite("g_direct_depth", &RenderOptions::g_direct_depth)
		.def_readwrite("g_direct_max_size", &RenderOptions::g_direct_max_size)
		.def_readwrite("g_direct_spp", &RenderOptions::g_direct_spp)
		.def_readwrite("g_direct_thold", &RenderOptions::g_direct_thold)
		.def_readwrite("g_eps", &RenderOptions::g_eps);

	py::class_<Integrator>(m, "Integrator");

	py::class_<PathTracer, Integrator>(m, "PathTracer")
		.def(py::init<>())
		.def("set_param", &PathTracer::SetParam)
		.def("renderC", &PathTracer::RenderC)
		.def("renderD", &PathTracer::RenderD);

	py::class_<ParticleTracer, Integrator>(m, "ParticleTracer")
		.def(py::init<>())
		.def("set_param", &ParticleTracer::SetParam)
		.def("renderC", &ParticleTracer::RenderC);

	py::class_<PrimaryBoundaryIntegrator, Integrator>(m, "PrimaryEdgeIntegrator")
		.def(py::init<>())
		.def("set_param", &PrimaryBoundaryIntegrator::SetParam)
		.def("renderD", &PrimaryBoundaryIntegrator::RenderD);

	py::class_<DirectBoundaryIntegrator, Integrator>(m, "DirectEdgeIntegrator")
		.def(py::init<>())
		.def("set_param", &DirectBoundaryIntegrator::SetParam)
		.def("renderD", &DirectBoundaryIntegrator::RenderD);

	py::class_<IndirectBoundaryIntegrator, Integrator>(m, "IndirectEdgeIntegrator")
		.def(py::init<>())
		.def("set_param", &IndirectBoundaryIntegrator::SetParam)
		.def("renderD", &IndirectBoundaryIntegrator::RenderD);

	py::class_<PixelBoundaryIntegrator, Integrator>(m, "PixelBoundaryIntegrator")
		.def(py::init<>())
		.def("set_param", &PixelBoundaryIntegrator::SetParam)
		.def("renderD", &PixelBoundaryIntegrator::RenderD);
}
