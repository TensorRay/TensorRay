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

#include "SceneLoader.h"
#include "Scene.h"
#include "pugixml/pugixml.hpp"
#include "Diffuse.h"
#include "Roughconductor.h"
#include "Microfacet.h"
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <array>

namespace EDX
{
	namespace TensorRay
	{
        template <int length>
        Vec<length, float> parse_vector(const char* str, bool allow_empty = false) 
        {
            Vec<length, float> result(0.0f);
            int tot = 0;
            for (int i = 0; ; ) 
            {
                while (str[i] && strchr(", ", str[i])) ++i;
                if (!str[i]) break;
                int j = i + 1;
                while (str[j] && strchr(", ", str[j]) == nullptr) ++j;
                Assert(tot < length);
                result[tot++] = (static_cast<float>(atof(str + i)));
                i = j;
            }

            if (tot < length) 
            {
                if (allow_empty) 
                {
                    float value = tot ? result[tot - 1] : 0.0f;
                    for (int k = tot; k < length; k++)
                        result[k] = value;
                }
                else 
                {
                    Assert(false);
                }
            }

            return result;
        }

        inline static std::string parse_bitmap(const pugi::xml_node& node) 
        {
            const char* texture_type = node.attribute("type").value();
            TENSORRAY_ASSERT_MSG(strcmp(texture_type, "bitmap") == 0, std::string("Unsupported texture type: ") + texture_type);
            const pugi::xml_node& fn_node = node.child("string");
            const char* tmp = fn_node.attribute("name").value(), * file_name = fn_node.attribute("value").value();
            TENSORRAY_ASSERT_MSG(strcmp(tmp, "filename") == 0 && file_name, "Failed to retrieve bitmap filename");
            return std::string(file_name);
        }

        static Vector3 load_rgb(const pugi::xml_node& node) 
        {
            if (strcmp(node.name(), "float") == 0) 
            {
                return Vector3(node.attribute("value").as_float());
            }
            else if (strcmp(node.name(), "rgb") == 0 || strcmp(node.name(), "spectrum") == 0) 
            {
                return parse_vector<3>(node.attribute("value").value(), true);
            }
            else 
            {
                TENSORRAY_ASSERT_MSG(false, std::string("Unsupported RGB type: ") + node.name());
            }
        }

        inline static pugi::xml_node find_child_by_name(const pugi::xml_node& parent,
            const std::unordered_set<std::string>& names,
            bool allow_empty = false) 
        {
            TENSORRAY_ASSERT(!names.empty());
            pugi::xml_node result = parent.find_child(
                [&](pugi::xml_node node) {
                    return names.find(node.attribute("name").value()) != names.end();
                }
            );
            TENSORRAY_ASSERT_MSG(allow_empty || result, std::string("Missing child node: ") + *names.begin());
            return result;
        }

        static std::pair<int, int> load_film(const pugi::xml_node& node) 
        {
            pugi::xml_node child;
            child = find_child_by_name(node, { "width" });
            int width = child.attribute("value").as_int();
            child = find_child_by_name(node, { "height" });
            int height = child.attribute("value").as_int();
            return { width, height };
        }

        void load_transform(const pugi::xml_node& parent, Expr& M, Expr& invM, Matrix m = Matrix::IDENTITY) 
        {
            if (parent) 
            {
                const char* node_name = parent.attribute("name").value();
                TENSORRAY_ASSERT_MSG(strcmp(node_name, "to_world") == 0 || strcmp(node_name, "toWorld") == 0,
                                     std::string("Invalid transformation name: ") + node_name);
                for (auto node = parent.first_child(); node; node = node.next_sibling()) 
                {
                    if (strcmp(node.name(), "translate") == 0) 
                    {
                        Vector3 offset(
                            node.attribute("x").as_float(0.0f),
                            node.attribute("y").as_float(0.0f),
                            node.attribute("z").as_float(0.0f)
                        );
                        m = Matrix::Mul(Matrix::Translate(offset), m);
                    }
                    else if (strcmp(node.name(), "rotate") == 0) 
                    {
                        Vector3 axis(
                            node.attribute("x").as_float(),
                            node.attribute("y").as_float(),
                            node.attribute("z").as_float()
                        );
                        float angle = node.attribute("angle").as_float();
                        m = Matrix::Mul(Matrix::Rotate(angle, axis), m);
                    }
                    else if (strcmp(node.name(), "scale") == 0) 
                    {
                        Vector3 scl(
                            node.attribute("x").as_float(1.0f),
                            node.attribute("y").as_float(1.0f),
                            node.attribute("z").as_float(1.0f)
                        );
                        m = Matrix::Mul(Matrix::Scale(scl.x, scl.y, scl.z), m);
                    }
                    else 
                    {
                        TENSORRAY_ASSERT_MSG(false, std::string("Unsupported transformation: ") + node.name());
                    }
                }
            }
            M = MatrixToTensor(m);
            invM = MatrixToTensor(Matrix::Inverse(m));
        }

        void SceneLoader::LoadFromFile(const char* filename, Scene& scene)
        {
            pugi::xml_document doc;
            TENSORRAY_ASSERT_MSG(doc.load_file(filename), "XML parsing failed");
            LoadScene(doc, scene);
        }

        void SceneLoader::LoadScene(const pugi::xml_document& doc, Scene& scene) 
        {
            const pugi::xml_node& root = doc.child("scene");

            // Load BSDFs
            for (auto node = root.child("bsdf"); node; node = node.next_sibling("bsdf"))
                LoadBsdf(node, scene);

            // Load shapes
            for (auto node = root.child("shape"); node; node = node.next_sibling("shape"))
                LoadShape(node, scene);

            // Load Sensors
            for (auto node = root.child("sensor"); node; node = node.next_sibling("sensor"))
                LoadSensor(node, scene);
        }

        void SceneLoader::LoadBsdf(const pugi::xml_node& node, Scene& scene)
        {
            const char* bsdf_id = node.attribute("id").value();
            TENSORRAY_ASSERT_MSG(bsdf_id && strcmp(bsdf_id, ""), "BSDF must have an id");
            const char* bsdf_type = node.attribute("type").value();
            if (strcmp(bsdf_type, "diffuse") == 0) 
            {
                // Diffuse BSDF
                pugi::xml_node refl_node = find_child_by_name(node, { "reflectance" });
                if (strcmp(refl_node.name(), "texture") == 0)
                {
                    scene.AddBsdf<Diffuse>(new TensorRay::ImageTexture(parse_bitmap(refl_node).c_str()));
                    scene.mBsdfs.back()->mStrId = bsdf_id;
                }
                else
                {
                    Vector3 rgb = load_rgb(refl_node);
                    scene.AddBsdf<Diffuse>(new TensorRay::ConstantTexture(rgb));
                    scene.mBsdfs.back()->mStrId = bsdf_id;
                }
            }
            else if (strcmp(bsdf_type, "roughconductor") == 0) 
            {
                // Rough conductor BSDF
                float roughness;
                Vector3 eta, k;

                // Load rough conductor properties
                pugi::xml_node roughness_node = find_child_by_name(node, { "alpha" });
                if (strcmp(roughness_node.name(), "float") == 0) 
                {
                    roughness = roughness_node.attribute("value").as_float();
                }
                else
                {
                    // TODO: support roughness map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported roughness: ") + roughness_node.name());
                }

                pugi::xml_node eta_node = find_child_by_name(node, { "eta" });
                if (strcmp(eta_node.name(), "spectrum") == 0) 
                {
                    eta = load_rgb(eta_node);
                }
                else
                {
                    // TODO: support eta map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported eta: ") + eta_node.name());
                }
                
                pugi::xml_node k_node = find_child_by_name(node, { "k" });
                if (strcmp(k_node.name(), "spectrum") == 0) 
                {
                    k = load_rgb(k_node);
                }
                else
                {
                    // TODO: support k map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported k: ") + k_node.name());
                }

                // Modify this to support texture map
                scene.AddBsdf<RoughConductor>(new TensorRay::ConstantTexture(roughness), new TensorRay::ConstantTexture(eta), new TensorRay::ConstantTexture(k));
                scene.mBsdfs.back()->mStrId = bsdf_id;
            }
            else if (strcmp(bsdf_type, "microfacet") == 0) 
            {
                // Disney BSDF
                float roughness;
                Vector3 diffuse, specular;

                pugi::xml_node roughness_node = find_child_by_name(node, { "roughness" });
                if (strcmp(roughness_node.name(), "float") == 0) 
                {
                    roughness = roughness_node.attribute("value").as_float();
                }
                else
                {
                    // TODO: support roughness map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported roughness: ") + roughness_node.name());
                }

                pugi::xml_node diffuse_node = find_child_by_name(node, { "diffuseReflectance" });
                if (strcmp(diffuse_node.name(), "spectrum") == 0) 
                {
                    diffuse = load_rgb(diffuse_node);
                }
                else
                {
                    // TODO: support diffuse map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported diffuseReflectance: ") + diffuse_node.name());
                }

                pugi::xml_node specular_node = find_child_by_name(node, { "specularReflectance" });
                if (strcmp(specular_node.name(), "spectrum") == 0) 
                {
                    specular = load_rgb(specular_node);
                }
                else
                {
                    // TODO: support specular map
                    TENSORRAY_ASSERT_MSG(false, std::string("Unsupported specularReflectance: ") + specular_node.name());
                }

                // Modify this to support texture map
                scene.AddBsdf<Microfacet>(new TensorRay::ConstantTexture(diffuse), new TensorRay::ConstantTexture(specular), new TensorRay::ConstantTexture(roughness));
                scene.mBsdfs.back()->mStrId = bsdf_id;
            }
            else
            {
                TENSORRAY_ASSERT_MSG(false, std::string("Unsupported BSDF: ") + bsdf_type);
            }
        }

        void SceneLoader::LoadShape(const pugi::xml_node& node, Scene& scene)
        {
            const char* mesh_id = node.attribute("id").value();
            const char* shape_type = node.attribute("type").value();
            TENSORRAY_ASSERT_MSG(strcmp(shape_type, "obj") == 0, std::string("Unsupported shape: ") + shape_type);
            // Set BSDF
            std::unordered_map<std::string, int> bsdfMap;
            for (int i = 0; i < scene.mBsdfs.size(); i++)
                bsdfMap.insert(std::make_pair(scene.mBsdfs[i]->mStrId, scene.mBsdfs[i]->mId));
            const pugi::xml_node& ref_node = node.child("ref");
            TENSORRAY_ASSERT_MSG(ref_node, std::string("Missing BSDF reference"));
            const char* bsdf_id = ref_node.attribute("id").value();
            TENSORRAY_ASSERT(bsdf_id);
            auto search = bsdfMap.find(bsdf_id);
            TENSORRAY_ASSERT_MSG(search != bsdfMap.end(), std::string("Unknown BSDF id: ") + bsdf_id);
            int bsdf_index = search->second;

            if (strcmp(shape_type, "obj") == 0)
            {
                // Set transformation
                Expr toWorld, toWorldInv;
                load_transform(node.child("transform"), toWorld, toWorldInv);

                const pugi::xml_node& name_node = node.child("string");
                TENSORRAY_ASSERT(strcmp(name_node.attribute("name").value(), "filename") == 0);
                // File name
                const char* file_name = name_node.attribute("value").value();
                // Handle face normals
                bool use_face_normals = true;
                const pugi::xml_node& fn_node = find_child_by_name(node, { "face_normals", "faceNormals" }, true);
                if (fn_node)
                    use_face_normals = (strcmp(fn_node.attribute("value").value(), "true") == 0);
                // Append the mesh to the scene
                scene.AddPrimitive<Primitive>(file_name, *scene.mBsdfs[bsdf_index], toWorld, toWorldInv, use_face_normals);
            }
            //else if (strcmp(shape_type, "rectangle") == 0)
            //{
            //    // Set transformation
            //    Matrix initM = Matrix::Mul(Matrix::Scale(1.f, -1.f, 1.f), Matrix::Rotate(90.f, Vector3::UNIT_X));
            //    Expr toWorld, toWorldInv;
            //    load_transform(node.child("transform"), toWorld, toWorldInv, initM);
            //    // Append the mesh to the scene
            //    scene.AddPrimitive<Primitive>(2, *scene.mBsdfs[bsdf_index], toWorld, toWorldInv);
            //}
            else
            {
                TENSORRAY_ASSERT_MSG(false, std::string("Unsupported shape: ") + shape_type);
            }

            // Set emitter (if necessary)
            const pugi::xml_node& emitter_node = node.child("emitter");
            if (emitter_node)
            {
                const char* emitter_type = emitter_node.attribute("type").value();
                TENSORRAY_ASSERT_MSG(strcmp(emitter_type, "area") == 0, std::string("Unsupported emitter: ") + emitter_type);
                auto radiance = load_rgb(find_child_by_name(emitter_node, { "radiance" }));
                scene.mPrims.back()->mIsEmitter = true;
                if (scene.mAreaLightIndex == -1) 
                {
                    scene.mAreaLightIndex = scene.mLights.size();
                    scene.AddLight<AreaLight>(radiance, scene.mPrims.size() - 1);
                }
                else 
                {
                    Assert(scene.mAreaLightIndex == 0);
                    AreaLight* ptr_light = dynamic_cast<AreaLight*>(scene.mLights[scene.mAreaLightIndex].get());
                    ptr_light->Append(radiance, scene.mPrims.size() - 1);
                }
            }
        }

        void SceneLoader::LoadSensor(const pugi::xml_node& node, Scene& scene)
        {
            const char* sensor_type = node.attribute("type").value();
            if (strcmp(sensor_type, "perspective") == 0)
            {
                // Camera film size
                int width, height;
                const pugi::xml_node& film_node = node.child("film");
                std::tie(width, height) = load_film(film_node);
                // Camera pose
                const pugi::xml_node& transform_node = node.child("transform");
                TENSORRAY_ASSERT(transform_node);
                const pugi::xml_node& lookat_node = transform_node.child("lookat");
                TENSORRAY_ASSERT_MSG(lookat_node, std::string("LookAt transformation for sensor is missing!"));
                Vector3 origin = parse_vector<3>(lookat_node.attribute("origin").value());
                Vector3 target = parse_vector<3>(lookat_node.attribute("target").value());
                Vector3 up     = parse_vector<3>(lookat_node.attribute("up").value());
                // Fov
                float fov_y = find_child_by_name(node, { "fov" }).attribute("value").as_float();
                scene.AddSensor<Camera>(origin, target, up, width, height, fov_y);
            }
            else
            {
                TENSORRAY_ASSERT_MSG(false, std::string("Unsupported sensor: ") + sensor_type);
            }
        }
    }
}