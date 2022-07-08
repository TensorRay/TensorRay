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

#include "../Tensor/Tensor.h"
#include "Distribution.h"
#include "Boundary.h"
#include "NEE.cuh"

namespace EDX
{
	namespace TensorRay
	{

        class AQLeaf {
            public:
                Tensorf         poly;
                Tensorf         p0, p1;
        };

        struct Tree3D {
            Tensorf         p0, p1;
        };

        struct GuidingOption {
			int   depth = 0;
			int   max_size = 100;
			int   spp = 16;
			float thold = 0.01f;
            float eps = 0.01f;
        };


        class AdaptiveQuadratureDistribution {
            public:
                Distribution1D    aq_distrb;
                AQLeaf            aq_leaf;

            Tensorf GuideDirectBoundary_Helper(const Camera& camera, const Scene& scene, const SecondaryEdgeInfo& secEdges, const Tensorf& rnd) {
                BoundarySegSampleDirect bss;
                int sample_size = rnd.LinearSize() / 3;
                if (SampleBoundarySegmentDirect(scene, secEdges, sample_size, rnd, Scalar(1.0), bss, true) == 0)
                {
                    return Zeros(Shape({ sample_size }));
                }
                Tensorf boundaryTerm;
                if (EvalBoundarySegmentDirect(camera, scene, 1, 1, bss, boundaryTerm, true) == 0) {
                    return Zeros(Shape({ sample_size }));
                }
                boundaryTerm = Abs(Detach(boundaryTerm));
                Tensorf result = IndexedWrite(boundaryTerm, bss.maskValid.index, Shape({ sample_size }), 0);
                return Detach(result);
            };

            void setup(const Camera& camera, const Scene& scene, const SecondaryEdgeInfo& secEdges, const Tensorf &cdfx, const GuidingOption &options) {
                std::cout << "Begin AQ Guiding" << std::endl;
                // Tensorf cdfxx = (Tensorf::ArrayRange(1)+Tensorf(1.0f)) / Tensorf(1.0f);

                Tensorf cdfy = (Tensorf::ArrayRange(1)+Tensorf(1.0f)) / Tensorf(1.0f);
                Tensorf cdfz = (Tensorf::ArrayRange(1)+Tensorf(1.0f)) / Tensorf(1.0f);
                int dimx = cdfx.LinearSize();
                int dimy = 1;
                int dimz = 1;

                size_t init_size = dimx*dimy*dimz;

                    
                std::cout << "init_size: " << init_size << std::endl;


                int max_tree_size = options.max_size;
                Tree3D tree_leaf;
                tree_leaf.p0 = Zeros(Shape({ max_tree_size }, VecType::Vec3));
                tree_leaf.p1 = Zeros(Shape({ max_tree_size }, VecType::Vec3));

                psdr_cuda::init_tree(cdfx.Data(), 
                                    cdfy.Data(),
                                    cdfz.Data(),
                                    dimx,
                                    dimy,
                                    dimz,
                                    tree_leaf.p0.Data(),
                                    tree_leaf.p1.Data(),
                                    max_tree_size);

                // test(tree_leaf.p0.Data());
                int fix_size = init_size;
                int cut_dim = 1;

                // std::cout << "tree_leaf.p0: " << tree_leaf.p0 << std::endl;
                // std::cout << "tree_leaf.p1: " << tree_leaf.p1 << std::endl;

                // for (int i=0; i<0; ++i) {
                //     Tensorf eval_rnd_buf = Zeros(Shape({ fix_size * 36 }, VecType::Vec3));
                //     psdr_cuda::generate_eval_point(fix_size,
                //                     tree_leaf.p0.Data(),
                //                     tree_leaf.p1.Data(),
                //                     eval_rnd_buf.Data(), i);

                //     Tensorf error_value = GuideDirectBoundary_Helper(camera, scene, secEdges, eval_rnd_buf);
                //     int app_size = psdr_cuda::cut_grid( error_value.Data(),
                //                                         tree_leaf.p0.Data(),
                //                                         tree_leaf.p1.Data(),
                //                                         fix_size, 0.001f, 1.0f);
                //     fix_size += app_size;

                //     std::cout << "\r"  << "Depth " << i << " with grid: " << fix_size;
                // }
                // std::cout << std::endl;
                // aqtree leaf

                Tensori tree_idx = Tensori::ArrayRange(fix_size);
                // std::cout << tree_idx << std::endl;
                // AQLeaf aq_leaf;
                aq_leaf.p0   = IndexedRead(tree_leaf.p0, tree_idx, 0);
                aq_leaf.p1   = IndexedRead(tree_leaf.p1, tree_idx, 0);
                // psdr_cuda::gather(aq_leaf.p0.Data(), tree_leaf.p0.Data(), tree_idx.Data(), fix_size);
                // psdr_cuda::gather(aq_leaf.p1.Data(), tree_leaf.p1.Data(), tree_idx.Data(), fix_size);

                // std::cout << "aq_leaf.p0: " << aq_leaf.p0 << std::endl;
                // std::cout << "aq_leaf.p1: " << aq_leaf.p1 << std::endl;
                aq_leaf.poly = Zeros(Shape({ fix_size }));

                int spp_per_grid = options.spp;
                for(int i=0; i<spp_per_grid; ++i) {
                    Tensorf rnd_aq = Tensorf::RandomFloat(Shape({ fix_size }, VecType::Vec3));
                    Tensorf lerp_rnd_aq = aq_leaf.p0 + (aq_leaf.p1-aq_leaf.p0)*rnd_aq;
                    Tensori sample_id = Tensori::ArrayRange(fix_size);
                    Tensorf temp_boundaryTerm = GuideDirectBoundary_Helper(camera, scene, secEdges, lerp_rnd_aq) / Scalar(static_cast<float>(spp_per_grid));
                    aq_leaf.poly = aq_leaf.poly + IndexedWrite(temp_boundaryTerm, sample_id, Shape({ fix_size }), 0);
                }
                    
                aq_leaf.poly /= Sum(aq_leaf.poly);

                // std::cout  << "tree_leaf.p0: " << tree_leaf.p0 << std::endl;
                // std::cout  << "tree_leaf.p1: " << tree_leaf.p1 << std::endl;

                // std::cout  << "aq_leaf.p0: " << aq_leaf.p0 << std::endl;
                // std::cout  << "aq_leaf.p1: " << aq_leaf.p1 << std::endl;
                // std::cout << aq_leaf.poly << std::endl;

                Tensorf aq_leaf_area = (X(aq_leaf.p1)-X(aq_leaf.p0))*(Y(aq_leaf.p1)-Y(aq_leaf.p0))*(Z(aq_leaf.p1)-Z(aq_leaf.p0));

                // psdr_cuda::get_area(aq_leaf_area.Data(), aq_leaf.p0.Data(), aq_leaf.p1.Data(), fix_size*3);

                // std::cout  << "aq_leaf_area: " << aq_leaf_area << std::endl;
                // std::cout  << "aq_leaf.poly: " << aq_leaf.poly << std::endl;

                float ueps = options.eps;
                aq_leaf.poly = Scalar(ueps) + (Scalar(1.0) - Scalar(ueps))*aq_leaf.poly;
                aq_distrb.SetFunction(aq_leaf.poly * aq_leaf_area);
            }

            Tensorf sample(const Tensorf &_rnd, Tensorf &pdf) {
                Tensori aqIdx;
                Tensorf aqPdf;
                int fix_size = pdf.LinearSize();
                aq_distrb.SampleDiscrete(Tensorf::RandomFloat(Shape({ fix_size })), &aqIdx, &aqPdf);
                Tensorf p0_new = IndexedRead(aq_leaf.p0, aqIdx, 0);
                Tensorf p1_new = IndexedRead(aq_leaf.p1, aqIdx, 0);
                Tensorf wt_new = IndexedRead(aq_leaf.poly, aqIdx, 0);
                Tensorf result = MakeVector3(
                    X(p0_new) + X(_rnd)*(X(p1_new)-X(p0_new)),
                    Y(p0_new) + Y(_rnd)*(Y(p1_new)-Y(p0_new)),
                    Z(p0_new) + Z(_rnd)*(Z(p1_new)-Z(p0_new))
                );
                pdf = wt_new / Sum(aq_distrb.mPDF);
                return result;
            }
        };

    } // namespace psdr

}