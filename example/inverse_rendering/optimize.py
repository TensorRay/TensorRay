# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import sys
import os
sys.path.insert(1, "../../")

import gin
import torch
import TensorRay as TR
import numpy as np
import scipy
import scipy.sparse.linalg
import dataclasses
import argparse
import datetime
from typing import List

from pyTensorRay.fwd import *
from pyTensorRay.common import get_int_ptr
from pyTensorRay.utils import image_tensor_to_torch, save_torch_image, uniform_laplacian, sparse_eye
from pyTensorRay.parameter_manager import SceneParameterManager
from pyTensorRay.renderer import Render
from pyTensorRay.loss import compute_render_loss_L2
from pyTensorRay.optimizer import LGDescent


scene_dir = "inverse_rendering/sphere_to_cube/"

@gin.configurable
@dataclasses.dataclass
class TrainRunner:
    scene_init_file: str = scene_dir + "scene.xml"
    scene_target_file: str = scene_dir + "sceneT.xml"
    out_dir: str = scene_dir + "results/"
    target_dir: str = scene_dir + "results/"

    shape_id: int = 0
    target_spp: int = 128
    niter: int = 500
    lr: float = 0.1
    lambda_value: float = 10.0

    options: TR.RenderOptions = None
    integrator: TR.Integrator = TR.PathTracer()
    edge_integrators: List = dataclasses.field(
        default_factory=lambda: [TR.PrimaryEdgeIntegrator, TR.DirectEdgeIntegrator, TR.IndirectEdgeIntegrator])

    def __post_init__(self):
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        # TODO: check if RenderOptions.seed is useful
        self.options = RenderOptions(
            1234, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED
        )
        # Set up target image
        scene_target = Scene(self.scene_target_file)
        self.width = scene_target.get_width(0)
        self.height = scene_target.get_height(0)

        scene_target.configure()
        spp = self.options.spp
        self.options.spp = self.target_spp
        target_tensor = self.integrator.renderC(scene_target, self.options)
        self.target_img = image_tensor_to_torch(target_tensor, self.height, self.width)
        save_torch_image(self.out_dir + "target.exr", self.target_img)
        print("[INFO] Target image rendered!")
        self.options.spp = spp

        # Set up optimization
        self.scene_init = Scene(self.scene_init_file)
        # TODO: passing a dict to set up requires_grad
        self.scene_init.shapes[self.shape_id].diff_all_vertex_pos()
        self.scene_init.configure()
        self.scene_param_manager = SceneParameterManager([0])
        self.scene_param_manager.get_values()
        print(self.scene_param_manager)

        self.render = Render(self.scene_init, None, self.scene_param_manager,
                             self.integrator, self.edge_integrators, self.options, None)

    def run(self):
        # Compute Laplacian and set up solver
        unknown_shape_ids = [self.shape_id]
        mesh_vertex_counts = []
        mesh_edge_counts = []
        for i, shape_id in enumerate(unknown_shape_ids):
            shape = self.scene_init.shapes[shape_id]
            mesh_vertex_counts.append(shape.get_vertex_count())
            mesh_edge_counts.append(shape.get_edge_count())

        vertex_offset = 0
        edge_data = None
        for i, shape_id in enumerate(unknown_shape_ids):
            shape = self.scene_init.shapes[shape_id]
            tmp_edge_data = torch.zeros([2, mesh_edge_counts[i]], dtype=torch.int32)
            shape.get_edge_data(get_int_ptr(tmp_edge_data))
            tmp_edge_data += vertex_offset
            vertex_offset += mesh_vertex_counts[i]
            if i == 0:
                edge_data = tmp_edge_data
            else:
                edge_data = torch.cat([edge_data, tmp_edge_data], axis=1)

        edge_data = edge_data.type(torch.LongTensor)
        L = uniform_laplacian(sum(mesh_vertex_counts), edge_data).detach() * self.lambda_value
        I = sparse_eye(L.shape[0])
        IL_term = I + L
        Lv = np.asarray(IL_term.coalesce().values())
        Li = np.asarray(IL_term.coalesce().indices())

        IL_term_sparse = scipy.sparse.coo_matrix((Lv, Li), shape=L.shape)
        IL_term_sparse_solver = scipy.sparse.linalg.factorized(IL_term_sparse)

        #optimizer = torch.optim.Adam(self.scene_param_manager.params, lr=self.lr, eps=1e-6)
        optimizer = LGDescent(
            params=[
                {"params": self.scene_param_manager.params[0],
                 "lr": self.lr}
            ],
            IL_term=IL_term,
            IL_solver=IL_term_sparse_solver
        )

        error = []
        for i_iter in range(self.niter):
            now = datetime.datetime.now()
            self.scene_param_manager.set_values()
            self.scene_init.configure()
            optimizer.zero_grad()
            # render
            img = self.render(self.scene_param_manager.params[0], 0,
                              {"seed": int(i_iter * 1e5)})
            img[img.isnan()] = 0.0
            # compute loss and backward
            img_loss = compute_render_loss_L2(img, self.target_img, 1.0)
            img_loss.backward()
            loss = img_loss.item()
            optimizer.step()
            # print stats
            end = datetime.datetime.now() - now
            print("[INFO] iter = {:d}, loss = {:.3f}, time = {:.3f}".format(
                i_iter, loss, end.seconds + end.microseconds / 1000000.0))
            error.append(loss)
            np.savetxt(self.out_dir + "loss.log", error)
            # export image
            img = img.detach().reshape([self.height, self.width, 3])
            save_torch_image(self.out_dir + "iter_%d.exr" % i_iter, img)


if __name__ == "__main__":
    default_config = "sphere_to_cube.conf"
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="config_file", type=str, nargs="?",
                        default=default_config, help="config file")
    args, unknown = parser.parse_known_args()
    gin.parse_config_file(args.config_file, skip_unknown=True)

    os.chdir("../")
    TR.env_create()
    train_runner = TrainRunner()
    train_runner.run()
    TR.env_release()
