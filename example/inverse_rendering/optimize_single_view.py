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
from random import seed, randrange

from pyTensorRay.fwd import *
from pyTensorRay.common import get_int_ptr, mkdir
from pyTensorRay.utils import image_tensor_to_torch, save_torch_image, read_images, update_render_batch_options
from pyTensorRay.parameter_manager import SceneParameterManager
from pyTensorRay.renderer_single_view import RenderSingleView
from pyTensorRay.loss import *
from pyTensorRay.optimizer import LargeSteps


@gin.configurable
@dataclasses.dataclass
class TrainRunner:
    scene_init_file: str = "./scenes/sphere_to_cube/init.xml"
    scene_target_file: str = "./scenes/sphere_to_cube/tar.xml"
    out_dir: str = "./output/sphere_to_cube/"
    target_dir: str = "./output/sphere_to_cube/target/"

    shape_id: int = 0
    niter: int = 500
    lr: float = 0.1
    lambda_value: float = 10.0
    print_size: int = 10
    target_spp: int = 1024

    options: TR.RenderOptions = None
    integrator: TR.Integrator = TR.PathTracer()
    edge_integrators: List = dataclasses.field(
        default_factory=lambda: [TR.PixelBoundaryIntegrator(), TR.PrimaryEdgeIntegrator(), TR.DirectEdgeIntegrator(), TR.IndirectEdgeIntegrator()])

    def __post_init__(self):
        mkdir(self.out_dir)
        mkdir(os.path.join(self.out_dir, "iter"))

        # Set up target scene
        scene_target = Scene(self.scene_target_file)
        scene_target.configure()

        self.width = scene_target.get_width(0)
        self.height = scene_target.get_height(0)

        # TODO: check if RenderOptions.seed is useful
        self.options = RenderOptions(
            1234, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED
        )
        batch_options = RenderBatchOptions(gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED)
        self.options = update_render_batch_options(self.options, batch_options)

        # Set up target image
        spp = self.options.spp
        self.options.spp = self.target_spp
        image_tensor = self.integrator.renderC(scene_target, self.options)
        self.train_image = image_tensor_to_torch(image_tensor, self.height, self.width).detach()
        save_torch_image(os.path.join(self.out_dir, "target.exr"), self.train_image)
        self.train_image_pyramid = build_pyramid(self.train_image)

        # Set up init scene
        self.options.spp = spp
        self.scene_init = Scene(self.scene_init_file)
        # TODO: passing a dict to set up requires_grad
        self.scene_init.shapes[self.shape_id].diff_all_vertex_pos()
        self.scene_init.configure()
        self.scene_param_manager = SceneParameterManager([0])
        self.scene_param_manager.get_values()
        print(self.scene_param_manager)
        self.scene_init.configure()

        self.render = RenderSingleView(self.scene_init, self.scene_param_manager,
                                       self.integrator, self.edge_integrators, self.options, None)

    def run(self):
        # Compute Laplacian and set up solver
        shape = self.scene_init.shapes[self.shape_id]
        face_count = shape.get_face_count()
        F = torch.zeros([face_count, 3], dtype=torch.int32)
        shape.get_face_indices(get_int_ptr(F))

        optimizer = torch.optim.Adam(self.scene_param_manager.params, lr=self.lr, eps=1e-6)
        #optimizer = LargeSteps(self.scene_param_manager.params, F,
        #                       self.scene_param_manager, self.lr, (0.9, 0.999), self.lambda_value)

        seed(1)
        error = []
        for i_iter in range(self.niter):
            now = datetime.datetime.now()
            optimizer.zero_grad()
            loss = 0.0

            # render
            img = self.render(self.scene_param_manager.params[0], {"seed": int(i_iter * 1e5)})
            img[img.isnan()] = 0.0
            # compute loss and backward
            img_loss = compute_render_loss_L1(img, self.train_image, 1.0)
            #img_loss = compute_render_loss_pyramid_L1(img, self.train_image_pyramid, 1.0)

            loss += img_loss.item()
            img_loss.backward()

            optimizer.step()
            self.scene_param_manager.set_values()
            self.scene_init.configure()
            # print stats
            end = datetime.datetime.now() - now
            print("[INFO] iter = {:d}, loss = {:.3f}, time = {:.3f}".format(
                i_iter, loss, end.seconds + end.microseconds / 1000000.0))
            error.append(loss)
            np.savetxt(os.path.join(self.out_dir, "loss.log"), error)
            # export files
            if i_iter == 0 or (i_iter + 1) % self.print_size == 0:
                self.scene_init.shapes[self.shape_id].export_mesh(
                    os.path.join(self.out_dir, "iter/iter_{:d}.obj".format(i_iter + 1)))
                print("Testing", end='\r')
                test_image = self.render(self.scene_param_manager.params[0], {"seed": int(i_iter * 1e5)}).detach()
                test_image = torch.cat([self.train_image, test_image], axis=0)
                save_torch_image(os.path.join(self.out_dir, "iter/iter_{:d}.png".format(i_iter + 1)), test_image)


if __name__ == "__main__":
    default_config = "sphere_to_cube.conf"
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", metavar="config_file", type=str, nargs="?",
                        default=default_config, help="config file")
    args, unknown = parser.parse_known_args()
    gin.parse_config_file(args.config_file, skip_unknown=True)

    TR.env_create()
    train_runner = TrainRunner()
    train_runner.run()
    TR.env_release()
