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
sys.path.insert(1, "../../")        # to import pyTensorRay
import argparse
import gin
import TensorRay as TR
import pyTensorRay as pyTR
from pyTensorRay.fwd import *
from pyTensorRay.utils import image_tensor_to_torch, save_torch_image, update_render_batch_options
import dataclasses
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# inject dependencies into the constructor

@gin.configurable
# generate the constructor
@dataclasses.dataclass
class TestRunner:
    scene: TR.Scene
    options: TR.RenderOptions
    integrator: TR.Integrator
    out_dir: str = "./"
    render_type: str = "forward"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.width = self.scene.get_width(0)
        self.height = self.scene.get_height(0)
        self.options.quiet = False
        batch_options = RenderBatchOptions(gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED, gin.REQUIRED)
        self.options = update_render_batch_options(self.options, batch_options)

    def renderC(self):
        self.scene.configure()
        img_tensor = self.integrator.renderC(self.scene, self.options)
        img = image_tensor_to_torch(img_tensor, self.height, self.width)
        save_torch_image(self.out_dir + 'TR_renderC.exr', img)
        print("[INFO] RenderC finished!")

    def renderD(self):
        xform = Transform(gin.REQUIRED)
        num_vars = 1
        x = TR.Tensorf(0, True)
        if xform.transformation.xform_type == "translate":
            T = xform.transformation.translation
            if xform.vertex_id == -1:
                self.scene.shapes[xform.shape_id].translation = TR.Tensorf(T[0], T[1], T[2]) * x
            else:
                self.scene.shapes[xform.shape_id].vertex_id = xform.vertex_id
                self.scene.shapes[xform.shape_id].vertex_translation = TR.Tensorf(T[0], T[1], T[2]) * x
        else:
            assert(False)
        self.scene.configure()
        self.options.export_deriv = True

        deriv_tensor = None
        integrators = [self.integrator, TR.PixelBoundaryIntegrator(), TR.PrimaryEdgeIntegrator(), TR.DirectEdgeIntegrator(), TR.IndirectEdgeIntegrator()]
        names = ["TR_interior", "TR_pixelB", "TR_primaryB", "TR_directB", "TR_indirectB"]
        for i in range(len(integrators)):
            img_tensor = integrators[i].renderD(self.scene, self.options, TR.Tensorf())
            if deriv_tensor is None:
                deriv_tensor = img_tensor
            else:
                deriv_tensor = deriv_tensor + img_tensor
            deriv_component = image_tensor_to_torch(img_tensor, num_vars * self.height, self.width)
            deriv_component[:, :, 1:] = 0
            save_torch_image(self.out_dir + names[i] + ".exr", deriv_component)

        derviv_all = image_tensor_to_torch(deriv_tensor, num_vars * self.height, self.width)
        derviv_all[:, :, 1:] = 0
        save_torch_image(self.out_dir + "TR_renderD.exr", derviv_all)
        print("[INFO] RenderD finished!")

    def render_fd(self):
        delta = 0.01
        self.options.spp = 8192
        self.options.spp_batch = 64
        self.scene.configure()
        TR.set_rnd_seed(self.options.seed)
        img_tensor = self.integrator.renderC(self.scene, self.options)
        img0 = image_tensor_to_torch(img_tensor, self.height, self.width)
        save_torch_image(self.out_dir + "TR_fd_img_0.exr", img0)

        xform = Transform(gin.REQUIRED)
        if xform.transformation.xform_type == "translate":
            T = xform.transformation.translation
            if xform.vertex_id == -1:
                self.scene.shapes[xform.shape_id].translation = TR.Tensorf(T[0] * delta, T[1] * delta, T[2] * delta)
            else:
                self.scene.shapes[xform.shape_id].vertex_id = xform.vertex_id
                self.scene.shapes[xform.shape_id].vertex_translation = TR.Tensorf(T[0] * delta, T[1] * delta, T[2] * delta)
        else:
            assert(False)
        self.scene.configure()
        TR.set_rnd_seed(self.options.seed)
        img_tensor = self.integrator.renderC(self.scene, self.options)
        img1 = image_tensor_to_torch(img_tensor, self.height, self.width)
        save_torch_image(self.out_dir + "TR_fd_img_1.exr", img1)

        fd = (img1 - img0) / delta
        fd[:, :, 1:] = 0
        save_torch_image(self.out_dir + "TR_fd_deriv.exr", fd)
        print("[INFO] FD finished!")

    def run(self, render_type):
        if render_type != "":
            self.render_type = render_type

        if self.render_type == "forward":
            self.renderC()
        elif self.render_type == "backward":
            self.renderD()
        elif self.render_type == "fd":
            self.render_fd()
        else:
            assert(False)

if __name__ == "__main__":
    TR.env_create()
    default_config = './cbox_bunny.conf'
    parser = argparse.ArgumentParser(description='Script for generating validation results')
    parser.add_argument('config_file', metavar='config_file',
                        type=str, nargs='?', default=default_config, help='config file')
    parser.add_argument('render_type', metavar='render_type',
                        type=str, nargs='?', default='', help='render type')
    args, unknown = parser.parse_known_args()
    # Dependency injection: Arguments are injected into the function from the gin config file.
    gin.parse_config_file(args.config_file, skip_unknown=True)
    test_runner = TestRunner()
    test_runner.run(args.render_type)
    TR.env_release()
