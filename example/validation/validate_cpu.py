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

import argparse
from cv2 import transform
import gin
import psdr_cpu
from pypsdr.validate import *
from pypsdr.utils.io import *
import dataclasses
import os
import numpy as np
import torch
# import yep

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# inject dependencies into the constructor


@gin.configurable
# generate the constructor
@dataclasses.dataclass
class TestRunner:
    scene: psdr_cpu.Scene
    options: psdr_cpu.RenderOptions
    integrator: psdr_cpu.Integrator
    guiding_options: GuidingOptions = GuidingOptions()
    out_dir: str = "./"
    render_type: str = "forward"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.width = self.scene.camera.width
        self.height = self.scene.camera.height
        psdr_cpu.set_verbose(True)
        self.integrator = Path2(False)

    def renderC(self):
        image = self.integrator.renderC(
            self.scene, self.options).reshape(self.height, self.width, 3)
        imwrite(image, os.path.join(self.out_dir, "psdr_renderC.exr"))

    def renderD(self):
        xform = Transform(gin.REQUIRED)

        if self.guiding_options.guide_type != "":
            self.scene.shapes[xform.shape_id].sort_config = self.guiding_options.sort_config
            self.scene.shapes[xform.shape_id].enable_draw = True
            self.scene.shapes[xform.shape_id].configure()

        sceneAD = psdr_cpu.SceneAD(self.scene)

        d_image = np.ones((self.height, self.width, 3))
        d_image[:, :, 1:3] = 0

        # dependency injection
        xform = Transform(gin.REQUIRED)
        assert(xform.shape_id >= 0)
        shape = sceneAD.val.shapes[xform.shape_id]
        shape.requires_grad = True
        if xform.vertex_id >= 0:
            shape.vertex_idx = xform.vertex_id
        if type(xform.transformation) is Translation:
            shape.setTranslation(xform.transformation.translation)
        elif type(xform.transformation) is Rotation:
            shape.setRotation(xform.transformation.rotation)
        else:
            assert(False)

        img = self.integrator.renderD(sceneAD, self.options, d_image.reshape(-1))
        imwrite(img.reshape((self.height, self.width, 3)), os.path.join(self.out_dir, "psdr_interior.exr"))

        boundary_integrator = psdr_cpu.PrimaryEdgeIntegrator(sceneAD.val)
        img_primary = boundary_integrator.renderD(sceneAD, self.options, d_image.reshape(-1))
        imwrite(img_primary.reshape((self.height, self.width, 3)), os.path.join(self.out_dir, "psdr_primaryB.exr"))
        img += img_primary

        boundary_integrator = psdr_cpu.DirectEdgeIntegrator(sceneAD.val)
        img_direct = boundary_integrator.renderD(sceneAD, self.options, d_image.reshape(-1))
        imwrite(img_direct.reshape((self.height, self.width, 3)), os.path.join(self.out_dir, "psdr_directB.exr"))
        img += img_direct

        boundary_integrator = psdr_cpu.IndirectEdgeIntegrator(sceneAD.val)
        img_indirect = boundary_integrator.renderD(sceneAD, self.options, d_image.reshape(-1))
        imwrite(img_indirect.reshape((self.height, self.width, 3)), os.path.join(self.out_dir, "psdr_indirectB.exr"))
        img += img_indirect

        boundary_integrator = psdr_cpu.PixelBoundaryIntegrator(sceneAD.val)
        img_pixel = boundary_integrator.renderD(sceneAD, self.options, d_image.reshape(-1))
        imwrite(img_pixel.reshape((self.height, self.width, 3)), os.path.join(self.out_dir, "psdr_pixelB.exr"))
        img += img_pixel

        # boundary_integrator = psdr_cpu.BoundaryIntegrator(sceneAD.val)

        # if self.guiding_options.guide_type != "":
        #     if self.guiding_options.guide_option == "direct" or self.guiding_options.guide_option == "both":
        #         boundary_integrator.recompute_direct_edge(sceneAD.val)
        #         if self.guiding_options.guide_type == "grid":
        #             boundary_integrator.preprocess_grid_direct(
        #                 sceneAD.val, self.guiding_options.grid_config_direct, self.options.max_bounces)
        #         else:
        #             boundary_integrator.preprocess_aq_direct(
        #                 sceneAD.val, self.guiding_options.aq_config_direct, self.options.max_bounces)
        #     if self.guiding_options.guide_option == "indirect" or self.guiding_options.guide_option == "both":
        #         boundary_integrator.recompute_indirect_edge(sceneAD.val)
        #         if self.guiding_options.guide_type == "grid":
        #             boundary_integrator.preprocess_grid_indirect(
        #                 sceneAD.val, self.guiding_options.grid_config_indirect, self.options.max_bounces)
        #         else:
        #             boundary_integrator.preprocess_aq_indirect(
        #                 sceneAD.val, self.guiding_options.aq_config_indirect, self.options.max_bounces)

        # img += boundary_integrator.renderD(
        #     sceneAD, self.options, d_image.reshape(-1))



        img = torch.tensor(img)
        img[img.isnan()] = 0
        img = img.reshape((self.height, self.width, 3))
        print("img grad:", img.clone().detach().sum())
        print("grad: ", torch.tensor(
            np.array(sceneAD.der.shapes[xform.shape_id].vertices))[0][2])
        imwrite(img.numpy(), os.path.join(self.out_dir, "psdr_renderD.exr"))

    def render_fd(self):
        delta = 0.01
        self.options.spp = 8192
        image1 = self.integrator.renderC(
            self.scene, self.options).reshape(self.height, self.width, 3)

        xform = Transform(gin.REQUIRED)
        shape = self.scene.shape_list[xform.shape_id]
        vertices = np.array(shape.vertices)
        if xform.vertex_id >= 0:
            vertices[xform.vertex_id] = xform.transformation.transform(
                vertices[xform.vertex_id], delta)
        else:
            for i in range(len(vertices)):
                vertices[i] = xform.transformation.transform(
                    vertices[i], delta)
        self.scene.shape_list[xform.shape_id].vertices = vertices
        self.scene.configure()
        image2 = self.integrator.renderC(
            self.scene, self.options).reshape(self.height, self.width, 3)
        fd = (image2 - image1) / delta
        fd[:, :, 1:] = 0
        imwrite(fd, os.path.join(self.out_dir, "psdr_fd.exr"))

    def d_render(self):
        sceneAD = psdr_cpu.SceneAD(self.scene)
        d_image = np.ones((self.height, self.width, 3))
        d_image[:, :, 1:3] = 0
        img = self.integrator.renderD(
            sceneAD, self.options, d_image.reshape(-1).astype(np.float32))
        print(torch.tensor(sceneAD.der.shapes[0].vertices).abs().sum())

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
    default_config = './cbox_bunny.conf'
    parser = argparse.ArgumentParser(
        description='Script for generating validation results')
    parser.add_argument('config_file', metavar='config_file',
                        type=str, nargs='?', default=default_config, help='config file')
    parser.add_argument('render_type', metavar='render_type',
                        type=str, nargs='?', default='', help='render type')
    args, unknown = parser.parse_known_args()
    # Dependency injection: Arguments are injected into the function from the gin config file.
    gin.parse_config_file(args.config_file, skip_unknown=True)
    test_runner = TestRunner()
    # yep.start("val.prof")

    test_runner.run(args.render_type)
    # yep.stop()
