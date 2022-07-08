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
sys.path.insert(1, "../../")

import argparse
import dataclasses
from typing import List

import TensorRay as TR
import torch
import numpy as np
from pyTensorRay.multi_pose import gen_random_rotations, gen_z_axis_rotations, matrices_to_tensor, render_save_multi_pose
from pyTensorRay.fwd import Camera, RenderOptions, Scene
from pyTensorRay.common import mkdir, get_ptr
import gin
import os
from random import seed


@gin.configurable
@dataclasses.dataclass
class PreprocessRunner:
    scene_file: str
    batch_size: int = 1
    seed: int = 1
    shape_id: int = 0
    out_dir: str = "./"
    integrator: TR.Integrator = TR.PathTracer()

    def run(self):
        self.scene = Scene(self.scene_file)
        self.scene.configure()
        self.width = self.scene.get_width(0)
        self.height = self.scene.get_height(0)
        with gin.config_scope('tar'):
            self.options = RenderOptions(gin.REQUIRED)
        self.options.quiet = False
        self.options.spp_batch = 64
        rotation_matrices = gen_random_rotations(batch=self.batch_size, seed=self.seed)
        #rotation_matrices = gen_z_axis_rotations(self.batch_size)
        rotation_tensors = matrices_to_tensor(rotation_matrices)

        mkdir(self.out_dir)
        np.savetxt(os.path.join(self.out_dir, "rotation_matrices.txt"), rotation_matrices)

        obj_center = self.scene.shapes[self.shape_id].get_obj_center()
        p = torch.zeros(3)
        obj_center.value_to_torch(get_ptr(p))
        np.savetxt(os.path.join(self.out_dir, "obj_center.txt"), p.numpy())

        mkdir(os.path.join(self.out_dir, "target"))
        render_save_multi_pose(self.scene, self.shape_id, rotation_tensors, self.options,
                               self.integrator, os.path.join(self.out_dir, "target"))


if __name__ == "__main__":
    seed(1)
    default_config = './sphere_to_cube.conf'
    parser = argparse.ArgumentParser(
        description='Script for generating validation results')
    parser.add_argument('config_file', metavar='config_file',
                        type=str, nargs='?', default=default_config, help='config file')
    args, unknown = parser.parse_known_args()
    # Dependency injection: Arguments are injected into the function from the gin config file.
    gin.parse_config_file(args.config_file, skip_unknown=True)
    TR.env_create()
    runner = PreprocessRunner()
    runner.run()
    TR.env_release()
