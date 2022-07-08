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

import os.path

from TensorRay import Camera
import TensorRay as TR
import numpy as np
import torch
import gin
from scipy.spatial.transform import Rotation as R

from pyTensorRay.utils import image_tensor_to_torch, save_torch_image, load_torch_image


def gen_random_rotations(batch, seed):
    matrices = R.random(batch, seed).as_matrix()
    matrices[0] = np.eye(3)
    return np.reshape(matrices, (batch, 9))


def gen_z_axis_rotations(batch):
    matrices = R.from_euler('z', np.linspace(0, 2 * np.pi, batch, endpoint=False)).as_matrix()
    return np.reshape(matrices, (batch, 9))


def matrices_to_tensor(matrices):
    res = []
    for i in range(matrices.shape[0]):
        mat = matrices[i]
        res.append(
            TR.Tensorf(
                mat[0], mat[1], mat[2], 0,
                mat[3], mat[4], mat[5], 0,
                mat[6], mat[7], mat[8], 0,
                0, 0, 0, 1
            )
        )
    return res


@gin.configurable
def rotation_matrices(filename):
    matrices = np.loadtxt(filename)
    return matrices


def render_save_multi_pose(scene, shape_id, rotation_tensors, options, integrator, out_dir):
    obj = scene.shapes[shape_id]
    obj_center = obj.get_obj_center()
    for i, rot in enumerate(rotation_tensors):
        # switch pose
        obj.obj_center = obj_center
        obj.rotate_matrix = rot
        scene.configure()
        # render
        img_tensor = integrator.renderC(scene, options)
        height = scene.get_height(0)
        width = scene.get_width(0)
        img = image_tensor_to_torch(img_tensor, height, width)
        # save
        save_torch_image(os.path.join(out_dir, "sensor_{:d}.exr".format(i)), img)


if __name__ == '__main__':
    from common import get_ptr
    rot = gen_random_rotations(10)
