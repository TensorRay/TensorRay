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

from pyTensorRay.utils import image_tensor_to_torch, save_torch_image, load_torch_image


def sample_on_sphere(batch, radius):
    """
    sample points on a sphere with radius=radius
    """
    cosPhi = np.linspace(1.0 - 0.01, -1.0 + 0.01, batch)
    theta  = np.linspace(0, np.pi * 10, batch) + np.random.uniform(0, 1, batch) * (1/float(batch))
    sinPhi = np.sqrt(1 - cosPhi * cosPhi)
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)

    return np.array([sinPhi * cosTheta * radius, cosPhi * radius, sinPhi * sinTheta * radius]).T


def sample_on_hemisphere(batch, radius):
    """
    sample points on a sphere with radius=radius
    """
    cosPhi = np.linspace(1.0 - 0.01, 0.1, batch)
    theta  = np.linspace(0, np.pi * 10, batch) + np.random.uniform(0, 1, batch) * (1/float(batch))
    sinPhi = np.sqrt(1 - cosPhi * cosPhi)
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)

    return np.array([sinPhi * cosTheta * radius, sinPhi * sinTheta * radius, cosPhi * radius]).T


def gen_camera(origin, target, up, fov, resolution):
    """
    generate a camera
    """
    return Camera(TR.Vector3f(origin[0], origin[1], origin[2]),
                  TR.Vector3f(target[0], target[1], target[2]),
                  TR.Vector3f(up[0], up[1], up[2]),
                  resolution[0], resolution[1], fov)


def gen_cameras(positions, target, up, fov, resolution):
    """
    generate a list of cameras from a list of positions
    """
    return [gen_camera(p, target, up, fov, resolution)
            for p in positions]


@gin.configurable
def camera_positions(filename):
    pos = np.loadtxt(filename)
    assert(pos.shape[1] == 3)
    return pos


@gin.configurable
def gen_camera_positions(batch, radius, on_hemisphere):
    np.random.seed(batch)
    if on_hemisphere:
        return sample_on_hemisphere(batch, radius)
    else:
        return sample_on_sphere(batch, radius)


def render_save_multi_views(scene, cameras, options, integrator, out_dir):
    for i, camera in enumerate(cameras):
        # switch camera
        scene.cameras[0].update(camera)
        # render
        img_tensor = integrator.renderC(scene, options)
        height = scene.get_height(0)
        width = scene.get_width(0)
        img = image_tensor_to_torch(img_tensor, height, width)
        # save
        save_torch_image(os.path.join(out_dir, "sensor_{:d}.exr".format(i)), img)


if __name__ == '__main__':
    from common import get_ptr
    camera = gen_camera([1., 2., 3.], [0., 0., 1.], [0., 1., 0.], 45., [512, 512])
    pos = torch.zeros([3], dtype=torch.float32)
    camera.posTensor.value_to_torch(get_ptr(pos))
    print(pos)
