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

import TensorRay as TR
import numpy as np
import torch
import cv2
import os
import re

import logging
import datetime
logger = logging.getLogger(__name__)

from pyTensorRay.common import get_ptr


def update_render_batch_options(options, batch_options):
    options.spp_batch = batch_options.spp_batch
    options.sppe_batch = batch_options.sppe_batch
    options.sppse0_batch = batch_options.sppse0_batch
    options.sppse1_batch = batch_options.sppse1_batch
    options.sppe0_batch = batch_options.sppe0_batch
    return options


def update_guiding_options(options, guiding_options):
    options.g_direct = guiding_options.g_direct
    options.g_direct_max_size = guiding_options.g_direct_max_size
    options.g_direct_spp = guiding_options.g_direct_spp
    options.g_eps = guiding_options.g_eps


def image_tensor_to_torch(img_tensor, height, width):
    """
    img_tensor: TR.Tensorf with TensorShape([width * height], TR.VecType.Vec3)
    return: torch.tensor with shape (height, width, 3)
    """
    img = torch.zeros(3, height * width)
    img_tensor.value_to_torch(get_ptr(img))
    img = img.t().reshape([height, width, 3])
    return img


def image_torch_to_tensor(img, height, width):
    """
    img: torch.tensor with shape(height, width, 3)
    return: TR.Tensorf with TensorShape([width * height], TR.VecType.Vec3)
    """
    img = img.reshape(height * width, 3)
    img = img.t().contiguous()
    tshape = TR.TensorShape([width * height], TR.VecType.Vec3)
    img_tensor = TR.Tensorf(get_ptr(img), tshape)
    return img_tensor


def load_torch_image(filename):
    """
    return: torch.tensor with shape (height, width, 3)
    """
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).float()
    return img


def load_image_to_tensor(filename):
    """
    return: TR.Tensorf with TensorShape([width * height], TR.VecType.Vec3)
    """
    img = load_torch_image(filename)
    height = img.shape[0]
    width = img.shape[1]
    # convert image to tensor
    return image_torch_to_tensor(img, height, width)


def save_torch_image(filename, img):
    """
    img: torch.tensor with shape (height, width, 3)
    """
    if filename.endswith(".png"):
        img = img.numpy().astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.power(img, 1 / 2.2)
        img = np.uint8(np.clip(img * 255., 0., 255.))
        cv2.imwrite(filename, img)
    elif filename.endswith(".exr"):
        output = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, output)
    else:
        assert False


def read_images(path, ext=".exr"):
    """
        Reads all images from a path.
        """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    images = []
    files = os.listdir(path)
    files.sort(key=alphanum_key)
    for f in files:
        if f.endswith(ext):
            images.append(load_torch_image(os.path.join(path, f)))
    return images


def uniform_laplacian(num_verts, edges):
    # compute L once per mesh subdiv.
    with torch.no_grad():
        V = num_verts
        e0, e1 = edges.unbind(0)
        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)
        # First, we construct the adjacency matrix,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=edges.device)

        # We construct the Laplacian matrix by adding the non diagonal values
        # i.e. L[i, j] = 1 if (i, j) is an edge
        L = torch.sparse.FloatTensor(idx, ones, (V, V))

        # Then we add the diagonal values L[i, i] = -1.
        idx = torch.arange(V, device=edges.device)
        idx = torch.stack([idx, idx], dim=0)
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=edges.device)
        L -= torch.sparse.FloatTensor(idx, -ones, (V, V))

        vals = torch.sparse.sum(L, dim=0).to_dense()
        indices = torch.arange(V, device=edges.device)
        idx = torch.stack([indices, indices], dim=0)
        L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L


def sparse_eye(size):
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))
