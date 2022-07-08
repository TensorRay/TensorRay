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

import torch
import math


def compute_render_loss_L2(img, target_img, weight):
    diff = img - target_img
    diff = diff.nan_to_num(nan=0)
    loss = 0.5 * torch.square(diff).sum()
    return loss * weight


def compute_render_loss_L1(img, target_img, weight):
    diff = img - target_img
    diff = diff.nan_to_num(nan=0)
    diff = diff.abs()
    loss = diff.sum()
    return loss * weight


def downsample(input):
    if input.size(0) % 2 == 1:
        input = torch.cat((input, torch.unsqueeze(input[-1,:], 0)), dim=0)
    if input.size(1) % 2 == 1:
        input = torch.cat((input, torch.unsqueeze(input[:,-1], 1)), dim=1)
    return (input[0::2, 0::2, :] + input[1::2, 0::2, :] + input[0::2, 1::2, :] + input[1::2, 1::2, :]) * 0.25


def build_pyramid(img):
    level = int(min(math.log2(img.shape[0]), math.log2(img.shape[1]))) + 1
    level = min(5, level)
    imgs = []
    for i in range(level):
        imgs.append(img)
        if i < level - 1:
            img = downsample(img)
    return imgs


def compute_render_loss_pyramid_L1(img, target_pyramid, weight):
    img_pyramid = build_pyramid(img)
    level = len(img_pyramid)
    loss = 0.0
    for i in range(level):
        loss = loss + compute_render_loss_L1(img_pyramid[i], target_pyramid[i], weight) * (4.0 ** i)
    return loss


def compute_render_loss_pyramid_L2(img, target_pyramid, weight):
    img_pyramid = build_pyramid(img)
    level = len(img_pyramid)
    loss = 0.0
    for i in range(level):
        loss = loss + compute_render_loss_L2(img_pyramid[i], target_pyramid[i], weight) * (4.0 ** i)
    return loss
