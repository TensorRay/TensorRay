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

import dataclasses
import TensorRay as TR
import gin
import os
import numpy as np
from typing import List


class RenderBatchOptions:
    spp_batch = 1
    sppe_batch = 1
    sppse0_batch = 1
    sppse1_batch = 1
    sppe0_batch = 1


class GuidingOptions:
    g_direct = False
    g_direct_max_size = 100000
    g_direct_spp = 16
    g_eps = 0.0


@gin.configurable
def RenderOptions(seed, max_bounce, spp, sppe, sppse0, sppse1, sppe0,
                  spp_batch = 1, sppe_batch = 1, sppse0_batch = 1, sppse1_batch = 1, sppe0_batch = 1,
                  g_direct = False, g_direct_max_size = 100000, g_direct_spp = 16, g_eps = 0.0,
                  quiet = False):
    options = TR.RenderOptions(seed, max_bounce, spp, sppe, sppse0, sppse1, sppe0)
    options.spp_batch = spp_batch
    options.sppe_batch = sppe_batch
    options.sppse0_batch = sppse0_batch
    options.sppse1_batch = sppse1_batch
    options.sppe0_batch = sppe0_batch

    # guiding
    options.g_direct = g_direct
    options.g_direct_max_size = g_direct_max_size
    options.g_direct_spp = g_direct_spp
    options.g_eps = g_eps

    options.quiet = quiet
    return options


@gin.configurable
def RenderBatchOptions(spp_batch = 1, sppe_batch = 1, sppse0_batch = 1, sppse1_batch = 1, sppe0_batch = 1):
    options = RenderBatchOptions
    options.spp_batch = spp_batch
    options.sppe_batch = sppe_batch
    options.sppse0_batch = sppse0_batch
    options.sppse1_batch = sppse1_batch
    options.sppe0_batch = sppe0_batch
    return options


@gin.configurable
def GuidingOptions(g_direct = False, g_direct_max_size = 100000, g_direct_spp = 16, g_eps = 0.0):
    options = GuidingOptions
    options.g_direct = g_direct
    options.g_direct_max_size = g_direct_max_size
    options.g_direct_spp = g_direct_spp
    options.g_eps = g_eps
    return options


@gin.configurable
def Path():
    return TR.PathTracer()


@gin.configurable
def PTracer():
    return TR.ParticleTracer()


@gin.configurable
def PrimaryEdgeIntegrator():
    return TR.PrimaryEdgeIntegrator()


@gin.configurable
def DirectEdgeIntegrator():
    return TR.DirectEdgeIntegrator()


@gin.configurable
def IndirectEdgeIntegrator():
    return TR.IndirectEdgeIntegrator()


@gin.configurable
def PixelBoundaryIntegrator():
    return TR.PixelBoundaryIntegrator()


@gin.configurable
def Scene(file_name):
    cur_dir = os.getcwd()
    os.chdir(os.path.dirname(file_name))
    scene = TR.Scene()
    scene.load_file(os.path.basename(file_name))
    os.chdir(cur_dir)
    return scene


@gin.configurable
@dataclasses.dataclass
class Camera:
    origin: List = dataclasses.field(default_factory=lambda: [1.0, 1.0, 1.0])
    target: List = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    up: List = dataclasses.field(default_factory=lambda: [0.0, 0.0, 1.0])
    fov: float = 10.0


# ================== Transformation ===================

@gin.configurable
class Transformation:
    pass

@gin.configurable
@dataclasses.dataclass
class Translation(Transformation):
    translation: list
    xform_type = "translate"

@gin.configurable
@dataclasses.dataclass
class Rotation(Transformation):
    rotation: list
    xform_type = "rotate"

@gin.configurable
@dataclasses.dataclass
class Transform:
    shape_id: int = -1
    vertex_id: int = -1  # -1 means all vertices
    transformation: Transformation = None
