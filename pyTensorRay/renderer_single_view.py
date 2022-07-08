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
import numpy as np
import TensorRay as TR

from pyTensorRay.common import get_ptr
from pyTensorRay.utils import image_tensor_to_torch, image_torch_to_tensor, save_torch_image
from pyTensorRay.optimizer import LargeSteps
from pyTensorRay.fwd import *


class RenderFunctionSingleView(torch.autograd.Function):
    """for optimization of a single shape"""
    @staticmethod
    def forward(ctx, V: torch.Tensor, context, params):
        ctx.context = context
        ctx.context.update(params)  # update the default context

        scene = context["scene"]
        integrator = context["integrator"]
        options = context["options"]
        seed = context["seed"]

        # render
        height, width = scene.cameras[0].height, scene.cameras[0].width
        TR.set_rnd_seed(seed)
        img_tensor = integrator.renderC(scene, options)
        img = image_tensor_to_torch(img_tensor, height, width)
        return img.clone().detach()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        context = ctx.context
        scene = context["scene"]
        scene_param_manager = context["scene_param_manager"]
        integrator = context["integrator"]
        edge_integrators = context["edge_integrators"]
        options = context["options"]
        seed = context["seed"]

        height, width = scene.cameras[0].height, scene.cameras[0].width
        dLdI = image_torch_to_tensor(grad_out, height, width)
        #save_torch_image("D:\\TensorRay\\example\\inverse_rendering\\output\\triangle_in_cbox\\iter\\dLdI.exr", grad_out)

        TR.set_rnd_seed(seed)
        integrator.renderD(scene, options, dLdI)
        for edge_integrator in edge_integrators:
            edge_integrator.renderD(scene, options, dLdI)

        # Note: here we hardcode grad[0] as the gradient of shape parameters!
        grads = scene_param_manager.get_grads()
        grad_vertex = grads[0]
        print(grad_vertex)
        grad_vertex[grad_vertex.isnan()] = 0.0

        result = (grad_vertex, None, None)
        return result


class RenderSingleView(torch.nn.Module):
    """for optimization of a single shape"""
    def __init__(self, scene, scene_param_manager,
                 integrator, edge_integrators, options, guiding_options):
        super(RenderSingleView, self).__init__()
        self.context = dict(
            scene=scene,
            scene_param_manager=scene_param_manager,
            integrator=integrator,
            edge_integrators=edge_integrators,
            options=options,
            guiding_options=guiding_options,
            seed=1234
        )

    def set_state(self, state):
        self.context.update(state)

    def forward(self, V: torch.Tensor, params={}):
        return RenderFunctionSingleView.apply(V, self.context, params)
