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
from torch import Tensor
from typing import List
import numpy as np

from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential


def adamax(original_params: List[Tensor],
           params: List[Tensor],
           grads: List[Tensor],
           m1_tp: List[Tensor],
           m2_tp: List[Tensor],
           state_steps: List[int],
           *,
           beta1: float,
           beta2: float,
           lr: float,
           IL_term, IL_solver):
    r"""Functional API that performs adamax algorithm computation.
    See :class:`~torch.optim.Adamax` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        m1_tp = m1_tp[i]
        m2_tp = m2_tp[i]
        step = state_steps[i]

        grad = torch.as_tensor(IL_solver(np.asarray(grad)))
        m1_tp.mul_(beta1).add_(grad, alpha=1 - beta1)
        m2_tp.mul_(beta2).add_(grad.square(), alpha=1 - beta2)
        u = torch.matmul(IL_term, param.detach())
        clr = lr / ((1-beta1 ** step) * (m2_tp.amax() /
                    (1-beta2 ** step)).sqrt()) * m1_tp
        u = u - clr
        new_param = torch.as_tensor(IL_solver(np.asarray(u)))
        new_param = new_param.transpose(0, 1).view(-1).contiguous()  # reshape from [n, 3] to [3, n] to [3 * n]
        original_params[i].copy_(new_param)


class LGDescent(torch.optim.Optimizer):
    """Take a coordinate descent step for a random parameter.
    And also, make every 100th step way bigger.
    """

    def __init__(self, params, IL_term, IL_solver, lr=2e-3, betas=(0.9, 0.999)):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas)
        self.IL_term = IL_term
        self.IL_solver = IL_solver
        super(LGDescent, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            original_params = []
            params_reshaped = []
            grads = []
            m1_tp = []
            m2_tp = []
            state_steps = []

            beta1, beta2 = group['betas']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                original_params.append(p)
                params_reshaped.append(p.view(3, -1).transpose(0, 1))  # reshape from [3 * n] to [3, n] to [n, 3]
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'Adamax does not support sparse gradients')
                grads.append(p.grad.view(3, -1).transpose(0, 1))

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m1_tp'] = torch.zeros_like(
                        grads[-1], memory_format=torch.preserve_format)
                    state['m2_tp'] = torch.zeros_like(
                        grads[-1], memory_format=torch.preserve_format)

                m1_tp.append(state['m1_tp'])
                m2_tp.append(state['m2_tp'])

                state['step'] += 1
                state_steps.append(state['step'])

            adamax(original_params,
                   params_reshaped,
                   grads,
                   m1_tp,
                   m2_tp,
                   state_steps,
                   beta1=beta1,
                   beta2=beta2,
                   lr=lr,
                   IL_term=self.IL_term, IL_solver=self.IL_solver)

        return loss


class LargeSteps(torch.optim.Optimizer):
    def __init__(self, V, F, scene_param_manager, lr=0.1, betas=(0.9, 0.999), lmbda=0.1):
        self.scene_param_manager = scene_param_manager
        vertex_count = self.scene_param_manager.param_counts[0]
        self.V = V[0]
        # reshape from [3 * n] to [3, n] to [n, 3]
        V_pos = self.V.view(3, -1).transpose(0, 1).contiguous().cuda()
        self.F = F.cuda()
        self.M = compute_matrix(V_pos, self.F, lmbda)
        self.u = to_differential(self.M, V_pos.detach()).clone().detach().requires_grad_(True)
        defaults = dict(F=self.F, lr=lr, betas=betas)
        self.optimizer = AdamUniform([self.u], lr=lr, betas=betas)
        super(LargeSteps, self).__init__(V, defaults)

    def step(self):
        # build compute graph from u to V
        V_pos = from_differential(self.M, self.u, 'Cholesky')
        # propagate gradients from V to u
        # reshape from [3 * n] to [3, n] to [n, 3]
        V_grad = self.V.grad.view(3, -1).transpose(0, 1).contiguous().cuda()
        V_pos.backward(V_grad)
        # step u
        self.optimizer.step()
        # update param
        V_pos = from_differential(self.M, self.u, 'Cholesky')
        # reshape from [n, 3] to [3, n] to [3 * n]
        V_pos = V_pos.transpose(0, 1).reshape(-1).contiguous().cpu()
        self.V.data.copy_(V_pos)

    def zero_grad(self):
        super(LargeSteps, self).zero_grad()
        self.optimizer.zero_grad()
