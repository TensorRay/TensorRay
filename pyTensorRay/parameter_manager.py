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
import torch

from pyTensorRay.common import get_ptr


class SceneParameterManager:
    """
    Data layout of unknown scene parameters
    ----------------------------------------------
    |     shape params     |     BSDF params     |
    ----------------------------------------------
    ^                      ^
    param_id_offsets[0]    param_id_offsets[1]
    """
    def __init__(self, param_id_offsets=[0]):
        self.param_group_count = len(param_id_offsets)
        self.param_id_offsets = param_id_offsets
        self.param_count_list = []
        self.total_param_count = TR.get_size_param(self.param_count_list)

        # group parameters
        self.param_counts = []
        self.params = []
        for i_group, val_start in enumerate(param_id_offsets):
            if i_group == self.param_group_count - 1:
                val_end = len(self.param_count_list)
            else:
                val_end = self.param_id_offsets[i_group + 1]
            self.param_counts.append(sum(self.param_count_list[val_start:val_end]))
            self.params.append(torch.tensor([0.0] * self.param_counts[i_group], requires_grad=True))
            assert(self.params[i_group].is_contiguous())

    def get_values(self):
        for i_group, val_start in enumerate(self.param_id_offsets):
            if i_group == self.param_group_count - 1:
                val_end = len(self.param_count_list)
            else:
                val_end = self.param_id_offsets[i_group + 1]

            offset = 0
            for i in range(val_start, val_end):
                TR.get_var(i, get_ptr(self.params[i_group]), offset)
                offset += self.param_count_list[i]

    def set_values(self):
        for i_group, val_start in enumerate(self.param_id_offsets):
            if i_group == self.param_group_count - 1:
                val_end = len(self.param_count_list)
            else:
                val_end = self.param_id_offsets[i_group + 1]

            offset = 0
            for i in range(val_start, val_end):
                TR.set_var(i, get_ptr(self.params[i_group]), offset)
                offset += self.param_count_list[i]

    def get_grads(self):
        res = []
        for i_group, val_start in enumerate(self.param_id_offsets):
            if i_group == self.param_group_count - 1:
                val_end = len(self.param_count_list)
            else:
                val_end = self.param_id_offsets[i_group + 1]

            grad = torch.tensor([0.0] * self.param_counts[i_group])
            offset = 0
            for i in range(val_start, val_end):
                TR.get_grad(i, get_ptr(grad), offset, True)
                offset += self.param_count_list[i]
            res.append(grad)
        return res

    def __str__(self):
        st = "Total param count = {:d}\n".format(self.total_param_count)
        st += "Param list:" + str(self.param_count_list) + "\n"
        st += "Param group count = {:d}\n".format(self.param_group_count)
        st += "Grouped params:" + str(self.param_counts)
        return st
