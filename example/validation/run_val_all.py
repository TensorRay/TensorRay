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

import os, sys
sys.path.insert(1, "../../")

import torch

from pyTensorRay.utils import load_torch_image


def gen_images():
    for filename in os.listdir("."):
        if filename.endswith(".conf"):
            cmd = "python validate_gpu.py "
            #cmd = "python validate_cpu.py "
            for suffix in [" forward", " backward"]: #, " fd"]:
                print(cmd + filename + suffix)
                os.system(cmd + filename + suffix)


def compare():
    ref_dir = "./output_TR/"
    out_dir = "./output/"

    for dir_name in os.listdir(ref_dir):
        for file_name in os.listdir(os.path.join(ref_dir, dir_name)):
            ref_file_name = os.path.join(ref_dir, dir_name, file_name)
            img_file_name = os.path.join(out_dir, dir_name, file_name)
            ref = load_torch_image(ref_file_name)
            img = load_torch_image(img_file_name)
            diff = torch.abs(ref - img).mean()
            if diff > 1e-5:
                print("Images don't match: ", diff,  img_file_name)

    print("Finish comparing.")


if __name__ == "__main__":
    gen_images()
    compare()
