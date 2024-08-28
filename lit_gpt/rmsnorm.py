# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
# Copyright (c) 2022, Tri Dao.
# Adapted from https://github.com/Dao-AILab/flash-attention/blob/7a983df74215e035e566e37125b0a71e3618f39d/flash_attn/ops/layer_norm.py#L16

import torch
from torch.nn import init

    
class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

try:
    import apex
    class FusedRMSNorm(apex.normalization.FusedRMSNorm):
        def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
            super().__init__(size, eps=eps, elementwise_affine=True)
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(size))
            self.dim = dim
            self.reset_parameters()

        def reset_parameters(self):
            init.ones_(self.weight)

        # def forward(self, x):
        #     return rms_norm(x, self.weight, self.eps)
except:
    print("Fail to import FusedRMSNorm, use RMSNorm instead.")
    FusedRMSNorm = RMSNorm
