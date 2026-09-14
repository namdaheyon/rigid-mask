"""DCNv2-compatible module backed by torchvision's packaged operator."""

import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


class DCN(nn.Module):
    """Drop-in replacement for the DCN class used by RigidMask's DLA network.

    Parameter names intentionally match the original extension so its
    checkpoint (`weight`, `bias`, `conv_offset_mask.*`) loads unchanged.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        channels = deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels, channels, kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, value):
        combined = self.conv_offset_mask(value)
        offset_x, offset_y, mask = torch.chunk(combined, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        return deform_conv2d(
            value, offset, self.weight, self.bias,
            self.stride, self.padding, self.dilation, torch.sigmoid(mask))
