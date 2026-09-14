"""Self-contained MiDaS v2.1 architecture without torch.hub side effects."""

import torch
from torch import nn
from torchvision.models import resnext101_32x8d


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, value):
        return nn.functional.interpolate(
            value, scale_factor=self.scale_factor, mode=self.mode,
            align_corners=self.align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        # MiDaS v2.1 applies ReLU in-place before the residual addition:
        # the skip below must contain ReLU(value), not the original value.
        # Changing this to out-of-place changes the pretrained network.
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value):
        result = self.conv1(self.relu(value))
        result = self.conv2(self.relu(result))
        return result + value


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        # Names preserve the official MiDaS v2.1 checkpoint layout.
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *values):
        result = values[0]
        if len(values) == 2:
            result = result + self.resConfUnit1(values[1])
        result = self.resConfUnit2(result)
        return nn.functional.interpolate(
            result, scale_factor=2, mode="bilinear", align_corners=True)


def _resnext101_wsl_shape_only():
    # WSL-ResNeXt101 has the same layer topology as torchvision's model. The
    # official MiDaS checkpoint supplies all parameters, so no network access
    # or ImageNet weights are needed here.
    model = resnext101_32x8d(weights=None)
    wrapped = nn.Module()
    wrapped.layer1 = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
    wrapped.layer2 = model.layer2
    wrapped.layer3 = model.layer3
    wrapped.layer4 = model.layer4
    return wrapped


def _scratch(features=256):
    result = nn.Module()
    result.layer1_rn = nn.Conv2d(256, features, 3, 1, 1, bias=False)
    result.layer2_rn = nn.Conv2d(512, features, 3, 1, 1, bias=False)
    result.layer3_rn = nn.Conv2d(1024, features, 3, 1, 1, bias=False)
    result.layer4_rn = nn.Conv2d(2048, features, 3, 1, 1, bias=False)
    result.refinenet4 = FeatureFusionBlock(features)
    result.refinenet3 = FeatureFusionBlock(features)
    result.refinenet2 = FeatureFusionBlock(features)
    result.refinenet1 = FeatureFusionBlock(features)
    result.output_conv = nn.Sequential(
        nn.Conv2d(features, 128, 3, 1, 1),
        Interpolate(2, "bilinear"),
        nn.Conv2d(128, 32, 3, 1, 1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, 1, 1, 0),
        nn.ReLU(True),
    )
    return result


class MidasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = _resnext101_wsl_shape_only()
        self.scratch = _scratch()

    def forward(self, value):
        layer1 = self.pretrained.layer1(value)
        layer2 = self.pretrained.layer2(layer1)
        layer3 = self.pretrained.layer3(layer2)
        layer4 = self.pretrained.layer4(layer3)
        path4 = self.scratch.refinenet4(self.scratch.layer4_rn(layer4))
        path3 = self.scratch.refinenet3(path4, self.scratch.layer3_rn(layer3))
        path2 = self.scratch.refinenet2(path3, self.scratch.layer2_rn(layer2))
        path1 = self.scratch.refinenet1(path2, self.scratch.layer1_rn(layer1))
        return torch.squeeze(self.scratch.output_conv(path1), dim=1)
