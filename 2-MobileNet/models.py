import torch
import torch.nn as nn

from typing import List
from functools import partial


def make_divisible(in_channel, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor

    new_ch = max(min_ch, (in_channel + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * in_channel:
        new_ch += divisor

    return int(new_ch)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer=None,
                 activation_layer=None):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU6

        self.conv1 = nn.Conv2d(in_channel,
                               out_channel,
                               (kernel_size, kernel_size),
                               (stride, stride),
                               padding=padding,
                               groups=groups,
                               )
        self.bn1 = norm_layer(out_channel)
        self.activation_layer = activation_layer()

    def forward(self, x):
        return self.activation_layer(self.bn1(self.conv1(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channel, squeeze_ratio: int = 4):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        squeeze_channel = make_divisible(in_channel / squeeze_ratio)
        self.fc1 = nn.Conv2d(in_channel, squeeze_channel, (1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channel, in_channel, (1, 1))
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.relu(self.fc1(x))
        x = self.hard_sigmoid(self.fc2(x))

        return x * identity


class InvertedResidualConfig:
    def __init__(self, in_channel, kernel_size, exp_channel, out_channel, use_se, activation, stride, width_multi):
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.exp_channel = exp_channel
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.out_channel = self.adjust_channels(out_channel, width_multi)

    @staticmethod
    def adjust_channels(channels, width_multi):
        return make_divisible(channels * width_multi)


class InvertedResidual(nn.Module):
    def __init__(self, config: InvertedResidualConfig, norm_layer):
        super(InvertedResidual, self).__init__()
        self.use_res_connect = (config.stride == 1 and config.in_channel == config.exp_channel)

        layers = []
        activation_layer = nn.Hardsigmoid if config.use_hs else nn.ReLU

        # expansion
        if config.exp_channel != config.in_channel:
            layers.append(ConvBNAct(config.in_channel,
                                    config.exp_channel,
                                    kernel_size=1,
                                    stride=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer))

        # depth-wise convolution
        layers.append(ConvBNAct(config.exp_channel,
                                config.exp_channel,
                                kernel_size=config.kernel_size,
                                stride=config.stride,
                                groups=config.exp_channel,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer))

        # SE block
        if config.use_se:
            layers.append(SEBlock(config.exp_channel))

        # 1x1 conv
        layers.append(ConvBNAct(config.exp_channel,
                                config.out_channel,
                                kernel_size=1,
                                stride=1,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer))

        self.block = nn.Sequential(*layers)
        self.out_channel = config.out_channel
        self.is_stride = config.stride > 1

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out += x

        return out


class MobileNetV3(nn.Module):
    def __init__(self,
                 ir_configs: List[InvertedResidualConfig],
                 out_channel: int = 1280,
                 num_classes: int = 1000,
                 block: InvertedResidual = None,
                 norm_layer: nn.Module = None):
        super(MobileNetV3, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []

        layers.append(ConvBNAct(3,
                                ir_configs[0].in_channel,
                                kernel_size=3,
                                stride=2,
                                norm_layer=norm_layer,
                                activation_layer=nn.Hardsigmoid))

        for config in ir_configs:
            layers.append(block(config, norm_layer=norm_layer))

        layers.append(ConvBNAct(ir_configs[-1].out_channel,
                                ir_configs[-1].out_channel * 6,
                                kernel_size=1,
                                stride=1,
                                norm_layer=norm_layer,
                                activation_layer=nn.Hardsigmoid))

        self.features = nn.Sequential(*layers)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_head = nn.Conv2d(ir_configs[-1].out_channel * 6,
                                   1280,
                                   kernel_size=1,
                                   stride=1)
        self.act = nn.Hardsigmoid()
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        out = self.act(self.conv_head(out))
        out = self.flatten(out)

        return self.classifier(out)


def mobilenet_v3_large(num_classes=1000):
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channel = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    inverted_residual_config_settings = [
        # in_c, kernel_size, exp_channel, out_channel, use_se, activation, stride, width_multi
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 3, 72, 40, True, "RE", 2),
        bneck_conf(40, 3, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 160, True, "HS", 2),
        bneck_conf(160, 3, 960, 160, True, "HS", 1),
        bneck_conf(160, 3, 960, 160, True, "HS", 1),
    ]

    return MobileNetV3(inverted_residual_config_settings)


if __name__ == '__main__':
    model = mobilenet_v3_large(1000)
    dummy_input = torch.randn((1, 3, 224, 224))

    print(model)
    print(model(dummy_input).shape)
