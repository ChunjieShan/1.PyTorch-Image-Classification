import copy

import torch
import torch.nn as nn
import math

from functools import partial


def _make_divisible(input_channel, divisor: int = 8, min_channel: int = None):
    if min_channel is None:
        min_channel = divisor

    new_channel = max(int(input_channel + divisor / 2) // divisor * divisor, min_channel)

    if new_channel < 0.9 * input_channel:
        new_channel += divisor

    return new_channel


class DropPath(nn.Module):
    def __init__(self, drop_rate):
        super(DropPath, self).__init__()
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor):
        if self.drop_rate == 0:
            return x
        keep_rate = 1 - self.drop_rate
        shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
        random_tensor = torch.randn(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_rate) * random_tensor

        return output


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: nn.Module = None,
                 act_layer: nn.Module = None):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              (kernel_size, kernel_size),
                              (stride, stride),
                              padding,
                              groups=groups)

        if norm_layer is None:
            self.bn = nn.BatchNorm2d(out_channel)

        else:
            self.bn = norm_layer(out_channel)

        if act_layer is None:
            self.act = nn.SiLU()

        else:
            self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out


class SEBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 expand_channel: int,
                 squeeze_factor: int = 4):
        super(SEBlock, self).__init__()
        squeeze_channel = in_channel // squeeze_factor

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(expand_channel, squeeze_channel, (1, 1))
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channel, expand_channel, (1, 1))
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.act1(self.fc1(out))
        out = self.act2(self.fc2(out))

        return out * x


class InvertedResidualConfig:
    def __init__(self,
                 kernel_size: int,
                 in_channel: int,
                 out_channel: int,
                 expand_ratio: int,
                 stride: int,
                 use_se: bool,
                 drop_rate: float,
                 width_coef: float):
        self.in_channel = self.adjust_channels(in_channel, width_coef)
        self.expand_channel = in_channel * expand_ratio
        self.out_channel = self.adjust_channels(out_channel, width_coef)
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        self.drop_rate = drop_rate

    @staticmethod
    def adjust_channels(channels: int, width_coef: float):
        return _make_divisible(channels * width_coef, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 config: InvertedResidualConfig,
                 norm_layer: nn.Module):
        super(InvertedResidual, self).__init__()
        self.use_res = (config.stride == 1 and config.in_channel == config.out_channel)
        layers = []

        act_layer = nn.SiLU

        # Expand Conv
        if config.expand_channel != config.in_channel:
            layers.append(ConvBNAct(config.in_channel,
                                    config.expand_channel,
                                    kernel_size=1,
                                    norm_layer=norm_layer,
                                    act_layer=act_layer))

        # Depthwise Conv
        layers.append(ConvBNAct(config.expand_channel,
                                config.expand_channel,
                                kernel_size=config.kernel_size,
                                stride=config.stride,
                                groups=config.expand_channel,
                                norm_layer=norm_layer,
                                act_layer=act_layer))

        if config.use_se:
            layers.append(SEBlock(config.in_channel,
                                  config.expand_channel))

        # Project Conv
        layers.append(ConvBNAct(config.expand_channel,
                                config.out_channel,
                                kernel_size=1,
                                stride=1,
                                norm_layer=norm_layer,
                                act_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channel = config.out_channel
        self.is_strided = config.stride == 2

        if self.use_res and config.drop_rate > 0:
            self.dropout = DropPath(config.drop_rate)

        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        if self.use_res:
            out += x

        return out


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coef: float,
                 depth_coef: float,
                 num_classes: int,
                 dropout_rate: float = 0.2,
                 block: nn.Module = InvertedResidual,
                 norm_layer: nn.Module = nn.BatchNorm2d):
        super(EfficientNet, self).__init__()
        default_cnf = [[3, 32, 16, 1, 1, True, dropout_rate, 1],
                       [3, 16, 24, 6, 2, True, dropout_rate, 2],
                       [5, 24, 40, 6, 2, True, dropout_rate, 2],
                       [3, 40, 80, 6, 2, True, dropout_rate, 3],
                       [5, 80, 112, 6, 1, True, dropout_rate, 3],
                       [5, 112, 192, 6, 2, True, dropout_rate, 4],
                       [3, 192, 320, 6, 1, True, dropout_rate, 1]]

        def round_repeat(repeat):
            return int(math.ceil(depth_coef * repeat))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_coef)
        b = 0
        bneck_conf = partial(InvertedResidualConfig, width_coef=width_coef)

        num_blocks = float(sum(round_repeat(config[-1]) for config in default_cnf))
        inverted_residual_settings = []
        for stage, args in enumerate(default_cnf):
            config = copy.copy(args)
            for i in range(round_repeat(config.pop(-1))):
                if i > 0:
                    config[-3] = 1  # stride
                    config[1] = config[2]  # make in_channel equals to out_channel

                config[-1] = args[-2] * b / num_blocks
                inverted_residual_settings.append(bneck_conf(*config))
                b += 1

        layers = [ConvBNAct(in_channel=3,
                            out_channel=adjust_channels(32),
                            kernel_size=3,
                            stride=2,
                            norm_layer=norm_layer)]

        for config in inverted_residual_settings:
            layers.append(block(config, norm_layer))

        last_conv_input = inverted_residual_settings[-1].out_channel
        last_conv_output = adjust_channels(1280)

        layers.append(ConvBNAct(in_channel=last_conv_input,
                                out_channel=last_conv_output,
                                kernel_size=1,
                                norm_layer=norm_layer))

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(last_conv_output, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def efficient_net_v2_b0(num_classes=1000):
    return EfficientNet(width_coef=1.0,
                        depth_coef=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


if __name__ == '__main__':
    # test = ConvBNAct(3, 32)
    model = efficient_net_v2_b0(1000)
    print(model)
    dummy_input = torch.randn((1, 3, 224, 224))
    print(model(dummy_input).shape)

