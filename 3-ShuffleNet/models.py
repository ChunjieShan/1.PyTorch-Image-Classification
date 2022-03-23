import torch
import torch.nn as nn

from typing import List


def channel_shuffle(x: torch.Tensor, groups: int):
    batch_size, in_channel, height, width = x.shape
    channel_per_group = in_channel // groups

    x = x.view(batch_size, groups, channel_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batch_size, in_channel, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super(InvertedResidual, self).__init__()
        split_channels = out_channel // 2
        self.stride = stride

        if stride == 2:
            self.branch_left = nn.Sequential(*[
                self.dw_conv(in_channel, in_channel, 3, self.stride, 1),
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, split_channels, (1, 1), (1, 1)),
                nn.BatchNorm2d(split_channels),
                nn.ReLU()
            ])

        else:
            self.branch_left = None

        self.branch_right = nn.Sequential(*[
            nn.Conv2d(in_channel if stride == 2 else split_channels, split_channels, (1, 1), (1, 1), 0),
            nn.BatchNorm2d(split_channels),
            nn.ReLU(),
            self.dw_conv(split_channels, split_channels, 3, self.stride, 1),
            nn.BatchNorm2d(split_channels),
            nn.Conv2d(split_channels, split_channels, (1, 1), (1, 1), 0),
            nn.BatchNorm2d(split_channels),
            nn.ReLU()
        ])

    def forward(self, x: torch.Tensor):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.concat((x1, self.branch_right(x2)), 1)

        else:
            out = torch.concat((self.branch_left(x), self.branch_right(x)), 1)

        out = channel_shuffle(out, 2)
        return out

    @staticmethod
    def dw_conv(in_channel: int, out_channel: int, kernel_size: int, stride: int, padding: int = 0):
        return nn.Conv2d(in_channels=in_channel,
                         out_channels=out_channel,
                         kernel_size=(kernel_size, kernel_size),
                         stride=(stride, stride),
                         padding=padding,
                         groups=in_channel)


class ShuffleNetV2(nn.Module):
    def __init__(self, blocks_num: List[int], out_channels_num: List[int], num_classes: int, block=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        in_channel = 3
        out_channel = out_channels_num[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), (2, 2), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        in_channel = out_channel

        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]

        for name, block_num, output_channel in zip(stage_names, blocks_num, out_channels_num[1:]):
            sequences = [block(in_channel, output_channel, 2)]

            for i in range(block_num - 1):
                sequences.append(block(output_channel, output_channel, 1))

            setattr(self, name, nn.Sequential(*sequences))
            in_channel = output_channel

        out_channel = out_channels_num[-1]

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1), padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channel, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pooling(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.avg_pooling(out)
        out = torch.flatten(out, 1)

        return self.fc(out)


def shufflenetv2_1x():
    model = ShuffleNetV2(blocks_num=[4, 8, 4], out_channels_num=[24, 116, 232, 464, 1024], num_classes=1000)

    return model


if __name__ == '__main__':
    dummy_input = torch.randn((1, 3, 224, 224))
    model = shufflenetv2_1x()
    print(model(dummy_input).shape)
    # dummy_input = channel_shuffle(dummy_input, 3)
