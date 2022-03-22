import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(in_channel, out_channel, (3, 3), (stride, stride), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, (3, 3), (1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.down_sample:
            identity = self.down_sample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        return self.relu(out)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BottleNeck, self).__init__()
        self.down_sample = down_sample

        self.conv1 = nn.Conv2d(in_channel, out_channel, (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, (3, 3), (stride, stride), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, (1, 1), (1, 1))
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        if self.down_sample:
            identity = self.down_sample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += identity

        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block=None, blocks_num=None, num_classes=1000, include_top=False):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.block = block
        self.blocks_num = blocks_num
        self.num_classes = num_classes
        self.include_top = include_top

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, self.in_channel, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
        ])

        self.block1 = self._make_layer(64, block, self.blocks_num[0], stride=1)
        self.block2 = self._make_layer(128, block, self.blocks_num[1])
        self.block3 = self._make_layer(256, block, self.blocks_num[2])
        self.block4 = self._make_layer(512, block, self.blocks_num[3])

        if include_top:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def _make_layer(self, out_channel, block, block_num, stride=2):
        down_sample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            down_sample = nn.Sequential(*[
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(out_channel * block.expansion),
            ])

        layers = [block(self.in_channel, out_channel, stride=stride, down_sample=down_sample)]
        self.in_channel = out_channel * block.expansion

        for i in range(1, block_num):
            layers.append(
                block(self.in_channel, out_channel)
            )

        return nn.Sequential(*layers)


if __name__ == '__main__':
    resnet = ResNet(BottleNeck, [3, 4, 6, 3], include_top=True)
    dummy_x = torch.randn((1, 3, 224, 224))

    print(resnet(dummy_x).shape)
    print(resnet)
