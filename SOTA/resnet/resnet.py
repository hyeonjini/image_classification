import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut == nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(x)))

        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)
        
        out += x
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, block_outs, ydim=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(block, block_outs[0], num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, block_outs[1], num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, block_outs[2], num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, block_outs[3], num_blocks[3], stride=2)

        self.linear = nn.Linear(512, ydim)

    def make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], [54, 128, 256, 512])