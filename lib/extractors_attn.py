from collections import OrderedDict
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# This function takes a target model and a source state dictionary,
# It uses an 'OrderedDict' to match the keys in the source and
# target state dictionaries
# ------------------------------------------------------------------

def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, v1), (k2, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)

# ------------------------------------------------------------------
# A helper function, that defines a 3x3 convolution layer with
# optional parameters for stride and dilation.
# ------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

# ------------------------------------------------------------------
# BasicBlock and Bottleneck are both building blocks for ResNet
# architectures. BasicBlock is for ResNet18 and ResNet34 while
# Bottleneck is for ResNet50, ResNet152. They define the basic
# structure of a residual block, consisting of one or more 
# conv layers.
# ------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ------------------------------------------------------------------
# Defines the ResNet architecture for RGB images. It consists of 
# layers with different numbers of blocks ('layers' parameter) and
# employs the previously defined 'BasicBlock' and 'Bottleneck'
# classes. It also includes custom weight initialization for 
# convolutional and batch normalization layers.
# ------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

# ------------------------------------------------------------------
# The weight initialization for convolutional layers follows a
# strategy where the weights are initialized with values sampled 
# from a normal distribution. The standard deviation is adjusted
# based on the size of the kernel and the number of output channels
# to promote stable training.
# ------------------------------------------------------------------

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def resnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    return model

# ------------------------------------------------------------------
# Similar to 'ResNet' but designed for 4-channel images 
# (RGB-D images). It has a modified 1st convolutional layer to
# accomodate the additional channel.
# ------------------------------------------------------------------

class ResNet4Ch(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        #super(ResNet, self).__init__()
        super(ResNet4Ch, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3


def resnet18_4Ch(pretrained=False):
    model = ResNet4Ch(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34_4Ch(pretrained=False):
    model = ResNet4Ch(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50_4Ch(pretrained=False):
    model = ResNet4Ch(Bottleneck, [3, 4, 6, 3])
    return model

def resnet101_4Ch(pretrained=False):
    model = ResNet4Ch(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152_4Ch(pretrained=False):
    model = ResNet4Ch(Bottleneck, [3, 8, 36, 3])
    return model





class ResNetDepth(nn.Module):
    def __init__(self, block, layers=(3, 4, 23, 3)):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)

        return x, x_3


# def resnet18Depth(pretrained=False):
#     model = ResNet4Ch(BasicBlock, [2, 2, 2, 2])
#     return model

# def resnet34Depth(pretrained=False):
#     model = ResNet4Ch(BasicBlock, [3, 4, 6, 3])
#     return model

# def resnet50Depth(pretrained=False):
#     model = ResNet4Ch(Bottleneck, [3, 4, 6, 3])
#     return model

def resnet18Depth(pretrained=False):
    model = ResNetDepth(BasicBlock, [2, 2, 2, 2])
    return model

def resnet34Depth(pretrained=False):
    model = ResNetDepth(BasicBlock, [3, 4, 6, 3])
    return model

def resnet50Depth(pretrained=False):
    model = ResNetDepth(Bottleneck, [3, 4, 6, 3])
    return model


def resnet101Depth(pretrained=False):
    model = ResNetDepth(Bottleneck, [3, 4, 23, 3])
    return model

def resnet152Depth(pretrained=False):
    model = ResNetDepth(Bottleneck, [3, 8, 36, 3])
    return model