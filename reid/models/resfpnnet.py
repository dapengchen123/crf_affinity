from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['ResFpnNet', 'resfpnnet18', 'resfpnnet34', 'resfpnnet50', 'resfpnnet101',
           'resfpnnet152']

class ResFpnNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth,  cut_at_pooling=False, num_features=0, dropout=0):

        super(ResFpnNet, self).__init__()

        self.depth = depth
        self.pretrained = True
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resfpnnet
        if depth not in ResFpnNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResFpnNet.__factory[depth](pretrained=True)
        self.num_feature = num_features
        self.dropout = dropout

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)




        #  Append new layers
        ## 1 smallest layers  after pool 8x4
        ## 2 smallest layers  after pool 16x8
        ## 3 smallest layers  after pool 32x16

        self.inplanes2  = self.base.layer2[3].bn3.num_features
        self.inplanes3  = self.base.layer3[5].bn3.num_features
        self.inplanes4 = self.base.layer4[2].bn3.num_features
        self.outplanes2 = self.inplanes2 * 16
        self.outplanes3 = self.inplanes3 * 4
        self.outplanes4 = self.inplanes4

        ## 1x1 convolution
        self.conv43 = nn.Conv2d(self.inplanes4, self.inplanes3, kernel_size=1, stride=1, padding=0)
        self.bn43 = nn.BatchNorm2d(self.inplanes3)
        self.conv32 = nn.Conv2d(self.inplanes3, self.inplanes2, kernel_size=1, stride=1, padding=0)
        self.bn32 = nn.BatchNorm2d(self.inplanes2)

        ## Upsampling
        self.Upsample43 = nn.UpsamplingNearest2d(scale_factor=2)
        self.Upsample32 = nn.UpsamplingNearest2d(scale_factor=2)

        ## FCN

        self.feat2 = nn.Linear(self.outplanes2, self.num_feature)
        self.feat3 = nn.Linear(self.outplanes3, self.num_feature)
        self.feat4 = nn.Linear(self.outplanes4, self.num_feature)
        self.feat2_bn = nn.BatchNorm1d(self.num_feature)
        self.feat3_bn = nn.BatchNorm1d(self.num_feature)
        self.feat4_bn = nn.BatchNorm1d(self.num_feature)

       ## initial other parameters
       ## init 1x1 convolution
        init.kaiming_normal(self.feat2.weight, mode='fan_out')
        init.constant(self.feat2.bias, 0)

        init.kaiming_normal(self.feat3.weight, mode='fan_out')
        init.constant(self.feat3.bias, 0)

        init.kaiming_normal(self.feat4.weight, mode='fan_out')
        init.constant(self.feat4.bias, 0)

        init.constant(self.feat2_bn.weight, 1)
        init.constant(self.feat2_bn.bias, 0)
        init.constant(self.feat3_bn.weight, 1)
        init.constant(self.feat3_bn.bias, 0)
        init.constant(self.feat4_bn.weight, 1)
        init.constant(self.feat4_bn.bias, 0)

        ## init 1x1 conbolution
        init.kaiming_normal(self.conv43.weight, mode='fan_out')
        init.constant(self.conv43.bias, 0)

        init.kaiming_normal(self.conv32.weight, mode='fan_out')
        init.constant(self.conv32.bias, 0)

        init.constant(self.bn43.weight, 1)
        init.constant(self.bn43.bias, 0)

        init.constant(self.bn32.weight, 1)
        init.constant(self.bn32.bias, 0)


    def forward(self, x):

        for name, module in self.base._modules.items():

            if name == 'layer3':
                xlayer2 = x

            if name == 'layer4':
                xlayer3 = x

            if name == 'avgpool':
                break
            x = module(x)

        ## x : layer4
        ### average pooling
        layer4branch = F.avg_pool2d(x, x.size()[2:])
        layer4branch = layer4branch.view(layer4branch.size(0), -1)
        layer4branch = self.feat4(layer4branch)
        layer4branch = self.feat4_bn(layer4branch)
        layer4branch = layer4branch/layer4branch.norm(2, 1).unsqueeze(1).expand_as(layer4branch)

        if self.dropout > 0:
            layer4branch = self.drop(layer4branch)

        ##  convolution upsampling + skip connection
        x = self.Upsample43(x)
        x = self.conv43(x)
        x = x + xlayer3

        ### 2x2 block average pooling
        height3 = x.size()[2]
        width3 = x.size()[3]
        split = 2
        height3_step = height3/split
        width3_step = width3/split

        layer3branch = 0
        for ih in range(split):
            for iw in range(split):
                hstart = int(ih*height3_step)
                hend = int((ih+1)*height3_step)
                wstart = int(iw*width3_step)
                wend = int((iw+1)*width3_step)
                xblock = x[:, :, hstart:hend, wstart:wend]
                xblock = F.avg_pool2d(xblock, xblock.size()[2:])
                if ih == 0 and iw == 0:
                    layer3branch = xblock
                else:
                    layer3branch = torch.cat((layer3branch, xblock), 1)

        layer3branch = layer3branch.view(layer3branch.size(0), -1)
        layer3branch = self.feat3(layer3branch)
        layer3branch = self.feat3_bn(layer3branch)
        layer3branch = layer3branch/layer3branch.norm(2, 1).unsqueeze(1).expand_as(layer3branch)
        if self.dropout > 0:
            layer3branch = self.drop(layer3branch)

        ## convolution upsampling _ skip connection
        x = self.Upsample32(x)
        x = self.conv32(x)
        x = x + xlayer2

        ##  4x4 block average pooling

        height2 = x.size()[2]
        width2 = x.size()[3]
        split = 4
        height2_step = height2/split
        width2_step = width2/split

        layer2branch = 0
        for ih in range(split):
            for iw in range(split):
                hstart = int(ih*height2_step)
                hend = int((ih+1)*height2_step)
                wstart = int(iw*width2_step)
                wend = int((iw+1)*width2_step)
                xblock = x[:, :, hstart:hend, wstart:wend]
                xblock = F.avg_pool2d(xblock, xblock.size()[2:])
                if ih == 0 and iw == 0:
                    layer2branch = xblock
                else:
                    layer2branch = torch.cat((layer2branch, xblock), 1)

        layer2branch = layer2branch.view(layer2branch.size(0), -1)
        layer2branch = self.feat2(layer2branch)
        layer2branch = self.feat2_bn(layer2branch)
        layer2branch = layer2branch/layer2branch.norm(2, 1).unsqueeze(1).expand_as(layer2branch)
        if self.dropout > 0:
            layer2branch = self.drop(layer2branch)

        return layer4branch, layer3branch, layer2branch


def resfpnnet18(**kwargs):
    return ResFpnNet(18, **kwargs)


def resfpnnet34(**kwargs):
    return ResFpnNet(34, **kwargs)


def resfpnnet50(**kwargs):
    return ResFpnNet(50, **kwargs)


def resfpnnet101(**kwargs):
    return ResFpnNet(101, **kwargs)


def resfpnnet152(**kwargs):
    return ResFpnNet(152, **kwargs)

