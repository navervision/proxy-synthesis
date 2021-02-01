'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from __future__ import print_function, division, absolute_import
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101

model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101}
output_dims = {
    'resnet18': 512,
    'resnet50': 2048}

class Resnet(nn.Module):
    def __init__(self, pretrained=True, resnet_type='resnet50'):
        super(Resnet, self).__init__()

        self.model = model_dict[resnet_type](pretrained)
        
        self.output_dim = output_dims[resnet_type]   

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        return x

