# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:12:44 2022

@author: phy71
"""

import torch
import torchvision
import torch.nn as nn


# hidden_length = 256
def get_model(name):
    model = None

    assert type(name) == str, 'Require a str as the name of the model'

    if name == 'TridentNetwork':
        model = TridentNetwork()
    elif name == 'Mobilenet':
        model = Mobilenet()
    elif name == 'Efficientnet':
        model = Efficientnet()
    elif name == 'SwinTransformer':
        model = SwinTransformer()
    else:
        print("There is no such model")
    return model


class SwinTransformer(nn.Module):
    def __init__(self, num_class=2):
        super(SwinTransformer, self).__init__()
        self.is_freeze = False
        self.num_class = num_class
        self.base = torchvision.models.swin_t(weights='IMAGENET1K_V1')
        self.base.head = nn.Identity()
        self.fc = nn.Linear(768, num_class)

    def forward(self, x):
        h = self.base(x)
        y = self.fc(h)
        return y

    def set_freeze(self, is_freeze=False):
        assert type(is_freeze) == bool, 'Wrong type of variable,requires bool'
        self.is_freeze = is_freeze
        if self.is_freeze:
            for parameter in self.base.parameters():
                parameter.requires_grad = self.is_freeze
        else:
            pass


class Efficientnet(nn.Module):
    def __init__(self, num_class=2):
        super(Efficientnet, self).__init__()
        self.is_freeze = False
        self.num_class = num_class
        self.base = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        self.base.classifier = nn.Identity()
        self.fc = nn.Linear(1280, num_class)

    def forward(self, x):
        h = self.base(x)
        y = self.fc(h)
        return y

    def set_freeze(self, is_freeze=False):
        assert type(is_freeze) == bool, 'Wrong type of variable,requires bool'
        self.is_freeze = is_freeze
        if self.is_freeze:
            for parameter in self.base.parameters():
                parameter.requires_grad = self.is_freeze
        else:
            pass


class Mobilenet(nn.Module):
    def __init__(self, num_class=2):
        super(Mobilenet, self).__init__()
        self.is_freeze = False
        self.num_class = num_class
        self.base = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        # self.base.classifier = nn.Identity()
        self.fc = nn.Linear(1000, num_class)

    def forward(self, x):
        h = self.base(x)
        y = self.fc(h)
        return y

    def set_freeze(self, is_freeze=False):
        assert type(is_freeze) == bool, 'Wrong type of variable,requires bool'
        self.is_freeze = is_freeze
        if self.is_freeze:
            for parameter in self.base.parameters():
                parameter.requires_grad = self.is_freeze
        else:
            pass


class TridentNetwork(nn.Module):
    def __init__(self, num_class=2):
        super(TridentNetwork, self).__init__()
        self.is_freeze = False
        self.num_class = num_class
        self.base1 = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.base2 = torchvision.models.swin_t(weights='IMAGENET1K_V1')
        self.base3 = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1')

        # self.base1.classifier[-1] = nn.Linear(1024, hidden_length)
        # self.base2.head = nn.Linear(768, hidden_length)
        # self.base3.classifier[-1] = nn.Linear(1280, hidden_length)

        self.base1.classifier[-1] = nn.Identity()
        self.base2.head = nn.Identity()
        self.base3.classifier[-1] = nn.Identity()
        self.fc1 = nn.Linear(1024 + 768 + 1280, 512)
        self.fc2 = nn.Linear(512,self.num_class)

    def forward(self, x):
        h1 = self.base1(x)
        h2 = self.base2(x)
        h3 = self.base3(x)
        h = torch.cat((h1, h2, h3), -1)
        y = self.fc1(h)
        y = nn.functional.leaky_relu(y)
        y = self.fc2(y)

        return y

    def set_freeze(self, is_freeze=False):
        assert type(is_freeze) == bool, 'Wrong type of variable,requires bool'
        self.is_freeze = is_freeze
        if self.is_freeze:
            for parameter in self.base1.parameters():
                parameter.requires_grad = False
            for parameter in self.base2.parameters():
                parameter.requires_grad = False
            for parameter in self.base3.parameters():
                parameter.requires_grad = False
        else:
            pass


if __name__ == "__main__":
    x = torch.rand(10, 3, 224, 224)
    # model = get_model('Efficientnet')
    # y = model(x)
