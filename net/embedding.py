'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
from __future__ import print_function, division, absolute_import
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)


    def forward(self, input_tensor):
        x = self.embedding(input_tensor)

        return x


def embedding(input_dim=1024, output_dim=64):
    module = Embedding(input_dim=input_dim, output_dim=output_dim)

    return module
