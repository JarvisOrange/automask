import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from libcity.model import loss
sigmoid = torch.nn.Sigmoid()
class SpatialMaskModel(nn.Module):
    def __init__(self):
        super(SpatialMaskModel, self).__init__()
        self.mask = nn.Parameter(torch.zeros(207,207,device='cuda',dtype=torch.float), requires_grad=True)

    def forward(self, batch):
        x = torch.tensor(batch['X'], device='cuda',dtype=torch.float)
        batch['X'] = torch.mul(x, self.mask)
        return batch


class TemporalMaskModel(nn.Module):
    def __init__(self):
        super(TemporalMaskModel, self).__init__()
        c_in= 12
        c_out = 12
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)


    def forward(self, batch):
        x = torch.tensor(batch['X'], device='cuda', dtype=torch.float)
        batch['X'] = torch.mul(x, self.get_mask(x))

        return batch

    def get_mask(self,input):
        mask = self.mlp(input)

        return mask
        # # self.mask = nn.Parameter(torch.ones(self.input_size, device='cuda', dtype=torch.float), requires_grad=True)
        # self.mask = nn.Parameter(torch.ones(self.input_size, device='cuda', dtype=torch.float), requires_grad=True)
        # # self.mask[:8,:,0] = 0
    # def forward(self, batch):
    #     x = torch.tensor(batch['X'], device='cuda' ,dtype=torch.float)
    #     batch['X'] = torch.mul(x, self.mask)
    #     return batch


class AutoMask(nn.Module):
    def __init__(self, model, method = 0):
        super(AutoMask, self).__init__()
        self.model = model
        self.method = method

        if method == 0:
            self.maskModel = TemporalMaskModel()
        else:
            self.maskModel = SpatialMaskModel()


    def forward(self, batch):

        output = self.maskModel(batch)

        outputs = self.model(output)
        return outputs





