# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class high_layers(nn.Module):

    def __init__(self,input_size,output_size):
        super(high_layers,self).__init__()
        self.alpha = torch.nn.Parameter(torch.rand((1,)), requires_grad=True)
        self.beta = 0.9
        self.gamma = -0.1
        self.eplison = 2

        w = torch.empty(input_size, output_size).cuda()
        self.u = torch.nn.Parameter(torch.nn.init.uniform_(w, 0, 1),
                                    requires_grad=True)

        w = torch.empty(input_size, output_size).cuda()
        self.w_params = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                           requires_grad=True)
    def forward(self,x):
        self.s = torch.sigmoid(torch.log(self.u) - torch.log(1 - self.u) + torch.log(self.alpha) / self.beta)
        print(self.s,"ssss")
        self.s_ = self.s * (self.eplison - self.gamma) + self.gamma

        self.z_params = (self.s_ > 0).float() * self.s_
        self.z_params = (self.z_params > 1).float() + (self.z_params <= 1).float() * self.z_params

        output = self.z_params * self.w_params
        output = torch.einsum('anc,cd -> and', x, output)
        print(output,"output")
        return output