import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_


class AntiderivTanh(nn.Module):
    def __init__(self):
        super(AntiderivTanh, self).__init__()

    def forward(self, x):
        return torch.abs(x) + torch.log(1 + torch.exp(-2.0 * torch.abs(x)))

ACTIVATION = {'gelu':nn.GELU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus,'ELU':nn.ELU,'silu':nn.SiLU, 'antiderivTanh':AntiderivTanh}


class Mlp(nn.Module):
    def __init__(
        self, 
        num_input, 
        num_hidden, 
        num_output, 
        num_layers=1, 
        activation='gelu', 
        isResNet=True,
        isBatchNorms=True,
    ):
        super(Mlp, self).__init__()

        if activation in ACTIVATION.keys():
            self.activation = ACTIVATION[activation]
        else:
            raise NotImplementedError
        self.num_layers = num_layers
        self.isResNet = isResNet
        self.isBatchNorms = isBatchNorms
        self.linear_pre = nn.Sequential(nn.Linear(num_input, num_hidden), self.activation())
        self.linear_out = nn.Linear(num_hidden, num_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(num_hidden, num_hidden), self.activation()) for _ in range(num_layers)])
        self.batchNorms = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.num_layers):
            if self.isBatchNorms:
                x_ = self.linears[i](x)
                x_ = self.batchNorms[i](x)
                x = (self.activation(x_) + x) if self.isResNet else self.activation(x_)
            else:
                x = self.linears[i](x) + x if self.isResNet else self.linear[i](x)
        x = self.linear_out(x)
        return x
