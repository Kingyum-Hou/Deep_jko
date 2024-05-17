# ResNet in DeepJKO
# Neural network to model the potential function
# Reference: Wonjun Lee

import torch
import torch.nn as nn
from models.base import Mlp


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.resnet = Mlp(
            args.space_dim+args.extraInput_dim,
            args.hidden_size,
            1,
            num_layers=args.num_layers,
            activation=args.activation,
            isResNet=True,
            isBatchNorms=False,
        )

    def forward(self, x):
        y = self.resnet(x)
        gradient = torch.autograd.grad(y, x, create_graph=True)[0]
        return gradient

    def gradAndHessian(self, x):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param isOnlyGrad: boolean, if True only return gradient, if False return (grad, trace-Hessisan)

        :return: gradient , trace(hessian)    OR    gradient
        """
        device = x.device
        y = self.resnet(x)
        gradient = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        
        # Hessian
        B, D = gradient.shape
        def iterate_columns():
            for i in range(D):
                yield gradient[:, i:i+1]

        tr_hessian = torch.zeros(B, 1).to(device)
        for itr, g in enumerate(iterate_columns()):
            second_gradient = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), retain_graph=True)[0]
            tr_hessian += second_gradient[:, itr].unsqueeze(dim=1)

        return gradient, tr_hessian
