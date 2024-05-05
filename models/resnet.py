# ResNet in DeepJKO
# Neural network to model the potential function
# Reference: Wonjun Lee

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        num_input,
        num_hidden,
        num_layer,
    ):
        pass

    def forward(self, x):
        pass

    def gradAndHessian(self, x, isOnlyGrad=False):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param isOnlyGrad: boolean, if True only return gradient, if False return (grad, trace-Hessisan)

        :return: gradient , trace(hessian)    OR    gradient
        """
        pass
