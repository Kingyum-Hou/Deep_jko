import torch
import os
import numpy as np
import random
from functools import reduce
import operator


def seed_everything(seed) -> int:
    if not isinstance(seed, int):
        seed = int(seed)

    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return seed


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    total_bytes = total_params * 4  # float32
    total_megabytes = total_bytes / (1024**2)
    print(f"Total Trainable Params: {total_params}")
    print(f"Which is approximately: {total_megabytes:.3f}Mb")
    return total_params, total_megabytes


def step4OTFlow_RK1(odefun, z, Phi, alphas, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alphas: list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    return z + (t1 - t0) * odefun(z, t0, Phi, alphas=alphas)


def step4OTFlow_RK4(odefun, z, Phi, alphas, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alphas: list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time

    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    k = h * odefun(z0, t0, Phi, alphas=alphas)
    z = z0 + (1.0/6.0) * k

    k = h * odefun( z0 + 0.5*k , t0+(h/2) , Phi, alphas=alphas)
    z += (2.0/6.0) * k

    k = h * odefun( z0 + 0.5*k , t0+(h/2) , Phi, alphas=alphas)
    z += (2.0/6.0) * k

    k = h * odefun( z0 + k , t0+h , Phi, alphas=alphas)
    z += (1.0/6.0) * k

    return z
