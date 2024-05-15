# Reference: Wonjun Lee
from utils.tools import step4OTFlow_RK1, step4OTFlow_RK4
from wassersteinGF.formula import odefun, calculateU
import torch


def internal_energy(x_next, rho_next, args):
    u = calculateU(x_next, rho_next, args)
    internalEnergy = 2 * torch.mean(u/rho_next) * args.outerTimeStep
    return internalEnergy

