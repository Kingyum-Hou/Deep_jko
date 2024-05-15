import torch
import torch.nn.functional as F


def referenceDensityQ(x_next, args):
    device = x_next.device
    # four Gaussian distributions
    centers = torch.tensor([[2., 2.], [2., -2.], [-2., 2.], [-2., -2.]]).to(device)
    device = x_next.device
    B = x_next.shape[0]
    pdf = torch.zeros(B, 1).to(device)
    for itr in range(4):
        scalingFactor = 1/(2*torch.pi*args.sigma**2)
        squaredDistance = torch.sum((x_next-centers[itr, :].unsqueeze(dim=0))**2, dim=1, keepdim=True)
        pdf +=  scalingFactor * torch.exp(-squaredDistance / (2*args.sigma**2))
    pdf /= 4
    return pdf


def calculateU(x_next, rho_next, args):
    logRho = torch.log(rho_next)
    logQ = torch.log(referenceDensityQ(x_next, args)+1e-5)
    return rho_next * (logRho - logQ)


def odefun(init_state, init_time, phi_net, args):
    """
    x - particle position
    l - log determinant
    wasserstein2Distance - accumulated transport costs (Lagrangian)
    """
    B, D = init_state.shape
    x = init_state[:, :args.space_dim]
    xt = F.pad(x, (0, 1, 0, 0), value=init_time)

    grad, trHessian = phi_net.gradAndHessian(xt)
    dx_dtau = -grad[:, :args.space_dim]
    dl_dtau = -trHessian
    wasserstein2Distance_dtau = torch.sum(torch.pow(dx_dtau, 2), dim=1, keepdim=True)
    return torch.cat([dx_dtau, dl_dtau, wasserstein2Distance_dtau], dim=1)
