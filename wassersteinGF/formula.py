import torch


def multiple_exp(Tx):
    pos1= Tensor([[2.0,-2.0]])
    pos2= Tensor([[-2.0,2.0]])
    pos3= Tensor([[2.0,2.0]])
    pos4= Tensor([[-2.0,-2.0]])
    return 1/(2*np.pi) * ( torch.exp(-torch.sum((Tx-pos1)**2, dim=1)/(2 * 0.5**2)) 
                         + torch.exp(-torch.sum((Tx-pos2)**2, dim=1)/(2 * 0.5**2))  
                         + torch.exp(-torch.sum((Tx-pos3)**2, dim=1)/(2 * 0.5**2))
                         + torch.exp(-torch.sum((Tx-pos4)**2, dim=1)/(2 * 0.5**2)) )


def compute_U(rho_next, Tx=None):
    if Tx != None:
        return rho_next * ( torch.log(rho_next) - torch.log(multiple_exp(Tx) + 1e-5) - 1 )
    else:
        return rho_next * ( torch.log(rho_next) - 1 )


def OTFlowProblemGradientFlowsPorous(x, rho, Phi, tspan , nt, tau, n_tau, net_list, stepper="rk4", alph =[1.0,1.0,1.0], z=None):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.
    
    :param x:       input data tensor nex-by-d
    :param rho:     input rho nex-by-1
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param n_tau: nth step in gradient flows
    :param net_list: list of nets of length n_tau
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    d = z.shape[1]-3 # dimension for x

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    rho_next = rho
    tk = tspan[0]

    # given the data from the list of nets, compute z for this iteration
    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # interaction cost
    n  = z.shape[0]
    Tx = z[:,0:d]

    rho_next = rho_next / torch.exp(z[:,d]) + 1e-5
    terminal_cost = (compute_U(rho_next,Tx) / (rho_next)).mean()
    costL  = torch.mean(z[:,-2]) * 0.5
    costC  = terminal_cost * tau
    costR = 0

    cs = [costL, costC, costR]
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs
