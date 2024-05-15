import torch
from utils.tools import step4OTFlow_RK1


def generateData_nth_outerIter(x_train_batch, y_train_batch, net_list, args):
    B = x_train_batch.shape[0]
    z = x_train_batch.clone()
    rho = y_train_batch.clone()
    innerSteps_dt = 1. / args.num_innerSteps
    
    with torch.no_grad():
        for outerIters_index in range(args.num_outerIters):
            l = torch.zeros(B, 1).to(device)         
            for innerSteps_index in range(args.num_innerSteps):
                innerSteps_time = innerSteps_index*innerSteps_dt
                z = None
            rho = rho / (torch.exp(l) + 1e-5)
    x_nth = z
    rho_nth = rho
    return x_nth, rho_nth
