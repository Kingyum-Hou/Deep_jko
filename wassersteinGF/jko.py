# Reference: Wonjun Lee

import os
import torch
from utils.tools import Tensor
from data.toy_data import inf_train_gen
from timeit import default_timer
from utils.tools import step4OTFlow_RK1
from wassersteinGF.formula import odefun, internal_energy
from visualization.plot import plot_scatter, plot_scatter_color
import torch.nn.functional as F


def sampling_current_outerIteration(args, sub_net_list, device):
    B, D = args.batch_size, args.space_dim+2
    num_current_outerIters = len(sub_net_list) - 1
    x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=B)
    x_init = Tensor(x_train_batch) 
    rho_init = Tensor(y_train_batch)
    z_init = torch.zeros(B, D).to(device)  # hidden vector for ODE:[x, l, v, r]
    z_init[:, :args.space_dim] = x_init
    TAPAN_START = 0.
    TSPAN = 1.
    integrate_timeStep = TSPAN / args.num_innerSteps

    x_next, rho_next, z_next = x_init, rho_init.unsqueeze(dim=1), z_init
    z_next.requires_grad_(True)
    for itr in range(num_current_outerIters):
        start_time = TAPAN_START
        end_time = start_time + integrate_timeStep
        for _ in range(args.num_innerSteps):
            if args.integrate_method=='rk1':
                z_next = step4OTFlow_RK1(odefun, start_time, end_time, z_next, sub_net_list[itr], args, device)
                start_time += integrate_timeStep
                end_time += integrate_timeStep
            elif args.integrate_method=='rk4':
                raise NotImplementedError
            else:
                raise NotImplementedError
        # renew all
        x_next = z_next[:, 0:args.space_dim]
        l_next = z_next[:, -2].unsqueeze(dim=1)
        rho_next = rho_next / (torch.exp(l_next)+1e-6)
        z_next[:, args.space_dim:]=0.
        
    return x_next, rho_next


def oneOuterIteration(args, sub_net_list, device):
    itr = 0
    # first sample
    x_current, rho_current = sampling_current_outerIteration(args, sub_net_list, device)
    x_current, rho_current = x_current.to(device), rho_current.to(device)
    
    # model
    # The first n layers have been frozen.
    net = sub_net_list[-1]
    if args.optimizer == 'Adam':
        optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
    else:
        raise NotImplementedError

    # train
    net.train()
    start = default_timer()
    best_loss = float('inf')
    best_params = None
    while True:
        # Re sample z0, Calculate for the current outerIteration, 
        if itr%args.reSampleFreq == 0 and itr > 0:
            x_current, rho_current = sampling_current_outerIteration(args, sub_net_list, device)
            x_current, rho_current = x_current.to(device), rho_current.to(device)
            if args.scheduler == 'StepLR':
                scheduler.step()


        # model pred(like autoregress)
        TAPAN_START = 0.
        TSPAN = 1.
        integrate_timeStep = TSPAN / args.num_innerSteps
        start_time = TAPAN_START
        end_time = start_time + integrate_timeStep
        # renew
        z_next = torch.zeros(args.batch_size, args.space_dim+2).to(device)
        z_next[:, :args.space_dim] = x_current.clone().detach()
        x_next = x_current.clone().detach()
        #xt = F.pad(x_next, (0, 1, 0, 0), value=0.)
        #xt.requires_grad_(True)
        #z_next[:, args.space_dim:args.space_dim+1] = torch.log(torch.abs(net.detHessian(xt).unsqueeze(dim=1)))
        #z_next.requires_grad_(True)
        rho_next = rho_current.clone().detach()

        for _ in range(args.num_innerSteps):
            if args.integrate_method=='rk1':
                z_next = step4OTFlow_RK1(odefun, start_time, end_time, z_next, net, args, device)
                start_time += integrate_timeStep
                end_time += integrate_timeStep
            elif args.integrate_method=='rk4':
                raise NotImplementedError
            else:
                raise NotImplementedError


        # interaction cost
        x_next = z_next[:, 0:args.space_dim]
        l_next = z_next[:, -2].unsqueeze(dim=1)
        rho_next = rho_next / torch.exp(l_next)
        
        wasserstein2Distance = torch.mean(z_next[:, -1])
        energy = internal_energy(x_next, rho_next, args)
        loss = wasserstein2Distance * args.alphas[0] + energy * args.alphas[1]
        if loss < best_loss:
            best_loss = loss
            best_params = net.state_dict()


        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler == 'OneCycleLR':
            scheduler.step()
        

        # stopping condition
        if itr >= args.num_iterations:
            # save model
            torch.save(best_params, os.path.join(args.save_path, f"{len(sub_net_list)-1}.pt"))
            plot_scatter_color(x_next.cpu().detach().numpy(), rho_next.cpu().detach().numpy(), args, os.path.join(args.save_path, f"{len(sub_net_list)}.png"))
            print(f'{len(sub_net_list)}-th Running finished... best loss is {best_loss:.3f}')
            break
        else:
            itr += 1
        

        # report
        if itr%500 == 0:
            end = default_timer()
            print(
                f"cost time: {end-start:.3f}s |"
                f"itr: {itr:>3} |"
                f"loss: {loss:.3f}"
            )
            start = default_timer()
