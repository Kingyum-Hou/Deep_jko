# Reference: Wonjun Lee

import os
import torch
from utils.tools import Tensor
from data.toy_data import inf_train_gen
from timeit import default_timer
from utils.tools import step4OTFlow_RK1
from wassersteinGF.formula import odefun
from wassersteinGF.loss import internal_energy


def sampling_current_outerIteration(args, sub_net_list, device):
    B, D = args.batch_size, args.space_dim+2
    num_current_outerIters = len(sub_net_list) - 1
    x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=B)
    x_init = Tensor(x_train_batch) 
    rho_init = Tensor(y_train_batch)
    z_init = torch.zeros(B, D).to(device)  # hidden vector for ODE:[x, l, v, r]
    z_init[:, :args.space_dim] = x_init
    # TO DO: 加上迭代到当前N OuterIteration的代码
    TAPAN_START = 0.
    TSPAN = 1.
    integrate_timeStep = TSPAN / args.num_innerSteps
    start_time = TAPAN_START
    end_time = start_time + integrate_timeStep

    x_next, rho_next, z_next = x_init, rho_init, z_init
    z_next.requires_grad_(True)
    for itr in range(num_current_outerIters):
        for _ in range(args.num_innerSteps):
            if args.integrate_method=='rk1':
                z_next = step4OTFlow_RK1(odefun, start_time, end_time, z_next, sub_net_list[itr], args)
                start_time += integrate_timeStep
                end_time += integrate_timeStep
            elif args.integrate_method=='rk4':
                raise NotImplementedError
            else:
                raise NotImplementedError
        x_next = z_next[:, 0:args.space_dim].clone().detach()
        l_next = z_next[:, -2].unsqueeze(dim=1).clone().detach()
        rho_next = rho_next / (torch.exp(l_next)+1e-5).clone().detach()

    return x_next, rho_next, z_next


def oneOuterIteration(args, sub_net_list, device):
    itr = 0
    # first sample
    x_current, rho_current, z_current = sampling_current_outerIteration(args, sub_net_list, device)
    
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
    while True:
        start_time = default_timer()

        # Re sample z0, Calculate for the current outerIteration, 
        if itr%args.reSampleFreq == 0 and itr > 0:
            x_current, rho_current, z_current = sampling_current_outerIteration(args, sub_net_list, device)
            if args.scheduler == 'StepLR':
                scheduler.step()

        # model pred(like autoregress)
        TAPAN_START = 0.
        TSPAN = 1.
        integrate_timeStep = TSPAN / args.num_innerSteps
        start_time = TAPAN_START
        end_time = start_time + integrate_timeStep
        
        # renew
        x_current, rho_current, z_current = x_current.clone().detach(), rho_current.clone().detach(), z_current.clone().detach()
        z_next = z_current
        z_next.requires_grad_(True)
        for _ in range(args.num_innerSteps):
            if args.integrate_method=='rk1':
                z_next = step4OTFlow_RK1(odefun, start_time, end_time, z_next, net, args)
                start_time += integrate_timeStep
                end_time += integrate_timeStep
            elif args.integrate_method=='rk4':
                raise NotImplementedError
            else:
                raise NotImplementedError

        # interaction cost
        x_next = z_next[:, 0:args.space_dim]
        l_next = z_next[:, -2].unsqueeze(dim=1)
        rho_next = rho_current / (torch.exp(l_next)+1e-5)
        
        wasserstein2Distance = torch.mean(z_next[:, -1])
        energy = internal_energy(x_next, rho_next, args)
        loss = wasserstein2Distance
        # loss = wasserstein2Distance * args.alphas[0] + energy * args.alphas[1]

        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler == 'OneCycleLR':
            scheduler.step()

        # stopping condition
        if itr >= args.num_iterations:
            # TO DO: save model
            torch.save(net.state_dict(), os.path.join(args.save_path, f"{len(sub_net_list)-1}.pt"))
            print(f'{len(sub_net_list)}-th Running finished...')
            break
        else:
            itr += 1
        
        # TO DO: create plots
        if itr%500 == 0:
            print(
                f"itr: {itr:>3} |"
                f"loss: {loss:.3f}"
            )
