# Reference: Wonjun Lee

import torch
from utils.tools import Tensor
from data.toy_data import inf_train_gen
from timeit import default_timer
from utils.tools import step4OTFlow_RK1
from wassersteinGF.formula import odefun
from wassersteinGF.loss import internal_energy


def oneOuterIteration(args, sub_net_list, device):
    itr = 0
    # first sample
    x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=args.batch_size)
    x_init = Tensor(x_train_batch) 
    rho_init = Tensor(y_train_batch)
    z_init = torch.zeros(args.batch_size, args.space_dim+2).to(device)  # hidden vector for ODE:[x, l, v, r]
    z_init[:, :args.space_dim] = x_init
    # TO DO: 加上迭代到当前N OuterIteration的代码
    x_current = x_init
    rho_current = rho_init.unsqueeze(dim=1)
    z_current = z_init
    
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
    B, D = args.batch_size, args.space_dim+2
    while True:
        start_time = default_timer()

        # Re sample z0, Calculate for the current outerIteration, 
        if itr%args.reSampleFreq == 0 and itr > 0:
            x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=args.batch_size)
            x_train_batch, y_train_batch = Tensor(x_train_batch), Tensor(y_train_batch)
            x_outerIter, y_outerIter = generateData_nth_outerIter(x_train_batch, y_train_batch, sub_net_list, args)
            if args.scheduler == 'StepLR':
                scheduler.step()

        # model pred(like autoregress)
        TAPAN_START = 0.
        TSPAN = 1.
        integrate_timeStep = TSPAN / args.num_innerSteps
        start_time = TAPAN_START
        end_time = start_time + integrate_timeStep
        
        z_current = z_current.clone().detach()  # renew z
        z_current.requires_grad_(True)
        for _ in range(args.num_innerSteps):
            if args.integrate_method=='rk1':
                z_next = step4OTFlow_RK1(odefun, start_time, end_time, z_current, net, args)
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
        loss = wasserstein2Distance * args.alphas[0] + energy * args.alphas[1]

        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler == 'OneCycleLR':
            scheduler.step()
        # stopping condition
        if itr >= args.num_iterations:
            # TO DO: save model
            print(f'{outerIters_index}-th Running finished...')
            return x, y
        else:
            itr += 1
        
        # TO DO: create plots
    
    return x, y