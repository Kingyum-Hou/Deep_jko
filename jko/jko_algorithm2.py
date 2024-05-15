import torch
from utils.tools import Tensor
from jko.jko_algorithm1 import generateData_nth_outerIter


def deepJKO(net_list, args, optimizer, scheduler, device, writer):
    # initial distribution rho0
    x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=args.batch_size)
    x_train_batch, y_train_batch = Tensor(x_train_batch), Tensor(y_train_batch)
    
    for outerIters_index in range(args.num_outerIters):
        pred_x, pred_y = oneOuterIter(OuterIters_index, net_list[:i+1], args, x_train_batch, y_train_batch)
        x_train_batch = pred_x
        y_train_batch = pred_y
    return 


def oneOuterIter(outerIters_index, net_list, args, x_train_batch, y_train_batch):
    iterations_index = 0
    while True:
        start_time = default_timer()

        # train
        for net in net_list:
            net.train()
        # Re sample x0
        if iterations_index%args.reSampleFreq == 0 and iterations_index > 0:
            x_train_batch, y_train_batch = inf_train_gen(args.data, batch_size=args.batch_size)
            x_train_batch, y_train_batch = Tensor(x_train_batch), Tensor(y_train_batch)
            x_outerIter, y_outerIter = generateData_nth_outerIter(x_train_batch, y_train_batch, net_list, args)
        # optim
        optim.zero_grad()

        pass

        # exit condition
        if iter_index >= args.num_iterations:
            print(f'{outerIters_index}th Running finished...')
            exit()
        else:
            iter_index += 1
    
    return x, y
