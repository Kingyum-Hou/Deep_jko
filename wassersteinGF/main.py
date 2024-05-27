import os
from utils.params import cli

# load param, set gpu
args = cli()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from tensorboardX import SummaryWriter
from models import get_model
from utils.tools import seed_everything, count_parameters, step4OTFlow_RK1, step4OTFlow_RK4, Tensor
from data.toy_data import inf_train_gen
import numpy as np
from visualization.plot import plot_scatter, plot_scatter_color
from wassersteinGF.jko import oneOuterIteration


seed = seed_everything(args.seed)
#torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.alphas = [float(item) for item in args.alphas.split(',')]
writer = SummaryWriter()


# build model, list contains K * phi network
net_list=[]
for i in range(args.num_outerIters):
    net = get_model(args.model, args).to(device)
    net_list.append(net)

x, y = inf_train_gen(args.data, batch_size=args.batch_size)
plot_scatter_color(x, y, args, os.path.join(args.save_path, "0.png"))


# training
for outerItr in range(0, args.num_outerIters):
    # load model
    for i in range(outerItr):
        model_save_path = os.path.join(args.save_path, f"{i}.pt")
        net_list[i].load_state_dict(torch.load(model_save_path, map_location=device))
        if i == outerItr-1:
            net_list[outerItr].load_state_dict(torch.load(os.path.join(args.save_path, f"{outerItr-1}.pt"), map_location=device))
    
    oneOuterIteration(args, net_list[:outerItr+1], device)
