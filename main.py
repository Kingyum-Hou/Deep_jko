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
from jko import oneOuterIteration


seed = seed_everything(args.seed)
#torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.alphas = [float(item) for item in args.alphas.split(',')]
writer = SummaryWriter()


# load data: lacation, rho
x, y = inf_train_gen(args.data, batch_size=args.num_totalData, dim=args.space_dim)
x = torch.tensor(x)
y = torch.tensor(y)
# plot_scatter_color(x, y, savePath=os.path.join(args.save_path, 'figs', f"data.png"))
print('Data is ready')


# load model, list contains K * phi network
net_list=[]
for i in range(args.num_outerIters):
    net = get_model(args.model, args).to(device)
    net_list.append(net)

# training
for outerItr in range(args.num_outerIters):
    pred_x, pred_y = oneOuterIteration(args, net_list[:outerItr+1], device)
