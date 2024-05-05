import os
from utils.params import cli

# load param, set gpu
args = cli()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from tensorboardX import SummaryWriter
from timeit import default_timer
from models import get_model
from utils.tools import seed_everything, count_parameters, step4OTFlow_RK1, step4OTFlow_RK4
from data.toy_data import inf_train_gen
import numpy as np
from visualization.plot import plot_scatter, plot_scatter_color
from wassersteinGF.loss import loss_OTFlowProblemGradientFlowsPorous


seed = seed_everything(args.seed)
#torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# base config
pass


# load data
pass


# load model
pass


# train
pass


# test
pass