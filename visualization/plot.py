import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_scatter(x, args, save_path):
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(x[:,0], x[:,1], marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(save_path)
    plt.close('all')

def plot_scatter_color(x, y, args, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x[:, 0], x[:, 1], c=y, marker='.')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(save_path)
    plt.close('all')

