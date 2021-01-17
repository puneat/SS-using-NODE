# Helpers adapted from https://pytorch.org/tutorials/beginner/nn_tutorial.html

import pandas as pd
from pandas import DataFrame
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from ModelBlocks import ConcatConv1d, ODENet, ODEfunc, ResBlock, count_parameters, norm, Flatten

def get_model(is_odenet=True, dim=1, adam=False, **kwargs):
    """
    Initialize ResNet or ODENet with optimizer.
    """
    downsampling_layers = [
#         nn.Conv1d(1, dim, 1, 1) 
#         norm(dim),
#         nn.ReLU(inplace=True),
#         nn.Conv1d(dim, dim, 4, 2, 1), 
#         norm(dim),
#         nn.ReLU(inplace=True),
#         nn.Conv1d(dim, dim, 4, 2, 1),
#         norm(dim),
#         nn.ReLU(inplace=True),
#         nn.Conv1d(dim, dim, 4, 2, 1)
    ]

    feature_layers = [ODENet(ODEfunc(dim), **kwargs)] if is_odenet else [ResBlock(dim) for _ in range(6)]
#     norm(dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), 

    fc_layers = [norm(dim), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1), Flatten(), nn.Linear(dim,2)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)

    opt = optim.Adam(model.parameters(), amsgrad=True) if adam else optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    return model, opt


def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    Calculate loss and update weights if training.
    """
    loss = loss_func(model(xb.float()), yb.long())

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fitModel(epochs, model, loss_func, opt, train_dl, valid_dl):
    """
    Train neural network model.
    """
    num_batches = len(train_dl)
    best_loss=1000
    
    for epoch in range(epochs):
        print(f"Training... epoch {epoch + 1}")
        
        model.train()   # Set model to training mode
        batch_count = 0
        #start = time.time()
        for rand,(xb, yb) in tqdm.tqdm(enumerate(train_dl), total = len(train_dl), leave=False):
            xb = xb.cuda()
            yb = yb.cuda()
            batch_count += 1
            # curr_time = time.time()
            # percent = round(batch_count/len(train_dl) * 100, 1)
            # elapsed = round((curr_time - start)/60, 1)
            # print(f"    Percent trained: {percent}%  Time elapsed: {elapsed} min", end='\r')
            loss_batch(model, loss_func, xb, yb, opt)
            
        model.eval()    # Set model to validation mode
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if val_loss < best_loss:
            best_loss = val_loss
            path = '/gdrive/My Drive/Spectrum_Sensing/SS_NODE_PyTorch_'+str(round(best_loss,4))+'.pt'
            torch.save(model, path)
            print('\n    best model saved')
            
        #writer.add_scalar("Loss/train", val_loss, epoch)
        #writer.flush()

        print(f"\n    val loss: {round(val_loss, 4)}\n")
        print(f"\n    best loss: {round(best_loss, 4)}\n")
