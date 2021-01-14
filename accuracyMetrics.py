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
from ModelBlocks import ConcatConv1d, ODENet, ODEfunc, count_parameters, norm, Flatten

def checkAccuracy (model, test_dataloader):

	model.eval()
	predicted=[]
	actuals=[]
	for i, (inputs, targets) in tqdm.tqdm(
		enumerate(test_dataloader), total = len(test_dataloader), leave=False):
		with torch.no_grad():
			logits = model(inputs.float())
			logits = logits.cpu()
			inputs=inputs.cpu()
			targets=targets.cpu()
		preds = torch.argmax(F.softmax(logits, dim=1), axis=1).numpy()
		targets = targets.numpy()
		actuals = np.concatenate((actuals,targets))
		predicted = np.concatenate((predicted,preds))

	acc = (predicted == actuals).mean()
	print(f"\n ODENet accuracy: {acc}")
	print(f"\n  Number of tunable parameters: {count_parameters(model)}")


