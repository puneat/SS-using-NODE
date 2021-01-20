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

def checkAccuracy (model, test_dataloader, return_predictions=False, return_prob = None):
	assert return_prob in [None,1,0], 'Choose from None, 1 or 0

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
		if return_prob==None:
			preds = torch.argmax(F.softmax(logits, dim=1), axis=1).numpy()
		elif return_prob != None:
			preds = F.softmax(logits, dim=1).numpy()
			preds = preds[:,int(return_prob)]
		targets = targets.numpy()
		actuals = np.concatenate((actuals,targets))
		predicted = np.concatenate((predicted,preds))
	if return_prob==None:
		acc = (predicted == actuals).mean()
		print(f"\n ODENet accuracy: {acc*100}")
	print(f"\n  Number of tunable parameters: {count_parameters(model)}")

	if return_predictions:
		return predicted, actuals


