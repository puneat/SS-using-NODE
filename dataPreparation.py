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

def loadData(path, modulation, signal=True, binary_task=True, snr=True):

	column_headers = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9',
	't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19',
	't20', 't21', 't22', 't23', 't24', 't25', 't26', 't27', 't28', 't29',
	't30', 't31', 't32', 't33', 't34', 't35', 't36', 't37', 't38', 't39',
	't40', 't41', 't42', 't43', 't44', 't45', 't46', 't47', 't48', 't49',
	't50', 't51', 't52', 't53', 't54', 't55', 't56', 't57', 't58', 't59',
	't60', 't61', 't62', 't63', 't64', 't65', 't66', 't67', 't68', 't69',
	't70', 't71', 't72', 't73', 't74', 't75', 't76', 't77', 't78', 't79',
	't80', 't81', 't82', 't83', 't84', 't85', 't86', 't87', 't88', 't89',
	't90', 't91', 't92', 't93', 't94', 't95', 't96', 't97', 't98', 't99',
	't100', 't101', 't102', 't103', 't104', 't105', 't106', 't107', 't108',
	't109', 't110', 't111', 't112', 't113', 't114', 't115', 't116', 't117',
	't118', 't119', 't120', 't121', 't122', 't123', 't124', 't125', 't126',
	't127', 't128', 't129', 't130', 't131', 't132', 't133', 't134', 't135',
	't136', 't137', 't138', 't139', 't140', 't141', 't142', 't143', 't144',
	't145', 't146', 't147', 't148', 't149', 't150', 't151', 't152', 't153',
	't154', 't155', 't156', 't157', 't158', 't159', 't160', 't161', 't162',
	't163', 't164', 't165', 't166', 't167', 't168', 't169', 't170', 't171',
	't172', 't173', 't174', 't175', 't176', 't177', 't178', 't179', 't180',
	't181', 't182', 't183', 't184', 't185', 't186', 't187', 't188', 't189',
	't190', 't191', 't192', 't193', 't194', 't195', 't196', 't197', 't198',
	't199', 't200', 't201', 't202', 't203', 't204', 't205', 't206', 't207',
	't208', 't209', 't210', 't211', 't212', 't213', 't214', 't215', 't216',
	't217', 't218', 't219', 't220', 't221', 't222', 't223', 't224', 't225',
	't226', 't227', 't228', 't229', 't230', 't231', 't232', 't233', 't234',
	't235', 't236', 't237', 't238', 't239', 't240', 't241', 't242', 't243',
	't244', 't245', 't246', 't247', 't248', 't249', 't250', 't251', 't252',
	't253', 't254', 't255', 't256', 't257', 't258', 't259', 't260', 't261',
	't262', 't263', 't264', 't265', 't266', 't267', 't268', 't269', 't270',
	't271', 't272', 't273', 't274', 't275', 't276', 't277', 't278', 't279',
	't280', 't281', 't282', 't283', 't284', 't285', 't286', 't287', 't288',
	't289', 't290', 't291', 't292', 't293', 't294', 't295', 't296', 't297',
	't298', 't299', 't300', 't301', 't302', 't303', 't304', 't305', 't306',
	't307', 't308', 't309', 't310', 't311', 't312', 't313', 't314', 't315',
	't316', 't317', 't318', 't319', 't320', 't321', 't322', 't323', 't324',
	't325', 't326', 't327', 't328', 't329', 't330', 't331', 't332', 't333',
	't334', 't335', 't336', 't337', 't338', 't339', 't340', 't341', 't342',
	't343', 't344', 't345', 't346', 't347', 't348', 't349', 't350', 't351',
	't352', 't353', 't354', 't355', 't356', 't357', 't358', 't359', 't360',
	't361', 't362', 't363', 't364', 't365', 't366', 't367', 't368', 't369',
	't370', 't371', 't372', 't373', 't374', 't375', 't376', 't377', 't378',
	't379', 't380', 't381', 't382', 't383', 't384', 't385', 't386', 't387',
	't388', 't389', 't390', 't391', 't392', 't393', 't394', 't395', 't396',
	't397', 't398', 't399', 't400', 't401', 't402', 't403', 't404', 't405',
	't406', 't407', 't408', 't409', 't410', 't411', 't412', 't413', 't414',
	't415', 't416', 't417', 't418', 't419', 't420', 't421', 't422', 't423',
	't424', 't425', 't426', 't427', 't428', 't429', 't430', 't431', 't432',
	't433', 't434', 't435', 't436', 't437', 't438', 't439', 't440', 't441',
	't442', 't443', 't444', 't445', 't446', 't447', 't448', 't449', 't450',
	't451', 't452', 't453', 't454', 't455', 't456', 't457', 't458', 't459',
	't460', 't461', 't462', 't463', 't464', 't465', 't466', 't467', 't468',
	't469', 't470', 't471', 't472', 't473', 't474', 't475', 't476', 't477',
	't478', 't479', 't480', 't481', 't482', 't483', 't484', 't485', 't486',
	't487', 't488', 't489', 't490', 't491', 't492', 't493', 't494', 't495',
	't496', 't497', 't498', 't499', 't500', 't501', 't502', 't503', 't504',
	't505', 't506', 't507', 't508', 't509', 't510', 't511', 't512', 'label',
	'snr']

	data = scipy.io.loadmat(path)

	if signal:
		json_header = str('signal_final_'+modulation)
	elif not signal:
		json_header = str('noise_final_'+modulation)

	data=DataFrame.from_dict(data[json_header])

	if snr and signal:
		data.columns = column_headers
		print('\n Loaded ' + str(modulation) + ' signal' + ' data with shape: '
			+ str(data.shape))
		if binary_task:
			if signal:
				data.iloc[:,-2] = 1

		##################     TBD      ################
		elif not binary_task:
			if modulation=='qam2':
				data.iloc[:,-2] = 2
		##########################################################


	elif snr and (not signal):
		data.columns = column_headers[:513]
		print('\n Loaded ' + 'noise' + ' data with shape: ' + str(data.shape))
	elif not snr:
		data.columns = column_headers
		print('\n Loaded '+str(modulation) + 'signal ' + signal + ' data with shape: '
			+ str(data.shape))
		if binary_task:
			if signal:
				data.iloc[:,-1] = 1
##################     TBD      ################
		elif not binary_task:
			if modulation=='b':
				data.iloc[:,-1] = 2
			elif modulation=='q':
				data.iloc[:,-1] = 4
####################################################

	return data

def prepareData(data, hasSNR=True, split=0.2):
	data = pd.concat(data);
	data = data.sample(frac=1).reset_index(drop=True) #shuffling for randomization

	X = data.iloc[:, :512].values
	y = data.iloc[:, 512].values
	if hasSNR:
		snr = data.iloc[:,513].values
		y = data.iloc[:, 512:].values

	print('\n Dimensions of input features, X is: '+ str(X.shape))
	print('\n Dimensions of output labels, y is: '+ str(y.shape))
	if hasSNR:
		print('\n Unique modulations: ' + str(np.unique(y[:,0], return_counts=True)))
	elif not hasSNR:
		print('\n Unique modulations: ' + str(np.unique(y, return_counts=True)))
		

	#Using sci-kit's train_test split function
	if split!=0:
		np.random.seed(0) # setting a seed value for reproducibility of results
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42, stratify=y[:,0])
		print('\n Rows in training data : '+ str(X_train.shape[0]))
		print('\n Rows in testing data : '+ str(X_test.shape[0]))
		return X_train, y_train, X_test, y_test
	elif split==0:
		print('\n Rows in data : '+ str(X.shape[0]))
		return X, y


def convertToTensor(X_train, y_train, X_test, y_test, train_bs=500, test_bs=1000, gpu = True):

	X_train, y_train, X_test, y_test = map( torch.from_numpy,(X_train, y_train, X_test, y_test))

    #shifting to GPU
	if gpu:
		X_train = X_train.to(device='cuda')
		y_train = y_train.to(device='cuda')
		X_test = X_test.to(device='cuda')
		y_test = y_test.to(device='cuda')

	# Convert to 3D tensor
	X_train = X_train.unsqueeze(1)
	X_test = X_test.unsqueeze(1)

	print('Training Set Shape: ', X_train.shape)
	print('Testing Set Shape: ', X_test.shape)

	train_ds = TensorDataset(X_train, y_train)
	train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)

	test_ds = TensorDataset(X_test, y_test)
	test_dl = DataLoader(test_ds, batch_size=test_bs)

	return train_dl, test_dl







