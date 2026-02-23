# Implementing an autoencoder from scratch
# https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
# https://medium.com/@dipikadhara06/understanding-hyperspectral-imaging-reading-and-visualizing-data-from-mat-files-using-python-295d5cb83e0a
# https://researchdatapod.com/building-autoencoders-pytorch-tutorial/


import scipy.io
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

#raw_data = str(sys.argv[1])
#raw_ground_truth = str(sys.argv[2])
data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('SalinasA_gt.mat')['salinasA_gt']

# For SalinasA, full data shape is 83 x 86 x 204
# Spatial grid is 83 x 86 
# 204-dimensional band vector (basically 204 channels instead of 3)
num_pixels = data.shape[0] * data.shape[1]
num_bands = data.shape[-1]

print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands:{num_bands}')

# # Normalizing band values to 1
# band_ind = int(sys.argv[1])
# band = data[:,:,band_ind]
# band_min = band.min()
# band_max = band.max()

# band_normalized = (band - band_min)/(band_max - band_min)

# # Plotting
# plt.imshow(band_normalized, cmap='gray',vmin=0,vmax=1)
# plt.show()

# Reshape data
# 83 x 86 = 7138 pixels
# So we need to break it down to 7138 1x204-dim vectors
data_reshaped = data.reshape(num_pixels, num_bands)

###########################################
########## Activation Functions ###########
###########################################

# ReLU
def relu(x):
	if x > 0 :
		return(x)
	else:
		return(0)

# Sigmoid
def sigmoid(x):
	f = 1 / (1 + np.exp(-x))
	return(f)

# Tanh
def tanh(x):
	f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	return(f)

############################################
############# Defining Layers ##############
############################################

# Need a gradual reduction in dimensionality
# e.g. 204 -> 64 -> 16 -> 8
# He weight initialization: scale by 2/sqrt(# inputs to layer)
# n: input size
# m: desired output size

def weight_init_He(n,m):
	# Desired variance in weights for He initialization
	var = 2 / np.sqrt(n)
	w = np.random.normal(0,var,size=(m,n))
	return(w)


############################################
############# Neural Network ###############
############################################

w0 = weight_init_He(num_bands,64)
w1 = weight_init_He(64,16)
w2 = weight_init_He(16,8)

# Assume initiating bias is 0
layer1 = w0 @ data_reshaped.transpose()
layer2 = w1 @ layer1
layer3 = w2 @ layer2












