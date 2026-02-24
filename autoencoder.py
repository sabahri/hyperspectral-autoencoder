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

#print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands:{num_bands}')

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
data_reshaped = data.reshape(num_pixels, num_bands).T

###########################################
########## Activation Functions ###########
##########     And Gradients    ###########
###########################################

# ReLU
def relu(x):
	return(np.maximum(x,0))

def relu_grad(x):
	return(x > 0)

# Sigmoid
def sigmoid(x):
	f = 1 / (1 + np.exp(-x))
	return(f)

def sigmoid_grad(x):
	return(sigmoid(x) * (1 - sigmoid(x)))

# Tanh
def tanh(x):
	f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	return(f)

def tanh_grad(x):
	return((1/np.cosh(x))**2)

############################################
############# Loss Functions ###############
############################################

# Mean Squared Error
def mse_cost(x,y,n):

	return(np.sum((x - y)**2, axis=0) / n)

def mse_der(x,y,n):
	return((x - y) * 2 / n)

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
	stdev = 2 / np.sqrt(n)
	w = np.random.normal(0,stdev,size=(m,n))
	return(w)

# Encoder
# Initialization
w0 = weight_init_He(num_bands,64)
w1 = weight_init_He(64,16)
w2 = weight_init_He(16,8)

# Assume initiating bias is 0
layer1 = relu(w0 @ data_reshaped)
layer2 = relu(w1 @ layer1)
# Finally, our Bottleneck layer
layer3 = relu(w2 @ layer2)

# Decoder
w3 = weight_init_He(8,16)
w4 = weight_init_He(16,64)
w5 = weight_init_He(64,num_bands)

layer4 = relu(w3 @ layer3)
layer5 = relu(w4 @ layer4)
# Reconstructed layer, no activation (linear output)
layer6 = w5 @ layer5

w_list = [w0, w1, w2, w3, w4, w5]
layer_list = [data_reshaped, layer1, layer2, layer3, layer4, layer5, layer6]

# Mean squared error to start with
cost = mse_cost(data_reshaped, layer6, num_bands)
cost_derivative = mse_der(data_reshaped,layer6, num_bands)

# Backpropagation
# Conceptually
#cost_grad = cost_derivative * w5 * relu_grad(layer5) * w4 * relu_grad(layer4) * w3 * relu_grad(layer3) * w2 * relu_grad(layer2) * w1 * relu_grad(layer1) * w0
cost_grad = (cost_derivative.T @ w5).T

#cost_grad = cost_grad @ w5 @ relu_grad(layer5)
# Note to self:
# keep track of what relu_grad argument is supposed to be for correct order of operations
# for tomorrow troubleshooting
for i in range(4,-1,-1):
	print(i)
	print((relu_grad(layer_list[i]).T @ w_list[i]).shape)
	#print(w_list[i].T.shape)
	#cost_grad = relu_grad(layer_list[i]).T @ w_list[i].T @ cost_grad

#print(cost_grad.shape)
