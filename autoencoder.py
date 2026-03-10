# Implementing an autoencoder from scratch
# https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
# https://medium.com/@dipikadhara06/understanding-hyperspectral-imaging-reading-and-visualizing-data-from-mat-files-using-python-295d5cb83e0a
# https://researchdatapod.com/building-autoencoders-pytorch-tutorial/
# https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/

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

## Normalizing band values to 1
#band_ind = int(sys.argv[1])
#band = data[:,:,band_ind]
#band_min = band.min()
#band_max = band.max()

#band_normalized = (band - band_min)/(band_max - band_min)

# # Plotting
# plt.imshow(band_normalized, cmap='gray',vmin=0,vmax=1)
# plt.show()

# Reshape data
# 83 x 86 = 7138 pixels
# So we need to break it down to 7138 1x204-dim vectors
#data_reshaped = data.reshape(num_pixels, num_bands)

###########################################
########## Activation Functions ###########
##########     And Gradients    ###########
###########################################

# ReLU
def relu(x):
	return(np.maximum(x,0))

def relu_der(x):
	return(x > 0)

# Sigmoid
def sigmoid(x):
	f = 1 / (1 + np.exp(-x))
	return(f)

# Tanh
def tanh(x):
	f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	return(f)

############################################
############# Loss Functions ###############
############################################

# Mean Squared Error
def mse(x,y,n):

	return(np.sum((x - y)**2, axis=0) / n)

def mse_der(x,y,n):
	return(-2*(x - y))

############################################
############### Weight Init ################
########## And Backpropagation #############
############################################

def weight_init_He(n,m):
	# He weight initialization: scale by 2/sqrt(# inputs to layer)
	# n: input layer size
	# m: desired output layer size

	stdev = 2 / np.sqrt(n)
	w = np.random.normal(0,stdev,size=(n,m))
	return(w)

def update_params(W, b, layer, d_J):
	# W : list of weights
	# W[i] : weight matrix applied to the ith layer
	# b : list of biases
	# layer: list of layers
	# d_J : derivative of cost function
	# layer_ind : layer we are backpropagating to

	# Recursively backpropagating with dz and dq base case
	dz = d_J												# starts at dz_6
	dq = W[-1].T @ dz										# dq_5

	W[-1] = W[-1] - learning_rate * dz @ layer[-2].T  		# note W[-1] = W[5] = w6 here
	b[-1] = b[-1] - learning_rate * dz

	for i in range(len(W) - 1, 0,-1):						# Iterating from i = 5 to 1
		dz = np.multiply(dq, relu_der(layer[i]))			# dz_5
		dq = W[i].T @ dz									# dq_4

		# Calculating gradients for weights / biases per layer
		dW = dz @ layer[i-1].T 
		db = dz

		# Updating parameters, layer by layer
		W[i] = W[i] - learning_rate * dW
		b[i] = b[i] - learning_rate * db
	
	return(W, b)

############################################
############## Initialization ##############
############################################

biases = np.zeros((num_pixels,1))

# Encoder
w1 = weight_init_He(num_bands,64)
b1 = 0
w2 = weight_init_He(64,16)
b2 = 0
w3 = weight_init_He(16,8)
b3 = 0

# Perform Per-Band Normalization to avoid exploding gradients

data_normalized = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
for j in range(data.shape[-1]):
	data_normalized[j] = (data[:,:,j] - data[:,:,j].min()) / (data[:,:,j].max() - data[:,:,j].min())

data_reshaped = data_normalized.reshape(num_pixels, num_bands)

# Assume initiating bias is 0
layer1 = relu(data_reshaped @ w1 + b1)
layer2 = relu(layer1 @ w2 + b2)
# Bottleneck layer
layer3 = relu(layer2 @ w3 + b3)

# Decoder
w4 = weight_init_He(8,16)
b4 = 0
w5 = weight_init_He(16,64)
b5 = 0
w6 = weight_init_He(64,num_bands)
b6 = 0

layer4 = relu(layer3 @ w4 + b4)
layer5 = relu(layer4 @ w5 + b5)
# Reconstructed layer, no activation (linear output)
layer6 = layer5 @ w6 + b6

# Mean squared error to start with
d_cost = mse_der(data_reshaped,layer6, num_bands)

w_list = [w1, w2, w3, w4, w5, w6]
b_list = [b1, b2, b3, b4, b5, b6]
l_list = [data_reshaped, layer1, layer2, layer3, layer4, layer5, layer6]

############################################
############# Gradient Descent #############
############################################

# Learning rate
lr = 0.1

iteration = 1
#while mse(layer6, data_reshaped) >= :
#	iteration += 1
#	update_params(w_list, b_list, l_list, d_cost)
