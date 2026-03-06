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

# W is my list of weights
# W[i] is the weight matrix applied to the ith layer
def backprop(W, layer, d_J, num_hidden_layers):

	d_z = d_J.T
	for i in range(len(W)-1, -1, -1):
		d_z = W[i].T @ d_z

		# need to fix indices for layers and weights
		d_z = np.multiply(relu_der(layer[i].T), d_z)

	return(d_z)

def gradients(W, layer, num_hidden_layers, d_J):

	d_z = backprop(W, layer, d_J, num_hidden_layers)
	dW = d_z @ layer.T 
	db = d_z

	return(dW, db)

def update_params(W, b, layer, num_hidden_layers, d_J, learning_rate):

	dW,db = gradients(W, layer, num_hidden_layers, d_J)
	new_W = W - learning_rate * dW
	new_b = b - learning_rate * db

	return(new_W, new_b)

############################################
############## Initialization ##############
############################################

biases = np.zeros((num_pixels,1))

# Encoder
w0 = weight_init_He(num_bands,64)
b0 = 0
w1 = weight_init_He(64,16)
b1 = 0
w2 = weight_init_He(16,8)
b2 = 0

# Assume initiating bias is 0
layer1 = relu(data_reshaped @ w0 + b0)
layer2 = relu(layer1 @ w1 + b1)
# Bottleneck layer
layer3 = relu(layer2 @ w2 + b2)

# Decoder
w3 = weight_init_He(8,16)
b3 = 0
w4 = weight_init_He(16,64)
b4 = 0
w5 = weight_init_He(64,num_bands)
b5 = 0

layer4 = relu(layer3 @ w3 + b3)
layer5 = relu(layer4 @ w4 + b4)
# Reconstructed layer, no activation (linear output)
layer6 = layer5 @ w5 + b5

# Mean squared error to start with
d_cost = mse_der(data_reshaped,layer6, num_bands)

w_list = [w0, w1, w2, w3, w4, w5]
b_list = [b0, b1, b2, b3, b4, b5]
l_list = [data_reshaped, layer1, layer2, layer3, layer4, layer5, layer6]

############################################
############# Parameter Updates ############
############################################

# Learning rate
lr = 0.1
# Number of hidden layers
nhr = len(w_list) - 1

grad5 = d_cost @ w5.T

w_list, b_list = update_params(w_list, b_list, l_list, nhr, grad5, lr)
