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
data_reshaped = data.reshape(num_pixels, num_bands).T

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
	# n: input size
	# m: desired output size

	stdev = 2 / np.sqrt(n)
	w = np.random.normal(0,stdev,size=(m,n))

	return(w)

def backprop(weights, layers, d_J):

	num_hidden_layers = len(weights) - 1 
	d_z = d_J
	for i in range(num_hidden_layers , -1, -1):

		d_z = weights.T @ d_J
		d_z = np.multiply(relu_der(layers[i]), d_J)

	return(d_z)

def gradients(weights, layers, d_J):

	d_z = backprop(weights, layers, d_J)
	dW = d_z @ layers.T 
	db = d_z

	return(dW, db)

def update_params(W, b, layers, d_J, learning_rate):

	dW,db = gradients(layers, layers, d_J)
	new_W = W - learning_rate * dW
	new_b = b - learning_rate * db

	return(new_w, new_b)

############################################
############## Initialization ##############
############################################

biases = np.zeros((num_pixels,1))

# Encoder
w0 = weight_init_He(num_bands,64)
b0 = np.zeros((64,1))
w1 = weight_init_He(64,16)
b1 = np.zeros((16,1))
w2 = weight_init_He(16,8)
b2 = np.zeros((8,1))

# Assume initiating bias is 0
layer1 = relu(w0 @ data_reshaped)
layer2 = relu(w1 @ layer1)
# Bottleneck layer
layer3 = relu(w2 @ layer2)

# Decoder
w3 = weight_init_He(8,16)
b3 = np.zeros((16,1))
w4 = weight_init_He(16,64)
b4 = np.zeros((64,1))
w5 = weight_init_He(64,num_bands)
b5 = np.zeros((num_bands, 1))

layer4 = relu(w3 @ layer3)
layer5 = relu(w4 @ layer4)
# Reconstructed layer, no activation (linear output)
layer6 = w5 @ layer5

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

for i in range(len(w_list)):
	w_list[i], b_list[i] = update_params(w_list[i], b_list[i], l_list[i], d_cost, lr)
