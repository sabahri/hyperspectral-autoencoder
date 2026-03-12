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
def mse_cost(x,y,n):
	loss = (y - x)**2
	return(np.sum(loss) / n)

def mse_der(x,y):
	return(-2*(y - x))

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

def forward_pass(W,b,data_array, channels):
	num_examples = data_array.shape[1]
	
	# Encoder
	layer1 = relu(data_array @ W[0] + b[0])
	layer2 = relu(layer1 @ W[1] + b[1])
	# Bottleneck layer
	layer3 = relu(layer2 @ W[2] + b[2])

	# Decoder
	layer4 = relu(layer3 @ W[3] + b[3])
	layer5 = relu(layer4 @ W[4] + b[4])

	# Reconstructed layer, no activation (linear output)
	layer6 = layer5 @ W[5] + b[5]

	# List of layers
	layer_list = [data_array, layer1, layer2, layer3, layer4, layer5, layer6]

	# Cost function
	J = mse_cost(data_array, layer6, num_examples)

	#print(J.shape)
	return(layer6, layer_list, J)

def update_params(W, b, layer, learning_rate, d_J):
	# W : list of weights
	# W[i] : weight matrix applied to the ith layer
	# b : list of biases
	# layer: list of layers
	# d_J : derivative of cost function
	# layer_ind : layer we are backpropagating to

	# Recursively backpropagating with dz and dq base case
	dz = d_J												# starts at dz_6
	dq = dz @ W[-1].T										# dq_5

	W[-1] = W[-1] - learning_rate * layer[-2].T @ dz 		# note W[-1] = W[5] = w6 here
	b[-1] = b[-1] - learning_rate * dz

	for i in range(len(W) - 1, 0,-1):
		print("weight", W[i].T.shape)							# Iterating from i = 5 to 1
		print("layer", layer[i].shape)
		dz = np.multiply(dq, relu_der(layer[i]))			# dz_5
		print("dz", dz.shape)
		dq = dz @ W[i].T									# dq_4

		# Calculating gradients for weights / biases per layer
		dW = layer[i-1].T @ dz 
		db = dz

		# Updating parameters, layer by layer
		W[i] = W[i] - learning_rate * dW
		b[i] = b[i] - learning_rate * db
	
	return(W, b)

############################################
########### Data Normalization #############
############################################

# Reshape data
# 83 x 86 = 7138 pixels
# So we need to break it down to 7138 1x204-dim vectors
data_reshaped = data.reshape(num_pixels, num_bands)

# Perform Per-Band Normalization to avoid exploding gradients
# Per band standard deviations shows that the standard deviations vary widely, indicating that
# Min-Max normaization would skew the MSE loss towards bands with high variance. Note: each 
# band is unlikely to be uniformly distributed

devs = np.zeros((num_bands))

for i in range(num_bands):
	devs[i] = np.std(data_reshaped[:,i])

## min-max scaling
# data_normalized = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
# for j in range(data.shape[-1]):
# 	data_normalized[:,:,j] = (data[:,:,j] - data[:,:,j].min()) / (data[:,:,j].max() - data[:,:,j].min())

# z-scoring
# Input data is normalized so that the pixels of each band are centered on 0, with stdev = 1
# As a result, a useless network will produce MSE loss ~ 1

data_z = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
epsilon = 10**-6

for j in range(data.shape[-1]):
	mean = np.mean(data[:,:,j])
	std = np.std(data[:,:,j])
	data_z[:,:,j] = (data[:,:,j] - mean) / (std + epsilon)		# add infinitesimal epsilon in case std = 0

data_z_reshaped = data_z.reshape(num_pixels, num_bands)

############################################
############## Initialization ##############
############################################

biases = np.zeros((num_pixels,1))

# Encoder inits
w1 = weight_init_He(num_bands,64)
b1 = 0
w2 = weight_init_He(64,16)
b2 = 0
w3 = weight_init_He(16,8)
b3 = 0

# Decoder inits
w4 = weight_init_He(8,16)
b4 = 0
w5 = weight_init_He(16,64)
b5 = 0
w6 = weight_init_He(64,num_bands)
b6 = 0

w_list = [w1, w2, w3, w4, w5, w6]
b_list = [b1, b2, b3, b4, b5, b6]
l_list = forward_pass(w_list, b_list, data_z_reshaped, num_bands)[1]

# Mean squared error to start with
output = forward_pass(w_list, b_list, data_z_reshaped, num_bands)[0]
d_cost = mse_der(data_z_reshaped,output)

############################################
############# Gradient Descent #############
#############   (Coarse) for   #############
############# Learning Rate Opt ############
############################################

# Learning rate
e = np.array((-5, -4, -3, -2, -1))
lr = 10.**e
num_rates = lr.shape[0]
num_epochs = 20
cost = np.zeros((num_rates, num_epochs))
epoch = 1

for i in range(num_rates):
	for j in range(num_epochs):
		print("Epoch:", epoch)
		output = forward_pass(w_list, b_list, data_z_reshaped, num_bands)[0]
		cost[i,j] = forward_pass(w_list, b_list, data_z_reshaped, num_bands)[2]	
		d_cost = mse_der(data_z_reshaped, output)
		w_list, b_list = update_params(w_list, b_list, l_list, lr[i], d_cost)
		epoch += 1
	
colors = cm.rainbow(np.linspace(0,1,num_rates))
fig, ax = plt.subplots()

for i in range(num_rates):
	ax.plot(num_epochs, cost[i,:])

ax.set(xlabel='epoch', ylabel='Cost (MSE Loss)')
ax.grid()
plt.show()


