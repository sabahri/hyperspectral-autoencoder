# Implementing an autoencoder from scratch
# https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
# https://medium.com/@dipikadhara06/understanding-hyperspectral-imaging-reading-and-visualizing-data-from-mat-files-using-python-295d5cb83e0a
# https://researchdatapod.com/building-autoencoders-pytorch-tutorial/
# https://www.freecodecamp.org/news/building-a-neural-network-from-scratch/

import scipy.io
from scipy.io import loadmat
import kneed as kn
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

###########################################
########### Plotting Original #############
###########################################

# # Normalizing band values to 1
# band_ind = int(sys.argv[1])
# band = data[:,:,band_ind]
# band_min = band.min()
# band_max = band.max()

# band_normalized = (band - band_min)/(band_max - band_min)

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
def mse_cost(x,y,n,m):
	# y: reconstruction
	# x: target
	# n: number of examples
	# m: number of bands

	loss = (y - x)**2
	return(np.sum(loss) / (n*m))

def mse_pixel_loss(x,y):
	return(np.mean((y - x)**2, axis=1))

def mse_der(x,y,n,m):
	# y: reconstruction
	# x: target
	# n: number of examples
	# m: number of bands

	return(2*(y - x) / (n*m))

############################################
############### Weight Init ################
########## And Backpropagation #############
############################################

def weight_init_He(n,m):
	# He weight initialization: scale by 2/sqrt(# inputs to layer)
	# n: input layer size
	# m: desired output layer size

	stdev = np.sqrt(2 / n)
	w = np.random.normal(0,stdev,size=(n,m))
	return(w)

def forward_pass(W,b,data_array):
	num_examples = data_array.shape[0]
	channels = data_array.shape[1]
	
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
	J = mse_cost(data_array, layer6, num_examples, channels)

	#print(J.shape)
	return(layer6, layer_list, J)

def update_params(W, b, layer, learning_rate, d_J, num_examples):
	# W : list of weights
	# W[i] : weight matrix applied to the ith layer
	# b : list of biases
	# layer: list of layers
	# d_J : derivative of cost function
	# layer_ind : layer we are backpropagating to
	# z : layer before activation
	# q: layer after activation

	# Recursively backpropagating with dz and dq base case

	dz = d_J												# 7138 x 204
	dq = dz @ W[-1].T										# dq_5, 7138 x 64

	W[-1] = W[-1] - learning_rate * layer[-2].T @ dz 		# note W[-1] = W[5] = w6 here
	b[-1] = b[-1] - learning_rate * dz


	# Iterating from 4 to 0
	for i in range(len(W) - 2, 0,-1):						# want start with w5 update, so W[4]
		dz = np.multiply(dq, relu_der(layer[i+1]))			# Layer 5, 7138 x 64		

		# For the next iteration
		dq = dz @ W[i].T  									# dq_4, 7138 x 16

		# Calculating gradients
		dW = layer[i].T @ dz / num_examples 					# dW should have dimensions 16 x 64
		db = np.sum(dz, axis=0, keepdims=True) / num_examples	# 1 x 64

		# Updating parameters
		W[i] = W[i] - learning_rate * dW					# W[4] = w5, should be 16 x 64
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

######################################################
########### Principal Component Analysis #############
######################################################
# Intrinsic dimensionality of data's linear structure

# Correlation matrix
corr = data_z_reshaped.T @ data_z_reshaped / num_pixels
eig_val, eig_vec = np.linalg.eigh(corr)				# using eigh instead of eig for symmetrix matrices

# Cumulative explained variance ratio
cumul = np.cumsum(eig_val[::-1]) / np.sum(eig_val)
pca_num = np.linspace(1, num_bands, num_bands)

# Kneedle algorithm to find curve elbow
# the eigh fxn stores in order of increasing values
# --> reversal in cumul should always be concave
kl = kn.KneeLocator(pca_num, cumul, curve="concave", direction="increasing")
kl.plot_knee()
plt.axvline(x = kl.knee,  color='red', linestyle='--', 
	label=f'Knee: {kl.knee:.2f} \n Explained Var:{100*cumul[int(kl.knee)]:.2f}%')
plt.xlabel('Principal Component Axis Number')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.legend()
plt.show()
#sys.exit()

# Conclusion: Knee location was at 10, so bottleneck should be around 10 dims

