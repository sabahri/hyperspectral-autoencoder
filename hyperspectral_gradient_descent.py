import scipy.io
from scipy.io import loadmat
import kneed as kn
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

import hyperspectral_functions as hf
from hyperspectral_functions import kl
from hyperspectral_functions import data_z_reshaped

data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('SalinasA_gt.mat')['salinasA_gt']

# For SalinasA, full data shape is 83 x 86 x 204
# Spatial grid is 83 x 86 
# 204-dimensional band vector (basically 204 channels instead of 3)
num_pixels = data.shape[0] * data.shape[1]
num_bands = data.shape[-1]

############################################
############# Gradient Descent #############
############################################
np.random.seed(42)

lr = 10.**-2
epoch = 1

biases = np.zeros((num_pixels,1))

# Encoder inits
w1 = hf.weight_init_He(num_bands,64)
b1 = np.zeros((1,64))
w2 = hf.weight_init_He(64,16)
b2 = np.zeros((1,16))
w3 = hf.weight_init_He(16,int(kl.knee))
b3 = np.zeros((1,int(kl.knee)))

# Decoder inits
w4 = hf.weight_init_He(int(kl.knee),16)
b4 = np.zeros((1,16))
w5 = hf.weight_init_He(16,64)
b5 = np.zeros((1,64))
w6 = hf.weight_init_He(64,num_bands)
b6 = np.zeros((1,num_bands))

w_list = [w1, w2, w3, w4, w5, w6]	
b_list = [b1, b2, b3, b4, b5, b6]


output = hf.forward_pass(w_list, b_list, data_z_reshaped)[0]
cost = hf.mse_cost(data_z_reshaped, output, num_pixels, num_bands)
d_cost = hf.mse_der(data_z_reshaped, output, num_pixels, num_bands)
cost_list = [cost]


# Desired cost optimum given z-score normalization
while cost > 0.3:
	epoch += 1
	output, l_list, cost = hf.forward_pass(w_list, b_list, data_z_reshaped)
	d_cost = hf.mse_der(data_z_reshaped, output, num_pixels, num_bands)
	w_list, b_list = hf.update_params(w_list, b_list, l_list, lr, d_cost, num_pixels)
	cost_list.append(cost)
	
	# if epoch == 2001:
	# 	print("Cost at epoch 2000:", cost)
	# 	break

l_list = hf.forward_pass(w_list,b_list,data_z_reshaped)[1]
bottleneck = l_list[3]

np.savez('trained_model.npz',
	w0=w_list[0], w1=w_list[1], w2=w_list[2], w3=w_list[3], w4=w_list[4], w5=w_list[5],
    b0=b_list[0], b1=b_list[1], b2=b_list[2], b3=b_list[3], b4=b_list[4], b5=b_list[5],
    bottleneck=bottleneck)

epochs = np.linspace(1, epoch, epoch)

fig, ax = plt.subplots()

ax.plot(epochs, np.asarray(cost_list))
ax.set(xlabel='epoch', ylabel='Cost (MSE Loss)')
ax.grid()
plt.show()

############################################
############# Gradient Descent #############
#############   (Coarse) for   #############
############# Learning Rate Opt ############
############################################

# # Learning rates being tested
# e = np.array((-5, -4, -3, -2, -1))
# lr = 10.**e
# num_rates = lr.shape[0]
# num_epochs = 100
# epochs = np.linspace(1,num_epochs,num_epochs)
# cost = np.zeros((num_rates, num_epochs))

# for i in range(num_rates):
# 	epoch = 1

# 	# Reinitializing values for each learning rate

# 	biases = np.zeros((num_pixels,1))

# 	# Encoder inits
# 	w1 = weight_init_He(num_bands,64)
# 	b1 = np.zeros((1,64))
# 	w2 = weight_init_He(64,16)
# 	b2 = np.zeros((1,16))
# 	w3 = weight_init_He(16,8)
# 	b3 = np.zeros((1,8))

# 	# Decoder inits
# 	w4 = weight_init_He(8,16)
# 	b4 = np.zeros((1,16))
# 	w5 = weight_init_He(16,64)
# 	b5 = np.zeros((1,64))
# 	w6 = weight_init_He(64,num_bands)
# 	b6 = np.zeros((1,num_bands))

# 	w_list = [w1, w2, w3, w4, w5, w6]
# 	b_list = [b1, b2, b3, b4, b5, b6]

# 	for j in range(num_epochs):

# 		#print("Epoch:", epoch)

# 		output, l_list, cost[i,j] = forward_pass(w_list, b_list, data_z_reshaped)
# 		d_cost = mse_der(data_z_reshaped, output, num_pixels, num_bands)

# 		w_list, b_list = update_params(w_list, b_list, l_list, lr[i], d_cost, num_pixels)

# 		epoch += 1
	
# colors = cm.rainbow(np.linspace(0,1,num_rates))
# fig, ax = plt.subplots()

# for i in range(num_rates):
# 	ax.plot(epochs, cost[i,:], label=f"lr: {lr[i]:.0e}")

# ax.set(xlabel='epoch', ylabel='Cost (MSE Loss)')
# ax.grid()
# ax.legend()
# plt.show()
