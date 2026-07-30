import scipy.io
from scipy.io import loadmat
import kneed as kn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

import neural_net as nn
np.seterr(all='raise')

##### Data import and z-normalization

data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('SalinasA_gt.mat')['salinasA_gt']

# SalinasA: 83 x 86 spatial grid
# 204 channels
num_pixels = data.shape[0] * data.shape[1]
num_bands = data.shape[-1]

# 7138 1x204-dim vectors
data_reshaped = data.reshape(num_pixels, num_bands)

# z-scoring
# Normalize input so that the pixels of each band are centered on 0, with stdev = 1
# --> A useless network will produce MSE loss ~ 1
data_z = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
epsilon = 10**-6

for j in range(data.shape[-1]):
	mean = np.mean(data[:,:,j])
	std = np.std(data[:,:,j])
	data_z[:,:,j] = (data[:,:,j] - mean) / (std + epsilon)		# add infinitesimal epsilon in case std = 0

data_z_reshaped = data_z.reshape(num_pixels, num_bands)

##### PCA and Kneedle analysis for bottleneck vector optimization:
##### Detect intrinsic dimensionality of data's linear structure
# Correlation matrix
corr = data_z_reshaped.T @ data_z_reshaped / num_pixels
eig_val, eig_vec = np.linalg.eigh(corr)

# Cumulative explained variance ratio
cumul = np.cumsum(eig_val[::-1]) / np.sum(eig_val)
pca_num = np.linspace(1, num_bands, num_bands)

# Kneedle algorithm to find curve elbow
kl = kn.KneeLocator(pca_num, cumul, curve="concave", direction="increasing")
kl.plot_knee()
if __name__ == "__main__":
	plt.axvline(x = kl.knee,  color='red', linestyle='--', 
		label=f'Knee: {kl.knee:.2f} \n Explained Var:{100*cumul[int(kl.knee)]:.2f}%')
	plt.xlabel('Principal Component Axis Number')
	plt.ylabel('Cumulative Explained Variance Ratio')
	plt.legend()
	plt.show(block=False)
	plt.pause(0.1)

bott_dim = int(kl.knee)
#print("Number of bottleneck dims:", bott_dim)


##### Gradient Descent

# # Learning Rate = 1.2
# lr = 1.2
# cost_minimum = 0.3

# np.random.seed(42)
# # Desired neural network architecture, bottleneck is after Tanh
# layers = [nn.Linear(num_bands, 64), nn.ReLU(), 
# 		nn.Linear(64, 16), nn.ReLU(), 
# #		nn.Linear(16, bott_dim), nn.Tanh(), 
# 		nn.Variational(16, bott_dim),
# 		nn.Linear(bott_dim, 16), nn.ReLU(), 
# 		nn.Linear(16,64), nn.ReLU(), 
# 		nn.Linear(64, num_bands)]

# # Bottleneck index
# bott_i = 4
# num_layers = (len(layers) + 1) / 2
# loss_function = nn.MSE()
# n_network = nn.MLP(layers, bott_i, loss_function, vae=True, lr)
# costs, output, b_neck = n_network.train(data_z_reshaped, cost_minimum)

# # Saving values from trained model
# saved_model = n_network.save_params()
# bott_output = n_network.save_output(b_neck,output)

# ##### GD Postmortem
# epoch = len(costs)
# epochs = np.linspace(1, epoch, epoch)

# fig, ax = plt.subplots()

# ax.plot(epochs, np.asarray(costs))
# ax.set(xlabel='epoch', ylabel='Cost (MSE Loss)')
# ax.grid()
# plt.show()



##### Optimizing learning rate
num_epochs = 1500
epochs = np.linspace(1,num_epochs,num_epochs)

#lr = np.array((0.0001, 0.001, 0.01, 0.1, 1.))
#lr = np.array((0.01,0.1,1.))
#lr = np.array((0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1))
lr = 0.04
#num_rates = lr.shape[0]
costs = np.zeros((num_epochs))
recon_costs = np.zeros_like(costs)
kl_costs = np.zeros_like(costs)

#colors = cm.rainbow(np.linspace(0,1,num_rates))
colors = cm.rainbow(np.linspace(0,1,3))
fig, axs = plt.subplots()

np.random.seed(42)
# Desired neural network architecture, bottleneck is after Tanh
layers = [nn.Linear(num_bands, 64), nn.ReLU(), 
		nn.Linear(64, 16), nn.ReLU(), 
#		nn.Linear(16, bott_dim), nn.Tanh(), 
		nn.Variational(16, bott_dim),
		nn.Linear(bott_dim, 16), nn.ReLU(), 
		nn.Linear(16,64), nn.ReLU(), 
		nn.Linear(64, num_bands)]
bott_i = 4
num_layers = (len(layers) + 1) / 2
loss_function = nn.MSE()
n_network = nn.MLP(layers, bott_i, loss_function, True, lr)
costs, recon_costs, kl_costs = n_network.opt_lr(data_z_reshaped, num_epochs)
axs.plot(epochs, costs, label="Total Cost")
axs.plot(epochs, recon_costs, label="Reconstruction Cost")
axs.plot(epochs, kl_costs, label="KL Cost")

# for i in range(num_rates):
# 	np.random.seed(42)
# 	# Desired neural network architecture, bottleneck is after Tanh
# 	layers = [nn.Linear(num_bands, 64), nn.ReLU(), 
# 			nn.Linear(64, 16), nn.ReLU(), 
# #			nn.Linear(16, bott_dim), nn.Tanh(), 
# 			nn.Variational(16, bott_dim),
# 			nn.Linear(bott_dim, 16), nn.ReLU(), 
# 			nn.Linear(16,64), nn.ReLU(), 
# 			nn.Linear(64, num_bands)]
# 	bott_i = 4
# 	num_layers = (len(layers) + 1) / 2
# 	loss_function = nn.MSE()
# 	n_network = nn.MLP(layers, bott_i, loss_function, True, lr[i])
# 	costs[i,:], recon_costs[i,:], kl_costs[i,:] = n_network.opt_lr(data_z_reshaped, num_epochs)
# 	axs[0,0].plot(epochs, costs[i,:], label=f"lr: {lr[i]:.2f}")
# 	axs[0,1].plot(epochs, recon_costs[i,:], label=f"lr: {lr[i]:.2f}")
# 	axs[0,2].plot(epochs, kl_costs[i,:], label=f"lr: {lr[i]:.2f}")

#axs.set(xlabel='epoch', ylabel='Cost (MSE Loss)')
#axs.grid()
axs.set_xlabel("Number of Epochs")
axs.set_ylabel("Cost")
axs.legend()
plt.show()



