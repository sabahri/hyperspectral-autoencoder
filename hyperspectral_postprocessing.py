import scipy.io
from scipy.io import loadmat
import kneed as kn
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

import hyperspectral_functions as hf
import hyperspectral_gradient_descent as hgd

######################################################
############# Visualizing Per-Pixel Loss #############
######################################################

# Un-normalizing the output
#recon = np.multiply(output, stdev) + mean
#recon = recon.reshape(data.shape[0], data.shape[1], data.shape[2])

p_loss = hf.mse_pixel_loss(data_z_reshaped, output)
ploss_h = np.percentile(p_loss, 99)
ploss_l = np.percentile(p_loss, 1)

# Clipping highest and lowest value pixels
p_loss = np.clip(p_loss, ploss_l, ploss_h)


p_loss = (p_loss - p_loss.min()) / (p_loss.max() - p_loss.min())

p_loss = p_loss.reshape(data.shape[0], data.shape[1])
plt.imshow(p_loss, cmap='gray',vmin=0,vmax=1)
plt.show()

#############################################
############# KMC on Bottleneck #############
#############################################

# To compare against ground truth labels

