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
print(f'Data Shape: {data.shape[:-1]}\nNumber of Bands:{data.shape[-1]}')

# Normalizing band values to 1
band_ind = int(sys.argv[1])
band = data[:,:,band_ind]
band_min = band.min()
band_max = band.max()

band_normalized = (band - band_min)/(band_max - band_min)

# Plotting
plt.imshow(band_normalized, cmap='gray',vmin=0,vmax=1)
plt.show()