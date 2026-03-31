import scipy.io
from scipy.io import loadmat
#import kneed as kn
#import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import umap


import hyperspectral_functions as hf
from hyperspectral_functions import data_z_reshaped

data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('SalinasA_gt.mat')['salinasA_gt']
ground_truth_flat = ground_truth.reshape(ground_truth.shape[0]*ground_truth.shape[1],)
unique_labels = np.unique(ground_truth_flat)
gt_classnum = len(unique_labels)

recoded = np.searchsorted(unique_labels,ground_truth_flat)

checkpoint = np.load('trained_model.npz')

w_list = [checkpoint[f'w{i}'] for i in range(6)]
b_list = [checkpoint[f'b{i}'] for i in range(6)]
bottleneck = checkpoint['bottleneck']
output = checkpoint['output']

######################################################
############# Visualizing Per-Pixel Loss #############
######################################################

# Raw data
# Normalizing band values to 1
band_ind = 100
band = data[:,:,band_ind]
band_min = band.min()
band_max = band.max()

band_normalized = (band - band_min)/(band_max - band_min)

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
#plt.imshow(p_loss, cmap='plasma',vmin=0,vmax=1)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(band_normalized, cmap='gray', vmin=0, vmax=1)
ax1.set_title('Band 100')

im = ax2.imshow(p_loss, cmap='plasma', vmin=0, vmax=1)
ax2.set_title('Per-Pixel Loss')

fig.colorbar(im, ax=[ax1, ax2], label='Per-Pixel Loss', fraction=0.015, pad=0.04)
plt.subplots_adjust(right=0.85)
################################
############# UMAP #############
################################

# https://umap-learn.readthedocs.io/en/latest/index.html

fit = umap.UMAP(n_components=2,init='random')
u = fit.fit_transform(bottleneck)

fig = plt.figure()

colors = ['gray', 'blue', 'orange', 'green', 'red', 'purple', 'brown']
cmap = plt.matplotlib.colors.ListedColormap(colors)

scatter = plt.scatter(u[:,0],u[:,1],c=recoded,cmap=cmap,vmin=0,vmax=6, edgecolor='k', linewidth=0.3)
class_names = ['Background', 'Brocoli_green_weeds_1', 'Corn_senesced_green_weeds', 
                'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 
                'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk']

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
           markersize=8, label=class_names[i]) for i in range(len(class_names))]

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.title('UMAP embedding of Bottleneck features')

#############################################
############# GMM on Bottleneck #############
#############################################

################ Expectation ################

# Cluster Weights
w_pi = np.random.random(gt_classnum)
w_pi /= w_pi.sum()

# Covariance matrix from mean-centered data
# n = number of pixels, d = number of dimensions
n,d = bottleneck.shape[0], bottleneck.shape[1]

# Constructing mean-centered feature matrix
# Then the covariance matrix
# 7138 x 10
Xb = np.zeros((n,d))
for i in range(d):
    Xb[:,i] = bottleneck[:,i] - np.mean(bottleneck[:,i])
cov = Xb.T @ Xb / n

# 7138 x 10
b = bottleneck

mu_init = []
ri = np.random.randint(1,n)
vec = b[ri,:]
mu_init.append(vec)
# 7137 x 10
b = np.delete(b, ri, 0)

# Farthest Point Sampling
for i in range(1, gt_classnum):
    dist = []
    # Parse through current mu vectors
    for j in range(len(mu_init)):
        # subtract mu vector from remaining bottleneck vectors
        dist.append(np.linalg.norm(b - mu_init[j], axis = 1))
    dist = np.vstack(dist)

    min_dist = np.min(dist,axis=0)
    new_mu_ind = np.argmax(min_dist)
    mu_init.append(b[new_mu_ind])

    b = np.delete(b, new_mu_ind, 0)

# 7 x 10
mu_init = np.asarray(mu_init)

################ Maximization ################

plt.show()




