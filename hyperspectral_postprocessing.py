import scipy.io
from scipy.io import loadmat
#import kneed as kn
#import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

Titles = ["SalinasA Band 100", "Per Pixel Loss"]
images = [band_normalized,p_loss] #, edges]
count = len(images)

plt.figure()

for i in range(count):
    plt.subplot(1, len(images), i+1)
    plt.title(Titles[i])
    plt.imshow(images[i])
    if i == 1:
        plt.imshow(p_loss, cmap='plasma',vmin=0,vmax=1)
    else:
        plt.imshow(band_normalized, cmap='gray',vmin=0,vmax=1)

plt.tight_layout

# # Plotting
# plt.imshow(band_normalized, cmap='gray',vmin=0,vmax=1)
# plt.show()

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

# Expectation

# Cluster Weights
w_pi = np.random.random(gt_classnum)
w_pi /= w_pi.sum()

# Covariance matrix from mean-centered data
# n = number of pixels, m = number of dimensions
n,m = bottleneck.shape[0], bottleneck.shape[1]

# Constructing mean-centered feature matrix
# 7138 x 10
Xb = np.zeros((n,m))

for i in range(m):
    Xb[:,i] = bottleneck[:,i] - np.mean(bottleneck[:,i])

# Covariance matrix
cov = Xb.T @ Xb / n

# mu = np.zeros((m,gt_classnum))
# mu[0,:] = bottleneck[np.random.randint(1,n),:]
# for i in range(n):

plt.show()




