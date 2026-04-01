import scipy.io
from scipy.io import loadmat
from scipy.stats import multivariate_normal
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

# Initializing model parameters

# Covariance matrix from mean-centered data
# n = number of pixels, d = number of dimensions
n,d = bottleneck.shape[0], bottleneck.shape[1]

# Initiate random cluster Weights
# (7,)
w_pi_init = np.random.random(gt_classnum)
w_pi_init /= w_pi_init.sum()

# 7138 x 10
b = bottleneck

# Initiate mean vectors using FPS
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

# Constructing mean-centered feature matrix
# Initial cov matrix is the same for every initial mean vector
# 7138 x 10
Xb = np.zeros((n,d))
for i in range(d):
    Xb[:,i] = bottleneck[:,i] - np.mean(bottleneck[:,i])
# 10 x 10
cov_1 = Xb.T @ Xb / n
cov_list = [cov_1]
cov_list = cov_list * gt_classnum
# 7 x 10 x 10
cov_init = np.stack(cov_list, axis=0) 

# Expectation Maximization Function

def expect_max(weights, means, covs, bneck, classnum):
    # Bottleneck : 7138 x 10
    # w_pi : (7,)
    # mu_init : 7 x 10
    # covariance : 10 x 10

    pixels = bneck.shape[0]
    dims = bneck.shape[1]
    # posterior : 7138 x 7
    posterior = np.zeros((pixels,classnum))
    # expectation
    for k in range(classnum):
        # 7138 x 1
        posterior[:,k] = weights[k] * multivariate_normal.pdf(bneck, means[k,:], covs[k,:,:])

    # normalizing posterior for each pixel across all clusters:
    # (7138,)
    post_sum = np.sum(posterior, axis=1)[:,None]
    print(post_sum)
    # 7138 x 7
    post_norm = posterior / post_sum

    log_like = np.sum(np.log(post_sum))

    # (7,)
    count = np.sum(post_norm,axis=0)

    l = 10**-7      # smaller than smallest diagonal element in initiating covariance matrix
    # maximization
    for k in range(classnum):
        weights[k] = count[k] / pixels
        means[k,:] = post_norm[:,k] @ bneck / count[k]

        # Deviation, 7138 x 10
        dev = (bneck - means[k,:])
        # Responsibility, 7138 x 1
        resp = post_norm[:,k][:,None]
        # 1 x 10 x 10
        covs[k,:,:] = dev.T @ (resp * dev) / count[k]
        covs[k,:,:] = covs[k,:,:] + l*np.eye((dims))

    # In the first run, the log likelihood corresponds to initiating values
    return(log_like, weights, means, covs)

# Running GMM

epsilon = 10**-3
LL_old, w_pi, mu, covariance = expect_max(w_pi_init, mu_init, cov_init, bottleneck, gt_classnum)
#print(covariance)
LL_new = expect_max(w_pi,mu, covariance, bottleneck, gt_classnum)[0]
epoch = 1

while np.abs(LL_new - LL_old) > epsilon:
    epoch =+ 1
    LL_old = LL_new
    LL_new, w_pi, mu, covariance = expect_max(w_pi,mu, covariance, bottleneck, gt_classnum)

print(epoch)
plt.show()




