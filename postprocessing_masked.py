# Resource for log-sum-exp trick: https://mc-stan.org/docs/2_27/stan-users-guide/log-sum-of-exponentials.html

import scipy.io
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
#import kneed as kn
#import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys
import itertools
import umap

plt.rcParams.update({'font.size': 14})

##### Loading data
data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('SalinasA_gt.mat')['salinasA_gt']
ground_truth_flat = ground_truth.reshape(ground_truth.shape[0]*ground_truth.shape[1],)
unique_labels = np.unique(ground_truth_flat)
gt_classnum = len(unique_labels)

recoded = np.searchsorted(unique_labels,ground_truth_flat)

checkpoint1 = np.load('trained_model.npz')
w_list = [checkpoint1[f'w{i}'] for i in range(6)]
b_list = [checkpoint1[f'b{i}'] for i in range(6)]

checkpoint2 = np.load('bott_output.npz')
bottleneck = checkpoint2['bottleneck']
output = checkpoint2['output']

##### Repeat normalizing

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
np.random.seed(42)

# Initializing model parameters

# Covariance matrix from mean-centered data
# n = number of pixels, d = number of dimensions
n,d = bottleneck.shape[0], bottleneck.shape[1]

# Initiate random cluster Weights
# (7,)
w_pi_init = np.random.random(gt_classnum)
w_pi_init /= w_pi_init.sum()

# 7138 x 10
bott = bottleneck

# Initiate mean vectors using FPS
mu_init = []
ri = np.random.randint(1,n)
vec = bott[ri,:]
mu_init.append(vec)

# 7137 x 10
bott = np.delete(bott, ri, 0)

# Farthest Point Sampling
for i in range(1, gt_classnum):
    dist = []
    # Parse through current mu vectors
    for j in range(len(mu_init)):
        # subtract mu vector from remaining bottleneck vectors
        dist.append(np.linalg.norm(bott - mu_init[j], axis = 1))
    dist = np.vstack(dist)

    min_dist = np.min(dist,axis=0)
    new_mu_ind = np.argmax(min_dist)
    mu_init.append(bott[new_mu_ind])

    bott = np.delete(bott, new_mu_ind, 0)

# 7 x 10
mu_init = np.asarray(mu_init)
mu_init_copy = mu_init.copy()

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
    #posterior = np.zeros((pixels,classnum))
    log_post = np.zeros((pixels,classnum))

    # expectation
    for k in range(classnum):
        # 7138 x 1
        log_post[:,k] = np.log(weights[k]) + multivariate_normal.logpdf(bneck, means[k,:], covs[k,:,:])

    # log-sum-exp trick
    # 7138 x 1
    log_post_max = np.max(log_post, axis=1, keepdims=True)
    log_post_diff = log_post - log_post_max
    exp_log_post_diff = np.exp(log_post_diff)
    # log_post_sum = log-sum-exp
    # 7138 x 1
    log_post_sum = log_post_max + np.log(np.sum(exp_log_post_diff, axis=1,keepdims=True))
    log_like = np.sum(log_post_sum)

    # 7138 x 7
    post_norm = np.exp(log_post - log_post_sum)
    label = np.argmax(post_norm, axis=1)

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
    return(label, log_like, weights, means, covs)

# Running GMM

epsilon = 10**-3
assign_gmm, LL_old, w_pi, mu_gmm, covariance = expect_max(w_pi_init, mu_init, cov_init, bottleneck, gt_classnum)
LL_new = expect_max(w_pi,mu_gmm, covariance, bottleneck, gt_classnum)[1]
epoch = 1

while np.abs(LL_new - LL_old) > epsilon:
    epoch +=1
    LL_old = LL_new
    assign_gmm, LL_new, w_pi, mu_gmm, covariance = expect_max(w_pi,mu_gmm, covariance, bottleneck, gt_classnum)

############################################
############# KMC on Bottleneck #############
#############################################

def kmc(means, bneck, classes):
    # means : 7 x 10
    dist_list = []
    for k in range(classes):
        # 7138 x 10
        #print(means[k,:][None,:].shape)
        diff = bneck - means[k,:][None,:]
        # 7138 x 1
        diff_norm = np.linalg.norm(diff, axis=1,keepdims=True)
        # length 7, with 7138 x 1 components
        dist_list.append(diff_norm)

    # 7138 x 7
    dist_array = np.hstack(dist_list)

    # 7138 x 1 each row indexes crop classes 0 to 6
    label = np.argmin(dist_array, axis=1)
    # 7 x 1
    labels, counts = np.unique(label, return_counts=True)

    # 7 x 10
    new_mean = np.zeros((means.shape[0], means.shape[1]))
    for k in range(classes):
        # 1 x 10
        new_mean[k,:] = np.sum(bneck[label == k,:], axis=0)[None,:]
    
    new_mean = new_mean / counts[:,None]

    return(label, new_mean)

assign_kmc, mu_kmc = kmc(mu_init_copy, bottleneck, gt_classnum)
for i in range(10):
    assign_kmc, mu_kmc = kmc(mu_kmc, bottleneck, gt_classnum)

###############################################
############# Hungarian Algorithm #############
###############################################

# Constructing confusion matrix
# Rows represent GT labels, Columns represent cluster assignment
# Each entry = num pixels
conf_kmc = np.zeros((gt_classnum, gt_classnum))
conf_gmm = np.zeros((gt_classnum, gt_classnum))

for p in range(n):
    conf_kmc[recoded[p], assign_kmc[p]] += 1
    conf_gmm[recoded[p], assign_gmm[p]] += 1

row_ind_kmc, col_ind_kmc = linear_sum_assignment(-conf_kmc)
row_ind_gmm, col_ind_gmm = linear_sum_assignment(-conf_gmm)

# Remap pixels from clustering assignment to ground truth assignment
def remap(labels, pixels, col_ind, classes):
    new_assignment = np.zeros((pixels, 1))
    new_map = np.zeros((classes,))

    for i in range(classes):
        new_map[col_ind[i]] = i

    for j in range(pixels):
        new_assignment[j] = new_map[labels[j]]

    return(new_assignment)

assign_kmc = remap(assign_kmc, n, col_ind_kmc, gt_classnum)
assign_gmm = remap(assign_gmm, n, col_ind_gmm, gt_classnum)

###############################################
############# Visualizing Results #############
###############################################

# Raw data
# Normalizing band values to 1
band_ind = 100
band = data[:,:,band_ind]
band_min = band.min()
band_max = band.max()

band_normalized = (band - band_min)/(band_max - band_min)
assign_map_gmm = assign_gmm.reshape(data.shape[0], data.shape[1])
assign_map_kmc = assign_kmc.reshape(data.shape[0], data.shape[1])

# Un-normalizing the output
#recon = np.multiply(output, stdev) + mean
#recon = recon.reshape(data.shape[0], data.shape[1], data.shape[2])

p_loss = np.mean((output - data_z_reshaped)**2, axis=1)
ploss_h = np.percentile(p_loss, 99)
ploss_l = np.percentile(p_loss, 1)

# Clipping highest and lowest value pixels
p_loss = np.clip(p_loss, ploss_l, ploss_h)
p_loss = (p_loss - p_loss.min()) / (p_loss.max() - p_loss.min())

p_loss = p_loss.reshape(data.shape[0], data.shape[1])

fig_loss, ax_loss = plt.subplots(figsize=(5, 5))
im = ax_loss.imshow(p_loss, cmap='plasma', vmin=0, vmax=1)
ax_loss.set_title('Per-Pixel Loss')
fig_loss.colorbar(im, ax=ax_loss, label='Per-Pixel Loss', fraction=0.03, pad=0.04)
plt.tight_layout()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

ax1.imshow(assign_map_kmc, cmap=cmap, vmin=0, vmax=gt_classnum-1)
ax1.set_title('Band 100, KMC Mapping')

ax2.imshow(assign_map_gmm, cmap=cmap, vmin=0, vmax=gt_classnum-1)
ax2.set_title('Band 100, GMM Mapping')

gt_map = recoded.reshape(data.shape[0], data.shape[1])
ax3.imshow(gt_map, cmap=cmap, vmin=0, vmax=gt_classnum-1)
ax3.set_title('Ground Truth')

plt.subplots_adjust(bottom=0.15)
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))
plt.tight_layout

####################################################
############# More Bottleneck Analysis #############
####################################################

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(d):
    axes[i].hist(bottleneck[:, i], bins=50, range=(-1, 1), color='#414fc8')
    axes[i].set_title(f'Dim {i}')
    axes[i].set_xlim(-1, 1)

plt.suptitle('Per-dimension bottleneck activation histograms')
plt.tight_layout()

plt.show()
