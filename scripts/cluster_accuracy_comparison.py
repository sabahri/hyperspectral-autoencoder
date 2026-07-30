# Quantitative KMC/GMM clustering accuracy (Hungarian-matched against ground
# truth) across representations: vanilla AE bottleneck, VAE mean vectors,
# VAE sampled bottleneck, and a plain PCA projection (no autoencoder at all).
# Written to check whether visual impressions from the UMAP/KMC/GMM figures
# ("PCA/vanilla AE look like they recover class structure") hold up
# quantitatively, rather than relying on eyeballing the images.

import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
import kneed as kn

data = loadmat('data/SalinasA_corrected.mat')['salinasA_corrected']
ground_truth = loadmat('data/SalinasA_gt.mat')['salinasA_gt']
gt_flat = ground_truth.reshape(-1)
unique_labels = np.unique(gt_flat)
gt_classnum = len(unique_labels)
recoded = np.searchsorted(unique_labels, gt_flat)

num_pixels = data.shape[0]*data.shape[1]
num_bands = data.shape[-1]

# Pixels (row=12, col=13) and (row=12, col=14) show ~19-sigma within-class
# spikes isolated to mutually exclusive band groups (edge bands / water-
# absorption bands respectively), consistent with a sensor glitch. Excluded
# from band statistics and zeroed post-normalization, matching the masked
# training pipeline.
bad_pixel_mask = np.zeros((data.shape[0], data.shape[1]), dtype=bool)
bad_pixel_mask[12, 13] = True
bad_pixel_mask[12, 14] = True
good_pixel_mask = ~bad_pixel_mask

data_z = np.zeros_like(data, dtype=float)
for j in range(num_bands):
    mean = np.mean(data[:, :, j][good_pixel_mask])
    std = np.std(data[:, :, j][good_pixel_mask])
    data_z[:, :, j] = (data[:, :, j] - mean) / (std + 1e-6)
data_z[bad_pixel_mask, :] = 0
data_z_reshaped = data_z.reshape(num_pixels, num_bands)


def expect_max(weights, means, covs, bneck, classnum):
    pixels = bneck.shape[0]
    dims = bneck.shape[1]
    log_post = np.zeros((pixels, classnum))
    for k in range(classnum):
        log_post[:, k] = np.log(weights[k]) + multivariate_normal.logpdf(bneck, means[k, :], covs[k, :, :])
    log_post_max = np.max(log_post, axis=1, keepdims=True)
    exp_log_post_diff = np.exp(log_post - log_post_max)
    log_post_sum = log_post_max + np.log(np.sum(exp_log_post_diff, axis=1, keepdims=True))
    log_like = np.sum(log_post_sum)
    post_norm = np.exp(log_post - log_post_sum)
    label = np.argmax(post_norm, axis=1)
    count = np.sum(post_norm, axis=0)
    l = 1e-7
    for k in range(classnum):
        weights[k] = count[k] / pixels
        means[k, :] = post_norm[:, k] @ bneck / count[k]
        dev = (bneck - means[k, :])
        resp = post_norm[:, k][:, None]
        covs[k, :, :] = dev.T @ (resp * dev) / count[k] + l * np.eye(dims)
    return (label, log_like, weights, means, covs)


def kmc(means, bneck, classes):
    dist_list = []
    for k in range(classes):
        diff = bneck - means[k, :][None, :]
        dist_list.append(np.linalg.norm(diff, axis=1, keepdims=True))
    dist_array = np.hstack(dist_list)
    label = np.argmin(dist_array, axis=1)
    labels, counts = np.unique(label, return_counts=True)
    new_mean = np.zeros((means.shape[0], means.shape[1]))
    for k in range(classes):
        new_mean[k, :] = np.sum(bneck[label == k, :], axis=0)[None, :]
    new_mean = new_mean / counts[:, None]
    return (label, new_mean)


def run_clustering(bottleneck, label_name):
    np.random.seed(42)
    n, d = bottleneck.shape
    w_pi_init = np.random.random(gt_classnum)
    w_pi_init /= w_pi_init.sum()
    bott = bottleneck.copy()
    mu_init = []
    ri = np.random.randint(1, n)
    mu_init.append(bott[ri, :])
    bott = np.delete(bott, ri, 0)
    for i in range(1, gt_classnum):
        dist = np.vstack([np.linalg.norm(bott - m, axis=1) for m in mu_init])
        min_dist = np.min(dist, axis=0)
        new_mu_ind = np.argmax(min_dist)
        mu_init.append(bott[new_mu_ind])
        bott = np.delete(bott, new_mu_ind, 0)
    mu_init = np.asarray(mu_init)
    mu_init_copy = mu_init.copy()

    Xb = bottleneck - bottleneck.mean(axis=0)
    cov_1 = Xb.T @ Xb / n
    cov_init = np.stack([cov_1] * gt_classnum, axis=0)

    assign_gmm, LL_old, w_pi, mu_gmm, covariance = expect_max(w_pi_init, mu_init, cov_init, bottleneck, gt_classnum)
    LL_new = expect_max(w_pi, mu_gmm, covariance, bottleneck, gt_classnum)[1]
    while np.abs(LL_new - LL_old) > 1e-3:
        LL_old = LL_new
        assign_gmm, LL_new, w_pi, mu_gmm, covariance = expect_max(w_pi, mu_gmm, covariance, bottleneck, gt_classnum)

    assign_kmc, mu_kmc = kmc(mu_init_copy, bottleneck, gt_classnum)
    for i in range(10):
        assign_kmc, mu_kmc = kmc(mu_kmc, bottleneck, gt_classnum)

    conf_kmc = np.zeros((gt_classnum, gt_classnum))
    conf_gmm = np.zeros((gt_classnum, gt_classnum))
    for p in range(n):
        conf_kmc[recoded[p], assign_kmc[p]] += 1
        conf_gmm[recoded[p], assign_gmm[p]] += 1

    row_ind_kmc, col_ind_kmc = linear_sum_assignment(-conf_kmc)
    row_ind_gmm, col_ind_gmm = linear_sum_assignment(-conf_gmm)

    acc_kmc = conf_kmc[row_ind_kmc, col_ind_kmc].sum() / n
    acc_gmm = conf_gmm[row_ind_gmm, col_ind_gmm].sum() / n

    return acc_kmc, acc_gmm


results = []

d1 = np.load('outputs/bott_output_masked.npz')
results.append(('Vanilla AE (tanh bottleneck)', *run_clustering(d1['bottleneck'], 'vanilla')))

d2 = np.load('outputs/bott_output_beta1_masked.npz')
results.append(('VAE mean (beta=1)', *run_clustering(d2['mean'], 'vae_mean')))
results.append(('VAE sampled bottleneck (beta=1)', *run_clustering(d2['bottleneck'], 'vae_sampled')))

corr = data_z_reshaped.T @ data_z_reshaped / num_pixels
eig_val, eig_vec = np.linalg.eigh(corr)
cumul = np.cumsum(eig_val[::-1]) / np.sum(eig_val)
pca_num = np.linspace(1, num_bands, num_bands)
kl_knee = kn.KneeLocator(pca_num, cumul, curve="concave", direction="increasing")
bott_dim = int(kl_knee.knee)
top_eig_vec = eig_vec[:, -bott_dim:]
pca_bottleneck = data_z_reshaped @ top_eig_vec
results.append(('PCA (no autoencoder)', *run_clustering(pca_bottleneck, 'pca')))

labels, counts = np.unique(recoded, return_counts=True)
majority_baseline = counts.max() / num_pixels

lines = []
lines.append(f"{'Representation':<35} {'KMC acc':>10} {'GMM acc':>10}")
for name, acc_kmc, acc_gmm in results:
    lines.append(f"{name:<35} {acc_kmc:>10.4f} {acc_gmm:>10.4f}")
lines.append(f"{'Majority-class baseline':<35} {majority_baseline:>10.4f}")

output_text = "\n".join(lines)
print(output_text)

with open('outputs/cluster_accuracy_comparison.txt', 'w') as f:
    f.write(output_text + "\n")
