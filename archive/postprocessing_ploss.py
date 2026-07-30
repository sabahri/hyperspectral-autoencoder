import scipy.io
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

##### Loading data
data = loadmat('SalinasA_corrected.mat')['salinasA_corrected']

# SalinasA: 83 x 86 spatial grid
# 204 channels
num_pixels = data.shape[0] * data.shape[1]
num_bands = data.shape[-1]

# z-scoring
# Normalize input so that the pixels of each band are centered on 0, with stdev = 1
data_z = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
epsilon = 10**-6

for j in range(data.shape[-1]):
	mean = np.mean(data[:,:,j])
	std = np.std(data[:,:,j])
	data_z[:,:,j] = (data[:,:,j] - mean) / (std + epsilon)

data_z_reshaped = data_z.reshape(num_pixels, num_bands)

# Band 100, normalized to [0,1] for display
band_ind = 100
band = data[:,:,band_ind]
band_normalized = (band - band.min()) / (band.max() - band.min())

##### Per-pixel loss + Band 100, for each checkpoint

checkpoints = {
	"orig": "bott_output.npz",
	"beta1": "bott_output_beta1.npz",
	"beta0.1": "bott_output_beta0.1.npz",
}

for label, filename in checkpoints.items():
	output = np.load(filename)['output']

	p_loss = np.mean((output - data_z_reshaped)**2, axis=1)
	ploss_h = np.percentile(p_loss, 99)
	ploss_l = np.percentile(p_loss, 1)

	# Clipping highest and lowest value pixels
	p_loss = np.clip(p_loss, ploss_l, ploss_h)
	p_loss = (p_loss - p_loss.min()) / (p_loss.max() - p_loss.min())
	p_loss = p_loss.reshape(data.shape[0], data.shape[1])

	fig, (ax_band, ax_loss) = plt.subplots(1, 2, figsize=(10, 5))

	ax_band.imshow(band_normalized, cmap='gray')
	ax_band.set_title('Band 100')

	im = ax_loss.imshow(p_loss, cmap='plasma', vmin=0, vmax=1)
	ax_loss.set_title('Per-Pixel Loss')
	fig.colorbar(im, ax=ax_loss, label='Per-Pixel Loss', fraction=0.046, pad=0.04)

	plt.tight_layout()
	plt.savefig(f'images/perpixel_loss_plasma_{label}.png')
	plt.show()
