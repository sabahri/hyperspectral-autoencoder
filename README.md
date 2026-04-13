Creating an autoencoder from scratch for hyperspectral remote sensing

# Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Script Summary](#3-script-summary)
4. [Future Work](#4-future-work)

# 1. Project Overview
This is a basic autoencoder with 5 hidden layers to analyze the SalinasA dataset. It comprises 83 x 86 pixel "images" aquired across 204 bands within the visible / IR range, downloaded here:
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

My aim was to build a self-supervised autoencoder for remote sensing, and to explore the variables that are important in general to decoding hyperspectral data in Earth Observation / Remote Sensing.

# 2. Architecture
The autoencoder breaks down the z-normalized input data from 204 to 10 dimensions (input -> decoder -> bottleneck) and then assesses the quality of the learned bottleneck layer by reconstructing the original data (bottleneck -> encoder -> output). Each hidden layer was ReLU-activated. The weights and biases were optimized via batch gradient descent, minimizing the cost based on mean-squared error between the input and output layers.

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/hyperspectral_autoencoder_nn_diagram_v8.svg" />

# 3. Script Summary

The input includes wide ranges of values, so minimizing the MSE cost on the raw data would give disproportionate weight to bands with high variance. I therefore chose to first z-score the data in order to normalize the per-band standard deviation to 1, causing the loss to now describe the reconstruction error in units of standard deviation.

Finally, postprocessing shows the following:

(1) The per-pixel loss, indicating ambiguous areas in the SalinasA scene for the network to interpret

<img width="640" height="480" alt="Figure1" src=https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/perpixel_loss_plasma_orig.png/>

(2) UMAP projection of the bottleneck layer colored by ground truth, showing that the network is able to preserve clustering information regarding detecting pixel types in the data, largely aligning with crop type and growth stage.

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/UMAP.png" />

(3) Finally, I used both K-means-based and Gaussian Mixture Model-based clustering to see whether the bottleneck accurately preserves geospectral signatures according to crop class. Here, it is again colored according to ground truth. The K-means and GMM clustering were calibrated to the ground truth according the Hungarian Algorithm to ensure that the color-labels correspond to the same classes.

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/remapped_4.png">

This bottleneck analysis shows that even though the per-pixel-loss condition shown above is achieved effectively, MSE loss did not ensure that the path the network chose through the given node constraints is particularly efficient. In fact, per-dimension activation histograms show that roughly 30% bottleneck vectors are carrying most of the structural information (dimensions 3, 5, 7) while about 20% are pushed towards the asymptotic edges relative to the tanh activation for the bottleneck layer (dims 1, 8), and the majority are soft-saturated somewhere in between:

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/histograms.png">

