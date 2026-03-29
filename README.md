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

<img width="640" height="480" alt="Figure1" src=https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/perpixel_loss.png/>

(2) UMAP projection of the bottleneck layer colored by ground truth, showing that the network is able to preserve clustering information regarding detecting pixel types in the data, largely aligning with crop type and growth stage.

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/UMAP.png" />