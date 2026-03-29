Creating an autencoder from scratch for hyperspectral remote sensing

# Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Script Summary](#3-script-summary)
4. [Future Work](#4-future-work)

# 1. Project Overview
This is a basic autoencoder with 5 hidden layers to analyze the SalinasA dataset. It comprises 83 x 86 pixel "images" aquired across 204 bands within the visible / IR range, downloaded here:
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

My aim was to build a self-supervised autoencoder for remote sensing, and to explore the variables that are important in general to decoding images in Earth Observation / Remote Sensing.

# 2. Architecture

<img width="640" height="480" alt="Figure1" src="https://github.com/sabahri/hyperspectral-autoencoder/blob/main/images/hyperspectral_autoencoder_nn_diagram_v8.svg" />

# 3. Script Summary
The autoencoder breaks down the data from 204 to 10 dimensions (input data -> decoder -> bottleneck) and then assesses the quality of the learned bottleneck layer by reconstructing the original data (bottleneck -> encoder -> output layer). The weights and biases were optimized via batch gradient descent, minimizing the mean-squared error between the input and output layers.