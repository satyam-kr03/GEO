# Generator-Enhanced Optimization

Demo implementation of the GEO strategy: A framework to solve optimization problems using generative models for efficient exploration of the large discrete search space.

Based on the paper titled [_GEO: Enhancing Combinatorial Optimization with Classical and Quantum Generative Models_ ](https://arxiv.org/abs/2101.06250) by Javier Alcazar, Mohammad Ghazi Vakili, Can B. Kalayci, and Alejandro Perdomo-Ortiz (2022).

![image](https://github.com/user-attachments/assets/8b645b34-8dab-4e6c-b946-0a89b449b847)

This work is part of the Seminar on Advanced Topics in Quantum Computing (WS 2024/25).

### Overview

The repository contains three different implementations of the algorithm, differing in the generative modeling core. The models used are: a classical Generative Adversarial Network (GAN), a classical Variational Auto Encoder (VAE) and a quantum Generative Adversarial Network (qGAN).

### Dataset

We use the latest stock prices dataset for portfolio optimization obtained from SimFin after processing it. The script for downloading the dataset can be found in /data/download.ipynb

### Methodology

#### Objective Function

The objective function calculates the objective values for a set of portfolio allocations by balancing the trade-off between risk and return, using a specified parameter (we use lambda\_ = 0.5). It calculates the risk as the quadratic form of the covariance matrix and the returns as the dot product of the mean returns and portfolio weights. The function returns the objective values as a PyTorch tensor.

#### Seed Dataset

The initial seed dataset is generated randomly using numpy. We keep updating it in each iteration of the algorithm.

#### Algorithm

The generic GEO algorithm broadly consists of three steps:

1. Initialize and Evaluate: Start with a random prior to generate initial solution candidates and evaluate them using the cost function.

2. Select and Generate: Select the best candidates based on the cost values, then use a generative model to propose new potential candidates with a lower cost value.

3. Update and Iterate: Update the generative model with the new merged seed dataset and repeat the process until a desired minimum value is found or a termination criterion is met.

#### Results

![alt text](/media/result_GAN.png)

Fig. 1: Portfolio optimization using a classical GAN over 200 iterations of the GEO algorithm.
