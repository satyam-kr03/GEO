from models import QuantumGenerator, ClassicalDiscriminator
import torch.nn as nn
from helpers import multiDistribution, random_noise
from problem_objective import objective
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# intital seed dataset
initial_seed = torch.from_numpy(np.array(pd.read_csv("../data/initial_seed.csv", index_col=0))).to(device)

sample_dim = 5 # number of assets
batch_size = 64
num_epochs = 200
learning_rate = 0.0002

xgenerator = QuantumGenerator(int(np.log2(16)))
generator = xgenerator.generator.to(device)
discriminator = ClassicalDiscriminator().to(device)

criterion = nn.BCELoss() 
generator_optimizer = Adam(generator.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005)
discriminator_optimizer = Adam(discriminator.parameters(), lr=0.01, betas=(0.7, 0.999), weight_decay=0.005)

distri = multiDistribution(initial_seed)

pbar = tqdm(range(num_epochs)) # progress bar for visualization 

new_samples = np.array([]) # new generated samples
initial_seed_cost = objective(initial_seed)
ground_val = initial_seed_cost.min().item()

# Initialize a list to store the cost values
cost_values = []
generator_loss_values = []
discriminator_loss_values = []

def make_positive_definite(matrix):
    # Compute the covariance matrix
    cov_matrix = torch.cov(matrix.T)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    # Ensure all eigenvalues are positive by applying a small threshold
    epsilon = 1e-6
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    # Reconstruct the positive definite covariance matrix
    stabilized_cov = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return stabilized_cov

import matplotlib.pyplot as plt

# Set up interactive mode for live plotting
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
cost_values = []  # Initialize list to store cost values
line, = ax.plot([], [], label='Objective Value')  # Create an empty line
ax.set_xlabel('Epoch')
ax.set_ylabel('Objective Value')
ax.set_title('Objective Value vs Number of Epochs')
ax.legend()

for epoch in pbar:

    if new_samples.any():
        initial_seed = torch.cat((initial_seed, new_samples), dim=0)
        initial_seed_cost = objective(initial_seed)

        sort_ind = torch.sort(initial_seed_cost, dim=0)[1]
        initial_seed = initial_seed[sort_ind]
        initial_seed = initial_seed[:5000]

        ground_val = initial_seed_cost.min().item()
        distri = multiDistribution(make_positive_definite(initial_seed))
        new_samples = np.array([])

    mean, std = torch.mean(initial_seed_cost), torch.std(initial_seed_cost)

    for batch_idx in range(10000 // batch_size):
        discriminator_optimizer.zero_grad()

        real_samples = distri.sample(sample_shape=[batch_size]).to(device).float()
        real_samples /= real_samples.sum(dim=1, keepdim=True)
        real_labels = torch.ones(batch_size, 1).to(device)

        fake_samples = xgenerator.generate_samples(64)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        d_real_output = discriminator(real_samples)
        d_real_loss = criterion(d_real_output, real_labels)

        d_fake_output = discriminator(fake_samples)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        discriminator_optimizer.step()
        discriminator_loss_values.append(d_loss.item())

        generator_optimizer.zero_grad()

        generated_samples = xgenerator.generate_samples(64)
        g_fake_output = discriminator(generated_samples)
        g_loss = criterion(g_fake_output, real_labels)
        g_loss.backward()
        generator_optimizer.step()
        generator_loss_values.append(g_loss.item())

    new_samples = xgenerator.generate_samples(1000).detach()
    new_samples /= new_samples.sum(dim=1, keepdim=True)
    pbar.set_description(f"{round(g_loss.item(),2)} :: {round(d_loss.item(),2)} :: {ground_val} :: {mean.item(), std.item()}")

    cost_values.append(ground_val)

    # Update live plot
    line.set_xdata(range(len(cost_values)))
    line.set_ydata(cost_values)
    ax.relim()  # Recompute the axes limits
    ax.autoscale_view()  # Rescale the view to fit data
    plt.draw()  # Redraw the plot
    plt.pause(0.1)  # Pause briefly to update the plot

    # Save plot as PNG
    plt.savefig(f'qGAN.png')

    print(ground_val)

# Turn off interactive mode
plt.ioff()

# Final plot display
plt.show()
