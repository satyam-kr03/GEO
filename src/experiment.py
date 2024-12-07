import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm

from helpers import multiDistribution, generate_random_noise
from problem_objective import objective
from models import Generator, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# intital seed dataset
initial_seed = torch.from_numpy(np.array(pd.read_csv("initial_seed.csv", index_col=0))).to(device)

sample_dim = 5 # number of assets
batch_size = 256
num_epochs = 200
learning_rate = 0.0002

generator = Generator(sample_dim, sample_dim).to(device)
discriminator = Discriminator(sample_dim).to(device)

criterion = nn.BCELoss() # Binary Cross Entropy Loss
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

distri = multiDistribution(initial_seed)

pbar = tqdm(range(num_epochs)) # progress bar for visualization 

new_samples = np.array([]) # new generated samples
initial_seed_cost = objective(initial_seed)
ground_val = initial_seed_cost.min().item()

# Initialize a list to store the cost values
cost_values = []

for epoch in pbar:
    if new_samples.any(): # if new generated samples are already present
        initial_seed = torch.cat((initial_seed, new_samples),dim=0) # concatenate new generated samples with initial seed
        initial_seed_cost = objective(initial_seed) # calculate the objective value of the new seed
        sort_ind = torch.sort(initial_seed_cost,dim=0)[1] # sort the new seed
        initial_seed = initial_seed[sort_ind] # sort the new seed
        initial_seed = initial_seed[:5000] # keep only the top 5000 samples
        ground_val = initial_seed_cost.min().item() # update the ground value
        distri = multiDistribution(initial_seed) # update the distribution of the new seed
        new_samples = np.array([]) # reset the new generated samples to empty 

    mean, std = torch.mean(initial_seed_cost), torch.std(initial_seed_cost) # calculate the mean and std of the new seed

    for batch_idx in range(10000 // batch_size): # batch size is 256

        # Train the Discriminator
        discriminator.zero_grad() 

        real_samples = distri.sample(sample_shape = [batch_size]).to(device).float() # sample from the distribution 
        real_samples /= real_samples.sum(dim=1, keepdim=True) # normalize the real samples
        real_labels = torch.ones(batch_size, 1).to(device) # real labels are 1

        fake_samples = generator(generate_random_noise(batch_size, 5)).detach() # generate fake samples
        fake_labels = torch.zeros(batch_size, 1).to(device) # fake labels are 0

        d_real_output = discriminator(real_samples) 
        d_real_loss = criterion(d_real_output, real_labels)

        d_fake_output = discriminator(fake_samples)
        d_fake_loss = criterion(d_fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        discriminator_optimizer.step()

        # Train the Generator
        generator.zero_grad()

        generated_samples = generator(generate_random_noise(batch_size, 5))
        g_fake_output = discriminator(generated_samples)
        g_loss = criterion(g_fake_output, real_labels)

        g_loss.backward()
        generator_optimizer.step()

    new_samples = generator(generate_random_noise(1000, 5)).detach() # generate new samples
    new_samples /= new_samples.sum(dim=1, keepdim=True) # normalize the new samples
    pbar.set_description(f"{round(g_loss.item(),2)} :: {round(d_loss.item(),2)} :: {ground_val} :: {mean.item(), std.item()}")

    # Append the cost value (ground_val) to the list
    cost_values.append(ground_val)

# Plot the cost values vs number of epochs
plt.plot(cost_values)
plt.xlabel('Epoch')
plt.ylabel('Cost Value')
plt.title('Cost Value vs Number of Epochs')
plt.show()