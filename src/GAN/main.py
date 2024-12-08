import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import multiDistribution, random_noise
from problem import objective
from models import Generator, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from IPython.display import clear_output

def plot_training_progress(generator_loss_values, discriminator_loss_values, objective_values):
    if len(generator_loss_values) < 2:
        return
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

    # Plot Generator and Discriminator Loss
    ax1.set_title("Loss")
    ax1.plot(generator_loss_values, label="Generator Loss", color="blue")
    ax1.plot(discriminator_loss_values, label="Discriminator Loss", color="red")
    ax1.legend(loc='best')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.grid()

    # Plot Objective Values
    ax2.set_title("Objective Values")
    ax2.plot(objective_values, label="Objective Value", color="green")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Objective Value")
    ax2.grid()
    plt.show()
    plt.savefig('progress.png')
    plt.close()

# intital seed dataset
initial_seed = torch.from_numpy(np.array(pd.read_csv("../data/initial_seed.csv", index_col=0))).to(device)

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
generator_loss_values = []
discriminator_loss_values = []
avg_generator_loss_values = []
avg_discriminator_loss_values = []

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

        fake_samples = generator(random_noise(batch_size, 5)).detach() # generate fake samples
        fake_labels = torch.zeros(batch_size, 1).to(device) # fake labels are 0
        d_real_output = discriminator(real_samples) 
        d_real_loss = criterion(d_real_output, real_labels)

        d_fake_output = discriminator(fake_samples)
        d_fake_loss = criterion(d_fake_output, fake_labels)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()

        discriminator_optimizer.step()
        discriminator_loss_values.append(d_loss.item())

        # Train the Generator
        generator.zero_grad()
        generated_samples = generator(random_noise(batch_size, 5))
        g_fake_output = discriminator(generated_samples)
        g_loss = criterion(g_fake_output, real_labels)
        g_loss.backward()

        generator_optimizer.step()
        generator_loss_values.append(g_loss.item())

    new_samples = generator(random_noise(1000, 5)).detach() # generate new samples
    new_samples /= new_samples.sum(dim=1, keepdim=True) # normalize the new samples

    pbar.set_description(f"{round(g_loss.item(),2)} :: {round(d_loss.item(),2)} :: {ground_val} :: {mean.item(), std.item()}")

    avg_gen_loss = np.mean(generator_loss_values[-(10000 // batch_size):])
    avg_disc_loss = np.mean(discriminator_loss_values[-(10000 // batch_size):])

    # Append averaged losses to new lists for plotting
    avg_generator_loss_values.append(avg_gen_loss)
    avg_discriminator_loss_values.append(avg_disc_loss)

    plot_training_progress(avg_generator_loss_values, avg_discriminator_loss_values, cost_values)

    # Append the cost value (ground_val) to the list
    cost_values.append(ground_val)

# Plot the cost values vs number of epochs
plt.plot(cost_values)
plt.xlabel('Epoch')
plt.ylabel('Objective Value')
plt.title('Objective Value vs Number of Epochs')
plt.show()


