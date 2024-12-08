import torch
from torch.distributions.multivariate_normal import MultivariateNormal

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def multiDistribution(data_seed): # data_seed is the initial seed dataset
    mean = torch.mean(data_seed, axis=0)
    covariance = torch.cov(torch.permute(data_seed, (1, 0)))
    distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance)
    return distribution

def random_noise(batch_size, sample_dim): #  we use batch size 256 and sample_dim 5
    return torch.randn(batch_size, sample_dim).to(device)
