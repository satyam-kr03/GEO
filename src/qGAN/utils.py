import torch
from torch.distributions.multivariate_normal import MultivariateNormal

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_positive_definite(matrix):
    cov_matrix = torch.cov(matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    # Ensure all eigenvalues are positive by applying a small threshold
    epsilon = 1e-6
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    # Reconstruct the positive definite covariance matrix
    stabilized_cov = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
    return stabilized_cov

def multiDistribution(matrix):
    mean = torch.mean(matrix, dim=0)
    covariance = make_positive_definite(matrix)
    # Create multivariate normal distribution
    distribution = MultivariateNormal(loc=mean, covariance_matrix=covariance)
    return distribution

def generate_random_noise(batch_size, sample_dim):
    return torch.randn(batch_size, sample_dim).to(device)
