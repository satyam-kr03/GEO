import torch
import pandas as pd
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prices = pd.read_csv("../data/prices.csv", index_col=0)

prices = prices.iloc[:, :5] # use only the first 5 assets

assets = list(prices.columns) # the assets in the portfolio
num_assets = len(assets)

mean = torch.Tensor(np.array(prices.pct_change().mean() * 252)).to(device)
covariance = torch.Tensor(np.array(prices.pct_change().cov() * 252)).to(device)
# pct_change() computes the percentage change between the current and a prior element

def objective(portfolio_allocations):
    lambda_ = 0.5 # return-risk trade-off parameter

    total_objective = []
    for portfolio in portfolio_allocations:
        global mean, covariance # the mean and covariance matrix of the returns of the assets

        risk = 0
        for i in range(portfolio_allocations.shape[1]): # for each asset
            for j in range(portfolio_allocations.shape[1]): # for each asset
                risk += covariance[i][j] * portfolio[i] * portfolio[j] # calculate the risk of the portfolio

        returns = 0
        for i in range(portfolio_allocations.shape[1]): # for each asset
            returns = returns + mean[i] * portfolio[i] # calculate the return of the portfolio

        objective_value = lambda_ * risk - (1 - lambda_) * returns
        total_objective.append(objective_value)

    return torch.Tensor(total_objective).to(device)

