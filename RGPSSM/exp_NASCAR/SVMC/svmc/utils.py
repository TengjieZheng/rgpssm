import torch


def compute_ess(weights):
    return 1 / torch.sum(weights ** 2, -1)


def compute_mse(X_true, X_particles, w_norm):
    X_est = torch.sum(X_particles * w_norm.unsqueeze(2), 1)
    return torch.mean((X_true - X_est) ** 2, -1)


def compute_elbo(weights):
    w_mean = torch.mean(weights, -1)
    elbo = torch.log(torch.cumprod(w_mean, 0))
    return elbo / torch.arange(1, w_mean.shape[0] + 1)