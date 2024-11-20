import numpy as np
import torch


def gaussian_loss(x, mu, logvar):
    """Negative diagonal Gaussian log-likelihood
    :param x:
    :param mu:
    :param logvar:
    :return:
    """
    var = torch.exp(logvar)
    loss = 0.5 * torch.sum((x - mu) ** 2 / var + logvar + torch.log(2 * torch.tensor(np.pi)), dim=-1)
    return loss
