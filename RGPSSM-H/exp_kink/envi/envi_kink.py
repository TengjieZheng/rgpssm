from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import numpy as np
import os
import torch
from tqdm import tqdm
import math

from .models.EnVI import GPSSMs

class EnVI_Kink():
    def __init__(self, var_noise, ips, device='cpu'):
        self.var_noise = var_noise
        self.std_noise = var_noise**0.5
        self.device = device


        self.state_dim = 1
        self.output_dim = 1
        self.input_dim = 0
        indices = [i for i in range(self.output_dim)]
        self.H_true = torch.eye(self.state_dim, device=device, dtype=torch.float)[indices]

        self.episode = 30
        self.seq_len = 20
        self.batch_size = self.episode      # full batch training
        self.lr = 0.01
        self.num_epoch = 1000               # number of epochs

        process_noise_sd = 0.05
        self.model = GPSSMs(dim_x=self.state_dim, dim_y=self.output_dim, seq_len=self.seq_len,
                       ips=ips, dim_c=0, N_MC=100,
                       process_noise_sd=process_noise_sd,
                       emission_noise_sd=self.std_noise,
                       consistentSampling=True).to(device)
        self.model.emission_likelihood.requires_grad_(False)

    def train(self, y):
        Y = y.reshape([self.episode, self.seq_len, self.output_dim])
        observe = torch.tensor(Y, dtype=torch.float).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        start_epoch = 0
        losses = []

        self.model.train()
        epochiter = tqdm(range(start_epoch, start_epoch + self.num_epoch), desc='Epoch:')
        for epoch in epochiter:
            self.model.train()
            optimizer.zero_grad()
            ELBO, X_filter = self.model(observe, self.H_true)
            loss = -ELBO
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

    def gp_pred(self, x_test: Tensor, condition_u=True):
        x_test = x_test.to(self.device).float()
        test_xx = x_test.reshape(-1, 1)
        if condition_u:
            # shape [state_dim x num_ips x (state_dim + input_dim)]
            ips = self.model.transition.variational_strategy.inducing_points

            # shape: state_dim x num_ips
            U1 = self.model.transition(ips).mean + ips[0,:,:].transpose(-1,-2)
            U = U1.mean(dim=[0])

            # x expected shape: state_dim x batch_size x (input_dim + state_dim)
            test_xx = test_xx.repeat(self.model.state_dim, 1, 1)

            # MultivariateNormal, shape: state_dim x batch_size
            func_pred = self.model.transition(test_xx)

            # MultivariateNormal
            observed_pred = self.model.likelihood(func_pred)
            # observed_pred = func_pred

            # shape: state_dim x batch_size (only for state_dim = 1 case)
            assert (self.model.state_dim == 1)
            pred_val_mean = observed_pred.mean.mean(dim=[0]) + test_xx[0,:,:].transpose(-1,-2).mean(dim=[0])

            # Get upper and lower confidence bounds
            # lower & upper,  shape: state_dim x batch_size
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                           upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])


        else:
            # x expected shape: state_dim x batch_size x (input_dim + state_dim)
            test_xx = test_xx.repeat(self.model.state_dim, 1, 1)

            func_pred = self.model.transition(test_xx)             # shape: state_dim x batch_size
            observed_pred = self.model.likelihood(func_pred)       # shape: state_dim x batch_size
            pred_val_mean = observed_pred.mean + test_xx[0, :, :].transpose(-1, -2)

            ips = self.model.transition.variational_strategy.inducing_points
            U = self.model.transition(ips).mean + ips[0, :, :].transpose(-1, -2)

            # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                           upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])


        f_mean = pred_val_mean.detach().view(-1)
        upper = upper.detach().view(-1)
        f_std = (upper - f_mean) / 2


        return f_mean, f_std

    def inducing_points(self):
        ips = self.model.transition.variational_strategy.inducing_points
        U1 = self.model.transition(ips).mean + ips[0, :, :].transpose(-1, -2)
        U = U1.mean(dim=[0])

        inducing_in = ips.detach().view(-1)
        inducing_out = U.detach().view(-1)

        return inducing_in, inducing_out