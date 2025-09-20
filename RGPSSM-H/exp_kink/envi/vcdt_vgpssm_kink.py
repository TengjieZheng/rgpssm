from typing import Tuple, Optional, List, Union
from types import SimpleNamespace

import matplotlib.pyplot as plt
from torch import Tensor
from numpy import ndarray

import numpy as np
import os
import torch
from tqdm import tqdm
import math

from .models.EnVI import GPSSMs
from .models.GPSSMs import VCDT, vGPSSM

class VCDT_vGPSSM_Kink():
    def __init__(self, name, var_noise, ips, device='cpu'):
        self.var_noise = var_noise
        self.std_noise = var_noise**0.5
        process_noise_sd = 0.05
        self.device = device

        self.state_dim = 1
        self.output_dim = 1
        self.input_dim = 0

        self.episode = 30
        self.seq_len = 20
        self.lr = 0.003
        self.num_epoch = 15000              # number of epochs
        self.residual_trans = True

        if name == 'vcdt':
            self.model = VCDT(state_dim=self.state_dim, output_dim=self.output_dim, seq_len=self.seq_len, inducing_points=ips,
                              input_dim=self.input_dim, N_MC=50, emission_noise_sd=self.std_noise, residual_trans=self.residual_trans).to(device)
        elif name == 'vgpssm':
            self.model = vGPSSM(state_dim=self.state_dim, output_dim=self.output_dim, seq_len=self.seq_len,
                       inducing_points=ips, input_dim=self.input_dim, emission_noise_sd=self.std_noise, residual_trans=self.residual_trans).to(device)
        else:
            raise ValueError('Model name is not correct')

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
            ELBO = self.model(observe)
            loss = -ELBO
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epochiter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})

        fig = plt.figure()
        plt.plot(losses)


    def gp_pred(self, x_test: Tensor, condition_u=True):
        x_test = x_test.to(self.device).float()
        test_xx = x_test.reshape(-1, 1)
        test_xx = test_xx.repeat(self.model.state_dim, 1, 1)

        func_pred = self.model.transition(test_xx)          # MultivariateNormal, shape: state_dim x batch_size
        observed_pred = self.model.likelihood(func_pred)    # MultivariateNormal

        # Mean
        assert (self.model.state_dim == 1)
        pred_val_mean = observed_pred.mean.mean(dim=[0])   # state_dim x batch_size (only for state_dim = 1 case)
        if self.residual_trans:
            pred_val_mean = pred_val_mean + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])

        # Upper and lower confidence bounds: shape: state_dim x batch_size
        lower, upper = observed_pred.confidence_region()
        lower, upper = lower.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0]), \
                       upper.reshape(-1, ) + test_xx[0, :, :].transpose(-1, -2).mean(dim=[0])

        # Arange the output
        f_mean = pred_val_mean.detach().view(-1)
        upper = upper.detach().view(-1)
        f_std = (upper - f_mean) / 2

        return f_mean, f_std


    def inducing_points(self):
        ips = self.model.transition.variational_strategy.inducing_points
        U = self.model.transition(ips).mean
        if self.residual_trans:
            U = U + ips[0, :, :].transpose(-1, -2)

        inducing_in = ips.detach().view(-1)
        inducing_out = U.mean(dim=[0]).detach().view(-1)

        return inducing_in, inducing_out