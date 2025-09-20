from typing import Tuple, Optional, List, Union
from types import SimpleNamespace

import numpy as np
from torch import Tensor
from numpy import ndarray

import torch
import torch.nn as nn
from tqdm import tqdm

from .svmc.base import KnowledgeModel
from .my_svmc_gp import My_SVMC_GP
from .utils import ToTensor


class KnoModelKink(KnowledgeModel):
    def __init__(self):
        d_in, d_out = 1, 1
        super().__init__(d_in, d_out)

    def fun_tran(self, x, u=None, f=None):
        """State transition function
        Args:
            x : system state
            u : system input
            f : GP prediction
        Returns:
            x_pre : state prediction
            Af : partial derivative dx_pre / df
        """
        x = torch.as_tensor(x).reshape(-1, 1)
        if u is not None:
            u = torch.as_tensor(u).reshape(-1, 1)
        if f is not None:
            f = torch.as_tensor(f).reshape(-1, 1)

        x_pre = 1 * x + f
        Af = torch.eye(1)

        return x_pre, Af

    def fun_input(self, x, u=None):
        """Input function
        Args:
            x : system state
            u : system input
        Returns:
            z : GP input
        """
        x = torch.as_tensor(x).reshape(-1, 1)
        if u is not None:
            u = torch.as_tensor(u).reshape(-1, 1)
        xu = x

        return xu.reshape(1, -1)


class SVMC_Kink():
    def __init__(self, var_noise):
        torch.set_default_dtype(torch.double)
        dx = 1
        dy = 1
        varx = 0.05**2 + 0.5
        vary = var_noise
        P = torch.eye(1) * 4

        u_inducing = np.linspace(-3, 1, 15).reshape(-1, 1)
        KnoModel = KnoModelKink()
        self.model = My_SVMC_GP(dx, dy, varx, vary, P, u_inducing, KnoModel, fvar=9, length_scale=1,
                 lr=1e-2, d_hidden_mlp=16, n_pf=50, n_opt=3, iter_opt=5)

    def train(self, y):
        y = y.ravel()
        train_gp = False
        for i in range(y.size):
            if i > 50:
                train_gp = True
            self.model.update(y[[i]], train_gp)

            if i % 100 == 0:
                print(f'i = {i}, ESS = {self.model.ESS}')

    def gp_pred(self, x_test: Tensor):
        f_mean, f_std = [], []
        for i in range(x_test.numel()):
            f_mean_i, f_var_i = self.model.GP_predict(x_test.view(-1)[i])
            f_mean.append(f_mean_i.view(-1).item())
            f_std.append(f_var_i.view(-1).item()**0.5)
        f_mean = ToTensor(f_mean).view(-1) + x_test.view(-1) * 1
        f_std = ToTensor(f_std).view(-1)

        return f_mean, f_std

    def inducing_points(self):
        qmean = 0
        for n in range(self.model.n_pf):
            qmean += self.model.w[n] * self.model.particles[2][n].qz.mean

        inducing_in = ToTensor(self.model.u_inducing).view(-1)
        inducing_out = ToTensor(qmean).view(-1) + inducing_in.view(-1) * 1

        return inducing_in, inducing_out