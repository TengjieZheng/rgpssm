from typing import Tuple, Optional, List, Union
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np

from data.utils import Torch2Np
from model.rgpssm.utils import ToTensor
from model.rgpssm_h.rgpssm_h import RGPSSM_H
from model.rgpssm_h.utils import IModel_H
from model.rgpssm_h.kernel import RBFKerH

class FunH(IModel_H):
    def __init__(self):
        pass

    def fun_tran(self, x: Tensor, c: Tensor, f: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        return f, None, None

    def fun_meas(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return x, None

    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        return x.unsqueeze(-2), None


class RGPSSM_Kink():
    def __init__(self, filter, var_noise):
        self.var_noise = var_noise
        self.std_noise = var_noise**0.5
        self.std_process = 0.05

        x0 = torch.zeros((1, 1))                        # Initial state mean
        P0 = torch.eye(1) * 1                           # Initial state covariance
        Q = torch.eye(1) * self.std_process**2 + 0.3    # Process noise covariance
        R = torch.eye(1) * self.std_noise**2            # Measurement noise covariance

        self.ker = RBFKerH(df=[1], dz=[1], std=[[3.]], ls=[[1.]], jitter=1e-4)
        self.gpssm = RGPSSM_H(x0, P0, Q, R, FunH(), self.ker, flag_chol=True, type_filter=filter.upper(),
                              budget=15, eps_tol=5e-4, num_opt_hp=1, lr_hp=5e-3)

    def train(self, y):
        y = y.ravel()
        for i in range(y.size):
            F, var_F, f, var_f = self.gpssm.predict()
            self.gpssm.correct(y[i])
            if i > 50:
                self.gpssm.hyperparam_opt()

            if i % 100 == 0 and i > 100:
                self.gpssm.prune_redundant_points()

            if i % 100 == 0:
                print(f'i = {i}, num_id = {self.gpssm.num_ips}')
                print(f'ls = {Torch2Np(self.ker.ls[0])}, var = {Torch2Np(self.ker.var[0])}')

    def gp_pred(self, x_test: Tensor):

        f_mean, f_var = self.gpssm.GP_predict(x_test.view(-1, 1))
        f_std = torch.diag(f_var)**0.5

        return f_mean, f_std

    def inducing_points(self):
        inducing_in = self.gpssm.Z
        inducing_out = self.gpssm.m

        return inducing_in.view(-1), inducing_out.view(-1)



