from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import torch.nn as nn
import numpy as np
import cholup, choldown

from ..rgpssm.utils import assemble_chol, nearest_positive_definite


class UKF_utils():

    @staticmethod
    def sample(x_mean, L, eta):
        """Sample x
        Args:
            x_mean: (dx, 1)
            L:      (dx, dx)
            eta:    float
        Returns:
            xu_s: (dx, 2*dx+1)
        """
        d = x_mean.numel()
        xu_s = x_mean.expand(d, 2 * d + 1).clone()
        xu_s[:, 1:d + 1] = xu_s[:, 1:d + 1] + eta * L
        xu_s[:, d + 1:] = xu_s[:, d + 1:] - eta * L

        return xu_s

    @staticmethod
    def pred_xw(xnew_s, w_s, w_mean, Wm, Wc, sqrt_Q, upper_x=True, Lw=None, eta=None):
        """Get x_mean and L_xw according to given the samples of the predicted x
           when w are not changed within the transition
        Args:
            xnew_s: samples of the predicted x (dx, n)
            w_s:    samples of the param w (dw, n)
            w_mean: mean of the param w (dw, 1)
            Wm: weight for the mean evaluation (n,)
            Wc: weight for the covariance evaluation (n,)
            sqrt_Q: the right cholesky factor of process noise covariance (dQ, dx+dw)
            upper_x: evaluate the cholesky factor of the predicted x-w when upper_x=True,
                     and evaluate the one of w-x when upper_x = False
            Lw： Cholesky factor of the covariance of w
            eta:    float
        Returns:
            x_mean: mean of the predicted x (dx, 1)
            L: cholesky factor of the covariance of the  predicted x-w (dx+dw, dx+dw)
        """

        # Mean
        x_mean = (xnew_s * Wm).sum(dim=-1).view(-1, 1)

        # Samples
        if upper_x:
            xnew_w_s = torch.cat((xnew_s, w_s), dim=0)
            xnew_w_m = torch.cat((x_mean, w_mean), dim=0)  # predicted mean
        else:
            xnew_w_s = torch.cat((w_s, xnew_s), dim=0)
            xnew_w_m = torch.cat((w_mean, x_mean), dim=0)  # predicted mean

        dx = x_mean.numel()
        if upper_x or Lw is None or eta is None or (not upper_x and (sqrt_Q[:-dx, :-dx] != 0).any()):
            # Covariance
            xnew_w_e = (Wc[1:]) ** 0.5 * (xnew_w_s[:, 1:] - xnew_w_m)  # Weighted difference
            right_factor = torch.cat((xnew_w_e.T, sqrt_Q), dim=0)
            _, R = torch.linalg.qr(right_factor)

            L = R.T
            f_chol = cholup.chol_update if Wc[0] >= 0 else choldown.chol_downdate
            L = f_chol(L, torch.abs(Wc[0]) ** 0.5 * (xnew_w_s[:, [0]] - xnew_w_m))
        else:
            dX = int((xnew_s.shape[1] - 1) / 2)
            dw = w_s.shape[0]
            e_x_s = (xnew_s - x_mean)               # (dx, n)
            We_x_s = e_x_s * Wc                     # (dx, n)
            tmp1 = We_x_s[:, 1:1+dw]                # (dx, dw)
            tmp2 = We_x_s[:, 1+dX:1+dX+dw]          # (dx, dw)
            low1 = eta * tmp1 - eta * tmp2          # (dx, dw)

            xnew_e = (Wc[1:]) ** 0.5 * (xnew_s[:, 1:] - x_mean)                 # (dx, n-1)
            tmp = torch.cat((xnew_e.T, sqrt_Q[-dx:, -dx:].T), dim=0)
            _, R = torch.linalg.qr(tmp)
            low2 = choldown.chol_downdate(R.T, low1)
            f_chol = cholup.chol_update if Wc[0] >= 0 else choldown.chol_downdate
            low2 = f_chol(low2, torch.abs(Wc[0]) ** 0.5 * (xnew_s[:, [0]] - x_mean))

            L = assemble_chol([[Lw], [low1, low2]])

        return x_mean, L

    @staticmethod
    def param(n, alpha=0.01, beta=2):
        """Get UKF param
        Args:
            n: dimension of state
            alpha: UKF param
            beta: UKF param
        Returns:
            eta: int
            Wm: (2*n+1)
            Wc: (2*n+1)
        """

        lam = n * (alpha ** 2 - 1)
        eta = (n + lam) ** 0.5
        Wm = torch.ones(2 * n + 1) / (2 * (n + lam))
        Wc = torch.ones(2 * n + 1) / (2 * (n + lam))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1 - alpha ** 2 + beta)

        return eta, Wm,  Wc