import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from .utils import *

class EKF():
    def __init__(self,
                 x0: Union[ndarray, Tensor],
                 P0: Union[ndarray, Tensor],
                 Q: Union[ndarray, Tensor],
                 R: Union[ndarray, Tensor],
                 fun: IModelEKF):
        """
        x0 : prior mean of state (nx, 1)
        P0 : prior variance of state (nx, nx)
        Q : process noise covariance (nx, nx)
        R : measurement noise covariance (ny, ny)
        fun: model information module
        """

        # Model
        self.fun = fun

        # Parameters of joint distribution
        self.x = Np2Torch(x0)
        self.P = Np2Torch(P0)
        self.Q = Np2Torch(Q)
        self.R = Np2Torch(R)
        self.nx = self.x.numel()

    def predict(self, c: Optional[Union[ndarray, Tensor]]=None, fun_tran=None, Q: Optional[Tensor]=None)-> Tuple[Tensor, Tensor]:
        """Prediction step
        Args:
            c : system input
            fun_tran : transition model [function: x, c -> F, Ax]
            Q : process noise covariance
        Returns
            x: mean of predicted state
            P: variance of predicted state
        """

        # Preparation
        if c is not None: c = Np2Torch(c).view(1, -1)
        if fun_tran is None: fun_tran = self.fun.fun_tran
        if Q is None: Q = self.Q

        # Prediction
        x = clone_required_grad(self.x)
        if c is not None: c = clone_required_grad(c).view(1, -1)
        x_pre, Ax = self.fun.fun_tran(x.view(1, -1), c)
        Ax = Jacobian(x, x_pre).detach() if Ax is None else Ax.detach()

        self.x = x_pre.detach().view(-1, 1)
        self.P = (Ax @ self.P @ Ax.T + Q).detach()


    def correct(self, y: Optional[Union[ndarray, Tensor]], c: Optional[Union[ndarray, Tensor]] = None, fun_meas=None, R: Optional[Tensor]=None):
        """Correction step
        Args:
            y : measurement
            c : system input
            fun_meas : measurement model [function: x, c -> y, Cx]
            R : measurement noise covariance
        Returns:
        """

        # Preparation
        if fun_meas is None: fun_meas = self.fun.fun_meas
        if c is not None: c = Np2Torch(c).view(1, -1)
        if R is None: R = self.R
        y = Np2Torch(y).view(-1, 1)

        x = clone_required_grad(self.x)
        if c is not None: c = clone_required_grad(c).view(1, -1)
        y_pre, Cx = self.fun.fun_meas(x.view(1, -1), c)
        Cx = Jacobian(x, y_pre).detach() if Cx is None else Cx.detach()

        Gam = self._keep_sym(Cx @ self.P @ Cx.T + R)
        rho = torch.linalg.cholesky(Gam)

        e = y.view(-1, 1) - y_pre.view(-1, 1)
        rhoinv_e = torch.linalg.solve_triangular(rho, e, upper=False)
        rhoinv_Cx_P = torch.linalg.solve_triangular(rho, Cx @ self.P, upper=False)

        self.x = x + rhoinv_Cx_P.T @ rhoinv_e
        self.P = self.P - rhoinv_Cx_P.T @ rhoinv_Cx_P


    def _keep_sym(self, A):
        """A should theoretically be symmetric, this function aims to eliminate numerical error"""
        return (A + A.T) * 0.5
