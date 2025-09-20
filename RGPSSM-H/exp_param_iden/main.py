from typing import Tuple, Optional, List, Union
from torch import Tensor

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.rgpssm_h.rgpssm_h import RGPSSM_H
from model.rgpssm_h.utils import IModel_H
from model.rgpssm_h.kernel import RBFKerH
from model.rgpssm.rgpssm_u import RGPSSM_U
from model.rgpssm.kernel import MultiTaskRBFKer
from model.rgpssm.utils import IModel

from system import Sys
from res import res_plot
from utils import Torch2Np, DataRecorder, error_bar_plot, timing, set_seed, save_show, save_pickle, load_pickle

class Fun_H(IModel_H):
    """System model for RGPSSM-H"""
    def __init__(self, dt):
        self.dt = dt

    def fun_tran(self, x: Tensor, c: Tensor, f:Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
            f: GP output (..., nf)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (nx, nx) or None
            Af: Jacobin for GP output dF/df (nx, nf) or None
        """

        f_bfn = f[..., [0]] * x + f[..., [1]]   # (*, dx)
        dot_x = f_bfn + c[..., [1]]             # (*, dx)
        F = x + dot_x * self.dt                 # (*, dx)

        return F, None, None

    def fun_meas(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (ny, nx) or None
        """
        y = x[..., [0]]
        Cx = torch.zeros((y.shape[-1], x.shape[-1]))
        Cx[:, 0] = 1.

        return y, Cx

    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., dx)
            c: system input (..., dc)
        Returns:
            z: GP input (..., nf, dz_max)
            dzdx: Jacobin of flattened z w.r.t. x with shape (..., dz_max*nf, nx) or None
        """
        z1 = c[..., [0]].unsqueeze(-2)                        # (*, 1, dz)
        z2 = c[..., [0]].unsqueeze(-2)                        # (*, 1, dz)
        z = torch.cat((z1, z2), dim=-2)                # (*, nf, dz)
        dzdx = None

        return z, dzdx


class Fun(IModel):
    """System model for RGPSSM"""
    def __init__(self, dt):
        self.dt = dt

    def fun_tran(self, x: Tensor, c: Tensor, f: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
            f: GP output (..., nf)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (nx, nx) or None
            Af: Jacobin for GP output dF/df (nx, nf) or None
        """
        phi = torch.zeros((1, 2))
        phi[0, 0] = x * 1.
        phi[0, 1] = 1.
        dot_x = f @ phi.T + c[0, 1]
        F = x + dot_x * self.dt

        return F, None, None

    def fun_meas(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (ny, nx) or None
        """
        y = x[..., [0]]
        Cx = torch.zeros((y.shape[-1], x.shape[-1]))
        Cx[:, 0] = 1.

        return y, Cx

    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            z: GP input (..., nz)
            dzdx: Jacobin of z w.r.t. x (..., nz, nx) or None
        """
        z = c[..., [0]]

        return z, None

class ParamKer(RBFKerH):
    """Additive Kernel which is a Gaussian kernel plus a basis function kernel"""
    def __init__(self, df, dz, std, ls, jitter):
        super().__init__(df, dz, std, ls, jitter)
        self.P0 = torch.eye(3)  # (1, 1)

    def ker_single(self, Z1: Tensor, Z2: Tensor, i: int) -> Tensor:
        """
        Args:
            Z1: (..., n1, 1)
            Z2: (..., n2, 1)
            i: index of function dimension
        Returns:
            K: (..., n1, n2)
        """
        K_rbf = super().ker_single(Z1, Z2, i)           # (..., n1, n2)
        K_param = self.ker_param_single(Z1, Z2, i)      # (..., n1, n2)
        return K_rbf + K_param

    def ker_param_single(self, Z1: Tensor, Z2: Tensor, i: int) -> Tensor:
        if i == 1:
            g1 = self.bfn(Z1)                                   # (..., n1, db)
            g2 = self.bfn(Z2)                                   # (..., n2, db)
            K = g1 @ self.P0 @ torch.transpose(g2, -2, -1)      # (..., n1, n2)
            return K
        else:
            return torch.zeros(1)

    def bfn(self, z):
        """
        Args:
            z: (..., 1)
        Returns:
            phi: (..., 5)
        """
        bfn_lst = [
            torch.cos(0.2 * z),
            torch.sin(0.5 * z),
            torch.sin(1. * z)]
        return torch.cat(bfn_lst, dim=-1)


class Exp():
    def __init__(self, seed=1):
        set_seed(seed)

        # Parameters
        self.sys = Sys(dt=0.01, std_noise=0.05)

        # RGPSSM initialization
        dim_latent = 1                                                      # dimension of latent state
        x0 = np.ones((dim_latent, 1))                                       # initial state mean
        P0 = np.eye(dim_latent) * 1                                         # initail state covariance
        Q = np.eye(dim_latent) * 1e-8 * self.sys.dt                         # process noise covariance
        R = np.eye(1) * self.sys.sigma_noise**2                             # measurement noise covariance
        budget = 80

        ker = MultiTaskRBFKer(df=2, dz=1, std=[1., 1.], ls=[1.], jitter=1e-4)
        self.gpssm = RGPSSM_U(x0, P0, Q, R, Fun(self.sys.dt), ker,
                               flag_chol=True, budget=budget, eps_tol=5e-3, num_opt_hp=1, lr_hp=1e-2, type_score="oldest")

        ker = RBFKerH(df=[1, 1], dz=[1, 1], std=[[1.], [1.]], ls=[[1.], [1.]], jitter=1e-4)
        self.gpssm_h = RGPSSM_H(x0, P0, Q, R, Fun_H(self.sys.dt), ker,
                                  flag_chol=True, budget=budget*2, eps_tol=5e-3, num_opt_hp=1, lr_hp=1e-2,
                                  type_score="oldest", type_filter='EKF')

        ker = ParamKer(df=[1, 1], dz=[1, 1], std=[[1.], [1.]], ls=[[1.], [1.]], jitter=1e-4)
        self.gpssm_p = RGPSSM_H(x0, P0, Q, R, Fun_H(self.sys.dt), ker,
                                  flag_chol=True, budget=budget*2, eps_tol=5e-3, num_opt_hp=1, lr_hp=1e-2,
                                  type_score="oldest", type_filter='EKF')

        self.data_recorder = DataRecorder()

    def data_record_update(self):

        data_name = ['t', 'y', 'x', 'xd', 'x_est', 'std_est', 'x_pre', 'f_pre', 'std_f', 'std_x', 'theta', 'ls', 'var',
                     'x_est_h', 'std_est_h', 'x_pre_h', 'f_pre_h', 'std_f_h', 'std_x_h', 'ls_h', 'var_h',
                     'x_est_p', 'std_est_p', 'x_pre_p', 'f_pre_p', 'std_f_p', 'std_x_p', 'ls_p', 'var_p']
        data_vec = [self.sys.t, self.sys.y, self.sys.x, self.sys.xd, self.x_est, self.std_est, self.x_pre, self.f_pre, np.diag(self.var_f_pre) ** 0.5,
                    np.diag(self.var_x_pre) ** 0.5, Torch2Np(self.sys.theta),
                    self.ls, self.var,
                    self.x_est_h, self.std_est_h, self.x_pre_h, self.f_pre_h, np.diag(self.var_f_pre_h) ** 0.5, np.diag(self.var_x_pre_h) ** 0.5,
                    self.ls_h, self.var_h,
                    self.x_est_p, self.std_est_p, self.x_pre_p, self.f_pre_p, np.diag(self.var_f_pre_p) ** 0.5,
                    np.diag(self.var_x_pre_p) ** 0.5,
                    self.ls_p, self.var_p
                    ]
        self.data_recorder.data_add(data_name, data_vec)

    @timing
    def run(self, tend=10):

        tend = tend
        for ii in range(int(tend / self.sys.dt)):
            # system update
            self.sys.update()
            # RGPSSM updaye
            in_gp = torch.zeros((2, 1))
            in_gp[0, 0] = self.sys.t
            in_gp[1, 0] = self.sys.u
            self.RGPSSM_update(in_gp.clone(), ii)
            self.RGPSSM_H_update(in_gp.clone(), ii)
            self.RGPSSM_P_update(in_gp.clone(), ii)
            # record
            self.update_hyperparam()
            self.data_record_update()
            if ii % 100 == 0:
                print(f'ii = {ii}')
                print(f'RGPSSM: ls = {self.ls}, var={self.var}, BV={self.gpssm.num_ips}')
                print(f'RGPSSM-H: ls = {self.ls_h}, var={self.var_h}, BV={self.gpssm_h.num_ips}')
                print(f'RGPSSM-P: ls = {self.ls_p}, var={self.var_p}, BV={self.gpssm_p.num_ips}')

    def RGPSSM_update(self, in_gp, ii):

        F, var_F, f, var_f = self.gpssm.predict(in_gp)
        self.x_pre = Torch2Np(F)
        self.var_x_pre = Torch2Np(var_F)
        self.f_pre = Torch2Np(f)
        self.var_f_pre = Torch2Np(var_f)

        self.gpssm.correct(self.sys.y)
        self.x_est = Torch2Np(self.gpssm.x)
        self.std_est = np.diag(Torch2Np(self.gpssm.P)) ** 0.5

        if ii > 100:
            self.gpssm.hyperparam_opt()

    def RGPSSM_H_update(self, in_gp, ii):

        F, var_F, f, var_f = self.gpssm_h.predict(in_gp)
        self.x_pre_h = Torch2Np(F)
        self.var_x_pre_h = Torch2Np(var_F)
        self.f_pre_h = Torch2Np(f)
        self.var_f_pre_h = Torch2Np(var_f)

        self.gpssm_h.correct(self.sys.y)
        self.x_est_h = Torch2Np(self.gpssm_h.x)
        self.std_est_h = np.diag(Torch2Np(self.gpssm_h.P)) ** 0.5

        if ii > 100:
            self.gpssm_h.hyperparam_opt()

        if ii % 1 == 0 and ii > 100:
            self.gpssm_h.prune_redundant_points()

    def RGPSSM_P_update(self, in_gp, ii):

        F, var_F, f, var_f = self.gpssm_p.predict(in_gp)
        self.x_pre_p = Torch2Np(F)
        self.var_x_pre_p = Torch2Np(var_F)
        self.f_pre_p = Torch2Np(f)
        self.var_f_pre_p = Torch2Np(var_f)

        self.gpssm_p.correct(self.sys.y)
        self.x_est_p = Torch2Np(self.gpssm_p.x)
        self.std_est_p = np.diag(Torch2Np(self.gpssm_p.P)) ** 0.5

        if ii > 100:
            self.gpssm_p.hyperparam_opt()

        if ii % 100 == 0 and ii > 100:
            self.gpssm_p.prune_redundant_points()

    def update_hyperparam(self):
        self.ls = [l.detach().clone().numpy().ravel() for l in self.gpssm.kernel.ls]
        self.var = [v.detach().clone().numpy().ravel() for v in self.gpssm.kernel.var]
        self.ls = np.concatenate(self.ls)
        self.var = np.concatenate(self.var)

        self.ls_h = [l.detach().clone().numpy().ravel() for l in self.gpssm_h.kernel.ls]
        self.var_h = [v.detach().clone().numpy().ravel() for v in self.gpssm_h.kernel.var]
        self.ls_h = np.concatenate(self.ls_h)
        self.var_h = np.concatenate(self.var_h)

        self.ls_p = [l.detach().clone().numpy().ravel() for l in self.gpssm_p.kernel.ls]
        self.var_p = [v.detach().clone().numpy().ravel() for v in self.gpssm_p.kernel.var]
        self.ls_p = np.concatenate(self.ls_p)
        self.var_p = np.concatenate(self.var_p)

    def res(self):
        res_plot(self.data_recorder.database, self.gpssm, self.gpssm_h, self.gpssm_p, flag_save=True, flag_show=True)


if __name__ == '__main__':
    flag_load = False
    if flag_load:
        exp = load_pickle('./log/exp.pkl')
    else:
        exp = Exp()
        exp.run(tend=30)
        save_pickle(exp, './log/exp.pkl')
    exp.res()
    plt.show()

