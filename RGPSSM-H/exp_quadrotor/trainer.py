from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import numpy as np
import torch
import time

from model.rgpssm.utils import ToTensor, clone_required_grad, Jacobian
from model.rgpssm_h.rgpssm_h import RGPSSM_H
from model.rgpssm_h.utils import IModel_H
from model.rgpssm_h.kernel import RBFKerH
from model.rgpssm.rgpssm_u import RGPSSM_U
from model.rgpssm.kernel import MultiTaskRBFKer
from model.rgpssm.utils import IModel
from ekf.ekf import EKF
from ekf.utils import IModelEKF

from utils import Torch2Np, DataRecorder, error_bar_plot, timing, set_seed, save_show, save_pickle, load_pickle
from utils import load_pickle, nMSE, Mnlp

class Sys():
    """System to provide data sequentially"""
    def __init__(self):
        self.data = load_pickle('./data/data_.pkl')
        self.dt = self.data['t'].ravel()[1] - self.data['t'].ravel()[0]
        self.t = 0

        self.data_read()
        self.num_update = 0

    def update(self):
        i = self.num_update

        self.p = self.P[i, :].reshape(-1, 1)
        self.v = self.V[i, :].reshape(-1, 1)
        self.a = self.A[i, :].reshape(-1, 1)
        self.q = self.Q_n[i, :].reshape(-1, 1)
        self.pwm = self.PWM_n[i, :].reshape(-1, 1)
        self.T_dir = self.T_Dir[i, :].reshape(-1, 1)

        self.t += self.dt
        self.num_update += 1

    def data_read(self):
        id_l = 0
        id_r = int(self.data['t'].size * 1.)

        self.P = self.data['p'][id_l:id_r, :]
        self.V = self.data['v'][id_l:id_r, :]
        self.A = self.data['a'][id_l:id_r, :]
        self.PWM = self.data['pwm'][id_l:id_r, :]
        self.Q = self.data['q'][id_l:id_r, :]
        self.T_Dir = self.data['T_dir'][id_l:id_r, :]

        self.num_data = self.P.shape[0]

        self.PWM_n = self.normal_data(self.PWM)
        self.Q_n = self.normal_data(self.Q)

    def normal_data(self, x):
        """
        Args:
            x: (n, dx)
        Returns:
            x_n: (n, dx)
        """
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        x_n = (x - mean) / std
        return x_n

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
        p, v = x[..., :3], x[..., 3:]
        dot_x = torch.cat((v, f), dim=-1)
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
        y = x[..., :3]

        return y, None


    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            z: GP input (..., nz)
            dzdx: Jacobin of z w.r.t. x (..., nz, nx) or None
        """
        z = c  / 5

        return z, None

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

        p, v = x[..., :3], x[..., 3:]
        dot_x = torch.cat((v, f), dim=-1)
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
        y = x[..., :3]

        return y, None

    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., dx)
            c: system input (..., dc)
        Returns:
            z: GP input (..., nf, dz_max)
            dzdx: Jacobin of flattened z w.r.t. x with shape (..., dz_max*nf, nx) or None
        """
        c_ = c.unsqueeze(-2) / 5
        z = torch.cat((c_, c_, c_), dim=-2)                # (*, nf, dz)

        return z, None

class Fun_EKF(IModelEKF):
    """System model for the EKF"""
    def __init__(self, dt, dw=2):
        self.dt = dt
        self.dw = dw

    def fun_tran(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: (p, v, w)
            c: (pwm, T_dir)
        """
        p, v, wb = x[..., :3], x[..., 3:6], x[..., 6:]
        a = self.acc(c, wb)

        dot_p = v
        dot_v = a

        p_new = p + self.dt * dot_p
        v_new = v + self.dt * dot_v
        F = torch.cat((p_new, v_new, wb), dim=-1)

        return F, None

    def fun_meas(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        y = x[..., :3]

        return y, None


    def acc(self, c, wb):
        w = wb[..., :-3]
        b = wb[..., -3:]
        pwm, T_dir, q = c[..., :4], c[..., 4:7], c[..., 7:11]

        phi = self.basis_fun(c)
        a_ = torch.sum(phi * w, dim=-1).unsqueeze(-1)   # (*, 1)
        a = a_ * T_dir                                  # (*, 3)
        a = a + b

        return a

    def basis_fun(self, c):
        pwm, T_dir, q = c[..., :4], c[..., 4:7], c[..., 7:11]

        phi_lst = [torch.sum(pwm**i, dim=-1).unsqueeze(-1) for i in range(self.dw)]
        phi = torch.cat(phi_lst, dim=-1)    # (*, dw)

        return phi


class Trainer():
    def __init__(self, name='rgpssm', seed=1):
        set_seed(seed)
        self.model_name = name

        # Parameters
        self.sys = Sys()
        self.sigma_noise = 0.045

        # RGPSSM initialization
        dx = 6
        dy = 3                                                      # dimension of latent state
        x0 = np.ones((dx, 1))                                       # initial state mean
        P0 = np.eye(dx) * 1                                         # initail state covariance
        Q = np.eye(dx) * 0.001**2 * self.sys.dt                     # process noise covariance
        Q[-3:, -3:] = np.eye(3) * 0.05**2 * self.sys.dt
        R = np.eye(dy) * self.sigma_noise**2                        # measurement noise covariance
        budget = 80

        ls = [3.8, 3.8, 3.8, 3.8, 1.3, 1.3, 1.3, 1.3]
        std_y = 0.5

        if name == 'rgpssm':
            ker = MultiTaskRBFKer(df=3, dz=8, std=[std_y, std_y, 0.05], ls=ls, jitter=1e-4)
            self.model = RGPSSM_U(x0, P0, Q, R, Fun(self.sys.dt), ker,
                                   flag_chol=True, budget=budget, eps_tol=8e-3, num_opt_hp=1, lr_hp=0.0015, type_score="full")
        elif 'rgpssm_h' in name:
            ker = RBFKerH(df=[1, 1, 1], dz=[8, 8, 8], std=[[std_y], [std_y], [std_y]], ls=[ls, ls, ls], jitter=1e-4)
            self.model = RGPSSM_H(x0, P0, Q, R, Fun_H(self.sys.dt), ker,
                                      flag_chol=True, budget=budget*3, eps_tol=8e-3, num_opt_hp=1, lr_hp=0.0015,
                                      type_score="full", type_filter=name[-3:].upper())
        elif name == 'ekf':
            dw = 3
            d_wb = dw + 3
            x0 = torch.zeros((6+d_wb, 1))
            P0 = torch.eye(6+d_wb) * 0.04
            P0[6:, 6:] = torch.eye(d_wb) * 1
            Q = torch.eye(6+d_wb)
            Q[:3, :3] = torch.eye(3) * 0.001**2 * self.sys.dt
            Q[3:6, 3:6] = torch.eye(3) * 0.05 ** 2 * self.sys.dt
            Q[-d_wb:, -d_wb:] = 0
            self.model = EKF(x0, P0, Q, R, Fun_EKF(self.sys.dt, dw=dw))
        else:
            raise ValueError(f"Unknown model name: {name}")

        self.data_recorder = DataRecorder()

    def data_record_update(self):
        data_name = ['t', 'p', 'v', 'a', 'pwm', 'q', 'x_est', 'std_est', 'x_pre', 'f_pre', 'std_f', 'std_x', 'ls', 'var']
        data_vec = [self.sys.t, self.sys.p, self.sys.v, self.sys.a, self.sys.pwm, self.sys.q,
                    self.x_est, self.std_est, self.x_pre, self.f_pre, np.diag(self.var_f_pre) ** 0.5, np.diag(self.var_x_pre) ** 0.5,
                    self.ls, self.var]
        self.data_recorder.data_add(data_name, data_vec)

    @timing
    def run(self, tend=60, t_pre=40):

        self.t_pre = t_pre
        tend = tend
        n = min(int(tend / self.sys.dt), self.sys.num_data)

        start = time.time()
        for ii in range(n):
            # system update
            self.sys.update()
            # RGPSSM update
            in_gp = np.concatenate((self.sys.pwm, self.sys.q), axis=0)
            in_gp = ToTensor(in_gp, view=(1, -1))
            if 'rgpssm' in self.model_name:
                self.RGPSSM_update(in_gp, ii)
            elif self.model_name == 'ekf':
                in_ekf = np.concatenate((self.sys.pwm, self.sys.T_dir, self.sys.q), axis=0)
                self.ekf_update(in_ekf, ii)

            # record
            self.update_hyperparam()
            self.data_record_update()
            if ii % 100 == 0:
                if 'rgpssm' in self.model_name:
                    print(f'ii = {ii}, BV={self.model.num_ips}')
                    print(f'ls = {self.ls}')
                    print(f'var = {self.var}')
                else:
                    print(f'ii = {ii}')

            if self.sys.t > self.t_pre and not ('t_sim' in self.data_recorder.database.keys()):
                end = time.time()
                t_train = end - start
                self.data_recorder.database['t_sim'] = t_train

        # save
        data = self.data_recorder.database
        data['t_pre'] = t_pre
        data['dt'] = self.sys.dt
        Recorder.save(self.model, self.model_name, data)

    def ekf_update(self, in_ekf, ii):
        in_ekf = ToTensor(in_ekf, view=(1, -1))
        T_dir = in_ekf[:, 4:]   # (1, 3)

        self.model.predict(in_ekf)
        self.x_pre = Torch2Np(self.model.x)
        self.var_x_pre = Torch2Np(self.model.P)

        wb = self.model.x[6:, :]  # (dw, 1)
        wb = clone_required_grad(wb, view=(1, -1))
        in_ekf = clone_required_grad(in_ekf, view=(1, -1))
        a = self.model.fun.acc(in_ekf, wb)
        da_dwb = Jacobian(wb, a).detach()

        var_a = da_dwb @ self.model.P[6:, 6:] @ da_dwb.T

        self.f_pre = Torch2Np(a)
        self.var_f_pre = Torch2Np(var_a)

        if self.sys.t < self.t_pre:
            self.model.correct(self.sys.p)
        self.x_est = Torch2Np(self.model.x)
        self.std_est = np.diag(Torch2Np(self.model.P)) ** 0.5

    def RGPSSM_update(self, in_gp, ii):

        F, var_F, f, var_f = self.model.predict(in_gp)
        self.x_pre = Torch2Np(F)
        self.var_x_pre = Torch2Np(var_F)
        self.f_pre = Torch2Np(f)
        self.var_f_pre = Torch2Np(var_f)

        if self.sys.t < self.t_pre:
            self.model.correct(self.sys.p)
        else:
            self.model.eps_tol = 1e100 # make sure not adding inducing point in the prediction phase
        self.x_est = Torch2Np(self.model.x)
        self.std_est = np.diag(Torch2Np(self.model.P)) ** 0.5

        if ii > 100 and self.sys.t < self.t_pre:
            self.model.hyperparam_opt()

        if 'rgpssm_h' in self.model_name:
            if (ii % 50 == 0 and ii > 50) and self.sys.t < self.t_pre:
                self.model.prune_redundant_points()

    def update_hyperparam(self):
        if 'rgpssm' in self.model_name:
            self.ls = [l.detach().clone().numpy().ravel() for l in self.model.kernel.ls]
            self.var = [v.detach().clone().numpy().ravel() for v in self.model.kernel.var]
            self.ls = np.concatenate(self.ls)
            self.var = np.concatenate(self.var)
        else:
            self.ls = np.eye(1) * np.nan
            self.var = np.eye(1) * np.nan


class Recorder():
    @staticmethod
    def save(model, name, data):
        d = [model, data]
        save_pickle(d, './log/' + name + '.pkl')

    @staticmethod
    def load(name):
        return load_pickle('./log/' + name + '.pkl')

    @staticmethod
    def eval(data, flag_print=False):
        d = data
        id_pre = np.sum(d['t'] < d['t_pre'])

        p = d['p'][id_pre:, :]
        p_pre = d['x_est'][id_pre:, :]
        std_pre = d['std_est'][id_pre:, :]

        mse = nMSE(p, p_pre)
        nll = Mnlp(p, p_pre, std_pre)

        if flag_print:
            print(f'mse: {mse}')
            print(f'nll: {nll}')

        return mse, nll