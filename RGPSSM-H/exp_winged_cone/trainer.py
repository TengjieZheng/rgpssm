from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from model.rgpssm_h.rgpssm_h import RGPSSM_H
from model.rgpssm_h.kernel import RBFKerH
from model.rgpssm_h.utils import IModel_H

from model.rgpssm.rgpssm_u import RGPSSM_U
from model.rgpssm.kernel import MultiTaskRBFKer
from model.rgpssm.utils import IModel

from vehicle.winged_cone import Winged_cone
from controller import Controller
from utils import DataRecorder, set_seed, save_show, ToTensor, Torch2Np, save_pickle, load_pickle, timing



class Fun_H(IModel_H):
    """System model for RGPSSM-H"""
    def __init__(self, dt, wc, scale_f):
        self.dt = dt
        self.I_mat = ToTensor(wc.I_mat)
        self.S = wc.S_ref
        self.b = wc.b
        self.c = wc.c
        self.Q = wc.Q
        self.scale_f = scale_f

        self.affine = self.Q * self.S * np.array([self.b, self.c, self.b]).reshape(-1, 1)    # (3, 1)
        self.affine = ToTensor(self.affine)
        self.C0 = torch.zeros((3, 1))

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

        f_ = f * self.scale_f

        omg = x.unsqueeze(-1)                               # (..., 3, 1)
        M_ad = self.affine * (self.C0 + f_.unsqueeze(-1))   # (..., 3, 1)
        M_i = -torch.cross(omg, self.I_mat @ omg, dim=-2)   # (..., 3, 1)
        dot_x = torch.linalg.solve(self.I_mat, M_i + M_ad)  # (..., 3, 1)
        dot_x = dot_x.squeeze(-1)                           # (..., 3)

        F = x + dot_x * self.dt                             # (..., dx)

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

        y = x                                   # (..., ny)
        Cx = torch.eye(x.shape[-1])             # (ny, nx)

        return y, Cx

    def fun_input(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., dx)
            c: system input (..., dc) - alpha, beta, delta_a, delta_e, delta_r
        Returns:
            z: GP input (..., nf, dz_max)
            dzdx: Jacobin of flattened z w.r.t. x with shape (..., dz_max*nf, nx) or None
        """

        p, q, r = x[..., [0]], x[..., [1]], x[..., [2]]
        z1 = torch.cat((c, p, r), dim=-1).unsqueeze(-2)                        # alpha, beta, delta_a, delta_e, delta_r, p, r (..., 1, 7)
        z2 = torch.cat((c[..., [0]], c[..., 2:], q), dim=-1).unsqueeze(-2)     # alpha, delta_a, delta_e, delta_r, q           (..., 1, 5)
        z3 = z1                                                                         # alpha, beta, delta_a, delta_e, delta_r, p, r (..., 1, 7)

        z2_ = torch.cat((z2, torch.zeros(z2.shape[:-1] + (2,))), dim=-1)        # (..., 1, 7) fill zero to make all dimension have same size
        z = torch.cat((z1, z2_, z3), dim=-2) * 10                               # (*, nf, dz_max)
        dzdx = None

        return z, dzdx

class Fun(IModel):
    """System model for RGPSSM"""
    def __init__(self, dt, wc, scale_f):
        self.dt = dt
        self.I_mat = ToTensor(wc.I_mat)
        self.S = wc.S_ref
        self.b = wc.b
        self.c = wc.c
        self.Q = wc.Q
        self.scale_f = scale_f

        self.affine = self.Q * self.S * np.array([self.b, self.c, self.b]).reshape(-1, 1)    # (3, 1)
        self.affine = ToTensor(self.affine)
        self.C0 = torch.zeros((3, 1))

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

        f_ = f * self.scale_f

        omg = x.unsqueeze(-1)  # (..., 3, 1)
        M_ad = self.affine * (self.C0 + f_.unsqueeze(-1))  # (..., 3, 1)
        M_i = -torch.cross(omg, self.I_mat @ omg, dim=-2)  # (..., 3, 1)
        dot_x = torch.linalg.solve(self.I_mat, M_i + M_ad)  # (..., 3, 1)
        dot_x = dot_x.squeeze(-1)  # (..., 3)

        F = x + dot_x * self.dt  # (..., dx)

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
        y = x                                   # (..., ny)
        Cx = torch.eye(x.shape[-1])             # (ny, nx)

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
        z = torch.cat((c, x), dim=-1)  # alpha, beta, delta_a, delta_e, delta_r, p, q, r (..., 1, 8)
        z = z * 10
        dzdx = None

        return z, dzdx


class Trainer():
    def __init__(self, name='rgpssm'):
        self.model_name = name
        set_seed(0)

        self.wc = Winged_cone(delta_t=0.05)
        self.wc.sigma_rate = 0.173 * np.array([10.6, 5.5, 1.6]) / 57.3    # noise standard deviation for rate
        print(f'Ma = {self.wc.state["Ma"]}')

        # RGPSSM initialization
        dx = 3                                                      # dimension of latent state
        x0 = np.ones((dx, 1))                                       # initial state mean
        P0 = np.eye(dx) * 4e-2                                      # initail state covariance
        Q = np.eye(dx) * 2e-3**2 * self.wc.delta_t                  # process noise covariance
        R = np.diag(self.wc.sigma_rate**2)                          # measurement noise covariance
        budget = 30

        self.scale_f = torch.Tensor([0.5e-3, 2.e-3, 0.5e-3]).view(-1)  # magnitude of f for each dimension, used for normal
        std1, std2, std3 = 1, 1, 1
        s_angle = 3 / 57.3 * 10
        s_delta = 25 / 57.3 * 10
        s_rate = 6 / 57.3 * 10
        ls1 = [s_angle, s_angle, s_delta, s_delta, s_delta, s_rate, s_rate]
        ls2 = [s_angle, s_delta, s_delta, s_delta, s_rate]
        ls3 = [s_angle, s_angle, s_delta, s_delta, s_delta, s_rate, s_rate]

        if 'rgpssm_h' in name:
            ker = RBFKerH(df=[1, 1, 1], dz=[7, 5, 7], std=[[std1], [std2], [std3]], ls=[ls1, ls2, ls3], jitter=1e-4)
            self.model = RGPSSM_H(x0, P0, Q, R, Fun_H(self.wc.delta_t, self.wc, self.scale_f), ker,
                                  flag_chol=True, budget=budget*3, eps_tol=1e-3, num_opt_hp=1, lr_hp=1e-2,
                                  type_score="full", type_filter=name[-3:].upper())
            self.model.alpha_ukf = 1e-1
            self.model.beta_ukf = 2
        elif name == 'rgpssm':
            ls = [s_angle, s_angle, s_delta, s_delta, s_delta, s_rate, s_rate, s_rate]
            ker = MultiTaskRBFKer(df=3, dz=8, std=[std1, std2, std3], ls=ls, jitter=1e-4)
            self.model = RGPSSM_U(x0, P0, Q, R, Fun(self.wc.delta_t, self.wc, self.scale_f), ker,
                                  flag_chol=True, budget=budget, eps_tol=1e-3, num_opt_hp=1, lr_hp=1e-2, type_score="full")
        else:
            raise ValueError(f"Unknown model name: {name}")

        self.ctrl = Controller(self.wc, fun_C=self.fun_C)

        self.alpha_old = self.wc.state['alpha']
        self.beta_old = self.wc.state['beta']
        self.data_recorder = DataRecorder()


    def data_record_update(self):

        self.data_name = (list(self.wc.state.keys())
                          + ['x_est', 'std_est', 'x_pre', 'f_pre', 'std_f', 'std_x', 'ls', 'var'])
        self.data_vector = (list(self.wc.state.values())
                          + [self.x_est, self.std_est, self.x_pre, self.f_pre, self.std_f_pre, self.std_x_pre, self.ls, self.var])
        self.data_recorder.data_add(self.data_name, self.data_vector)

    @timing
    def run(self, tend=60, t_ctrl=30):
        for ii in range(int(tend/self.wc.delta_t)):
            # vehicle update
            self.wc.target_update()
            if self.wc.t > t_ctrl:
                self.u = self.ctrl.update(self.wc.state, self.wc.angle_c, fun_C=self.fun_C_gp) # use the learning controller
            else:
                self.u = self.ctrl.update(self.wc.state, self.wc.angle_c)                       # use the basic  controller

            self.wc.update(u=self.u, update_tgt=True)
            self.wc.state['x1_d'] = self.ctrl.x_ref[0:3, :]
            self.wc.state['x2_c'] = self.ctrl.x2_c
            self.wc.state['x2_d'] = self.ctrl.x2_f

            # GP update
            s = self.wc.state
            in_gp = np.array([self.alpha_old, self.beta_old, s['delta_a'], s['delta_e'], s['delta_r']])
            in_gp = ToTensor(in_gp)
            self.y = np.array([s['p_meas'], s['q_meas'], s['r_meas']])
            self.RGPSSM_update(in_gp.clone(), ii)

            # record
            self.update_hyperparam()
            self.data_record_update()
            self.alpha_old = s['alpha']
            self.beta_old = s['beta']
            if ii % 100 == 0:
                print(f'ii = {ii}, t = {self.wc.t:.2f}s, BV={self.model.num_ips}')
                print(f'ls = {self.ls}')
                print(f'var = {self.var}')

        # Save
        self.data_recorder.database['scale_f'] = Torch2Np(self.scale_f)
        self.data_recorder.database['t_ctrl'] = t_ctrl
        Recorder.save(self.model,  self.model_name, self.data_recorder.database)

    def RGPSSM_update(self, in_gp, ii):

        self.model.fun.C0 = ToTensor(self.wc.state['C_moment_0'].reshape(-1, 1))

        F, var_F, f, var_f = self.model.predict(in_gp)
        self.x_pre = Torch2Np(F)
        self.std_x_pre = np.diag(Torch2Np(var_F)) ** 0.5
        self.f_pre = Torch2Np(f)
        self.std_f_pre = np.diag(Torch2Np(var_f)) ** 0.5

        self.model.correct(self.y)
        self.x_est = Torch2Np(self.model.x)
        self.std_est = np.diag(Torch2Np(self.model.P)) ** 0.5

        if ii > 100:
            self.model.hyperparam_opt()

        if 'rgpssm_h' in self.model_name:
            if ii % 50 == 0 and ii > 50:
                self.model.prune_redundant_points()

    def fun_C(self, alpha, beta, omg, delta, h, V, flag_pull=False):
        """
        Args:
            omg: [p, q, r]
            delta: [delta_a, delta_e, delta_r]
        Returns:
            C: [Cl, Cm, Cn]
        """
        CL, CD, CY, Cl, Cm, Cn = self.wc.ad.get_C(alpha, beta, *delta.ravel(), *omg.ravel(), V, h, flag_pull=flag_pull)
        return np.array([Cl, Cm, Cn]).reshape(-1, 1)

    def fun_C_gp(self, alpha, beta, omg, delta, h, V):
        c = np.array([alpha, beta, *delta.ravel()]).reshape(1, -1)
        C, _ = self.model.GP_predict(omg.reshape(1, -1), c)
        C = C.view(-1) * self.scale_f.view(-1)

        C0 = self.fun_C(alpha, beta, omg, delta, h, V, flag_pull=False)

        return Torch2Np(C).reshape(-1, 1) + C0.reshape(-1, 1)

    def update_hyperparam(self):
        self.ls = [l.detach().clone().numpy().ravel() for l in self.model.kernel.ls]
        self.var = [v.detach().clone().numpy().ravel() for v in self.model.kernel.var]
        self.ls = np.concatenate(self.ls)
        self.var = np.concatenate(self.var)


class Recorder():
    @staticmethod
    def save(model, name, data):
        d = [model, data]
        save_pickle(d, './log/' + name + '.pkl')

    @staticmethod
    def load(name):
        return load_pickle('./log/' + name + '.pkl')






