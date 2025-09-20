# -*- coding:utf-8 -*-
# @author  : Zheng Tengjie
# @time    : 2025/03/28 21:37
# @function: Recursive Gaussian process state space model (RGPSSM)
# @version : V2

from typing import Optional, Union, Tuple, List
import copy

import torch
from numpy import ndarray
from torch import Tensor
import torch.nn as nn
from torch.linalg import solve_triangular as solve_tri
import numpy as np

from .utils import *
from .kernel import IKer
import cholup, choldown


class RGPSSM():
    """Recursive Gaussian process state space model (RGPSSM)"""

    def __init__(self,
                 x0: Union[ndarray, Tensor],
                 P0: Union[ndarray, Tensor],
                 Q: Union[ndarray, Tensor],
                 R: Union[ndarray, Tensor],
                 fun: IModel, kernel: IKer,
                 flag_chol:bool =True,
                 budget:int =50, eps_tol: float =1e-2,
                 num_opt_hp: int =0, lr_hp: float =1e-3,
                 type_score: str ='full',
                 Z: Optional[Union[ndarray, Tensor]]=None,
                 Qu: float =0e-6):
        """
        x0 : prior mean of state (dx, 1)
        P0 : prior variance of state (dx, dx)
        Q : process noise covariance (dx, dx)
        R : measurement noise covariance (dy, dy)
        fun: model information module
        kernel : kernel of GP
        flag_chol : flag to use Cholesky decomposition for joint covariance or not
        budget : the maximum number of inducing points [int]
        eps_tol : threshold for determination of adding new inducing points [float]
        num_opt_hp : optimization number for hyperparameter at each correction step [int]
        lr_hp : learning rate for hyperparameter optimization [float]
        type_score: type of score for deleting inducing points， ”full“, "mean", "oldest"
        Z : inducing inputs [tensor]
        Qu : process noise variance for inducing outputs [float]
        """

        # GP kernel and Model
        self.kernel = kernel
        self.fun = fun

        # Parameters of joint distribution
        self.x = ToTensor(x0, view=(-1, 1))
        self.P = ToTensor(P0)
        self.Q = ToTensor(Q)
        self.R = ToTensor(R)
        self.flag_chol = flag_chol
        self.Qu = Qu
        if Z is not None: self._init_q(Z)
        self.dx = self.x.numel()

        # Parameters about optimization
        self.budget = budget
        self.eps_tol = eps_tol
        self.lr_hp = lr_hp
        self.num_hp = num_opt_hp
        self.type_score = type_score

    def predict(self, c: Optional[Union[ndarray, Tensor]]=None, fun_tran=None, Q: Optional[Tensor]=None)\
            ->Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Prediction step
        Args:
            c : system input
            fun_tran : transition model [function: x, c, f -> F, Ax, Af]
            Q : process noise covariance
        Returns
            F : mean of predicted state
            var_F : variance of predicted state
            f: mean of predicted GP
            var_f: variance of predicted GP
        """

        c = ToTensor(c, view=(1, -1))
        F, var_F, f, var_f = self._moments_propagate_EKF(c, fun_tran, Q)
        self._inducing_points_opt()

        return F, var_F, f, var_f

    def correct(self, y: Optional[Union[ndarray, Tensor]], c: Optional[Tensor]=None, fun_meas=None, R: Optional[Tensor]=None):
        """Correction step
        Args:
            y : measurement
            c : system input
            fun_meas : measurement model [function: x, c -> y, Cx]
            R : measurement noise covariance
        Returns:
        """

        if fun_meas is None: fun_meas = self.fun.fun_meas
        if R is None: R = self.R
        y = ToTensor(y, view=(1, -1))

        # Calculate some quantities
        x = clone_required_grad(self.x)
        c = clone_required_grad(c, view=(1, -1))
        y_pre, Cx = fun_meas(x.view(1, -1), c)
        y_pre = y_pre.view(-1, 1)
        if Cx is None: Cx = Jacobian(x, y_pre).detach()
        e = y - y_pre.detach()

        if self.flag_chol:
            P, V, _ = self._get_cov(self.L, isP=True, isV=True, isS=False)

            H = torch.cat((Cx, torch.zeros((Cx.shape[0], V.shape[1]))), dim=1)
            Gamma = keep_sym(Cx @ P @ Cx.T + R)
            rho = torch.linalg.cholesky(Gamma)
            inv_rho_H = torch.linalg.solve(rho, H)
            x = self.L @ (self.L.T @ inv_rho_H.T)
            self.L = choldown.chol_downdate(self.L, x)
            self.P, self.V, self.S = self._get_cov(self.L)

            q = Cx.T @ torch.linalg.solve(Gamma, e)
            self.x = self.x + P @ q
            self.m = self.m + V.T @ q

        else:
            L = torch.linalg.cholesky(Cx @ self.P @ Cx.T + R)
            inv_L_C = torch.linalg.solve(L, Cx)
            r = -inv_L_C.T @ inv_L_C
            q = inv_L_C.T @ torch.linalg.solve(L, e)

            # Correce the moments of the joint distribution
            P, V = self.P, self.V
            # mean
            self.x = self.x + P @ q
            self.m = self.m + V.T @ q
            # moments
            self.P = self.P + P @ r @ P
            self.V = self.V + P @ r @ V
            self.S = self.S + V.T @ r @ V

    def hyperparam_opt(self):
        """Hyperparameter optimization"""

        loss = np.nan
        if self.num_hp > 0:
            # Optimization
            hp0 = copy.deepcopy(self.kernel.state_dict())
            if not hasattr(self, 'optimizer'):
                self.optimizer = torch.optim.Adam(self.kernel.parameters(), lr=self.lr_hp)

            for ii in range(self.num_hp):
                self.optimizer.zero_grad()
                loss = self._loss_hp()
                loss.backward()
                self.optimizer.step()

            # Update joint distribution
            with torch.no_grad():
                Knew = self.kernel(self.Z)
                L_Knew = self._chol_decompose(Knew)
            inv_S_DK = self._get_inv_S_DK(self.L_Kuu, L_Knew)

            V, S = self.V, self.S
            q = -inv_S_DK @ self.m
            r = -inv_S_DK

            flag_update = False
            if (not torch.isnan(loss)):
                # Update joint distribution using new hyperparameters
                try:
                    P = self.P + V @ r @ V.T
                    V = self.V + V @ r @ S
                    S = self.S + S @ r @ S
                    Sigma = assemble_cov(P, V.T, S)
                    L = torch.linalg.cholesky(keep_sym(Sigma))
                    flag_update = True
                except RuntimeError:
                    pass

            if flag_update:
                if self.flag_chol:
                    self.L = L
                    self.P, self.V, self.S = self._get_cov(L)
                else:
                    self.P, self.V, self.S = P, V, S

                self.x = self.x + V @ q
                self.m = self.m + S @ q
                self._update_K_chol()
            else:
                # Not update
                self.kernel.load_state_dict(hp0)

            loss = loss.item()

        return loss

    def GP_predict(self, x: Optional[Union[ndarray, Tensor]], c: Optional[Union[ndarray, Tensor]]=None)\
            ->Tuple[Tensor, Tensor]:
        """Get mean and variance of GP prediction
        Args:
            x: system state (dx, 1)
            c: system input (nc, 1)
        Returns:
            f: mean of predicted GP (nf, 1)
            var_f: variance of predicted GP (nf, nf)
        """

        x = ToTensor(x, view=(1, -1))
        c = ToTensor(c, view=(1, -1))
        z, dzdx = self.fun.fun_input(x.view(1, -1), c)
        with torch.no_grad():
            Kff = self.kernel(z)

        if hasattr(self, 'Z'):
            with torch.no_grad():
                Kfu = self.kernel(z, self.Z)
            kfu = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, Kfu.T, upper=False), upper=True).T
            f = kfu @ self.m
            var_f = Kff - kfu @ Kfu.T + kfu @ self.S @ kfu.T
        else:
            f = torch.zeros((self.kernel.df, 1))
            var_f = Kff

        return f, var_f

    def GP_derivative_predict_ad(self, x: Union[ndarray, Tensor], c: Optional[Union[ndarray, Tensor]] = None)\
            -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """Get the moments of GP prediction and its partial derivative using automatic differentiation
        Args:
            x: system state (dx, 1)
            c: system input (nc, 1)
        Returns:
            f: mean of predicted GP (nf, 1)
            var_f: variance of predicted GP (nf, nf)
            dfdz: mean of partial derivative dfdz (nf, nz)
            var_dfdz: variance of partial derivative dfdz List[(nz, nz)]
        """

        x = ToTensor(x, view=(1, -1))
        c = ToTensor(c, view=(1, -1))
        z, dzdx = self.fun.fun_input(x, c)
        zl = clone_required_grad(z)
        zr = clone_required_grad(z)

        def get_Hessian(zl, zr, var_f):
            dvar_dzl = Jacobian(zl, var_f)
            var_dfdz = []
            for ii in range(var_f.numel()):
                J = Jacobian(zr, dvar_dzl[ii, :]).T
                var_dfdz.append(J.detach())

            return var_dfdz

        if hasattr(self, 'Z'):
            Kff = self.kernel(zl, zr)
            Kfu_l = self.kernel(zl, self.Z)
            kfu_l = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, Kfu_l.T, upper=False), upper=True).T
            Kfu_r = self.kernel(zr, self.Z)
            kfu_r = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, Kfu_r.T, upper=False), upper=True).T

            f = kfu_l @ self.m
            var_f = Kff - kfu_l @ Kfu_r.T + kfu_l @ self.S @ kfu_r.T
        else:
            f = torch.zeros((self.kernel.df, 1))
            var_f = self.kernel(zl, zr)

        return f.detach(), var_f.detach(), Jacobian(zl, f).detach(), get_Hessian(zl, zr, torch.diag(var_f).view(-1, 1))

    def GP_derivative_predict(self, x: Union[ndarray, Tensor], c: Optional[Union[ndarray, Tensor]] = None) \
            -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """ Get the moments of GP prediction and its partial derivative using analytical method (now only for nf=1)
        Args:
            x: system state (dx, 1)
            c: system input (nc, 1)
        Returns:
            f: mean of predicted GP (nf, 1)
            var_f: variance of predicted GP (nf, nf)
            dfdz: mean of partial derivative dfdz (nf, nz)
            var_dfdz: variance of partial derivative dfdz List[(nz, nz)]
        Notation:
            kernel:
                K(x1, x2) = v * exp[-0.5 * (x1 - x2)^T @ Lam^-1 @ (x1 - x2)]
                where Lam = diag(lam) and lam = [l1^2, ..., ln^2]
            Then,
                dK(x1, x2)/dx1      = -K(x1,x2) * Lam^-1 @ (x1 - x2)
                d^2K(x1, x2)/dx1dx2 = K(x1,x2) * Lam^-1
        """

        x = ToTensor(x, view=(1, -1))
        c = ToTensor(c, view=(1, -1))
        z, dzdx = self.fun.fun_input(x, c)
        Kff = self.kernel(z)

        if hasattr(self, 'Z'):
            f, dfdz, Kfu, kfu, dKdz = self._dfdz(z)  # f:(nf, 1), dfdz:(nf, nz), dKdz:(nz, nf, Nz*nf)
            var_f = Kff - kfu @ Kfu.T + kfu @ self.S @ kfu.T  # (nf, nf)

            dKdzT = dKdz.permute(0, 2, 1)  # (nz, Nz*nf, nf）
            dkdzT = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, dKdzT, upper=False), upper=True)  # (nz, NZ*nf, nf)
            dkdz = dkdzT.permute(0, 2, 1) # (nz, nf, NZ*nf)
            tmp = torch.einsum('ijk, kl, nlm->injm', dkdz, self.S - self.Kuu, dkdzT)  # (nz, nz, nf, nf)
            dKdzdz = self.kernel.dK_dz1dz2(z, z)  # (nz, nz, nf, nf)
            var_dfdz = dKdzdz + tmp
        else:
            f = torch.zeros((self.kernel.df, 1))                    # (nf, 1)
            var_f = torch.diag(Kff)                                 # (nf, nf)
            dfdz = torch.zeros((self.kernel.df, self.kernel.dz))    # (nf, nz)
            dKdzdz = self.kernel.dK_dz1dz2(z, z)                    # (nz, nz, nf, nf)
            var_dfdz = dKdzdz                                       # (nz, nz, nf, nf)

        var_dfdz = [var_dfdz.detach()[:, :, i, i] for i in range(self.kernel.df)]
        return f.detach(), var_f.detach(), dfdz.T.detach(), var_dfdz

    def _dfdz(self, z):
        Kfu = self.kernel(z, self.Z)
        kfu = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, Kfu.T, upper=False), upper=True).T
        alpha = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, self.m, upper=False), upper=True)

        f = Kfu @ alpha
        dKdz = self.kernel.dK_dz1(z, self.Z)  # (nz, nf, NZ*nf)
        dfdz = torch.einsum('ijk, k->ij', dKdz, alpha.view(-1))  # (nz, nf)
        dfdz = dfdz.T  # (nf, nz)

        return f, dfdz, Kfu, kfu, dKdz

    def _init_q(self, Z: Union[ndarray, Tensor]):
        """Initialize inducing point distribution
        Args:
            Z: each row is a inducing input (N, nz)
        """
        # V presents the cov between x and inducing outputs u
        # S presents the var of inducing points u
        # z presents a inducing point input, and Z is the point set
        # m is the output mean of z

        self.Z = ToTensor(Z)
        self.m = torch.zeros((self.kernel.df * Z.shape[0], 1))
        with torch.no_grad():
            self.S = self.kernel(self.Z)
        self.V = torch.zeros(self.dx, self.S.shape[0])
        self._update_K_chol()

        if self.flag_chol:
            Sigma = assemble_cov(self.P, self.V.T, self.S)
            self.L = torch.linalg.cholesky(Sigma)

    def _moments_propagate_EKF(self, c=None, fun_tran=None, Q=None):

        # Preparation
        if fun_tran is None: fun_tran = self.fun.fun_tran
        if Q is None: Q = self.Q

        # Update the moments of joint distribution
        if not hasattr(self, 'Z'):
            # Give an initial inducing point
            x = self.x.detach().clone()
            z, dzdx = self.fun.fun_input(x.view(1, -1), c)
            
            ls = self.kernel.ls.detach().clone().view(1, -1)
            # choose a non-zero initial point to avoid zero partial derivative
            self._init_q(z + ls*torch.randn(1)*0.1)

        z, mt, Ft, Ax, Af, gam, Vxu, Vxt, Stt, Stu, Suu, ktu = self._get_tmp(c, fun_tran)
        self.x = Ft

        if self.flag_chol:
            # Update of Cholesky version
            tr_var = torch.sum(self.kernel.var).item()
            K_diag = torch.diag(self.Kuu).reshape(-1, self.kernel.df)  # (nb, df)
            tr_max = torch.sum(K_diag, dim=-1).max()
            # if torch.trace(gam) > (self.eps_tol + self.kernel.jitter) * tr_var:
            if torch.trace(gam) > tr_max * self.eps_tol:
                # Update with adding point
                v_T = solve_tri(self.L, torch.cat((Vxt, Stu.T), dim=0), upper=False)
                beta = torch.linalg.cholesky(gam)
                L1 = torch.cat((self.L, torch.zeros((self.L.shape[0], beta.shape[1]))), dim=1)
                L2 = torch.cat((v_T.T, beta), dim=1)
                L_new = torch.cat((L1, L2), dim=0)
                # Suppose L_new = [L, 0; v, beta]，and then mat_new = [L*L^T, L*v^T;v*L^T, beta*beta^T + v * v^T]
                # Based on this, we can get v and beta，thus getting L_new

                dx, nu, nu_new = self.dx, Vxu.shape[1], Vxu.shape[1] + self.kernel.df
                Phi1 = torch.cat((Ax, torch.zeros((dx, nu)), Af), dim=1)
                Phi2 = torch.cat((torch.zeros((nu_new, dx)), torch.eye(nu_new)),dim=1)
                Phi = torch.cat((Phi1, Phi2), dim=0)
                sqrt_noise = torch.cat((torch.diag(Q).view(-1, 1), self.Qu * torch.ones((nu_new, 1))), dim=0) ** 0.5
                right_factor = torch.cat((L_new.T @ Phi.T, torch.diag_embed(sqrt_noise.view(-1))), dim=0)
                _, R = torch.linalg.qr(right_factor)

                self.L = R.T
                self.m = torch.cat((self.m, mt), dim=0)
                self.P, self.V, self.S = self._get_cov(self.L)
                self.Z = torch.cat((self.Z, z), dim=0)
                self._update_K_chol()
            else:
                # Update without adding point
                dx, nu = self.dx, Vxu.shape[1]
                Phi1 = torch.cat((Ax, Af@ktu), dim=1)
                Phi2 = torch.cat((torch.zeros((nu, dx)), torch.eye(nu)), dim=1)
                Phi = torch.cat((Phi1, Phi2), dim=0)

                R_bias = torch.eye(dx+nu)
                R_bias[:dx, :dx] = torch.linalg.cholesky(self.Q + Af @ gam @ Af.T).T
                R_bias[dx:, dx:] = torch.eye(nu) * self.Qu**0.5
                tmp = torch.cat((self.L.T @ Phi.T, R_bias), dim=0)
                _, R = torch.linalg.qr(tmp)
                self.L = R.T
                self.P, self.V, self.S = self._get_cov(self.L)
        else:
            # Update of regular version
            self.P = keep_sym(Ax @ self.P @ Ax.T + Ax @ Vxt @ Af.T + Af @ Vxt.T @ Ax.T + Af @ Stt @ Af.T + Q)
            self.V = Ax @ Vxu + Af @ Stu

            tr_var = torch.sum(self.kernel.var).item()
            if torch.trace(gam) > (self.eps_tol + self.kernel.jitter) * tr_var:
                # adapt jitter and var
                # Update with adding point
                self.m = torch.cat((self.m, mt), dim=0)
                self.V = torch.cat((self.V, Ax @ Vxt + Af @ Stt), dim=1)
                Sun = torch.cat((Suu, Stu.T), dim=1)
                Stn = torch.cat((Stu, Stt), dim=1)
                self.S = torch.cat((Sun, Stn), dim=0)
                self.S = self.S + torch.eye(self.S.shape[0]) * self.Qu
                self.Z = torch.cat((self.Z, z), dim=0)
                self._update_K_chol()

        return self.x, self.P, mt, Stt

    def _get_tmp(self, c, fun_tran):

        x = clone_required_grad(self.x)
        c = clone_required_grad(c, view=(1, -1))
        z, dzdx = self.fun.fun_input(x.view(1, -1), c)

        Vxu, Suu = self.V, self.S
        Ktt = self.kernel(z)
        Ktu = self.kernel(z, self.Z)
        ktu = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, Ktu.T, upper=False), upper=True).T
        gam = Ktt - ktu @ Ktu.T
        Vxt = Vxu @ ktu.T
        Stu = ktu @ Suu
        Stt = gam + Stu @ ktu.T

        mt = ktu @ self.m
        x0 = 1*x # x0 is not x
        Ft, Ax0, Af = fun_tran(x0.view(1, -1), c, mt.view(1, -1))
        Ft = Ft.view(-1, 1)

        if Ax0 is None: Ax0 = Jacobian(x0, Ft).detach()
        if Af is None: Af = Jacobian(mt, Ft).detach()
        if dzdx is None: dzdx = Jacobian(x, z).detach()
        dfdz = Jacobian(z, mt).detach()
        Ax = Ax0 + Af @ dfdz @ dzdx

        return z.detach(), mt.detach(), Ft.detach(), Ax.detach(), Af.detach(), gam.detach(), \
               Vxu.detach(), Vxt.detach(), Stt.detach(), Stu.detach(), Suu.detach(), ktu.detach()

    def _inducing_points_opt(self, flag_opt=False, type_score=None):
        """Optimize the inducing-point set"""

        if type_score is None:
            type_score = self.type_score

        if self.Z.shape[0] > self.budget or flag_opt:
            if self.type_score == "oldest":
                id_discard = 0
                l, r = id_discard * self.kernel.df, (id_discard + 1) * self.kernel.df
            else:
                id_discard, l, r = self._get_score(type_score)

            self.Z = get_vec_left(self.Z, id_discard, id_discard+1, dim=0)
            self.m = get_vec_left(self.m, l, r, dim=0)

            if self.flag_chol:
                l, r = l + self.dx, r + self.dx
                self.L = chol_delete(self.L, l, r)
                self.P, self.V, self.S = self._get_cov(self.L)
            else:
                self.V = get_vec_left(self.V, l, r, dim=1)
                self.S = get_mat_left(self.S, l, r)

            self._update_K_chol()

    def _get_score(self, type_score):
        score = torch.zeros(self.Z.shape[0]).view(-1)
        alpha = solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, self.m, upper=False), upper=True)
        nf = self.kernel.df
        if self.flag_chol:
            inv_Sigma = torch.cholesky_inverse(self.L)
        else:
            Sigma = assemble_cov(self.P, self.V.T, self.S)
            self.L = torch.linalg.cholesky(Sigma)
            inv_Sigma = torch.cholesky_inverse(self.L)
        inv_Sigma_lower_right = inv_Sigma[self.dx:, self.dx:]
        self.inv_Kuu = torch.cholesky_inverse(self.L_Kuu)

        for d in range(self.Z.shape[0]):
            l, r = d * nf, (d + 1) * nf
            alpha_d = alpha[l:r, :]
            Qdd = self.inv_Kuu[l:r, l:r]
            inv_Qdd = torch.inverse(Qdd)
            if type_score == 'full':
                Qdu = self.inv_Kuu[l:r, :]
                _, ld_Q = torch.slogdet(Qdd)
                _, ld_S = torch.slogdet(inv_Sigma_lower_right[l:r, l:r])
                score[d] = ld_S - ld_Q + alpha_d.T @ inv_Qdd @ alpha_d + torch.trace(Qdu @ self.S @ Qdu.T @ inv_Qdd)
            elif type_score == 'mean':
                score[d] = alpha_d.T @ inv_Qdd @ alpha_d

        id_discard = torch.argmin(score)
        l, r = id_discard * nf, (id_discard + 1) * nf

        return id_discard, l, r

    def _get_inv_S_DK(self, L_Kold, L_Knew):
        """Get (S + Delta_K)^-1"""

        # input cholesky decompose of Kold and Knew
        Kold = L_Kold @ L_Kold.T
        Knew = L_Knew @ L_Knew.T
        Delta_K = Kold - Knew

        tmp = solve_tri(L_Knew.T, solve_tri(L_Knew, Delta_K, upper=False), upper=True)
        inv_DK = solve_tri(L_Kold.T, solve_tri(L_Kold, tmp.T, upper=False), upper=True).T
        # 自协方差矩阵， Knew, Kold 均对称，上式可计算出 DK^-1 = Knew^-1 - Kold^-1
        inv_S_DK = inv_DK - inv_DK @ self.S @ inv_DK  # 近似DK极大，inv_DK很小，采用近似

        return inv_S_DK

    def _loss_hp(self):
        """Loss for optimizing GP hyperparameters"""
        # need Knew be calculated in real time

        Knew = self.kernel(self.Z)
        L_Knew = self._chol_decompose(Knew)
        inv_S_DK = self._get_inv_S_DK(self.L_Kuu, L_Knew)
        # inverse of S + \Delta K
        tmp = (self.Kuu - Knew) @ solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, self.S, upper=False), upper=True)
        # calculate inv_Kuu @ S in one step, decrease calculate error
        _, ld = torch.slogdet(Knew + tmp)
        # sign of det, log of abs det
        loss = ld + self.m.T @ inv_S_DK @ self.m

        return loss

    def _get_cov(self, L, isP=True, isV=True, isS=True):
        """Get covariance regarding the joint distribution"""

        dx = self.dx

        P, V, S = None, None, None
        A, B, C = L[:dx, :dx], L[dx:, :dx], L[dx:, dx:]
        if isP:
            P = A @ A.T
        if isV:
            V = A @ B.T
        if isS:
            S = B @ B.T + C @ C.T

        return P, V, S

    def _update_K_chol(self):
        """Update Kuu and its cholesky factor"""

        with torch.no_grad():
            self.Kuu = self.kernel(self.Z)
        self.L_Kuu = self._chol_decompose(self.Kuu)

    def _chol_decompose(self, A):
        """Cholesky decomposition"""
        A = keep_sym(A)
        L = torch.linalg.cholesky(A)

        return L

    @property
    def num_ips(self):
        return self.Z.shape[0]


class RGPSSMa(RGPSSM):
    """Evaluate partial derivative using analytical method"""

    def _get_tmp(self, c, fun_tran):

        x = clone_required_grad(self.x).view(1, -1)
        z, dzdx = self.fun.fun_input(x.view(1, -1), c)

        Vxu, Suu = self.V, self.S
        Ktt = self.kernel(z)
        mt, dfdz, Ktu, ktu, dKdz = self._dfdz(z)
        gam = Ktt - ktu @ Ktu.T
        Vxt = Vxu @ ktu.T
        Stu = ktu @ Suu
        Stt = gam + Stu @ ktu.T

        x0 = 1*x # x0 is not x
        Ft, Ax0, Af = fun_tran(x0.view(1, -1), c, mt.view(1, -1))
        Ft = Ft.view(-1, 1)

        if Ax0 is None: Ax0 = Jacobian(x0, Ft).detach()
        if Af is None: Af = Jacobian(mt, Ft).detach()
        if dzdx is None: dzdx = Jacobian(x, z).detach()
        Ax = Ax0 + Af @ dfdz @ dzdx

        return z.detach(), mt.detach(), Ft.detach(), Ax.detach(), Af.detach(), gam.detach(), \
               Vxu.detach(), Vxt.detach(), Stt.detach(), Stu.detach(), Suu.detach(), ktu.detach()