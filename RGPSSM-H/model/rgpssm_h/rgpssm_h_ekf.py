# -*- coding:utf-8 -*-
# @author  : Zheng Tengjie
# @time    : 2025/04/02 15:07
# @function: Recursive Gaussian process state space model for heterogeneous multi-output (RGPSSM-H)
# @version : V1

from typing import Optional, Union, Tuple, List
import copy

import torch
from numpy import ndarray
from torch import Tensor
from torch.linalg import solve_triangular as solve_tri
import numpy as np

from exp_kink.envi.utils.settings import dtype
from ..rgpssm.utils import (Jacobian, ToTensor, clone_required_grad, assemble_cov, assemble_chol, chol_add, insert_row,
                          get_mat_left, get_vec_left, keep_sym, chol_delete, nearest_positive_definite)
from .utils import IModel_H, extract_diag_blocks
from .kernel import IKerH
from ..rgpssm.rgpssm_u import RGPSSM_U
import cholup, choldown


class RGPSSM_H_EKF(RGPSSM_U):
    """Recursive Gaussian process state space model for heterogeneous input (RGPSSM-H) with EKF-based moment matching"""

    def __init__(self,
                 x0: Union[ndarray, Tensor],
                 P0: Union[ndarray, Tensor],
                 Q: Union[ndarray, Tensor],
                 R: Union[ndarray, Tensor],
                 fun: IModel_H, kernel: IKerH,
                 flag_chol: bool = True,
                 budget: int = 50, eps_tol: float = 1e-2,
                 num_opt_hp: int = 0, lr_hp: float = 1e-3,
                 type_score: str = 'full',
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
        self.dx = self.x.numel()

        # Parameters about optimization
        self.budget = budget
        self.eps_tol = eps_tol
        self.lr_hp = lr_hp
        self.num_hp = num_opt_hp
        self.type_score = type_score

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


            flag_update = False
            if (not torch.isnan(loss)):
                # Update joint distribution using new hyperparameters
                try:
                    # Update joint distribution
                    with torch.no_grad():
                        Knew = self.kernel(self.Z, self.Id_Z, flag_sort=True)
                        L_Knew = self._chol_decompose(Knew)
                    inv_S_DK = self._get_inv_S_DK(self.L_Kuu, L_Knew)

                    V, S = self.V, self.S
                    q = -inv_S_DK @ self.m
                    r = -inv_S_DK

                    P = self.P + V @ r @ V.T
                    V = self.V + V @ r @ S
                    S = self.S + S @ r @ S
                    Sigma = assemble_cov(S, V, P)
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

    def GP_predict(self, x: Optional[Union[ndarray, Tensor]], c: Optional[Union[ndarray, Tensor]] = None) \
            -> Tuple[Tensor, Tensor]:
        """Get mean and variance of GP prediction
        Args:
            x: system state (dx, 1)
            c: system input (nc, 1)
        Returns:
            f: mean of predicted GP (nf, 1)
            var_f: variance of predicted GP (nf, nf)
        """

        x = ToTensor(x)
        c = ToTensor(c)
        z, _ = self.fun.fun_input(x, c)
        z = z.reshape(-1, z.shape[-1])
        n = int(z.shape[0] / self.kernel.id_z.numel())
        id_z = self.kernel.id_z.repeat(n)
        with torch.no_grad():
            Kff = self.kernel(z, id_z, flag_sort=True)

        if hasattr(self, 'Z'):
            with torch.no_grad():
                Kfu = self.kernel(z, id_z, self.Z, self.Id_Z, flag_sort=True)
            kfu = torch.cholesky_solve(Kfu.T, self.L_Kuu).T
            f = kfu @ self.m
            var_f = Kff - kfu @ Kfu.T + kfu @ self.S @ kfu.T
        else:
            f = torch.zeros((self.kernel.df, 1))
            var_f = Kff

            f = f.repeat(n, 1)
            var_f = torch.block_diag(*[var_f for _ in range(n)])

        return f, var_f

    def _init_q_hi(self, Z: Tensor, Id: Union[Tensor, List[int]]):
        """Initialize inducing point distribution
        Args:
            Z: N elements and each element is a GP input [(1, nz)]
            Id: each elements represent the id of a dimension of f (N,)
        """
        # V presents the cov between x and inducing outputs u, namely, Sxu
        # S presents the var of inducing points u, namely, Suu
        # m is the output mean of u, namely, mu

        self.Z = Z
        self.Id_Z = ToTensor(Id, dtype=torch.int)       # the idx of GP for inputs Z
        self.Id_f = self.kernel.idZ_to_idf(self.Id_Z)   # the idx of GP for output u

        # nu = torch.sum(self.kernel.df[self.Id_Z]).item()
        nu = self.Id_f.numel()
        self.m = torch.zeros((nu, 1))
        with torch.no_grad():
            self.S = self.kernel(self.Z, self.Id_Z, flag_sort=True)
        self.V = torch.zeros(self.dx, nu)
        self._update_K_chol()

        if self.flag_chol:
            Sigma = assemble_cov(self.S, self.V, self.P)
            self.L = torch.linalg.cholesky(Sigma)

    def _moments_propagate_EKF(self, c=None, fun_tran=None, Q=None):

        # Preparation
        if fun_tran is None: fun_tran = self.fun.fun_tran
        Q = self.Q  if Q is None else ToTensor(Q)
        c = ToTensor(c, view=(1, -1))

        # Update the moments of joint distribution
        if not hasattr(self, 'Z'):
            Z_init = self._get_Z_init(c)
            self._init_q_hi(Z_init, self.kernel.id_z)

        z, mt, Ft, Ax, Af, gam, Vxu, Vxt, Stt, Stu, Suu, ktu = self._get_tmp(c, fun_tran)
        self.x = Ft

        if self.flag_chol:
            # Update of Cholesky version
            flag_add = self._find_add(gam)
            if flag_add.any():
                L_new, A_unew = self._add_point_EKF(flag_add, z, Vxt, Stu, Stt, Af, ktu, mt)
            else:
                L_new = self.L
                A_unew = Af @ ktu

            flag_no_add = ~flag_add
            A_no_add = Af[:, flag_no_add]
            gam_no_add = gam[flag_no_add, :][:, flag_no_add]
            Q_gp = A_no_add @ gam_no_add @ A_no_add.T
            P = keep_sym(Ax @ self.P @ Ax.T + Ax @ Vxt @ Af.T + Af @ Vxt.T @ Ax.T + Af @ Stt @ Af.T + Q + Q_gp)
            V = Ax @ self.V + A_unew @ self.S
            L_S = L_new[:-self.dx, :-self.dx]

            rho = torch.linalg.solve_triangular(L_S, V.T, upper=False).T
            # tmp = nearest_positive_definite(P - rho @ rho.T)
            tmp = P - rho @ rho.T
            beta = torch.linalg.cholesky(tmp)
            self.L = assemble_chol([[L_S], [rho, beta]])
            self.P, self.V, self.S = self._get_cov(self.L)

        else:
            raise NotImplementedError

        return self.x, self.P, mt, Stt

    def _find_add(self, gam):
        """
        Args:
            gam: GP prediction prior variance (dh, dh)
        Returns:
            # flag_add: Tensor (n_gp,) indicates which new inducing points should be added to the GP model. 
        """
        flag_add = []
        for ii in self.kernel.id_z:
            gam_i = gam[self.kernel.id_f == ii, :][:, self.kernel.id_f == ii]
            Kuu_i = self.Kuu[self.Id_f == ii, :][:, self.Id_f == ii]

            K_diag = torch.diag(Kuu_i).reshape(-1, self.kernel.df[ii])  # (nb, dfi)
            tr_max = torch.sum(K_diag, dim=-1).max()
            tr_gam = torch.trace(gam_i) if gam_i.dim() == 2 else torch.sum(gam_i)

            flag = tr_gam.item() > tr_max.item() * self.eps_tol
            flag_add = flag_add + [flag] * self.kernel.df[ii]
        flag_add = ToTensor(flag_add, dtype=torch.bool)

        return flag_add

    def _add_point_EKF(self, flag_add, z, Vxt, Stu, Stt, Af, ktu, mt):
        """Update the relevant variables after adding points for EKF-based moment propagation
        Args:
            flag_add: Tensor (n_gp,) indicates which new inducing points should be added to the GP model. 
        Returns:
            L_new: Cholesky factor of (u,x) after adding points (dux, dux)
            A_unew: dF/du_new (dx, du)
        """
        L_new = self.L
        S_ux_a = torch.cat((Stu.T, Vxt), dim=0)

        A_unew = Af[:, flag_add == False] @ ktu[flag_add == False, :]

        for i in range(flag_add.numel()):
            if flag_add[i]:
                idx_z = (self.Id_Z <=i).sum()
                idx_u = (self.Id_f <=i).sum()

                bool_f = self.kernel.id_f == i
                L_new = chol_add(L_new, S_ux_a[:, bool_f], Stt[bool_f, :][:, bool_f], idx_u)
                S_ux_a = insert_row(S_ux_a, Stt[bool_f, :], idx_u)

                self.Z = insert_row(self.Z, z[[i], :], idx_z)
                id_add = torch.tensor(i).view(-1)
                self.Id_Z = insert_row(self.Id_Z, id_add, idx_z)
                self.Id_f = insert_row(self.Id_f, self.kernel.idZ_to_idf(id_add), idx_u)
                self.m = insert_row(self.m, mt[bool_f, :], idx_u)

                A_unew = insert_row(A_unew.T, Af[:, [i]].T, idx_u).T
        self._update_K_chol()
        self.P, self.V, self.S = self._get_cov(L_new)

        return L_new, A_unew

    def _get_tmp(self, c, fun_tran):

        x = clone_required_grad(self.x)
        c = clone_required_grad(c, view=(1, -1))
        z, dzdx = self.fun.fun_input(x.view(1, -1), c)
        z = z.squeeze(0)
        dzdx = None if dzdx is None else dzdx.squeeze(0)
        id_z = self.kernel.id_z

        Vxu, Suu = self.V, self.S
        Ktt = self.kernel(z, id_z, flag_sort=True)
        Ktu = self.kernel(z, id_z, self.Z, self.Id_Z, flag_sort=True)
        ktu = torch.cholesky_solve(Ktu.T, self.L_Kuu).T
        gam = Ktt - ktu @ Ktu.T
        Vxt = Vxu @ ktu.T
        Stu = ktu @ Suu
        Stt = gam + Stu @ ktu.T

        mt = ktu @ self.m
        x0 = 1 * x              # x0 is not x
        Ft, Ax0, Af = fun_tran(x0.view(1, -1), c, mt.view(1, -1))
        Ft = Ft.view(-1, 1)

        if Ax0 is None: Ax0 = Jacobian(x0, Ft).detach()
        if Af is None: Af = Jacobian(mt, Ft).detach()
        z_flat = z.view(1, -1)
        if dzdx is None:
            dzdx = Jacobian(x, z_flat).detach()
        dfdz = Jacobian(z_flat, mt)
        Ax = Ax0 + Af @ dfdz @ dzdx

        return z.detach(), mt.detach(), Ft.detach(), Ax.detach(), Af.detach(), gam.detach(), \
            Vxu.detach(), Vxt.detach(), Stt.detach(), Stu.detach(), Suu.detach(), ktu.detach()

    def _inducing_points_opt(self, flag_opt=False, type_score=None, n_discard=None):
        """Optimize the inducing-point set
        Args:
            flag_opt: whether to adjust the inducing points
            type_score: type of the score for evaluating importance
        """

        n_discard = self.kernel.id_z.numel() if n_discard is None else n_discard

        if type_score is None:
            type_score = self.type_score

        if len(self.Z) > self.budget or flag_opt:
            if type_score == "oldest":
                Id_discard = torch.arange(0, n_discard, dtype=torch.int)
                L, R = [], []
                l = 0
                for d in Id_discard:
                    r = l + self.kernel.df[self.Id_Z[d]]
                    L.append(l)
                    R.append(r)
                    l = r
            else:
                Id_discard, L, R = self._get_score(type_score, n_discard)
            # Id_discard: Tensor (n_gp,) the indices of inducing points to discard
            # L: List: left index of the block of m, V, S to discard
            # R: List: right index of the block of m, V, S to discard

            self._delete_points(Id_discard, L, R)

    def prune_redundant_points(self):
        """optimize the inducing points by deleting
         the points too close
        Note: the current version only suitable for df=1 for all GPs
        """

        Id_discard = []
        idx_all = torch.arange(0, self.Z.shape[0])
        for k in range(self.kernel.id_z.numel()):
            bool_k = self.Id_Z == k
            idx_k = idx_all[bool_k]
            Kuu_k = self.Kuu[bool_k, :][:, bool_k]
            L_Kuu_k = self.L_Kuu[bool_k, :][:, bool_k]
            Q_k = torch.cholesky_inverse(L_Kuu_k)
            gam = 1 / torch.diag(Q_k)
            tr_max = torch.diag(Kuu_k).max()
            # novelty = gam / (tr_max * self.eps_tol * 0.05 * self.kernel.dz[k])
            novelty = gam / (tr_max * self.eps_tol * 0.1)
            novelty = torch.cat((novelty[:-2], novelty[[-1]]))
            idx_k = torch.cat((idx_k[:-2], idx_k[[-1]]))
            if (novelty < 1).any() and len(idx_k) > 1:
                idx_min = torch.argmin(novelty)
                Id_discard.append(idx_k[idx_min])

        if len(Id_discard) > 0:
            Id_discard = ToTensor(Id_discard, dtype=torch.int)
            L = Id_discard
            R = L + 1

            self._delete_points(Id_discard, L, R)
            # print(f'delete inducing points: {Id_discard} / {idx_all.numel()}')
        else:
            pass
            # print('no delete inducing points!')



    def _delete_points(self, Id_discard, L, R):
        """Delete inducing points
        Args:
            Id_discard: List/Tensor (n_gp,) the indices of inducing points to discard
            L: List: left index of the block of m, V, S to discard
            R: List: right index of the block of m, V, S to discard
        """
        Id_discard = ToTensor(Id_discard, view=(-1,), dtype=torch.int)
        L, R = ToTensor(L, dtype=torch.int), ToTensor(R, dtype=torch.int)
        sort = torch.sort(Id_discard)
        Id_discard = sort.values
        L = L[sort.indices]
        R = R[sort.indices]

        for ii in range(len(Id_discard)):
            idx = -ii - 1
            id_discard = Id_discard[idx]
            l, r = L[idx], R[idx]

            self.Z = get_vec_left(self.Z, id_discard, id_discard + 1, dim=0)
            self.Id_Z = get_vec_left(self.Id_Z.view(-1, 1), id_discard, id_discard + 1, dim=0).view(-1)
            self.Id_f = get_vec_left(self.Id_f.view(-1, 1), l, r, dim=0).view(-1)
            self.m = get_vec_left(self.m, l, r, dim=0)
            self.V = get_vec_left(self.V, l, r, dim=1)
            self.S = get_mat_left(self.S, l, r)

            if self.flag_chol:
                self.L = chol_delete(self.L, l, r)

        if self.flag_chol: self.P, self.V, self.S = self._get_cov(self.L)
        self._update_K_chol()

    def _get_score(self, type_score, n_discard):
        """
        Args:
            type_score: type of the score for evaluating importance
        Returns:
            Id_discard: Tensor (n_gp,) the indices of inducing points to discard
            L: List: left index of the block of m, V, S to discard
            R: List: right index of the block of m, V, S to discard
        """

        score = torch.zeros(len(self.Z)).view(-1)
        alpha =  torch.cholesky_solve(self.m, self.L_Kuu)
        if self.flag_chol:
            inv_Sigma = torch.cholesky_inverse(self.L)
        else:
            Sigma = assemble_cov(self.S, self.V, self.P)
            self.L = torch.linalg.cholesky(Sigma)
            inv_Sigma = torch.cholesky_inverse(self.L)
        inv_Sigma_u = inv_Sigma[:-self.dx, :-self.dx]
        self.inv_Kuu = torch.cholesky_inverse(self.L_Kuu)

        if (self.kernel.df == 1).all():
            q_dd = torch.diag(self.inv_Kuu).view(-1)
            omg_dd = torch.diag(inv_Sigma_u).view(-1)

            sqrt_qdd = q_dd**0.5
            alpha_ = alpha.view(-1) / sqrt_qdd
            D1 = alpha_**2

            if type_score == 'full':
                tmp = self.S @ (self.inv_Kuu / q_dd.view(1, -1))
                D2 = torch.sum(self.inv_Kuu * tmp.T, dim=1)
                D3 = torch.log(omg_dd / q_dd)
                score = D1 + D2 + D3
            elif type_score == 'mean':
                score = D1

            l_list = torch.arange(0, len(self.Z), dtype=torch.int32)
            r_list = l_list + 1
        else:
            l_list, r_list = [], []
            l = 0
            for d in range(len(self.Z)):
                r = l + self.kernel.df[self.Id_Z[d]]
                # l, r = d * nf, (d + 1) * nf
                alpha_d = alpha[l:r, :]
                Qdd = self.inv_Kuu[l:r, l:r]
                inv_Qdd = torch.linalg.inv(Qdd)
                if type_score == 'full':
                    Qdu = self.inv_Kuu[l:r, :]
                    _, ld_Q = torch.slogdet(Qdd)
                    _, ld_S = torch.slogdet(inv_Sigma_u[l:r, l:r])
                    score[d] = ld_S - ld_Q + alpha_d.T @ inv_Qdd @ alpha_d + torch.trace(Qdu @ self.S @ Qdu.T @ inv_Qdd)
                elif type_score == 'mean':
                    score[d] = alpha_d.T @ inv_Qdd @ alpha_d

                l_list.append(l)
                r_list.append(r)
                l = r

        score[-len(self.kernel.df):] = torch.inf
        Id_discard = torch.topk(score, k=n_discard, largest=False).indices  # first k smallest elements

        # Avoid discarding all the inducing points for a function dimension
        new_Id_discard = []
        nu_lst = [torch.sum(self.Id_Z == d) for d in range(self.kernel.nf)]
        for d in Id_discard:
            if nu_lst[self.Id_Z[d]] > 1:
                new_Id_discard.append(d)
                nu_lst[self.Id_Z[d]] -= 1
        Id_discard = new_Id_discard

        L, R = [l_list[d] for d in Id_discard], [r_list[d] for d in Id_discard]

        return Id_discard, L, R

    def _get_inv_S_DK(self, L_Kold, L_Knew):
        """Get (S + Delta_K)^-1"""

        # input cholesky decompose of Kold and Knew
        Kold = L_Kold @ L_Kold.T
        Knew = L_Knew @ L_Knew.T
        Delta_K = Kold - Knew

        # tmp = solve_tri(L_Knew.T, solve_tri(L_Knew, Delta_K, upper=False), upper=True)
        # inv_DK = solve_tri(L_Kold.T, solve_tri(L_Kold, tmp.T, upper=False), upper=True).T
        tmp = torch.cholesky_solve(Delta_K, L_Knew)
        inv_DK = torch.cholesky_solve(tmp.T, L_Kold).T
        # 自协方差矩阵， Knew, Kold 均对称，上式可计算出 DK^-1 = Knew^-1 - Kold^-1
        inv_S_DK = inv_DK - inv_DK @ self.S @ inv_DK  # 近似DK极大，inv_DK很小，采用近似

        return inv_S_DK

    def _loss_hp(self):
        """Loss for optimizing GP hyperparameters"""
        # need Knew be calculated in real time

        Knew = self.kernel(self.Z, self.Id_Z, flag_sort=True)
        L_Knew = self._chol_decompose(Knew)
        inv_S_DK = self._get_inv_S_DK(self.L_Kuu, L_Knew)
        # inverse of S + \Delta K
        # tmp = (self.Kuu - Knew) @ solve_tri(self.L_Kuu.T, solve_tri(self.L_Kuu, self.S, upper=False), upper=True)
        tmp = (self.Kuu - Knew) @ torch.cholesky_solve(self.S, self.L_Kuu)
        # calculate inv_Kuu @ S in one step, decrease calculate error
        _, ld = torch.slogdet(Knew + tmp)
        # sign of det, log of abs det
        loss = ld + self.m.T @ inv_S_DK @ self.m

        return loss

    def _get_Z_init(self, c):
        z, _ = self.fun.fun_input(self.x.view(1, -1), c)
        z = z.squeeze(0)

        ls = self.kernel.ls
        Z = []
        for ii in range(len(ls)):
            pert = ls[ii].detach().clone().view(-1) * torch.randn(1) * 0.1
            zi = self.kernel.z_fill(self.kernel.z_extract(z[[ii], :], ii) + pert)
            Z.append(zi)

        return torch.cat(Z, dim=0)

    def _update_K_chol(self):
        """Update Kuu and its cholesky factor"""

        with torch.no_grad():
            self.Kuu = self.kernel(self.Z, self.Id_Z, flag_sort=True)
        self.L_Kuu = self._chol_decompose(self.Kuu)

    @property
    def num_ips(self):
        """Get number of inducing points
        Returns: List with element denoting the number of inducing points for each GP
        """

        Df = len(self.kernel.df)
        N = [torch.sum(self.Id_Z == ii).item() for ii in range(Df)]

        return N


