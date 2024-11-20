# -*- coding:utf-8 -*-
# @author  : Zheng Tengjie
# @time    : 2024/11/20 15:15
# @function: Recursive Gaussian process state space model (RGPSSM)
# @version : V1

import copy
from method import *
import cholup, choldown

def Np2Torch(x, device=None):
    """ndarray to tensor"""
    if x is None:
        return x
    else:
        if isinstance(x, torch.Tensor):
            y = x.to(torch.float32)
        else:
            y = torch.tensor(x, dtype=torch.float32)

        if device is not None:
            return y.to(device)

    return y

def Jacobian(x, y):
    """Get Jacobian matrix dy/dx based on torch
    Arguments:
        x : [tensor]
        y : [tensor]
    Returns:
        J : Jacobian matrxi [tensor]
    """

    y.view(-1)
    J = torch.zeros((y.numel(), x.numel())).to(x.device)
    for ii in range(y.numel()):
        yy = y[ii]
        dyy_dx = torch.autograd.grad(yy, x, create_graph=True)[0]
        J[ii, :] = dyy_dx.view(1, -1)

    return J

class RGPSSM():
    """Recursive Gaussian process state space model (RGPSSM)"""

    def __init__(self, x0, P0, Q, R, fun, kernel, flag_chol=True,
                 jitter=1e-4, budget=50, eps_tol=1e-2, num_opt_hp=0, lr_hp=1e-3, Z=None, Qu=0e-6):
        """
        x0 : prior mean of state [tensor]
        P0 : prior variance of state [tensor]
        Q : process noise covariance [tensor]
        R : measurement noise covariance [tensor]
        fun: model information function
        kernel : kernel of GP [gpytorch.kernels.MultitaskKernel]
        flag_chol : flag to use Cholesky decomposition for joint covariance or not [bool]
        jitter : a positive real number to enhance numerical stability [int]
        budget : the maximum number of pseudo points [int]
        eps_tol : threshold for determination of adding new pseudo points [float]
        num_opt_hp : optimization number for hyperparameter at each correction step [int]
        lr_hp : learning rate for hyperparameter optimization [float]
        Z : inducing inputs [tensor]
        Qu : process noise variance for inducing outputs [float]
        """

        # GP kernel
        self.kernel = kernel

        # Function
        self.fun = fun

        # Parameters of joint distribution
        self.x = Np2Torch(x0)
        self.P = Np2Torch(P0)
        self.Q = Np2Torch(Q)
        self.R = Np2Torch(R)
        self.jitter = jitter
        self.flag_chol = flag_chol
        self.Qu = Qu
        if Z is not None:
            self.init_q(Z)

        # Parameters about optimization
        self.budget = budget
        self.eps_tol = eps_tol
        self.lr_hp = lr_hp
        self.num_hp = num_opt_hp
        self.type_score = 'full'

    def init_q(self, Z):
        """Initialize inducing point distribution"""

        self.Z = Np2Torch(Z)
        self.m = torch.empty((0, 1))
        for z in self.Z:
            m = self.fun.fun_mean(z)
            self.m = torch.cat((self.m, m.view(-1, 1)), dim=0)
        with torch.no_grad():
            self.S = self._add_jiiter(self.kernel(self.Z).to_dense())
        self.V = torch.zeros(self.x.numel(), self.S.shape[0])
        self._update_K_invK()

        if self.flag_chol:
            Sigma1 = torch.cat((self.P, self.V), dim=1)
            Sigma2 = torch.cat((self.V.T, self.S), dim=1)
            Sigma = torch.cat((Sigma1, Sigma2), dim=0)
            self.L = torch.linalg.cholesky(Sigma)


    def predict(self, u=None, fun_tran=None, Q=None):
        """Prediction step
        Arguments:
            u : system input [tensor]
            fun_tran : transition model [function: x, u, f -> F]
            Q : process noise covariance [tensor]
        Returns
            F : predicted state [tensor]
            var_F : variance of predicted state [tensor]
        """

        if u is not None:
            u = Np2Torch(u).view(-1, 1)

        F, var_F, f, var_f = self._moments_propagate_EKF(u, self.fun.fun_tran, Q)
        self._pseudo_points_opt()

        return F, var_F, f, var_f

    def correct(self, y, R=None):
        """Correction step
        Arguments:
            y : measurement [tensor]
            R : measurement noise covariance [tensor]
        Returns:
        """

        if R is None:
            R = self.R
        y = Np2Torch(y)

        # Calculate some quantities
        x = self.x.detach().clone().requires_grad_(True)
        y_pre = self.fun.fun_meas(x)
        Cx = Jacobian(x, y_pre).detach()
        e = y - y_pre.detach()

        if self.flag_chol:
            P, V, S = self.get_cov(self.L)

            H = torch.cat((Cx, torch.zeros((Cx.shape[0], V.shape[1]))), dim=1)
            Gamma = self.keep_sym(Cx @ P @ Cx.T + R)
            rho = torch.linalg.cholesky(Gamma)
            inv_rho_H = torch.linalg.solve(rho, H)
            x = self.L @ (self.L.T @ inv_rho_H.T)
            self.L = choldown.chol_downdate(self.L, x)
            self.P, self.V, self.S = self.get_cov(self.L)

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
            self.x = self.x + P @ q
            self.m = self.m + V.T @ q
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
                Knew = self._add_jiiter(self.kernel(self.Z).to_dense())
            inv_S_DK = self.get_inv_S_DK(self.Kuu, Knew)
            V, S = self.V, self.S
            q = -inv_S_DK @ self.m
            r = -inv_S_DK

            try:
                torch.linalg.cholesky(self.P + V @ r @ V.T)
                torch.linalg.cholesky(self.S + S @ r @ S)
                flag_update = True
            except RuntimeError:
                flag_update = False

            if (not torch.isnan(loss)) and flag_update:
                # Update joint distribution using new hyperparameters

                self.x = self.x + V @ q
                self.m = self.m + S @ q
                self.P = self.P + V @ r @ V.T
                self.V = self.V + V @ r @ S
                self.S = self.S + S @ r @ S
                if self.flag_chol:
                    try:  # Try to Cholesky Decomposition
                        Sigma1 = torch.cat((self.P, self.V), dim=1)
                        Sigma2 = torch.cat((self.V.T, self.S), dim=1)
                        Sigma = torch.cat((Sigma1, Sigma2), dim=0)
                        self.L = torch.linalg.cholesky(self.keep_sym(Sigma))
                        self.P, self.V, self.S = self.get_cov(self.L)
                    except RuntimeError:
                        self.P, self.V, self.S = self.get_cov(self.L)

                self._update_K_invK()
            else:
                # Not update
                self.kernel.load_state_dict(hp0)

            loss = loss.item()

        return loss

    def GP_predict(self, x, u=None):

        x = Np2Torch(x).view(1, -1)
        if u is not None:
            u = Np2Torch(u).view(1, -1)
        z = self.fun.fun_input(x, u).view(1, -1)
        with torch.no_grad():
            Kff = self._add_jiiter(self.kernel(z).to_dense())

        if hasattr(self, 'Z'):
            with torch.no_grad():
                Kfu = self.kernel(z, self.Z).to_dense()
            kfu = torch.linalg.solve(self.Kuu, Kfu.T).T
            f = kfu @ self.m + self.fun.fun_mean(z)
            var_f = Kff - kfu @ Kfu.T + kfu @ self.S @ kfu.T
        else:
            f = self.fun.fun_mean(z)
            var_f = Kff

        return f, var_f

    def _moments_propagate_EKF(self, u=None, fun_tran=None, Q=None):

        # Preparation
        if fun_tran is None:
            fun_tran = self.fun.fun_tran
        if Q is None:
            Q = self.Q

        # Update the moments of joint distribution
        if not hasattr(self, 'Z'):
            # Give an initial inducing point
            x = self.x.detach().clone()
            z = self.fun.fun_input(x, u).view(1, -1)
            ls = self.kernel.data_covar_module.lengthscale.detach().clone().view(1, -1)
            self.init_q(z + ls*torch.randn(1)*0.1)

        z, mt, Ft, Ax, Af, gam, Vxu, Vxt, Stt, Stu, Suu, ktu = self._get_tmp(u, fun_tran)
        self.x = Ft

        if self.flag_chol:
            # Update of Cholesky version
            tr_var = torch.sum(self.kernel.task_covar_module.var).item()
            if torch.trace(gam) > (self.eps_tol + self.jitter) * tr_var:
                # Update with adding point
                v_T = torch.linalg.solve_triangular(self.L, torch.cat((Vxt, Stu.T), dim=0), upper=False)
                beta = torch.linalg.cholesky(gam)
                L1 = torch.cat((self.L, torch.zeros((self.L.shape[0], beta.shape[1]))), dim=1)
                L2 = torch.cat((v_T.T, beta), dim=1)
                L_new = torch.cat((L1, L2), dim=0)
                # Suppose L_new = [L, 0; v, beta]，and then mat_new = [L*L^T, L*v^T;v*L^T, beta*beta^T + v * v^T]
                # Based on this, we can get v and beta，thus getting L_new

                nx, nu, nu_new = Ax.shape[0], Vxu.shape[1], Vxu.shape[1] + Stt.shape[0]
                Phi1 = torch.cat((Ax, torch.zeros((nx, nu)), Af), dim=1)
                Phi2 = torch.cat((torch.zeros((nu_new, nx)), torch.eye(nu_new)),dim=1)
                Phi = torch.cat((Phi1, Phi2), dim=0)
                sqrt_noise = torch.cat((torch.diag(Q).view(-1, 1), self.Qu * torch.ones((nu_new, 1))), dim=0) ** 0.5
                right_factor = torch.cat((L_new.T @ Phi.T, torch.diag_embed(sqrt_noise.view(-1))), dim=0)
                _, R = torch.linalg.qr(right_factor)

                self.L = R.T
                self.m = torch.cat((self.m, mt), dim=0)
                self.P, self.V, self.S = self.get_cov(self.L)
                self.Z = torch.cat((self.Z, z), dim=0)
                self._update_K_invK()
            else:
                # Update without adding point
                nx, nu = Ax.shape[0], Vxu.shape[1]
                Phi1 = torch.cat((Ax, Af@ktu), dim=1)
                Phi2 = torch.cat((torch.zeros((nu, nx)), torch.eye(nu)), dim=1)
                Phi = torch.cat((Phi1, Phi2), dim=0)

                R_bias = torch.eye(nx+nu)
                R_bias[:nx, :nx] = torch.linalg.cholesky(self.Q + Af @ gam @ Af.T).T
                R_bias[nx:, nx:] = torch.eye(nu) * self.Qu**0.5
                tmp = torch.cat((self.L.T @ Phi.T, R_bias), dim=0)
                _, R = torch.linalg.qr(tmp)
                self.L = R.T
                self.P, self.V, self.S = self.get_cov(self.L)
        else:
            # Update of regular version
            self.P = self.keep_sym(Ax @ self.P @ Ax.T + Ax @ Vxt @ Af.T + Af @ Vxt.T @ Ax.T + Af @ Stt @ Af.T + Q)
            self.V = Ax @ Vxu + Af @ Stu

            tr_var = torch.sum(self.kernel.task_covar_module.var).item() * (1 + self.jitter)
            if torch.trace(gam) > (self.eps_tol + self.jitter) * tr_var:
                # Update with adding point
                self.m = torch.cat((self.m, mt), dim=0)
                self.V = torch.cat((self.V, Ax @ Vxt + Af @ Stt), dim=1)
                Sun = torch.cat((Suu, Stu.T), dim=1)
                Stn = torch.cat((Stu, Stt), dim=1)
                self.S = torch.cat((Sun, Stn), dim=0) + torch.eye(self.S.shape[0]) * self.Qu
                self.Z = torch.cat((self.Z, z), dim=0)
                self._update_K_invK()

        return self.x, self.P, mt, Stt

    def _get_tmp(self, u, fun_tran):

        x = self.x.detach().clone().requires_grad_(True)
        z = self.fun.fun_input(x, u).view(1, -1)

        Vxu, Suu = self.V, self.S
        Ktt = self._add_jiiter(self.kernel(z).to_dense())
        Ktu = self.kernel(z, self.Z).to_dense()
        ktu = torch.linalg.solve(self.Kuu, Ktu.T).T
        gam = Ktt - ktu @ Ktu.T
        Vxt = Vxu @ ktu.T
        Stu = ktu @ Suu
        Stt = gam + Stu @ ktu.T

        mt = ktu @ self.m + self.fun.fun_mean(z).view(-1, 1)
        Ft = fun_tran(x, u, mt)
        Ax = Jacobian(x, Ft).detach()
        Af = Jacobian(mt, Ft).detach()

        return z.detach(), mt.detach(), Ft.detach(), Ax, Af, gam.detach(), \
               Vxu.detach(), Vxt.detach(), Stt.detach(), Stu.detach(), Suu.detach(), ktu.detach()

    def _pseudo_points_opt(self, flag_opt=False):
        """Optimize the inducing-point set"""

        if self.Z.shape[0] > self.budget or flag_opt:
            score = torch.zeros(self.Z.shape[0]).view(-1)
            alpha = torch.linalg.solve(self.Kuu, self.m)
            num_task = self.kernel.task_covar_module.var.numel()
            if self.flag_chol:
                inv_Sigma = torch.cholesky_inverse(self.L)
            else:
                Sigma1 = torch.cat((self.P, self.V), dim=1)
                Sigma2 = torch.cat((self.V.T, self.S), dim=1)
                Sigma = torch.cat((Sigma1, Sigma2), dim=0)
                self.L = torch.linalg.cholesky(Sigma)
                inv_Sigma = torch.cholesky_inverse(self.L)
            inv_Sigma_lower_right = inv_Sigma[self.P.shape[0]:, self.P.shape[0]:]

            for d in range(self.Z.shape[0]):
                l, r = d * num_task, (d + 1) * num_task
                alpha_d = alpha[l:r, :]
                Qdd = self.inv_Kuu[l:r, l:r]
                inv_Qdd = torch.inverse(Qdd)
                if self.type_score == 'full':
                    Qdu = self.inv_Kuu[l:r, :]
                    _, ld_Q = torch.slogdet(Qdd)
                    _, ld_S = torch.slogdet(inv_Sigma_lower_right[l:r, l:r])
                    score[d] = ld_S - ld_Q + alpha_d.T @ inv_Qdd @ alpha_d + torch.trace(Qdu @ self.S @ Qdu.T @ inv_Qdd)
                elif self.type_score == 'mean':
                    score[d] = alpha_d.T @ inv_Qdd @ alpha_d

            id_discard = torch.argmin(score)
            l, r = id_discard * num_task, (id_discard + 1) * num_task
            self.Z = self._get_vec_left(self.Z, id_discard, id_discard+1, dim=0)
            self.m = self._get_vec_left(self.m, l, r, dim=0)
            self.V = self._get_vec_left(self.V, l, r, dim=1)
            self.S = self._get_mat_left(self.S, l, r)

            if self.flag_chol:
                if id_discard == self.Z.shape[0]:
                    n = self.x.numel() + self.m.numel()
                    self.L = self.L[:n, :n]
                else:
                    l, r = l + self.x.numel(), r + self.x.numel()
                    A, B, C = self.L[:l, :l], self.L[r:, :l], self.L[r:, r:]
                    c = self.L[r:, l:r]
                    C_new = cholup.chol_update(C, c)
                    L1 = torch.cat((A, torch.zeros((A.shape[0], C_new.shape[1]))), dim=1)
                    L2 = torch.cat((B, C_new), dim=1)
                    self.L = torch.cat((L1, L2), dim=0)
                    """
                    Suppose original Cholesky factor is L = [A, 0, 0;  and L^T = [A^T, a^T, B^T;
                                                             a, b, 0;             0,   b^T, c^T
                                                             B, c, C]             0,   0,   C^T]
                    then the original joint covariance is [A*A^T, A*a^T,       A*B^T; 
                                                           a*A^T, a*a^T+b*b^T, a*B^T+b*c^T
                                                           B*A^T, B*a^T+c*b^T, B*B^T+c*c^T+C*C^T]
                    The new joint covariance = the original matrix with the rows and columns corresponding to the lowercase letters removed, 
                    and the bottom-right block becomes -> B*B^T+C_new*C_new^T（solve C_new based on this）
                    """

                self.P, self.V, self.S = self.get_cov(self.L)

            self._update_K_invK()

    def get_inv_S_DK(self, Kold, Knew):
        Delta_K = Kold - Knew
        tmp = torch.linalg.solve(Knew, Delta_K)
        inv_DK = torch.linalg.solve(Kold, tmp.T).T
        inv_S_DK = inv_DK - inv_DK @ self.S @ inv_DK  # 近似inv_DK很小，发现这个计算更稳定

        return inv_S_DK

    def _loss_hp(self):
        """Loss for optimizing GP hyperparameters"""

        Knew = self._add_jiiter(self.kernel(self.Z).to_dense())
        inv_S_DK = self.get_inv_S_DK(self.Kuu, Knew)
        tmp = (self.Kuu - Knew) @ torch.linalg.solve(self.Kuu, self.S)
        _, ld = torch.slogdet(Knew + tmp)
        loss = ld + self.m.T @ inv_S_DK @ self.m

        return loss

    def get_cov(self, L):
        """Get covariance regarding the joint distribution"""

        nx = self.x.numel()
        A, B, C = L[:nx, :nx], L[nx:, :nx], L[nx:, nx:]
        P = A @ A.T
        V = A @ B.T
        S = B @ B.T + C @ C.T

        return P, V, S

    def _update_K_invK(self):

        with torch.no_grad():
            self.Kuu = self._add_jiiter(self.kernel(self.Z).to_dense())
        self.inv_Kuu = self._inv_chol(self.Kuu)


    def _add_jiiter(self, K, jitter=None):

        if jitter is None:
            jitter = self.jitter

        var = (self.kernel.task_covar_module.var).detach().clone()
        num_task = var.numel()
        num_input = int(K.shape[0] / num_task)
        Jitter = torch.kron(torch.eye(num_input), torch.diag_embed(var) * jitter)
        Knew = K + Jitter

        return Knew

    def _inv_chol(self, A):

        A = self.keep_sym(A)
        L = torch.linalg.cholesky(A)
        inv_A = torch.cholesky_inverse(L)

        return inv_A


    def _get_mat_left(self, mat, l, r):

        mu1 = mat[:l, :l]
        mu2 = mat[:l, r:]
        md1 = mat[r:, :l]
        md2 = mat[r:, r:]

        mu = torch.cat([mu1, mu2], dim=1)
        md = torch.cat([md1, md2], dim=1)
        m = torch.cat([mu, md], dim=0)

        return m

    def _get_vec_left(self, vec, l, r, dim):

        if dim == 0:
            v = torch.cat((vec[:l, :], vec[r:, :]), dim=dim)
        elif dim == 1:
            v = torch.cat((vec[:, :l], vec[:, r:]), dim=dim)

        return v

    def keep_sym(self, A):
        return (A + A.T) * 0.5
