from ..rgpssm.utils import nearest_positive_definite, chol_near_singular
from .rgpssm_h_ukf import *
from torch import cholesky_solve as chol_solve
import cholup, choldown

class RGPSSM_H_ADF(RGPSSM_H_UKF):
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
                 type_filter: str = 'UKF'):
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
        type_filter: type of filter, "EKF", "UKF"， "ADF"
        """

        super().__init__(x0, P0, Q, R, fun, kernel, flag_chol, budget, eps_tol,
                       num_opt_hp, lr_hp, type_score, type_filter)


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

        if self.type_filter == 'EKF':
            F, var_F, f, var_f = self._moments_propagate_EKF(c, fun_tran, Q)
        elif self.type_filter == 'UKF':
            F, var_F, f, var_f = self._moments_propagate_UKF(c, fun_tran, Q)
        elif self.type_filter == 'ADF':
            F, var_F, f, var_f = self._moments_propagate_ADF(c, fun_tran, Q)
        else:
            raise ValueError('type_filter must be "EKF" or "UKF" or "ADF"')

        self._inducing_points_opt()

        return F, var_F, f, var_f

    def _moments_propagate_ADF(self, c=None, fun_tran=None, Q=None):
        # Preparation
        if fun_tran is None: fun_tran = self.fun.fun_tran
        Q = self.Q if Q is None else ToTensor(Q)
        c = ToTensor(c, view=(1, -1))

        # Update the moments of joint distribution
        if not hasattr(self, 'Z'):
            Z_init = self._get_Z_init(c)
            self._init_q_hi(Z_init, self.kernel.id_z)

        z, mt, gam, Vxt, Stt, Stu, Ktt = self._get_tmp_UKF(c)

        if self.flag_chol:
            # Update of Cholesky version
            # Add points
            flag_add = self._find_add(gam)
            L_new = self._add_point_UKF(flag_add, z, Vxt, Stu, Stt, mt) if flag_add.any() else self.L

            # predict h (Denote (h, x_old) as w)
            mh, Sww, Swu, Shu, L_w = self._predict_h_ADF(L_new, c)

            # Get UKF param
            dhx = self.dx + self.kernel.nf
            eta, Wm, Wc = ukf.param(dhx, self.alpha_ukf, self.beta_ukf)

            # Samples
            w_m = torch.cat((mh, self.x), dim=0) # mean of w
            w_s = ukf.sample(w_m, L_w, eta)             # sample of w

            # Predict x
            xnew_s = self._get_state_sample_ADF(w_s.T, c).T

            # Update L
            self.L, self.x = self._update_x_L_ADF(L_w, L_new, Shu, xnew_s, eta, Wm, Wc, Q)

            # Covariance
            self.P, self.V, self.S = self._get_cov(self.L)


        return self.x, self.P, mt, Stt

    def _predict_h_ADF(self, L, c):
        """Latene function prediction based on ADF
        Args:
            L：Cholesky factor of (u, x) (dux, dux)
            c: control input (1, dc)
        Returns:
            mh, Sww, Swu, Shu
            L_w： Cholesky factor of w = (h,x)
        """

        # Calculate mean of z and Jacobian C = dz/dx
        mz_lst, C_lst = self._get_mz_dzdx_ADF(c)
        ls, var = [l.detach() for l in self.kernel.ls], [v.detach() for v in self.kernel.var]

        # Prepare some necessary moments, where we denote (u, x) as o, and (h,x) as w
        Sxx, Sxu, Suu = self._get_cov(L)
        mo = torch.cat((self.m, self.x), dim=0)  # (do, 1)
        Sux = Sxu.T
        Sxo = torch.cat((Sxu, Sxx), dim=1)
        Suo = torch.cat((Suu, Sux), dim=1)

        # Get moments
        mh, Sho = self._get_mh_Sho_ADF(mz_lst, C_lst, ls, var, mo, Sux, Suo, Sxo)
        Shh = self._get_Shh_ADF(mz_lst, C_lst, ls, var, mh, Sux, Suu)

        # Arange
        Shx = Sho[:, -self.dx:]
        Shu = Sho[:, :-self.dx]
        Sww = assemble_cov(Shh, Shx.T, Sxx)
        Swu = torch.cat((Shu, Sxu), dim=0)

        Sww = nearest_positive_definite(Sww)
        L_w = torch.linalg.cholesky(Sww)

        return mh, Sww, Swu, Shu, L_w


    def _get_mh_Sho_ADF(self, mz_lst, C_lst, ls, var, mo, Sux, Suo, Sxo):
        """Get mh and Sho based on ADF-based moment propagation"""

        mh = torch.zeros((self.kernel.nf, 1))
        Sho_1 = torch.zeros((self.kernel.nf, self.dx + self.m.shape[0])) # first component to calculate Sho

        for k in range(self.kernel.nf):
            idx = self.Id_Z == k
            mz = mz_lst[k]                                      # (1, dz)
            Zk = self.kernel.z_extract(self.Z[idx, :], k)       # (n, dz)
            L_Kuu_k = self.L_Kuu[idx, :][:, idx]                # (n, n)
            Ck = C_lst[k]

            mv = torch.cholesky_solve(self.m[idx, :], L_Kuu_k)  # (n, 1)
            Pz = Ck @ self.P @ Ck.T                             # (dz, dz)
            Svz = torch.cholesky_solve(Sux[idx, :] @ Ck.T, L_Kuu_k)     # (n, dz)
            Svo = torch.cholesky_solve(Suo[idx, :], L_Kuu_k)            # (n, do)
            Szo = Ck @ Sxo                                      # (dz, do)

            ls_k = ls[k].detach().view(-1)
            ls_k_square = ls_k**2
            var_k = var[k].detach().item()
            Lam = torch.diag_embed(ls_k_square)
            inv_Lam = torch.diag_embed(1 / ls_k_square)
            Pz_Lam = Pz + Lam
            L_Pz_Lam = torch.linalg.cholesky(Pz_Lam)

            ez = Zk - mz.view(1, -1)                                      # (n, dz)
            Av = torch.cholesky_solve(Svz.T, L_Pz_Lam).T                    # (n, dz)
            Ao = torch.cholesky_solve(Szo, L_Pz_Lam).T                      # (do, dz)

            det = torch.linalg.det(torch.eye(Pz.shape[0]) + Pz @ inv_Lam)   # (1,)
            Kz = self.kernel.ker_single_base(Zk, mz, Pz_Lam)                # (n, 1)
            bz = var_k * det**(-0.5) * Kz                                   # (n, 1)

            mv_z = mv + torch.sum(Av * ez, dim=1).unsqueeze(-1)             # (n, 1)
            mo_z = mo + Ao @ ez.T                                           # (do, n)
            Svo_z = Svo - Av @ Szo                                          # (n, do)

            mh[k, :] = (bz * mv_z).sum(dim=0)
            Sho_1[k, :] = (bz * (Svo_z + mv_z * mo_z.T)).sum(dim=0)

        Sho = Sho_1 - mh @ mo.T

        return mh, Sho

    def _get_Shh_ADF(self, mz_lst, C_lst, ls, var, mh, Sux, Suu):
        """Get Shh based on ADF-based moment propagation"""

        Shh = torch.zeros((self.kernel.nf, self.kernel.nf))

        for k in range(self.kernel.nf):
            for l in range(k, self.kernel.nf): # Shh is a symmetric matrix so we only need to calculate half of the matrix

                Ckl, Zkl, mZ, PZ, PZ_Lam, ls_kl_2, var_k, var_l, n_k, n_l, dZ \
                    = self._get_tmp_Shh_ADF(mz_lst, C_lst, ls, var, k, l)

                idx_k = self.Id_Z == k
                idx_l = self.Id_Z == l
                L_Kuu_k = self.L_Kuu[idx_k, :][:, idx_k]                        # (nk, nk)
                L_Kuu_l = self.L_Kuu[idx_l, :][:, idx_l]                        # (nl, nl)

                m_vk = chol_solve(self.m[idx_k, :], L_Kuu_k)                    # (nk, 1)
                m_vl = chol_solve(self.m[idx_l, :], L_Kuu_l)                    # (nl, 1)
                S_vkZ = chol_solve(Sux[idx_k, :] @ Ckl.T, L_Kuu_k)              # (nk, dZ)
                S_vlZ = chol_solve(Sux[idx_l, :] @ Ckl.T, L_Kuu_l)              # (nl, dZ)
                S_vkul = chol_solve(Suu[idx_k, :][:, idx_l], L_Kuu_k)
                S_vkvl = chol_solve(S_vkul.T, L_Kuu_l).T                        # (nk, nl)

                # bZ
                inv_Lam = torch.diag_embed(1/ls_kl_2)
                det = torch.linalg.det(torch.eye(PZ.shape[0]) + PZ @ inv_Lam)       # (1,)
                KZ = self.kernel.ker_single_base(Zkl, mZ.view(1, -1), PZ_Lam)       # (nk, nl, 1)
                KZ = KZ.squeeze(-1)                                                 # (nk, nl)
                bZ = var_k * var_l * det**(-0.5) * KZ                               # (nk, nl)

                # S_vkvl, m_vk, m_vl
                L_PZ_Lam = torch.linalg.cholesky(PZ_Lam)                            # (dZ, dZ)
                A_vk = chol_solve(S_vkZ.T, L_PZ_Lam).T                              # (nk, dZ)
                A_vl = chol_solve(S_vlZ.T, L_PZ_Lam).T                              # (nk, dZ)
                # A_vk = torch.linalg.solve(PZ_Lam, S_vkZ.T).T                        # (nk, dZ)
                # A_vl = torch.linalg.solve(PZ_Lam, S_vlZ.T).T                        # (nl, dZ)
                A_vk_expd = A_vk.unsqueeze(-2).expand(n_k, n_l, dZ)                 # (nk, nl, dZ)
                A_vl_expd = A_vl.unsqueeze(-3).expand(n_k, n_l, dZ)                 # (nk, nl, dZ)
                e = Zkl - mZ.view(1, -1)                                            # (nk, nl, dZ)

                m_vk_expd = m_vk.expand(n_k, n_l)                   # (nk, nl)
                m_vl_expd = m_vl.T.expand(n_k, n_l)                 # (nk, nl)

                m_vk_Z = m_vk_expd + (A_vk_expd * e).sum(dim=-1)    # (nk, nl)
                m_vl_Z = m_vl_expd + (A_vl_expd * e).sum(dim=-1)    # (nk, nl)
                S_vkvl_Z = S_vkvl - A_vk @ S_vlZ.T                  # (nk, nl)

                if k == l:
                    Qk = torch.cholesky_inverse(L_Kuu_k)
                    Shh_1 = -torch.sum(bZ * Qk)
                else:
                    Shh_1 = 0
                Shh_2 = torch.sum(bZ * (S_vkvl_Z + m_vk_Z * m_vl_Z))

                Shh[k, l] = Shh_1 + Shh_2

        Shh = Shh + Shh.T - torch.diag(torch.diag(Shh)) #  Get the full matrix
        Khh = torch.diag_embed((1 + self.kernel.jitter) * (torch.cat(var, dim=0).detach().view(-1)))
        Shh = Shh + Khh - mh @ mh.T

        return Shh

    def _get_tmp_Shh_ADF(self, mz_lst, C_lst, ls, var, k, l):
        """Get tmp for calculating Shh"""
        mz_k = mz_lst[k]                                            # (1, dz_k)
        mz_l = mz_lst[l]                                            # (1, dz_l)
        mZ = torch.cat((mz_k, mz_l), dim=-1)                 # (1, dZ)
        Zk = self.kernel.z_extract(self.Z[self.Id_Z == k, :], k)    # (nk, dz_k)
        Zl = self.kernel.z_extract(self.Z[self.Id_Z == l, :], l)    # (nl, dz_l)

        n_k, n_l, dZ = Zk.shape[0], Zl.shape[0], mz_k.shape[-1] + mz_l.shape[-1]

        Zk_expd = Zk.unsqueeze(-2).expand(n_k, n_l, Zk.shape[-1])  # (nk, nl, dz_k)
        Zl_expd = Zl.unsqueeze(-3).expand(n_k, n_l, Zl.shape[-1])  # (nk, nl, dz_l)
        Zkl = torch.cat((Zk_expd, Zl_expd), dim=-1)  # (nk, nl, dZ)

        ls_k = ls[k].detach().view(-1)                  # (dz_k)
        ls_l = ls[l].detach().view(-1)                  # (dz_l)
        ls_kl = torch.cat((ls_k, ls_l), dim=0)   # (dZ_kl)
        ls_kl_2 = ls_kl ** 2                            # (dZ_kl)
        var_k = var[k].detach().item()
        var_l = var[l].detach().item()

        Ck, Cl = C_lst[k], C_lst[l]
        Ckl = torch.cat((Ck, Cl), dim=0)
        PZ = Ckl @ self.P @ Ckl.T
        PZ_Lam = PZ + torch.diag_embed(ls_kl_2)

        return Ckl, Zkl, mZ, PZ, PZ_Lam, ls_kl_2, var_k, var_l, n_k, n_l, dZ

    def _get_mz_dzdx_ADF(self, c):
        """Get mean of z^k and Jacobian dz^k/dx
        Args:
            mz_lst: List whose element is mean of z^k
            C_lst: List whose element if dz^k/dx
        """
        x = clone_required_grad(self.x)
        c = clone_required_grad(c, view=(1, -1))
        z, _ = self.fun.fun_input(x.view(1, -1), c)
        z = z.squeeze(0)
        mz_lst, C_lst = [], []
        for i in range(z.shape[0]):
            zi = self.kernel.z_extract(z[[i], :], i)
            C_lst.append(Jacobian(x, zi).detach())
            mz_lst.append(zi.detach())

        return mz_lst, C_lst

    def _update_x_L_ADF(self, Lw, L_ux, Shu, xnew_s, eta, Wm, Wc, Qx):
        """Update the Cholesky factor of (u, x) for the ADF-based moment propagation
        Args:
            L_w_xnew: Cholesky factor of (w, xnew)
            L_ux: Cholesky factor of (u, x)
            Shu: covariance of (h,u)
            xnew_s: samples of the predicted x (dx, n)
            Wm: weight for the mean evaluation (n,)
            Wc: weight for the covariance evaluation (n,)
        Returns:
            L: Cholesky factor of (u, xnew)
        """

        # Get blocks
        A, B = L_ux[:-self.dx, :-self.dx], L_ux[-self.dx:, :-self.dx]  # (du, du), (dx, du)

        # Get the left lower block of L
        tmp1 = torch.cat((solve_tri(A, Shu.T, upper=False), B.T), dim=-1)       # (du, dw)
        x_mean = (xnew_s * Wm).sum(dim=-1).view(-1, 1)  # (dx, 1)
        e_x_s = (xnew_s - x_mean)                       # (dx, n)
        We_x_s = e_x_s * Wc                             # (dx, n)
        dw = int((xnew_s.shape[1]-1)/2)
        tmp2 = eta * solve_tri(Lw.T, (We_x_s[:, 1:1+dw] - We_x_s[:, 1+dw:]).T, upper=True)  # (dw, dx)
        low_1 = (tmp1 @ tmp2).T                                                             # (dx, du)

        # Get the right lower block of L
        sqrt_Q = Qx**0.5
        xnew_e = (Wc[1:]) ** 0.5 * (xnew_s[:, 1:] - x_mean)  # (dx, n-1)
        tmp = torch.cat((xnew_e.T, sqrt_Q.T), dim=0)
        _, R = torch.linalg.qr(tmp)
        low2 = choldown.chol_downdate(R.T, low_1)
        f_chol = cholup.chol_update if Wc[0] >= 0 else choldown.chol_downdate
        low_2 = f_chol(low2, torch.abs(Wc[0]) ** 0.5 * (xnew_s[:, [0]] - x_mean))

        L = assemble_chol([[A], [low_1, low_2]])

        return L, x_mean

    def _get_state_sample_ADF(self, w_s, c):
        """Get state prediction sample for ADF-based moment propagation
        Args:
            w_s: samples of h_x (n, dw)
            c: system input (1, dc) or None
        Returns:
            xnew_s: samples of the predicted x (n, dx)
        """

        n = w_s.shape[0]
        c_expd = c.expand(n, c.shape[-1]) if c is not None else None

        h_s = w_s[:, :-self.dx]
        x_s = w_s[:, -self.dx:]
        xnew_s, _, _ = self.fun.fun_tran(x_s, c_expd, h_s)

        return xnew_s
