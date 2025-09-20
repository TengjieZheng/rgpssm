
from .rgpssm_h_ekf import *
from .ukf import UKF_utils as ukf


class RGPSSM_H_UKF(RGPSSM_H_EKF):
    """Recursive Gaussian process state space model for heterogeneous input (RGPSSM-H) with UKF-based moment matching"""
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
        type_filter: type of filter, "UKF", "EKF"
        """

        super().__init__(x0, P0, Q, R, fun, kernel, flag_chol, budget, eps_tol,
                         num_opt_hp, lr_hp, type_score, 0.)

        self.type_filter = type_filter
        self.init_ukf()

    def init_ukf(self):
        if self.type_filter == 'UKF' or self.type_filter == 'UKF2':
            self.alpha_ukf = 5e-1
            self.beta_ukf = 2
            # eta == dx**0.5 * alpha
        elif self.type_filter == 'ADF':
            self.alpha_ukf = 5e-1
            self.beta_ukf = 2

    def predict(self, c: Optional[Union[ndarray, Tensor]] = None, fun_tran=None, Q: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
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
        else:
            raise ValueError('type_filter must be "EKF" or "UKF"')

        self._inducing_points_opt()

        return F, var_F, f, var_f

    def _moments_propagate_UKF(self, c=None, fun_tran=None, Q=None):
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

            # Get UKF param
            dh = self.kernel.nf
            dxu = L_new.shape[0]
            du = dxu - self.dx
            dxue = dxu + dh
            eta, Wm, Wc = ukf.param(dxue, self.alpha_ukf, self.beta_ukf)

            # Samples
            Lu = L_new[:du, :du]                                                            # Cholesktr factor of u
            Luxe = torch.block_diag(L_new, torch.eye(dh))                           # Cholesky factor of (u,x,e)
            uxe_m = torch.cat((self.m, self.x, torch.zeros((dh, 1))), dim=0)         # mean of (u,x,e) (duxe, 1)
            uxe_s = ukf.sample(uxe_m, Luxe, eta)                                            # sample of (u,e,e) (duxe, ns)
            u_s = uxe_s[:du, :]

            # Prediction
            xnew_s = self._get_state_sample_UKF(uxe_s.T, c, du, dh, gam).T                                 # get sample of x_new (dx, ns)
            sqrt_Q = torch.block_diag(self.Qu ** 0.5 * torch.eye(du), Q ** 0.5)     # Cholesky factor of process noise covariance w.r.t. (u, x)
            self.x, self.L = ukf.pred_xw(xnew_s, u_s, self.m, Wm, Wc, sqrt_Q, upper_x=False, Lw=Lu, eta=eta) # get mean of x and Cholesky factor of (u, x)

            # Covariance
            self.P, self.V, self.S = self._get_cov(self.L)
        else:
            raise NotImplementedError('The standard version of UKF-based moment matching is not implemented. It is recommended to use the Cholesky version.')

        return self.x, self.P, mt, Stt

    def _get_state_sample_UKF(self, uxe, c, du, dh, gam):
        """Get state prediction sample for UKF-based moment propagation
        Args:
            uxe: sample for (u,x,e) (n, duxe)
            c: control input (1, dc) or None
            du: dimension of inducing points u
            dh: dimension of latent function prediction h
        returns:
            xnew: sample for new state xnew (n, dx)
        """

        n = uxe.shape[0]
        u, x, e = uxe[:, :du], uxe[:, du:du+self.dx], uxe[:, -dh:]
        c_expd = c.expand(n, c.shape[-1]) if c is not None else None

        # GP mean prediction
        z, _ = self.fun.fun_input(x, c_expd)                # (n, dh, dz_max)

        id_zs = self.kernel.id_z.view(-1, 1).expand(-1, n)  # each row is a f output
        id_zs = id_zs.reshape(-1)                           # (dh*n) sorted by the f output
        zs = z.permute(1, 0, 2)                             # (dh, n, dz_max)
        zs = zs.reshape(-1, zs.shape[-1])                   # (dh*n, dz_max) sorted by the f output
        with torch.no_grad():
            Kxu = self.kernel(zs, id_zs, self.Z, self.Id_Z, flag_sort=True)  # (n*dh, du) sorted by the f output
        kxu = torch.cholesky_solve(Kxu.T, self.L_Kuu).T     # (n*dh, du) sorted by the f output
        ur = u.repeat(self.kernel.nf, 1)                    # (n*dh, du)
        f = torch.sum(kxu * ur, dim=-1).unsqueeze(-1)       # (n*dh, 1)
        f = f.reshape(-1, n).T                              # (n, dh)

        # Add GP prior noise
        noise = self._get_gp_prior_noise_UKF(e, gam, du, dh) # (n, dh)
        f = f + noise                                       # (n, dh)

        # State prediction
        xnew, _, _ = self.fun.fun_tran(x, c_expd, f)        # (n, dx)

        return xnew.detach()

    def _get_gp_prior_noise_UKF(self, e, gam, du, dh):
        """Get sample for the GP prior prediction noise, namely, \Sigma_{gp}^{1/2}(x)e
        Args:
            e: (2*duxe+1, dh)
            gam: (dh, dh)
            du: int
            dh: int
        Returns:
            noise: (n, dh)
        """
        dux = self.dx + du

        e_ = torch.cat((e[dux+1:dux+1+dh], e[-dh:]), dim=0)     # (2*dh, dh)
        e_ = e_.unsqueeze(-1)                                           # (2*dh, dh, 1)
        L_gam = torch.linalg.cholesky(gam)                              # (dh, dh)
        noise_ = (L_gam @ e_).squeeze(-1)                              # (2*dh, dh)
        zeros1 = torch.zeros((dux+1, dh))
        zeros2 = torch.zeros((dux, dh))
        noise = torch.cat((zeros1, noise_[:dh], zeros2, noise_[dh:]), dim=0)    # (2*duxe+1, dh)
        # Here, we utilize the fact that e[:dux+1] = 0 and e[duxe+1:-dh] = 0

        return noise

    def _add_point_UKF(self, flag_add, z, Vxt, Stu, Stt, mt):
        """Update Update the relevant variables after adding points for EKF-based moment propagation
        Args:
            flag_add: Tensor (n_gp,) indicates which new inducing points should be added to the GP model.
        """

        L_new = self.L
        S_ux_a = torch.cat((Stu.T, Vxt), dim=0)

        for i in range(flag_add.numel()):
            if flag_add[i]:
                idx_z = (self.Id_Z <= i).sum()
                idx_u = (self.Id_f <= i).sum()

                bool_f = self.kernel.id_f == i
                L_new = chol_add(L_new, S_ux_a[:, bool_f], Stt[bool_f, :][:, bool_f], idx_u)
                S_ux_a = insert_row(S_ux_a, Stt[bool_f, :], idx_u)

                self.Z = insert_row(self.Z, z[[i], :], idx_z)
                id_add = torch.tensor(i).view(-1)
                self.Id_Z = insert_row(self.Id_Z, id_add, idx_z)
                self.Id_f = insert_row(self.Id_f, self.kernel.idZ_to_idf(id_add), idx_u)
                self.m = insert_row(self.m, mt[bool_f, :], idx_u)

        self._update_K_chol()
        self.P, self.V, self.S = self._get_cov(L_new)

        return L_new

    def _get_tmp_UKF(self, c):

        x = clone_required_grad(self.x)
        c = clone_required_grad(c, view=(1, -1))
        z, _ = self.fun.fun_input(x.view(1, -1), c)
        z = z.squeeze(0)
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

        return z.detach(), mt.detach(), gam.detach(), Vxt.detach(), Stt.detach(), Stu.detach(), Ktt.detach()