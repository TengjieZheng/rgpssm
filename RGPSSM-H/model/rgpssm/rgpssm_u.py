from .rgpssm import *

class RGPSSM_U(RGPSSM):
    """
    The RGPSSM in which the order of x and u in the covariance matrix is with u on top and x on the bottom.
    This modification can improve computational efficiency because Suu remains unchanged during the prediction step.
    """
    def __init__(self,
                 x0: Union[ndarray, Tensor],
                 P0: Union[ndarray, Tensor],
                 Q: Union[ndarray, Tensor],
                 R: Union[ndarray, Tensor],
                 fun: IModel, kernel: IKer,
                 flag_chol: bool = True,
                 budget: int = 50, eps_tol: float = 1e-2,
                 num_opt_hp: int = 0, lr_hp: float = 1e-3,
                 type_score: str = 'full',
                 Z: Optional[Union[ndarray, Tensor]] = None,
                 Qu: float = 0e-6):
        """
        x0 : prior mean of state (dx, 1)
        P0 : prior variance of state (dx, dx)
        Q : process noise covariance (dx, dx)
        R : measurement noise covariance (ny, ny)
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
        super().__init__(x0, P0, Q, R, fun, kernel, flag_chol, budget, eps_tol, num_opt_hp, lr_hp, type_score, Z, Qu)

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
            Sigma = assemble_cov(self.S, self.V, self.P)
            self.L = torch.linalg.cholesky(Sigma)

    def correct(self, y: Optional[Union[ndarray, Tensor]], c: Optional[Tensor]=None, fun_meas=None, R: Optional[Tensor]=None):
        if fun_meas is None: fun_meas = self.fun.fun_meas
        c = ToTensor(c, view=(1, -1))
        if R is None: R = self.R
        y = ToTensor(y, view=(-1, 1))

        # Calculate some quantities
        x = clone_required_grad(self.x)
        c = clone_required_grad(c)
        y_pre, Cx = fun_meas(x.view(1, -1), c)
        y_pre = y_pre.view(-1, 1)
        if Cx is None: Cx = Jacobian(x, y_pre).detach()
        e = y - y_pre.detach()

        if self.flag_chol:
            P, V, _ = self._get_cov(self.L, isP=True, isV=True, isS=False)

            H = torch.cat((torch.zeros((Cx.shape[0], V.shape[1])), Cx), dim=1)
            Gamma = keep_sym(Cx @ P @ Cx.T + R)
            rho = torch.linalg.cholesky(Gamma)
            inv_rho_H = torch.linalg.solve(rho, H)
            a = self.L @ (self.L.T @ inv_rho_H.T)
            L_new = choldown.chol_downdate(self.L, a)

            if torch.isnan(self.L.reshape(-1)).any():
                print('L NaN in the correction step')
            else:
                self.L = L_new
                self.P, self.V, self.S = self._get_cov(self.L)

                q = Cx.T @ torch.linalg.solve(Gamma, e)
                self.x = self.x + P @ q
                self.m = self.m + V.T @ q
        else:
            raise NotImplementedError



    def hyperparam_opt(self):
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


    def _inducing_points_opt(self, flag_opt=False, type_score=None):
        """Optimize the inducing-point set"""

        if type_score is None:
            type_score = self.type_score

        if self.Z.shape[0] > self.budget or flag_opt:
            if type_score == "oldest":
                id_discard = 0
                l, r = id_discard * self.kernel.df, (id_discard + 1) * self.kernel.df
            else:
                id_discard, l, r = self._get_score(type_score)

            self.Z = get_vec_left(self.Z, id_discard, id_discard + 1, dim=0)
            self.m = get_vec_left(self.m, l, r, dim=0)

            if self.flag_chol:
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
            Sigma = assemble_cov(self.S, self.V, self.P)
            self.L = torch.linalg.cholesky(Sigma)
            inv_Sigma = torch.cholesky_inverse(self.L)
        inv_Sigma_u = inv_Sigma[:-self.dx, :-self.dx]
        self.inv_Kuu = torch.cholesky_inverse(self.L_Kuu)

        for d in range(self.Z.shape[0]):
            l, r = d * nf, (d + 1) * nf
            alpha_d = alpha[l:r, :]
            Qdd = self.inv_Kuu[l:r, l:r]
            inv_Qdd = torch.inverse(Qdd)
            if type_score == 'full':
                Qdu = self.inv_Kuu[l:r, :]
                _, ld_Q = torch.slogdet(Qdd)
                _, ld_S = torch.slogdet(inv_Sigma_u[l:r, l:r])
                score[d] = ld_S - ld_Q + alpha_d.T @ inv_Qdd @ alpha_d + torch.trace(
                    Qdu @ self.S @ Qdu.T @ inv_Qdd)
            elif type_score == 'mean':
                score[d] = alpha_d.T @ inv_Qdd @ alpha_d

        id_discard = torch.argmin(score)
        l, r = id_discard * nf, (id_discard + 1) * nf

        return id_discard, l, r


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
            self._init_q(z + ls * torch.randn(1) * 0.1)

        z, mt, Ft, Ax, Af, gam, Vxu, Vxt, Stt, Stu, Suu, ktu = self._get_tmp(c, fun_tran)
        self.x = Ft

        if self.flag_chol:
            # Update of Cholesky version
            K_diag = torch.diag(self.Kuu).reshape(-1, self.kernel.df)  # (nb, df)
            tr_max = torch.sum(K_diag, dim=-1).max()
            tr_gam = torch.trace(gam)
            if tr_gam > self.eps_tol * tr_max:
                v = torch.cat((Stu.T, Vxt), dim=0)
                L_new = chol_add(self.L, v, Stt, id=self.m.shape[0])

                P = keep_sym(Ax @ self.P @ Ax.T + Ax @ Vxt @ Af.T + Af @ Vxt.T @ Ax.T + Af @ Stt @ Af.T + Q)
                Vxu_new = torch.cat((Vxu, Vxt), dim=1)
                Stu_new = torch.cat((Stu, Stt), dim=1)
                V = Ax @ Vxu_new + Af @ Stu_new

                L_S = L_new[:-self.dx, :-self.dx]

                self.m = torch.cat((self.m, mt), dim=0)
                self.Z = torch.cat((self.Z, z), dim=0)
                self._update_K_chol()
            else:
                P = keep_sym(Ax @ self.P @ Ax.T + Ax @ Vxt @ Af.T + Af @ Vxt.T @ Ax.T + Af @ Stt @ Af.T + Q)
                V = Ax @ Vxu + Af @ Stu

                L_S = self.L[:-self.dx, :-self.dx]

            rho = torch.linalg.solve_triangular(L_S, V.T, upper=False).T
            beta = torch.linalg.cholesky(P - rho @ rho.T)
            self.L = assemble_chol([[L_S], [rho, beta]])
            self.P, self.V, self.S = self._get_cov(self.L)
        else:
            raise NotImplementedError

        return self.x, self.P, mt, Stt

    def _get_cov(self, L, isP=True, isV=True, isS=True):
        """Get covariance regarding the joint distribution
        Returns:
            P: Sxx
            V: Sxu
            S: Suu
        """

        dx = self.dx

        P, V, S = None, None, None
        A, B, C = L[:-dx, :-dx], L[-dx:, :-dx], L[-dx:, -dx:]
        if isS:
            S = A @ A.T
        if isV:
            V = B @ A.T
        if isP:
            P = B @ B.T + C @ C.T

        return P, V, S






