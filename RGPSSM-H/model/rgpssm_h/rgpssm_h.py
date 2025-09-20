
from .rgpssm_h_adf import *

class RGPSSM_H(RGPSSM_H_ADF):
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
