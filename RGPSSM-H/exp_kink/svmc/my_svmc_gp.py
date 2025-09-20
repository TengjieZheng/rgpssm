import torch.linalg

from .svmc.covfun import SquaredExponential
from .svmc import SVMC_GP
from .svmc import proposal
from .svmc import likelihood
from .svmc.dynamics import SGPDynamics, KSGPDynamics
from .svmc.gp import SGP
from .utils import *


class My_SVMC_GP():
    """A organized version of the original SVMC-GP implementation"""

    def __init__(self, dx, dy, varx, vary, P, u_inducing, KnoModel, fvar, length_scale,
                 lr=1e-3, d_hidden_mlp=32, n_pf=10, n_opt=5, iter_opt=5,
                 learn_emission=False, Cobs=None, dobs=None):
        """
        :param dx: dimension of state x
        :param dy: dimension of measurement y
        :param varx: process noise variance
        :param vary: measurement noise variance
        :param P: covariance of the initial state x0
        :param u_inducing: inducing point
        :param KnoModel: system model
        :param fvar: signal variance of the kernel
        :param length_scale: length scale of the kernel
        :param lr: learning rate
        :param d_hidden_mlp: dimension of hidden layer of the proposal network
        :param n_pf:  number of particles
        :param n_opt: number of particles to evaluate the loss
        :param iter_opt: number of iterations to optimize the proposal network
        :param learn_emission: whether to learn the emission function
        :param Cobs: measurement matrix
        :param dobs: measurement bias
        """

        self.dx = dx
        self.dy = dy
        self.varx = ToTensor(varx).double()
        self.vary = ToTensor(vary).double()
        self.P = ToTensor(P).double()
        self.KnoModel = KnoModel
        self.u_inducing = ToTensor(u_inducing).double()
        self.lr = lr
        self.d_hidden_mlp = d_hidden_mlp
        self.n_pf = n_pf
        self.n_opt = n_opt
        self.iter_opt = iter_opt
        self.learn_emission = learn_emission

        self.Cobs = ToTensor(Cobs).double() if Cobs is not None else torch.eye(max(dx, dy))[:dy, :dx].double()
        self.dobs = ToTensor(dobs).double().view(-1) if dobs is not None else torch.zeros(dy, 1).double().view(-1)
        self._init_emission()

        self.r = proposal.MlpProposal(dx + dy, d_hidden_mlp, dx, log_flag=False)  # MLP proposal

        # initial particles and weights
        self.x_particles = torch.randn(n_pf, dx).double() @ torch.linalg.cholesky(P).T
        self.w = np.ones(n_pf) / n_pf

        # GP particles
        self.cov_func = SquaredExponential(fvar, length_scale)
        self.cov_func.loggamma.requires_grad = False  # not optimize hyperparameter
        self.cov_func.logvar.requires_grad = False
        Q = varx * torch.eye(dx) # process noise covariance
        self.z_SS = [KSGPDynamics(u_inducing, None, self.cov_func, Q, KnoModel, diffusion=0) for n in range(n_pf)]  # Construct sparse GP objects
        self.pf = SVMC_GP(n_pf, n_opt, dx, dy, self.log_like, self.r, self.z_SS, lr=lr, gp_diffusion=0)

        self.particles = (self.x_particles, np.zeros((n_pf, dx)), self.z_SS)

    def _init_emission(self):
        self.log_like = likelihood.ISOGaussian(self.dy, self.dx, log_flag=False)
        self.log_like.input_to_output.weight.data = self.Cobs
        self.log_like.input_to_output.weight.requires_grad = self.learn_emission  # should we learn emission distribution

        self.log_like.input_to_output.bias.data = self.dobs
        self.log_like.input_to_output.bias.requires_grad = self.learn_emission  # should we learn emission distribution

        self.log_like.tau.data.fill_(torch.log(self.vary))
        self.log_like.tau.requires_grad = self.learn_emission  # should we learn emission distribution


    def update(self, y, train_GP=False):
        """Update SVMC-GP
        y: measurement (1, dy)
        train_GP: whether to train GP
        """
        y = ToTensor(y).double()

        self.particles, _, self.w = self.pf.filter(y, self.particles, self.w, self.iter_opt, nongradient=train_GP)
        self.x_particles = self.particles[0].detach()
        self.ESS = 1 / np.sum(self.w ** 2)

        self.x_est = torch.zeros((1, self.dx))
        for n in range(self.n_pf):
            self.x_est += self.w[n] * self.x_particles[n, :]

    def GP_predict(self, x, sgp=None):
        x = ToTensor(x, view=(1, -1)).double()
        sgp = self.get_sgp() if sgp is None else sgp
        gp_in = self.KnoModel.fun_input(x, u=None)
        f, fvar = sgp.predict(gp_in, full_cov=False)

        return f.detach(), fvar.detach()

    def get_sgp(self):
        sgp = SGP(self.KnoModel.d_out, self.KnoModel.d_in, self.u_inducing, None, self.cov_func, self.varx)

        qmean, qcov = 0, 0
        for n in range(self.n_pf):
            qmean += self.w[n] * self.particles[2][n].qz.mean
        for n in range(self.n_pf):
            vi = self.particles[2][n].qz.mean
            qcov += self.w[n] * (self.particles[2][n].qz.cov + vi @ vi.T)
        qcov = qcov - qmean @ qmean.T
        sgp.qz.mean = qmean
        sgp.qz.cov = qcov

        return sgp