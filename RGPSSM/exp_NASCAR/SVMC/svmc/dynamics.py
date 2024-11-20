import torch
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Linear, Parameter, Sequential, ReLU, Tanh
import numpy as np
from .operation import kron
from .gp import SGP, KnoSGP
from .covfun import CovarianceFunction
from .base import Dynamics, KnowledgeModel
from .metric import gaussian_loss


class LDS(Dynamics):
    """Linear dynamical system, x_t = Ax_{t-1} + B  + e_t, e_t ~ N(0, sI)"""
    def __init__(self, d_latent, log_flag=True):
        super().__init__()
        self.d_latent = d_latent  # dimension of latent dynamics
        self.log_flag = log_flag
        self.add_module("trans_matrix", Linear(d_latent, d_latent))  # [A, B]
        self.register_parameter("tau", Parameter(torch.zeros(1)))  # noise is from isotropic Gaussian

    def log_weight(self, xt, x):
        mean = self.trans_matrix(x)

        # compute isotropic covariance matrix
        var = torch.exp(self.tau)
        if self.log:
            var = torch.log(1 + var)

        # Evaluate log density
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(xt), -1)
        return log_weight

    def forward(self, xt, x):
        return self.log_weight(xt, x)

    def one_step_sim(self, x):
        mean = self.trans_matrix(x)
        return mean

    def bootstrap(self, x):
        return self.one_step_sim(x)

    def sample(self, x):
        mean = self.trans_matrix(x)

        # compute isotropic covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        return mean + torch.sqrt(var) * torch.randn(1, self.d_latent)


class DiagLDS(Dynamics):
    """Linear dynamical system, x_t = Ax_{t-1} + B  + e_t, e_t ~ N(0, diag(s))"""
    def __init__(self, d_latent, log_flag=True):
        super().__init__()
        self.d_latent = d_latent  # dimension of latent dynamics
        self.log_flag = log_flag
        self.add_module("trans_matrix", Linear(d_latent, d_latent))  # [A, B]
        self.register_parameter("tau", Parameter(torch.zeros(d_latent)))  # noise is from isotropic Gaussian

    def log_weight(self, xt, x):
        mean = self.trans_matrix(x)

        # compute diagonal covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        # Evaluate log density
        log_weight = MultivariateNormal(loc=mean, covariance_matrix=torch.diag(var)).log_prob(xt)
        return log_weight

    def forward(self, xt, x):
        return self.log_weight(xt, x)

    def one_step_sim(self, x):
        mean = self.trans_matrix(x)
        return mean

    def sample(self, x):
        mean = self.trans_matrix(x)

        # compute covariance matrix
        var = torch.exp(self.tau)
        if self.log:
            var = torch.log(1 + var)

        return mean + torch.sqrt(var) * torch.randn(1, self.d_latent)


class NullDynamics(Dynamics):
    def __init__(self):
        super().__init__()

    def sample(self):
        pass

    def log_weight(self):
        pass


class SGPDynamics(Dynamics):
    def __init__(self, ydim, xdim, inducing, mean, cov: CovarianceFunction, noise, K="I", diffusion=0):
        super().__init__()

        self.add_module("sgp", SGP(ydim, xdim, inducing, mean, cov, noise, K, diffusion))

    @property
    def qz(self):
        return self.sgp.qz

    def predict(self, x, u=None, z=None, m=None, G=None, full_cov=True):
        if u is not None: # u is control inout
            xu = torch.cat((x, u), dim=-1)
        else:
            xu = x
        dx, cov = self.sgp.predict(xu, z, m, G, full_cov)
        return x + dx, cov

    def sample(self, size=1):
        # mean_z, cov_z = self.sgp.Qosterior
        mean_z = self.sgp.qz.mean
        cov_z = self.sgp.qz.cov
        L = cov_z.cholesky(upper=False)
        return mean_z + L @ torch.randn(mean_z.shape[0], size)

    def log_weight(self, x_prev, x_curr, u=None, z=None):
        if z is None:
            fmean, fcov = self.predict(x_prev, u, z)
        else:
            z = self.sample()  # first obtain sample from posterior of inducing points
            fmean, fcov = self.predict(x_prev, u, z)
        fcov += kron(self.sgp.Q, torch.eye(x_prev.shape[0]))  # state noise
        # Compute density of normal pdf with input x and output y
        return MultivariateNormal(loc=fmean.squeeze(), covariance_matrix=fcov).log_prob(x_curr.squeeze())

    def update(self, x, y, u=None):
        if u is not None:
            xu = torch.cat((x, u.repeat((x.shape[0], 1))), dim=-1)
        else:
            xu = x
        self.sgp.update(xu, y - x)

    def change_inducing(self, inducing):
        self.sgp.change_inducing(inducing)


class KSGPDynamics(Dynamics):
    """Knowledge informed sparse Gaussian process dynamics"""

    def __init__(self, inducing, mean, cov: CovarianceFunction, Q, KnoModel: KnowledgeModel, K="I", diffusion=0):
        """
        KnoModel: Knowledge informed dynamics model
        """
        super().__init__()

        self.KnoModel = KnoModel
        self.add_module("sgp", KnoSGP(KnoModel, inducing, mean, cov, Q, K, diffusion))

    @property
    def qz(self):
        return self.sgp.qz

    def predict(self, x, u=None, z=None, m=None, G=None, full_cov=True):

        gp_in = self.KnoModel.fun_input(x, u)
        f, fcov = self.sgp.predict(gp_in, z, m, G, full_cov)
        F, Af = self.KnoModel.fun_tran(x, u, f)
        Fcov = Af @ fcov @ Af.T

        return F.reshape(1, -1), Fcov

    def sample(self, size=1):
        # mean_z, cov_z = self.sgp.Qosterior
        mean_z = self.sgp.qz.mean
        cov_z = self.sgp.qz.cov
        L = cov_z.cholesky(upper=False)
        return mean_z + L @ torch.randn(mean_z.shape[0], size)

    def log_weight(self, x_prev, x_curr, u=None, z=None):
        if z is None:
            fmean, fcov = self.predict(x_prev, u, z)
        else:
            z = self.sample()  # first obtain sample from posterior of inducing points
            fmean, fcov = self.predict(x_prev, u, z)
        fcov += kron(self.sgp.Q, torch.eye(x_prev.shape[0]))  # state noise
        # Compute density of normal pdf with input x and output y
        return MultivariateNormal(loc=fmean.squeeze(), covariance_matrix=fcov).log_prob(x_curr.squeeze())

    def update(self, x, y, u=None):

        self.sgp.update(x, y, u)

    def change_inducing(self, inducing):
        self.sgp.change_inducing(inducing)

class ArtificialCircuit(Dynamics):
    def __init__(self, var=1e-3, dt=1e-3):
        super().__init__()
        self.d_latent = 3
        self.var = var  # covariance of state noise
        self.dt = dt  # size of step in Euler integration
        self.alpha = 1.5 * np.cos(np.pi/5)
        self.beta = 1.5 * np.sin(np.pi/5)

    def auto_dynamics(self, x):
        deriv = torch.zeros(x.shape[0], self.d_latent)
        deriv[:, 0] = (5 * x[:, 2] - 5) * (x[:, 0] - torch.tanh(self.alpha * x[:, 0] - self.beta * x[:, 1]))  # dX
        deriv[:, 1] = (5 * x[:, 2] - 5) * (x[:, 1] - torch.tanh(self.beta * x[:, 0] + self.alpha * x[:, 1]))  # dY
        return self.dt * deriv

    def stim_dynamics(self, x, u):
        g = u * ((1 - u) * x[:, 0] ** 2 + (2 * u - 1) * x[:, 0] * x[:, 1] - u * x[:, 1] ** 2) + 2 * u  # stimulator
        deriv = torch.zeros(x.shape[0], self.d_latent)
        deriv[:, 0] = 5 * g * (x[:, 0] - torch.tanh(self.alpha * x[:, 0] - self.beta * x[:, 1]))  # dX
        deriv[:, 1] = 5 * g * (x[:, 1] - torch.tanh(self.beta * x[:, 0] + self.alpha * x[:, 1]))  # dY
        deriv[:, 2] = -0.5 * (x[:, 2] + g - torch.tanh(1.5 * (x[:, 2] + g))) + g
        return self.dt * deriv

    def one_step_sim(self, x, u):
        return x + self.auto_dynamics(x) + self.stim_dynamics(x, u)

    def log_weight(self, x_curr, x_prev):
        u = x_prev[:, -1]  # stimulus appended at the end
        x_prev = x_prev[:, -1]
        mu = self.one_step_sim(x_prev, u)

        # Evaluate log density
        log_weight = MultivariateNormal(loc=mu, covariance_matrix=self.var * torch.eye(self.d_latent)).log_prob(x_curr)
        return log_weight

    def forward(self, x_curr, x_prev):
        return self.log_weight(x_prev, x_curr)

    def sample(self, x, u):
        mu = self.one_step_sim(x, u)
        return mu + torch.sqrt(self.var) * torch.randn(self.d_latent)


class NarendraLi(Dynamics):
    def __init__(self, cov=0.1*torch.eye(2)):
        super().__init__()
        self.d_latent = 2
        self.cov = cov

    def one_step_sim(self, x, u):
        xt = torch.zeros(x.shape[0], self.d_latent)
        xt[:, 0] = torch.sin(x[:, 1]) * (x[:, 0]/(1 + x[:, 0] ** 2) + 1)
        xt[:, 1] = x[:, 1] * torch.cos(x[:, 1]) + x[:, 0] * torch.exp(-(x[:, 0] ** 2 + x[:, 1] ** 2)/8) + u ** 3 / (1 + u ** 2 + 0.5 *
                                                                                                  torch.cos(x[:, 0] + x[:, 1]))
        return xt

    def log_weight(self, x_curr, x_prev):
        u = x_prev[:, -1]  # stimulus appended at the end
        x_prev = x_prev[:, -1]
        mu = self.one_step_sim(x_prev, u)

        # Evaluate log density
        log_weight = MultivariateNormal(loc=mu, covariance_matrix=self.cov).log_prob(x_curr)
        return log_weight

    def forward(self, x_curr, x_prev):
        return self.log_weight(x_curr, x_prev)

    def sample(self, x, u):
        mu = self.one_step_sim(x, u)
        return mu + torch.sqrt(self.cov) @ torch.randn(self.d_latent)

    def bootstrap(self, x):
        return self.one_step_sim(x[:, :-1], x[:, -1])


class VanillaRNN(Dynamics):
    def __init__(self, d_latent, var, bias=False, dt=1e-3, gamma=2.5, tau=0.025):
        super().__init__()
        self.d_latent = d_latent
        self.add_module("W", Linear(d_latent, d_latent, bias=bias))
        self.dt = dt
        self.gamma = gamma
        self.tau = tau
        self.var = var * torch.ones(1)

    def one_step_sim(self, x):
        dx = self.dt * (-x + self.gamma * self.W(torch.tanh(x)))/self.tau
        return x + dx

    def log_weight(self, x_curr, x_prev):
        mu = self.one_step_sim(x_prev)

        # Evaluate log density
        log_weight = -gaussian_loss(x_curr, mu, torch.log(self.var))
        return log_weight

    def forward(self, x_curr, x_prev):
        return self.log_weight(x_curr, x_prev)

    def sample(self, x):
        mu = self.one_step_sim(x)
        return mu + torch.sqrt(self.var) * torch.randn(self.d_latent)

    def bootstrap(self, x):
        return self.one_step_sim(x)


class SnowMan(Dynamics):
    def __init__(self, r=2, dt=0.01, var=0.01 ** 2):
        super().__init__()
        self.dt = dt
        self.dx = 2
        self.var = var * torch.ones(1)
        self.r = r * torch.ones(1)

    def ccwring(self, x):
        """counter-clockwise ring limit cycle"""
        return torch.cat((-x[:, [1]], x[:, [0]]), 1) + x * (self.r ** 2 - x[:, [0]] ** 2 - x[:, [1]] ** 2)

    def cwring(self, x):
        """clockwise ring limit cycle"""
        return torch.cat((x[:, [1]], -x[:, [0]]), 1) + x * (self.r ** 2 - x[:, [0]] ** 2 - x[:, [1]] ** 2)

    def mask(self, x):
        """soft-mask over latent space"""
        return torch.sigmoid(100 * (x[:, [1]] - self.r))

    def one_step_sim(self, x):
        if len(x.shape) == 1:
            x.unsqueeze_(0)
        mask = self.mask(x)
        dx = (1 - mask) * self.cwring(x) + mask * self.ccwring(torch.cat((x[:, [0]], x[:, [1]] - 2 * self.r), 1))
        return x + self.dt * dx

    def sample(self, x):
        mu = self.one_step_sim(x)
        return mu + torch.sqrt(self.var) * torch.randn(mu.shape)

    def log_weight(self, x_curr, x_prev):
        mu = self.one_step_sim(x_prev)

        # Evaluate log density
        log_weight = -gaussian_loss(x_curr, mu, torch.log(self.var))
        return log_weight

    def forward(self, x_curr, x_prev):
        return self.log_weight(x_curr, x_prev)

    def bootstrap(self, x):
        return self.one_step_sim(x)


class MLPDynamics(Dynamics):
    def __init__(self, d_in, d_hidden, d_out,
                 n_layers=2, log_flag=True, relu_flag=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.input_to_mean = Sequential(*[Linear(d_in, d_hidden),
                                          ReLU() if relu_flag else Tanh()] +
                                         sum([
                                             [Linear(d_hidden, d_hidden),
                                              ReLU() if relu_flag else Tanh()
                                              ]
                                             for _ in range(n_layers - 1)],
                                             [])
                                        + [Linear(d_hidden, d_out)]
                                        )
        self.log_flag = log_flag
        self.register_parameter("tau", Parameter(0.1 * torch.randn(d_in)))  # noise is from isotropic Gaussian

    def log_weight(self, xt, x):
        mean = x + self.input_to_mean(x)
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(xt), -1)
        return log_weight

    def forward(self, xt, x):
        return self.log_weight(xt, x)

    def one_step_sim(self, x):
        return x + self.input_to_mean(x)

    def sample(self, x):
        mu = self.one_step_sim(x)
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)
        return mu + torch.sqrt(var) * torch.randn(mu.shape)


class BootstrapDynamics(Dynamics):
    def __init__(self, d_in, d_out, dynamics, log_flag=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dynamics = dynamics
        self.log_flag = log_flag
        self.register_parameter("tau", Parameter(0.1 * torch.randn(d_in)))  # noise is from isotropic Gaussian

    def log_weight(self, xt, x):
        mean = self.dynamics(x)
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(xt), -1)
        return log_weight

    def forward(self, xt, x):
        return self.log_weight(xt, x)

    def one_step_sim(self, x):
        return self.dynamics(x)

    def sample(self, x):
        mu = self.one_step_sim(x)
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)
        return mu + torch.sqrt(var) * torch.randn(mu.shape)