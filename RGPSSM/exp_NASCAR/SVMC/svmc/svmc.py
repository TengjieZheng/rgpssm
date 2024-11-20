"""
NOTE:
Torch matrices are M by Dim
"""
import numpy as np
import torch
import copy
from torch.optim import Adam

from .base import Filter
from .dynamics import NullDynamics
from .metric import gaussian_loss
from .resample import resample

__all__ = ["SVMC", "SVMC_GP"]

class BoostrapPF(Filter):
    def __init__(self, n_pf, d_latent, d_obs, log_like, proposal, scheme='multinomial'):

        super().__init__()
        self.n_pf = n_pf  # number of particle streams to keep
        self.d_latent = d_latent  # dimension of latent space
        self.d_obs = d_obs  # dimension of observation space
        self.scheme = scheme  # resampling scheme to follow
        self.add_module("log_like", log_like)  # log_likelihood of the model
        self.add_module("proposal", proposal)  # proposal distribution for sampling

    def compute_weight(self, Y, particles, W, M, U=None, beta=1):
        """
        function for computing importance sampling weights
        :param Y: 1 by d_obs tensor which is the observation of the system at time t
        :param particles: n_pf by d_latent tensor containing the particles generated at time t-1 i.e. q(x_{t-1})
        :param W: n_pf numpy array containing NORMALIZED importance weights from the previous time step
        :param M: number of samples to generate from proposal
        :param U: 1 by du tensor corresponding to external stimulus
        :param beta: scalar used to downweight influence of dynamics model. useful for GP
        :return:
        """
        X = particles
        with torch.no_grad():
            # Draw ancestor variables
            ancestors = resample(W, M, self.scheme)

            # Select streams i.e. particles generated at the previous time instant
            x_prev = X[ancestors, :].detach()

        if U is None:
            cat = torch.cat((x_prev, Y.repeat(M, 1)), 1)
        else:
            cat = torch.cat((x_prev, U.repeat(M, 1), Y.repeat(M, 1)), 1)

        # Obtain samples from proposal distribution
        mu, var = self.proposal(cat)
        L_var = torch.linalg.cholesky(var)
        x_samples = mu + torch.randn(M, self.d_latent) @ L_var

        #  Compute log weights
        log_weight = self.log_like(Y, x_samples)

        return log_weight, x_samples

    def log_sum_exp(self, logW):
        max_logW = torch.max(logW)
        w = torch.exp(logW - max_logW)
        return w, max_logW


    def filter(self, Y, particles, W, U=None, beta=1,
               nongradient=False, project=False, n_sgd=1):
        """
        Perform filtering step of SVMC
        :param Y: 1 by d_obs tensor which is the observation of the system at time t
        :param particles: n_pf by d_latent tensor containing the particles generated at time t-1 i.e. q(x_{t-1})
        :param W: n_pf numpy array containing NORMALIZED importance weights from the previous time step
        :param U: 1 by du tensor corresponding to external stimulus
        :param beta: scalar used to downweight influence of dynamics model. useful for GP
        :param nongradient:
        :param project: boolean flag used to determine whether projected sgd should be done
        :param n_sgd: number of samples used to compute gradient
        :return: X, W i.e. q(x_t)
        """

        assert np.isclose(np.sum(W), 1, 1e-3)  # make sure the previous weights define a valid pmf
        # Optimize proposal
        if U is not None:
            U = torch.as_tensor(U)[None, :]

        # Filter one step forward
        with torch.no_grad():
            self.proposal.eval()
            logW, particles = self.compute_weight(Y, particles, W, self.n_pf, U, beta)
            w, _ = self.log_sum_exp(logW)  # log_sum_exp trick for numerical stability
            w_norm = w / torch.sum(w)  # normalize the weights

        return particles, logW.numpy(), w_norm.numpy()

    def forward(self, *args):
        self.filter(*args)


class SVMC(Filter):
    """
    Base class for streaming variational monte carlo (SVMC) for online filtering. This base class assumes the
    following:
    x_t = f(x_{t-1}; theta) + e_t,
    y_t = g(x_t;psi) + v_t
    The model will learn the parameters of the proposal distribution but can also jointly learn the
    parameters of the emission and transition distribution.
    """
    def __init__(self, n_pf, n_optim, d_latent, d_obs, log_like, log_dynamics, proposal, lr=0.001,
                 scheme='multinomial', weight_decay=0.):
        """
        Constructor for SVMC class
        :param n_pf: Number of particles used for approximating p(x_t | y_{<=t})
        :param n_optim: Number of particles used to compute elbo for computing stochastic gradients
        :param d_latent: Dimension of x_t
        :param d_obs: Dimension of observation, y_t
        :param log_like: log likelihood, log p(y_t | x_t), for generative model.
        :param log_dynamics: log dynamics, log p(x_t | x_{t-1}), for generative model.
        :param proposal: proposal object used to generate samples and evaluate log proposal
        :param lr: learning rate of optimizer
        :param scheme: string that dictates the resampling scheme
        :param weight_decay: strength of weight decay regularizer
        """
        super().__init__()
        self.n_pf = n_pf  # number of particle streams to keep
        self.n_optim = n_optim  # number of particles to sample when computing stochastic gradients
        self.d_latent = d_latent  # dimension of latent space
        self.d_obs = d_obs  # dimension of observation space
        self.scheme = scheme  # resampling scheme to follow
        self.add_module("log_like", log_like)  # log_likelihood of the model
        self.add_module("log_dynamics", log_dynamics)  # log of the transition distribution
        self.add_module("proposal", proposal)  # proposal distribution for sampling
        self.optim = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer


    def compute_weight(self, Y, particles, W, M, U=None, beta=1):
        """
        function for computing importance sampling weights
        :param Y: 1 by d_obs tensor which is the observation of the system at time t
        :param particles: n_pf by d_latent tensor containing the particles generated at time t-1 i.e. q(x_{t-1})
        :param W: n_pf numpy array containing NORMALIZED importance weights from the previous time step
        :param M: number of samples to generate from proposal
        :param U: 1 by du tensor corresponding to external stimulus
        :param beta: scalar used to downweight influence of dynamics model. useful for GP
        :return:
        """
        X = particles
        with torch.no_grad():
            # Draw ancestor variables
            ancestors = resample(W, M, self.scheme)

            # Select streams i.e. particles generated at the previous time instant
            x_prev = X[ancestors, :].detach()

        if U is None:
            cat = torch.cat((x_prev, Y.repeat(M, 1)), 1)
        else:
            cat = torch.cat((x_prev, U.repeat(M, 1), Y.repeat(M, 1)), 1)

        # Obtain samples from proposal distribution
        mu, var = self.proposal(cat)
        x_samples = mu + torch.sqrt(var) * torch.randn(M, self.d_latent)

        #  Compute log weights
        log_weight = self.log_like(Y, x_samples)
        log_weight += beta * self.log_dynamics(x_samples, cat[:, :-self.d_obs])
        log_weight += gaussian_loss(x_samples, mu, torch.log(var))
        return log_weight, x_samples

    def elbo(self, w, max_logW):
        return max_logW + torch.log(torch.sum(w) / self.n_optim)

    def loss(self, w, max_logW):
        return -self.elbo(w, max_logW)

    def log_sum_exp(self, logW):
        max_logW = torch.max(logW)
        w = torch.exp(logW - max_logW)
        return w, max_logW

    def filter(self, Y, particles, W, iter_optim, U=None, beta=1,
               nongradient=False, project=False, n_sgd=1):
        """
        Perform filtering step of SVMC
        :param Y: 1 by d_obs tensor which is the observation of the system at time t
        :param particles: n_pf by d_latent tensor containing the particles generated at time t-1 i.e. q(x_{t-1})
        :param W: n_pf numpy array containing NORMALIZED importance weights from the previous time step
        :param iter_optim: number of gradient steps to take
        :param U: 1 by du tensor corresponding to external stimulus
        :param beta: scalar used to downweight influence of dynamics model. useful for GP
        :param nongradient:
        :param project: boolean flag used to determine whether projected sgd should be done
        :param n_sgd: number of samples used to compute gradient
        :return: X, W i.e. q(x_t)
        """

        assert np.isclose(np.sum(W), 1, 1e-3)  # make sure the previous weights define a valid pmf
        # Optimize proposal
        if U is not None:
            U = torch.as_tensor(U)[None, :]
        if iter_optim > 0:
            self.proposal.train()
            for it in range(iter_optim):
                self.optim.zero_grad()
                loss = 0
                for j in range(n_sgd):  # todo: can totally parallelize this computation
                    logW, _ = self.compute_weight(Y[None, :], particles, W, self.n_optim, U, beta)
                    w, max_logW = self.log_sum_exp(logW)  # log_sum_exp trick for numerical stability
                    loss += self.loss(w, max_logW) / n_sgd  # compute lower bound
                loss.backward()  # compute gradient
                self.optim.step()  # take gradient step
                if project:
                    # Project onto your favorite manifold
                    self.log_like.project_unit_norm()

        # Filter one step forward
        with torch.no_grad():
            self.proposal.eval()
            logW, particles = self.compute_weight(Y, particles, W, self.n_pf, U, beta)
            w, _ = self.log_sum_exp(logW)  # log_sum_exp trick for numerical stability
            w_norm = w / torch.sum(w)  # normalize the weights

        if nongradient:
            self.nongradient(particles, U)

        return particles, logW.numpy(), w_norm.numpy()

    def forward(self, *args):
        self.filter(*args)

    def nongradient(self, *args, **kwargs):
        pass

    @staticmethod
    def build(*args, **kwargs):
        """Factory method to build objects of SVMC and subclasses"""
        if kwargs["dynamcs"] == "SGP":
            return SVMC_GP(*args, **kwargs)
        else:
            return SVMC(*args, **kwargs)


class SVMC_GP(SVMC):
    """
    SVMC where sparse GP prior is placed on dynamics
    x_t = f(x_{t-1}) + e_t, f ~ GP(0, K)
    y_t = g(x_t) + v_t
    z_1, .., z_M are inducing points
    u_1, ..., u_M are inputs for inducing points
    """
    def __init__(self, n_pf, n_optim, d_latent, d_obs, log_like, proposal, rz, lr=0.001,
                 scheme='multinomial', weight_decay=0., gp_diffusion=1e-2):

        """
        Constructor for SVMC_GP
        :param n_pf: Number of particles used for approximating p(x_t | y_{<=t})
        :param n_optim: Number of particles used to compute elbo for computing stochastic gradients
        :param d_latent: Dimension of x_t
        :param d_obs: Dimension of observation, y_t
        :param log_like: log likelihood, log p(y_t | x_t), for generative model.
        :param proposal: proposal object used to generate samples and evaluate log proposal
        :param rz: proposal for inducing points
        :param lr: learning rate of optimizer
        :param scheme: string that dictates the resampling scheme
        :param weight_decay: strength of weight decay regularizer
        :param gp_diffusion: magnitude of variance for diffusion process on inducing points
        """
        super().__init__(n_pf=n_pf, n_optim=n_optim, d_latent=d_latent, d_obs=d_obs,
                         log_like=log_like, log_dynamics=NullDynamics(), proposal=proposal,
                         lr=lr, scheme=scheme, weight_decay=weight_decay)
        self.z_proposals = rz
        self.optim = Adam(self.parameters(), lr=lr)  # optimizer
        self.gp_diffusion = torch.tensor(gp_diffusion)

    def compute_weight(self, Y, particles, W, M, U=None, beta=1):
        X, _, z_proposals = particles

        with torch.no_grad():
            # Draw ancestor variables
            ancestors = resample(W, M, self.scheme)

            # Select streams i.e. last particle generated and each streams sufficient statistics
            x_prev = X[ancestors, :]
            z_proposals = [copy.deepcopy(z_proposals[ancestors[i]]) for i in range(M)]

        # Obtain samples from proposal distribution
        if U is None:
            cat = torch.cat((x_prev, Y.repeat(M, 1)), 1)
        else:
            cat = torch.cat((x_prev, U.repeat(M, 1), Y.repeat(M, 1)), 1)

        mu, var = self.proposal(cat)
        x_samples = mu + torch.sqrt(var) * torch.randn(M, self.d_latent)

        # Compute unnormalized log weights
        log_weight = self.log_like(Y, x_samples)  # compute contribution from log-likelihood
        log_weight += gaussian_loss(x_samples, mu, torch.log(var))  # compute contribution from proposal

        # TODO: write multivariate gaussian loss for batch
        for m in range(M):
            log_weight[m] += z_proposals[m].log_weight(x_prev[[m], :], x_samples[[m], :], u=U)

        return log_weight, (x_samples, x_prev.detach(), z_proposals)

    def nongradient(self, particles, U=None):
        with torch.no_grad():
            x_samples, x_prevs, z_proposals = particles

            for x_sample, x_prev, z_proposal in zip(x_samples, x_prevs, z_proposals):
                z_proposal.update(x_prev[None, :], x_sample[None, :], u=U)

