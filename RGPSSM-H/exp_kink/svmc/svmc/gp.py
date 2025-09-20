"""
Gaussian Process
"""
from abc import ABCMeta

import numpy as np
import torch
from torch.nn import Module, Parameter
from torch.linalg import solve

from .covfun import CovarianceFunction
from .operation import kron, yakron
from .base import KnowledgeModel


__all__ = ["GP", "SGP", "predict", "update"]

eps = 1e-16


class Qosterior(Module):
    def __init__(self, mean, cov):
        super().__init__()

        self.register_parameter("q_mean", Parameter(mean, requires_grad=False))
        self.register_parameter("q_cov", Parameter(cov, requires_grad=False))

    @property
    def mean(self):
        return self.q_mean

    @mean.setter
    def mean(self, value):
        self.q_mean.data = value

    @property
    def cov(self):
        return self.q_cov

    @cov.setter
    def cov(self, value):
        self.q_cov.data = value

    def forward(self, *input):
        raise NotImplementedError


class GP(Module, metaclass=ABCMeta):
    """Gaussian Process
    y = f(x) + e
    """

    def __init__(self, ydim, xdim, mean, cov: CovarianceFunction, noise, K="I"):
        """
        :param ydim: dimension of y
        :param xdim: dimension of x
        :param mean: mean function, (n, xdim) => (n, ydim)
        :param cov: covariance function
        :param noise: variance of noise e
        :param K: prior covariance between dimensions of f
        """
        super().__init__()

        self.ydim = ydim
        self.xdim = xdim
        if mean is None:
            self.mean_func = self._mean
        else:
            self.mean_func = mean
        self.register_parameter(
            "logvar", Parameter(torch.tensor(np.log(noise + eps)), requires_grad=False)
        )
        if K == "I":
            self.K = torch.eye(self.ydim)
        else:
            raise ValueError("unsupported covariance")
        self.add_module("cov_func", cov)

    def _mean(self, x):
        return torch.zeros(x.shape[0], self.ydim).double()

    @property
    def Q(self):
        return torch.exp(self.logvar) * torch.eye(self.ydim)


class SGP(GP):
    """Sparse Gaussian Process"""

    def __init__(
        self,
        ydim,
        xdim,
        inducing,
        mean,
        cov: CovarianceFunction,
        noise,
        K="I",
        diffusion=0.,
    ):
        """
        :param ydim: dimension of y
        :param xdim: dimension of x
        :param inducing: pseudo inputs, row vectors
        :param mean: mean function
        :param cov: covariance function
        :param noise: variance of noise
        :param K: prior covariance between dimensions of f
        :param diffusion: variance of diffusion for SGP
        """
        super().__init__(ydim, xdim, mean, cov, noise, K)

        self.inducing = torch.as_tensor(inducing)
        self.inducing.requires_grad_(False)
        self.mz = self.mean_func(self.inducing).t().reshape(-1, 1) + torch.zeros(
            self.ydim * self.inducing.shape[0], 1)
        # self.diffusion = diffusion * torch.eye(inducing.size)
        self.diffusion = torch.tensor(diffusion)
        self.initialize()

    def initialize(self):
        mean = self.mean_func(self.inducing).t().reshape(-1, 1) + torch.zeros(
            self.ydim * self.inducing.shape[0], 1
        )
        Kzz = self.cov_func(self.inducing)
        self.Kzz = Kzz
        Kzz = kron(self.K, Kzz)
        self.add_module("qz", Qosterior(mean, Kzz))

    def precompute(self, x):
        Kff = self.cov_func(x)
        # Kff = torch.diag(torch.diag(Kff))
        # Kff = kron(self.K, Kff)
        Kfz = self.cov_func(x, self.inducing)
        # Kfz = kron(self.K, Kfz)
        # Kzz = self.cov_func(self.inducing)
        # Kzz = kron(self.K, Kzz)
        # Kzz = (Kzz + Kzz.t())/2
        A = solve(self.Kzz, Kfz.t())
        B = Kff - torch.mm(Kfz, A)
        # B = 0.5 * (B + B.t())
        A = A.t()
        A = kron(self.K, A)
        B = kron(self.K, B)
        C = B + kron(self.Q, torch.eye(x.shape[0]))
        # C = B + self.Q
        return A, B, C

    def predict(self, x, z=None, m=None, G=None, full_cov=True):
        """f(x)
        :param x: predictor
        :param z: inducing variable
        :param m: mean of z
        :param G: covariance of z
        :param full_cov: return full covariance matrix of f if True or return only variance (diagonals)
        :return:
        """
        x = torch.as_tensor(x)
        A, B, C = self.precompute(x)
        # Kfz Kzz^{-1} z, AtΓtAt' + Bt
        if z is None:
            if m is None and G is None:
                m, G = self.qz.mean, self.qz.cov
            elif m is not None and G is not None:
                pass
            else:
                raise ValueError("m and G must be specified together")
            fcov = A @ G @ A.t() + B
            fmean = A @ (m - self.mz)
        else:
            fmean = A @ (z - self.mz) # 使用时z是从诱导输出的后验采样的，因此该预测为采样预测，为何不用减去mz
            fcov = B

        fmean = torch.reshape(
            fmean, (self.ydim, -1)
        ).t()  # (ydim * n,) => (ydim, n) => (n, ydim)
        fmean = fmean + self.mean_func(x)  # mean function

        if not full_cov:
            fcov = torch.diag(fcov)
            fcov = torch.reshape(fcov, (self.ydim, -1)).t()

        return fmean, fcov

    def forward(self, x, z=None, full_cov=True):
        return self.predict(x, z, full_cov)

    def update(self, x, y):
        with torch.no_grad():
            x = torch.as_tensor(x)
            y = torch.as_tensor(y) - self.mean_func(x)  # (n, q)
            y = torch.reshape(y.t(), (-1, 1))  # (qn, 1)

            A, B, C = self.precompute(x)
            m, G = self.qz.mean, self.qz.cov
            Gprev = G
            AG = A @ G
            # Inversion by Woodbury identity
            D = C + AG @ A.t()
            G = G - AG.t() @ solve(D, AG)
            G = 0.5 * (G + G.t())  # symmetrize
            m = G @ (solve(Gprev, m) + A.t() @ solve(C, y - A @ self.mz))
            # m = m + G @ (A.t() @ solve(C, y))

            self.qz.mean = m
            self.qz.cov = G + torch.eye(G.shape[0]) * self.diffusion

    def change_inducing(self, inducing):
        """Change inducing points
        :param inducing: new pseudo inputs, row vectors
        :return:
        """
        inducing = torch.as_tensor(inducing)
        inducing = inducing.clone().detach().requires_grad_(False)
        fmean, fcov = self.predict(inducing, full_cov=True)
        self.inducing = inducing
        self.qz.mean = torch.reshape(fmean.t(), (-1, 1))
        self.qz.cov = fcov
        # inclusive KL projection

class KnoSGP(SGP):
    """SGP used in the Knowledge informed SGP dynamics"""
    def __init__(
        self,
        KnoModel: KnowledgeModel,
        inducing,
        mean,
        cov: CovarianceFunction,
        Q,
        K="I",
        diffusion=0.,
    ):
        """
        :param KnoModel: knowledge model
        :param inducing: pseudo inputs, row vectors
        :param mean: mean function
        :param cov: covariance function
        :param noise: variance of noise
        :param K: prior covariance between dimensions of f
        :param diffusion: variance of diffusion for SGP
        """

        self.KnoModel = KnoModel
        self.Q_Kno = torch.as_tensor(Q)
        super().__init__(KnoModel.d_out, KnoModel.d_in, inducing, mean, cov, 0, K, diffusion)

    def update(self, x, y, u=None):
        with torch.no_grad():
            x = torch.as_tensor(x).reshape(-1, 1)
            y = torch.as_tensor(y).reshape(-1, 1)
            if u is not None:
                u = torch.as_tensor(u).reshape(-1, 1)
            gp_in = self.KnoModel.fun_input(x, u)

            A, B, C = self.precompute(gp_in)
            m, G = self.qz.mean, self.qz.cov

            Gprev = G
            AG = A @ G
            f = A @ (m - self.mz) + self.mean_func(gp_in).reshape(-1, 1)
            fcov = AG @ A.t() + B
            F, Af = self.KnoModel.fun_tran(x, u, f)
            gam = Af @ fcov @ Af.T + self.Q_Kno

            # Inversion by Woodbury identity
            HG = Af @ AG
            K = solve(gam, HG).T # Kalman gain
            G = Gprev - K @ HG
            G = 0.5 * (G + G.t())  # symmetrize
            m = m + K @ (y - F)

            self.qz.mean = m
            self.qz.cov = G + torch.eye(G.shape[0]) * self.diffusion

            a = 1

    @property
    def Q(self):
        return self.Q_Kno

class DiagSGP(GP):
    def __init__(
        self,
        ydim,
        xdim,
        mean,
        cov: CovarianceFunction,
        noise,
        inducing,
        diffusion=0.,
        rep=1,
    ):
        """
        :param ydim: dimension of y
        :param xdim: dimension of x
        :param mean: mean function
        :param cov: covariance function
        :param noise: variance of noise
        :param K: prior covariance between dimensions of f
        :param inducing: pseudo inputs, row vectors
        :param diffusion: variance of diffusion for SGP
        :param rep: number of sets of inducing variables
        """
        super().__init__(ydim, xdim, mean, cov, noise, "I")

        self.inducing = torch.as_tensor(inducing)
        self.inducing.requires_grad_(False)

        self.diffusion = torch.tensor(diffusion)
        self.rep = rep
        self.initialize()

    @property
    def fvar(self):
        return torch.exp(self.logvar)

    def initialize(self):
        mean = self.mean_func(self.inducing).t().reshape(-1, 1) + torch.zeros(
            self.ydim * self.inducing.shape[0], 1
        )  # (pm, 1)
        mean.unsqueeze_(0)  # (1, pm, 1)
        mean = mean.repeat(self.rep, 1, 1)  # (n, pm, 1)
        Kzz = torch.ones_like(mean) * self.cov_func.var  # (n, pm, 1)
        self.add_module("qz", Qosterior(mean, Kzz))

    def predict(self, x, *, z=None, gamma=None):
        if z is None:
            z = self.qz.mean
            gamma = self.qz.cov

        return predict(x, z, self.inducing, self.K, self.cov_func, gamma=gamma)

    def update(self, x, y):
        with torch.no_grad():
            z = self.qz.mean
            gamma = self.qz.cov
            z_shape = z.shape
            gamma_shape = gamma.shape

            assert z_shape == gamma_shape

            m, gamma = update(
                x, y, z, gamma, self.inducing, self.K, self.cov_func, self.fvar
            )

            assert z_shape == m.shape and gamma_shape == gamma.shape
            self.qz.mean, self.qz.cov = m, gamma


def predict(x, z, inducing, K, cov_func, *, gamma=None):
    """Predict f given x and (distribution of) z
    :param x: (n, q)
    :param z: (n, pm, 1)
    :param inducing: (m, q)
    :param K: (p, p)
    :param cov_func: covariance function
    :param gamma: (n, pm, 1)
    :returns
        fmean: f(x) or E[f(x)]
        fvar: var[f(x)] given gamma
    """
    x = torch.as_tensor(x)

    Kff = cov_func(x)  # (n, n)
    Kfz = cov_func(x, inducing)  # (n, m)
    Kzz = cov_func(inducing)  # (m, m)

    Kff = Kff.diagonal().unsqueeze_(-1).unsqueeze(-1)  # (n, 1, 1)
    Kfz.unsqueeze_(1)  # (n, 1, m)
    Kzz.unsqueeze_(0)  # (1, m, m)

    A = solve(Kzz, Kfz.transpose(1, 2))  # (n, m, 1)
    B = Kff - Kfz @ A  # (n, 1, 1)
    A.transpose_(1, 2)  # (n, 1, m)
    A = yakron(K, A)  # (n, p, pm)
    B = yakron(K, B)  # (n, p, p)

    # A = A.reshape(n, -1, A.shape[-1])

    fmean = A @ z  # (n, p, 1)
    fmean.squeeze_(-1)  # (n, p)
    fvar = B.diagonal(dim1=1, dim2=2)  # (n, p)

    if gamma is not None:
        fvar += torch.squeeze((A ** 2) @ gamma, -1)  # (n, p)

    return fmean, fvar


def update(x, y, z, gamma, inducing, K, cov_func, state_var):
    """Calculate the updated posterior of z given new data
    param x: state (n, q)
    param y: observation (n, p)
    param z: old posterior mean (n, pm, 1)
    param gamma: old posterior variance (n, pm, 1)
    param inducing: pseudo input (m, q)
    param K: covariance of f (p, p)
    param cov_func: covariance function
    param state_var: variance of state noise, scalar
    :returns
        new posterior mean
        new posterior variance
    """
    x = torch.as_tensor(x)  # (n, q)
    y = torch.as_tensor(y)  # (n, p)
    y = y.unsqueeze(-1)  # (n, p, 1)

    Kff = cov_func(x)  # (n, n)
    Kfz = cov_func(x, inducing)  # (n, m)
    Kzz = cov_func(inducing)  # (m, m)

    Kff = Kff.diagonal().unsqueeze_(-1).unsqueeze(-1)  # (n, 1, 1)
    Kfz.unsqueeze_(1)  # (n, 1, m)
    Kzz.unsqueeze_(0)  # (1, m, m)

    A = solve(Kzz, Kfz.transpose(1, 2))  # (n, m, 1)
    B = Kff - Kfz @ A  # (n, 1, 1)
    A.transpose_(1, 2)  # (n, 1, m)
    A = yakron(K, A)  # (n, p, pm)
    B = yakron(K, B)  # (n, p, p)

    b = B.diagonal(dim1=1, dim2=2)  # (n, p)
    c = b + state_var  # (n, p)
    c.unsqueeze_(-1)  # (n, p, 1)

    g = gamma  # (n, pm, 1)
    # Inversion by Woodbury identity
    d = (A ** 2) @ gamma + c  # (n, p, 1)

    GAT = gamma * A.transpose(1, 2)  # (n, pm, p)
    g = g - (GAT ** 2) @ (1 / d)  # (n, pm, 1)
    m = g * (z / gamma + A.transpose(2, 1) @ (y / c))

    return m, g
