from abc import abstractmethod

import numpy as np
import torch
from math import pi
from torch.nn import Module, Parameter

from .operation import squared_scaled_dist, scaled_dist


class CovarianceFunction:
    """Covariance function"""

    @abstractmethod
    def __call__(self, a, b=None):
        pass

    def __add__(self, other):
        return Additive(self, other)


class Additive(CovarianceFunction):
    def __init__(self, covfunc1, covfunc2):
        self._cov1 = covfunc1
        self._cov2 = covfunc2

    def __call__(self, a, b=None):
        return self._cov1(a, b) + self._cov1(a, b)


class SquaredExponential(Module, CovarianceFunction):
    """Squared exponential covariance function"""

    def __init__(self, var, scale, jitter=1e-6):
        super().__init__()
        self.register_parameter(
            "logvar", Parameter(torch.tensor(np.log(var)), requires_grad=True)
        )
        self.register_parameter(
            "loggamma", Parameter(torch.tensor(-np.log(scale)), requires_grad=True)
        )
        self.jitter = jitter

    @property
    def scale(self):
        return torch.exp(-self.loggamma)

    @property
    def var(self):
        return torch.exp(self.logvar)

    @property
    def gamma(self):
        """Inverse length scale"""
        return torch.exp(self.loggamma)

    def forward(self, a, b=None):
        if b is None:
            sym = True
            b = a
        else:
            sym = False

        d2 = squared_scaled_dist(a, b, gamma=self.gamma)
        cov = self.var * torch.exp(-0.5 * d2)

        if sym:
            cov += torch.eye(a.shape[0]) * self.jitter

        return cov


class RationalQuadratic(CovarianceFunction):
    r"""Rational quadratic covariance function
    \sigma^2 (1 + \frac{r^2}{2 \alpha l^2})^{- \alpha}
    """

    def __init__(self, var, scale, alpha, jitter=1e-6):
        self._logvar = torch.tensor(np.log(var))
        self._loggamma = torch.tensor(-np.log(scale))
        self._logalpha = torch.tensor(np.log(alpha))
        self.jitter = jitter

    @property
    def scale(self):
        return torch.exp(-self._loggamma)

    @property
    def var(self):
        return torch.exp(self._logvar)

    @property
    def alpha(self):
        return torch.exp(self._logalpha)

    @property
    def gamma(self):
        return torch.exp(self._loggamma)

    def __call__(self, a, b=None):
        if b is None:
            sym = True
            b = a
        else:
            sym = False

        r2 = squared_scaled_dist(a, b, gamma=self.gamma)
        cov = self.var * (1 + 0.5 * r2 / self.alpha) ** (-self.alpha)

        if sym:
            cov += torch.eye(a.shape[0]) * self.jitter

        return cov


class Linear(CovarianceFunction):
    r"""Linear covariance function
    \sigma_b^2 + \sigma_v^2 (x - c)'(x - c)
    """

    def __init__(self, sigma_b, sigma_v, c):
        self._logsigma_b = torch.tensor(np.log(sigma_b))
        self._logsigma_v = torch.tensor(np.log(sigma_v))
        self._c = torch.tensor(c)

    @property
    def sigma_b(self):
        return torch.exp(self._logsigma_b)

    @property
    def sigma_v(self):
        return torch.exp(self._logsigma_b)

    @property
    def c(self):
        return self._c

    def __call__(self, a, b=None):
        if b is None:
            b = a

        cov = self.sigma_b + self.sigma_v * torch.matmul(a - self.c, b - self.c)

        return cov


class Periodic_SE(Module, CovarianceFunction):
    "Periodic covariance function + SE. Assume: var * exp( -2 * sin^2(pi * |x - c| / p) / l)"
    def __init__(self, var, scale, period, jitter=1e-6):
        self._logvar = torch.tensor(np.log(var))
        self._loggamma = torch.tensor(-np.log(scale))
        self._logfreq = torch.tensor(-np.log(period))
        self.jitter = jitter

    @property
    def var(self):
        return torch.exp(self._logvar)

    @property
    def gamma(self):
        return torch.exp(self._loggamma)

    @property
    def freq(self):
        return torch.exp(self._logfreq)

    def __call__(self, a, b=None):
        "Assume that time is concataneted as the last element of each data point"
        time_a = a[:, [-1]]
        a = a[:, :-1]
        if b is None:
            sym = True
            b = a
            time_b = time_a
        else:
            sym = False
            time_b = b[:, [-1]]
            b = b[:, :-1]
        dim = a.shape[1]

        d2 = squared_scaled_dist(a, b, gamma=self.gamma)
        cov = self.var * torch.exp(-0.5 * d2)

        sin_dist = torch.sin(pi * (time_a - time_b.t()) * self.freq) ** 2
        cov *= self.var * torch.exp(-2 * sin_dist * self.gamma ** 2)


        if sym:
            cov = (cov + cov.t()) / 2
            cov += torch.eye(a.shape[0]) * self.jitter
        return cov
