import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal as mvn
from torch.distributions import Normal, StudentT, Poisson
from torch.nn import Parameter

from .metric import gaussian_loss


class ISOGaussian(nn.Module):
    """Isotropic Gaussian likelihood, y ~ N(Cx+D, rI)"""
    def __init__(self, d_obs, d_in, log_flag=True, bias=True):
        super().__init__()
        self.d_in = d_in
        self.d_obs = d_obs
        self.log_flag = log_flag

        self.add_module("input_to_output", nn.Linear(d_in, d_obs, bias=bias))
        self.register_parameter("tau", Parameter(torch.zeros(1)))

    def compute_log_weight(self, y, x):
        mean = self.input_to_output(x)  # compute mean

        # compute isotropic covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        # Evaluate log density
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(y), -1)
        return log_weight

    def forward(self, y, x):
        return self.compute_log_weight(y, x)


class DiagGaussian(nn.Module):
    """Diagonal Gaussian likelihood, y ~ N(Cx+D, RR') where R is a diagonal matrix"""
    def __init__(self, d_obs, d_in, log_flag=True, mask_flag=False, normalize=False, bias=True):
        super().__init__()
        self.d_in = d_in
        self.d_obs = d_obs
        self.log_flag = log_flag

        self.add_module("input_to_output", nn.Linear(d_in, d_obs, bias=bias))
        self.register_parameter("tau", Parameter(torch.zeros(d_obs)))
        self.mask = torch.ones((d_obs, d_in))  # masking matrix

        if mask_flag:
            # Constrain loading matrix to be upper triangular
            for j in range(d_in - 1):
                self.mask[j, j + 1:] = torch.zeros(d_in - 1 - j)
        self.input_to_output.weight.data *= self.mask  # Make upper triangular
        if normalize:  # Only normalize if specified
            self.project_unit_norm()  # make weights be unit norm

    def compute_log_weight(self, y, x):
        mean = self.input_to_output(x)  # compute mean

        # compute diagonal covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        # Evaluate log density
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(y), -1)
        return log_weight

    def forward(self, y, x):
        return self.compute_log_weight(y, x)

    def project_unit_norm(self):
        """Project onto the manifold of unit column norm matrices"""
        with torch.no_grad():
            "No gradients are needed!"
            self.input_to_output.weight.data *= self.mask  # Make upper triangular  # TODO: Might not need this...
            norm = torch.sqrt(torch.sum(self.input_to_output.weight.data ** 2, 0))  # obtain column norms
            self.input_to_output.weight.data /= norm.unsqueeze(0)  # make weights have unit column norm


class ISOStudentT(nn.Module):
    """Isotropic Student's T distribution"""
    def __init__(self, d_obs, d_in, bias=False, df=2, log_flag=True):
        super().__init__()
        self.d_in = d_in
        self.d_obs = d_obs
        self.log_flag = log_flag
        self.df = df
        self.add_module("input_to_output", nn.Linear(d_in, d_obs, bias=bias))
        self.register_parameter("tau", Parameter(torch.zeros(1)))

    def compute_log_weight(self, y, x):
        mean = self.input_to_output(x)  # compute mean

        # compute isotropic covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        log_weight = torch.sum(StudentT(loc=mean, df=self.df, scale=var).log_prob(y), dim=-1)
        return log_weight

    def forward(self, y, x):
        return self.compute_log_weight(y, x)


class ISOGaussianNorm(nn.Module):
    """Isotropic Gaussian likelihood, y ~ N(Cx+D, rI)"""
    def __init__(self, d_obs, d_in, log=True):
        super().__init__()
        self.d_in = d_in
        self.d_obs = d_obs
        self.log = log

        self.add_module("input_to_output", nn.utils.weight_norm(nn.Linear(d_in, d_obs)))
        self.register_parameter("tau", Parameter(torch.zeros(1)))

    def compute_log_weight(self, y, x):
        mean = self.input_to_output(x)  # compute mean

        # compute isotropic covariance matrix
        var = torch.exp(self.tau)
        if self.log_flag:
            var = torch.log(1 + var)

        # Evaluate log density
        log_weight = torch.sum(Normal(loc=mean, scale=torch.sqrt(var)).log_prob(y), -1)
        return log_weight

    def forward(self, y, x):
        return self.compute_log_weight(y, x)


class Bern(nn.Module):
    """Bernoulli Likelihood, y ~ Bern(p) where p=sigmoid(Cx)"""
    def __init__(self, d_obs, d_in):
        super().__init__()
        self.d_in = d_in  # dimension of latent space
        self.d_obs = d_obs  # dimension of observation space i.e. number of neurons

        self.add_module("input_to_prob", nn.Linear(d_in, d_obs))

    def compute_log_weight(self, y, x):
        m = self.input_to_prob(x)
        p = torch.sigmoid(m)  # make it between 0 and 1
        log_weight = y * torch.log(p) + (1 - y) * torch.log(1 - p)
        return log_weight

    def forward(self, y, x):
        return self.compute_log_weight(y, x)


class NarendraLiObsv(nn.Module):
    "Likelihood used in Narendra-Li system"
    def __init__(self):
        super().__init__()
        self.d_obs = 1
        self.logvar = torch.log(0.1 * torch.ones(1))

    def compute_obsv(self, x):
        return x[:, 0]/(1 + 0.5 * torch.sin(x[:, 1])) + x[:, 1]/(1 + 0.5 * torch.sin(x[:, 0]))

    def compute_log_weight(self, y, x):
        m = self.compute_obsv(x)
        return -gaussian_loss(y, m, self.logvar)

    def forward(self, y, x):
        return self.compute_log_weight(y, x)


class NarendraLiObsvT(NarendraLiObsv):
    def __init__(self):
        super().__init__()

    def compute_log_weight(self, y, x):
        m = self.compute_obsv(x)
        return StudentT(loc=m, scale=self.scale, df=2).log_prob(y)


# TODO: Do we need the masking matrix? If we initialize the loading matrix to be 0, will the gradient for those elements also be 0?
class PoissLike(nn.Module):
    """y_t ~ Poiss(r) r = exp(Cx)"""
    def __init__(self, dy, dx, mask_flag=False, normalize=False):
        super().__init__()
        self.d_obs = dy
        self.d_latent = dx
        self.add_module("C", nn.Linear(dx, dy))
        self.mask = torch.ones((dy, dx))  # masking matrix

        if mask_flag:
            # Constrain loading matrix to be upper triangular"
            for j in range(dx - 1):
                self.mask[j, j + 1:] = torch.zeros(dx - 1 - j)
        self.C.weight.data *= self.mask  # Make upper triangular
        if normalize:  # Only normalize if specified
            self.project_unit_norm()  # make weights be unit norm

    def compute_log_weight(self, y, x):
        r = torch.exp(self.C(x))  # compute rate
        if len(y.shape) == 1:
            y = y.unsqueeze(0)
        return torch.sum(Poisson(rate=r).log_prob(y[:, -self.d_obs:]), 1)

    def forward(self, y, x):
        return self.compute_log_weight(y, x)

    def project_unit_norm(self):
        """Project onto the manifold of unit column norm matrices"""
        with torch.no_grad():
            # No gradients are needed!
            self.C.weight.data *= self.mask  # Make upper triangular  # TODO: Might not need this...
            norm = torch.sqrt(torch.sum(self.C.weight.data ** 2, 0))  # obtain column norms
            self.C.weight.data /= norm.unsqueeze(0)  # make weights have unit column norm
