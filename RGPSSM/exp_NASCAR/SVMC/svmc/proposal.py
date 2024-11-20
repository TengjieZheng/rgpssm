import torch
import torch.nn as nn
from torch.nn import Parameter


class MlpProposal(nn.Module):
    # Gaussian proposal where mean and variance are parameterized by MLP
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, log_flag=True, relu_flag=True, resnet=True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log_flag = log_flag
        self.resnet = resnet

        self.input_to_hidden = nn.Sequential(*[nn.Linear(d_in, d_hidden),
                                               nn.ReLU() if relu_flag else nn.Tanh()] +
                                             sum([
                                                 [nn.Linear(d_hidden, d_hidden),
                                                  nn.ReLU() if relu_flag else nn.Tanh()
                                                  ]
                                                 for _ in range(n_layers - 1)],
                                                 [])
                                             )
        self.hidden_to_mean = nn.Linear(d_hidden, d_out)  # proposal mean readout
        self.hidden_to_tau = nn.Linear(d_hidden, d_out)  # proposal variance readout

    def get_proposal_params(self, x):
        hidden_layer = self.input_to_hidden(x)
        mean = self.hidden_to_mean(hidden_layer)
        if self.resnet:
            mean = x[:, :self.d_out] + mean
        var = torch.exp(torch.min(self.hidden_to_tau(hidden_layer), torch.tensor(10.)))
        if self.log_flag:
            var = torch.log(1 + var)  # to avoid constrained optimization (Softplus)
        return mean, var

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, x):
        return self.get_proposal_params(x)


class MlpOptimal(nn.Module):
    # Gaussian proposal where mean and variance are parameterized by MLP where dynamics are also incorporated
    def __init__(self, d_in, d_hidden, d_out, dynamics, n_layers=2,
                 log_flag=True, relu_flag=True, d_stim=0):
        super().__init__()
        self.d_in = d_in
        self.d_stim = d_stim
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.log_flag = log_flag
        self.dynamics = dynamics
        self.loc = torch.zeros(1, d_in)
        self.scale = torch.ones(1, d_in)

        self.input_to_hidden = nn.Sequential(*[nn.Linear(d_in, d_hidden),
                                               nn.ReLU() if relu_flag else nn.Tanh()] +
                                              sum([
                                                  [nn.Linear(d_hidden, d_hidden),
                                                   nn.ReLU() if relu_flag else nn.Tanh()
                                                   ]
                                                  for _ in range(n_layers - 1)],
                                                  [])
                                             )
        self.hidden_to_mean = nn.Linear(d_hidden, d_out)  # proposal mean readout
        self.hidden_to_tau = nn.Linear(d_hidden, d_out)  # proposal variance readout

    def get_proposal_params(self, x):
        z = torch.cat((self.dynamics(x[:, :self.d_out + self.d_stim]),
                       x[:, self.d_out + self.d_stim:]), 1)
        # z = (z - self.loc) / self.scale
        hidden_layer = self.input_to_hidden(z)
        mean = x[:, :self.d_out] + self.hidden_to_mean(hidden_layer)
        var = torch.exp(
            torch.min(self.hidden_to_tau(hidden_layer), torch.tensor(10.)))  # to avoid constrained optimization
        if self.log_flag:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, x):
        return self.get_proposal_params(x)


class LinearProposal(nn.Module):
    # Gaussian proposal where mean and variance are parameterized by linear layer
    def __init__(self, d_in, d_out, log_flag=True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.log_flag = log_flag

        self.input_to_mean = nn.Linear(d_in, d_out)  # proposal mean readout
        self.input_to_tau = nn.Linear(d_in, d_out)  # proposal variance readout

    def get_proposal_params(self, x):
        mean = self.input_to_mean(x)
        var = torch.exp(torch.min(self.input_to_tau(x), torch.tensor(5.)))
        if self.log_flag:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, x):
        return self.get_proposal_params(x)


class ScaledLinearProposal(nn.Module):
    def __init__(self, d_in, d_out, log_flag=True, bias_flag=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.log_flag = log_flag

        self.A = nn.Linear(d_out, d_out, bias=bias_flag)
        self.tau_t = Parameter(torch.zeros(d_out))
        self.beta_t = Parameter(torch.zeros(d_out))
        self.mu_t = Parameter(torch.zeros(d_out))

    def get_proposal_params(self, x):
        "x_t ~ N(mu_t + diag(beta_t)*A*x_{t-1}, diag(sigma_t))"
        mean = self.mu_t + self.beta_t * self.A(x[:, :self.d_out])  # don't use observations for proposal
        var = torch.exp(self.tau_t)  # to avoid constrained optimization
        if self.log:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), diag(var(theta)))

    def forward(self, x):
        return self.get_proposal_params(x)


class BootstrapProposal(nn.Module):
    """Proposal used to run a bootstrap particle filter"""
    def __init__(self, d_in, d_out, d_obs, var, dynamics):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_obs = d_obs  # used to remove observation appended to the input
        self.var = var
        self.dynamics = dynamics  # function handle of the true dynamics

    def get_proposal_params(self, x):
        mean = self.dynamics(x[:, :-self.d_obs])
        var = self.var
        return mean, var

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, x):
        return self.get_proposal_params(x)


class AffineBootstrapProposal(nn.Module):
    """Proposal mean is an affine transformation of the dynamics"""
    def __init__(self, d_in, d_out, d_obs, dynamics, log_flag=True, bias_flag=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_obs = d_obs  # used to remove observation appended to the input
        self.dynamics = dynamics  # function handle of the true dynamics
        self.log_flag = log_flag

        self.A = nn.Linear(d_out, d_out, bias=bias_flag)
        self.tau_t = Parameter(torch.zeros(d_out))
        self.beta_t = Parameter(torch.zeros(d_out))
        self.mu_t = Parameter(torch.zeros(d_out))

    def get_proposal_params(self, x):
        mean = self.mu_t + self.beta_t * self.A(self.dynamics(x[:, :-self.d_obs]))  # don't use observations for proposal
        var = torch.exp(self.tau_t)  # to avoid constrained optimization
        if self.log_flag:
            var = torch.log(1 + var)  # to avoid constrained optimization
        return mean, var * torch.ones(mean.shape[0], 1)

    def sample(self, x, no_samples):
        mean, var = self.get_proposal_params(x)  # obtain mean and variance
        return mean + torch.sqrt(var) * torch.randn(no_samples,
                                                    self.d_out)  # return samples from N(mu(theta), var(theta)I)

    def forward(self, x):
        return self.get_proposal_params(x)


class DiffusingInducingProposal(nn.Module):
    def __init__(self, sgp, diffusion):
        self.sgp = sgp
        self.diffusion = diffusion

    def forward(self, *x):
        return self.sgp.qz.mean, self.sgp.qz.cov + torch.diag(self.diffusion)
