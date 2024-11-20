import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from method import *
from wing_rock import WingRock


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def ExactGP(train_x, train_y, training_iter=100, lr=0.1):
    """Get GP hyperparameters. Only suitable for single dimensional GP"""

    kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]), num_tasks=train_y.shape[1], rank=0)  # Kxx \otimes V
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, kernel, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def get_hp():
        ls = model.covar_module.data_covar_module.lengthscale.detach().clone().numpy()
        var = model.covar_module.task_covar_module.var.detach().clone().numpy()
        noise = likelihood.noise.detach().clone().numpy()

        return ls, var, noise

    Loss = []
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.view(-1))
        loss.backward()
        ls, var, noise = get_hp()
        print(
            f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f} - lengthscale: {ls} - var: {var} - noise: {noise}')
        optimizer.step()
        Loss.append(loss.item())

    figure()
    plt.plot(Loss)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')

    return get_hp()

class RefHyperparam():
    """Attain reference value of GP hyperparameters"""

    def __init__(self, tend):
        self.tend = tend
        self.path = 'log/hp.pkl'

    def run(self):
        """Get hyperparameters"""

        self.log = load_pickle('wr.pkl')
        self.data = self.log.data_recorder.database

        idx = np.where(self.data['t'] >= self.tend)[0][0]
        theta, p, Delta = self.data['theta'][:idx, 0], self.data['p'][:idx, 0], self.data['Delta'][:idx, 0]
        Delta = Delta + np.random.randn(Delta.size) * 0.03
        self.train_x = Np2Torch(np.hstack((theta.reshape(-1, 1), p.reshape(-1, 1))))
        self.train_y = Np2Torch(Delta.reshape(-1, 1))

        self.ls, self.var, self.noise = ExactGP(self.train_x, self.train_y, training_iter=150, lr=0.2)
        self.save()

    def save(self):
        data = [self.ls, self.var, self.noise]
        save_pickle(data, self.path)

    def load(self):
        self.ls, self.var, self.noise = load_pickle(self.path)


if __name__ == '__main__':
    hp = RefHyperparam(tend=100)
    hp.run()
    plt.show()