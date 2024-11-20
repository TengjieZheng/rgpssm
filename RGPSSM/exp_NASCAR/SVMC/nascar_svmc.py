
import numpy as np
import numpy.random as npr
import torch
from scipy.stats import multivariate_normal as mvn

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.special import expit as sigmoid
from sympy import lerchphi
from tqdm import tnrange, tqdm_notebook

import svmc
from svmc.covfun import SquaredExponential
from svmc import SVMC_GP
from svmc import proposal
from svmc import likelihood
from svmc.dynamics import SGPDynamics, KSGPDynamics
from svmc.base import KnowledgeModel
from svmc.gp import SGP
from method import *

torch.set_default_dtype(torch.float64)
randseed = 523

class KnoModelNascar(KnowledgeModel):
    def __init__(self):
        d_in, d_out = 2, 2
        super().__init__(d_in, d_out)

    def fun_tran(self, x, u=None, f=None):
        """State transition function
        Args:
            x : system state
            u : system input
            f : GP prediction
        Returns:
            x_pre : state prediction
            Af : partial derivative dx_pre / df
        """
        x = torch.as_tensor(x).reshape(-1, 1)
        if u is not None:
            u = torch.as_tensor(u).reshape(-1, 1)
        if f is not None:
            f = torch.as_tensor(f).reshape(-1, 1)

        x_pre = x + f
        Af = torch.eye(2)

        return x_pre, Af

    def fun_input(self, x, u=None):
        """Input function
        Args:
            x : system state
            u : system input
        Returns:
            z : GP input
        """
        x = torch.as_tensor(x).reshape(-1, 1)
        if u is not None:
            u = torch.as_tensor(u).reshape(-1, 1)
        xu = x
        # xu = torch.cat((x, u), dim=0)

        return xu.reshape(1, -1)

def GP_predict(sgp, n_pf, w, particles, KnoModel, x):
    qmean = 0
    qcov = 0

    for n in range(n_pf):
        qmean += w[n] * particles[2][n].qz.mean
        qcov += w[n] ** 2 * particles[2][n].qz.cov
    sgp.qz.mean = qmean
    sgp.qz.cov = qcov

    gp_in = KnoModel.fun_input(x, u=None)
    f, fvar = sgp.predict(gp_in, full_cov=False)

    return f.detach().numpy().ravel() + x.ravel(), fvar.detach().numpy()

# RSLDS functions
def rslds(x, A, B, R, r, K, dx=False):
    v = np.zeros(K - 1)
    p = np.zeros(K)

    for k in range(K - 1):
        v[k] = R[:, k] @ x + r[k]

    "Compute weighted LDS"
    fx = 0
    for k in range(K):
        w = 1
        # Compute weight
        for j in range(k):
            w *= sigmoid(-v[j])
        if k != K - 1:
            w *= sigmoid(v[k])

        fx += w * (A[:, :, k] @ x + B[:, k])

    if dx:
        fx = fx - x

    return fx

# Parameters
d_latent = 2        # latent space dimensionality
d_obs = 4           # observation dimenionality
d_hidden = 10       # MLP proposal hidden units
m_inducing = 20     # number of GP inducing points
vary = 0.01         # variance of observation noise
varx = 1e-3         # variance of state noise
T1 = 500            # length of training set
T2 = 500            # length of forecasting
T = T1 + T2         # total length

# RSLDS
rslds_data = np.load("rslds_nascar.npy", allow_pickle=True)[()]
K = 4
dim = d_latent

Atrue = rslds_data['A']
btrue = rslds_data['b']
Rtrue = np.zeros((dim, K - 1))
Rtrue[0, 0] = 100
Rtrue[0, 1] = -100
Rtrue[1, 2] = 100
rtrue = -200*np.ones(K - 1)
rtrue[-1] = 0

def get_data():
    np.random.seed(randseed)
    x = np.random.randn(T + 1, d_latent) * 10
    for t in range(T):
        x[t + 1, :] = rslds(x[t, :], A=Atrue, B=btrue, R=Rtrue, r=rtrue, K=K) + np.sqrt(varx) * np.random.randn(d_latent)
    x = x[1:, :].T

    C = npr.rand(d_obs, d_latent + 1) - 0.5  # parameters for emission (C = [C, D] for C*x + D
    Cobs = C[:, :-1]
    dobs = np.zeros((d_obs, 1)).ravel()
    y = Cobs @ x + dobs[:, None] + np.sqrt(vary) * npr.randn(d_obs, T)
    Y = torch.from_numpy(y.T).double()  # Convert to tensor

    return x, y, Y, Cobs, dobs
@ timing
def learning_and_prediction(fvar, length_scale, Y, Cobs, dobs):
    # Learning parapmeters
    n_pf = 50  # number of particles
    n_optim = 5  # number of particles to compute weights
    iter_optim = 5  # number of SGD iterations per step
    lr = 1e-4  # learning rate
    Tu = 50  # train GP after how many steps
    log = False
    emission_grad = False

    # GP hyperparameters
    gp_diffusion = 0.0

    # likelihood model
    log_like = likelihood.ISOGaussian(d_obs, d_latent, log_flag=log)
    log_like.input_to_output.weight.data = torch.from_numpy(Cobs)
    log_like.input_to_output.weight.requires_grad = emission_grad  # should we learn emission distribution

    log_like.input_to_output.bias.data = torch.from_numpy(dobs)
    log_like.input_to_output.bias.requires_grad = emission_grad  # should we learn emission distribution

    log_like.tau.data.fill_(np.log(vary))
    log_like.tau.requires_grad = emission_grad  # should we learn emission distribution

    # inducing points
    xmin, xmax = -12, 12
    ymin, ymax = -10, 10
    ux, uy = np.meshgrid(np.linspace(xmin, xmax, 5), np.linspace(ymin, ymax, 4))    # 网格化诱导点输入
    u_inducing = np.stack([ux.ravel(), uy.ravel()]).T

    # proposal
    torch.manual_seed(randseed)
    r = proposal.MlpProposal(d_latent + d_obs, d_hidden, d_latent, log_flag=False)  # MLP proposal

    # initial particles and weights
    x_particles = 10 * torch.randn(n_pf, d_latent)
    x_particles = x_particles.double()
    w = np.ones(n_pf) / n_pf
    x_est = np.zeros((d_latent, T1))
    noise = varx

    # GP particles
    cov_func = SquaredExponential(fvar, length_scale)
    cov_func.loggamma.requires_grad = False  # not optimize hyperparameter
    cov_func.logvar.requires_grad = False
    KnoModel = KnoModelNascar()
    Q = noise*np.eye(2)
    z_SS = [KSGPDynamics(u_inducing, None,
                cov_func, Q, KnoModel, diffusion=gp_diffusion) for n in range(n_pf)]  # Construct sparse GP objects
    # rz = SGP(d_latent, d_latent, u_inducing, None, cov_func, noise)
    pf = SVMC_GP(n_pf, n_optim, d_latent, d_obs, log_like, r, z_SS, lr=lr, gp_diffusion=gp_diffusion)
    # z_seq = SGP(d_latent, d_latent, u_inducing, None,
    #             cov_func, noise) # 没用上

    ESS = []
    Z_particles = []
    W = []
    f_seq = []
    fvar_seq = []
    particles = (x_particles, np.zeros((n_pf, d_latent)), z_SS)

    ## Learning
    nongradient = False  # train GP
    try:
        for t in range(T1):
            # print(t)
            if t > Tu:
                nongradient = True
            particles, _, w = pf.filter(Y[t, :], particles, w, iter_optim, nongradient=nongradient)
            x_particles = particles[0].detach()
            # particles = (x_particles, particles[1], 0, particles[3])
            ESS.append(1 / np.sum(w ** 2))  # compute effective sample size
            print("ESS:", ESS[-1])
            for n in range(n_pf):
                x_est[:, t] += w[n] * x_particles[n, :].numpy()
            Z_particles.append(particles[2])
            W.append(w)

            sgp = SGP(KnoModel.d_out, KnoModel.d_in, u_inducing, None, cov_func, noise)
            f, fvar = GP_predict(sgp, n_pf, w, particles, KnoModel, x_est[:, t].reshape(1, -1))
            f_seq.append(f.ravel())
            fvar_seq.append(fvar.ravel())
    except KeyboardInterrupt:
        pass

    f_seq = np.array(f_seq)
    fvar_seq = np.array(fvar_seq)

    ## Prediction
    # Construct a mean GP using the GP particles
    sgp = svmc.gp.SGP(d_latent, d_latent, u_inducing, None, cov_func, noise)
    qmean = 0
    qcov = 0

    for n in range(n_pf):
        qmean += w[n] * particles[2][n].qz.mean
        qcov += w[n] ** 2 * particles[2][n].qz.cov
    sgp.qz.mean = qmean
    sgp.qz.cov = qcov

    # fig1 = gp_velocity_field(sgp, None, None, [xmin, xmax], [ymin, ymax], l=xmin, u=xmax)

    # Forecast future state
    x_pred = np.zeros((T2 + 1, d_latent))
    x_pred[0, :] = x_est[:, -1]
    for t in range(T2):
        x_pred[t + 1, :] = x_pred[t, :] + sgp.predict(x_pred[[t], :])[0].numpy() + np.random.randn(d_latent) * np.sqrt(varx)

    return x_est, x_pred

def traj_plot(x, x_est, name_method=''):
    """Trajectory plot"""
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(*x[:, 1:T1], color='tab:blue', label='True')
    ax1.plot(*x_est, color='tab:red', label='Inferred')
    ax1.scatter(x[0, 1], x[1, 1], marker='o', color='green', s=50, zorder=5, label='start')
    ax1.scatter(x_est[0, 0], x_est[1, 0], marker='o', color='green', s=50, zorder=5)
    ax1.scatter(x[0, T1-1], x[1, T1-1], marker='o', color='yellow', s=50, zorder=5, label='stop')
    ax1.scatter(x_est[0, -1], x_est[1, -1], marker='o', color='yellow', s=50, zorder=5)
    plt.legend(loc='upper left')
    fig.show()
    save_show(flag_save=True, flag_show=True, filename='nascar1_'+name_method+'.pdf', fig=fig)

def state_plot(T1, T2, T, x, x_est, x_pred, name_method=''):
    """State Profile
        Args:
            T1 (float): learning time length
            T2 (float): prediction time length
            T (float): total time length
            x (float): state true value
            x_est (ndarray): state estimated value
            x_pred (ndarray): state predicted value
        Returns:
            None
    """

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(211)
    ax.plot(np.arange(T1 - 500, T),     x[0, -500 - T2:], color='blue')                 # state true value
    ax.plot(np.arange(T1 - 500, T1),    x_est[0, -500:], color='red', linestyle=':')    # state estimated value
    ax.plot(np.arange(T1, T),           x_pred[1:, 0], color='red', linestyle='-.')     # state predicted value
    ax.set_ylabel("x1")
    ax.axvline(x=T1, ymin=-25, ymax=25, color="grey")

    ax = fig.add_subplot(212)
    ax.plot(np.arange(T1 - 500, T), x[1, -500 - T2:], color='blue', label="true")                       # state true value
    ax.plot(np.arange(T1 - 500, T1), x_est[1, -500:], color='red', linestyle=':', label="filtered")     # state estimated value
    ax.plot(np.arange(T1, T), x_pred[1:, 1], color='red', linestyle='-.', label="predicted")            # state predicted value
    ax.set_xlabel("t")
    ax.set_ylabel("x2")
    ax.legend()
    ax.axvline(x=T1, ymin=-25, ymax=25, color="grey")
    ax.legend(loc="upper left")
    fig.show()

    save_show(flag_save=True, flag_show=True, filename='nascar2_' + name_method + '.pdf', fig=fig)

    RMSE = np.sqrt(np.mean(np.sum((x[:, -501:].T - x_pred) ** 2, axis=1)))
    print(f'RMSE = {RMSE:.4f}')

if __name__ == '__main__':
    name_method = 'svmc'

    # GP kernel parameters
    length_scale = 5.
    fvar = 10.

    x, y, Y, Cobs, dobs = get_data()
    x_est, x_pred = learning_and_prediction(fvar, length_scale, Y, Cobs, dobs)
    traj_plot(x, x_est, name_method)
    state_plot(T1, T2, T, x, x_est, x_pred, name_method)
    plt.show()