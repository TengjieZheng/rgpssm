import numpy as np
import numpy.random as npr
import random
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.special import expit as sigmoid
from tqdm import tqdm
import gpytorch

from rgpssm import RGPSSM
from method import *

randseed = 523
random.seed(randseed)
np.random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed(randseed)

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

x, y, Y, Cobs, dobs = get_data()
Cobs_tensor = Np2Torch(Cobs)
dobs_tensor = Np2Torch(dobs).view(-1, 1)


class Fun():
    """Some functions defining the GP model and GPSSMs"""

    def fun_input(self, x, u):
        """Get GP input
        Arg:
            x: system state
            u: control input
        Returns:
            z: GP input
        """
        z = x.view(1, -1)

        return z

    def fun_meas(self, x):
        """Measurement model
        Args:
            x: system state
        Returns:
            y: measurement
        """
        x = x.view(-1, 1)
        y = (Cobs_tensor @ x + dobs_tensor).view(-1, 1)

        return y

    def fun_mean(self, z):
        """prior mean function of GP
        Args:
            z: GP input
        Returns:
            f: mean
        """
        f = torch.zeros((d_latent, 1)) * z.view(-1)[0]
        f = f.view(-1, 1)

        return f

    def fun_tran(self, x, u, f):
        """Transition model
        Args:
            x: system state
            u: control input
            f: GP prediction
        Returns:
            F: next state
        """
        F = x + f.view(-1, 1)

        return F

@ timing
def learning_and_prediction():
    # GPSSM parameters
    num_task = 2                            # GP output dimension
    var_task = [10., 10.]                   # kernel hyperparameters: signal variance
    len_scale = [5., 5.]                    # kernel hyperparameters: length scale
    x0 = Np2Torch(x[:, 0]).view(-1, 1)      # initial state mean
    P0 = torch.eye(d_latent) * 1.           # initail state covariance
    Q = torch.eye(d_latent) * (varx + 0.00) # process noise covariance
    R = torch.eye(d_obs) * vary * 1.        # measurement noise covariance

    # GP kernel
    kernel = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d_latent),
                                              num_tasks=num_task, rank=0)  # Kxx \otimes V
    kernel.task_covar_module.var = torch.tensor(var_task).view(-1, 1)
    kernel.data_covar_module.lengthscale = torch.tensor(len_scale).view(-1, 1)

    # GPSSM
    gpssm = RGPSSM(x0, P0, Q, R, Fun(), kernel, flag_chol=True, jitter=1e-4, budget=20,
                          eps_tol=1e-5, num_opt_hp=0, lr_hp=0., Z=None, Qu=0)

    x_est = np.empty((2, 0))
    for ii in range(T):

        gpssm.predict()                                     # Prediction step
        if ii < T1:
            # Learning phase
            gpssm.correct(Np2Torch(y[:, ii]).view(-1, 1))   # Correction step
            gpssm.hyperparam_opt()
            x_est = np.hstack((x_est, Torch2Np(gpssm.x).reshape(-1, 1)))
            x_pred = Torch2Np(gpssm.x).reshape(-1, 1)
        else:
            # Prediction phase
            noise = torch.randn_like(gpssm.x) * torch.sqrt(torch.as_tensor(varx))
            gpssm.x = gpssm.x + noise
            x_pred = np.hstack((x_pred, Torch2Np(gpssm.x).reshape(-1, 1)))

        if ii % 10 == 0:
            num_id = gpssm.Z.shape[0]
            print(f'ii = {ii}, inducing points = {num_id}')

    return x_est, x_pred.T

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
    name_method = 'rgpssm'
    x_est, x_pred = learning_and_prediction()
    traj_plot(x, x_est, name_method)
    state_plot(T1, T2, T, x, x_est, x_pred, name_method)
    plt.show()


