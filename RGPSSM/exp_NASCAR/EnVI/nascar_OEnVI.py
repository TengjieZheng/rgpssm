import numpy as np
import numpy.random as npr
import torch
import matplotlib.pyplot as plt
import matplotlib

from exp_NASCAR.RGPSSM.method import save_show

matplotlib.use('TkAgg')
from scipy.special import expit as sigmoid
from models.EnVI import OnlineEnVI
from tqdm import tqdm
from utils import settings as cg
from method import *

randseed = 523
cg.reset_seed(randseed)                      # setting random seed
device = cg.device                           # setting device
dtype = torch.float                          # setting dtype
plt.rcParams["figure.figsize"] = (20,15)


# The function below implements the RSLDS (recurrent switching linear dynamical systems).
def rslds(x, A, B, R, r, K, dx=False):
    v = np.zeros(K - 1) # 用来表示是在哪个区域
    p = np.zeros(K)

    for k in range(K - 1):
        v[k] = R[:, k] @ x + r[k]

    "Compute weighted LDS" # 计算下一个状态
    fx = 0
    for k in range(K):
        w = 1
        # Compute weight
        for j in range(k):
            w *= sigmoid(-v[j])
        if k != K - 1:
            w *= sigmoid(v[k])

        fx += w * (A[:, :, k] @ x + B[:, k])

    # 计算状态增量
    if dx:
        fx = fx - x

    return fx


# Settings
d_latent = 2  # latent space dimensionality
d_obs = 4  # observation dimenionality
vary = 0.01  # variance of observation noise
varx = 1e-3  # variance of state noise
T1 = 500  # length of training set
T2 = 500   # length of forecasting
T = T1 + T2 # total length

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
def learning_and_prediction(y):
    # Initialize the model
    fixEmission = True
    fixTransition = False
    m_inducing = 20             # number of GP inducing points
    var_task = 10.
    len_scale = 5.
    ips = 10 * torch.randn((d_latent, m_inducing, d_latent), dtype=dtype)
    model = OnlineEnVI(dim_x=d_latent, dim_y=d_obs, ips=ips, dim_c=0, N_MC=50, process_noise_sd=1,
                       emission_noise_sd=np.sqrt(vary), consistentSampling=False, learn_emission=False, residual_trans=False,
                       H=torch.tensor(Cobs, dtype=dtype), var_task=var_task, len_scale=len_scale).to(device)
    if fixEmission:
        model.emission_likelihood.requires_grad_(False)
    if fixTransition:
        model.likelihood.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Learning
    initial_state = torch.tensor(x[:,0], dtype=dtype).expand(model.N_MC, model.state_dim, 1, model.state_dim)
    maxEpoch=20

    dataIter = tqdm(range(0, T1), desc='Data index')
    # one-step prediction (with filtering at each step
    x_t_1 = initial_state

    filter_x_all = []
    filter_x_var_all = []
    for i in dataIter:
        optimizer.zero_grad()
        elbo, x_t, filtered_mean, filtered_var = model(x_t_1=x_t_1,  y_t=torch.tensor(y[:, i][None, None], device=device, dtype=dtype),)

        filter_x_all.append(filtered_mean)
        filter_x_var_all.append(filtered_var)

        loss = -elbo
        loss.backward()
        optimizer.step()
        x_t_1 = x_t.detach()
        dataIter.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # # Plot the inferred trajectory and velocity field.
    x_est = torch.cat(filter_x_all, dim=1).detach().cpu().numpy()
    x_est = x_est.squeeze().T             # shape: 2 x T1
    x_est_var = torch.cat(filter_x_var_all, dim=1).detach().cpu().numpy()
    x_est_var = x_est_var.squeeze().T    # shape: 2 x T1

    # make prediction
    # pred_x,          shape: batch_size x seq_len x state_dim
    # pred_x_var,      shape: batch_size x seq_len x state_dim
    x_test_ini = filtered_mean[:, -1].expand(model.N_MC, model.state_dim, -1, -1)
    pred_x, pred_x_var = model.Forcasting(T=T2, x_0=x_test_ini)
    x_pred = pred_x.squeeze().detach().cpu().numpy()
    x_pred = np.vstack((x_est[:, -1].reshape(1, -1), x_pred))

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
    name_method = 'oenvi'
    x, y, Y, Cobs, dobs = get_data()
    x_est, x_pred = learning_and_prediction(y)
    traj_plot(x, x_est, name_method)
    state_plot(T1, T2, T, x, x_est, x_pred, name_method)
    plt.show()