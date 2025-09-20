import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pickle
from scipy.spatial.transform import Rotation as R

from utils import load_pickle, save_pickle

def quaternion_to_T_dir(q0, q1, q2, q3):

    quaternions = np.column_stack((q0, q1, q2, q3))                             # SciPy uses the order (x, y, z, w)
    rotation_matrices = R.from_quat(quaternions).as_matrix()
    # rotation_matrices = np.transpose(rotation_matrices, (0, 2, 1))

    T_dir = rotation_matrices[:, :, 2]

    return T_dir, rotation_matrices

def preprocess(filename):
    log = load_pickle(filename)
    data = log[0]

    dt = data['t'].ravel()[1] - data['t'].ravel()[0]
    print("dt:", dt)
    p = data['p']       # (n,3) position
    v = data['v']       # (n,3) velocity
    acc = data['acc']   # (n,3) acceleration
    pwm = data['pwm']   # (n,4) PWM

    p_smth = data_smooth(data['p'], n_smooth=5)
    v_diff = get_diff(p_smth, dt)
    v_smth = data_smooth(v_diff, n_smooth=5)

    a_diff = get_diff(v_smth, dt)
    # a_diff[:, 2] = a_diff[:, 2] + 9.8

    # state curves
    id_l, id_r = int(data['t'].size * 0.2), int(data['t'].size * 0.9)
    state_plot(data['t'][:id_r-id_l, :], [data['pwm'][id_l:id_r, :]],
               label_lst=['pwm'], name_lst=['$u_1$', '$u_2$', '$u_3$', '$u_4$'])

    state_plot(data['t'][:id_r-id_l, :], [data['p'][id_l:id_r, :]],
               label_lst=['meas'], name_lst=['$x$', '$y$', '$z$'])

    state_plot(data['t'][:id_r-id_l, :], [data['v'][id_l:id_r, :], v_diff[id_l:id_r, :], v_smth[id_l:id_r, :]],
               label_lst=['meas', 'diff', 'smth'], name_lst=['$v_x$', '$v_y$', '$v_z$'])
    state_plot(data['t'][:id_r-id_l, :], [acc[id_l:id_r, :], a_diff[id_l:id_r, :]],
               label_lst=['meas', 'diff'], name_lst=['$a_x$', '$a_y$', '$a_z$'])

    # 3D trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(p[id_l:id_r,0], p[id_l:id_r,1], p[id_l:id_r,2], label='Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

    # data for learning
    data_ = {}
    data_['p'] = data['p'][id_l:id_r, :]
    data_['v'] = v_diff[id_l:id_r, :]
    data_['a'] = a_diff[id_l:id_r, :]
    data_['pwm'] = data['pwm'][id_l:id_r, :]
    data_['q'] = data['q'][id_l:id_r, :]
    data_['t'] = data['t'][:id_r-id_l, :]
    T_dir, _ = quaternion_to_T_dir(data_['q'][:, 0], data_['q'][:, 1], data_['q'][:, 2], data_['q'][:, 3])
    # T_dir = get_T_dir_a(data_['a'])
    data_['T_dir'] = T_dir   # direction vector of the Z axis of the quadrotor

    save_pickle(data_, './data/data_.pkl')

def verify_thrust_direction():
    data = load_pickle('./data/data_.pkl')

    T_dir_acc = get_T_dir_a(data['a'])
    T_dir_q = get_T_dir_q(data)

    fig = plt.figure(figsize=(14, 4))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        plt.plot(data['t'], T_dir_acc[:, i], label='acc')
        plt.plot(data['t'], T_dir_q[:, i], label='q')
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Direction of thrust')
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

def get_T_dir_q(data):
    """Get the direction vector of thrust using quaternion"""
    T_dir, _ = quaternion_to_T_dir(data['q'][:, 0], data['q'][:, 1], data['q'][:, 2], data['q'][:, 3])

    return T_dir

def get_T_dir_a(a):
    """Get the direction vector of thrust using acceleration"""
    T_dir_acc = a
    T_dir_acc[:, -1] = T_dir_acc[:, -1] + 9.8
    T_dir_acc = T_dir_acc / np.linalg.norm(T_dir_acc, ord=2, axis=-1).reshape(-1, 1)

    return T_dir_acc

def state_plot(t, s_lst, label_lst, name_lst):
    n_col = s_lst[0].shape[-1]
    fig = plt.figure(figsize=(14, 4))
    for i in range(n_col):
        ax = fig.add_subplot(1, n_col, i+1)
        for s, l in zip(s_lst, label_lst):
            plt.plot(t, s[:, i], label=l, alpha=0.7)
        plt.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(name_lst[i])
    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.3, hspace=None)

def data_smooth(x, n_smooth=3):
    """
    Smooth data by averaging over neighboring frames.
    Args:
        x: (n, dx) numpy array
        n_smooth: half window size for smoothing,
                  the actual window size = 2 * n_smooth + 1
    Returns:
        x_smooth: (n, dx) numpy array
    """
    n, dx = x.shape
    x_smooth = np.zeros_like(x)

    for i in range(n):
        # 窗口范围： [i-n_smooth, i+n_smooth]
        start = max(0, i - n_smooth)
        end = min(n, i + n_smooth + 1)
        x_smooth[i] = np.mean(x[start:end], axis=0)

    return x_smooth

def get_diff(x, dt):
    """numerical difference
    x: (n, dx)
    """
    xo = x[:-1, :]
    xn = x[1:, :]
    diff_x = (xn - xo) / dt
    diff_x = np.concatenate((diff_x, diff_x[[-1], :]), axis=0)
    return diff_x


if __name__ == '__main__':
    preprocess(filename='./data/1_helx_log.pkl')
    # verify_thrust_direction()
    plt.show()