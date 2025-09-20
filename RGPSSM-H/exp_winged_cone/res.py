from typing import Tuple, Optional, List, Union
from types import SimpleNamespace
from torch import Tensor
from numpy import ndarray

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from trainer import Recorder
from utils_res import subfig_plot, error_bar, set_ylim, area_plot, area_single
from utils import get_color, get_color_sys, nMSE, Mnlp, save_show


def res_all(alg_lst):
    name_map = {
        'rgpssm': 'RGPSSM',
        'rgpssm_h_ekf': 'RGPSSM-H (EKF)',
        'rgpssm_h_ukf': 'RGPSSM-H (UKF)',
        'rgpssm_h_adf': 'RGPSSM-H',
    }
    if alg_lst is None:
        alg_lst = name_map.keys()

    color_map = {
        'rgpssm': get_color('b'),
        'rgpssm_h_ekf': get_color('o'),
        'rgpssm_h_ukf': get_color('g'),
        'rgpssm_h_adf': get_color('r'),
        'gt': 'gray',
        'meas': get_color('g'),
        'basic': get_color_sys(1),
        'learn': get_color_sys(2),
    }

    attitude_angle_plot(alg_lst, name_map, color_map)
    aerodynamic_angle_plot(alg_lst, name_map, color_map)
    learn_plot(alg_lst, name_map, color_map)

def attitude_angle_plot(alg_lst, name_map, color_map):
    """plot attitude angle, angular rate and control surface deflection"""
    fig, axs = plt.subplots(3, 3, figsize=(16, 6))

    # Command
    _, d = Recorder.load(alg_lst[0])
    angle_d = d['x1_d'] * 57.3
    subfig_plot(d['t'], [angle_d], label_lst=['command'], name_lst=['roll angle', 'pitch angle', 'yaw angle'],
                ax_lst=axs[0], color_lst=[color_map['gt']], flag_label=False)

    error_dict = {}
    # State profile
    for alg in alg_lst:
        _, d = Recorder.load(alg)
        angle = np.concatenate((d['roll'], d['pitch'], d['yaw']), axis=-1) * 57.3
        rate = np.concatenate((d['p'], d['q'], d['r']), axis=-1) * 57.3
        delta = np.concatenate((d['delta_a'], d['delta_e'], d['delta_r']), axis=-1) * 57.3

        subfig_plot(d['t'], [angle], label_lst=[name_map[alg]],
                    ax_lst=axs[0], color_lst=[color_map[alg]], flag_label=False,
                    legend_set={'ncol': 3, 'loc': 'upper right', 'fontsize': 8})
        set_ylim(axs[0], upper=0.3, lower=0.05, x_arr=angle_d)

        subfig_plot(d['t'], [rate], label_lst=[name_map[alg]], name_lst=['$p$', '$q$', '$r$'],
                    ax_lst=axs[1], color_lst=[color_map[alg]])

        subfig_plot(d['t'], [delta], label_lst=[name_map[alg]], name_lst=['$\delta_a$', '$\delta_e$', '$\delta_r$'],
                    ax_lst=axs[2], color_lst=[color_map[alg]], flag_xticks=True)

        idx_ctrl = np.sum(d['t'] < d['t_ctrl'])
        error_dict[alg] = nMSE(angle_d[idx_ctrl:, :], angle[idx_ctrl:, :])
        print(f'control nMSE = {error_dict[alg]}')

    # Background plots
    for i in range(3):
        area_plot(axs[i], d['t_ctrl'], color_map['basic'], color_map['learn'])
    axs[0][-1].legend(**{'ncol': 3, 'loc': 'upper right', 'fontsize': 8})

    plt.tight_layout()

def aerodynamic_angle_plot(alg_lst, name_map, color_map):
    """Plot aerodynamic angles: angle of attack, sideslip angle, and bank angle"""
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
    # Command
    _, d = Recorder.load(alg_lst[0])
    ad_angle_c = np.concatenate((d['alpha_c'], d['beta_c'], d['phi_v_c']), axis=-1) * 57.3
    subfig_plot(d['t'], [ad_angle_c], label_lst=['command'],
                name_lst=[r'Angle of attack $\alpha$ [deg]', r'Sideslip angle $\beta$ [deg]', r'Bank angle $\phi_v$ [deg]'],
                ax_lst=axs, color_lst=[color_map['gt']], flag_label=False, flag_xticks=True, linewidth_lst=[2])

    # State
    for alg in alg_lst:
        _, d = Recorder.load(alg)
        ad_angle = np.concatenate((d['alpha'], d['beta'], d['phi_v']), axis=-1) * 57.3
        subfig_plot(d['t'], [ad_angle], label_lst=[name_map[alg]],
                    ax_lst=axs, color_lst=[color_map[alg]], flag_label=True, flag_xticks=True,
                    xlabel='Time [s]', legend_set={'ncol':3, 'loc': 'upper left', 'fontsize':8})
    area_plot(axs, d['t_ctrl'], color_map['basic'], color_map['learn'])
    axs[0].text(14., -8.9, 'basic controller', size=8, color=color_map['basic'])
    axs[0].text(46., -8.9, 'GP controller', size=8, color=color_map['learn'])
    set_ylim(axs, upper=0.1, lower=0.08)

    plt.tight_layout()
    save_show(flag_save=True, flag_show=True, filename='./fig/wc_ctrl.pdf', fig=fig)

def learn_plot(alg_lst, name_map, color_map):
    """Plot angular rate estimation and moment coefficient prediction"""
    fig, axs = plt.subplots(2, 3, figsize=(14, 5.5))

    # Ground truth
    _, d = Recorder.load(alg_lst[0])
    omg = np.concatenate((d['p'], d['q'], d['r']), axis=-1) * 57.3
    omg_meas = np.concatenate((d['p_meas'], d['q_meas'], d['r_meas']), axis=-1) * 57.3
    Delta_C_true = d['C_moment'] - d['C_moment_0']

    id_r = np.where(d['t']>d['t_ctrl'])[0][0] if d['t'].ravel()[-1] > d['t_ctrl'] else d['t'].size
    subfig_plot(d['t'][:id_r, :], [omg[:id_r, :], omg_meas[:id_r, :]], label_lst=['ground truth', 'measurement'],
                name_lst=['roll rare $p$ [deg/s]', 'pitch rate $q$ [deg/s]', 'yaw rate $r$ [deg/s]'],
                ax_lst=axs[0], color_lst=[color_map['gt'], color_map['meas']], flag_label=True, linewidth_lst=[2, 0.5])
    subfig_plot(d['t'][:id_r, :], [Delta_C_true[:id_r, :]], label_lst=['ground truth'],
                name_lst=['Roll moment coefficient bias $\Delta C_l$', 'Pitch moment coefficient bias $\Delta C_m$', 'Yaw moment coefficient bias $\Delta C_n$'],
                ax_lst=axs[1], color_lst=[color_map['gt']], flag_label=False, flag_xticks=True, linewidth_lst=[2])

    # Estimation
    mse_dict = {}
    nll_lst = {}
    id_l = int(id_r/2)

    for alg in alg_lst:
        _, d = Recorder.load(alg)
        subfig_plot(d['t'][:id_r, :], [d['x_est'][:id_r, :]*57.3], label_lst=[name_map[alg]],
                    ax_lst=axs[0], color_lst=[color_map[alg]], flag_label=True, legend_set={'ncol':3, 'loc': 'upper left', 'fontsize':8},
                    ls_lst=[(0, (12, 5)), (0, (8, 4))])

        mse_dict[alg] = nMSE(Delta_C_true[id_l:id_r, :], d['f_pre'][id_l:id_r, :] * d['scale_f'])
        nll_lst[alg] = Mnlp(Delta_C_true[id_l:id_r, :], d['f_pre'][id_l:id_r, :] * d['scale_f'], d['std_est'][id_l:id_r, :]* d['scale_f'])

    def C_plot(ax_lst, channel_lst, flag_xlabel=False, flag_gt=False):
        for alg in alg_lst:
            _, d = Recorder.load(alg)
            f_pre = (d['f_pre'][:id_r, :]*d['scale_f'])[:, channel_lst]
            std_f = (d['std_f'][:id_r, :]*d['scale_f'])[:, channel_lst]
            if flag_gt:
                subfig_plot(d['t'][:id_r, :], [Delta_C_true[:id_r, channel_lst]], label_lst=['ground truth'],
                        ax_lst=ax_lst, color_lst=[color_map['gt']], flag_label=False, flag_xticks=True, linewidth_lst=[2])
            subfig_plot(d['t'][:id_r, :], [f_pre], label_lst=[name_map[alg]],
                        ax_lst=ax_lst, color_lst=[color_map[alg]], xlabel='Time [s]' if flag_xlabel else None, flag_xticks=True,
                        ls_lst=[(0, (12, 5)), (0, (8, 4))])
            error_bar(ax_lst, d['t'][:id_r, :], [f_pre], [std_f], color_lst=[color_map[alg]], alpha=0.1)


    C_plot(axs[1], channel_lst=[0, 1, 2], flag_xlabel=True)
    set_ylim(ax_lst=axs[0], upper=0.3, lower=0.1, x_arr=omg[:id_r, :])
    set_ylim(ax_lst=axs[1], upper=0.2, lower=0.2, x_arr=Delta_C_true[:id_r, :])

    # inset
    def inset_plot(idx, x1, x2, b2a=(0.82, 0.7, 0.1, 0.1), upper=0.1, lower=0.1):
        ax = axs[1][idx]
        axins = inset_axes(ax, width=1., height=0.7, loc="center", bbox_to_anchor=b2a, bbox_transform=ax.transAxes)
        C_plot([axins], channel_lst=[idx], flag_gt=True)
        ymin, ymax = Delta_C_true[x1:x2, idx].min(), Delta_C_true[x1:x2, idx].max()
        y1 = ymin - (ymax - ymin) * lower
        y2 = ymax + (ymax - ymin) * upper
        axins.set_xlim(d['t'][x1], d['t'][x2])
        axins.set_ylim(y1, y2)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # inset_plot(0, x1=745, x2=780, b2a=(0.25, 0.03, 0.5, 0.5), upper=0.25, lower=0.4)
    # inset_plot(1, x1=770, x2=787, b2a=(0.12, 0.03, 0.5, 0.5), upper=0.2, lower=0.2)
    # inset_plot(2, x1=740, x2=790, b2a=(0.15, 0.03, 0.5, 0.5), upper=0.1, lower=0.1)

    area_single(axs[0], 0, d['t_ctrl'], color_map['basic'])
    area_single(axs[1], 0, d['t_ctrl'], color_map['basic'])
    plt.tight_layout()
    save_show(flag_save=True, flag_show=True, filename='./fig/wc_learn.pdf', fig=fig)

    # learn nMSE
    for alg in alg_lst:
        print(alg)
        print(f'learn nMSE = {mse_dict[alg]}')
        print(f'learn mnll = {nll_lst[alg]}')

    improve_mse = mse_dict['rgpssm_h_adf'] / mse_dict['rgpssm']
    improve_nll = nll_lst['rgpssm_h_adf'] / nll_lst['rgpssm']
    print(f'improve_mse = {(1 - improve_mse).mean()}')
    print(f'improve_nll = {(1 - improve_nll).mean()}')

    # omg variance
    var_omg = np.mean(omg[:id_r, :]**2, axis=0)
    print(f'var_omg = {var_omg}')
    print(f'std_omg = {var_omg**0.5}')



