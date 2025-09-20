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

from utils import ToTensor, Torch2Np, error_bar_plot, save_show, get_color

color_gt, color, color_h = 'gray', get_color('b'), get_color('r')
color_p = get_color('g')

def res_plot(data, gpssm, gpssm_h, gpssm_p, flag_save=True, flag_show=True):
    param_plot(data, gpssm, gpssm_h, gpssm_p, flag_save=flag_save, flag_show=flag_show)
    hyperparam_plot(data, gpssm, gpssm_h, gpssm_p, flag_save=flag_save, flag_show=flag_show)

def hyperparam_plot(data, gpssm, gpssm_h, gpssm_p, flag_save=True, flag_show=True):
    """Hyperparameters profile"""
    fig = plt.figure(figsize=(5.2, 2.9))
    plt.plot(data['t'], data['ls_h'][:, 0], ls=(2, (4, 2)), label='RGPSSM-H: $l^1$', color=color_h, linewidth=1.9)
    plt.plot(data['t'], data['ls_h'][:, 1], '-', label='RGPSSM-H: $l^2$', color=color_h, linewidth=1.9)
    plt.plot(data['t'], data['ls_p'][:, 0], ls=(2, (4, 2)), label='RGPSSM-H-P: $l^1$', color=color_p, linewidth=1.9)
    plt.plot(data['t'], data['ls_p'][:, 1], '-', label='RGPSSM-H-P: $l^2$', color=color_p, linewidth=1.9)
    plt.plot(data['t'], data['ls'], ls=(0, (4, 2)), label='RGPSSM: $l$', color=color, linewidth=1.9)
    plt.xlabel("Time [s]")
    plt.ylabel(fr'Length scale')
    ymin, ymax = plt.ylim()
    plt.ylim([ymin * 1.5, ymax * 1.2])
    plt.legend(loc='lower right', ncol=3, fontsize=7)
    plt.semilogy()

    plt.tight_layout()

    save_show(flag_save, flag_show, './fig/hyperparam_tvpi.pdf', fig)

def param_plot(data, gpssm, gpssm_h, gpssm_p, flag_save=True, flag_show=True):
    """Model parameters profile"""
    f_gp, var_gp, f_gp_h, var_gp_h, f_gp_p, var_gp_p = get_smoothing_est(data['t'], data['x'], gpssm, gpssm_h, gpssm_p)
    data['f_smth'] = f_gp
    data['std_f_smth'] = var_gp ** 0.5
    data['f_smth_h'] = f_gp_h
    data['std_f_smth_h'] = var_gp_h ** 0.5
    data['f_smth_p'] = f_gp_p
    data['std_f_smth_p'] = var_gp_p ** 0.5

    theta_lst = [1, 1, 2, 2]
    type_est_lst = ['filter', 'smooth', 'filter', 'smooth']
    title_map = {
        'filter': 'Filtering estimation',
        'smooth': 'Smoothing estimation'
    }

    fig = plt.figure(figsize=(10., 4.))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        single_fig(ax, data, gpssm, gpssm_h, gpssm_p, theta_lst[i], type_est_lst[i], flag_legend=i == 0, size_legend=7)
        if i < 2:
            ax.set_xticks([])
            ax.set_title(title_map[type_est_lst[i]])
        else:
            ax.set_xlabel('Time [s]')
        if i == 0 or i == 2:
            ax.set_ylabel(fr'Parameter $\theta_{theta_lst[i]}$')
        else:
            ax.set_yticks([])
    plt.tight_layout()

    save_show(flag_save, flag_show, './fig/param_iden.pdf', fig)

def single_fig(ax, data, gpssm, gpssm_h, gpssm_p, idx_tht=1, type_est='filter', flag_legend=True, size_legend=10):
    aph_line, aph_line_h, aph_line_p = 0.99, 0.99, 0.99
    aph_e, aph_e_h, aph_e_p = 0.15, 0.25, 0.1

    t, theta = data['t'], data['theta']
    f_pre, std_f = data['f_pre'], data['std_f']
    f_pre_h, std_f_h = data['f_pre_h'], data['std_f_h']
    f_pre_p, std_f_p = data['f_pre_p'], data['std_f_p']
    f_smth, std_f_smth = data['f_smth'], data['std_f_smth']
    f_smth_h, std_f_smth_h = data['f_smth_h'], data['std_f_smth_h']
    f_smth_p, std_f_smth_p = data['f_smth_p'], data['std_f_smth_p']

    tht = theta[:, idx_tht - 1]
    tht_max, tht_min = np.max(tht), np.min(tht)
    d_tht = tht_max - tht_min
    label_gt = fr'ground truth' if flag_legend else None
    ax.plot(t, tht, label=label_gt, linewidth=3, color=color_gt)
    if type_est == 'filter':
        ax.plot(t, f_pre[:, idx_tht - 1], ls=(0, (10, 5)), label=fr'RGPSSM', color=color, alpha=aph_line)
        ax.plot(t, f_pre_h[:, idx_tht - 1], ls=(0, (8, 4)), label=fr'RGPSSM-H', color=color_h, alpha=aph_line_h)
        ax.plot(t, f_pre_p[:, idx_tht - 1], ls=(0, (9, 8)), label=fr'RGPSSM-H-P', color=color_p, alpha=aph_line_p)
        error_bar_plot(t, f_pre[:, idx_tht - 1], std_f[:, idx_tht - 1] * 1.96, label=None, color=color, alpha=aph_e)
        error_bar_plot(t, f_pre_h[:, idx_tht - 1], std_f_h[:, idx_tht - 1] * 1.96, label=None, color=color_h, alpha=aph_e_h)
        error_bar_plot(t, f_pre_p[:, idx_tht - 1], std_f_p[:, idx_tht - 1] * 1.96, label=None, color=color_p, alpha=aph_e_p)
    else:
        ax.plot(t, f_smth[:, idx_tht - 1], ls=(0, (10, 5)), color=color, alpha=aph_line)
        ax.plot(t, f_smth_h[:, idx_tht - 1], ls=(0, (8, 4)), color=color_h, alpha=aph_line_h)
        ax.plot(t, f_smth_p[:, idx_tht - 1], ls=(0, (15, 7)), color=color_p, alpha=aph_line_p)
        error_bar_plot(t, f_smth[:, idx_tht - 1], std_f_smth[:, idx_tht - 1] * 1.96, label=None, color=color, alpha=aph_e)
        error_bar_plot(t, f_smth_h[:, idx_tht - 1], std_f_smth_h[:, idx_tht - 1] * 1.96, label=None, color=color_h, alpha=aph_e_h)
        error_bar_plot(t, f_smth_p[:, idx_tht - 1], std_f_smth_p[:, idx_tht - 1] * 1.96, label=None, color=color_p, alpha=aph_e_p)

        inducing_point_plot(ax, gpssm, gpssm_h, gpssm_p, idx_tht, color, color_h, size_legend)
        if idx_tht == 1:
            # Parameters for the inset
            width = 0.6
            height = 0.5
            bbox_to_anchor = (0.82, 0.7, 0.1, 0.1)
            # x1, x2 = 1180, 1340
            x1, x2 = 1790, 1970

            # Add inset
            axins = inset_axes(ax, width=width, height=height, loc="center",
                               bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes)

            # Plot curves
            axins.plot(t, tht, linewidth=3, color=color_gt)
            axins.plot(t, f_smth[:, 0], ls=(0, (10, 5)), color=color, alpha=aph_line)
            axins.plot(t, f_smth_h[:, 0], ls=(0, (8, 4)), color=color_h, alpha=aph_line_h)
            axins.plot(t, f_smth_p[:, 0], ls=(0, (15, 7)), color=color_p, alpha=aph_line_p)
            error_bar_plot(t, f_smth[:, 0], std_f_smth[:, 0] * 1.96, label=None, color=color, alpha=aph_e, ax=axins)
            error_bar_plot(t, f_smth_h[:, 0], std_f_smth_h[:, 0] * 1.96, label=None, color=color_h, alpha=aph_e_h, ax=axins)
            error_bar_plot(t, f_smth_p[:, 0], std_f_smth_p[:, 0] * 1.96, label=None, color=color_p, alpha=aph_e_p, ax=axins)
            inducing_point_plot(axins, gpssm, gpssm_h, gpssm_p, idx_tht, color, color_h, size_legend, flag_legend=False)

            # Set inset axis range (zoom into a local time interval)
            y1 = tht[x1:x2].min() + d_tht * 0.04
            y2 = tht[x1:x2].max() + d_tht * 0.02
            axins.set_xlim(t[x1], t[x2])
            axins.set_ylim(y1, y2)

            # Draw a rectangle indicating the inset area
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    if idx_tht == 1:
        ax.set_ylim([tht_min - 0.27 * d_tht, tht_max + 0.45 * d_tht])
    else:
        ax.set_ylim([tht_min - 0.45 * d_tht, tht_max + 0.35 * d_tht])
    if flag_legend: plt.legend(loc='upper left', ncol=4, fontsize=size_legend)

def inducing_point_plot(ax, gpssm, gpssm_h, gpssm_p, idx_tht, color='b', color_h='r', size_legend=10, flag_legend=True):

    ip_in = Torch2Np(gpssm.Z).ravel()
    idx_h = gpssm_h.Id_Z == idx_tht - 1
    idx_p = gpssm_p.Id_Z == idx_tht - 1
    ip_in_h = Torch2Np(gpssm_h.Z[idx_h, 0]).ravel()
    ip_in_p = Torch2Np(gpssm_p.Z[idx_p, 0]).ravel()

    ylim = ax.set_ylim()
    ymin, ymax = ylim
    y_ = ymin + (ymax - ymin) * 0.03

    y = np.ones_like(ip_in) * y_
    y_h = np.ones_like(ip_in_h) * y_
    y_p = np.ones_like(ip_in_p) * y_
    ax.scatter(ip_in, y, marker='+', s=14, color=color, linewidths=1.2, label=f'{ip_in.size} inducing points')
    ax.scatter(ip_in_h, y_h, marker='+', s=14, color=color_h, linewidths=1.2, label=f'{ip_in_h.size} inducing points')
    ax.scatter(ip_in_p, y_p, marker='+', s=14, color=color_p, linewidths=1.2, label=f'{ip_in_p.size} inducing points')

    if flag_legend: ax.legend(loc='upper left', ncol=2, fontsize=size_legend)

def get_smoothing_est(t, x, gpssm, gpssm_h, gpssm_p):
    def pred(x, t, alg='rgpssm'):
        in_gp = torch.zeros((2, 1))
        in_gp[0, 0] = t
        in_gp[1, 0] = t
        if alg == 'rgpssm':
            f, var_f = gpssm.GP_predict(ToTensor(x).view(1, -1), ToTensor(in_gp).view(1, -1))
        elif alg == 'rgpssm_h':
            f, var_f = gpssm_h.GP_predict(ToTensor(x).view(1, -1), ToTensor(in_gp).view(1, -1))
        elif alg == 'rgpssm_p':
            f, var_f = gpssm_p.GP_predict(ToTensor(x).view(1, -1), ToTensor(in_gp).view(1, -1))

        var_f = torch.diag(var_f)

        return f.detach().clone().numpy(), var_f.detach().clone().numpy()

    f_gp, var_gp = [], []
    f_gp_h, var_gp_h = [], []
    f_gp_p, var_gp_p = [], []
    for ii in range(t.size):
        xx = x.ravel()[ii]
        tt = t.ravel()[ii]

        f_single, v_single = pred(xx, tt, alg='rgpssm')
        f_gp.append(f_single)
        var_gp.append(v_single)
        f_single, v_single = pred(xx, tt, alg='rgpssm_h')
        f_gp_h.append(f_single.ravel())
        var_gp_h.append(v_single.ravel())
        f_single, v_single = pred(xx, tt, alg='rgpssm_p')
        f_gp_p.append(f_single.ravel())
        var_gp_p.append(v_single.ravel())
    f_gp, var_gp = np.array(f_gp), np.array(var_gp)
    f_gp_h, var_gp_h = np.array(f_gp_h), np.array(var_gp_h)
    f_gp_p, var_gp_p = np.array(f_gp_p), np.array(var_gp_p)

    return f_gp, var_gp, f_gp_h, var_gp_h, f_gp_p, var_gp_p